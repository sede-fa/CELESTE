"""
Vapour-liquid equilibrium with the Peng-Robinson equation of state, in JAX.

The isothermal flash :func:`flash_pt` is built for robustness rather than for
the smallest number of iterations:

1. **Stability first.** Michelsen's tangent plane test is run from a
   vapour-like and a liquid-like trial phase. The feed is only split if a trial
   phase lowers the Gibbs energy, so a single-phase feed can never produce a
   spurious two-phase "solution" and a two-phase feed is not missed because of
   a poor initial K-value estimate.
2. **K-values from the stability test** start the phase split, which puts the
   successive substitution inside the two-phase region from the first step.
3. **Rachford-Rice** is solved by a bracketed Newton iteration between its
   asymptotes; it cannot diverge.
4. Everything runs in fixed-shape ``lax`` loops: the flash can be jit compiled,
   vectorised with ``vmap`` and differentiated in forward mode (``jacfwd``).

Units: T in K, P in bar.
"""
from __future__ import annotations

from typing import NamedTuple

from . import enable_x64

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from . import eos  # noqa: E402
from .eos import LIQUID, VAPOUR, STABLE, PRParams  # noqa: E402

__all__ = ["FlashResult", "StabilityResult", "wilson_k", "rachford_rice", "stability", "flash_pt",
           "LIQUID", "VAPOUR", "TWO_PHASE"]

TWO_PHASE = 2
_TINY = 1e-30


class StabilityResult(NamedTuple):
    stable: jax.Array       #: True when the feed is stable as a single phase
    tpd: jax.Array          #: smallest (modified) tangent plane distance found; negative = unstable
    K: jax.Array            #: (Ns,) K-value estimate for a subsequent phase split
    iterations: jax.Array


class FlashResult(NamedTuple):
    phase: jax.Array        #: 0 liquid, 1 vapour, 2 two-phase
    beta: jax.Array         #: vapour fraction, mol vapour / mol feed (0 or 1 for a single phase)
    x: jax.Array            #: (Ns,) liquid composition (= feed for a single-phase result)
    y: jax.Array            #: (Ns,) vapour composition (= feed for a single-phase result)
    K: jax.Array            #: (Ns,) equilibrium ratios y/x
    converged: jax.Array    #: convergence flag of the phase split
    iterations: jax.Array   #: successive substitution iterations used by the phase split
    tpd: jax.Array          #: tangent plane distance of the stability test


def wilson_k(T, P, p: PRParams, omega=None):
    """
    Wilson estimate of the equilibrium ratios.

    The acentric factor is recovered from the slope of the alpha function at the
    critical point when it is not given.
    """
    if omega is None:
        # kappa(omega) of the original Peng-Robinson alpha, inverted for omega (mc1 plays the role of kappa)
        kappa = p.mc[:, 0]
        omega = (1.54226 - jnp.sqrt(jnp.maximum(1.54226**2 - 4*0.26992*(kappa - 0.37464), 0.0)))/(2*0.26992)
    return p.Pc/P*jnp.exp(5.373*(1.0 + omega)*(1.0 - p.Tc/T))


def rachford_rice(z, K, iterations: int = 60):
    """
    Vapour fraction ``beta`` solving ``sum_i z_i (K_i - 1) / (1 + beta (K_i - 1)) = 0``.

    The root is searched between the asymptotes ``1/(1 - K_max)`` and
    ``1/(1 - K_min)`` with a Newton iteration that falls back to bisection
    whenever a step leaves the current bracket. Values outside [0, 1] are
    returned as they are ("negative flash"): they tell on which side of the
    two-phase region the feed lies.
    """
    k_max = jnp.max(K)
    k_min = jnp.min(K)
    # no sign change of K - 1: single phase, return the limiting value
    all_vapour = k_min >= 1.0
    all_liquid = k_max <= 1.0
    lo = 1.0/(1.0 - jnp.where(k_max > 1.0, k_max, 2.0))
    hi = 1.0/(1.0 - jnp.where(k_min < 1.0, k_min, 0.5))
    eps = 1e-12*(hi - lo)
    lo, hi = lo + eps, hi - eps

    def g(b):
        d = 1.0 + b*(K - 1.0)
        return jnp.sum(z*(K - 1.0)/d), -jnp.sum(z*((K - 1.0)/d)**2)

    def body(_, state):
        lo, hi, b = state
        f, df = g(b)
        # g decreases monotonically between the asymptotes
        lo = jnp.where(f > 0.0, b, lo)
        hi = jnp.where(f > 0.0, hi, b)
        newton = b - f/jnp.where(df < -_TINY, df, -_TINY)
        b = jnp.where((newton > lo) & (newton < hi), newton, 0.5*(lo + hi))
        return lo, hi, b

    b0 = jnp.clip(0.5, lo, hi)
    _, _, beta = lax.fori_loop(0, iterations, body, (lo, hi, b0))
    return jnp.where(all_vapour, 1.0 + 1e-9, jnp.where(all_liquid, -1e-9, beta))


def _normalise(z):
    z = jnp.maximum(jnp.asarray(z, dtype=float), _TINY)
    return z/jnp.sum(z)


def stability(T, P, z, p: PRParams, max_iter: int = 200, tol: float = 1e-11) -> StabilityResult:
    """
    Michelsen tangent plane stability test of a mixture ``z`` at (T, P).

    Two trial phases (vapour-like ``z*K`` and liquid-like ``z/K`` with Wilson K)
    are converged by successive substitution on the stationarity condition

        ln W_i + ln phi_i(W) - ln z_i - ln phi_i(z) = 0

    The feed is unstable if a non-trivial stationary point has a negative
    modified tangent plane distance ``tm = 1 - sum(W)``.
    """
    z = _normalise(z)
    d = jnp.log(z) + eos.ln_phi(T, P, z, p, STABLE)
    k_w = wilson_k(T, P, p)

    v_feed = eos.molar_volume(T, P, z, p, STABLE)

    def run(W0):
        def cond(state):
            _, err, it = state
            return (err > tol) & (it < max_iter)

        def body(state):
            lnW, _, it = state
            W = jnp.exp(lnW)
            lnW_new = d - eos.ln_phi(T, P, W/jnp.sum(W), p, STABLE)
            return lnW_new, jnp.max(jnp.abs(lnW_new - lnW)), it + 1

        lnW, _, it = lax.while_loop(cond, body, (jnp.log(W0), jnp.inf, 0))
        W = jnp.exp(lnW)
        tm = 1.0 - jnp.sum(W)                       # value of the modified TPD at a stationary point
        trivial = jnp.sum((lnW - jnp.log(z))**2) < 1e-8
        tm = jnp.where(trivial, 0.0, tm)
        # the incipient phase is the vapour if it is lighter than the feed: K = y/x accordingly
        lighter = eos.molar_volume(T, P, W/jnp.sum(W), p, STABLE) > v_feed
        K = jnp.where(lighter, W/z, z/W)
        return K, tm, it

    K_v, tm_v, it_v = run(z*k_w)
    K_l, tm_l, it_l = run(z/k_w)
    tpd = jnp.minimum(tm_v, tm_l)
    stable = tpd > -1e-9
    K = jnp.where(stable, k_w, jnp.where(tm_v <= tm_l, K_v, K_l))
    return StabilityResult(stable, tpd, K, it_v + it_l)


def _split(T, P, z, K0, p: PRParams, max_iter, tol):
    """Two-phase split by successive substitution on ln K (accelerated every few steps)."""
    def phases(lnK):
        K = jnp.exp(lnK)
        beta = rachford_rice(z, K)
        x = z/(1.0 + beta*(K - 1.0))
        y = K*x
        x, y = x/jnp.sum(x), y/jnp.sum(y)
        return beta, x, y

    def cond(state):
        _, _, err, it = state
        return (err > tol) & (it < max_iter)

    def body(state):
        lnK, d_prev, _, it = state
        _, x, y = phases(lnK)
        lnK_new = eos.ln_phi(T, P, x, p, LIQUID) - eos.ln_phi(T, P, y, p, VAPOUR)
        d = lnK_new - lnK
        # dominant eigenvalue extrapolation every 5th step (speeds up the slow approach near a critical point)
        lam = jnp.sum(d*d)/jnp.where(jnp.abs(jnp.sum(d*d_prev)) > _TINY, jnp.sum(d*d_prev), 1.0)
        accelerate = (it % 5 == 4) & (lam > 0.0) & (lam < 0.95)
        lnK_next = lnK_new + jnp.where(accelerate, lam/(1.0 - lam), 0.0)*d
        return lnK_next, d, jnp.max(jnp.abs(d)), it + 1

    lnK0 = jnp.log(jnp.maximum(K0, _TINY))
    lnK, _, err, it = lax.while_loop(cond, body, (lnK0, jnp.zeros_like(lnK0), jnp.inf, 0))
    beta, x, y = phases(lnK)
    return beta, x, y, jnp.exp(lnK), err <= tol, it


def _single_phase_label(T, P, z, p: PRParams):
    """Liquid (0) or vapour (1) label for a stable single phase."""
    Z, A, B, *_ = eos._state(T, P, z, p, STABLE)
    z_l, z_v = eos.z_roots(A, B)
    two_roots = (z_v - z_l) > 1e-10
    by_root = jnp.where(jnp.abs(Z - z_l) < jnp.abs(Z - z_v), LIQUID, VAPOUR)
    # one root only: dense (v/b below the critical value 3.95) and below the pseudo-critical temperature -> liquid
    dense = (Z/B < 3.9514) & (T < jnp.sum(z*p.Tc))
    return jnp.where(two_roots, by_root, jnp.where(dense, LIQUID, VAPOUR))


def flash_pt(T, P, z, p: PRParams, max_iter: int = 2000, tol: float = 1e-11) -> FlashResult:
    """
    Isothermal, isobaric flash of a mixture ``z``.

    Parameters
    ----------
    T, P     : temperature [K] and pressure [bar]
    z        : (Ns,) feed mole fractions (normalised internally, zeros are allowed)
    p        : :class:`celeste.jx.eos.PRParams` from :func:`celeste.jx.eos.pr_params`
    max_iter : iteration limit of the phase split
    tol      : convergence tolerance on ``max |d ln K|``

    Returns
    -------
    FlashResult. For a single-phase result ``x = y = z`` and ``beta`` is 0 (liquid) or 1 (vapour).
    """
    z = _normalise(z)
    stab = stability(T, P, z, p)

    def two_phase(_):
        return _split(T, P, z, stab.K, p, max_iter, tol)

    def one_phase(_):
        return jnp.zeros(()), z, z, stab.K, jnp.array(True), jnp.array(0)

    beta, x, y, K, converged, it = lax.cond(stab.stable, one_phase, two_phase, None)

    label = _single_phase_label(T, P, z, p)
    # the split may still end outside the two-phase region for feeds that sit on a phase boundary
    is_two = (~stab.stable) & (beta > 0.0) & (beta < 1.0)
    single = jnp.where(stab.stable, label, jnp.where(beta >= 1.0, VAPOUR, LIQUID))
    phase = jnp.where(is_two, TWO_PHASE, single)
    beta = jnp.where(is_two, beta, jnp.where(single == VAPOUR, 1.0, 0.0))
    x = jnp.where(is_two, x, z)
    y = jnp.where(is_two, y, z)
    return FlashResult(phase, beta, x, y, K, converged, it, stab.tpd)
