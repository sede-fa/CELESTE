"""
Peng-Robinson equation of state for mixtures, written in JAX.

    P = R T / (v - b) - a(T) / (v^2 + 2 b v - b^2)

* alpha function: Mathias-Copeman below the critical temperature (parameters
  fitted to saturation pressures, see ``tools/build_eos_table.py``) and the
  Boston-Mathias extrapolation above it (monotonic for T >> Tc),
* van der Waals one-fluid mixing rules with binary interaction parameters k_ij,
* compressibility factor, fugacity coefficients, enthalpy / entropy departure.

Every function is pure, works on JAX arrays and can be passed through
``jax.jit``, ``jax.vmap`` and ``jax.grad``. Units: T in K, P in bar, molar
volume in m3/mol, departure enthalpy in kJ/mol, departure entropy in kJ/mol/K.

Phase selection uses the integer codes ``LIQUID`` (smallest root), ``VAPOUR``
(largest root) and ``STABLE`` (the root with the lower Gibbs energy).
"""
from __future__ import annotations

import csv
from functools import lru_cache
from importlib.resources import files
from typing import NamedTuple

import numpy as np

from . import enable_x64

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

__all__ = [
    "R_BAR", "R_KJ", "LIQUID", "VAPOUR", "STABLE", "PRParams", "pr_params", "species_available", "default_kij",
    "alpha", "pure_ab", "mixture_ab", "z_factor", "ln_phi", "departure", "molar_volume", "PhaseProps", "phase_props",
]

R_BAR = 8.314462618e-5      #: gas constant, bar m3/mol/K
R_KJ = 8.314462618e-3       #: gas constant, kJ/mol/K
_BAR_M3_TO_KJ = 100.0       # 1 bar m3 = 100 kJ

LIQUID, VAPOUR, STABLE = 0, 1, 2

_OMEGA_A = 0.45723552892138218
_OMEGA_B = 0.07779607390388846
_SQRT2 = 1.4142135623730951
_D1, _D2 = 1.0 + _SQRT2, 1.0 - _SQRT2

#: Binary interaction parameters k_ij for the Peng-Robinson equation (symmetric, default 0).
#: Hydrocarbon / CO2 / N2 pairs: Knapp et al., DECHEMA Chemistry Data Series VI (1982).
#: Pairs with water or methanol are indicative values only - check them against data for design work.
DEFAULT_KIJ = {
    ("CH4", "CO2"): 0.0919, ("CH4", "N2"): 0.0311, ("CH4", "C2H6"): -0.0026, ("CH4", "C3H8"): 0.0140,
    ("CH4", "C4H10"): 0.0133, ("C2H6", "C3H8"): 0.0011, ("C2H6", "CO2"): 0.1322, ("C3H8", "CO2"): 0.1241,
    ("C4H10", "CO2"): 0.1333, ("CO2", "N2"): -0.0170, ("C2H6", "N2"): 0.0515, ("C3H8", "N2"): 0.0852,
    ("CO", "N2"): 0.0307, ("CH4", "CO"): 0.0300, ("CO2", "H2"): -0.1622, ("CH4", "H2"): 0.0156,
    ("H2", "N2"): 0.1030, ("CO", "H2"): 0.0919,
    ("CO2", "H2O"): 0.1200, ("CH3OH", "H2O"): -0.0778, ("CH3OH", "CO2"): 0.0230,
}


class PRParams(NamedTuple):
    """Peng-Robinson parameters of a species set (JAX arrays; a valid pytree)."""
    Tc: jax.Array       #: (Ns,) critical temperature, K
    Pc: jax.Array       #: (Ns,) critical pressure, bar
    mc: jax.Array       #: (Ns, 3) Mathias-Copeman parameters
    kij: jax.Array      #: (Ns, Ns) binary interaction parameters


@lru_cache(maxsize=1)
def _table() -> dict:
    out = {}
    with files("celeste.data").joinpath("eos_species.csv").open("r") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for r in rows:
            out[r["species"]] = dict(nasa=r["nasa"], Tc=float(r["Tc_K"]), Pc=float(r["Pc_bar"]),
                                     omega=float(r["omega"]), mc=(float(r["mc1"]), float(r["mc2"]), float(r["mc3"])))
    return out


def species_available() -> list[str]:
    """Species that have Peng-Robinson parameters."""
    return list(_table())


def eos_name(species: str) -> str:
    """Name of a species in the EOS table (``'H2O(L)' -> 'H2O'``)."""
    name = str(species)
    return name[:-3] if name.endswith("(L)") else name


def nasa_names(species) -> list[str]:
    """Names of the ideal-gas species in the NASA database for a list of EOS species."""
    tab = _table()
    return [tab[eos_name(s)]["nasa"] for s in species]


def default_kij(species) -> np.ndarray:
    """Symmetric k_ij matrix from :data:`DEFAULT_KIJ` (zero for pairs without an entry)."""
    names = [eos_name(s) for s in species]
    k = np.zeros((len(names), len(names)))
    for i, si in enumerate(names):
        for j, sj in enumerate(names):
            if i != j:
                k[i, j] = DEFAULT_KIJ.get((si, sj), DEFAULT_KIJ.get((sj, si), 0.0))
    return k


def pr_params(species, kij=None) -> PRParams:
    """
    Parameters for an ordered list of species.

    Parameters
    ----------
    species : names from :func:`species_available` (a trailing ``(L)`` is ignored)
    kij     : ``None`` for the built-in values, a full (Ns, Ns) matrix, or a dict
              ``{('CO2', 'H2O'): 0.12, ...}`` that overrides single pairs
    """
    tab = _table()
    names = [eos_name(s) for s in species]
    missing = [s for s in names if s not in tab]
    if missing:
        raise KeyError(f"No Peng-Robinson parameters for {missing}. Available: {species_available()}")
    if kij is None or isinstance(kij, dict):
        k = default_kij(names)
        for (si, sj), val in (kij or {}).items():
            i, j = names.index(eos_name(si)), names.index(eos_name(sj))
            k[i, j] = k[j, i] = val
    else:
        k = np.asarray(kij, dtype=float)
        if k.shape != (len(names), len(names)):
            raise ValueError(f"kij must have shape {(len(names), len(names))}, got {k.shape}")
    return PRParams(Tc=jnp.array([tab[s]["Tc"] for s in names]), Pc=jnp.array([tab[s]["Pc"] for s in names]),
                    mc=jnp.array([tab[s]["mc"] for s in names]), kij=jnp.asarray(k))


# --------------------------------------------------------------------------
# Pure component and mixture parameters
# --------------------------------------------------------------------------
def alpha(T, p: PRParams):
    """Temperature dependence of the attraction parameter, shape (Ns,)."""
    Tr = T/p.Tc
    c1, c2, c3 = p.mc[:, 0], p.mc[:, 1], p.mc[:, 2]
    # sub-critical: Mathias-Copeman
    x = 1.0 - jnp.sqrt(jnp.minimum(Tr, 1.0))
    sub = (1.0 + c1*x + c2*x**2 + c3*x**3)**2
    # super-critical: Boston-Mathias (same value and slope at Tr = 1, decays monotonically)
    d = 1.0 + 0.5*c1
    c = 1.0 - 1.0/d
    sup = jnp.exp(2.0*c*(1.0 - jnp.maximum(Tr, 1.0)**d))
    return jnp.where(Tr <= 1.0, sub, sup)


def pure_ab(T, p: PRParams):
    """Pure component a_i(T) [bar m6/mol2] and b_i [m3/mol]."""
    a = _OMEGA_A*(R_BAR*p.Tc)**2/p.Pc*alpha(T, p)
    b = _OMEGA_B*R_BAR*p.Tc/p.Pc
    return a, b


def mixture_ab(T, x, p: PRParams):
    """
    One-fluid mixing rules.

    Returns ``(a, b, a_i, b_i, s_i)`` with ``s_i = sum_j x_j a_ij``.
    """
    a_i, b_i = pure_ab(T, p)
    sq = jnp.sqrt(a_i)
    a_ij = sq[:, None]*sq[None, :]*(1.0 - p.kij)
    s_i = a_ij @ x
    return x @ s_i, x @ b_i, a_i, b_i, s_i


# --------------------------------------------------------------------------
# Cubic
# --------------------------------------------------------------------------
def _cubic_residual(Z, A, B):
    return ((Z - (1.0 - B))*Z + (A - 3.0*B*B - 2.0*B))*Z - (A*B - B*B - B**3)


def _cubic_slope(Z, A, B):
    return (3.0*Z - 2.0*(1.0 - B))*Z + (A - 3.0*B*B - 2.0*B)


def _real_roots(A, B):
    """The three roots of the PR cubic as (r1, r2, r3, three_real); for a single real root r1 = r2 = r3."""
    c2 = -(1.0 - B)
    c1 = A - 3.0*B*B - 2.0*B
    c0 = -(A*B - B*B - B**3)
    p = c1 - c2*c2/3.0
    q = 2.0*c2**3/27.0 - c2*c1/3.0 + c0
    disc = 0.25*q*q + p**3/27.0
    three = disc < 0.0

    # one real root (Cardano, written to avoid cancellation)
    sd = jnp.sqrt(jnp.where(three, 0.0, disc))
    sgn = jnp.where(q >= 0.0, 1.0, -1.0)
    u = jnp.cbrt(-0.5*q - sgn*sd)
    v = jnp.where(u != 0.0, -p/(3.0*jnp.where(u != 0.0, u, 1.0)), 0.0)
    t_single = u + v

    # three real roots (trigonometric form); p < 0 is guaranteed in this branch
    pm = jnp.where(three, p, -1.0)
    m = 2.0*jnp.sqrt(-pm/3.0)
    arg = jnp.clip(3.0*q/(pm*m), -1.0, 1.0)
    th = jnp.arccos(arg)/3.0
    shift = -c2/3.0
    r_a = jnp.where(three, m*jnp.cos(th), t_single) + shift
    r_b = jnp.where(three, m*jnp.cos(th - 2.0943951023931953), t_single) + shift
    r_c = jnp.where(three, m*jnp.cos(th - 4.1887902047863905), t_single) + shift
    return r_a, r_b, r_c, three


def _polish(Z, A, B):
    """Two Newton steps on the cubic (guards against a vanishing slope)."""
    for _ in range(2):
        f, df = _cubic_residual(Z, A, B), _cubic_slope(Z, A, B)
        step = jnp.where(jnp.abs(df) > 1e-14, f/jnp.where(jnp.abs(df) > 1e-14, df, 1.0), 0.0)
        Z = Z - step
    return Z


def z_roots(A, B):
    """
    Liquid-like and vapour-like compressibility factor ``(Z_liquid, Z_vapour)``.

    The smallest and the largest root above ``B`` are returned; when the cubic
    has a single admissible root both values coincide. Gradients with respect
    to ``A`` and ``B`` follow from the implicit function theorem.
    """
    r_a, r_b, r_c, _ = _real_roots(A, B)
    roots = jnp.stack([_polish(r, A, B) for r in (r_a, r_b, r_c)])
    valid = roots > B*(1.0 + 1e-12)
    z_v = jnp.max(jnp.where(valid, roots, -jnp.inf))
    z_l = jnp.min(jnp.where(valid, roots, jnp.inf))
    # no admissible root can only come from round-off in a degenerate case: fall back to the ideal gas limit
    z_v = jnp.where(jnp.isfinite(z_v), z_v, 1.0 + B)
    z_l = jnp.where(jnp.isfinite(z_l), z_l, z_v)

    def implicit(z):
        z0 = jax.lax.stop_gradient(z)
        df = _cubic_slope(z0, A, B)
        df = jnp.where(jnp.abs(df) > 1e-14, df, 1e-14)
        return z0 - _cubic_residual(z0, A, B)/jax.lax.stop_gradient(df)

    return implicit(z_l), implicit(z_v)


def _log_term(Z, B):
    return jnp.log((Z + _D1*B)/(Z + _D2*B))


def _ln_phi_root(Z, A, B, a, b, b_i, s_i):
    return b_i/b*(Z - 1.0) - jnp.log(Z - B) - A/(2.0*_SQRT2*B)*(2.0*s_i/a - b_i/b)*_log_term(Z, B)


def _gibbs_departure_rt(Z, A, B):
    """Residual Gibbs energy g^R/(RT) of the mixture for a root Z."""
    return Z - 1.0 - jnp.log(Z - B) - A/(2.0*_SQRT2*B)*_log_term(Z, B)


def _select(phase, z_l, z_v, A, B):
    """Root for the requested phase code."""
    stable = jnp.where(_gibbs_departure_rt(z_l, A, B) < _gibbs_departure_rt(z_v, A, B), z_l, z_v)
    return jnp.where(phase == LIQUID, z_l, jnp.where(phase == VAPOUR, z_v, stable))


def _state(T, P, x, p: PRParams, phase):
    a, b, a_i, b_i, s_i = mixture_ab(T, x, p)
    A = a*P/(R_BAR*T)**2
    B = b*P/(R_BAR*T)
    z_l, z_v = z_roots(A, B)
    Z = _select(phase, z_l, z_v, A, B)
    return Z, A, B, a, b, b_i, s_i


def z_factor(T, P, x, p: PRParams, phase=STABLE):
    """Compressibility factor Z = P v / (R T)."""
    return _state(T, P, x, p, phase)[0]


def molar_volume(T, P, x, p: PRParams, phase=STABLE):
    """Molar volume in m3/mol."""
    return z_factor(T, P, x, p, phase)*R_BAR*T/P


def ln_phi(T, P, x, p: PRParams, phase=STABLE):
    """Natural logarithm of the fugacity coefficients, shape (Ns,)."""
    Z, A, B, a, b, b_i, s_i = _state(T, P, x, p, phase)
    return _ln_phi_root(Z, A, B, a, b, b_i, s_i)


def departure(T, P, x, p: PRParams, phase=STABLE):
    """
    Departure from the ideal gas at the same T, P and composition.

    Returns ``(h_dep [kJ/mol], s_dep [kJ/mol/K], Z)``.
    """
    Z, A, B, a, b, _, _ = _state(T, P, x, p, phase)
    # temperature derivative of the mixture attraction parameter by forward-mode AD
    _, da_dT = jax.jvp(lambda t: mixture_ab(t, x, p)[0], (jnp.asarray(T, dtype=float),), (jnp.ones(()),))
    L = _log_term(Z, B)
    h_dep = R_KJ*T*(Z - 1.0) + (T*da_dT - a)/(2.0*_SQRT2*b)*L*_BAR_M3_TO_KJ
    s_dep = R_KJ*jnp.log(Z - B) + da_dT/(2.0*_SQRT2*b)*L*_BAR_M3_TO_KJ
    return h_dep, s_dep, Z


class PhaseProps(NamedTuple):
    """Properties of a single phase."""
    Z: jax.Array          #: compressibility factor
    v: jax.Array          #: molar volume, m3/mol
    ln_phi: jax.Array     #: (Ns,) ln fugacity coefficients
    h_dep: jax.Array      #: departure enthalpy, kJ/mol
    s_dep: jax.Array      #: departure entropy, kJ/mol/K


def phase_props(T, P, x, p: PRParams, phase=STABLE) -> PhaseProps:
    """Everything the equation of state knows about one phase, in one call."""
    Z, A, B, a, b, b_i, s_i = _state(T, P, x, p, phase)
    _, da_dT = jax.jvp(lambda t: mixture_ab(t, x, p)[0], (jnp.asarray(T, dtype=float),), (jnp.ones(()),))
    L = _log_term(Z, B)
    h_dep = R_KJ*T*(Z - 1.0) + (T*da_dT - a)/(2.0*_SQRT2*b)*L*_BAR_M3_TO_KJ
    s_dep = R_KJ*jnp.log(Z - B) + da_dT/(2.0*_SQRT2*b)*L*_BAR_M3_TO_KJ
    return PhaseProps(Z, Z*R_BAR*T/P, _ln_phi_root(Z, A, B, a, b, b_i, s_i), h_dep, s_dep)
