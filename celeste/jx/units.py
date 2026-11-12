"""
Process units as Equinox modules - a prototype of the next model generation.

The units of ``celeste.CSL`` are stateful objects that are solved one after the
other. The units in this module are *pure functions with parameters*: a unit is
an :class:`equinox.Module` (an immutable pytree), calling it maps inlet streams
to outlet streams. A flowsheet written this way can be

* compiled as a whole with ``jax.jit``,
* evaluated for many operating points at once with ``jax.vmap``,
* differentiated with ``jax.grad`` / ``jax.jacfwd`` (exact sensitivities and
  Jacobians for optimisation and equation-oriented solving).

Only ideal-gas units are provided here; they reproduce their CSL counterparts
and show the pattern. See docs/jax_roadmap.md for the migration plan.

    >>> gas = ideal.ideal_gas(['O2', 'N2'])
    >>> feed = Stream(n=1.0, T=300.0, P=1.0, x=jnp.array([0.21, 0.79]))
    >>> out, work = Compressor(gas, P_out=5.0, eff=0.8)(feed)
    >>> out, duty = Heater(gas, T_out=350.0)(out)
"""
from __future__ import annotations

from . import enable_x64

enable_x64()

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from . import ideal  # noqa: E402
from ..CSL import equilibrium as eq  # noqa: E402

__all__ = ["Stream", "Heater", "Compressor", "Mixer", "ShiftReformingEquilibrium", "newton"]


class Stream(eqx.Module):
    """State of a stream: molar flow [mol/s], temperature [K], pressure [bar], mole fractions."""
    n: jax.Array
    T: jax.Array
    P: jax.Array
    x: jax.Array

    def __init__(self, n, T, P, x):
        self.n, self.T, self.P = (jnp.asarray(v, dtype=float) for v in (n, T, P))
        self.x = jnp.asarray(x, dtype=float)

    def h(self, gas: ideal.IdealGas):
        """Molar enthalpy, kJ/mol"""
        return ideal.enthalpy(self.T, self.x, gas)

    def s(self, gas: ideal.IdealGas):
        """Molar entropy, kJ/mol/K"""
        return ideal.entropy(self.T, self.P, self.x, gas)

    def H(self, gas: ideal.IdealGas):
        """Enthalpy flow, kJ/s"""
        return self.n*self.h(gas)


def newton(residual, x0, iterations: int = 40, tol: float = 1e-12):
    """
    Damped Newton iteration for ``residual(x) = 0`` with the exact Jacobian from
    forward-mode AD. A step is halved until the residual norm decreases (steps
    into regions where the residual is not finite are rejected the same way).
    The solution carries implicit derivatives with respect to everything the
    residual closes over.
    """
    def norm(x):
        r = residual(x)
        return jnp.where(jnp.all(jnp.isfinite(r)), jnp.sum(r*r), jnp.inf)

    def cond(state):
        _, f, it = state
        return (f > tol*tol) & (it < iterations)

    def body(state):
        x, f, it = state
        r = residual(x)
        J = jax.jacfwd(residual)(x)
        dx = jnp.linalg.solve(J + 1e-14*jnp.eye(x.shape[0]), -r)

        def shrink(carry):
            t, _ = carry
            return 0.5*t, norm(x + 0.5*t*dx)

        t, f_new = lax.while_loop(lambda c: (c[1] >= f) & (c[0] > 1e-10), shrink, (1.0, norm(x + dx)))
        x_new = jnp.where(f_new < f, x + t*dx, x)
        return x_new, jnp.minimum(f_new, f), it + 1

    x0 = jnp.asarray(x0, dtype=float)
    x, _, _ = lax.while_loop(cond, body, (x0, norm(x0), 0))
    # one differentiable Newton step at the solution: implicit function theorem for the derivatives
    xs = lax.stop_gradient(x)
    J = lax.stop_gradient(jax.jacfwd(residual)(xs))
    return xs - jnp.linalg.solve(J + 1e-14*jnp.eye(xs.shape[0]), residual(xs))


class Heater(eqx.Module):
    """Heat (or cool) a stream to ``T_out``. Returns ``(outlet, Q)`` with the duty Q in kJ/s."""
    gas: ideal.IdealGas
    T_out: jax.Array

    def __call__(self, feed: Stream):
        out = Stream(feed.n, self.T_out, feed.P, feed.x)
        return out, out.H(self.gas) - feed.H(self.gas)


class Compressor(eqx.Module):
    """
    Adiabatic compressor / turbine with isentropic efficiency ``eff``.
    Returns ``(outlet, W)`` with the shaft work W in kJ/s (negative for an expansion).

    The isentropic outlet temperature follows from s(T_s, P_out) = s(T_in, P_in)
    with the real temperature dependence of cp (the CSL model uses a mean cp).
    """
    gas: ideal.IdealGas
    P_out: jax.Array
    eff: jax.Array = 1.0

    def __call__(self, feed: Stream):
        gas = self.gas
        s_in, h_in = feed.s(gas), feed.h(gas)
        guess = feed.T*(self.P_out/feed.P)**(ideal.R/ideal.heat_capacity(feed.T, feed.x, gas))
        T_s = newton(lambda t: (ideal.entropy(t[0], self.P_out, feed.x, gas) - s_in)[None]*1e3, guess[None])[0]
        dh_s = ideal.enthalpy(T_s, feed.x, gas) - h_in
        dh = jnp.where(self.P_out >= feed.P, dh_s/self.eff, dh_s*self.eff)
        T_out = newton(lambda t: (ideal.enthalpy(t[0], feed.x, gas) - h_in - dh)[None], T_s[None])[0]
        return Stream(feed.n, T_out, self.P_out, feed.x), feed.n*dh


class Mixer(eqx.Module):
    """Adiabatic mixing of two streams at the pressure of the first one. Returns the outlet stream."""
    gas: ideal.IdealGas

    def __call__(self, a: Stream, b: Stream):
        gas = self.gas
        n = a.n + b.n
        x = (a.n*a.x + b.n*b.x)/n
        H = a.H(gas) + b.H(gas)
        T0 = (a.n*a.T + b.n*b.T)/n
        T = newton(lambda t: (n*ideal.enthalpy(t[0], x, gas) - H)[None], T0[None])[0]
        return Stream(n, T, a.P, x)


class ShiftReformingEquilibrium(eqx.Module):
    """
    Isothermal reactor at water-gas-shift and methane-reforming equilibrium -
    the Equinox counterpart of ``celeste.CSL.reactors.MR``.

    Species order of ``media.Fuel_CH4``: H2, CH4, CO2, CO, H2O, N2. The two
    reaction extents are the only unknowns. Returns ``(outlet, Q)``.
    """
    gas: ideal.IdealGas
    T: jax.Array
    P: jax.Array

    #                      H2    CH4   CO2   CO    H2O   N2
    _nu = jnp.array([[1.0, 0.0, 1.0, -1.0, -1.0, 0.0],      # CO + H2O <-> CO2 + H2
                     [3.0, -1.0, 0.0, 1.0, -1.0, 0.0]])     # CH4 + H2O <-> CO + 3 H2

    def __call__(self, feed: Stream):
        n0 = feed.n*feed.x
        K_wgs, K_msr = eq.K_WGS(self.T, jnp.exp), eq.K_MSR(self.T, jnp.exp)

        def moles(xi):
            return n0 + xi @ self._nu

        def residual(xi):
            # logarithmic form of the two mass action laws: well scaled also for trace species
            n = moles(xi)
            reacting = jnp.any(self._nu != 0.0, axis=0)
            ok = jnp.all((n > 0.0) | ~reacting)
            ln_x = jnp.log(jnp.where(ok & reacting, n, 1.0)/jnp.sum(n))
            r = self._nu @ ln_x + jnp.stack([-jnp.log(K_wgs), 2.0*jnp.log(self.P) - jnp.log(K_msr)])
            return jnp.where(ok, r, jnp.nan)

        # Start strictly inside the feasible region (all mole numbers positive). Species missing from the
        # feed are created by moving each reaction in turn to the middle of the range it currently allows.
        xi0 = jnp.zeros(2)
        for r in (0, 1, 0, 1):
            n, nu = moles(xi0), self._nu[r]
            lo = jnp.max(jnp.where(nu > 0, -n/jnp.where(nu > 0, nu, 1.0), -jnp.inf))
            hi = jnp.min(jnp.where(nu < 0, -n/jnp.where(nu < 0, nu, -1.0), jnp.inf))
            xi0 = xi0.at[r].add(0.5*(lo + hi))
        xi = newton(residual, xi0, iterations=100)
        n = moles(xi)
        out = Stream(jnp.sum(n), self.T, self.P, n/jnp.sum(n))
        return out, out.H(self.gas) - Stream(feed.n, self.T, feed.P, feed.x).H(self.gas)
