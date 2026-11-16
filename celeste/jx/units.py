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

import numpy as np  # noqa: E402

__all__ = ["Stream", "Heater", "Compressor", "Mixer", "Splitter", "Separator", "HeatExchanger",
           "ShiftReformingEquilibrium", "SOC", "newton", "flows", "stream_vector", "stream_from_vector"]

F_CONST = 96485.0      #: Faraday constant, C/mol (as in celeste.CSL.electrochem)


class Stream(eqx.Module):
    """State of a stream: molar flow [mol/s], temperature [K], pressure [bar], mole fractions."""
    n: jax.Array
    T: jax.Array
    P: jax.Array
    x: jax.Array

    def __init__(self, n, T, P, x):
        self.n, self.T, self.P = (jnp.asarray(v, dtype=float) for v in (n, T, P))
        self.x = jnp.asarray(x, dtype=float)

    def replace(self, **changes) -> "Stream":
        """
        A copy with some of ``n``, ``T``, ``P``, ``x`` changed, e.g. ``feed.replace(T=600.0, P=20.0)``.
        Streams are immutable (Equinox modules), so this is how a state is "set"; it works under
        ``jit``, ``vmap`` and ``grad``.
        """
        unknown = set(changes) - {'n', 'T', 'P', 'x'}
        if unknown:
            raise TypeError(f"Stream has no field(s) {sorted(unknown)}; use n, T, P, x")
        return Stream(**{k: changes.get(k, getattr(self, k)) for k in ('n', 'T', 'P', 'x')})

    def h(self, gas: ideal.IdealGas):
        """Molar enthalpy, kJ/mol"""
        return ideal.enthalpy(self.T, self.x, gas)

    def s(self, gas: ideal.IdealGas):
        """Molar entropy, kJ/mol/K"""
        return ideal.entropy(self.T, self.P, self.x, gas)

    def H(self, gas: ideal.IdealGas):
        """Enthalpy flow, kJ/s"""
        return self.n*self.h(gas)


def flows(s: Stream):
    """Species flows n*x [mol/s]."""
    return s.n*s.x


def stream_vector(s: Stream):
    """The variables of a stream in an equation-oriented model: species flows and temperature."""
    return jnp.append(flows(s), s.T)


def stream_from_vector(v, P) -> Stream:
    """Inverse of :func:`stream_vector` at pressure P."""
    n = v[:-1]
    return Stream(jnp.sum(n), v[-1], P, n/jnp.sum(n))


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

    def residuals(self, feed: Stream, out: Stream):
        """Equations of the unit for an equation-oriented model (zero at the solution)."""
        return jnp.append((flows(out) - flows(feed))/feed.n, (out.T - self.T_out)/self.T_out)


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

    def residuals(self, a: Stream, b: Stream, out: Stream):
        """Species balances and energy balance (zero at the solution)."""
        n = a.n + b.n
        return jnp.append((flows(out) - flows(a) - flows(b))/n,
                          (out.H(self.gas) - a.H(self.gas) - b.H(self.gas))/(n*ideal.R*1e3))


class Splitter(eqx.Module):
    """Splits a stream: the fraction ``split`` goes to the first outlet, the rest to the second."""
    split: jax.Array

    def __call__(self, feed: Stream):
        return feed.replace(n=self.split*feed.n), feed.replace(n=(1 - self.split)*feed.n)

    def residuals(self, feed: Stream, out1: Stream, out2: Stream):
        return jnp.concatenate(((flows(out1) - self.split*flows(feed))/feed.n,
                                (flows(out2) - (1 - self.split)*flows(feed))/feed.n,
                                jnp.stack([out1.T - feed.T, out2.T - feed.T])/feed.T))


class Separator(eqx.Module):
    """
    Black-box separator (the counterpart of ``CSL.components.GPU_2SV2``): of every species the fraction
    ``recovery[i]`` goes to the first outlet, the rest to the second; both leave at ``T_out`` and the inlet
    pressure. Returns ``(out1, out2, Q)`` with the duty Q in kJ/s.
    """
    gas: ideal.IdealGas
    recovery: jax.Array
    T_out: jax.Array

    def __call__(self, feed: Stream):
        n1, n2 = self.recovery*flows(feed), (1 - self.recovery)*flows(feed)
        out1 = Stream(jnp.sum(n1), self.T_out, feed.P, n1/jnp.sum(n1))
        out2 = Stream(jnp.sum(n2), self.T_out, feed.P, n2/jnp.sum(n2))
        return out1, out2, out1.H(self.gas) + out2.H(self.gas) - feed.H(self.gas)

    def residuals(self, feed: Stream, out1: Stream, out2: Stream):
        return jnp.concatenate(((flows(out1) - self.recovery*flows(feed))/feed.n,
                                (flows(out2) - (1 - self.recovery)*flows(feed))/feed.n,
                                (jnp.stack([out1.T, out2.T]) - self.T_out)/self.T_out))


class HeatExchanger(eqx.Module):
    """
    Counter-current heat exchanger (recuperator) with the effectiveness ``eff``: the duty is ``eff`` times the
    largest duty the two streams allow (each heated or cooled to the inlet temperature of the other), with the
    full temperature dependence of the heat capacities. Returns ``(hot_out, cold_out, Q)``, Q in kJ/s.
    """
    gas_hot: ideal.IdealGas
    gas_cold: ideal.IdealGas
    eff: jax.Array

    def duty(self, hot: Stream, cold: Stream):
        Q_max = jnp.minimum(hot.H(self.gas_hot) - hot.replace(T=cold.T).H(self.gas_hot),
                            cold.replace(T=hot.T).H(self.gas_cold) - cold.H(self.gas_cold))
        return self.eff*Q_max

    def __call__(self, hot: Stream, cold: Stream):
        Q = self.duty(hot, cold)

        def outlet(s, gas, H):
            T = newton(lambda t: (s.n*ideal.enthalpy(t[0], s.x, gas) - H)[None]/s.n, s.T[None])[0]
            return s.replace(T=T)
        return outlet(hot, self.gas_hot, hot.H(self.gas_hot) - Q), outlet(cold, self.gas_cold, cold.H(self.gas_cold) + Q), Q

    def residuals(self, hot: Stream, cold: Stream, hot_out: Stream, cold_out: Stream):
        Q, scale = self.duty(hot, cold), (hot.n + cold.n)*ideal.R*1e3
        return jnp.concatenate(((flows(hot_out) - flows(hot))/hot.n, (flows(cold_out) - flows(cold))/cold.n,
                                jnp.stack([hot.H(self.gas_hot) - hot_out.H(self.gas_hot) - Q,
                                           cold_out.H(self.gas_cold) - cold.H(self.gas_cold) - Q])/scale))


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


class SOC(eqx.Module):
    r"""
    Solid oxide cell stack - the Equinox counterpart of :class:`celeste.CSL.electrochem.SOC`, built on the
    same equations (:func:`celeste.CSL.equilibrium.fuel_residuals`).

    Fuel side H2, CH4, CO2, CO, H2O, N2 at water-gas-shift and reforming equilibrium; air side O2, N2. The
    current fixes the oxygen exchanged, n_O = Ncells*I/(2F) (I < 0: electrolysis). Ideal voltage: the Nernst
    potential averaged over the charge passed ('mean' of the CSL model); operating voltage with the ASR of
    M. Riedel 2020.

    Energy specification: ``T`` (isothermal stack) or ``Q`` (heat added, kJ/s; 0 is adiabatic).
    Calling the unit solves it: ``fuel_out, air_out, W, Q = soc(fuel, air)`` with the electrical work W and the
    heat Q added, kJ/s. ``residuals`` gives its equations for an equation-oriented flowsheet.
    """
    fuel: ideal.IdealGas
    air: ideal.IdealGas
    I: jax.Array
    Ncells: jax.Array
    cellArea: jax.Array = 100.0
    f_ASR: jax.Array = 1.0
    T: jax.Array | None = None
    Q: jax.Array | None = None
    Qloss: jax.Array = 0.0
    n_nodes: int = eqx.field(static=True, default=12)

    def __check_init__(self):
        if (self.T is None) == (self.Q is None):
            raise ValueError("SOC: give either T (isothermal) or Q (heat added; 0 is adiabatic)")

    # ---- charge, voltage, work ----------------------------------------------------------------------
    @property
    def n_O(self):
        """Oxygen transferred to the fuel side, mol/s (negative in electrolysis)."""
        return self.Ncells*self.I/2/F_CONST

    def air_flows(self, air: Stream, lam=1.0):
        return flows(air) - lam*jnp.array([0.5, 0.0])*self.n_O

    @staticmethod
    def nernst(T, P, x_H2, x_H2O, x_O2):
        """Nernst potential of H2 + 1/2 O2 -> H2O, V."""
        dG0 = -ideal.R*T*jnp.log(eq.K_eq(T, 'H2_ox', jnp.exp))
        dG = dG0 + ideal.R*T*(jnp.log(x_H2O) - jnp.log(x_H2) - 0.5*jnp.log(x_O2*P))
        return -dG*1e3/(2*F_CONST)

    @staticmethod
    def _guess(n0, O_in):
        """
        Starting point of the fuel-side solve: the oxygen exchanged moves H2O <-> H2 first, then CO2 <-> CO
        (each by at most 95 % of what is there); ln of the flows, kept away from zero.
        """
        H2, CH4, CO2, CO, H2O, N2 = n0
        out = jnp.maximum(-O_in, 0.0)                         # oxygen removed (electrolysis)
        a = jnp.minimum(out, 0.95*H2O)
        b = jnp.minimum(out - a, 0.95*CO2)
        inn = jnp.maximum(O_in, 0.0)                          # oxygen added (fuel cell)
        c = jnp.minimum(inn, 0.95*H2)
        d = jnp.minimum(inn - c, 0.95*CO)
        n = jnp.stack([H2 + a - c, CH4, CO2 - b + d, CO + b - d, H2O - a + c, N2])
        return jnp.log(jnp.maximum(n, 1e-9*jnp.sum(n0)))

    def _equilibrium(self, n0, T, P, lam, ln_guess):
        return newton(lambda u: eq.fuel_residuals(jnp.exp(u), n0, T, P, O_in=lam*self.n_O, xp=jnp), ln_guess,
                      iterations=60)

    def V_mean(self, fuel: Stream, air: Stream, fuel_out: Stream):
        """Nernst potential averaged over the charge passed (Gauss-Legendre in the charge fraction), V."""
        nodes, weights = np.polynomial.legendre.leggauss(self.n_nodes)
        lam, w = 0.5*(jnp.asarray(nodes) + 1), 0.5*jnp.asarray(weights)
        n0 = flows(fuel)

        def node(l):
            T = fuel.T + l*(fuel_out.T - fuel.T)
            n = jnp.exp(self._equilibrium(n0, T, fuel.P, l, self._guess(n0, l*self.n_O)))
            na = self.air_flows(air, l)
            return self.nernst(T, fuel.P, n[0]/jnp.sum(n), n[4]/jnp.sum(n), na[0]/jnp.sum(na))
        return jnp.sum(w*jax.vmap(node)(lam))

    def ASR(self, T):
        """Area specific resistance, Ohm cm2 (M. Riedel 2020, DOI 10.1149/1945-7111/ab6820)."""
        return 7.28e2*jnp.exp(-8.29e-3*(T - 273.15))

    def voltage(self, fuel: Stream, air: Stream, fuel_out: Stream):
        """(ideal, operating) cell voltage, V."""
        nodes, weights = np.polynomial.legendre.leggauss(4)
        T = fuel.T + 0.5*(jnp.asarray(nodes) + 1)*(fuel_out.T - fuel.T)
        ASR = 0.5*jnp.sum(jnp.asarray(weights)*self.ASR(T))
        Vid = self.V_mean(fuel, air, fuel_out)
        return Vid, Vid - self.f_ASR*ASR*self.I/self.cellArea

    def work(self, fuel: Stream, air: Stream, fuel_out: Stream):
        """Electrical work added, kJ/s (positive in electrolysis)."""
        return self.voltage(fuel, air, fuel_out)[1]*(-self.I)*self.Ncells*1e-3

    def heat(self, fuel, air, fuel_out, air_out, W):
        """Heat added, kJ/s (net of Qloss)."""
        return (fuel_out.H(self.fuel) + air_out.H(self.air) - fuel.H(self.fuel) - air.H(self.air)) - W - self.Qloss

    # ---- equations and solution ---------------------------------------------------------------------
    def _outlets(self, fuel: Stream, air: Stream, n, T):
        na = self.air_flows(air)
        return (Stream(jnp.sum(n), T, fuel.P, n/jnp.sum(n)), Stream(jnp.sum(na), T, fuel.P, na/jnp.sum(na)))

    def residuals(self, fuel: Stream, air: Stream, fuel_out: Stream, air_out: Stream):
        """Element balances and equilibria of the fuel side, air-side balance, energy specification."""
        T = fuel_out.T
        r = [eq.fuel_residuals(flows(fuel_out), flows(fuel), T, fuel.P, O_in=self.n_O, xp=jnp),
             (flows(air_out) - self.air_flows(air))/air.n, jnp.stack([air_out.T - T])/T]
        if self.Q is None:
            r.append(jnp.stack([T - self.T])/self.T)
        else:
            W = self.work(fuel, air, fuel_out)
            r.append(jnp.stack([self.heat(fuel, air, fuel_out, air_out, W) - self.Q])/(fuel.n*ideal.R*1e3))
        return jnp.concatenate(r)

    def __call__(self, fuel: Stream, air: Stream):
        n0 = flows(fuel)
        ln_feed = self._guess(n0, self.n_O)
        if self.Q is None:
            ln_n = self._equilibrium(n0, self.T, fuel.P, 1.0, ln_feed)
            fuel_out, air_out = self._outlets(fuel, air, jnp.exp(ln_n), self.T)
        else:
            def residual(x):
                f_out, a_out = self._outlets(fuel, air, jnp.exp(x[:6]), x[6])
                return self.residuals(fuel, air, f_out, a_out)[jnp.array([0, 1, 2, 3, 4, 5, 9])]
            ln_T = jnp.append(self._equilibrium(n0, fuel.T, fuel.P, 1.0, ln_feed), fuel.T)
            x = newton(residual, ln_T, iterations=60)
            fuel_out, air_out = self._outlets(fuel, air, jnp.exp(x[:6]), x[6])
        W = self.work(fuel, air, fuel_out)
        return fuel_out, air_out, W, self.heat(fuel, air, fuel_out, air_out, W)
