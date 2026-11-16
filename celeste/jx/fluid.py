"""
Real-fluid mixture properties: NASA ideal gas + Peng-Robinson departure,
with vapour-liquid equilibrium resolved by :func:`celeste.jx.flash.flash_pt`.

This is the JAX replacement for an external property package such as CoolProp:
one differentiable function returns the state of a stream that may be a gas,
a liquid or a two-phase mixture.

Phase codes for the ``phase`` argument: ``LIQUID`` (0), ``VAPOUR`` (1) force a
single phase with that root of the cubic, ``AUTO`` (3) runs a flash.
"""
from __future__ import annotations

from typing import NamedTuple

from . import enable_x64

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from . import eos, flash, ideal, nasa  # noqa: E402
from .eos import LIQUID, VAPOUR  # noqa: E402

__all__ = ["Fluid", "FluidState", "fluid", "state_pt", "enthalpy_pt", "LIQUID", "VAPOUR", "AUTO"]

AUTO = 3


class Fluid(NamedTuple):
    """Property model of a species set (a valid JAX pytree)."""
    gas: ideal.IdealGas
    pr: eos.PRParams


class FluidState(NamedTuple):
    h: jax.Array        #: molar enthalpy of the stream, kJ/mol
    s: jax.Array        #: molar entropy of the stream, kJ/mol/K
    v: jax.Array        #: molar volume of the stream, m3/mol
    beta: jax.Array     #: vapour fraction, mol/mol
    phase: jax.Array    #: 0 liquid, 1 vapour, 2 two-phase
    x: jax.Array        #: (Ns,) liquid composition
    y: jax.Array        #: (Ns,) vapour composition
    h_liq: jax.Array    #: molar enthalpy of the liquid phase, kJ/mol
    h_vap: jax.Array    #: molar enthalpy of the vapour phase, kJ/mol
    s_liq: jax.Array    #: molar entropy of the liquid phase, kJ/mol/K
    s_vap: jax.Array    #: molar entropy of the vapour phase, kJ/mol/K
    converged: jax.Array


def fluid(species, kij=None) -> Fluid:
    """Property model for an ordered list of species (see :func:`celeste.jx.eos.species_available`)."""
    return Fluid(ideal.ideal_gas(eos.nasa_names(species)), eos.pr_params(species, kij))


def _phase_hsv(T, P, x, fl: Fluid, root):
    """h, s, v of one phase with composition x."""
    pp = eos.phase_props(T, P, x, fl.pr, root)
    cp_i, h_i, s_i = ideal.species_props(T, fl.gas)
    _, h_ig, _, s_ig = nasa.mix_props(jnp, x, P, cp_i, h_i, s_i, 0.0)
    return h_ig + pp.h_dep, s_ig + pp.s_dep, pp.v


def state_pt(T, P, z, fl: Fluid, phase=AUTO) -> FluidState:
    """
    State of a stream of overall composition ``z`` at T [K] and P [bar].

    With ``phase=AUTO`` the stream is flashed and may come back as two phases in
    equilibrium; ``LIQUID`` / ``VAPOUR`` skip the flash and evaluate that phase.
    """
    z = jnp.maximum(jnp.asarray(z, dtype=float), 0.0)
    z = z/jnp.sum(z)

    def auto(_):
        r = flash.flash_pt(T, P, z, fl.pr)
        return r.beta, r.x, r.y, r.phase, r.converged

    def forced(_):
        beta = jnp.where(phase == VAPOUR, 1.0, 0.0)
        return beta, z, z, jnp.where(phase == VAPOUR, VAPOUR, LIQUID), jnp.array(True)

    beta, x, y, ph, converged = jax.lax.cond(phase == AUTO, auto, forced, None)
    h_l, s_l, v_l = _phase_hsv(T, P, x, fl, LIQUID)
    h_v, s_v, v_v = _phase_hsv(T, P, y, fl, VAPOUR)
    mix = lambda liq, vap: beta*vap + (1.0 - beta)*liq
    return FluidState(mix(h_l, h_v), mix(s_l, s_v), mix(v_l, v_v), beta, ph, x, y, h_l, h_v, s_l, s_v, converged)


def enthalpy_pt(T, P, z, fl: Fluid, phase=AUTO):
    """Molar enthalpy in kJ/mol; differentiate with ``jax.jacfwd`` for the apparent heat capacity."""
    return state_pt(T, P, z, fl, phase).h
