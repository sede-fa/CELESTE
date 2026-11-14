"""
Ideal-gas mixture properties in JAX (jit / vmap / grad ready).

This is the JAX instantiation of the kernel in :mod:`celeste.jx.nasa`; the
stateful ``celeste.CSL.thermo.Thermo`` runs the very same code under NumPy.

    >>> from celeste.jx import ideal
    >>> gas = ideal.ideal_gas(['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'])
    >>> h = jax.jit(ideal.enthalpy)(900.0, x, gas)                    # one state
    >>> h = jax.vmap(ideal.enthalpy, (0, None, None))(T_array, x, gas)  # many states
    >>> cp = jax.grad(ideal.enthalpy)(900.0, x, gas)                  # exact derivative
"""
from __future__ import annotations

from typing import NamedTuple

from . import enable_x64, nasa

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

__all__ = ["IdealGas", "ideal_gas", "species_props", "mixture", "enthalpy", "entropy", "gibbs", "heat_capacity",
           "molar_mass", "R"]

R = nasa.R


class IdealGas(NamedTuple):
    """NASA-9 data of a species set as JAX arrays (a valid pytree)."""
    a: jax.Array            #: (Ns, K, 9)
    T_break: jax.Array      #: (Ns, K-1)
    T_lo: jax.Array         #: (Ns,)
    T_hi: jax.Array         #: (Ns,)
    mw: jax.Array           #: (Ns,) g/mol
    condensed: jax.Array    #: (Ns,) 1 for condensed species


def ideal_gas(species) -> IdealGas:
    """Data for an ordered list of species names of the NASA database."""
    d = nasa.nasa_data(species)
    return IdealGas(*(jnp.asarray(v) for v in (d.a, d.T_break, d.T_lo, d.T_hi, d.mw, d.condensed)))


def species_props(T, gas: IdealGas):
    """Standard-state ``(cp_i, h_i, s_i)`` of every species; kJ/mol/K, kJ/mol, kJ/mol/K."""
    return nasa.pure_props(jnp, T, gas.a, gas.T_break, gas.T_lo, gas.T_hi)


def mixture(T, P, x, gas: IdealGas):
    """Mixture ``(cp, h, s)`` at T [K], P [bar] and mole fractions x."""
    cp_i, h_i, s_i = species_props(T, gas)
    cp, h, _, s = nasa.mix_props(jnp, x, P, cp_i, h_i, s_i, gas.condensed)
    return cp, h, s


def enthalpy(T, x, gas: IdealGas):
    """Molar enthalpy in kJ/mol (includes the enthalpy of formation)."""
    return jnp.sum(x*species_props(T, gas)[1])


def heat_capacity(T, x, gas: IdealGas):
    """Molar isobaric heat capacity in kJ/mol/K."""
    return jnp.sum(x*species_props(T, gas)[0])


def entropy(T, P, x, gas: IdealGas):
    """Molar entropy in kJ/mol/K."""
    return mixture(T, P, x, gas)[2]


def gibbs(T, P, x, gas: IdealGas):
    """Molar Gibbs energy in kJ/mol."""
    _, h, s = mixture(T, P, x, gas)
    return h - T*s


def molar_mass(x, gas: IdealGas):
    """Molar mass of the mixture in g/mol."""
    return jnp.sum(x*gas.mw)
