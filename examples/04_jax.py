"""
The JAX layer: batched properties, exact derivatives, a compiled flowsheet.

Run with:  python examples/04_jax.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from celeste.CSL import media
from celeste.jx import eos, flash, ideal, units

# --- 1. properties for 100 000 states at once ----------------------------------------------------------
gas = ideal.ideal_gas(['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'])
x = jnp.array([0.6, 0.05, 0.1, 0.05, 0.2, 0.0])
T = jnp.linspace(300.0, 1500.0, 100_000)
enthalpy = jax.jit(jax.vmap(ideal.enthalpy, (0, None, None)))
enthalpy(T, x, gas)                                    # compile
t = time.perf_counter(); h = enthalpy(T, x, gas).block_until_ready(); t_jax = time.perf_counter() - t

ref = media.Fuel_CH4(); ref.Xi = np.asarray(x)
t = time.perf_counter()
for Ti in np.asarray(T[:2000]):
    ref.T = float(Ti)
t_loop = (time.perf_counter() - t)*50
print(f'100 000 enthalpies: {t_jax*1e3:.1f} ms vectorised, {t_loop:.2f} s one object at a time')

# --- 2. derivatives without finite differences ---------------------------------------------------------
cp = jax.grad(ideal.enthalpy)(800.0, x, gas)
print(f'cp = dh/dT at 800 K: {float(cp)*1e3:.4f} J/mol/K (automatic differentiation)')

# --- 3. a flowsheet as one differentiable function -----------------------------------------------------
def methane_slip(T_reactor, P):
    """Heater + equilibrium reactor; returns the methane leaving the reactor in mol/s."""
    feed = units.Stream(n=1.0, T=500.0, P=P, x=jnp.array([0.6, 0.01, 0.2, 0.1, 0.08, 0.01]))
    hot, duty = units.Heater(gas, T_reactor)(feed)
    out, q_reaction = units.ShiftReformingEquilibrium(gas, T_reactor, P)(hot)
    return out.n*out.x[1]

sweep = jax.jit(jax.vmap(methane_slip, (0, None)))(jnp.linspace(500.0, 1000.0, 6), 10.0)
print('methane vs temperature :', np.round(np.asarray(sweep), 4))
print(f'd(methane)/dT at 700 K  : {float(jax.grad(methane_slip)(700.0, 10.0)):.3e} mol/s/K')

# --- 4. phase equilibrium on a grid --------------------------------------------------------------------
params = eos.pr_params(['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'])
z = jnp.array([0.6, 0.05, 0.1, 0.05, 0.2, 0.0])
grid_T, grid_P = jnp.meshgrid(jnp.linspace(280.0, 460.0, 7), jnp.array([1.0, 20.0, 80.0]))
beta = jax.jit(jax.vmap(jax.vmap(lambda t, p: flash.flash_pt(t, p, z, params).beta)))(grid_T, grid_P)
print('\nvapour fraction (rows: 1, 20, 80 bar; columns: 280 ... 460 K)')
print(np.round(np.asarray(beta), 3))
