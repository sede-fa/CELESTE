"""
An SOEC system built from Equinox units, solved two ways:

* sequential modular (SM): the units are called in flowsheet order and the recycle is converged on its tear
  stream - the way the CSL systems run, but as one compiled function;
* equation oriented (EO): every stream is an unknown, every unit contributes its equations, and Newton's
  method with the exact Jacobian (``jax.jacfwd``) solves them all at once, starting from the SM solution.
  EO then also solves design specifications directly (here: the current for a given hydrogen production).

The plant is a ``jx.Flowsheet``: its streams are declared once with ``link`` (as in CSL); the solve order, the
tear streams, the unknowns and the equations are generated from them (``plant.print_equations()``).

Flowsheet (fuel side; the air enters the stack at the stack temperature):

    steam --> MIX --> REC(cold) --> HT --> SOC --> REC(hot) --> KO --> SPL --> hydrogen product
               ^                                                 |      |
               |                                                 v      |
               |                                               water    |
               +--------------------- hydrogen recycle -----------------+

Run with:  python examples/05_soec_flowsheet.py
"""
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from celeste.jx import ideal, units
from celeste.jx.flowsheet import Flowsheet
from celeste.jx.units import Stream, flows

FUEL = ideal.ideal_gas(['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'])
AIR = ideal.ideal_gas(['O2', 'N2'])
SPECIES = {6: ['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'], 2: ['O2', 'N2']}      # names for the PFD
TRACE = 1e-9                                                       # mole fraction of absent species


class SOECPlant(Flowsheet):
    """The feeds, the units with their design inputs and the streams (an Equinox module: jit, vmap and grad apply)."""
    steam: Stream
    air: Stream
    mix: units.Mixer
    rec: units.HeatExchanger
    ht: units.Heater
    soc: units.SOC
    ko: units.Separator
    spl: units.Splitter

    def __init__(self, I=-50.0, Ncells=300.0, T_soc=1073.15, eff_rec=0.8, recycle=0.15):
        self.steam = Stream(0.1, 423.15, 1.0, jnp.array([TRACE, TRACE, TRACE, TRACE, 1.0 - 5*TRACE, TRACE]))
        self.air = Stream(0.3, 1073.15, 1.0, jnp.array([0.21, 0.79]))
        self.mix = units.Mixer(FUEL)
        self.rec = units.HeatExchanger(FUEL, FUEL, eff_rec)
        self.ht = units.Heater(FUEL, T_soc)
        self.soc = units.SOC(FUEL, AIR, I=I, Ncells=Ncells, T=T_soc)
        self.ko = units.Separator(FUEL, jnp.array([1.0, 1.0, 1.0, 1.0, 0.07, 1.0]), 313.15)  # 7 % of the water stays
        self.spl = units.Splitter(recycle)

        self.link(self.steam, self.mix.a)
        self.link(self.mix.out, self.rec.cold)
        self.link(self.rec.cold_out, self.ht.feed)
        self.link(self.ht.out, self.soc.fuel_in)
        self.link(self.air, self.soc.air_in)
        self.link(self.soc.fuel_out, self.rec.hot)
        self.link(self.rec.hot_out, self.ko.feed)
        self.link(self.ko.out1, self.spl.feed)
        self.link(self.spl.out1, self.mix.b)          # hydrogen recycle; spl.out2 (product), ko.out2 (water) leave


def solve_with_spec(plant, start, H2_product):
    """EO with a design specification: the current that gives ``H2_product`` mol/s of hydrogen."""
    def residuals(z):
        p = eqx.tree_at(lambda p: p.soc.I, plant, z[-1])
        s = p.unpack(z[:-1], start)
        return jnp.append(p.residuals(z[:-1], start), (flows(s['spl.out2'])[0] - H2_product)/H2_product)
    z = units.newton(residuals, jnp.append(plant.pack(start), plant.soc.I))
    return plant.unpack(z[:-1], start), z[-1]


def summary(plant, s):
    """Duties and power of a solved flowsheet, kJ/s."""
    W = plant.soc.work(s['ht.out'], plant.air, s['soc.fuel_out'])
    return dict(W_stack=W, Q_heater=s['ht.out'].H(FUEL) - s['rec.cold_out'].H(FUEL),
                Q_stack=plant.soc.heat(s['ht.out'], plant.air, s['soc.fuel_out'], s['soc.air_out'], W),
                H2=flows(s['spl.out2'])[0])


if __name__ == '__main__':
    plant = SOECPlant()
    plant.print_equations()

    sm = eqx.filter_jit(lambda p: p.sequential())
    t = time.perf_counter(); s = sm(plant); jax.block_until_ready(s); t_c = time.perf_counter() - t
    t = time.perf_counter(); s = sm(plant); jax.block_until_ready(s); t_r = time.perf_counter() - t
    print(f'SM : compile + run {t_c:.1f} s, run {t_r*1e3:.1f} ms')
    print('     ' + ', '.join(f'{k} = {float(v):.5f}' for k, v in summary(plant, s).items()))

    eo = eqx.filter_jit(lambda p, s: p.solve(s))
    t = time.perf_counter(); s_eo = eo(plant, s); jax.block_until_ready(s_eo); t_c = time.perf_counter() - t
    x = plant.pack(s_eo)
    print(f'EO : {x.size} unknowns from the SM solution, max |residual| = '
          f'{float(jnp.max(jnp.abs(plant.residuals(x, s)))):.1e}, compile + run {t_c:.1f} s')

    # a different operating point solved by EO from the SM solution of the base case
    s_spec, I = eqx.filter_jit(lambda p, s: solve_with_spec(p, s, 0.09))(plant, s)
    print(f'EO with a design spec: H2 product 0.09 mol/s needs I = {float(I):.3f} A per cell')

    # many operating points at once: the SM solve vectorised over the current
    currents = jnp.linspace(-30.0, -60.0, 7)
    sweep = eqx.filter_jit(jax.vmap(lambda i: summary(SOECPlant(I=i), SOECPlant(I=i).sequential())))(currents)
    print('current sweep (A per cell) :', np.round(np.asarray(currents), 1))
    print('  H2 product [mol/s]       :', np.round(np.asarray(sweep['H2']), 4))
    print('  stack power [kW]         :', np.round(np.asarray(sweep['W_stack']), 3))

    plant.pfd(s_eo, species=SPECIES, path='soec_flowsheet.html')
    print('process flow diagram: soec_flowsheet.html')
