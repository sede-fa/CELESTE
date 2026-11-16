"""
An SOEC system built from Equinox units, solved two ways:

* sequential modular (SM): the units are called in flowsheet order and the recycle is converged on its tear
  stream - the way the CSL systems run, but as one compiled function;
* equation oriented (EO): every stream is an unknown, every unit contributes its equations, and Newton's
  method with the exact Jacobian (``jax.jacfwd``) solves them all at once, starting from the SM solution.
  EO then also solves design specifications directly (here: the current for a given hydrogen production).

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
from celeste.jx.units import Stream, flows, stream_from_vector, stream_vector

FUEL = ideal.ideal_gas(['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'])
AIR = ideal.ideal_gas(['O2', 'N2'])
TRACE = 1e-9                                                       # mole fraction of absent species

STEAM = Stream(0.1, 423.15, 1.0, jnp.array([TRACE, TRACE, TRACE, TRACE, 1.0 - 5*TRACE, TRACE]))
AIR_IN = Stream(0.3, 1073.15, 1.0, jnp.array([0.21, 0.79]))


class SOECPlant(eqx.Module):
    """The units of the flowsheet with their design inputs (an Equinox module: jit, vmap and grad apply)."""
    mix: units.Mixer
    rec: units.HeatExchanger
    ht: units.Heater
    soc: units.SOC
    ko: units.Separator
    spl: units.Splitter

    def __init__(self, I=-50.0, Ncells=300.0, T_soc=1073.15, eff_rec=0.8, recycle=0.15):
        self.mix = units.Mixer(FUEL)
        self.rec = units.HeatExchanger(FUEL, FUEL, eff_rec)
        self.ht = units.Heater(FUEL, T_soc)
        self.soc = units.SOC(FUEL, AIR, I=I, Ncells=Ncells, T=T_soc)
        self.ko = units.Separator(FUEL, jnp.array([1.0, 1.0, 1.0, 1.0, 0.07, 1.0]), 313.15)  # 7 % of the water stays
        self.spl = units.Splitter(recycle)

    # ---- sequential modular -------------------------------------------------------------------------
    def pass_(self, recycle: Stream, hot_guess: Stream):
        """One pass through the flowsheet for a guessed recycle (and recuperator hot side); returns all streams."""
        s = {'recycle_in': recycle}
        s['mix'] = self.mix(STEAM, recycle)
        _, s['rec_cold'], _ = self.rec(hot_guess, s['mix'])
        s['ht'], _ = self.ht(s['rec_cold'])
        s['soc_fuel'], s['soc_air'], _, _ = self.soc(s['ht'], AIR_IN)
        s['rec_hot'], _, _ = self.rec(s['soc_fuel'], s['mix'])
        s['ko_gas'], s['water'], _ = self.ko(s['rec_hot'])
        s['recycle'], s['product'] = self.spl(s['ko_gas'])
        return s

    def solve_sm(self, tol=1e-10, max_iter=100):
        """Converge the tear streams (recycle and recuperator hot inlet) by successive substitution."""
        def tear(s):
            return jnp.concatenate((stream_vector(s['recycle']), stream_vector(s['soc_fuel'])))

        def run(x):
            return self.pass_(stream_from_vector(x[:7], STEAM.P), stream_from_vector(x[7:], STEAM.P))

        def cond(c):
            return (c[1] > tol) & (c[2] < max_iter)

        def body(c):
            x, _, k = c
            x_new = tear(run(x))
            return x_new, jnp.max(jnp.abs(x_new - x)/(jnp.abs(x) + 1e-3)), k + 1

        x0 = jnp.concatenate((stream_vector(STEAM.replace(n=0.1*STEAM.n, T=313.15)), stream_vector(STEAM.replace(T=1073.15))))
        x, err, k = jax.lax.while_loop(cond, body, (x0, jnp.inf, 0))
        return run(x), k

    # ---- equation oriented --------------------------------------------------------------------------
    UNKNOWN = ('mix', 'rec_cold', 'ht', 'soc_fuel', 'soc_air', 'rec_hot', 'ko_gas', 'water', 'recycle', 'product')

    @staticmethod
    def pack(s):
        return jnp.concatenate([stream_vector(s[k]) for k in SOECPlant.UNKNOWN])

    @staticmethod
    def unpack(x):
        s, i = {}, 0
        for k in SOECPlant.UNKNOWN:
            m = 3 if k == 'soc_air' else 7
            s[k] = stream_from_vector(x[i:i + m], STEAM.P)
            i += m
        return s

    def residuals(self, x):
        """Equations of all units; the unknowns are the stream variables."""
        s = self.unpack(x)
        return jnp.concatenate((
            self.mix.residuals(STEAM, s['recycle'], s['mix']),
            self.rec.residuals(s['soc_fuel'], s['mix'], s['rec_hot'], s['rec_cold']),
            self.ht.residuals(s['rec_cold'], s['ht']),
            self.soc.residuals(s['ht'], AIR_IN, s['soc_fuel'], s['soc_air']),
            self.ko.residuals(s['rec_hot'], s['ko_gas'], s['water']),
            self.spl.residuals(s['ko_gas'], s['recycle'], s['product']),
        ))

    def solve_eo(self, x0):
        return self.unpack(units.newton(self.residuals, x0))


def solve_eo_with_spec(x0, H2_product, I0=-50.0, **design):
    """EO with a design specification: the current that gives ``H2_product`` mol/s of hydrogen."""
    def residuals(z):
        plant = SOECPlant(I=z[-1], **design)
        s = plant.unpack(z[:-1])
        return jnp.append(plant.residuals(z[:-1]), (flows(s['product'])[0] - H2_product)/H2_product)
    z = units.newton(residuals, jnp.append(x0, I0))
    return SOECPlant.unpack(z[:-1]), z[-1]


def summary(plant, s):
    """Duties and power of a solved flowsheet, kJ/s."""
    W = plant.soc.work(s['ht'], AIR_IN, s['soc_fuel'])
    return dict(W_stack=W, Q_heater=s['ht'].H(FUEL) - s['rec_cold'].H(FUEL),
                Q_stack=plant.soc.heat(s['ht'], AIR_IN, s['soc_fuel'], s['soc_air'], W),
                H2=flows(s['product'])[0])


if __name__ == '__main__':
    plant = SOECPlant()

    sm = eqx.filter_jit(lambda p: p.solve_sm())
    t = time.perf_counter(); (s, passes) = sm(plant); jax.block_until_ready(s); t_c = time.perf_counter() - t
    t = time.perf_counter(); (s, passes) = sm(plant); jax.block_until_ready(s); t_r = time.perf_counter() - t
    print(f'SM : {int(passes)} passes, compile + run {t_c:.1f} s, run {t_r*1e3:.1f} ms')
    print('     ' + ', '.join(f'{k} = {float(v):.5f}' for k, v in summary(plant, s).items()))

    x0 = SOECPlant.pack(s)
    eo = eqx.filter_jit(lambda p, x: p.solve_eo(x))
    t = time.perf_counter(); s_eo = eo(plant, x0); jax.block_until_ready(s_eo); t_c = time.perf_counter() - t
    print(f'EO : {x0.size} unknowns from the SM solution, max |residual| = '
          f'{float(jnp.max(jnp.abs(plant.residuals(SOECPlant.pack(s_eo))))):.1e}, compile + run {t_c:.1f} s')

    # a different operating point solved by EO from the SM solution of the base case
    s_spec, I = eqx.filter_jit(lambda x: solve_eo_with_spec(x, 0.09))(x0)
    print(f'EO with a design spec: H2 product 0.09 mol/s needs I = {float(I):.3f} A per cell')

    # many operating points at once: the SM solve vectorised over the current
    currents = jnp.linspace(-30.0, -60.0, 7)
    sweep = eqx.filter_jit(jax.vmap(lambda i: summary(SOECPlant(I=i), SOECPlant(I=i).solve_sm()[0])))(currents)
    print('current sweep (A per cell) :', np.round(np.asarray(currents), 1))
    print('  H2 product [mol/s]       :', np.round(np.asarray(sweep['H2']), 4))
    print('  stack power [kW]         :', np.round(np.asarray(sweep['W_stack']), 3))
