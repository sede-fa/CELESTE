"""
Unit models on their own: create, solve, inspect.

Run with:  python examples/01_components.py
"""
import numpy as np

from celeste.CSL import components, flow, heat, interfaces, media, reactors

# --- a stream: the Flange -------------------------------------------------------------------------
air = interfaces.Flange('air', media.Air(), n=1.0, Xi=np.array([0.21, 0.79]), T=300.0, P=1.0)
print(air, '\n')
print(f'exergy {air.Ex:.4f} kJ/mol, entropy {air.s:.5f} kJ/mol/K\n')

# --- compressor and cooler --------------------------------------------------------------------------
comp = components.CompIsent('comp', media.Air(), eff=0.8)
comp.F0.T = 300.0
comp.F0.P = 1.0
comp.F1.P = 5.0
sol = comp.run_sol()
print(f'compressor: converged={sol.success}, outlet {comp.F1.T:.1f} K, work {comp.W:.3f} kW, '
      f'exergy destroyed {comp.Exloss:.3f} kW')

cooler = heat.Heater('cooler', media.Air(), media.Air())
cooler.F1.T = 310.0
cooler.F0.Medium.PTX = comp.F1.Medium.PTX          # hand the stream over (systems do this with connect_seq)
cooler.run_sol()
print(f'cooler: duty {cooler.Q:.3f} kW\n')

# --- a reactor ----------------------------------------------------------------------------------------
feed = np.array([0.60, 0.01, 0.20, 0.10, 0.08, 0.01])      # H2, CH4, CO2, CO, H2O, N2
reactor = reactors.MR('methanation', T=600.0)
reactor.F0.n, reactor.F0.Xi, reactor.F0.P = 10.0, feed, 10.0
sol = reactor.run_sol()
print(f'methanation: converged={sol.success} ({sol.method}), residual {sol.residual_norm:.1e}')
print(reactor.ViewFlange(reactor).round(4), '\n')
print(f'heat duty {reactor.Q:.2f} kW, exergy efficiency {reactor.Exeff:.3f}')
print(f'mass balance error {reactor.mass_balance():.2e} g/s')

# --- mixing -------------------------------------------------------------------------------------------
mixer = flow.Mixer('mix', media.Air())
mixer.F0.n = 1.0
mixer.F1.n = 2.0
mixer.F0.T, mixer.F1.T = 300.0, 600.0
mixer.run_sol()
print(f'\nmixer outlet {mixer.F2.T:.2f} K, entropy generated {mixer.Sgen*1e3:.3f} W/K')
