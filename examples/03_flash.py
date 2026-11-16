"""
Vapour-liquid equilibrium: the Flash unit and real-fluid media.

Run with:  python examples/03_flash.py     (the first flash of a species set compiles for a few seconds)
"""
import numpy as np

from celeste.CSL import heat, interfaces, media, separation

syngas = np.array([0.60, 0.05, 0.10, 0.05, 0.20, 1e-8])        # H2, CH4, CO2, CO, H2O, N2

# --- 1. knock-out drum in a flowsheet that uses the ideal-gas (NASA) media -----------------------------
drum = separation.Flash('drum', Medium=media.Fuel_CH4())
drum.F0.n = 10.0
drum.F0.Xi = syngas
drum.F0.T = 303.15
drum.F0.P = 20.0
drum.run_sol()
print(f'{drum.phase}: vapour fraction {drum.beta:.4f}')
print(f'gas    {drum.F1.n:.3f} mol/s, x_H2O = {drum.F1.Xi[4]:.5f}')
print(f'liquid {drum.F2.n:.3f} mol/s, x_H2O = {drum.F2.Xi[4]:.5f}  (medium species: {drum.F2.Medium.species})')
print(f'water recovered {100*drum.recovery[4]:.2f} %, heat removed {-drum.Q:.1f} kW\n')

# condensation curve: vapour fraction against temperature
for T in (280.0, 320.0, 360.0, 400.0, 440.0):
    drum.T_spec = T
    drum.run_sol()
    print(f'  T = {T:5.1f} K  ->  {drum.phase:9s} beta = {drum.beta:.4f}')

# --- 2. adiabatic flash: the temperature follows from the energy balance -------------------------------
crude = np.array([1e-8, 1e-8, 0.02, 1e-8, 0.48, 1e-8, 0.50])   # CO2 / water / methanol
letdown = separation.Flash('letdown', Medium=media.Fuel_MeOH_PR(), P=1.2, Q=0.0)
letdown.F0.n = 5.0
letdown.F0.Xi = crude
letdown.F0.T = 420.0
letdown.F0.P = 30.0
sol = letdown.run_sol()
print(f'\nadiabatic let-down from 30 to 1.2 bar: feed is {letdown.F0.Medium.state}, '
      f'drum at {letdown.T:.1f} K, {100*letdown.beta:.1f} % vapour (converged: {sol.success})')
print(f'methanol in vapour {letdown.F1.Xi[6]:.3f}, in liquid {letdown.F2.Xi[6]:.3f}; '
      f'exergy destroyed {letdown.Exloss:.2f} kW\n')

# --- 3. real-fluid media make any unit phase-aware -----------------------------------------------------
cooler = heat.Heater('cooler', media.Fuel_CH4_PR(), media.Fuel_CH4_PR())
cooler.F0.n = 10.0
cooler.F0.Xi = syngas
cooler.F0.T = 600.0
cooler.F0.P = 20.0
cooler.F1.T = 303.15
cooler.F1.P = 20.0
cooler.run_sol()
out = cooler.F1.Medium
print(f'cooler with a Peng-Robinson medium: outlet is {out.state} (beta = {out.beta:.4f}), duty {cooler.Q:.1f} kW')

water = interfaces.Flange('w', media.Water_PR(), T=298.15, P=1.0)
print(f'water at 25 degC: {water.Medium.state}, h = {water.h:.2f} kJ/mol, exergy = {water.Ex:.2f} kJ/mol')
