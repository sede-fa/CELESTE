# Solid oxide cell (`CSL.electrochem.SOC`)

## Model

The fuel side holds H2, CH4, CO2, CO, H2O and N2 and leaves at water-gas-shift
(CO + H2O ⇌ CO2 + H2) and methane-reforming (CH4 + H2O ⇌ CO + 3 H2)
equilibrium. The current fixes the oxygen exchanged with the air side,
n_O = Ncells·I/(2F) (negative in electrolysis). The system of equations (`Eqs`):

| Unknown | Equation |
|---|---|
| ln n_H2, ln n_CH4, ln n_CO2, ln n_CO, ln n_H2O, ln n_N2 (fuel outlet) | C, H, N balances; O balance including n_O |
| | WGS equilibrium, ln(x_CO2·x_H2 / (x_CO·x_H2O)) − ln K_WGS = 0 |
| | MSR equilibrium, ln(x_CO·x_H2³·P² / (x_CH4·x_H2O)) − ln K_MSR = 0 |
| T_out (only with `Q=`) | energy balance, Q − Q_spec = 0 |

The balances are scaled by the element inflow and the equilibria are in log
form, so all residuals are of order one. Solving for the logarithms of the flows
keeps them positive, also for trace species (1e-8 CH4 in an electrolysis feed),
so `celeste.solvers.solve` needs no bounds. `run_sol` solves `f_sol_T`
(isothermal, 6 unknowns) or `f_sol_Q` (heat duty, 7 unknowns, outlet
temperature solved together with the compositions). The constants come from the
NASA Gibbs energies (`CSL.equilibrium`). The extents are reported from the
flows: `xi_WGS` is the CO2 formed, `xi_MSR` the CH4 converted.

## Energy specification

```python
soc = electrochem.SOC('soc', media.Fuel_CH4(), media.Air(), I=-50, Ncells=100, T=1073.15)  # isothermal
soc = electrochem.SOC('soc', media.Fuel_CH4(), media.Air(), I=-50, Ncells=100, Q=0)        # adiabatic
```

- `T=`: the outlets leave at `T`; the heat duty `Q` is computed.
- `Q=`: the heat added is fixed (`Q=0` adiabatic) and the outlet temperature is
  solved from the energy balance, together with the voltage and the ohmic loss,
  which depend on it.
- `Qloss` is heat exchanged with the surroundings at T0, positive into the stack
  like `Q` and `W`; the energy balance is `ΔH = Q + W + Qloss`.

`soc.T = ...` and `soc.Q = ...` switch between the two. The inlet state is set
on `F0` and `A0`.

## Ideal voltage

The reversible potential at a point in the cell is the Nernst potential of
H2 + ½ O2 → H2O at the local temperature and composition:

    E = −ΔG_r(T, x, P)/(2F),   ΔG_r = g°_H2O − g°_H2 − ½ g°_O2 + RT ln( x_H2O / (x_H2 · (x_O2·P)^½) )

It changes along the cell as the fuel converts. `V_model` chooses which value
the stack uses:

| `V_model` | Ideal voltage | Use |
|---|---|---|
| `'mean'` (default) | E averaged over the charge passed, Vid = ∫₀¹ E(λ) dλ, where λ is the fraction of the current drawn so far, the composition at λ is the local equilibrium and T is linear in λ from the fuel inlet to the outlet | reversible work of the stack: Wid = Vid·(−I)·Ncells |
| `'outlet'` | E at the outlet | limit for a stack whose cells run at one voltage: the outlet is where the local Nernst potential is highest (electrolysis) or lowest (fuel cell) |
| `'streams'` | (G_out − G_in)/(−I·Ncells) of the streams | the former model, for comparison |

All three are available as `V_mean`, `V_nernst_out` and `V_streams`
(`V_nernst_in` gives the inlet value). `Vop = Vid − f_ASR·ASR·j`, with the
area-specific resistance averaged along the cell in the same way for `'mean'`
and taken at the outlet otherwise.

**The Gibbs-energy approach when the temperature changes.** For an isothermal
stack fed with an equilibrium mixture, `V_mean` equals `V_streams` (the tests
check this to 1e-9 V): the Gibbs energy difference of the streams is the
integral of ΔG_r over the extent of the electrochemical reaction, and the
shift and reforming reactions add nothing because their ΔG_r is zero at
equilibrium. When the temperature changes along the cell, the stream difference
G_out(T1) − G_in(T0) also contains −∫S dT, the change of Gibbs energy from
heating the gas, which is not electrochemical work. `V_streams` then stops being a
voltage (in the adiabatic electrolysis test case it differs from the Nernst
values by several volts). The charge-averaged Nernst potential is the
Gibbs-energy approach carried over to a stack whose temperature changes: it
evaluates ΔG_r of the reaction at the temperature where the charge passes.

A feed that is not at equilibrium relaxes at the inlet. That relaxation releases
Gibbs energy that `V_streams` counts as electrochemical work and `V_mean` does
not; for the co-electrolysis feed of the tests the two differ by 8 mV.
