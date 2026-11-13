# Solid oxide cell (`CSL.electrochem.SOC`)

## Model

The fuel side holds H2, CH4, CO2, CO, H2O and N2 and leaves at water-gas-shift
and methane-reforming equilibrium. Its outlet follows from three extents:

    n_out = n_in + ξ_EC·ν_EC + ξ_WGS·ν_WGS + ξ_MSR·ν_MSR

| Reaction | Stoichiometry | Extent |
|---|---|---|
| electrochemical (fuel side) | H2 + O²⁻ → H2O + 2e⁻ | ξ_EC = Ncells·I/(2F), fixed by the current |
| water-gas shift | CO + H2O ⇌ CO2 + H2 | unknown |
| methane reforming | CH4 + H2O ⇌ CO + 3 H2 | unknown |

The element balances therefore hold by construction. ξ_WGS and ξ_MSR follow from
the two mass-action laws, written as ln(Q/K) = 0 and solved by a damped Newton
method with the analytical Jacobian; the constants come from the NASA Gibbs
energies (`CSL.equilibrium`). The air side loses ½·ξ_EC of O2. With the
CO2/CO couple instead of H2/H2O as the electrochemical reaction the outlet would
be the same, because the shift reaction is at equilibrium.

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
