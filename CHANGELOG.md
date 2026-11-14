# Changelog

## Unreleased – framework review

Existing models keep their names, arguments and attributes. The sections
**Results that change** and **Deprecated** list everything a user of an earlier
version may notice.

### Added

- **Type hints** on the public signatures of the models (`celeste._types.Array` for float arrays); `pyright` clean
- **`CSL.heat.heater_to`** (heater to a given outlet temperature, optional change of medium) and **`SysBase.solve(*units)`**,
  used to shorten the system boxes (`ElectBox`, `SOCsystem`, `ProcessBoxes`); results unchanged
- Transport properties (`mu`, `k`, `Pr`) for the real-fluid `Thermo_PR` media
- **`celeste.jx`** – numerical kernels in JAX
  - `nasa` / `ideal`: NASA-9 ideal-gas properties; one kernel for NumPy and JAX (`jit`, `vmap`, `grad`)
  - `eos`: Peng–Robinson equation of state for mixtures (fugacity coefficients, departure functions)
  - `flash`: tangent-plane stability test, Rachford–Rice, isothermal flash
  - `fluid`: real-fluid stream state (gas, liquid or two-phase)
  - `units`: Equinox prototypes of process units
- **`CSL.separation.Flash`** – vapour–liquid flash drum (isothermal, adiabatic or with a given duty)
- **`FT.componentsFT.Flash`** – Raoult flash for FT media: ideal hydrocarbon solution, immiscible liquid water,
  permanent gases stay in the vapour; `FT.thermoFT.vapour_pressures`, `psat_water`, `raoult_split`
- **`CSL.media.Medium_PRbase`** and `Fuel_CH4_PR`, `Fuel_MeOH_PR`, `HB_PR`, `Air_PR`, `Water_PR` – real-fluid
  media; `CSL.thermo.Thermo_PR`
- **`CSL.reactors.ExtentReactor`** – base for reactors solved for their reaction extents: element balances
  close by construction; energy specification `T` (isothermal) or `Q` (`Q=0` adiabatic)
- **`CSL.reactors.MR`** on the pattern of the SOC: unknowns are the logarithms of the outlet flows, equations the
  element balances and the shift and reforming equilibria (`CSL.equilibrium.fuel_residuals`, shared with the SOC);
  isothermal, adiabatic (`Q=0`) or with a given heat duty; `reactor.T = ...` / `reactor.Q = ...` switch an existing
  reactor; `xi` returns the shift and reforming extents. `CSL.equilibrium.ln_K`
- **`CSL.reactors.HB`** – one Haber–Bosch model for isothermal, adiabatic and cooled operation (kinetics of the
  former `HB_Adiabatic`); **`HB_1D`** – plug-flow bed integrated along its volume (profiles in `.profile`)
- **`CSL.reactors.MeOH`** – one methanol reactor: `model='equilibrium' | 'yield' | 'kinetic'`, isothermal,
  adiabatic or with a given duty (replaces `MeOH_Equil`, `MeOH_Yield`, `MeOH_Kinetic`, `MeOH_WGS`)
- `Flange.compatible(other)` and `Flange.incompatibility(other)` (the reason as
  text)
- **`celeste.HEN.HENSynthesiser`** – heat exchanger network synthesis from the pinch stream table
  (`solve(objective='cost' | 'mer' | 'min_units')`, default MILP time limit 600 s)
- **`CSL.systems.ProcessBoxes`** – `CondenserBox`, `MethanolLoop`
- **`RecycleMixin.Recycle_Tear`** – recycle convergence on the full tear stream (Anderson, Aitken/Steffensen,
  Wegstein with adaptive or fixed q, direct; `n_direct` start-up passes). See `docs/recycles.md`, which also
  covers feed–effluent heat exchangers
- **`celeste.solvers`** – `solve`, `solve_scalar`, `ConvergenceWarning`
- **`celeste.CSL.exergy`** – the exergy model now lives in the base package (`ThermoExergy`,
  `standard_chemical_exergy`, `exergy_table`). The FT module extends it (`FT.thermoFT.ThermoExergyFT`);
  a medium chooses its model with the class attribute `ExergyModel`. `celeste.CSL` no longer imports
  `celeste.FT`.
- `Flange.Ex_ph`, `Flange.Ex_ch` (physical / chemical exergy), `Flange.EX` (exergy flow, kW);
  `ThermoExergy.Ex_ph`, `Ex_ch`, `condenses`
- **`CSL.electrochem.SOC`**: unknowns are the logarithms of the fuel outlet flows (positive also for trace
  species, no bounds needed); equations are the C, H, O, N balances and the shift and reforming equilibria, plus
  the energy balance for a specified heat duty. Energy specification `T=` (isothermal, `f_sol_T`) or `Q=`
  (`Q=0` adiabatic, `f_sol_Q`: outlet temperature solved together with the composition). Ideal voltage
  `V_model='mean'` (charge-averaged Nernst potential, default), `'outlet'` or `'streams'` (former model), all
  three available as `V_mean`, `V_nernst_out`, `V_streams`; extents `xi_WGS`, `xi_MSR` from the flows. See
  `docs/electrochemistry.md`
- `CSL.components.ChgMedium` (moved from the FT module, which now extends it)
- **`CSL.electrochem.PCC`** (from main, rebuilt on the SOC machinery): proton conducting cell with the same
  equations, energy specification (`T=` / `Q=`), ideal voltage models and solution as the SOC; `SOC` and `PCC` share
  `_FuelStack`. `PCC_base`, `PCC_delG`, `PCC_Nernst_avg` and `SOC_Nernst_avg` remain as thin variants. SOC and PCC
  gain `V_model='average'`, `Wid`, `AU`, `Lambda` and a `j` setter
- **`CSL.turbomachinery`** (from main): compressors and turbines with performance maps. The compressor models
  live here (`Compressor`, `CompIsot`, `CompIsent`, `CompressorMapped`, `Turbine`, `TurbineMapped`); one
  real-gas path (`_RealGasPath`) serves the isentropic compressor, the polytropic compressor and the mapped
  compressors and turbines. **`CompPolytropic`**: polytropic efficiency; closed form for ideal-gas (NASA) media, any cp(T): the outlet
  entropy is s0 + (1/eff − 1)·R·ln(P1/P0); other media integrate in `n_steps` isentropic steps
- **`ComponentModels.Ncomp`** (and `SimpleCompressorSys`) take the compressor model as `compressor=` (a class or any
  callable `(name, Medium, eff)`, one for all stages or a list per stage), one value or a list for `eff_comp` and `T_HT`, a
  final pressure `P_out` (equal pressure ratios) instead of `P_comp`, and `add_stage(..., compressor=)`
- Transport properties (from main): `mu`, `k`, `Pr` of the media (Chung viscosity, Eucken conductivity, Wilke and
  Mason–Saxena mixing; CoolProp for `*_CP`) and of the flanges, `Flange.rho`; vectorised in NumPy (no Numba)
- `media.Fluegas`, `FluegasIncomplete`, `AirHumid` (from main)
- `HexNTU.run_sol_fixOutletTemp(Ttarg, stream)` (from main): UA that reaches an outlet temperature
- `CSL.equilibrium` – equilibrium constants shared by reactors and the SOC, computed from the NASA Gibbs
  energies (`K_eq(T, reaction)`; `K_WGS`, `K_rWGS`, `K_MSR`, `K_PSR`, `K_NH3`, `K_MeOH_CO2`, `K_MeOH_CO`)
- `jx.units.Stream.replace(n=, T=, P=, x=)`: a copy with a changed state (streams are immutable)
- `ModelBase.inlets`, `outlets`, `mass_balance()` and a default `MBcheck()`; `energy_balance()`, `heat_and_work()`
- `SysBase.boundary_flanges()`, `unit_balances()`
- `IntrospectionMixin.get_Electric()` (was called by the reports but did not exist)
- `Thermo.in_range`, `Thermo_CP.rho`, `Thermo_CP.LHVmix`
- test suite (`tests/`), examples (`examples/`), documentation (`docs/`)

### Changed

- Numba is no longer used or required. Property data, dead-state values and chemical exergies are cached per
  species set. See the performance table in the README.
- JAX and Equinox are new dependencies. Cantera and IPython are optional extras and imported on first use;
  CoolProp is imported on first use.
- `celeste.CSL.<module>` and `celeste.<package>` are importable lazily (`import celeste; celeste.CSL.media.Air()`).
- Unit models solve through `celeste.solvers.solve`: bounded where the unknowns have physical limits, success
  judged by the residual, failures reported as `ConvergenceWarning` instead of `print`.
- `Mixer.run_sol` always returns a result object with `.success`.
- Media: every NASA medium is declared by `_default_species` / `_default_Xi` only; constructors accept
  `species`, `T`, `Xi`, `P`. `Medium_Base.get_Idx` works for lists.
- `PinchAnalyser` accepts the stream names as index (the format of `pinch_stream_table()`).
- `create_graph(..., save=...)`: writing the `.mmd` file can be switched off or redirected.
- `exdest_df` groups all reactors under *Reactor* and all black-box separators under *GPU*; new row *Separation*.
- Unit constructors take design quantities only (medium, efficiency, volume, split, model); the operating state
  is set on the flanges (see `docs/conventions.md`). `Heater(name, Medium0)` uses `Medium0` on both sides when
  `Medium1` is not given. The internal systems, examples and tutorial use the new style.
- **`Valve`** is isenthalpic: the outlet temperature follows from h(T1, P1) = h(T0, P0) (unchanged for ideal-gas
  media, Joule–Thomson cooling for `*_PR` media). Outlet pressure from `P_out=`, `dP=` or the pressure set on `F1`.
  Stand-alone default changed to a pressure drop (3 → 1 bar; was 1 → 3 bar).
- **`Pump(name, Medium, eff, P_out=None, dP=None)`**; a falling pressure gives a hydraulic turbine (W < 0).
- `connect` / `connect_seq` check the species list as well as the flange and medium type, and the error names
  the reason. Flanges compare by identity (`==` used to mean "same flange and medium type"), so they can be
  used in sets and as dictionary keys.
- `SysBase.MBcheck()` / `EBcheck()` balance the streams that cross the system boundary (found from the recorded
  connections), so they also work for boxes without sources and sinks and include every unit's heat and work.
  `MBcheck()` now returns in − out like the unit models (was out − in). `connect_seq` always records the
  connection; `store=True` is only needed for the flowsheet graph.
- GPU models share one base (`GPUbase`): work from the exergy change and `eff`, the work dissipated to the
  surroundings (`Q_amb`, so the energy balance closes), consistent `Exloss` and `Sgen`. `GPU_2S` is `GPU_2SV2`
  with one component removed; `GPU_3S` sets the F3 composition to the species marked −1; an empty outlet keeps a
  valid composition; `Sep` is checked. State keywords as in the other units.
- Removed `.gitattributes` (it referred to `CELESTE2/_version.py`, which does not exist); releases are tags on
  `main` (`docs/conventions.md`).
- `SOC` takes design inputs like the other units: the inlet state is set on `F0`/`A0`, the outlet temperature
  is `T=`. `f_sol` takes the logarithms of the outlet flows (was outlet fractions and flow) and dispatches to `f_sol_T` / `f_sol_Q`; `MSR_extent` is
  replaced by `xi_MSR`. The ohmic loss uses the area-specific resistance averaged along the cell (unchanged for an isothermal
  stack).

### Fixed

- `HexNTU.run_sol_fixOutletTemp` (main) missed the factor `Cmin` in the UA of a counter-flow exchanger with equal capacity rates (Cr = 1), so the outlet target was not reached

- `SOC`: `Qloss` follows the sign convention of `Q` and `W` (positive into the stack, a loss is negative) and
  now enters the entropy generation (heat exchanged at T0). Before, it was described as a loss but entered the
  energy balance as a heat input, and `Sgen` ignored it. Results with the default `Qloss = 0` are unchanged.
- `SOC.EnthalpyEfficiency`: the methane heating value used CO instead of CH4 (CO2 + 2 H2O − CO − 2 O2), and the
  heating values were taken at the stack temperature. It now uses the lower heating values of the medium
  (`Flange.LHV`, 25 °C): default electrolysis case 0.9707 → 0.9453.
- `MR` at low temperature: the solver reported convergence for outlets that were not at equilibrium (the absolute
  mass-action residuals are tiny when products are trace species). At 500 K and 1–30 bar, 15 of 84 test feeds were
  off by up to 0.09 in a mole fraction; all 84 now satisfy both equilibria to 1e-13, and the solve is about 40 times
  faster.
- `Mixer`: two inlets within 0.001 % in temperature were treated as equal, so the outlet temperature jumped by up
  to a few mK; this stalled recycle solvers near convergence.
- A medium created with its own species list (`media.Fuel_CH4(species=[...], Xi=...)`) was reset to the
  default species when it was given to a flange.
- **FT liquid phase**: setting `phase='Liquid'` on an FT flange had no effect (the phase switch was only read
  when the medium was built), so liquid FT streams returned gas properties. The phase can now be set and
  changed at any time; water becomes `H2O(L)`, hydrocarbons below their critical temperature become an ideal
  liquid solution (no pressure term in the entropy). `'liquid'` spelt in lower case no longer sends every species
  to the NASA list.
- **FT heat of vaporisation**: `B*(1 - T/Tc)*n` replaced by the documented Watson correlation
  `B*(1 - T/Tc)**n`; below the correlation range a sub-cooled liquid with the tabulated liquid heat capacity
  (was `hvap(Tmin) + cpl*(T - Tmin)`, discontinuous at Tmin and of the wrong sign); the liquid entropy includes the
  vapour pressure (Clausius–Clapeyron) and the liquid `cp` the temperature derivative of the heat of vaporisation.
- **FT `RStoic`**: heat duty from flow × enthalpy (was `F0.n*F0.n + FH0.n*FH0.n`, flow squared).
- **FT exergy**: the C5+ alkanes have liquid standard chemical exergies in `Chem_exergies.csv` but were referred
  to the ideal gas at T0; their dead state is now the liquid.
- `pyproject.toml` declares the BSD-3-Clause licence of `LICENSE` (was "Proprietary").

- The package could not be installed on case-sensitive file systems (`README.md` vs `readme.md`).
- **FT module**: `Medium_Base` overwrote the reference composition of the FT media with `None`; every FT flange
  failed. `ExBs()` methods called a non-existent `Flange.ExBs()`; `Exloss` could not be assigned.
- **Systems**: all SOC based systems passed keyword arguments that `SOC` did not accept; `SimpleHeaterSys` and
  `SimpleCompressorSys` called themselves recursively and built the heater with wrong arguments; `SOCBasicSys`
  connected to a non-existent sink flange.
- **Reports**: `work_df`/`tot_work_*`, `heat_df`, `tot_Qloss`, `exdest_df`, `SBcheck`, `get_success`,
  `tot_work_comp` raised or returned wrong values (missing imports, wrong keys, empty tables, a string test
  that was always true).
- `MeOH_Equil.ExBs` raised; the methanol reactors had no `Exloss` until `ExBs()` was called.
- `MeOH_Kinetic`: adsorption constants were frozen at the construction temperature.
- `GPU_3S` (CSL and FT) failed with NumPy ≥ 2.
- `Recycle_Hybrid_FPI_Aitken` returned after the initial iterations with a zero error; `Recycle_Steffensen`
  divided by zero at convergence; progress printing failed for scalar flows.
- `Thermo_CP` media had no density (`Medium_CPbase.rho` used a missing attribute).
- `Base.__init__` stored the class object of the caller as an attribute (`__class__` ended up in `to_dict()`).

### Results that change

| What | Before | Now | Size of the effect |
|---|---|---|---|
| `CompIsent` outlet state | ideal-gas formula with the mean heat capacity of inlet and outlet, solved with `root` | real-gas isentropic outlet from the medium, then the efficiency correction (also used by the mapped compressors and turbines); explicit, so `run_sol` returns a result with `success=True` and `get_success()` no longer lists compressors | outlet temperature 511.43 → 512.49 K and work 6.290 → 6.321 kJ/s for air, 300 K, 1 → 5 bar, eff 0.8; the reversible case now has Sgen = 0 exactly (was 1e-4) |
| `GPU_*` `Exloss`, `Sgen` | exergy in − exergy out (negative for a separation), ΔS | exergy in + W − exergy out = W − Wid ≥ 0; ΔS − Q_amb/T0, so Exloss = T0·Sgen | GPU rows of `exdest_df` grow by the separation work, e.g. 0.94 kW in `SOECSys` |
| `SysBase.MBcheck()`, `EBcheck()` | sources and sinks only, `MBcheck` out − in; `EBcheck` from the heater / heat / work tables | boundary streams from the recorded connections, in − out; Σ(Q + W) of every unit | sign of `MBcheck`; `EBcheck` now closes for boxes without sources and sinks (e.g. `SOCsysRecupe`, `CO2Sys`) |
| `Valve` outlet temperature | equal to the inlet | isenthalpic | none for ideal-gas media; Joule–Thomson cooling for `*_PR` media |
| Species below a mole fraction of 1e-6 | left out of `cp`, `h`, `s` (a step at 1e-6) | included; entropy term goes smoothly to zero | ≤ 1e-6 × species value, e.g. 6e-6 kJ/mol in `h` |
| Temperature outside the fitted range of a species | properties returned as **zero** | heat capacity frozen at the range limit, `h`, `s` continuous | large where it applies, e.g. `H2O(L)` above 600 K |
| Liquid water as **first** species of a medium (`Water`, `Media_AEL`) | received the ideal-gas mixing/pressure entropy | no mixing or pressure entropy, as at any other position | `s`, `g`, `Sgen` at P ≠ 1 bar; exergy of liquid water at 60 bar: 11.05 → 0.90 kJ/mol |
| Exergy of `H2O(L)` in media that also contain `H2O` (`Media_AEL`, `Steam_Water`) | gas pressure and mixing term applied to the liquid | none, as in the entropy of these media; exergy destruction now equals T0·Sgen | up to R·T0·ln(P/bar) per mol of liquid, e.g. 10 kJ/mol at 60 bar |
| `ChgMedium` | an outlet species missing in the inlet received the `H2O(L)` fraction when the inlet contained `H2O(L)` | species mapped by name, `H2O` ↔ `H2O(L)` | only media with several species missing from the inlet |
| `Flange.Ex` | overwrote fractions below 1e-6 in the flange with 1e-12 and relabelled FT flanges as gas | reads only | downstream values at the 1e-6 level |
| Exergy of media with species outside the FT table (`CH3OH`, `C3H8`, …) | dead state from the FT correlation: `nan` for methanol, error for `CH3OH(L)` | dead state from the property model of the medium | methanol media now have an exergy; `media_C3`: ≤ 0.02 kJ/mol |
| `MeOH_Kinetic` | converged to negative mole fractions for the documented example | physical solution | different outlet |
| `ASU`, `DAC` | `F2.T` stale; `DAC` overwrote the inlet temperature with the outlet's | both outlets at the inlet temperature | energy and exergy balance of these units |
| `Thermo_CP.cp` | J/mol/K | kJ/mol/K, as for every other medium | factor 1000 |
| FT streams with `phase='Liquid'` | gas properties | liquid: e.g. `Wax` at 298 K: `h` −653 → −766 kJ/mol, `cp` 0.67 → 1.05 kJ/mol/K; `Water`: −241.8 → −285.8 kJ/mol | every liquid FT stream; the `FT` reactor duty of the test case −32.628 → −32.643 kW |
| Exergy of gaseous FT streams with C5+ | liquid chemical exergy with a gas dead state | gas carries the exergy of vaporisation, −R·T0·ln(Psat/P0) | `Wax` +53 kJ/mol (0.3 %), `Naphtha` +8 kJ/mol |
| FT `RStoic.Q` | flow squared | enthalpy balance | any use of `RStoic.Q` or its `ExBs()` |
| Equilibrium constants | empirical fits: linear Gibbs fit (MR, SOC, MeOH_WGS, Reformer with 32.115), Graaf (MeOH_Equil, MeOH_Yield) | from the NASA Gibbs energies, consistent with `h`, `s` | WGS at 450 K: 257 → 399 (fit), 378 → 399 (Graaf); within 0.6 % at SOC temperatures. Methanol loop carbon efficiency 0.9002 → 0.9087 |
| `HB` (isothermal) | Nielsen constants of the original `HB`, outlet at the inlet temperature | constants of `HB_Adiabatic` (one rate law); isothermal at `T` | test case NH3 0.3658 → 0.3677 |
| Isothermal methanol reactors | outlet at the inlet temperature (constructor `T` set both) | at the specified `T` | only when the feed temperature differs from `T` in a system |
| `MR` outlet at ≤ 500 K | not at equilibrium for many feeds (see Fixed) | equilibrium | up to 0.09 in a mole fraction at 500 K, 1 bar; unchanged to 1e-6 above 700 K |
| `MR(Q=...)` | ignored (the duty was always computed) | heat duty; `Q=0` adiabatic | only callers that passed `Q` |
| Reactor and SOC results | – | same roots, solver tolerance tighter | relative 1e-6 or smaller |
| `SOC.Vid` | Gibbs energy difference of the streams (with `extent=True` minus a reforming term that is zero at equilibrium) | charge-averaged Nernst potential (`V_model='mean'`); the former value is `V_streams` / `V_model='streams'` | identical for an isothermal stack fed at equilibrium. Feed not at equilibrium: its relaxation is no longer counted as work, e.g. `SOECSys` 1.2212779 → 1.2212782 V, `SOCsysRecupe` 1.397895 → 1.397950 V, co-electrolysis test feed 0.8325 → 0.8402 V, fuel cell fed with 25 % CH4 (internal reforming) by 0.43 V, because the reforming of the feed towards equilibrium was counted as electrical work. With a temperature change the former value was not a voltage |

The reference values of the original implementation are stored in
`tests/data/golden_thermo.json`; `tests/test_thermo.py` and
`tests/test_exergy.py` document every deviation above.

### Deprecated

These keep working and emit a `CelesteDeprecationWarning`:

| Name | Use instead |
|---|---|
| `PCC_base`, `SOC_Nernst_avg` (no warning) | `PCC`, `SOC(..., V_model='average')` |
| `PCC(Tpcc0=, Tpcc1=, extent=)` | `pcc.F0.T = ...`, `PCC(..., T=...)`, nothing |
| `HB_Adiabatic(...)` | `HB(..., Q=0)`; outlet pressure `F1_P` → pressure drop `dP` |
| `MeOH_Equil`, `MeOH_Yield`, `MeOH_Kinetic` | `MeOH(..., model='equilibrium' / 'yield' / 'kinetic', T=...)` |
| `MeOH_WGS` | `MeOH(..., model='yield', Q=0)` |
| state keywords of unit constructors (`F0_n`, `F0_Xi`, `F0_T`, `F0_P`, outlet specs such as `F1_T`, `F1_P`; reactor `P` (also `MR(..., P=)`); `T`/`P` of `Mixer`/`Splitter`; `T`/`n`/`Xi` of `ASU`/`DAC`) – `HB`, `MeOH`, `MR`, `SOC`, `Heater`, `Compressor`, `CompIsot`, `CompIsent`, `ASU`, `DAC`, `Mixer`, `Splitter`, `Sink`, `Flash`, `Valve`, `Pump` (also their `n`, `Xi`), CSL `GPU_*` | set the state on the flanges: `unit.F0.n = ...`, `heater.F1.T = ...`, `comp.F1.P = ...` |
| `Recycle_FPI`, `Recycle_Wegstein_q`, `Recycle_Wegstein_Generic`, `Recycle_Wegstein_Generic2`, `Recycle_Steffensen`, `Recycle_Aitken`, `Recycle_Hybrid_FPI_Aitken`, `recycle_solver` | `Recycle_Tear(..., method='direct' / 'wegstein' / 'aitken')`; the wrappers converge the full tear stream, so they may take a different number of passes |
| `Thermo.updateBoolArr()`, `thermoArr()`, `getCoeffs()` | nothing – handled inside `calcThermoSep()` |
| `EquilBase.ExBs()` (methanol reactors) | `Exth`, `Exloss`, `Exeff` are properties |
| `ASU.MBCheck()`, `DAC.MBCheck()` | `MBcheck()` |
| `SOC(Tsoc0=, T_soc0=, P_soc=)` | `soc.F0.T = ...`, `soc.A0.T = ...`, `soc.F0.P = ...` |
| `SOC(Tsoc1=, T_soc1=)` | `SOC(..., T=...)` (or `Q=` for a heat duty) |
| `SOC(extent=)`, `SOECSys` / `SOECSysV2` / `CO2Sys` `(ext=)` | nothing – the ideal voltage excludes the shift and reforming reactions |

`celeste.FT.thermoFT.ThermoExergy` and `standard_chemical_exergy` remain importable from their old
location and dispatch to the new classes. `HB.for_sol` / `HB_Adiabatic.for_sol` remain as aliases of `f_sol`.
