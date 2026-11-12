# Changelog

## Unreleased – framework review

Existing models keep their names, arguments and attributes. The sections
**Results that change** and **Deprecated** list everything a user of an earlier
version may notice.

### Added

- **`celeste.jx`** – numerical kernels in JAX
  - `nasa` / `ideal`: NASA-9 ideal-gas properties; one kernel for NumPy and JAX (`jit`, `vmap`, `grad`)
  - `eos`: Peng–Robinson equation of state for mixtures (fugacity coefficients, departure functions)
  - `flash`: tangent-plane stability test, Rachford–Rice, isothermal flash
  - `fluid`: real-fluid stream state (gas, liquid or two-phase)
  - `units`: Equinox prototypes of process units
- **`CSL.separation.Flash`** – vapour–liquid flash drum (isothermal, adiabatic or with a given duty)
- **`CSL.media.Medium_PRbase`** and `Fuel_CH4_PR`, `Fuel_MeOH_PR`, `HB_PR`, `Air_PR`, `Water_PR` – real-fluid
  media; `CSL.thermo.Thermo_PR`
- **`CSL.reactors.ExtentReactor`** – base for reactors solved for their reaction extents: element balances
  close by construction; energy specification `T` (isothermal) or `Q` (`Q=0` adiabatic)
- **`CSL.reactors.HB`** – one Haber–Bosch model for isothermal, adiabatic and cooled operation (kinetics of the
  former `HB_Adiabatic`); **`HB_1D`** – plug-flow bed integrated along its volume (profiles in `.profile`)
- **`CSL.reactors.MeOH`** – one methanol reactor: `model='equilibrium' | 'yield' | 'kinetic'`, isothermal,
  adiabatic or with a given duty (replaces `MeOH_Equil`, `MeOH_Yield`, `MeOH_Kinetic`, `MeOH_WGS`)
- `Flange.set(n=, Xi=, T=, P=)`
- **`celeste.HEN.HENSynthesiser`** – heat exchanger network synthesis from the pinch stream table
- **`CSL.systems.ProcessBoxes`** – `CondenserBox`, `MethanolLoop`
- **`RecycleMixin.Recycle_Tear`** – recycle convergence on the full tear stream (Anderson, Wegstein, direct)
- **`celeste.solvers`** – `solve`, `solve_scalar`, `ConvergenceWarning`
- **`celeste.CSL.exergy`** – the exergy model now lives in the base package (`ThermoExergy`,
  `standard_chemical_exergy`, `exergy_table`). The FT module extends it (`FT.thermoFT.ThermoExergyFT`);
  a medium chooses its model with the class attribute `ExergyModel`. `celeste.CSL` no longer imports
  `celeste.FT`.
- `Flange.Ex_ph`, `Flange.Ex_ch` (physical / chemical exergy), `Flange.EX` (exergy flow, kW);
  `ThermoExergy.Ex_ph`, `Ex_ch`, `condenses`
- `CSL.components.ChgMedium` (moved from the FT module, which now extends it)
- `CSL.equilibrium` – equilibrium constants shared by reactors and the SOC, computed from the NASA Gibbs
  energies (`K_eq(T, reaction)`; `K_WGS`, `K_rWGS`, `K_MSR`, `K_PSR`, `K_NH3`, `K_MeOH_CO2`, `K_MeOH_CO`)
- `ModelBase.inlets`, `outlets`, `mass_balance()` and a default `MBcheck()`
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
- `SOC` accepts `T_soc0`, `T_soc1`, `P_soc` in addition to `Tsoc0`, `Tsoc1`.

### Fixed

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
| Reactor and SOC results | – | same roots, solver tolerance tighter | relative 1e-6 or smaller |

The reference values of the original implementation are stored in
`tests/data/golden_thermo.json`; `tests/test_thermo.py` and
`tests/test_exergy.py` document every deviation above.

### Deprecated

These keep working and emit a `CelesteDeprecationWarning`:

| Name | Use instead |
|---|---|
| `HB_Adiabatic(...)` | `HB(..., Q=0)`; outlet pressure `F1_P` → pressure drop `dP` |
| `MeOH_Equil`, `MeOH_Yield`, `MeOH_Kinetic` | `MeOH(..., model='equilibrium' / 'yield' / 'kinetic', T=...)` |
| `MeOH_WGS` | `MeOH(..., model='yield', Q=0)` |
| state keywords of unit constructors (`F0_n`, `F0_Xi`, `F0_T`, `F0_P`, outlet specs such as `F1_T`, `F1_P`; reactor `P`; `T`/`P` of `Mixer`/`Splitter`; `T`/`n`/`Xi` of `ASU`/`DAC`) – `HB`, `MeOH`, `Heater`, `Compressor`, `CompIsot`, `CompIsent`, `ASU`, `DAC`, `Mixer`, `Splitter`, `Sink`, `Flash` | set the state on the flanges: `unit.F0.set(n=..., Xi=..., T=..., P=...)`, `heater.F1.set(T=...)`, `comp.F1.set(P=...)` |
| `Thermo.updateBoolArr()`, `thermoArr()`, `getCoeffs()` | nothing – handled inside `calcThermoSep()` |
| `EquilBase.ExBs()` (methanol reactors) | `Exth`, `Exloss`, `Exeff` are properties |
| `ASU.MBCheck()`, `DAC.MBCheck()` | `MBcheck()` |

`celeste.FT.thermoFT.ThermoExergy` and `standard_chemical_exergy` remain importable from their old
location and dispatch to the new classes. `HB.for_sol` / `HB_Adiabatic.for_sol` remain as aliases of `f_sol`.
