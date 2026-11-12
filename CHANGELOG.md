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
- **`celeste.HEN.HENSynthesiser`** – heat exchanger network synthesis from the pinch stream table
- **`CSL.systems.ProcessBoxes`** – `CondenserBox`, `MethanolLoop`
- **`RecycleMixin.Recycle_Tear`** – recycle convergence on the full tear stream (Anderson, Wegstein, direct)
- **`celeste.solvers`** – `solve`, `solve_scalar`, `ConvergenceWarning`
- `CSL.equilibrium` – equilibrium constants shared by reactors and the SOC
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
- `SOC` accepts `T_soc0`, `T_soc1`, `P_soc` in addition to `Tsoc0`, `Tsoc1`.

### Fixed

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
| `Flange.Ex` | overwrote fractions below 1e-6 in the flange with 1e-12 and relabelled FT flanges as gas | reads only | downstream values at the 1e-6 level |
| Exergy of media with species outside the FT table (`CH3OH`, `C3H8`, …) | dead state from the FT correlation: `nan` for methanol, error for `CH3OH(L)` | dead state from the property model of the medium | methanol media now have an exergy; `media_C3`: ≤ 0.02 kJ/mol |
| `MeOH_Kinetic` | converged to negative mole fractions for the documented example | physical solution | different outlet |
| `ASU`, `DAC` | `F2.T` stale; `DAC` overwrote the inlet temperature with the outlet's | both outlets at the inlet temperature | energy and exergy balance of these units |
| `Thermo_CP.cp` | J/mol/K | kJ/mol/K, as for every other medium | factor 1000 |
| Reactor and SOC results | – | same roots, solver tolerance tighter | relative 1e-6 or smaller |

The reference values of the original implementation are stored in
`tests/data/golden_thermo.json`; `tests/test_thermo.py` and
`tests/test_exergy.py` document every deviation above.

### Deprecated

These keep working and emit a `CelesteDeprecationWarning`:

| Name | Use instead |
|---|---|
| `Thermo.updateBoolArr()`, `thermoArr()`, `getCoeffs()` | nothing – handled inside `calcThermoSep()` |
| `EquilBase.ExBs()` (methanol reactors) | `Exth`, `Exloss`, `Exeff` are properties |
| `ASU.MBCheck()`, `DAC.MBCheck()` | `MBcheck()` |

`HB.for_sol` / `HB_Adiabatic.for_sol` remain as aliases of `f_sol`.
