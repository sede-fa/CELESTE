# Open points from the review

Items that were found while reviewing the code and deliberately **not**
changed, because the right answer depends on modelling intent or on data.

## Fischer–Tropsch module

The FT module was restored (it could not run) and moved off Numba with
identical numerical results. Two things in `FT/thermoFT.py` look unintended:

1. **Liquid phase is not applied when the phase is set on a flange.**
   `interfacesFT.Flange(..., phase='Liquid')` sets `Medium.phase`, but the
   switch that selects liquid properties (`Thermo._phaseID`) is only evaluated
   when the medium is constructed, and the media in `mediaFT` are constructed
   as gas. Liquid FT flanges therefore return gas-phase properties. The swap of
   `H2O` for `H2O(L)` in `getSpecies` is evaluated once at construction too.
2. **Enthalpy of vaporisation.** `_VapCalc` uses `B*(1 - T/Tc)*n` inside the
   valid range and `B*(1 - Tmin/Tc)**n` below it; the docstring of `calVAP`
   gives the Watson form `B*(1 - T/Tc)**n`. The two branches do not meet at
   `Tmin`.

Fixing either changes every liquid-phase FT result. Related: `getSpecies`
reads `species == 'H2O' and phase == 'Liquid' or phase == 'liquid'`, which
sends *every* species to the NASA list when the phase is spelled `'liquid'`.

Smaller points in `FT/reactorsFT.py`: `RStoic.Eqs` computes `Q` from
`F0.n*F0.n` (flow squared instead of flow times enthalpy), and the feed water
flange `FH0` is created with `mediaFT.Water()` but commented as hydrogen.

## Standard library

- **Reference composition of `mediaFT.Syncrude`.** Its first two entries (H2,
  CO2) are zero, so an `FT` reactor created without `F0_Xi` has no hydrogen and
  no conversion.
- **`HB` and `HB_Adiabatic`** use the same rate expressions with different
  constants (`alpha`, `w`, `k_2`, `K_3`). They were left as two models; if one
  parameter set is the intended one, the rate functions can be shared.
- **`Reformer.K_WGS`** uses 32.115 where `MR`, `SOC` and `MeOH_WGS` use 32.1153.
  The shared `CSL.equilibrium.K_WGS` was not applied to the reformer to keep
  its results unchanged.
- **`Valve` and `Pump` defaults** (`F1_P=3.0` for a valve whose inlet default
  is 1 bar; composition defaults tied to `Fuel_CH4`) only make sense inside a
  system that overwrites them.
- **`Thermo_CP` with a liquid-forming default composition.** The reference
  state is evaluated without a phase specification while the stream state uses
  the given phase; for `Fuel_CH4_CP()` at its default composition (90 % water,
  298 K) the enthalpy differs from the NASA value by the heat of vaporisation.
- **`Flange.__eq__`** means "same type of port and medium" and is used by
  `connect_seq`. As a side effect flanges cannot be used in sets or as
  dictionary keys.
- **`SysBase.EBcheck`** counts sources and sinks only; for boxes with their own
  inlet and outlet flanges it does not close.
- **Mixed-model flash.** `Flash` with NASA media takes the phase split from
  Peng–Robinson and the properties from the media. Its exergy balance is
  approximate; in the methanol loop example the drum shows −0.3 % of the total
  exergy destruction. Use real-fluid media where this matters.
- **Black-box separators.** `GPU_2S` is the single-component special case of
  `GPU_2SV2`. Both are kept; a future clean-up could make `GPU_2S` a thin
  wrapper.
- **Legacy recycle methods** (`Recycle_FPI`, `Recycle_Wegstein_*`,
  `Recycle_Steffensen`, `Recycle_Aitken`, `Recycle_Hybrid_FPI_Aitken`)
  accelerate the total flow only. They are kept for existing systems;
  `Recycle_Tear` supersedes them (methanol loop: 21 passes instead of no
  convergence after 500).

## Packaging

- The repository has no version tags; the version is derived from the commit
  count (`0.0.1.devN`).
- `.gitattributes` refers to `CELESTE2/_version.py`, which does not exist.

## Heat integration

- The HEN topology search slows down quickly beyond about ten streams
  (see `docs/heat_integration.md`).
- The cost defaults in `HENCosts` are placeholders.
