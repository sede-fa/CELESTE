# Open points from the review

Items that were found while reviewing the code and deliberately **not**
changed, because the right answer depends on modelling intent or on data.

## Fischer–Tropsch module

- The FT model does not determine the phase of a stream; it evaluates the phase
  set on the flange. A liquid flange above the bubble point is still liquid.

## Standard library

- **Reference composition of `mediaFT.Syncrude`.** Its first two entries (H2,
  CO2) are zero, so an `FT` reactor created without `F0_Xi` has no hydrogen and
  no conversion.
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
