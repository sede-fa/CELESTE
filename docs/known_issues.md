# Open points from the review

Open problems that were found while reviewing the code and not yet changed.
Behaviour that is intended but surprising is listed in
[sharp_corners.md](sharp_corners.md).

## Standard library

- **`SysBase.EBcheck`** counts sources and sinks only; for boxes with their own
  inlet and outlet flanges it does not close.
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
