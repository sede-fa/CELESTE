# Open points from the review

Open problems that were found while reviewing the code and not yet changed.
Behaviour that is intended but surprising is listed in
[sharp_corners.md](sharp_corners.md).

## Standard library

- **`SOC.Qloss`** enters the energy balance like a heat input (`Q` is defined as
  ΔH − W − Qloss). If `Qloss` is meant as a loss to the surroundings, its sign
  in `SOC.Q` should be flipped; it defaults to 0, so no example is affected.

## Packaging

- The repository has no version tags; the version is derived from the commit
  count (`0.0.1.devN`).
- `.gitattributes` refers to `CELESTE2/_version.py`, which does not exist.

## Heat integration

- The HEN topology search slows down quickly beyond about ten streams
  (see `docs/heat_integration.md`).
- The cost defaults in `HENCosts` are placeholders.
