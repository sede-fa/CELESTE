# Open points from the review

Open problems that were found while reviewing the code and not yet changed.
Behaviour that is intended but surprising is listed in
[sharp_corners.md](sharp_corners.md).

## Packaging

- The repository has no release tag yet, so the version is derived from the
  commit count (`0.0.1.devN`) until the first release is tagged (see
  `docs/conventions.md`, *Releases*).

## Exergy data

- For reacting units, `Exloss` and `T0·Sgen` differ by up to a few tenths of a
  percent of `Exloss` (SOC test cases: 0.001–0.09 kW, 0.15–0.35 %). Two causes:
  the tabulated standard chemical exergies (`celeste/data/Chem_exergies.csv`)
  do not match the NASA Gibbs energies exactly (per mol of reaction 0.34 kJ for
  methane reforming, −0.02 kJ for the shift, 0.004 kJ for hydrogen oxidation),
  and water that condenses at the dead state makes the composition term of the
  exergy nonlinear (see the exergy definition in `docs/thermodynamics.md`).
  Deriving the chemical exergies of the reference species from the NASA data
  would remove the first part.

## Heat integration

- The HEN topology search slows down quickly beyond about ten streams
  (see `docs/heat_integration.md`).
- The cost defaults in `HENCosts` are placeholders.
