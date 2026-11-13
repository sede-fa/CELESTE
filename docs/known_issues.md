# Open points from the review

Open problems that were found while reviewing the code and not yet changed.
Behaviour that is intended but surprising is listed in
[sharp_corners.md](sharp_corners.md).

## Packaging

- The repository has no release tag yet, so the version is derived from the
  commit count (`0.0.1.devN`) until the first release is tagged (see
  `docs/conventions.md`, *Releases*).

## Heat integration

- The HEN topology search slows down quickly beyond about ten streams
  (see `docs/heat_integration.md`).
- The cost defaults in `HENCosts` are placeholders.
