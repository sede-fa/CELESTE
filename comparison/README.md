# Component and system comparison notebook

`component_models.ipynb` builds a fixed set of models (media, flow, components,
reactors, electrochemistry, system models, Fischer-Tropsch), solves them, and
stores every readable number in `snapshots/<label>.json`. The last section
diffs two snapshots (`RTOL`/`ATOL` at the top of that cell).

## Use

1. Check out the code version to test and run the notebook from `comparison/`.
   Set `CELESTE_LABEL` to name the snapshot (default is set in the setup cell):
   `CELESTE_LABEL=reworked jupyter nbconvert --execute --inplace component_models.ipynb`
2. Repeat on the other version with a different label.
3. Set `FILE_A` / `FILE_B` in the comparison cell and run it.

`snapshots/main.json` is the baseline from `main` as of the merge of "Mirror/develop" (9ef960f: type hints, turbomachinery, PCC,
transport properties). The code needs `numba` and `cantera` installed.

The notebook uses the former constructor keywords (`F0_n=`, `F0_T=`, `T=`,
`P=`), which still work on the reworked code (deprecated). Errors in a model are
recorded as results rather than raised.

## Notes on the baseline

- Compressor classes are looked up in `celeste.CSL.turbomachinery` when it exists, else in `components`.
- Imports of the system modules are guarded: cases that need a module that fails to import are recorded as errors.
  On `main` as of 9ef960f, `ElectBox` (and `celeste.FT`) fail to import because `components.CompIsent` moved to
  `turbomachinery`, so the SOEC/CO2/AEL systems, `Ncomp` and all FT cases are errors there.
- Also errors on `main`: `SOCBasicSys`, `SimpleCompressorSys`, `SimpleHeaterSys`, `gpu_3s`, the polytropic compressors
  (they do not exist there).
- `SOCsysRecupe_fuel_cell` does not converge on either version; treat it as unreliable.

## Expected differences

Equilibrium constants (NASA-Gibbs), isentropic compressor outlet (real-gas instead of mean-cp), PCC (about 1e-3),
`Lambda` of SOC/PCC (main divides by the electron flow), exergy loss / Sgen of GPU and methanol
components, SOC ideal voltage (stream dG vs Nernst) and SOEC system totals.
See "Results that change" in `CHANGELOG.md` on the reworked branch.
