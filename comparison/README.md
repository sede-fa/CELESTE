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

`snapshots/main.json` is the baseline from the original code (before the
overhaul). The original code needs `numba` and `cantera` installed.

The notebook uses the former constructor keywords (`F0_n=`, `F0_T=`, `T=`,
`P=`), which still work on the reworked code (deprecated). Errors in a model are
recorded as results rather than raised.

## Notes on the baseline

- The original `SOC` lacks `T_soc0`, `T_soc1`, `P_soc`; a notebook-only shim adds
  them so the SOC system models can be built. Applied shims are stored in the snapshot.
- On the original code these error and are recorded as such: all FT cases (numba
  TypingError), `SOCBasicSys`, `SimpleCompressorSys`, `SimpleHeaterSys`, `gpu_3s`, `AELSys`.
- `SOCsysRecupe_fuel_cell` does not converge on either version; treat it as unreliable.

## Expected differences

Equilibrium constants (NASA-Gibbs), exergy loss / Sgen of GPU and methanol
components, SOC ideal voltage (stream dG vs Nernst) and SOEC system totals.
See "Results that change" in `CHANGELOG.md` on the reworked branch.
