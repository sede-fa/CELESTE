# JAX and Equinox: where the framework stands and where it can go

## What JAX is used for today

| Layer | Module | Status |
|---|---|---|
| ideal-gas properties | `celeste.jx.nasa`, `celeste.jx.ideal` | in use – the kernel behind all NASA media |
| real-gas / liquid properties | `celeste.jx.eos`, `celeste.jx.fluid` | in use – behind `Medium_PRbase` |
| phase equilibrium | `celeste.jx.flash` | in use – behind `separation.Flash` |
| network cost gradients | `celeste.HEN.synthesis` | in use |
| unit models | `celeste.jx.units` | prototype |

Importing a JAX module of CELESTE switches JAX to 64-bit floats
(`jax_enable_x64`); single precision is not sufficient for these calculations.

## Why the unit models still run on NumPy

The CSL units are objects that are updated one state at a time. For that
pattern the measured cost of one property update is

| Backend | per call |
|---|---|
| Numba (previous) | 7 µs |
| NumPy (now) | 16 µs |
| `jax.jit` call | 15 µs + conversion of the results |

JAX has no advantage for single scalar calls: the time is dispatch overhead,
not arithmetic. Its advantages appear when a whole calculation is inside one
compiled function:

- 100 000 states: 90 ms with `jax.vmap`, 1.6 s as a Python loop;
- exact derivatives (`jax.grad`, `jax.jacfwd`) instead of finite differences;
- a complete flowsheet as one compiled, differentiable function.

So the kernel is written once (`celeste.jx.nasa`) and instantiated for both
backends. Nothing has to be rewritten when a model moves to JAX.

## The Equinox pattern

A unit is an immutable `equinox.Module`; calling it maps inlet streams to
outlet streams and results. No state is stored.

```python
import jax, jax.numpy as jnp
from celeste.jx import ideal, units

gas = ideal.ideal_gas(['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'])

def methane_slip(T, P):
    feed = units.Stream(n=1.0, T=500.0, P=P, x=jnp.array([0.6, 0.01, 0.2, 0.1, 0.08, 0.01]))
    hot, duty = units.Heater(gas, T)(feed)
    out, q = units.ShiftReformingEquilibrium(gas, T, P)(hot)
    return out.n*out.x[1]

jax.jit(jax.vmap(methane_slip, (0, None)))(jnp.linspace(500., 1000., 50), 10.0)   # a temperature sweep
jax.grad(methane_slip)(700.0, 10.0)                                               # sensitivity
```

Available prototypes: `Stream`, `Heater`, `Compressor`, `Mixer`,
`ShiftReformingEquilibrium`, and the solver `newton` (damped Newton with an
exact Jacobian; derivatives of the solution by the implicit function theorem).
They are tested against their CSL counterparts (`tests/test_jx_units.py`).

Differences to the CSL worth knowing:

| | CSL | Equinox prototype |
|---|---|---|
| equilibrium reactor unknowns | outlet flow and all mole fractions (7) | reaction extents (2), logarithmic mass action laws |
| compressor | mean heat capacity between inlet and outlet | exact isentropic path (work differs by ≈ 0.5 %) |
| failure handling | `ConvergenceWarning` | none yet – see below |

## Real-gas properties without CoolProp

CoolProp cannot be called from compiled JAX code. `celeste.jx.eos` and
`celeste.jx.fluid` replace it inside the JAX layer:

- Peng–Robinson with an alpha function fitted to saturation pressures, so
  vapour pressures are within 1 % of the reference equations;
- gas, liquid and two-phase states from one function, `fluid.state_pt`;
- 16 species, extensible through `tools/build_eos_table.py`.

Accuracy is that of a cubic equation (see `docs/thermodynamics.md`). If higher
accuracy is needed for a specific fluid, options that stay inside JAX are:

1. volume translation (Péneloux) for liquid densities – a small addition to `eos.py`;
2. multi-parameter Helmholtz equations for single fluids (water, CO2, H2):
   the functional forms are explicit in T and ρ and differentiable, but each
   fluid needs its coefficient set and a density solve;
3. tabulated properties with differentiable interpolation, generated once from
   CoolProp.

`Thermo_CP` remains available for the stateful CSL models.

## Suggested migration path

1. **Now** – keep modelling in the CSL; use `celeste.jx` for sweeps,
   sensitivities and property work. Nothing in existing models has to change.
2. **Port units one at a time** into `celeste.jx.units`, each with a test
   against its CSL counterpart, starting with explicit units (splitter, valve,
   black-box separators), then the equilibrium and yield reactors, then SOC and
   AEL. Formulating reactors in reaction extents removes most convergence
   problems.
3. **Add what the prototype lacks before relying on it:**
   - exergy and entropy generation as pure functions (`Thermo_PR.exergy` shows
     the formula that needs no dead-state flash);
   - a result structure that reports convergence, since JAX code cannot raise
     warnings from compiled functions;
   - real-fluid units on `fluid.state_pt` (flash drum, condenser, pump);
   - a system layer: sequential-modular composition is plain function
     composition; recycles become a fixed-point or root-finding problem with
     implicit differentiation (Optimistix provides these solvers).
4. **Then** equation-oriented solving and gradient-based optimisation of whole
   flowsheets become possible, and the CSL objects can turn into thin wrappers
   around the Equinox units, as `Thermo` already is around the property kernel.

Constraints to plan for:

- compilation time on first use (seconds per distinct flowsheet structure);
- array shapes must be static: the number of species is fixed per compiled
  function;
- reverse-mode differentiation does not pass through `lax.while_loop`; the
  flash is differentiable in forward mode, and solutions of `newton` in both
  modes through the implicit function theorem.
