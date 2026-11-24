# Building and solving systems

Three tools on every `SysBase` system, independent of each other:

| | |
|---|---|
| `link(outlet, inlet)` | declare the streams once; the solve order and the tear streams are generated (see [recycles](recycles.md#declaring-the-flowsheet-instead)) |
| `add_spec(target, value, vary=...)` | design specifications instead of hand-written `Optimiser` methods |
| `equations()` / `print_equations()` | what each unit solves: its equations, unknowns and current residual, or "algorithmic" for units that assign the outlet from the inlet |
| `pfd()` | interactive process flow diagram with stream tables on hover |

## Design specifications

```python
plant.add_spec('sink1.F0.n', 2.0, vary='source1.F1.n', bounds=(0.1, 10.0))
plant.add_spec(lambda s: s.sink1.F0.Xi[0], 0.45, vary='reformer.T', name='H2 fraction')
plant.solve_specs()
plant.specs            # table: value, current value, adjusted input, bounds
```

- `target`: a dotted path relative to the system (indices allowed: `'reactor.F1.Xi[6]'`, properties such as
  `'carbon_efficiency'`) or a function of the system.
- `vary`: the dotted path of the input that is adjusted (any attribute or property that can be set:
  `'feed.F1.n'`, `'heater.F1.T'`, `'split.split'`, `'reformer.T'`).
- All specifications are solved together with `solvers.solve` (bounded when `bounds` are given); every
  iteration is one full `run_sol` of the system, recycles included. `solve_specs(run=...)` takes another
  solve function, `clear_specs()` removes them.
- Residuals are divided by `scale` (default `|value|`), so specifications of different size are balanced.
- The derivatives are finite differences of a converged flowsheet: keep the recycle tolerance (`eps`) well
  below the spec tolerance (default 1e-8 relative). The equation-oriented form in the JAX layer
  (`examples/05_soec_flowsheet.py`) solves specifications with exact derivatives instead.

## Process flow diagram

```python
plant.run_sol()
plant.pfd('plant.html')     # or just `plant.pfd()` as the last line of a notebook cell
```

![PFD of the methanol loop, hovering the recycle](figures/pfd_methanol_loop.png)

One self-contained HTML file: an SVG drawing and a few lines of JavaScript, no downloads, light and dark mode.
Hovering a stream shows n, m, T, P, h, Ex and the composition; hovering a unit its class, Q, W, exergy loss and
whether its solver converged. A click pins the table.

The streams drawn are the declared ones (`link`) or, for a system with a hand-written `seq_sol`, the connections
made by its last run. The layout is the layered method used by Graphviz dot and ELK: recycles are torn (dashed,
the same choice as `solve_order`), units are placed by their longest path from the feeds, the order within a
column is improved by barycentre sweeps, and the streams are routed orthogonally - recycles along lanes below
the units. Colours mark the unit type (heat exchange, reactor, electrochemical, separation, pressure change,
mixer/splitter, subsystem). `create_graph('mermaid')` and `create_graph('networkx')` remain available.

### What the diagram shows

The drawing lives in its own package, `celeste.pfd`, which knows nothing about CELESTE units: every unit and
stream describes itself with a `pfd_view()` method.

- `ModelBase.pfd_view()` returns a `UnitView`: `kind` (the colour of the box, class attribute `pfd_kind`) and the
  rows Q, W, Ex loss and converged. Unit families add their own rows: heat exchangers the effectiveness and the
  temperatures of both sides, heaters T in / out, compressors and turbines PR, efficiency and T out, SOC/PCC the
  current, j, ideal and cell voltage, conversion and T out, reactors T, P and the extents, the flash drum T, P and
  the vapour fraction, the splitter its split.
- `Flange.pfd_view()` returns a `StreamView`: n, m, T, P, h, Ex (and the phase) and the composition.

To show something else, override `pfd_view` in your class:

```python
class MyStack(electrochem.SOC):
    def pfd_view(self):
        return super().pfd_view().add('ASR', self.ASR_mean, 'ohm cm2')
```

or replace the view of some units in one drawing, by name or by class:

```python
from celeste.pfd import UnitView
plant.pfd(views={'soc': lambda u: UnitView('electrochem').add('V', u.Vop, 'V'),
                 heat.Heater: lambda u: UnitView('heat').add('Q', u.Q, 'kW')})
```

`celeste.pfd.draw(name, units, streams)` draws any flowsheet from `{label: unit}` and
`[(source, destination, stream, title)]` (sources and destinations are labels or `Terminal`s), so other models can
use the same diagrams by giving their objects a `pfd_view`.

## Equation-oriented solve

The declared streams can also be solved all at once, on top of the sequential-modular solve (which stays as it is):

```python
eo = plant.eo_compile()      # 1. compile: unknowns and equations from the links and the units
print(eo)                    #    EOSystem(socsys: 48 unknowns, 48 equations; ...)
eo.initialise()              # 2. starting vector from one sequential sweep ('current' starts from the flanges)
res = eo.solve()             # 3. solve every equation together; the solution is in the flanges
plant.eo_sol()               # the three steps in one call
plant.solve_specs(eo=True)   # every design-spec iteration is an eo_sol, warm-started from the previous one
```

Unknowns are the state (n, T, P, Xi) of every stream between two units and the unknowns of every unit that solves a
system of equations (`f_sol`, for example the SOC extents). Equations are "stream = outlet of the unit it leaves" and the
`f_sol` residuals of those units (`print_equations()` lists them). A unit without `f_sol` is an explicit function of its
inlet. The sweep matters: from a bad start the equations can converge to a wrong root, or not at all.
