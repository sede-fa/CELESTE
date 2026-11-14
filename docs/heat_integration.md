# Heat integration

Both tools read the same stream table.

```python
streams = system.pinch_stream_table()      # from the Heater units of a solved system
```

| Column | Meaning |
|---|---|
| index (or `segment_id`) | name of the stream; for a system the unit path, e.g. `cond.cooler` |
| `segment_type` | `sensible` or `latent` |
| `side` | `hot` (to be cooled) or `cold` (to be heated) |
| `T_in`, `T_out`, `CP` | temperatures in °C and heat capacity flow in kW/K (sensible) |
| `T_phase`, `Q` | temperature in °C and duty in kW (latent, isothermal) |
| `h` (optional, HEN only) | film heat transfer coefficient in kW/m²/K |

`pinch_stream_table(include=..., exclude=..., min_duty=..., min_delta_t=...)`
filters the heaters; the table is an ordinary DataFrame and can be edited or
written by hand.

## Pinch analysis

```python
from celeste.HEN import PinchAnalyser

pa = PinchAnalyser(streams, dt_min=10).solve()
pa.summary()                      # hot_utility, cold_utility, heat_recovery, pinch temperatures
pa.problem_table, pa.heat_cascade
pa.plot_summary()                 # composite curves + grand composite curve
PinchAnalyser.sweep_dt_min(streams)   # targets against dt_min
```

## Heat exchanger network synthesis

```python
from celeste.HEN import HENSynthesiser, HENCosts, Utility

hen = HENSynthesiser(streams, dt_min=10).solve()
hen.summary()
hen.network          # process-process exchangers
hen.utilities        # heaters and coolers
hen.check()          # energy balance of every stream and smallest approach temperature
hen.plot_grid_diagram()
```

`summary()` lists the utilities of the network next to the pinch targets, the
number of units, total area and annual cost. `network` has one row per
exchanger: streams, stage, duty, the four temperatures, approach temperatures,
LMTD, U, area and cost.

### Options

| Argument | Default | Meaning |
|---|---|---|
| `dt_min` | – | minimum approach temperature of every exchanger |
| `n_stages` | max(hot, cold streams), at most 4 | stages of the superstructure; more stages allow more complex networks and take longer |
| `costs` | `HENCosts()` | utility prices and exchanger cost law |
| `hot_utility`, `cold_utility` | just outside the stream temperatures | `Utility(T_in, T_out, h)` |
| `latent_dT` | 1 K | temperature glide given to isothermal streams |
| `h_default` | 1 kW/m²/K | film coefficient of streams without an `h` entry |
| `solve(refine=True)` | | optimise duties and areas after the topology search |
| `solve(objective='cost')` | `'cost'` | what the topology search minimises, see below |
| `solve(time_limit=600, mip_gap=1e-4)` | | limits of each topology MILP (seconds, relative gap) |

### Objectives

| `objective=` | Topology search | Typical use |
|---|---|---|
| `'cost'` | utility cost + fixed charge per exchanger (areas added by the refinement) | design studies with project cost figures |
| `'mer'` | least hot utility the superstructure allows, then the fewest exchangers that achieve it | energy targets as a hard requirement; the classic MER design |
| `'min_units'` | fewest exchangers, heaters and coolers, then the least utility cost with that many | simple networks, retrofit screening |

`'mer'` and `'min_units'` solve two MILPs in sequence (each with `time_limit`),
and their refinement keeps the heat recovery of the topology, so the utilities
do not grow. For the four-stream example of Linnhoff and Hindmarsh (dt_min 20 K):

| Objective | Exchangers + heaters + coolers | Hot utility / kW |
|---|---|---|
| `'cost'` (default costs) | 3 + 1 + 1 | 137.5 |
| `'mer'` | 4 + 2 + 1 | 107.5 (the pinch target) |
| `'min_units'` | 2 + 1 + 1 | 262.5 |

Stream splitting is part of the superstructure for every objective: a stream can
meet several partners in the same stage (parallel branches, mixed isothermally).

`HENCosts(hot_utility=100, cold_utility=15, unit_fixed=8000, area_coeff=500, area_exp=0.8)`:
utilities per kW and year, exchangers `unit_fixed + area_coeff · A^area_exp` per
year. **These defaults are placeholders** – the network depends on them, so set
them to project values.

### Method

The stage-wise superstructure of Yee and Grossmann (1990) is used: in each
stage every hot stream may exchange heat with every cold stream; split streams
are mixed isothermally; utilities are placed at the stream ends. Solving
topology and areas at once is a non-convex MINLP that needs a dedicated solver.
The problem is decomposed instead:

1. **Topology – MILP.** With isothermal mixing all constraints are linear.
   Utility cost plus a fixed charge per exchanger is minimised with HiGHS
   (`scipy.optimize.milp`), `dt_min` enforced at both ends of every exchanger.
   The solver proves optimality of this step.
2. **Duties and areas – NLP.** For the chosen matches all temperatures are
   linear in the duties. The total annual cost including area cost is minimised
   under linear constraints (SLSQP, exact gradients from JAX). The result is
   kept only if it is feasible and cheaper than the starting point.

Consequences:

- The network is always feasible; `check()` confirms it.
- It is the optimum of the two-step procedure, not a proven global optimum of
  the full cost function. The topology is chosen without knowing the area cost.
- With utility prices that dominate the exchanger cost the network reaches the
  pinch targets exactly (tested).

### Size and run time

| Problem | Binary variables | Time |
|---|---|---|
| 2 hot × 2 cold, 2 stages | 12 | 0.2 s |
| 3 hot × 2 cold, 3 stages | 23 | 0.7 s |
| 6 hot × 6 cold, 4 stages | 156 | hits a 30 s limit with a 14 % gap; use the 600 s default or more |

For large tables reduce `n_stages`, raise `time_limit`, or remove small streams
with `pinch_stream_table(min_duty=...)`. `summary()['milp_optimal']` and
`['milp_gap']` tell whether the topology search finished.

### Limits

- Constant heat capacity flow per stream. Split a stream with strongly varying
  cp into several segments (several heaters in the system model).
- Each segment is an independent stream; segments that belong to one physical
  stream are not forced to be exchanged in sequence.
- Isothermal mixing of split streams; no bypasses.
- One hot and one cold utility.
