<p align="center">
  <img src="./misc/CELESTE logo FINAL.svg" alt="CELESTE logo" width="500">
</p>

# CELESTE

CELESTE is a Python framework for conceptual process modelling and exergy
analysis. A process is described as unit models (reactors, compressors, heat
exchangers, electrolysers, separators, …) connected by streams. CELESTE solves
the mass and energy balances and reports energy, entropy and exergy for every
unit and for the whole system. Pinch analysis and heat exchanger network
synthesis work directly on a solved system.

It is aimed at early design studies of energy and chemical conversion
processes (power-to-X, fuel synthesis, electrolysis, heat integration) where
transparent models and a complete second-law balance matter more than
rigorous equipment design.

```python
import numpy as np
from celeste.CSL import media, reactors

reactor = reactors.MR('methanation', T=600.0)                            # design: isothermal at 600 K
reactor.F0.n = 10.0                                                        # feed, mol/s
reactor.F0.P = 10.0                                                        # bar
reactor.F0.Xi = np.array([0.60, 0.01, 0.20, 0.10, 0.08, 0.01])             # H2, CH4, CO2, CO, H2O, N2
reactor.run_sol()

reactor.F1.Xi      # outlet composition at chemical equilibrium
reactor.Q          # heat duty, kW
reactor.Exloss     # exergy destroyed, kW
```

## Core ideas

- **Flanges carry streams.** Every port of a unit is a `Flange` holding molar
  flow, composition, temperature and pressure; enthalpy, entropy and exergy
  follow from the medium. Units are connected flange to flange.
- **Design inputs in the constructor, operating state on the flanges.** A
  unit is created with its design quantities (medium, efficiency, volume,
  model choice); feeds and operating specifications are set on the flanges,
  e.g. `comp.F1.P = 30.0` for a discharge pressure or
  `heater.F1.T = 600.0` for an outlet temperature.
- **Systems are classes.** A system derives from `SysBase`, creates its units
  and connects them in flow order in `seq_sol`. Recycles are converged with
  `Recycle_Tear`. A system can be used inside another system like a unit.
- **Every balance is checked.** Units report `Q`, `W`, `Exloss` and `Sgen`;
  systems check mass and energy balances over their boundary and break the
  exergy destruction down by unit type.
- **One property kernel for NumPy and JAX.** The same code serves the unit
  models and JAX, so sweeps, derivatives and compiled flowsheets are
  available where needed.

## What is in the package

| Package | Content |
|---|---|
| `celeste.CSL` | **CELESTE Standard Library** – unit models, media and systems |
| &nbsp;&nbsp;`interfaces` | `Flange`: the port that connects units and carries a stream (n, Xi, T, P → h, s, g, cp, exergy) |
| &nbsp;&nbsp;`media` | Fluid definitions: ideal-gas NASA media (`Air`, `Fuel_CH4`, `Fuel_MeOH`, …), Peng–Robinson real-fluid media (`Fuel_CH4_PR`, `Water_PR`, …), CoolProp media |
| &nbsp;&nbsp;`flow`, `heat`, `components` | Sources, sinks, mixers, splitters; heaters and heat exchangers; compressors, pumps, valves, black-box separators, ASU, DAC |
| &nbsp;&nbsp;`reactors`, `electrochem` | Equilibrium, yield and kinetic reactors (methanation, Haber–Bosch incl. plug-flow bed, methanol, reforming); solid oxide cell and alkaline electrolyser |
| &nbsp;&nbsp;`separation` | `Flash`: vapour–liquid flash drum with a stability-tested phase split |
| &nbsp;&nbsp;`equilibrium` | Equilibrium constants from NASA Gibbs energies, shared by reactors and cells |
| &nbsp;&nbsp;`systems` | Ready-made subsystems (“boxes”) and example systems: compression train, SOEC and AEL boxes, condenser box, methanol loop |
| `celeste.HEN` | `PinchAnalyser` (targets, composite curves) and `HENSynthesiser` (heat exchanger network) |
| `celeste.FT` | Fischer–Tropsch media (C1–C50 with liquid phase), units and a Raoult flash |
| `celeste.jx` | JAX kernels: properties, Peng–Robinson equation of state, flash, Equinox unit prototypes |
| `celeste.solvers` | Robust nonlinear solves shared by all unit models |

## Installation

Python 3.12 or newer.

```bash
pip install git+https://github.com/DLR-TT-ESI-EHT/CELESTE.git
```

Dependencies (installed automatically): NumPy, SciPy, pandas, JAX, Equinox,
CoolProp, NetworkX, Matplotlib, openpyxl. Optional extras:

```bash
pip install "celeste[cantera] @ git+https://github.com/DLR-TT-ESI-EHT/CELESTE.git"
```

| Extra | Needed for |
|---|---|
| `cantera` | carbon deposition check (`Flange.getCarbon`, `celeste.functions.carbonFraction`) |
| `notebook` | rendering Mermaid graphs in notebooks (`celeste.functions.mm`) |
| `test` | running the test suite |

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/DLR-TT-ESI-EHT/CELESTE.git
```

```bash
pip install -e "CELESTE[dev]"
```

Check the installation with `python -c "import celeste; print(celeste.__version__)"`.

## Quick start

### A system

A system is a class derived from `SysBase`. Units are created in `__init__`,
`seq_sol` connects them in flow order and solves them one after the other.

```python
from celeste.CSL import SysBase, components, flow, heat, media, turbomachinery

class AirCompression(SysBase):
    def __init__(self, name, n=1.0, P_out=5.0, T_out=310.0):
        super().__init__(locals())
        self.source = flow.SourceM(name + '.source', media.Air(), F1_n=n, F1_T=300.0, F1_P=1.0)
        self.comp = turbomachinery.CompIsent(name + '.comp', media.Air(), eff=0.8)
        self.comp.F1.P = P_out  # operating spec on the flange
        self.cooler = heat.Heater(name + '.cooler', media.Air(), media.Air())
        self.cooler.F1.T = T_out
        self.sink = flow.Sink(name + '.sink', media.Air())

    def seq_sol(self, store=False):
        self.connect_seq(self.source.F1, self.comp.F0, store=store)
        self.comp._sol = self.comp.run_sol()
        self.connect_seq(self.comp.F1, self.cooler.F0, store=store)
        self.cooler._sol = self.cooler.run_sol()
        self.connect_seq(self.cooler.F1, self.sink.F0, store=store)

plant = AirCompression('plant')
plant.run_sol()

plant.work_df()            # work of every unit
plant.heater_df()          # heaters and coolers
plant.exdest_df()          # exergy destruction by unit type
plant.MBcheck(), plant.EBcheck()                 # overall mass and energy balance
print(plant.create_graph('mermaid', save=False)) # flowsheet as a Mermaid graph
```

### Heat integration

```python
from celeste.HEN import PinchAnalyser, HENSynthesiser

streams = plant.pinch_stream_table()                 # one row per heater / cooler
pinch = PinchAnalyser(streams, dt_min=10).solve()
pinch.summary()                                      # minimum hot and cold utility, pinch temperature
pinch.plot_summary()                                 # composite and grand composite curves

hen = HENSynthesiser(streams, dt_min=10).solve()     # same table, one more line
hen.summary()                                        # utilities, number of units, area, annual cost vs. targets
hen.network                                          # matches: duty, temperatures, LMTD, area, cost
hen.plot_grid_diagram()
```

### Vapour–liquid separation

```python
from celeste.CSL import media, separation

drum = separation.Flash('drum', Medium=media.Fuel_CH4())
drum.F0.n = 10.0
drum.F0.T = 303.15
drum.F0.P = 20.0
drum.F0.Xi = np.array([0.60, 0.05, 0.10, 0.05, 0.20, 1e-8])
drum.run_sol()
drum.beta          # vapour fraction: 0.802
drum.F1, drum.F2   # gas and liquid outlet
```

### JAX

```python
import jax, jax.numpy as jnp
from celeste.jx import ideal

gas = ideal.ideal_gas(['O2', 'N2'])
x = jnp.array([0.21, 0.79])
h = jax.vmap(ideal.enthalpy, (0, None, None))(jnp.linspace(300., 1500., 100_000), x, gas)   # 100 000 states
cp = jax.grad(ideal.enthalpy)(800.0, x, gas)                                                # exact derivative
```

More complete, runnable scripts are in [`examples/`](examples):

| Script | Shows |
|---|---|
| [`01_components.py`](examples/01_components.py) | flanges and single unit models |
| [`02_system_pinch_hens.py`](examples/02_system_pinch_hens.py) | methanol loop with recycle → stream table → pinch analysis → heat exchanger network |
| [`03_flash.py`](examples/03_flash.py) | flash drum, adiabatic flash, real-fluid media |
| [`04_jax.py`](examples/04_jax.py) | batched properties, derivatives, a compiled flowsheet, phase diagrams |

The notebook [`Tutorial/Tutorial.ipynb`](Tutorial/Tutorial.ipynb) explains step
by step how to write your own unit models and how to keep them in your own
library.

## Documentation

| Document | Content |
|---|---|
| [docs/conventions.md](docs/conventions.md) | units, signs, naming, structure of unit and system models, balances, releases |
| [docs/thermodynamics.md](docs/thermodynamics.md) | property models, their limits, exergy definition |
| [docs/separation.md](docs/separation.md) | flash models: usage, algorithm, validation, limits |
| [docs/recycles.md](docs/recycles.md) | converging recycles and heat-recovery loops |
| [docs/electrochemistry.md](docs/electrochemistry.md) | solid oxide cell: extents, energy specification, ideal voltage |
| [docs/heat_integration.md](docs/heat_integration.md) | pinch analysis and heat exchanger network synthesis |
| [docs/jax_roadmap.md](docs/jax_roadmap.md) | the JAX layer, Equinox units, equation-oriented solution |
| [docs/sharp_corners.md](docs/sharp_corners.md) | intended behaviour that can surprise a new user |
| [docs/known_issues.md](docs/known_issues.md) | open problems |
| [CHANGELOG.md](CHANGELOG.md) | what changed, including everything that changes numerical results |

## Units

| Quantity | Unit | | Quantity | Unit |
|---|---|---|---|---|
| temperature | K (°C in the pinch / HEN stream table) | | enthalpy, Gibbs energy, exergy | kJ/mol |
| pressure | bar | | entropy, heat capacity | kJ/mol/K |
| molar flow | mol/s | | heat, work, exergy loss | kJ/s = kW |
| mass flow | g/s | | density | kg/m³ |

Heat `Q` and work `W` are positive when they enter a unit.

## Performance

Properties are evaluated by one kernel (`celeste.jx.nasa`) through NumPy for
the unit models and through JAX for batches and derivatives. Typical timings
on a laptop:

| Operation | Time |
|---|---|
| `import celeste.CSL` | 0.55 s |
| create a `Flange` | 56 µs |
| set `T` on a flange and read `h` | 16 µs |
| exergy of a stream (`Flange.Ex`) | 24 µs |
| two-stage compression: build, solve, exergy loss | 1.7 ms |
| methanation equilibrium reactor: build, solve, exergy loss | 1.9 ms |
| 100 000 enthalpies with `jax.vmap` | 0.09 s |

## Tests

```bash
pip install -e ".[test]"
```

```bash
python -m pytest
```

The suite covers the property kernels (against reference values recorded with
the original implementation), every unit model (mass, element and energy
balances, second law), the system models and reports, the equation of state and
the flash (thermodynamic consistency, reference fluids, an independent flash
implementation), heat integration and the JAX layer.

## Contributing

Contributions are welcome: new unit models, media, validation cases, fixes and
documentation.

- **Workflow.** Branch from `main`, open a pull request; the test suite runs
  on every pull request.
- **Conventions.** Read [docs/conventions.md](docs/conventions.md) before
  writing a unit model: port names, sign conventions, design inputs versus
  flange state, how to solve, how to test.
- **Tests.** Every model needs tests for its mass, element and energy balances
  and the second law (`Exloss = T0·Sgen` where the exergy model allows it),
  plus reference values where data exist.
- **Results that change.** A change that alters numbers must be listed in
  `CHANGELOG.md` under *Results that change*, with the size of the effect.
- **Retiring an interface.** Keep the old name or keyword working and emit a
  `CelesteDeprecationWarning` (see *Retiring a name* in the conventions); list
  it under *Deprecated* in the changelog.
- **Releases** are tags `vX.Y.Z` on `main`; the version comes from the tag.

## Licence

BSD 3-Clause, see [LICENSE](LICENSE). Developed at the German Aerospace Center
(DLR), Institute of Engineering Thermodynamics.
