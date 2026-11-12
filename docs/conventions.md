# Conventions

The rules below are what the standard library follows. New models should follow
them too; existing user models are not affected by any of this.

## Units and signs

| Quantity | Symbol | Unit |
|---|---|---|
| temperature | `T` | K |
| pressure | `P` | bar |
| molar flow | `n` | mol/s |
| mass flow | `m` | g/s |
| mole / mass fractions | `Xi` / `Yi` | – |
| enthalpy, Gibbs energy, exergy, LHV | `h`, `g`, `Ex`, `LHV` | kJ/mol |
| entropy, heat capacity | `s`, `cp` | kJ/mol/K |
| heat, work, exergy loss | `Q`, `W`, `Exloss` | kJ/s (kW) |
| entropy generation | `Sgen` | kJ/K/s |

`Q` and `W` are positive when they **enter** the unit. Enthalpies include the
enthalpy of formation, so reaction heat needs no separate term.

## Ports

A port is a `Flange` stored as an attribute of the unit.

| Name | Meaning |
|---|---|
| `F0` | inlet |
| `F1`, `F2`, … | outlets |
| `H0`/`H1`, `C0`/`C1` | hot and cold side of a heat exchanger |
| `A0`/`A1` | air side of an electrochemical cell |

A port whose name ends in `0` is an inlet, every other port is an outlet. A
unit that deviates (the `Mixer` has two inlets `F0`, `F1`) declares
`_inlet_ports = ('F0', 'F1')`. `unit.inlets`, `unit.outlets`,
`unit.mass_balance()` and the default `MBcheck()` rely on this.

Flange names are `'<unit name>.<port name>'`, unit names inside a system are
`'<system name>.<attribute name>'`.

## Structure of a unit model

```python
from celeste.CSL import ModelBase, interfaces
from celeste import solvers

class MyUnit(ModelBase):
    """One paragraph on what the unit does, then Parameters / Ports / Properties."""

    #: default state of a stand-alone unit, per port
    _default_state = {'F0': dict(n=1.0, T=298.15, P=1.0), 'F1': dict(P=3.0)}

    def __init__(self, name, Medium, eff=1.0, **legacy):
        super().__init__(locals())                    # every design argument becomes an attribute
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self._init_state(legacy)                      # defaults, plus the deprecated F0_n=... keywords

    def Eqs(self):
        """Explicit relations, then the residual vector of the implicit ones."""
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        return np.array([...])

    def f_sol(self, x):
        """Write the unknowns into the model and return the residuals."""
        self.F1.T = x[0]
        return self.Eqs()

    def run_sol(self):
        return solvers.solve(self.f_sol, [self.F0.T], bounds=([1.0], [1e4]), name=self.name)

    @property
    def Q(self): ...          # derived quantities are properties, not attributes set by a method
    @property
    def Exloss(self): ...
    @property
    def Sgen(self): ...
```

### Design inputs and operating state

Constructor arguments are **design** quantities: the medium, efficiencies,
volumes, areas, split fractions, the model choice. The **operating state** lives
on the flanges and is set there, both for a stand-alone unit and inside a
system:

```python
comp = components.CompIsent('comp', media.Air(), eff=0.8)   # design
comp.F0.set(n=10.0, T=300.0, P=1.0)                        # feed
comp.F1.set(P=5.0)                                         # operating spec: discharge pressure
comp.run_sol()
```

Outlet values that a unit does not compute are its specification: the outlet
temperature of a `Heater`, the outlet pressure of a compressor. Inlet values are
overwritten by the upstream unit once the unit is connected. `_default_state`
gives a stand-alone unit a sensible starting point.

The former state keywords (`F0_n=`, `F0_Xi=`, `F1_T=`, `F1_P=`, ...) are still
accepted and land on the same flanges, with a `CelesteDeprecationWarning`.
`SourceM` keeps its `F1_*` arguments: they define the source. `SOC`, `AEL`, `MR`
and the GPU and FT units have not been migrated yet.

| Member | Purpose |
|---|---|
| `Eqs()` | sets explicitly known outlet values, returns residuals of the remaining equations |
| `f_sol(x)` | maps the vector of unknowns onto the model, returns `Eqs()` |
| `run_sol()` | solves the unit; returns a result with `.success` for iterative units |
| `Q`, `W` | heat and work added |
| `Exth` | exergy of the heat duty |
| `Exloss`, `Sgen`, `Exeff` | exergy destroyed, entropy generated, exergy efficiency |
| `MBcheck()` | prints the relative mass balance error, returns in − out |

Units that are evaluated explicitly need no `f_sol`.

### Solving

Use `celeste.solvers.solve(fun, x0, bounds=..., scale=..., name=...)` instead of
calling SciPy directly:

- give **bounds** whenever the unknowns have physical limits – a root with
  negative mole fractions is then never returned as a solution;
- `scale` is the typical size of the residuals (usually the feed flow);
- the result has `success`, `residual_norm` and `method`; the model is left at
  the returned point;
- a failed solve raises a `ConvergenceWarning` (turn it into an error with
  `warnings.simplefilter('error', celeste.solvers.ConvergenceWarning)`).

`solvers.solve_scalar(fun, lo, hi)` solves one equation on an interval with a
bracketing method.

## Structure of a system model

```python
class MySystem(SysBase):
    def __init__(self, name, ...):
        super().__init__(locals())
        self.unit_a = ...(name + '.unit_a', ...)
        self.unit_b = ...(name + '.unit_b', ...)

    def seq_sol(self, store=False):
        self.connect_seq(self.unit_a.F1, self.unit_b.F0, store=store)
        self.unit_b._sol = self.unit_b.run_sol()
```

- `seq_sol(store=False)` connects and solves the units in flow order; pass
  `store` on to `connect_seq` so that `create_graph()` can draw the flowsheet.
- Store the result of iterative units in `unit._sol`; `get_success()` reads it.
- `run_sol()` defaults to one call of `seq_sol()`. Systems with a recycle
  override it: `return self.Recycle_Tear(self.split.F1, self.mix.F1)`.
- A **box** is a system with its own inlet and outlet flanges instead of
  sources and sinks (`Ncomp`, `SOECSys`, `CondenserBox`). It is used inside
  another system like a unit.

## Media

A NASA medium is a class with two attributes:

```python
class Syngas(media.Medium_NASAbase):
    _default_species = ['H2', 'CO', 'CO2', 'H2O']
    _default_Xi = np.array([0.5, 0.3, 0.1, 0.1])
```

`_liquid = True` selects the liquid water density. Real-fluid media derive from
`media.Medium_PRbase` in the same way. A flange copies the medium it is given,
so one medium object can be passed to many flanges.

## Naming

Existing public names are kept as they are (`run_sol`, `Eqs`, `MBcheck`,
`get_Flanges`, `F0_n`, …). For new code:

- classes in `CamelCase`, functions, methods and arguments in `snake_case`;
- flange arguments as `<port>_<variable>`: `F0_n`, `F0_Xi`, `F1_T`, `F1_P`;
- temperatures of a unit as `T_<what>` (`T_cond`, `T_soc0`);
- no `print` in models – return values, or warn through `warnings`.

## Retiring a name

```python
from celeste._deprecation import deprecated

@deprecated('use new_method()')
def old_method(self):
    return self.new_method()
```

The old name keeps working and warns once per call site. Record it in
`CHANGELOG.md` under *Deprecated*.

## Tests

Every model has at least one test in `tests/` that checks the mass (or element)
balance, the energy balance, the sign of the entropy generation and one
reference value. Run `python -m pytest` before sharing a change.
