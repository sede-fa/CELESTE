# Recycles and heat-recovery loops

A sequential-modular flowsheet with a loop is solved by **tearing** one stream:
guess it, run the units once in flow order, compare the computed stream with
the guess and update the guess until the two agree. `SysBase.Recycle_Tear`
does this.

```python
class Loop(SysBase):
    def seq_sol(self, store=False):
        self.connect_seq(self.feed.F1, self.mix.F0, store=store)
        self.mix._sol = self.mix.run_sol()            # mix.F1 holds the guessed recycle
        self.connect_seq(self.mix.F2, self.reactor.F0, store=store)
        ...
        self.split.run_sol()                          # split.F1 is the computed recycle

    def run_sol(self):
        return self.Recycle_Tear(self.split.F1, self.mix.F1)   # (computed, guessed)
```

`Recycle_Tear(F_source, F_sink, ...)` returns `(passes, error)` and warns with a
`ConvergenceWarning` if `max_iter` passes are not enough.

## What is converged

The tear variables are the **component molar flows and the temperature** of
`F_sink`; its pressure is copied from `F_source`. Converging only the total flow
is not enough: in a loop with a small purge the composition of the recycle
(inerts building up) moves much more slowly than its total flow, so a solver
that watches the total flow stops long before the composition has settled. The
error is the largest relative change of any tear variable between guess and
result.

## Methods

| `method=` | Update | Use |
|---|---|---|
| `'anderson'` (default) | combines the last `memory` passes, accounts for the coupling between variables | most loops; fewest passes |
| `'aitken'` (alias `'steffensen'`) | Δ² extrapolation every second pass (vector form of Irons and Tuck) | smooth loops dominated by one slow mode |
| `'wegstein'` | per-variable secant factor q within `q_bounds`, or a fixed `q=` | weakly coupled loops |
| `'direct'` | the result becomes the next guess | robust but slow when the loop gain is close to 1 |

Passes needed for the methanol loop (`ProcessBoxes.MethanolLoop`, tolerance 1e-9):
Anderson 21, Aitken 94, Wegstein 208; direct substitution has not converged
after 500 passes.

`n_direct=k` runs k direct passes before the acceleration starts, which helps
when the first guess is far off.

The former methods `Recycle_FPI`, `Recycle_Wegstein_q`, `Recycle_Wegstein_Generic`,
`Recycle_Wegstein_Generic2`, `Recycle_Steffensen`, `Recycle_Aitken` and
`Recycle_Hybrid_FPI_Aitken` still work: they call `Recycle_Tear` with the
matching `method` and emit a `CelesteDeprecationWarning`.

## Choosing the tear stream

- Tear a stream that every pass computes completely (flow, composition,
  temperature), typically the recycle entering a mixer.
- Give the torn flange a sensible first guess before calling `Recycle_Tear`
  (its current state is used). A zero recycle is a valid start for material
  recycles.
- One tear per loop. Nested loops need the inner loop converged inside the
  `flow_method` of the outer one.

## Heat-recovery loops

A feed–effluent heat exchanger (the reactor product heats the reactor feed) is a
loop as well: the hot inlet of the exchanger depends on the reactor, whose inlet
depends on the exchanger. Tear the hot inlet:

```python
def seq_sol(self, store=False):
    self.connect_seq(self.feed.F1, self.hx.C0, store=store)
    self.hx.run_sol()                                   # hx.H0 holds the guessed reactor effluent
    self.connect_seq(self.hx.C1, self.reactor.F0, store=store)
    self.reactor.run_sol()
    self.connect_seq(self.hx.H1, self.product.F0, store=store)

def run_sol(self):
    self.hx.H0.set(n=..., Xi=..., T=800.0, P=...)       # first guess of the effluent
    return self.Recycle_Tear(self.reactor.F1, self.hx.H0)
```

`tests/test_boxes.py::FEHELoop` is a complete example (adiabatic ammonia bed,
converged in under 20 passes from a hot or a cold guess).

Cautions:

- **Multiple steady states.** An exothermic reactor with heat feedback can have
  an ignited and an extinguished steady state, and an unstable one between
  them. Which one the iteration reaches depends on the first guess and on the
  reactor model. `HB` and `MeOH` return the ignited state of the reactor
  itself; start the loop from a hot guess when the ignited loop is wanted, and
  check the result from a second guess.
- **Temperature cross-over.** If the exchanger is too large, the reactor
  inlet approaches the outlet temperature and an equilibrium-limited reactor
  converts little; this is a correct steady state, not a solver failure.
- The tear converges the temperature, so the error of `Recycle_Tear` also
  reflects how accurately the units inside the loop are solved; ask for a
  tolerance looser than the units' own.
