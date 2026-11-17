"""
Design specifications: "adjust this input until that result has this value", declared on the system.

    plant.add_spec('product.F0.n', 2.0, vary='feed.F1.n', bounds=(0.1, 10.0))
    plant.add_spec(lambda s: s.cond.F1.T - s.cond.F0.T, -5.0, vary='cooler.F1.T')
    plant.solve_specs()

Targets and varied inputs are dotted paths relative to the system (``rgetattr``/``rsetattr``); a target can also
be a function of the system. All specifications are solved together, each solver iteration being a full
``run_sol`` of the system, so they replace the hand-written ``Optimiser`` methods.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ... import solvers


@dataclass
class Spec:
    """One design specification: ``target`` == ``value`` by adjusting ``vary``."""
    target: object              #: dotted path (str) or callable(system) -> float
    value: float
    vary: str                   #: dotted path of the adjusted input
    bounds: tuple = (-np.inf, np.inf)
    scale: float = 1.0          #: typical magnitude of the target (residuals are divided by it)
    name: str = ''


class SpecMixin:
    """``add_spec`` / ``specs`` / ``clear_specs`` / ``solve_specs``."""

    def add_spec(self, target, value: float, vary: str, bounds: tuple | None = None, scale: float | None = None,
                 name: str | None = None) -> Spec:
        """
        Declare a design specification.

        Parameters:
            target:  dotted path of the result (``'sink1.F0.n'``, ``'reactor.F1.Xi[6]'``) or a function of the system
            value:   value the target must have
            vary:    dotted path of the input that is adjusted (``'source1.F1.n'``, ``'heater.F1.T'``)
            bounds:  (lower, upper) of the adjusted input
            scale:   typical magnitude of the target; default |value| (1 if value is 0)
            name:    label in ``specs``
        """
        self.rgetattr(vary)                                           # fails early on a wrong path
        spec = Spec(target, float(value), vary, tuple(bounds) if bounds is not None else (-np.inf, np.inf),
                    float(scale) if scale is not None else (abs(float(value)) or 1.0),
                    name or (target if isinstance(target, str) else getattr(target, '__name__', 'spec')))
        if '_specs' not in self.__dict__:
            self.__dict__['_specs'] = []
        self._specs.append(spec)
        return spec

    def clear_specs(self) -> None:
        """Remove all design specifications."""
        self.__dict__['_specs'] = []

    def _target(self, spec: Spec) -> float:
        return float(spec.target(self) if callable(spec.target) else self.rgetattr(spec.target))

    @property
    def specs(self) -> pd.DataFrame:
        """Table of the design specifications with the current target and input values."""
        rows = []
        for s in self.__dict__.get('_specs', []):
            try:
                current = self._target(s)
            except Exception:
                current = np.nan
            rows.append({'spec': s.name, 'value': s.value, 'current': current, 'vary': s.vary,
                         'input': float(self.rgetattr(s.vary)), 'lower': s.bounds[0], 'upper': s.bounds[1]})
        return pd.DataFrame(rows, columns=['spec', 'value', 'current', 'vary', 'input', 'lower', 'upper'])

    def solve_specs(self, run=None, tol: float = 1e-8, **solve_kw):
        """
        Solve the system with all design specifications: the inputs ``vary`` are adjusted until every target
        has its value. ``run`` is the solve of the system for given inputs (default: ``run_sol``).
        Returns the solver result (``x`` = the adjusted inputs, ``success``).
        """
        specs = self.__dict__.get('_specs', [])
        if not specs:
            raise ValueError(f"{self.name}: no design specifications (add_spec)")
        run = run or self.run_sol

        def residuals(x):
            for s, xi in zip(specs, x):
                self.rsetattr(s.vary, float(xi))
            run()
            return np.array([(self._target(s) - s.value)/s.scale for s in specs])

        x0 = np.array([float(self.rgetattr(s.vary)) for s in specs])
        bounds = None
        if any(np.isfinite(b).any() for s in specs for b in [s.bounds]):
            bounds = (np.array([s.bounds[0] for s in specs], dtype=float), np.array([s.bounds[1] for s in specs], dtype=float))
        sol = solvers.solve(residuals, x0, bounds=bounds, tol=tol, name=f'{self.name} design specs', **solve_kw)
        self._spec_sol = sol
        return sol
