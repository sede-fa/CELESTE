"""
Equation-oriented solve of a flowsheet, on top of the declared streams (``link``).

    eo = plant.eo_compile()        # 1. compile: unknowns and equations from the links and the units
    eo.initialise()                # 2. initialise: x0 from a first sequential sweep (or from the current state)
    eo.solve()                     # 3. solve: all equations together
    plant.eo_sol()                 # the three steps at once

The sequential-modular solve (``auto_sol``, ``run_sol``) is not touched: this layer only reads the declared streams
and the units, and writes the state it solves back into the flanges.

Unknowns   - the state (n, T, P, Xi) of every stream between two units, and the unknowns of every unit that solves a
             system of equations (``f_sol``, for example the reaction extents of the SOC).
Equations  - one per stream variable, "stream = outlet of the unit it leaves" (the connection equations), and the
             residuals of the units' ``f_sol``. A unit without ``f_sol`` is an explicit function: it computes its
             outlet from its inlet (``run_sol``), so it adds no unknowns and no equations of its own.
Feeds and products of the system are not unknowns: feeds are copied into the units, products out of them.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ... import solvers


@dataclass
class _Stream:
    """A stream between two units: its unknowns are the slice ``idx`` of the unknown vector."""
    outlet: object
    inlet: object
    idx: slice
    name: str


class EOSystem:
    """
    The equations of a flowsheet built from its declared streams (see the module description).

    ``variables`` lists the unknowns, ``residuals(x)`` evaluates all equations (and leaves the flanges in the
    state of ``x``), ``initialise`` fills the starting vector and ``solve`` runs the root finding.
    """

    def __init__(self, system):
        self.system = system
        owners = system._owner_map()
        self.units = list(system.getObjs())
        self.feeds = [(o, i) for o, i in system.streams if owners[id(o)] is system]
        self.products = [(o, i) for o, i in system.streams if owners[id(i)] is system]
        inner = [(o, i) for o, i in system.streams if owners[id(o)] is not system and owners[id(i)] is not system]

        self.streams, names, lo, hi, n = [], [], [], [], 0
        for outlet, inlet in inner:
            ns = len(np.atleast_1d(inlet.Xi))
            label = f'{outlet.name} -> {inlet.name}'
            self.streams.append(_Stream(outlet, inlet, slice(n, n + 3 + ns), label))
            names += [f'{label} {v}' for v in ['n', 'T', 'P']] + [f'{label} Xi[{k}]' for k in range(ns)]
            lo += [0.0, 1.0, 1e-6] + [0.0]*ns
            hi += [np.inf, np.inf, np.inf] + [1.0]*ns
            n += 3 + ns
        self.n_streams = n

        self.solved = []                                   # units with equations: (unit, slice of the unknowns)
        self.black_boxes = []                              # explicit units, evaluated through run_sol
        for unit in self.units:
            if not hasattr(unit, 'run_sol'):
                continue
            k = getattr(unit, 'n_unknowns', None)
            if hasattr(unit, 'f_sol') and k:
                self.solved.append((unit, slice(n, n + k)))
                names += [f'{unit.name} unknown {j}' for j in range(k)]
                lo += [-np.inf]*k
                hi += [np.inf]*k
                n += k
            else:
                self.black_boxes.append(unit)
        self.names, self._lo, self._hi = names, np.array(lo), np.array(hi)
        self.scale = np.ones(n)
        self.x = None
        self.result = None

    # ---- description ------------------------------------------------------------------------------------
    @property
    def n_unknowns(self) -> int:
        return len(self.names)

    @property
    def n_equations(self) -> int:
        """Stream connection equations plus the equations of the units that solve ``f_sol``."""
        return self.n_streams + sum(k.stop - k.start for _, k in self.solved)

    @property
    def variables(self) -> pd.DataFrame:
        """The unknowns with their current value (``None`` before ``initialise``)."""
        x = self.x if self.x is not None else [None]*self.n_unknowns
        return pd.DataFrame({'unknown': self.names, 'value': x})

    def __repr__(self) -> str:
        return (f'EOSystem({self.system.name}: {self.n_unknowns} unknowns, {self.n_equations} equations; '
                f'{len(self.streams)} streams, {len(self.solved)} units with equations, '
                f'{len(self.black_boxes)} explicit units)')

    # ---- equations --------------------------------------------------------------------------------------
    @staticmethod
    def _state(flange) -> np.ndarray:
        return np.concatenate([[flange.n, flange.T, flange.P], np.atleast_1d(flange.Xi)])

    @staticmethod
    def _set_state(flange, v) -> None:
        flange.n, flange.Xi, flange.T, flange.P = float(v[0]), np.asarray(v[3:], dtype=float), float(v[1]), float(v[2])

    def residuals(self, x) -> np.ndarray:
        """
        All equations at the unknowns ``x`` (physical units): the streams are written into the inlets of the units,
        every unit is evaluated once, then

        * stream equations: stream state - outlet state of the unit it comes from,
        * unit equations: ``f_sol`` of the units that solve a system.

        Streams and unit outlets are scaled by the starting values, so all entries are of order one.
        """
        sys = self.system
        x = np.asarray(x, dtype=float)
        for outlet, inlet in self.feeds:
            sys.connect_seq(outlet, inlet)
        for s in self.streams:
            self._set_state(s.inlet, x[s.idx])
        out = []
        for unit in self.black_boxes:
            unit.run_sol()
        for unit, idx in self.solved:
            out.append(np.atleast_1d(unit.f_sol(x[idx])))
        stream_res = [(x[s.idx] - self._state(s.outlet))/self.scale[s.idx] for s in self.streams]
        for outlet, inlet in self.products:
            sys.connect_seq(outlet, inlet)
        return np.concatenate(stream_res + out) if (stream_res or out) else np.zeros(0)

    # ---- initialisation ---------------------------------------------------------------------------------
    def initialise(self, method: str = 'sweep', **sweep_kw) -> np.ndarray:
        """
        Build the starting vector.

        ``'sweep'``    one run of the sequential-modular solve (``auto_sol``; ``eps``, ``max_iter`` are passed on,
                       a loose ``eps`` is enough): streams from the converged flanges, unit unknowns from the
                       solutions of the units.
        ``'current'``  the present state of the flanges (the solution of an earlier solve, for example while a
                       design specification adjusts an input); units that solve a system are run once.
        """
        sys = self.system
        if method == 'sweep':
            sys.auto_sol(**sweep_kw)
        elif method != 'current':
            raise ValueError("method must be 'sweep' or 'current'")
        for outlet, inlet in self.feeds:
            sys.connect_seq(outlet, inlet)
        x = np.zeros(self.n_unknowns)
        for s in self.streams:
            x[s.idx] = self._state(s.outlet)
        for unit, idx in self.solved:
            sol = None
            if method == 'current':
                sol = getattr(unit, '_sol', None)
            if sol is None or not hasattr(sol, 'x'):
                for s in self.streams:                     # inlets from the starting streams, then the unit alone
                    if any(s.inlet is f for f in unit.get_Flanges()):
                        self._set_state(s.inlet, x[s.idx])
                sol = unit.run_sol()
                unit._sol = sol
            x[idx] = sol.x
        self.x = x
        self.scale = np.maximum(np.abs(x), 1.0)          # T, P, flows and extents of order one or larger
        return x

    # ---- solve ------------------------------------------------------------------------------------------
    def solve(self, tol: float = 1e-9, **kw):
        """
        Solve all equations together, from the vector of ``initialise`` (called with the sweep if it was not).
        The solution is left in the flanges of the system; the result carries ``x``, ``success`` and
        ``residual_norm``. ``x`` is also kept in ``variables``.
        """
        if self.x is None:
            self.initialise()
        y0 = self.x/self.scale
        bounds = (self._lo/self.scale, self._hi/self.scale)
        res = solvers.solve(lambda y: self.residuals(y*self.scale), y0, bounds=bounds, tol=tol,
                            name=f'{self.system.name} equation-oriented solve', **kw)
        res.x = res.x*self.scale
        self.x, self.result = res.x, res
        return res


class EquationOrientedMixin:
    """``eo_compile`` and ``eo_sol``: the equation-oriented solve of the declared streams."""

    def eo_compile(self) -> EOSystem:
        """Build the equation system from the declared streams and the units (see ``EOSystem``)."""
        if not self.streams:
            raise ValueError(f"{self.name}: no streams declared (link) to build equations from")
        eo = EOSystem(self)
        self.__dict__['_eo'] = eo
        return eo

    def eo_sol(self, init: str = 'sweep', tol: float = 1e-9, recompile: bool = False, sweep_kw: dict | None = None, **kw):
        """
        Equation-oriented solve: compile (once), initialise (``init`` = ``'sweep'`` or ``'current'``) and solve.
        Returns the solver result.
        """
        eo = self.__dict__.get('_eo')
        if eo is None or recompile:
            eo = self.eo_compile()
        eo.initialise(init, **(sweep_kw or {}))
        return eo.solve(tol=tol, **kw)
