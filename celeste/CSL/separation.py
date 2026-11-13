'''
Separation models

1) Flash: vapour-liquid flash drum (isothermal, adiabatic or with a given heat duty)

The phase equilibrium is computed with the Peng-Robinson equation of state and
a stability-tested flash (celeste.jx.flash). In contrast to the black-box
separators in ``components`` (GPU_2S, ...), which split a stream by prescribed
fractions, the split of a Flash follows from thermodynamics.
'''
import numpy as np

from . import ModelBase, interfaces, media
from .. import solvers
from ..jx import nasa as _nasa
from .thermo import _PRModel


def _liquid_medium(Medium):
    """NASA medium for the liquid outlet: species with liquid-phase data switch to it ('H2O' -> 'H2O(L)')."""
    from .exergy import exergy_table
    known = set(_nasa.species_names())
    with_exergy = set(exergy_table().index)
    species = []
    for s in Medium.species:
        liquid = f'{s}(L)'
        species.append(liquid if (liquid in known and liquid in with_exergy) else str(s))
    return media.Medium_NASAbase(f'{Medium.name}_liquid', species=species, Xi=np.array(Medium._reference_X, dtype=float))


class Flash(ModelBase):
    """
    Vapour-liquid flash drum: one feed (F0) is split into a vapour (F1) and a
    liquid (F2) stream in thermodynamic equilibrium.

    Parameters:
        name:    objects name
        Medium:  medium of the feed and of the vapour outlet. Either
                 * a NASA medium (media.Fuel_CH4(), media.Fuel_MeOH(), ...): the phase split comes from the
                   Peng-Robinson equation, the stream properties from the media, so the unit drops into
                   existing flowsheets (e.g. after a cooler, instead of GPU_2S), or
                 * a Peng-Robinson medium (media.Medium_PRbase): fully consistent real-fluid properties.
        MediumL: medium of the liquid outlet. Defaults to the liquid counterpart of Medium
                 ('H2O' -> 'H2O(L)', 'CH3OH' -> 'CH3OH(L)'; dissolved gases keep their species).
        T:       drum temperature in [K]. None: taken from the feed (plain phase separator) unless Q is given
        P:       drum pressure in [bar]. None: feed pressure minus dP
        Q:       heat added in [kJ/s]. Give Q (e.g. 0.0 for an adiabatic flash) instead of T to let the
                 drum temperature follow from the energy balance
        dP:      pressure drop in [bar], used when P is None. Defaults to 0
        kij:     binary interaction parameters for the equation of state (see celeste.jx.eos.pr_params)
        T_bounds: search interval for the drum temperature when Q is specified
        F0_n, F0_Xi, F0_T, F0_P: feed state for stand-alone use (overwritten when connected in a system)

    Ports:
        F0: feed,  F1: vapour outlet,  F2: liquid outlet

    Properties:
        beta:   vapour fraction F1.n/F0.n [-]
        K:      equilibrium ratios y_i/x_i
        phase:  'liquid', 'vapour' or 'two-phase'
        Q:      heat added [kJ/s];  Exth, Exloss, Sgen as for the other units

    Notes:
        * The feed is tested for phase stability first; a single-phase feed leaves completely through
          F1 (vapour) or F2 (liquid) and the other outlet carries no flow.
        * Two phases at most are considered (no second liquid phase).
        * The first call for a new set of species compiles the property functions (a few seconds).

    Examples:
        1) knock-out drum behind a syngas cooler: water leaves as liquid
        ko = Flash('ko', Medium=media.Fuel_CH4(), F0_n=10., F0_Xi=np.array([0.6, 0.05, 0.1, 0.05, 0.2, 1e-8]),
                   F0_T=303.15, F0_P=20.)
        ko.run_sol()
        ko.beta, ko.F2.n, ko.F1.Xi

        2) adiabatic flash of a methanol / water mixture across a valve
        fl = Flash('fl', Medium=media.Fuel_MeOH(), P=1.2, Q=0.0, ...)
    """

    def __init__(self, name:str, Medium:media.Medium_Base, MediumL:media.Medium_Base|None=None,
                 T:float|None=None, P:float|None=None, Q:float|None=None, dP:float=0.0, kij=None,
                 T_bounds:tuple=(150.0, 1500.0), **legacy):
        args = dict(locals())
        for key in ('T', 'P', 'Q'):
            args.pop(key)
        super().__init__(args)
        if T is not None and Q is not None:
            raise ValueError(f"{name}: give either the drum temperature T or the heat duty Q, not both")
        self.T_spec, self.P_spec, self.Q_spec = T, P, Q

        self._real_fluid = isinstance(Medium, media.Medium_PRbase)
        if self._real_fluid:
            MediumV = Medium.with_phase('vapour')
            MediumL = Medium.with_phase('liquid') if MediumL is None else MediumL
            kij = Medium._kij if kij is None else kij
        else:
            MediumV = Medium
            MediumL = _liquid_medium(Medium) if MediumL is None else MediumL
        if len(MediumL.species) != len(Medium.species):
            raise ValueError(f"{name}: MediumL must list the same components as Medium, in the same order")

        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=MediumV)
        self.F2 = interfaces.Flange(name + '.F2', Medium=MediumL)
        self._init_state(legacy)
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        self.F1.T = self.F0.T
        self.F1.P = self.F0.P
        self.F2.n = 0.0
        self.F2.Xi = self.F0.Xi
        self.F2.T = self.F0.T
        self.F2.P = self.F0.P

        self._eos = _PRModel.get(Medium.species, kij)
        self.beta = 1.0
        self.K = np.ones(len(Medium.species))
        self.phase = 'vapour'
        self.converged = True

    # ------------------------------------------------------------------
    def _split(self, T:float, P:float) -> None:
        """Flash the feed at (T, P) and write the result to the outlet flanges."""
        z = np.maximum(np.asarray(self.F0.Xi, dtype=float), 0.0)
        r = self._eos.flash(float(T), float(P), z)
        beta = float(r.beta)
        self.beta = beta
        self.K = np.asarray(r.K)
        self.phase = ('liquid', 'vapour', 'two-phase')[int(r.phase)]
        self.converged = bool(r.converged)

        # feed fractions that do not add up to exactly one still give closed component balances
        n_feed = self.F0.n*float(z.sum())
        self.F1.n = beta*n_feed
        self.F2.n = (1.0 - beta)*n_feed
        self.F1.Medium.PTX = P, T, np.asarray(r.y)
        self.F2.Medium.PTX = P, T, np.asarray(r.x)

    def Eqs(self):
        """Energy balance residual in [kJ/s] for the current drum temperature (zero unless Q is specified)."""
        if self.Q_spec is None:
            return np.zeros(1)
        return np.array([self.Q - self.Q_spec])

    @property
    def P(self):
        """Drum pressure [bar]"""
        return self.F0.P - self.dP if self.P_spec is None else self.P_spec

    @property
    def T(self):
        """Drum temperature [K]"""
        return self.F1.T

    @property
    def Q(self):
        """Heat added to the drum [kJ/s]"""
        return self.F1.n*self.F1.h + self.F2.n*self.F2.h - self.F0.n*self.F0.h

    @property
    def Exth(self):
        return self.Q*(1 - 298.15/self.F1.T)

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex + self.Exth

    @property
    def Sgen(self):
        return self.F1.n*self.F1.s + self.F2.n*self.F2.s - self.F0.n*self.F0.s - self.Q/self.F1.T

    @property
    def recovery(self):
        """Fraction of every component of the feed that leaves with the liquid (F2)"""
        feed = self.F0.n*np.asarray(self.F0.Xi, dtype=float)
        liquid = self.F2.n*np.asarray(self.F2.Xi, dtype=float)
        return np.divide(liquid, feed, out=np.zeros_like(feed), where=feed > 0)

    def f_sol(self, x):
        self._split(x[0], self.P)
        return self.Eqs()

    def run_sol(self):
        P = self.P
        if self.Q_spec is None:
            T = self.F0.T if self.T_spec is None else self.T_spec
            self._split(T, P)
            sol = solvers.trivial_solution([T])
            sol.success = self.converged
        else:
            # the enthalpy of the products rises monotonically with temperature: bracketed search
            scale = max(abs(self.F0.n), 1e-12)
            sol = solvers.solve_scalar(lambda T: float(self.f_sol([T])[0])/scale, self.T_bounds[0], self.T_bounds[1],
                                       x0=self.F0.T, tol=1e-8, name=self.name)
            sol.success = bool(sol.success and self.converged)
        if not self.converged:
            import warnings
            warnings.warn(f'{self.name}: phase split not converged', solvers.ConvergenceWarning, stacklevel=2)
        return sol
