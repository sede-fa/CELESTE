"""
Reactors models

1) EquilBase:     Former base of the isothermal reactors, kept for user models (no CSL model uses it any more)
2) MR:            Methanation / steam reforming + water-gas shift at equilibrium (ln-flow unknowns, ExtentReactor energy options)
3) ExtentReactor: Base for reactors solved for their reaction extents (isothermal, adiabatic or given Q)
4) HB:            Haber-Bosch, kinetic (HB_Adiabatic: former adiabatic class, now HB(..., Q=0)); HB_1D: plug-flow bed
5) MeOH:         Methanol synthesis: equilibrium, yield or kinetic (Van den Bussche & Froment); isothermal,
                  adiabatic or given Q. Former classes MeOH_Equil, MeOH_Yield, MeOH_Kinetic, MeOH_WGS are wrappers
9) Reformer:      Kinetic propane / methane steam reformer (CSTR), Reformer_1D: series of them
"""

import numpy as np
from .._types import Array
from numpy.typing import NDArray as npArray
import scipy.optimize as spo

from . import ModelBase, SysBase, interfaces, media
from . import equilibrium as eq
from .. import solvers
from .._deprecation import deprecated, CelesteDeprecationWarning


class EquilBase(ModelBase):
    """
    Base for isothermal, isobaric reactors with one inlet (F0) and one outlet (F1).

    Setting ``T`` or ``P`` on the reactor sets both flanges. Sub-classes provide
    ``Eqs()`` (residuals; also stores the heat duty ``Q`` in kJ/s) and ``f_sol(x)``
    which maps the unknown vector ``x = [F1.n, *F1.Xi]`` onto the model.

    Properties:
        Exth:   exergy of the heat duty at the reactor temperature [kJ/s]
        Exloss: exergy destroyed [kJ/s]
        Exeff:  exergy efficiency [-]
        Sgen:   entropy generated [kJ/K/s]
    """

    def __init__(self, name:str, Medium:media.Medium_Base, 
                 T:float=298.15, P:float=1.0, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None, #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None,
                 ):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P)
        self._P = P
        self._T = T

    @property
    def P(self) -> float:
        return self._P

    @P.setter
    def P(self, P_: float) -> None:
        self.__dict__['P'] = P_
        self._P = P_
        self.F0.P = P_
        self.F1.P = P_   

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, T_: float) -> None:
        self.__dict__['T'] = T_
        self._T = T_
        self.F0.T = T_
        self.F1.T = T_   

    def Z(self):
        return ((1000/self.F1.T)-1)

    @property
    def Exth(self) -> float:
        return self.Q*(1 - (298.15/self.F1.T))

    @property
    def Exloss(self) -> float:
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth

    @property
    def Exeff(self) -> float:
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.maximum(0, self.Exth))

    @property
    def Sgen(self) -> float:
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n - self.Q/self.F0.T

    @deprecated('Exth, Exloss and Exeff are properties now and need no update call', removal=None)
    def ExBs(self) -> None:
        return None

    def _solve_composition(self, f_sol, x0, n_scale=None):
        """Solve for x = [F1.n, *F1.Xi] with physical bounds (n > 0, 0 <= Xi <= 1)."""
        x0 = np.asarray(x0, dtype=float)
        scale = self.F0.n if n_scale is None else n_scale
        lower = np.concatenate(([1e-12*scale], np.zeros(x0.size - 1)))
        upper = np.concatenate(([np.inf], np.ones(x0.size - 1)))
        return solvers.solve(f_sol, x0, bounds=(lower, upper), scale=scale, name=self.name)

class ExtentReactor(ModelBase):
    """
    Base of reactors solved for their reaction extents (one inlet F0, one outlet F1).

    The outlet follows from the stoichiometry, ``n_out = n_in + nu.T @ xi``, so the
    element balances hold by construction. The unknowns are the extents (one per
    reaction) and, when a heat duty is specified, the outlet temperature.

    Energy specification (exactly one of them):
        T:  isothermal at T [K]
        Q:  heat added [kJ/s]; Q=0 is adiabatic
        neither: isothermal at the inlet temperature

    The outlet pressure is the inlet pressure minus ``dP``. The inlet state is set on F0.

    Sub-classes define ``reactions`` ({name: {species: coefficient}}) and
    ``rate_residuals(xi, T, P, x)`` - one residual per reaction, e.g.
    ``xi - r(T, P, x)*Vol`` (kinetic) or ``ln(Q/K)`` (equilibrium).
    """
    reactions: dict = {}

    def __init__(self, name: str, Medium: media.Medium_Base, T=None, Q=None, dP: float = 0.0):
        if T is not None and Q is not None:
            raise ValueError(f"{name}: give either T (isothermal) or Q (heat duty), not both")
        ModelBase.__init__(self, dict(name=name, Medium=Medium, dP=dP))
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self.T_spec, self.Q_spec = T, Q
        species = [str(sp) for sp in Medium.species]
        self._nu = np.zeros((len(self.reactions), len(species)))
        for r, stoich in enumerate(self.reactions.values()):
            for sp, v in stoich.items():
                self._nu[r, species.index(sp)] = v
        self._xi = np.zeros(len(self.reactions))

    # ---- energy specification ---------------------------------------------------------------------
    @property
    def isothermal(self) -> bool:
        return self.Q_spec is None

    @property
    def T(self) -> float:
        '''Reactor temperature: the specified T, or the outlet temperature.'''
        return self.T_spec if self.T_spec is not None else self.F1.T

    @T.setter
    def T(self, value: float) -> None:
        '''Setting T makes the reactor isothermal at T.'''
        self.T_spec, self.Q_spec = value, None
        self._select_mode()

    @property
    def Q(self) -> float:
        '''Heat added to the reactor [kJ/s].'''
        return self.F1.n*self.F1.h - self.F0.n*self.F0.h

    @Q.setter
    def Q(self, value: float) -> None:
        '''Setting Q (0 for adiabatic) makes the outlet temperature follow from the energy balance.'''
        self.Q_spec, self.T_spec = value, None
        self._select_mode()

    def _select_mode(self):
        '''Hook run when T or Q is set (models that pick their equations by energy specification override it).'''

    @property
    def xi(self):
        '''Reaction extents [mol/s], in the order of ``reactions``.'''
        return self._xi.copy()

    # ---- balances -----------------------------------------------------------------------------------
    @property
    def Exth(self) -> float:
        return self.Q*(1 - 298.15/self.F1.T) if not (self.Q_spec == 0) else 0.0

    @property
    def Exloss(self) -> float:
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth

    @property
    def Exeff(self) -> float:
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.maximum(0, self.Exth))

    @property
    def Sgen(self) -> float:
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n - self.Q/self.F1.T

    @deprecated('Exth, Exloss and Exeff are properties now and need no update call', removal=None)
    def ExBs(self) -> None:
        return None

    # ---- solution ----------------------------------------------------------------------------------
    def rate_residuals(self, xi, T, P, x):
        raise NotImplementedError

    def _moles(self, xi):
        return self.F0.n*np.asarray(self.F0.Xi, dtype=float) + xi @ self._nu

    def _extent_bounds(self):
        '''Range of each extent that keeps every mole number non-negative (with the other extents free).'''
        n0 = self.F0.n*np.asarray(self.F0.Xi, dtype=float)
        R = self._nu.shape[0]
        lo, hi = np.empty(R), np.empty(R)
        for r in range(R):
            for sign, out in ((1.0, hi), (-1.0, lo)):
                c = np.zeros(R); c[r] = -sign
                res = spo.linprog(c, A_ub=-self._nu.T, b_ub=n0, bounds=[(None, None)]*R, method='highs')
                out[r] = sign*(-res.fun) if res.status == 0 else sign*np.inf
        return lo, hi

    def _apply(self, xi, T):
        n = self._moles(xi)
        floor = 1e-14*max(self.F0.n, 1e-300)
        n_safe = np.maximum(n, floor)
        self._xi = np.asarray(xi, dtype=float)
        self.F1.P = self.F0.P - self.dP
        self.F1.T = T
        self.F1.n = float(n.sum())
        self.F1.Xi = n_safe/n_safe.sum()
        return n

    def _outlet_T(self, xi):
        '''Outlet temperature for the extents xi: the specification, or the energy balance (monotonic in T).'''
        if self.isothermal:
            return self.T_spec if self.T_spec is not None else self.F0.T
        H_in = self.F0.n*self.F0.h + self.Q_spec

        def balance(T):
            self._apply(xi, T)
            return (self.F1.n*self.F1.h - H_in)/max(self.F0.n, 1e-300)

        res = solvers.solve_scalar(balance, 150.0, 4000.0, x0=self.F1.T, tol=1e-10, name=self.name, warn=False)
        return float(res.x[0])

    def f_sol(self, xi):
        '''Residuals of the rate / equilibrium relations for the extents ``xi`` (T from the energy specification).'''
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        T = self._outlet_T(xi)
        self._apply(xi, T)
        return np.atleast_1d(np.asarray(self.rate_residuals(xi, T, self.F1.P, np.asarray(self.F1.Xi)), dtype=float))

    def Eqs(self):
        return self.f_sol(self._xi)

    def initial_extents(self, lo, hi):
        '''Starting point inside the feasible range (sub-classes may give a better one).'''
        return np.where(np.isfinite(lo) & np.isfinite(hi), lo + 0.5*(hi - lo), 0.0)

    def run_sol(self, n_scan: int = 60):
        '''
        Solve for the extents. With one reaction the whole feasible range is scanned and the
        steady state with the largest extent is returned (for an adiabatic exothermic reactor
        that is the ignited state); with several reactions a bounded Newton-type solve is used.
        '''
        lo, hi = self._extent_bounds()
        span = np.where(np.isfinite(hi - lo), hi - lo, self.F0.n)
        lo_b, hi_b = lo + 1e-10*span, hi - 1e-10*span
        if self._nu.shape[0] == 1 and np.all(np.isfinite([lo[0], hi[0]])):
            g = lambda v: float(self.f_sol([v])[0])
            grid = np.linspace(lo_b[0], hi_b[0], n_scan)
            vals = np.array([g(v) for v in grid])
            roots = [k for k in range(n_scan - 1) if np.isfinite(vals[k]) and np.isfinite(vals[k + 1])
                     and vals[k]*vals[k + 1] <= 0]
            if roots:
                k = roots[-1]
                res = solvers.solve_scalar(g, grid[k], grid[k + 1], tol=1e-8, name=self.name)
                self.f_sol(res.x)
                return self._check_energy(res)
        starts = [np.clip(self.initial_extents(lo, hi), lo_b, hi_b)]
        if not self.isothermal:
            # starting points from isothermal solutions at and above the inlet temperature (continuation);
            # of several steady states the one with the largest first extent is returned
            spec = self.Q_spec
            for dT in (0.0, 50.0, 100.0, 200.0):
                self.Q_spec, self.T_spec = None, self.F0.T + dT
                iso = solvers.solve(self.f_sol, starts[0], bounds=(lo_b, hi_b), scale=1.0, name=self.name, warn=False)
                if iso.success:
                    starts.append(np.clip(iso.x, lo_b, hi_b))
            self.Q_spec, self.T_spec = spec, None
        best = None
        for x0 in starts:
            res = solvers.solve(self.f_sol, x0, bounds=(lo_b, hi_b), scale=1.0, name=self.name, warn=False)
            if best is None or (res.success and (not best.success or res.x[0] > best.x[0] + 1e-9*self.F0.n)) \
                    or (not best.success and res.residual_norm < best.residual_norm):
                best = res
            if res.success and self.isothermal:
                break
        self.f_sol(best.x)
        if not best.success:
            import warnings
            warnings.warn(f"{self.name}: not converged (max |residual| = {best.residual_norm:.3e})",
                          solvers.ConvergenceWarning, stacklevel=2)
        return self._check_energy(best)

    def _check_energy(self, res):
        if not self.isothermal:
            err = abs(self.Q - self.Q_spec)
            if err > 1e-6*max(1.0, abs(self.Q_spec)):
                import warnings
                warnings.warn(f"{self.name}: the heat duty Q={self.Q_spec} cannot be met (reached {self.Q:.4g} at the "
                              f"temperature limit)", solvers.ConvergenceWarning, stacklevel=3)
                res.success = False
        return res

    def MBcheck(self) -> float:
        '''Mass in - mass out [g/s] (zero by construction).'''
        return self.F0.m - self.F1.m


class MR(ExtentReactor):
    """
    Methanation / steam reforming reactor with the fuel at water-gas-shift and methane-reforming equilibrium.

    System of equations (``Eqs``), the same as in the SOC without current:
        unknowns:  ln of the outlet flows of H2, CH4, CO2, CO, H2O, N2   (+ T_out when Q is specified)
        equations: C, H, O, N balances, WGS and MSR equilibrium         (+ energy balance)
    Solving for the logarithms keeps every flow positive, also for trace species. The extents ``xi_WGS``
    and ``xi_MSR`` follow from the flows.

    Isothermal (give ``T``, or nothing: inlet temperature), adiabatic (``Q=0``) or with a given heat duty ``Q``;
    ``reactor.T = ...`` / ``reactor.Q = ...`` switch an existing reactor.

    Parameters (design):
        Medium: media.Fuel_CH4() (H2, CH4, CO2, CO, H2O, N2)
        T, Q:   energy specification (see :class:`ExtentReactor`)
        dP:     pressure drop [bar]
        P:      deprecated, set ``reactor.F0.P``

    Inlet state: set ``reactor.F0.n``, ``.Xi``, ``.T``, ``.P``; stand-alone default 500.15 K and 23 bar.

    Properties: Q, xi (WGS, MSR) [mol/s], Exloss, Exeff, Sgen
    """
    reactions = {'WGS': eq.REACTIONS['WGS'], 'MSR': eq.REACTIONS['MSR']}
    _default_state = {'F0': dict(n=1.0, T=500.15, P=23.0)}

    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_CH4(), T:float|None=None, P:float|None=None,
                 Q:float|None=None, dP:float=0.0, **legacy):
        if [str(sp) for sp in Medium.species] != eq.FUEL_SPECIES:
            raise ValueError(f"{name}: Medium must hold {eq.FUEL_SPECIES} in this order (media.Fuel_CH4())")
        legacy.pop('F1_n', None); legacy.pop('F1_Xi', None)
        super().__init__(name, Medium, T=T, Q=Q, dP=dP)
        if P is not None:
            legacy.setdefault('F0_P', P)
        self._init_state(legacy)
        if T is not None and 'F0_T' not in legacy:
            self.F0.T = T                       # stand-alone: feed at the reactor temperature, as before
        self.F1.n, self.F1.Xi, self.F1.T, self.F1.P = self.F0.n, self.F0.Xi, self.F0.T, self.F0.P
        self._select_mode()

    def _select_mode(self):
        if self.isothermal:
            self.Eqs, self.f_sol, self.n_unknowns = self.Eqs_T, self.f_sol_T, 6
        else:
            self.Eqs, self.f_sol, self.n_unknowns = self.Eqs_Q, self.f_sol_Q, 7

    @property
    def xi(self):
        '''Extents of the water-gas shift (CO2 formed) and of methane reforming (CH4 converted) [mol/s].'''
        return np.array([self.F1.n*self.F1.Xi[2] - self.F0.n*self.F0.Xi[2],
                         self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1]])

    def Eqs_T(self):
        '''Isothermal: element balances, WGS and MSR equilibrium.'''
        self.Eq = eq.fuel_residuals(self.F1.n*np.asarray(self.F1.Xi), self.F0.n*np.asarray(self.F0.Xi),
                                    self.F1.T, self.F1.P)
        return self.Eq

    def Eqs_Q(self):
        '''Specified heat duty: element balances, WGS and MSR equilibrium, energy balance.'''
        self.Eq = np.append(self.Eqs_T(), (self.Q - self.Q_spec)/self.F0.n)
        return self.Eq

    def set_outlet(self, n, T):
        self.F1.n, self.F1.Xi, self.F1.T, self.F1.P = n.sum(), n/n.sum(), T, self.F0.P - self.dP

    def f_sol_T(self, x):
        '''Isothermal reactor: x = ln of the outlet flows.'''
        self.set_outlet(np.exp(x), self.T_spec if self.T_spec is not None else self.F0.T)
        return self.Eqs_T()

    def f_sol_Q(self, x):
        '''Specified heat duty: x = (ln of the outlet flows, outlet temperature).'''
        self.set_outlet(np.exp(x[:6]), x[6])
        return self.Eqs_Q()

    def run_sol(self, x0=None, method=None):
        '''
        Solve ``f_sol`` (chosen by T= / Q=), starting from the feed (and the inlet temperature).
        ``method`` may be a scipy-style solver ``method(fun, x0)`` to bypass the default.
        '''
        if x0 is None:
            ln_feed = np.log(np.maximum(self.F0.n*np.asarray(self.F0.Xi, dtype=float), 1e-12*self.F0.n))
            x0 = np.append(ln_feed, self.F0.T)[:self.n_unknowns]
        sol = solvers.solve(self.f_sol, x0, tol=1e-9, name=self.name) if method is None else method(self.f_sol, x0)
        self.f_sol(sol.x)
        return sol

    def K_WGS(self):
        return eq.K_WGS(self.F1.T)

    def K_MSR(self):
        return eq.K_MSR(self.F1.T)


class HB(ExtentReactor):
    """
    Haber-Bosch reactor, 1/2 N2 + 3/2 H2 <-> NH3, lumped (well-mixed) catalyst bed.

    Isothermal (give ``T``, or nothing: inlet temperature), adiabatic (``Q=0``) or with a
    given heat duty ``Q``. For the profile along a bed use :class:`HB_1D`.

    Parameters (design):
        Medium: medium with H2, N2, NH3 in this order (media.HB())
        Vol:    catalyst bed volume [m3]
        AF:     catalyst activity factor [-]
        r_type: True - Nielsen rate (Aspen Plus ammonia model, refitted), False - Temkin-Pyzhev
        T, Q:   energy specification (see :class:`ExtentReactor`)
        dP:     pressure drop [bar]

    Inlet state: set ``reactor.F0.n``, ``.Xi``, ``.T``, ``.P``; stand-alone default
    0.74 H2 / 0.25 N2 / 0.01 NH3 at 673.15 K and 200 bar.

    Properties: Q, xi (mol NH3/s), conversion (of H2), Exloss, Exeff, Sgen
    """
    reactions = {'NH3': {'H2': -1.5, 'N2': -0.5, 'NH3': 1.0}}
    _default_state = {'F0': dict(n=1.0, Xi=[0.74, 0.25, 0.01], T=673.15, P=200.0)}

    def __init__(self, name:str, Medium:media.Medium_Base=media.HB(), Vol:float=1.0, AF:float=1.0,
                 r_type:bool=True, T:float|None=None, Q:float|None=None, dP:float=0.0, **legacy):
        P_legacy = legacy.pop('P', None)
        legacy.pop('F1_n', None); legacy.pop('F1_Xi', None)
        super().__init__(name, Medium, T=T, Q=Q, dP=dP)
        self.Vol, self.AF, self.r_type = Vol, AF, r_type
        if P_legacy is not None:
            legacy.setdefault('F0_P', P_legacy)
        self._init_state(legacy)
        if T is not None and 'F0_T' not in legacy:
            self.F0.T = T                       # stand-alone: feed at the reactor temperature, as before
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        self.F1.T = self.F0.T
        self.F1.P = self.F0.P

    @property
    def r(self):
        '''Rate expression r(T, P, Xi, AF) [mol NH3/m3/s] selected by ``r_type``.'''
        return self.r_Nielsen if self.r_type else self.r_TP

    def rate_residuals(self, xi, T, P, x):
        return (xi - self.r(T, P, x, self.AF)*self.Vol)/self.F0.n

    def initial_extents(self, lo, hi):
        return np.array([lo[0] + 0.3*(hi[0] - lo[0])])

    @property
    def conversion(self) -> float:
        '''Hydrogen conversion [-].'''
        return (self.F0.Xi[0]*self.F0.n - self.F1.Xi[0]*self.F1.n)/(self.F0.Xi[0]*self.F0.n)

    @property
    def tau0(self) -> float:
        return self.F0.P*self.Vol*1e3/0.08314/self.F0.T/self.F0.n

    @property
    def tau1(self) -> float:
        return self.F1.P*self.Vol*1e3/0.08314/self.F1.T/self.F1.n

    @staticmethod
    def r_Nielsen(T, P, Xi, AF):
        """
        Kinetic from Aspen Plus, 2015, "Aspen Plus ammonia model", page 16-17 and from, Nielsen, 1964, "Ammonia
        synthesis rate and mechanism", with the parameters refitted as in A. Nielsen, 1995, "Ammonia catalysis
        and manufacture", page 169. Iron catalyst, activities of H2, N2 and NH3.
        Returns mol NH3 per m3 of catalyst bed per second.
        """
        R = 1.987 # ideal gas const cal/mol/K
        alpha = 0.654
        w = 1.523
        Ru = 0.08206 # atm*L/mol/K
        Patm_bar = 1/1.01325 #atm/bar
        A = np.array([0.1975, 1.3445, 2.393])
        B = np.array([0.02096, 0.05046, 0.03415])
        C = np.array([5.04e2, 4.2e4, 4.7687e6])
        Xi = np.maximum(np.asarray(Xi, dtype=float), 1e-12)
        k_2 = 3.945e10*np.exp(-11171/T/R)
        K_3 = 2.94e-4*np.exp(24052/T/R)
        a = Xi*P*Patm_bar*np.exp((P*Patm_bar/Ru/T)*(B - A/Ru/T - C/(T**3) + ((A**0.5 - np.dot(Xi, A**0.5))**2)/Ru/T))
        K_a = 10**(-2.691122*np.log10(T) - 5.519265e-5*T + 1.848863e-7*T**2 + 2001.6/T + 2.6899)
        return 1e3*AF*k_2*((a[1]*K_a**2 - (a[2]**2)/(a[0]**3))/((1. + K_3*a[2]/(a[0]**w))**(2*alpha)))/3600

    @staticmethod
    def r_TP(T, P, Xi, AF):
        """
        Temkin-Pyzhev rate (Morud, 1998, "Analysis of instability in an industrial ammonia reactor", adjusted
        from Nielsen, 1964). Returns mol NH3 per m3 of reactor per second. The rate diverges for NH3 -> 0;
        the ammonia fraction is floored at 1e-8.
        """
        R = 8.314
        alpha = 0.5
        Xi = np.maximum(np.asarray(Xi, dtype=float), 1e-8)
        k1 = 1.79e4*np.exp(-87090/R/T)
        k2 = 2.57e16*np.exp(-198464/R/T)
        return 2*AF*1e3*(k1*Xi[1]*P*((Xi[0]*P)**3/(Xi[2]*P)**2)**alpha
                         - k2*((Xi[2]*P)**2/(Xi[0]*P)**3)**(1 - alpha))/3600

    for_sol = ExtentReactor.f_sol   # former name, kept for existing code


class HB_Adiabatic(HB):
    """
    Adiabatic Haber-Bosch reactor - former class, now ``HB(..., Q=0)``.
    The former outlet pressure ``F1_P`` becomes the pressure drop ``dP = F0_P - F1_P``.
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.HB(), AF: float = 1.0, Vol:float=1.0, r_type:bool=True,
                 P:float|None=None, F0_P:float=200.0, F1_P:float|None=None, F1_T=None, **legacy):
        from .._deprecation import warn_deprecated
        warn_deprecated("HB_Adiabatic is deprecated: use HB(name, Medium, Vol=..., Q=0)", stacklevel=3)
        legacy = {k: v for k, v in legacy.items() if k not in ('F1_n', 'F1_Xi')}
        legacy['F0_P'] = F0_P
        dP = 0.0 if F1_P is None else F0_P - F1_P
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', CelesteDeprecationWarning)
            super().__init__(name, Medium, Vol=Vol, AF=AF, r_type=r_type, Q=0.0, dP=dP, **legacy)


class HB_1D(ModelBase):
    """
    Haber-Bosch catalyst bed as a plug-flow reactor: one bed, integrated along its volume.

        dxi/dV = r(T, P, x)                                  (mol NH3/s per m3)
        dT/dV  = (q_wall - dH_r(T) r) / (sum n_i cp_i)       (q_wall = UA_V (T_wall - T), kW per m3)

    with the reaction enthalpy and heat capacities of the medium, so the energy balance is
    exact: F1.n*F1.h - F0.n*F0.h = Q (the heat through the wall). ``UA_V=0`` (default) is an
    adiabatic bed. Several beds with intercooling are built in a system: bed -> cooler -> bed.

    Parameters (design):
        Medium, Vol, AF, r_type: as :class:`HB`
        UA_V:   heat transfer coefficient times area per bed volume [kW/K/m3]; 0: adiabatic
        T_wall: coolant temperature [K] (used when UA_V > 0)
        dP:     pressure drop over the bed [bar], linear along the bed
        n_points: number of points stored for the profile

    Properties:
        profile: DataFrame along the bed (V, xi, T, P, mole fractions)
        Q (from the outlet enthalpy), Q_wall (integrated wall heat; equal to Q), conversion, Exloss, Sgen
    """
    _default_state = HB._default_state

    def __init__(self, name:str, Medium:media.Medium_Base=media.HB(), Vol:float=1.0, AF:float=1.0, r_type:bool=True,
                 UA_V:float=0.0, T_wall:float=673.15, dP:float=0.0, n_points:int=50, **legacy):
        super().__init__(dict(name=name, Medium=Medium, Vol=Vol, AF=AF, r_type=r_type, UA_V=UA_V, T_wall=T_wall,
                              dP=dP, n_points=n_points))
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self._init_state(legacy)
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        self.F1.T = self.F0.T
        self.F1.P = self.F0.P
        self._nu = np.array([-1.5, -0.5, 1.0])
        self.profile = None

    @property
    def r(self):
        return HB.r_Nielsen if self.r_type else HB.r_TP

    def _state(self, xi, T, V):
        n = self.F0.n*np.asarray(self.F0.Xi, dtype=float) + xi*self._nu
        P = self.F0.P - self.dP*V/self.Vol
        return n, P

    def _rhs(self, V, y):
        xi, T = y[0], y[1]
        n, P = self._state(xi, T, V)
        n = np.maximum(n, 1e-14*self.F0.n)
        med = self._med
        med.PTX = P, T, n/n.sum()
        rate = self.r(T, P, med.Xi, self.AF)
        dH = float(self._nu @ med.h_i)                          # kJ/mol NH3
        q = self.UA_V*(self.T_wall - T)
        return [rate, (q - dH*rate)/float(n @ med.cp_i), q]

    def run_sol(self):
        import copy
        from scipy.integrate import solve_ivp
        self._med = copy.copy(self.F0.Medium)
        xi_max = self.F0.n*min(self.F0.Xi[0]/1.5, self.F0.Xi[1]/0.5)
        stop = lambda V, y: xi_max*(1 - 1e-9) - y[0]
        stop.terminal = True
        V_eval = np.linspace(0.0, self.Vol, self.n_points)
        sol = solve_ivp(self._rhs, (0.0, self.Vol), [0.0, self.F0.T, 0.0], method='LSODA', t_eval=V_eval,
                        rtol=1e-9, atol=[1e-12*self.F0.n, 1e-9, 1e-12], events=stop)
        self.Q_wall = float(sol.y[2, -1])
        xi_end, T_end = sol.y[0, -1], sol.y[1, -1]
        n, P = self._state(xi_end, T_end, self.Vol)
        self.F1.n = float(n.sum())
        self.F1.Xi = n/n.sum()
        self.F1.T = float(T_end)
        self.F1.P = float(P)
        rows = []
        for V, xi, T in zip(sol.t, sol.y[0], sol.y[1]):
            nV, PV = self._state(xi, T, V)
            rows.append(dict(V=V, xi=xi, T=T, P=PV, **{f'x_{sp}': v for sp, v in zip(self.F0.Medium.species, nV/nV.sum())}))
        import pandas as pd
        self.profile = pd.DataFrame(rows)
        sol.residual_norm = 0.0 if sol.success else np.inf
        return sol

    @property
    def Q(self) -> float:
        '''Heat added through the wall [kJ/s].'''
        return self.F1.n*self.F1.h - self.F0.n*self.F0.h

    @property
    def conversion(self) -> float:
        return (self.F0.Xi[0]*self.F0.n - self.F1.Xi[0]*self.F1.n)/(self.F0.Xi[0]*self.F0.n)

    @property
    def Exth(self) -> float:
        return self.Q*(1 - 298.15/self.T_wall)

    @property
    def Exloss(self) -> float:
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth

    @property
    def Sgen(self) -> float:
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n - self.Q/self.T_wall


class MeOH(ExtentReactor):
    """
    Methanol synthesis reactor with two reactions:

        1) CO2 + 3 H2 <-> CH3OH + H2O      (extent xi[0])
        2) CO2 +   H2 <-> CO    + H2O      (reverse water-gas shift, extent xi[1])

    CO hydrogenation (CO + 2 H2 -> CH3OH) is the sum of both and needs no extra extent.

    Parameters (design):
        Medium: medium with H2, CO2, CO, H2O, CH3OH (media.Fuel_MeOH(); CH4 and N2 are inert)
        model:  'equilibrium' - both reactions at equilibrium (NASA Gibbs energies)
                'yield'       - methanol formed = Yield x (CO + CO2) of the feed, shift at equilibrium
                'kinetic'     - Van den Bussche & Froment (1996) rates over ``M_cat`` kg of catalyst
        Yield:  per-pass methanol yield for model='yield' [-]
        M_cat:  catalyst mass for model='kinetic' [kg]
        T, Q:   energy specification: T isothermal, Q heat added (Q=0 adiabatic), neither: inlet temperature
        dP:     pressure drop [bar]

    Inlet state: set ``reactor.F0.n``, ``.Xi``, ``.T``, ``.P``; stand-alone default
    0.75 H2 / 0.05 CH4 / 0.1 CO2 / 0.1 CO at 473.15 K and 50 bar.

    Properties: Q, xi, conversion (H2, CO2, carbon), Exloss, Exeff, Sgen

    Notes:
        Equilibrium constants: celeste.CSL.equilibrium (NASA Gibbs energies). The kinetic model keeps
        the equilibrium constants it was fitted with (Graaf et al., 1986) inside its rate expressions.
    """
    reactions = {'MeOH': {'CO2': -1.0, 'H2': -3.0, 'CH3OH': 1.0, 'H2O': 1.0},
                 'rWGS': {'CO2': -1.0, 'H2': -1.0, 'CO': 1.0, 'H2O': 1.0}}
    _default_state = {'F0': dict(n=1.0, Xi=[0.75, 0.05, 0.1, 0.1, 1e-8, 1e-8, 1e-8], T=473.15, P=50.0)}
    R = 8.314462618     # J/mol/K (kinetics)
    _MODELS = ('equilibrium', 'yield', 'kinetic')

    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_MeOH(), model:str='equilibrium',
                 Yield:float=0.35, M_cat:float|None=None, T:float|None=None, Q:float|None=None, dP:float=0.0,
                 **legacy):
        if model not in self._MODELS:
            raise ValueError(f"model must be one of {self._MODELS}")
        if model == 'kinetic' and M_cat is None:
            raise ValueError("model='kinetic' needs the catalyst mass M_cat [kg]")
        P_legacy = legacy.pop('P', None)
        legacy.pop('F1_n', None); legacy.pop('F1_Xi', None)
        super().__init__(name, Medium, T=T, Q=Q, dP=dP)
        self.model, self.Yield, self.M_cat = model, Yield, M_cat
        sp = [str(s) for s in Medium.species]
        self._i = {k: sp.index(k) for k in ('H2', 'CO2', 'CO', 'H2O', 'CH3OH')}
        if P_legacy is not None:
            legacy.setdefault('F0_P', P_legacy)
        self._init_state(legacy)
        if T is not None and 'F0_T' not in legacy:
            self.F0.T = T
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        self.F1.T = self.F0.T
        self.F1.P = self.F0.P

    # ---- relations per model ----------------------------------------------------------------------
    def rate_residuals(self, xi, T, P, x):
        i = self._i
        ln = lambda k: np.log(x[i[k]])
        eq_shift = ln('CO') + ln('H2O') - ln('CO2') - ln('H2') - np.log(eq.K_rWGS(T))
        if self.model == 'equilibrium':
            eq_meoh = ln('CH3OH') + ln('H2O') - ln('CO2') - 3*ln('H2') - 2*np.log(P) - np.log(eq.K_MeOH_CO2(T))
            return np.array([eq_meoh, eq_shift])
        if self.model == 'yield':
            n0 = self.F0.n*np.asarray(self.F0.Xi, dtype=float)
            target = self.Yield*(n0[i['CO2']] + n0[i['CO']])
            return np.array([(xi[0] - target)/self.F0.n, eq_shift])
        r_meoh, r_rwgs = self.rates(T, P, x)
        return np.array([xi[0] - r_meoh*self.M_cat, xi[1] - r_rwgs*self.M_cat])/self.F0.n

    def initial_extents(self, lo, hi):
        if self.model == 'yield':
            n0 = self.F0.n*np.asarray(self.F0.Xi, dtype=float)
            x1 = self.Yield*(n0[self._i['CO2']] + n0[self._i['CO']])
            return np.array([x1, 0.0])
        return np.array([0.1*hi[0], 0.0])

    # ---- Van den Bussche & Froment (1996) ---------------------------------------------------------
    _K_a = 3453.38

    def _K_b_at(self, T):
        return 0.499*np.exp(17197/(self.R*T))

    def _K_c_at(self, T):
        return 6.62e-11*np.exp(124119/(self.R*T))

    @property
    def _K_b(self):
        return self._K_b_at(self.T)

    @property
    def _K_c(self):
        return self._K_c_at(self.T)

    def rates(self, T, P, x):
        '''(r_MeOH, r_rWGS) in mol/kg_cat/s (Van den Bussche & Froment, 1996).'''
        i = self._i
        H2, CO2, CO, H2O, MeOH = (max(float(x[i[k]]), 1e-12) for k in ('H2', 'CO2', 'CO', 'H2O', 'CH3OH'))
        K_meoh = 10**((3066/T) - 10.592)
        K_wgs = 10**((2073/T) - 2.029)
        k_meoh = 1.07*np.exp(36696/(self.R*T))
        k_rwgs = 1.22e10*np.exp(-94765/(self.R*T))
        den = 1 + self._K_a*(H2O/H2) + self._K_b_at(T)*(H2*P)**0.5 + self._K_c_at(T)*H2O*P
        r_meoh = k_meoh*H2*CO2*P**2*(1 - (1/K_meoh)*H2O*MeOH/(H2**3*CO2*P**2))/den**3
        r_rwgs = k_rwgs*CO2*P*(1 - K_wgs*H2O*CO/(H2*CO2))/den
        return r_meoh, r_rwgs

    def r_MeOH(self):
        return self.rates(self.F1.T, self.F1.P, self.F1.Xi)[0]

    def r_rWGS(self):
        return self.rates(self.F1.T, self.F1.P, self.F1.Xi)[1]

    # ---- results ------------------------------------------------------------------------------------
    @property
    def conversion(self) -> float:
        i, f0, f1 = self._i, self.F0, self.F1
        c_in = f0.n*(f0.Xi[i['CO2']] + f0.Xi[i['CO']])
        c_out = f1.n*(f1.Xi[i['CO2']] + f1.Xi[i['CO']])
        return {"H2 Conversion": 1 - f1.n*f1.Xi[i['H2']]/(f0.n*f0.Xi[i['H2']]),
                "CO2 Conversion": 1 - f1.n*f1.Xi[i['CO2']]/(f0.n*f0.Xi[i['CO2']]),
                "Carbon Conversion": 1 - c_out/c_in}

    def MBcheck(self) -> float:
        """Print the relative mass balance error and return out - in, g/s."""
        MB = self.F1.m - self.F0.m
        print(f"{(self.F1.m/self.F0.m - 1)*100}% difference")
        return MB


def _former_meoh(model, adiabatic: bool = False):
    def init(self, name:str, Medium:media.Medium_Base=media.Fuel_MeOH(), Yield:float=0.35, M_cat:float|None=None,
             T:float=473.15, P:float=50.0, Q:float=0.0, F0_n:float=1.0, F0_Xi=None, F1_n=None, F1_Xi=None):
        from .._deprecation import warn_deprecated
        new = "MeOH(name, Medium, model='%s'%s)" % (model, ", Q=0" if adiabatic else ", T=...")
        warn_deprecated(f"{type(self).__name__} is deprecated: use {new}", stacklevel=3)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', CelesteDeprecationWarning)
            MeOH.__init__(self, name, Medium, model=model, Yield=Yield, M_cat=M_cat,
                          T=None if adiabatic else T, Q=0.0 if adiabatic else None,
                          F0_n=F0_n, F0_Xi=F0_Xi, F0_T=T, F0_P=P)
    return init


class MeOH_Equil(MeOH):
    """Former isothermal equilibrium methanol reactor: ``MeOH(name, Medium, model='equilibrium', T=...)``."""
    __init__ = _former_meoh('equilibrium')


class MeOH_Yield(MeOH):
    """Former isothermal yield reactor: ``MeOH(name, Medium, model='yield', Yield=..., T=...)``."""
    __init__ = _former_meoh('yield')


class MeOH_Kinetic(MeOH):
    """Former isothermal kinetic reactor: ``MeOH(name, Medium, model='kinetic', M_cat=..., T=...)``."""
    __init__ = _former_meoh('kinetic')


class MeOH_WGS(MeOH):
    """Former adiabatic yield reactor: ``MeOH(name, Medium, model='yield', Yield=..., Q=0)``."""
    __init__ = _former_meoh('yield', adiabatic=True)


class Reformer(ModelBase):
    '''
    Medium = media.media_C3()
    '''
    def __init__(self, name: str, Medium: media.Medium_Base, F0_n: float = 1., F0_Xi: float = 1., F1_n: float = 1., F1_Xi=None, T: float = 500., P: float = 1., V: float = 1., por: float = 0.75, A = [17400., 23950., 5000.], 
                 Ea = [103191., 0.82*231266., 159241.], beta3: int = -1, alpha3: float = 2.8, Twall: float = 1000., UA: float = 1e-3):
        
        super().__init__(locals())
        
        self.F0 = interfaces.Flange(name + '.F0', Medium=self.Medium, n=F0_n, Xi=F0_Xi, T=T, P=P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=self.Medium, n=F1_n, Xi=F1_Xi, T=T, P=P)
        
        self.v_i = np.array([
            [1, 0, 1, -1, -1, 0, 0],
            [3, -1, 0, 1, -1, 0, 0],
            [7, 0, 0, 3, -3, 0, -1],
            #[2, 0, -3, 6, 0, 0, -1]
        ], dtype=float)
        self.Eq = np.zeros(self.Medium.nX+2, dtype=float)
        self.r_i = np.zeros(3, dtype=float) #mol/m3cat/s
        
    def Eqs(self): 
        
        self.r_i[0] = self.A[0]*np.exp(-self.Ea[0]/8.314/self.F1.T)*(self.F1.Xi[3]*self.F1.Xi[4]*self.F1.P**2 - (1/self.K_WGS())*self.F1.Xi[2]*self.F1.Xi[0]*self.F1.P**2)
        self.r_i[1] = self.A[1]*np.exp(-self.Ea[1]/8.314/self.F1.T)*(self.F1.Xi[1]*self.F1.Xi[4]*self.F1.P**2 - (1/self.K_MSR())*self.F1.Xi[3]*self.F1.Xi[0]**3*self.F1.P**4)
        self.r_i[2] = self.A[2]*((self.F1.T/1000)**self.beta3)*np.exp(-self.Ea[2]/8.314/self.F1.T)*(self.F1.Xi[6]*self.F1.P)**self.alpha3
        #self.r_i[3] = self.A[3]*np.exp(-self.Ea[3]/8.314/self.F1.T)*(self.F1.Xi[6]*self.F1.P)
        
        self.Q = self.UA*(self.Twall - self.F1.T)
        
        self.Eq[0] = np.sum(self.F1.Xi) - 1. #self.F0.n*self.F0.mw - self.F1.n*self.F1.mw #
        self.Eq[1] = self.F0.n*self.F0.h - self.F1.n*self.F1.h + self.Q
        self.Eq[2:] = self.F0.n*self.F0.Xi - self.F1.n*self.F1.Xi + np.einsum('i, ij->j', self.r_i, self.v_i)*self.V*(1-self.por)
        
        return self.Eq
    
    def Conv(self):
        X = (self.F1.Xi*self.F1.n - self.F0.Xi*self.F0.n)/(self.F0.Xi*self.F0.n)
        return X
    
    def MBcheck(self) -> float:
        return self.F0.n*self.F0.mw/(self.F1.n*self.F1.mw) 
    
    def K_PSR(self):
        return eq.K_PSR(self.F1.T)
    
    def K_MSR(self):
        return eq.K_MSR(self.F1.T)
    
    def K_WGS(self):
        return eq.K_WGS(self.F1.T)
    
    def Q_WGS(self):
        return self.F1.Xi[2]*self.F1.Xi[0]/self.F1.Xi[3]/self.F1.Xi[4]
    
    def Eq_WGS(self):
        return self.Q_WGS()/self.K_WGS()
    
    def f_sol(self, x: Array) -> Array:
        self.F1.n = x[0]
        self.F1.T = x[1]
        self.F1.Xi = x[2:]
        
        return self.Eqs()
    
    def run_sol(self, f_method=spo.root):
        x0 = np.concatenate(([self.F0.n, self.F0.T], self.F0.Xi))
        
        sol = f_method(self.f_sol, x0, method='krylov', tol=1e-10)
        self.f_sol(sol.x)
        if not sol.success:
            import warnings
            warnings.warn(f'{self.name}: not converged ({sol.message})', solvers.ConvergenceWarning, stacklevel=2)
        return sol

class Reformer_1D(SysBase):
    '''
    Medium = media.media_C3()
    '''
    def __init__(self, name: str, Medium: media.Medium_Base, F0_n: float = 1., F0_Xi: float = 1., T: float = 500., P: float = 1., V: float = 1., N: int = 10, por: float = 0.75, A = [17400., 23950., 5000.], 
                 Ea = [103191., 0.82*231266., 159241.], beta3: int = -1, alpha3: float = 2.8, Twall: float = 1000., UA: float = 1e-3):
        super().__init__(locals())
        self.Refs = np.array([Reformer(f'{name}[{i}]', Medium=Medium, F0_n = F0_n, F0_Xi=F0_Xi, F1_n=F0_n, F1_Xi=F0_Xi, 
                                       T=T, P=P, V=V/N, por=por, A = A, Ea = Ea, Twall=Twall, UA=UA, beta3=beta3, alpha3=alpha3) for i in range(N)], 
                             dtype=object)
        
    def run_sol(self):
        self.Refs[0]._sol= self.Refs[0].run_sol()
        
        for i in range(self.N-1):
            self.connect_seq(self.Refs[i].F1, self.Refs[i+1].F0)
            self.Refs[i+1]._sol= self.Refs[i+1].run_sol()
            
    def getArrAttr(self, a):
        return np.array([self.Refs[i].rgetattr(a) for i in range(self.N)])
    
    @property
    def Xd(self):
        X = self.getArrAttr('F1.Xi')
        X[:,4] = 0
        X = (1/np.sum(X, axis=1)).reshape(self.N, 1)*X
        return X
