"""
Reactors models

1) EquilBase:     Base model for isothermal reactors (ports F0 -> F1, shared exergy / entropy balances)
2) MR:            Methanation / steam reforming + water-gas shift at equilibrium
3) ExtentReactor: Base for reactors solved for their reaction extents (isothermal, adiabatic or given Q)
4) HB:            Haber-Bosch, kinetic (HB_Adiabatic: former adiabatic class, now HB(..., Q=0)); HB_1D: plug-flow bed
5) MeOH_Equil:    Methanol synthesis at equilibrium
6) MeOH_Yield:    Methanol synthesis with a given yield, rWGS at equilibrium
7) MeOH_Kinetic:  Methanol synthesis, kinetic (Van den Bussche & Froment)
8) MeOH_WGS:      Adiabatic methanol synthesis with a given yield, WGS at equilibrium
9) Reformer:      Kinetic propane / methane steam reformer (CSTR), Reformer_1D: series of them
"""

import numpy as np
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
    def P(self):
        return self._P

    @P.setter
    def P(self, P_):
        self.__dict__['P'] = P_
        self._P = P_
        self.F0.P = P_
        self.F1.P = P_   

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, T_):
        self.__dict__['T'] = T_
        self._T = T_
        self.F0.T = T_
        self.F1.T = T_   

    def Z(self):
        return ((1000/self.F1.T)-1)

    @property
    def Exth(self):
        return self.Q*(1 - (298.15/self.F1.T))

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth

    @property
    def Exeff(self):
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.maximum(0, self.Exth))

    @property
    def Sgen(self):
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

class MR(EquilBase):
    """ 
        Medium: media.Fuel_CH4()
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_CH4(), 
                 T:float=500.15, P:float=23.0, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None, #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None
                ):          
        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q) 
        self._N_H = np.array([2,4,0,0,2,0])
        self._N_C = np.array([0,1,1,1,0,0])
        self._N_O = np.array([0,0,2,1,1,0])
        self._N_N = np.array([0,0,0,0,0,2])
        
    def Eqs(self):
        
        self.Eq:npArray[np.float64] = np.array([
            # MB
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)), 
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)), 
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)), 
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)), 
            np.sum(self.F1.Xi) - 1, 
            # Equil
            self.F1.Xi[3]*self.F1.Xi[4]*self.K_WGS() - (self.F1.Xi[2]*self.F1.Xi[0]), 
            self.F1.Xi[1]*self.F1.Xi[4]*self.K_MSR() - (self.F1.Xi[3]*self.F1.Xi[0]**3*self.F1.P**2), 
        ])
        self.Q = (self.F1.n*self.F1.h - self.F0.n*self.F0.h )
        
        self.F0_n = self.F0.n
        self.F0_Xi = self.F0.Xi
        self.F1_n = self.F1.n
        self.F1_Xi = self.F1.Xi
        
        return self.Eq
    
    def K_WGS(self):
        return eq.K_WGS(self.F1.T)

    def K_MSR(self):
        return eq.K_MSR(self.F1.T)

    def f_sol(self, x):  
        self.F1.n = x[0]
        self.F1.Xi = x[1:7] 
        return self.Eqs()
        
    def run_sol(self, method=None):
        """Solve for the outlet. ``method`` may be a scipy-style solver ``method(fun, x0)`` to bypass the default."""
        x0 = np.concatenate(([self.F0.n], self.F0.Xi)) 
        if method is not None:
            sol = method(self.f_sol, x0)
            self.f_sol(sol.x)
            return sol
        return self._solve_composition(self.f_sol, x0)

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

    def __init__(self, name, Medium, T=None, Q=None, dP=0.0):
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
    def T(self):
        '''Reactor temperature: the specified T, or the outlet temperature.'''
        return self.T_spec if self.T_spec is not None else self.F1.T

    @T.setter
    def T(self, value):
        '''Setting T makes the reactor isothermal at T.'''
        self.T_spec, self.Q_spec = value, None

    @property
    def Q(self):
        '''Heat added to the reactor [kJ/s].'''
        return self.F1.n*self.F1.h - self.F0.n*self.F0.h

    @Q.setter
    def Q(self, value):
        '''Setting Q (0 for adiabatic) makes the outlet temperature follow from the energy balance.'''
        self.Q_spec, self.T_spec = value, None

    @property
    def xi(self):
        '''Reaction extents [mol/s], in the order of ``reactions``.'''
        return self._xi.copy()

    # ---- balances -----------------------------------------------------------------------------------
    @property
    def Exth(self):
        return self.Q*(1 - 298.15/self.F1.T) if not (self.Q_spec == 0) else 0.0

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth

    @property
    def Exeff(self):
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.maximum(0, self.Exth))

    @property
    def Sgen(self):
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
        x0 = np.clip(self.initial_extents(lo, hi), lo_b, hi_b)
        return self._check_energy(solvers.solve(self.f_sol, x0, bounds=(lo_b, hi_b), scale=1.0, name=self.name))

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

    Inlet state: ``reactor.F0.set(n=..., Xi=..., T=..., P=...)``; stand-alone default
    0.74 H2 / 0.25 N2 / 0.01 NH3 at 673.15 K and 200 bar.

    Properties: Q, xi (mol NH3/s), conversion (of H2), Exloss, Exeff, Sgen
    """
    reactions = {'NH3': {'H2': -1.5, 'N2': -0.5, 'NH3': 1.0}}
    _inlet_state = {'F0': dict(n=1.0, Xi=[0.74, 0.25, 0.01], T=673.15, P=200.0)}

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
        self.F1.set(n=self.F0.n, Xi=self.F0.Xi, T=self.F0.T, P=self.F0.P)

    @property
    def r(self):
        '''Rate expression r(T, P, Xi, AF) [mol NH3/m3/s] selected by ``r_type``.'''
        return self.r_Nielsen if self.r_type else self.r_TP

    def rate_residuals(self, xi, T, P, x):
        return (xi - self.r(T, P, x, self.AF)*self.Vol)/self.F0.n

    def initial_extents(self, lo, hi):
        return np.array([lo[0] + 0.3*(hi[0] - lo[0])])

    @property
    def conversion(self):
        '''Hydrogen conversion [-].'''
        return (self.F0.Xi[0]*self.F0.n - self.F1.Xi[0]*self.F1.n)/(self.F0.Xi[0]*self.F0.n)

    @property
    def tau0(self):
        return self.F0.P*self.Vol*1e3/0.08314/self.F0.T/self.F0.n

    @property
    def tau1(self):
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
    def __init__(self, name:str, Medium:media.Medium_Base=media.HB(), AF=1.0, Vol:float=1.0, r_type:bool=True,
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
    _inlet_state = HB._inlet_state

    def __init__(self, name:str, Medium:media.Medium_Base=media.HB(), Vol:float=1.0, AF:float=1.0, r_type:bool=True,
                 UA_V:float=0.0, T_wall:float=673.15, dP:float=0.0, n_points:int=50, **legacy):
        super().__init__(dict(name=name, Medium=Medium, Vol=Vol, AF=AF, r_type=r_type, UA_V=UA_V, T_wall=T_wall,
                              dP=dP, n_points=n_points))
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self._init_state(legacy)
        self.F1.set(n=self.F0.n, Xi=self.F0.Xi, T=self.F0.T, P=self.F0.P)
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
        self.F1.set(n=float(n.sum()), Xi=n/n.sum(), T=float(T_end), P=float(P))
        rows = []
        for V, xi, T in zip(sol.t, sol.y[0], sol.y[1]):
            nV, PV = self._state(xi, T, V)
            rows.append(dict(V=V, xi=xi, T=T, P=PV, **{f'x_{sp}': v for sp, v in zip(self.F0.Medium.species, nV/nV.sum())}))
        import pandas as pd
        self.profile = pd.DataFrame(rows)
        sol.residual_norm = 0.0 if sol.success else np.inf
        return sol

    @property
    def Q(self):
        '''Heat added through the wall [kJ/s].'''
        return self.F1.n*self.F1.h - self.F0.n*self.F0.h

    @property
    def conversion(self):
        return (self.F0.Xi[0]*self.F0.n - self.F1.Xi[0]*self.F1.n)/(self.F0.Xi[0]*self.F0.n)

    @property
    def Exth(self):
        return self.Q*(1 - 298.15/self.T_wall)

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n - self.Q/self.T_wall


class MeOH_Equil(EquilBase):
    """
    Equilibrium isothermal methanol synthesis reactor.
    Two reactions are considered to reach equilibrium:
        1) CO2 + 3H2 <-> CH3OH + H2O
        2) CO2 + H2 <-> CO + H2O
    
    Parameters:
        name: object's name
        Medium: media of the inlet stream. Defaults to media.Fuel_MeOH()
        T: temperature in [K]. Defaults to 473.15 
        P: pressure in [bar]. Defaults to 50.0
        Q: heat released/absorbed in [kJ/s]. Defaults to 0.0 
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium1. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium2. Defaults to None   
    
    Attributes: 
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        Exth: thermal exergy of the object in [kJ/s]
        Exloss: exergy loss in the object in [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Notes:
        Reference: Graaf 1985 "Chemical Equilibria in Methanol Synthesis"
    
    Examples:
        MeOH_Equilibrium = MeOH_Equil.Reactors('MeOH_example', Medium=media.Fuel_MeOH(), F0_n=100, 
        F0_Xi=np.array([0.75, 0.05, 0.1, 0.1, 1e-8, 1e-8, 1e-8]), T=473.15, P=50.0)
         
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_MeOH(), 
                 T:float=473.15, P:float=50.0, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None, #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None,
                ):
        
        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q)   
        self._N_H = np.array([2,4,0,0,2,0,4])
        self._N_C = np.array([0,1,1,1,0,0,1])
        self._N_O = np.array([0,0,2,1,1,0,1])
        self._N_N = np.array([0,0,0,0,0,2,0])
        
    def Eqs(self):

        self.F1.P = self.F0.P
        self.F1.T = self.F0.T 

        self.Eq:npArray[np.float64] = np.array([
            # MB
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)), # C balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)), # H balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)), # O balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)), # N balance
            self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1], # CH4
            1 - np.sum(self.F1.Xi), 
            # Equil
            self.F1.Xi[0]*self.F1.Xi[2]*self.K_rWGS() - (self.F1.Xi[3]*self.F1.Xi[4]), 
            self.F1.Xi[0]**3*self.F1.Xi[2]*self.P**2*self.K_MeOH() - (self.F1.Xi[6]*self.F1.Xi[4]), # for CO2
            #(self.F1.Xi[0]**2*self.F1.Xi[3]*self.P**2*self.K_MeOH()) - self.F1.Xi[6] # for CO   
        ])

        self.Q = (self.F1.n*self.F1.h - self.F0.n*self.F0.h )  

        self.F0_n = self.F0.n
        self.F0_Xi = self.F0.Xi
        self.F1_n = self.F1.n
        self.F1_Xi = self.F1.Xi

        return self.Eq
    
    def K_rWGS(self):
        return eq.K_rWGS(self.F1.T)

    def K_MeOH(self):
        return eq.K_MeOH_CO2(self.F1.T) # for CO2

    def MBcheck(self)->float:
        """Print the relative mass balance error and return out - in, g/s."""
        self.m0:float = self.F0.n*self.F0.mw
        self.m1:float = self.F1.n*self.F1.mw
        MB:float = self.m1 - self.m0
        print(f"{(self.F1.n*self.F1.mw/(self.F0.n*self.F0.mw) - 1)*100}% difference")
        return MB
    
    def f_sol(self, x):
        self.F1.n = x[0]
        self.F1.Xi = x[1:8]
        return self.Eqs()
        
    def run_sol(self):
        x0 = np.concatenate(([self.F0.n], media.Fuel_MeOH._default_Xi))
        return self._solve_composition(self.f_sol, x0)

class MeOH_Yield(EquilBase):
    """
    Methanol synthesis isothermal reactor where the methanol yield is given and the rWGS reaction is in equilibrium 
    Two reactions are considered:
        1) CO2 + 3H2 <-> CH3OH + H2O (yield)
        2) CO2 + H2 <-> CO + H2O (equilibrium) 
    
    Parameters:
        name: object's name
        Medium: media of the inlet stream. Defaults to media.Fuel_MeOH()
        Yield: methanol yield. Its goes from 0 to 1. Defaults to 0.35
        T: temperature in [K]. Defaults to 473.15 
        P: pressure in [bar]. Defaults to 50.0
        Q: heat released/absorbed in [kJ/s]. Defaults to 0.0 
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium1. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium2. Defaults to None   
    
    Attributes: 
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        Exth: thermal exergy of the object in [kJ/s]
        Exloss: exergy loss in the object in [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Notes:
        Reference: Graaf 1985 "Chemical Equilibria in Methanol Synthesis"
    
    Examples:
        MeOH_Yield = MeOH_yield.Reactors('MeOH_example', Medium=media.Fuel_MeOH(), Yield=0.33, F0_n=100, 
        F0_Xi=np.array([0.75, 0.05, 0.1, 0.1, 1e-8, 1e-8, 1e-8]), T=473.15, P=50.0)
    
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_MeOH(), 
                 Yield:float=0.35, T:float=473.15, P:float=50.0, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None,  #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None
                 ):
        
        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q)   
        self._N_H = np.array([2,4,0,0,2,0,4])
        self._N_C = np.array([0,1,1,1,0,0,1])
        self._N_O = np.array([0,0,2,1,1,0,1])
        self._N_N = np.array([0,0,0,0,0,2,0])
        self.Yield = Yield
        
    def Eqs(self):

        self.F1.P = self.F0.P
        self.F1.T = self.F0.T
        
        self.Eq:npArray[np.float64] = np.array([
            # MB
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)), # C balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)), # H balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)), # O balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)), # N balance
            self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1], # CH4
            1 - np.sum(self.F1.Xi), # Xi balance
            (self.Yield*self.F0.n*(self.F0.Xi[2]+self.F0.Xi[3]) + self.F0.n*self.F0.Xi[6]) - (self.F1.n*self.F1.Xi[6]), #MeOH            
            # Equil
            self.F1.Xi[0]*self.F1.Xi[2]*self.K_rWGS() - (self.F1.Xi[3]*self.F1.Xi[4]), 
        ])

        self.Q = (self.F1.n*self.F1.h - self.F0.n*self.F0.h )

        self.F0_n = self.F0.n
        self.F0_Xi = self.F0.Xi
        self.F1_n = self.F1.n
        self.F1_Xi = self.F1.Xi
        
        return self.Eq
    
    def K_rWGS(self):
        return eq.K_rWGS(self.F1.T)

    MBcheck = MeOH_Equil.MBcheck

    def f_sol(self, x): 
        self.F1.n = x[0]
        self.F1.Xi = x[1:8]
        return self.Eqs()
        
    def run_sol(self):
        x0 = np.concatenate(([self.F0.n], media.Fuel_MeOH._default_Xi))
        return self._solve_composition(self.f_sol, x0)

class MeOH_Kinetic(EquilBase):
    """
    Isothermal kinetic based reactor model for methanol synthesis
    Two reactions are considered:
        1) CO2 + 3H2 <-> CH3OH + H2O 
        2) CO2 + H2 <-> CO + H2O 
    
    Parameters:
        name: object's name
        Medium: media of the inlet stream. Defaults to media.Fuel_MeOH()
        M_cat: mass of the catalyst in [kg]. Defaults to None
        T: temperature in [K]. Defaults to 473.15 
        P: pressure in [bar]. Defaults to 50.0
        Q: heat released/absorbed in [kJ/s]. Defaults to 0.0 
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium1. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium2. Defaults to None   
    
    Attributes: 
        R: ideal gas constant in [J/mol]
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        Exth: thermal exergy of the object in [kJ/s]
        Exloss: exergy loss in the object in [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        r_MeOH: determination of reaction rate of the CO2 to methanol reaction (CO2-Hydrogenation). Returns: float in [mol/kg_cat*s]
        r_rWGS: determination of reaction rate of the reverse water gas shift reaction (rWGS). Returns: float in [mol/kg_cat*s]
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Notes:
        Reference: Van den Bussche and Froment, 1996, "A steady-state kinetic model for methanol synthesis and the water 
        gas shift reaction in a commercial catalyst"
       
    Examples:
        MeOH_Kinetic = MeOH_Kinetic.Reactors('MeOH_example', Medium=media.Fuel_MeOH(), M_cat=1000, F0_n=100, 
        F0_Xi=np.array([0.75, 0.05, 0.1, 0.1, 1e-8, 1e-8, 1e-8]), T=473.15, P=50.0)

    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_MeOH(), M_cat:float=None,
                 T:float=473.15, P:float=50, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None,  #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None
                 ):

        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q)   
        self.M_cat = M_cat
        self.R:float = 8.314462618 # J/mol

        self.F1.P = self.F0.P
        self.F1.T = self.F0.T   

    # adsorption constants (evaluated at the current reactor temperature)
    _K_a = 3453.38

    @property
    def _K_b(self):
        return 0.499*np.exp(17197/(self.R*self.F1.T))

    @property
    def _K_c(self):
        return 6.62e-11*np.exp(124119/(self.R*self.F1.T))

    def Eqs(self): 
        rrWGS = self.r_rWGS()   
        rMeOH = self.r_MeOH()
        self.Eq:npArray[np.float64] = np.array([          
            # MB
            (self.F0.n*self.F0.Xi[0] - 3*rMeOH*self.M_cat - rrWGS*self.M_cat) - self.F1.n*self.F1.Xi[0], # H2 balance
            (self.F0.n*self.F0.Xi[1]) - self.F1.n*self.F1.Xi[1], # CH4 balance 
            (self.F0.n*self.F0.Xi[2] - rMeOH*self.M_cat - rrWGS*self.M_cat) - self.F1.n*self.F1.Xi[2], # CO2 balance 
            (self.F0.n*self.F0.Xi[3] + rrWGS*self.M_cat) - self.F1.n*self.F1.Xi[3], # CO balance            
            (self.F0.n*self.F0.Xi[4] + rMeOH*self.M_cat + rrWGS*self.M_cat) - self.F1.n*self.F1.Xi[4], # H2O balance
            (self.F0.n*self.F0.Xi[5]) - self.F1.n*self.F1.Xi[5], # N2 balance
            (self.F0.n*self.F0.Xi[6] + rMeOH*self.M_cat) - self.F1.n*self.F1.Xi[6], # CH3OH balance                                
            1 - np.sum(self.F1.Xi), # Xi balance
        ])
        
        self.Q = (self.F1.n*self.F1.h - self.F0.n*self.F0.h )

        self.F0_n = self.F0.n
        self.F0_Xi = self.F0.Xi
        self.F1_n = self.F1.n
        self.F1_Xi = self.F1.Xi

        return self.Eq

    def r_MeOH(self)->float: 
        # Equilibrium constant
        K_meoh = 10**((3066/self.F1.T) - 10.592)
        # kinetic factor
        k_meoh = 1.07*np.exp(36696/(self.R*self.F1.T)) #mol/kg_cat*s*bar^2
        r_num = k_meoh*self.F1.Xi[0]*self.F1.Xi[2]*self.F1.P**2*(1 - (1/K_meoh)*self.F1.Xi[4]*self.F1.Xi[6]/(self.F1.Xi[0]**3*self.F1.Xi[2]*self.F1.P**2))
        r_den = (1 + self._K_a*(self.F1.Xi[4]/self.F1.Xi[0]) + self._K_b*(self.F1.Xi[0]*self.F1.P)**0.5 + self._K_c*self.F1.Xi[4]*self.F1.P)**3 
        return r_num/r_den 
    
    def r_rWGS(self)->float: 
        # Equilibrium constant
        K_wgs = 10**((2073/self.F1.T) - 2.029)
        # kinetic factor
        k_rwgs = 1.22e10*np.exp(-94765/(self.R*self.F1.T)) #mol/kg_cat*s*bar
        r_num = k_rwgs*self.F1.Xi[2]*self.F1.P*(1 - K_wgs*self.F1.Xi[4]*self.F1.Xi[3]/(self.F1.Xi[0]*self.F1.Xi[2]))
        r_den = (1 + self._K_a*(self.F1.Xi[4]/self.F1.Xi[0]) + self._K_b*(self.F1.Xi[0]*self.F1.P)**0.5 + self._K_c*self.F1.Xi[4]*self.F1.P) 
        return r_num/r_den 

    MBcheck = MeOH_Equil.MBcheck

    def f_sol(self, x): 
        self.F1.n = x[0]
        self.F1.Xi = x[1:8]
        return self.Eqs()
        
    def run_sol(self, guess_x0=np.array([0.709,0.048,0.074,0.031,0.023,0.051, 0.064])):
        x0 = np.concatenate(([self.F0.n*0.9], guess_x0)) 
        # the rate expressions divide by x_H2 and x_CO2: keep the iterates strictly positive
        lower = np.concatenate(([1e-12*self.F0.n], np.full(7, 1e-12)))
        upper = np.concatenate(([np.inf], np.ones(7)))
        return solvers.solve(self.f_sol, x0, bounds=(lower, upper), scale=self.F0.n, name=self.name)

#WGS kinetics
class MeOH_WGS(EquilBase):
    """
    Methanol synthesis adiabatic reactor where the methanol yield is given and the WGS reaction is in equilibrium 
    Inlet and outlet tempeartures can be different
    Two reactions are considered:
        1) CO + 2H2 <-> CH3OH  (yield)
        2) CO2 + 3H2 <-> CH3OH + H2O (yield)
        3) CO + H2O <-> CO2 + H2 (equilibrium) WGS
    
    Parameters:
        name: object's name
        Medium: media of the inlet stream. Defaults to media.Fuel_MeOH()
        Yield: methanol yield. Its goes from 0 to 1. Defaults to 0.35
        T: temperature in [K]. Defaults to 473.15 
        P: pressure in [bar]. Defaults to 50.0
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium1. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium2. Defaults to None   
    
    Attributes: 
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        Exth: thermal exergy of the object in [kJ/s] defaulted to 0
        Exloss: exergy loss in the object in [kJ/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Notes:
        Reference: Graaf 1985 "Chemical Equilibria in Methanol Synthesis"
        Water gas shift reaction kinetics are used 
        Methanol yield is based on concentrations of CO and CO2 in the incoming streams.
    
    Examples:
        MeOH_Yield = MeOH_yield.Reactors('MeOH_example', Medium=media.Fuel_MeOH(), Yield=0.33, F0_n=100, 
        F0_Xi=np.array([0.75, 0.05, 0.1, 0.1, 1e-8, 1e-8, 1e-8]), T=473.15, P=50.0)
    
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_MeOH(), 
                 Yield:float=0.35, T:float=473.15, P:float=50.0, Q:float=0.0,
                 F0_n:float=1.0, F0_Xi=None,  #F0_Xi:npArray[np.float64]|None=None, 
                 F1_n:float=1.0, F1_Xi=None #F1_Xi:npArray[np.float64]|None=None
                 ):
        
        super().__init__(name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q)   
        self._N_H = np.array([2,4,0,0,2,0,4])
        self._N_C = np.array([0,1,1,1,0,0,1])
        self._N_O = np.array([0,0,2,1,1,0,1])
        self._N_N = np.array([0,0,0,0,0,2,0])
        self.Yield = Yield
        
    def Eqs(self):

        self.F1.P = self.F0.P
        
        self.Eq:npArray[np.float64] = np.array([
            # MB
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)), # C balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)), # H balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)), # O balance
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)), # N balance
            self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1], # CH4
            1 - np.sum(self.F1.Xi), # Xi balance       
            # Equil
            self.F1.Xi[3]*self.F1.Xi[4]*self.K_WGS() - (self.F1.Xi[2]*self.F1.Xi[0]), #WGS

            (self.Yield*self.F0.n*(self.F0.Xi[3]+self.F0.Xi[2]) + self.F0.n*self.F0.Xi[6]) - (self.F1.n*self.F1.Xi[6]), #CO and CO2 to MeOH 
            self.Q, #EBal
        ])
        
        return self.Eq

    @property
    def Q(self):
        return self.F0.n*self.F0.h - self.F1.n*self.F1.h 
    
    def K_WGS(self):
        return eq.K_WGS(self.F1.T)
        
    def MBcheck(self)->float:
        MB = self.F1.m - self.F0.m
        print(f"{(self.F1.m/(self.F0.m) - 1)*100}% difference")
        return MB
    
    @property
    def Exth(self):
        return 0 

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth
    
    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n 

    @property
    def conversion(self): 
        H2Conv = ((self.F0.Xi[0])*self.F0.n - (self.F1.Xi[0])*self.F1.n)/((self.F0.Xi[0])*self.F0.n)
        CO2Conv = ((self.F0.Xi[2])*self.F0.n - (self.F1.Xi[2])*self.F1.n)/((self.F0.Xi[2])*self.F0.n)
        CConv = ((self.F0.Xi[2]+self.F0.Xi[3])*self.F0.n - (self.F1.Xi[2]+self.F0.Xi[3])*self.F1.n)/((self.F0.Xi[2]+self.F0.Xi[3])*self.F0.n)
        return {"H2 Conversion": H2Conv, "CO2 Conversion": CO2Conv, "Carbon Conversion": CConv}
        
    def f_sol(self, x): 
        self.F1.n = np.abs(x[0])
        self.F1.Xi = x[1:8]#np.abs(x[1:8])
        self.F1.T = np.abs(x[8])
        return self.Eqs()
        
    def run_sol(self):
        x0 = np.concatenate(([self.F0.n*0.9], np.array([0.69, 0.003, 0.17, 0.09, 0.027, 1e-10, 0.02 ]) , [self.F0.T*1.05]))
        lower = np.concatenate(([1e-12*self.F0.n], np.zeros(7), [200.0]))
        upper = np.concatenate(([np.inf], np.ones(7), [3000.0]))
        return solvers.solve(self.f_sol, x0, bounds=(lower, upper), scale=self.F0.n, name=self.name)

class Reformer(ModelBase):
    '''
    Medium = media.media_C3()
    '''
    def __init__(self, name, Medium, F0_n=1., F0_Xi=1., F1_n=1., F1_Xi=None, T=500., P=1., V=1., por=0.75, A = [17400., 23950., 5000.], 
                 Ea = [103191., 0.82*231266., 159241.], beta3=-1, alpha3=2.8, Twall=1000., UA=1e-3):
        
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
    
    def MBcheck(self):
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
    
    def f_sol(self, x):
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
    def __init__(self, name, Medium, F0_n=1., F0_Xi=1., T=500., P=1., V=1., N=10, por=0.75, A = [17400., 23950., 5000.], 
                 Ea = [103191., 0.82*231266., 159241.], beta3=-1, alpha3=2.8, Twall=1000., UA=1e-3):
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
