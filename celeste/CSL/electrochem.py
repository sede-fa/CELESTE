'''
Electrochemical reactors
'''

from . import interfaces, ModelBase
from . import equilibrium as eq
from .exergy import T0
from functools import lru_cache
from ..jx import nasa as _nasa
from .. import solvers
import warnings
import numpy as np
import scipy.optimize as spo
from celeste import functions as funs

F_CONST = 96485  #: Faraday constant used by the electrochemical models, C/mol


_REACTIONS = {**eq.REACTIONS, 'H2ox': {'H2': -1, 'O2': -0.5, 'H2O': 1}}


@lru_cache(maxsize=4096)
def _lnK(reaction: str, T: float) -> float:
    """ln K of a reaction at T, from the NASA Gibbs energies (cached: the solvers revisit the same T)."""
    return float(np.log(eq.K_eq(float(T), _REACTIONS[reaction])))


class SOC(ModelBase):
    r'''
    Solid oxide cell stack (electrolysis for I < 0, fuel cell for I > 0) with the fuel side at
    water-gas-shift and methane-reforming equilibrium.

    Unknowns:   the extents xi_WGS and xi_MSR [mol/s] (and the outlet temperature when Q is specified)
    Equations:  WGS equilibrium, MSR equilibrium (and the energy balance when Q is specified)

    The fuel outlet follows from three extents,

        n_out = n_in + n_O*nu_EC + xi_WGS*nu_WGS + xi_MSR*nu_MSR

    where the electrochemical extent n_O = Ncells*I/(2F) (H2 + O -> H2O on the fuel side) is fixed by
    the current, so the element balances hold by construction.

    Energy specification (one of them):
        T:  isothermal stack, outlet temperature T [K] (default 1073.15); Q is computed
        Q:  heat added to the stack [kJ/s]; Q=0 is adiabatic and the outlet temperature is solved

    Ideal voltage (``V_model``), see docs/electrochemistry.md:
        'mean':    Nernst potential averaged over the charge passed (default); with T linear in the charge
                   from the fuel inlet to the outlet. Isothermal with an equilibrium feed it equals the
                   Gibbs energy approach (G_out - G_in)/(-I*Ncells).
        'outlet':  Nernst potential at the outlet
        'streams': Gibbs energy difference of the streams (the former model)
    All three are available as ``V_mean``, ``V_nernst_out`` and ``V_streams``.

    Parameters:
        name:      objects name
        MediumF:   fuel side medium with H2, CH4, CO2, CO, H2O, N2 (media.Fuel_CH4())
        MediumA:   air side medium, media.Air() (O2, N2)
        I:         current per cell in [A]; negative in electrolysis mode
        cellArea:  active cell area in [cm2]
        Ncells:    number of cells
        f_ASR:     multiplier on the area specific resistance correlation
        Qloss:     heat exchanged with the surroundings in [kJ/s]; like Q and W positive into the stack, so a
                   loss is negative (Qloss = -5 means 5 kW lost)
        T, Q:      energy specification, see above
        V_model:   'mean' (default), 'outlet' or 'streams'
        n_nodes:   quadrature points of the 'mean' voltage

    Deprecated: ``Tsoc0``/``T_soc0`` (set ``F0.T``/``A0.T``), ``Tsoc1``/``T_soc1`` (use ``T``),
    ``P_soc`` (set ``F0.P``/``A0.P``), ``extent`` (no effect).

    Ports: F0 -> F1 (fuel side), A0 -> A1 (air side); the outlets leave at the outlet temperature and the
    fuel inlet pressure.
    '''
    V_models = ('mean', 'outlet', 'streams')

    def __init__(self, name, MediumF, MediumA, Tsoc0:float|None=None, Tsoc1:float|None=None, I:float=0.0,
                 cellArea:float=100., Ncells:int|float=1, f_ASR:float=1., extent:bool|None=None, Qloss:float=0.0, *,
                 T:float|None=None, Q:float|None=None, V_model:str='mean', n_nodes:int=12,
                 T_soc0:float|None=None, T_soc1:float|None=None, P_soc:float|None=None):
        if V_model not in self.V_models:
            raise ValueError(f"{name}: V_model must be one of {self.V_models}, not {V_model!r}")
        T_in = T_soc0 if T_soc0 is not None else Tsoc0
        T_out = T_soc1 if T_soc1 is not None else Tsoc1
        if T_out is not None:
            if T is not None or Q is not None:
                raise ValueError(f"{name}: give the outlet temperature as T only")
            T = T_out
        if T is not None and Q is not None:
            raise ValueError(f"{name}: give either T (isothermal) or Q (heat duty), not both")
        if T is None and Q is None:
            T = 1073.15
        super().__init__(dict(name=name, MediumF=MediumF, MediumA=MediumA, I=I, cellArea=cellArea, Ncells=Ncells,
                              f_ASR=f_ASR, Qloss=Qloss, V_model=V_model, n_nodes=n_nodes,
                              Tsoc0=T_in, Tsoc1=T_out, extent=bool(extent)))
        self.T_spec, self.Q_spec = T, Q

        T_init = T_in if T_in is not None else (T if T is not None else 1073.15)
        P_init = 1. if P_soc is None else P_soc
        self.F0 = interfaces.Flange(name + '.F0', Medium=MediumF, T=T_init, P=P_init)
        self.F1 = interfaces.Flange(name + '.F1', Medium=MediumF, T=T or T_init, P=P_init)
        self.A0 = interfaces.Flange(name + '.A0', Medium=MediumA, T=T_init, P=P_init)
        self.A1 = interfaces.Flange(name + '.A1', Medium=MediumA, T=T or T_init, P=P_init)

        #                     H2 CH4 CO2 CO H2O N2
        self._nu = np.array([[-1,  0,  0,  0,  1, 0],     # EC:  H2 + O(2-) -> H2O + 2e-
                             [ 1,  0,  1, -1, -1, 0],     # WGS: CO + H2O <-> CO2 + H2
                             [ 3, -1,  0,  1, -1, 0]])    # MSR: CH4 + H2O <-> CO + 3 H2
        self._N_H = np.array([2,4,0,0,2,0])
        self._N_C = np.array([0,1,1,1,0,0])
        self._N_O = np.array([0,0,2,1,1,0])
        self._N_N = np.array([0,0,0,0,0,2])
        if [str(sp) for sp in MediumF.species] != ['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2']:
            raise ValueError(f"{name}: MediumF must hold H2, CH4, CO2, CO, H2O, N2 in this order")

        self.xi_WGS, self.xi_MSR = 0.0, 0.0
        self._V_mean, self._ASR_mean = np.nan, np.nan

        deprecated_kw = [k for k, v in (('Tsoc0', Tsoc0), ('T_soc0', T_soc0), ('Tsoc1', Tsoc1), ('T_soc1', T_soc1),
                                        ('P_soc', P_soc), ('extent', extent)) if v is not None]
        if deprecated_kw:
            from .._deprecation import warn_deprecated
            warn_deprecated(f"SOC({', '.join(deprecated_kw)}=...) is deprecated: give the outlet temperature as T= "
                            f"(or a heat duty Q=), set the inlet state on the flanges (unit.F0.T = ..., unit.F0.P = ...); "
                            f"extent has no effect, the ideal voltage excludes the reforming reactions",
                            stacklevel=2)

    # ---- specification ------------------------------------------------------------------------------
    @property
    def isothermal(self) -> bool:
        return self.Q_spec is None

    @property
    def T(self):
        '''Stack (outlet) temperature: the specified T, or the solved outlet temperature.'''
        return self.T_spec if self.T_spec is not None else self.F1.T

    @T.setter
    def T(self, value):
        '''Setting T makes the stack isothermal at T.'''
        self.T_spec, self.Q_spec = value, None

    @property
    def Q(self):
        '''Heat added to the stack [kJ/s] (net of Qloss).'''
        return self.F1.n*self.F1.h + self.A1.n*self.A1.h - self.F0.n*self.F0.h - self.A0.n*self.A0.h - self.W - self.Qloss

    @Q.setter
    def Q(self, value):
        '''Setting Q (0 for adiabatic) makes the outlet temperature follow from the energy balance.'''
        self.Q_spec, self.T_spec = value, None

    @property
    def P_soc(self):
        return self.F1.P

    @P_soc.setter
    def P_soc(self, P):
        self.F1.P = P
        self.A0.P = P
        self.A1.P = P

    @property
    def n_O(self):
        '''Oxygen ions transferred to the fuel side [mol/s] (negative in electrolysis): the electrochemical extent.'''
        return self.Ncells*self.I/2/F_CONST

    @property
    def xi(self):
        '''Extents of the electrochemical, water-gas-shift and reforming reactions [mol/s].'''
        return np.array([self.n_O, self.xi_WGS, self.xi_MSR])

    @property
    def MSR_extent(self):
        return self.xi_MSR

    # ---- voltage ------------------------------------------------------------------------------------
    @staticmethod
    def nernst(T, P, x_H2, x_H2O, x_O2):
        '''Nernst potential of H2 + 1/2 O2 -> H2O [V] at T [K], P [bar] and the given mole fractions.'''
        dG0 = -_nasa.R*T*_lnK('H2ox', T)                                    # kJ/mol
        dG = dG0 + _nasa.R*T*(np.log(x_H2O) - np.log(x_H2) - 0.5*np.log(x_O2*P))
        return -dG*1e3/(2*F_CONST)

    def charge_average(self, T1):
        '''
        Nernst potential and ASR averaged over the charge passed (Gauss-Legendre in lam), with the fuel at
        local equilibrium and T linear in lam from the fuel inlet to T1.
        '''
        if self.n_O == 0:
            return np.nan, float(self.ASR(T1))
        nodes, weights = np.polynomial.legendre.leggauss(int(self.n_nodes))
        E, ASR, xi = 0.0, 0.0, (self.xi_WGS, self.xi_MSR)
        for lam, w in zip(0.5*(nodes + 1), 0.5*weights):
            T = self.F0.T + lam*(T1 - self.F0.T)
            xi = self.local_equilibrium(lam, T, xi)
            n = self.fuel_moles(lam, *xi)
            n_O2, n_air = self.air_O2(lam)
            E += w*self.nernst(T, self.F0.P, n[0]/n.sum(), n[4]/n.sum(), n_O2/n_air)
            ASR += w*self.ASR(T)
        return float(E), float(ASR)

    def update_voltage(self):
        '''Recompute the charge-averaged voltage and ASR for the current outlet temperature.'''
        if self.V_model == 'mean':
            self._V_mean, self._ASR_mean = self.charge_average(self.F1.T)

    @property
    def V_mean(self):
        '''Charge-averaged Nernst potential [V].'''
        return self._V_mean

    @property
    def V_nernst_out(self):
        '''Nernst potential at the outlet [V].'''
        return float(self.nernst(self.F1.T, self.F1.P, self.F1.Xi[0], self.F1.Xi[4], self.A1.Xi[0]))

    @property
    def V_nernst_in(self):
        '''Nernst potential of the inlet streams as fed [V].'''
        return float(self.nernst(self.F0.T, self.F0.P, self.F0.Xi[0], self.F0.Xi[4], self.A0.Xi[0]))

    @property
    def V_streams(self):
        '''(G_out - G_in)/(-I*Ncells) of the streams [V]; the former ideal voltage.'''
        dG = self.F1.n*self.F1.g + self.A1.n*self.A1.g - self.F0.n*self.F0.g - self.A0.n*self.A0.g
        return dG*1e3/(-self.I*self.Ncells)

    @property
    def Vid(self):
        '''Ideal (reversible) cell voltage [V] of the chosen ``V_model``.'''
        return {'mean': self.V_mean, 'outlet': self.V_nernst_out, 'streams': self.V_streams}[self.V_model]

    @property
    def Wid(self):
        '''Reversible electrical power [kJ/s].'''
        return self.Vid*(-self.I)*self.Ncells*1e-3

    def ASR(self, T=None):
        '''Area specific resistance [Ohm cm2], M. Riedel 2020 DOI 10.1149/1945-7111/ab6820.'''
        T = self.F1.T if T is None else T
        return 7.28e2*np.exp(-8.29e-3*(T - 273.15))

    @property
    def ASR_eff(self):
        '''ASR of the ohmic loss: averaged like the voltage ('mean') or at the outlet.'''
        return self._ASR_mean if self.V_model == 'mean' else float(self.ASR())

    @property
    def j(self):
        return self.I/self.cellArea

    @property
    def Vop(self):
        return self.Vid - self.f_ASR*self.ASR_eff*self.j

    @property
    def W(self):
        return self.Vop*(-self.I)*self.Ncells*1e-3

    def heat_and_work(self) -> float:
        """Q + W + Qloss: ``Q`` is defined net of ``Qloss``, so the energy balance needs all three."""
        return float(self.Q + self.W + self.Qloss)

    # ---- second law ---------------------------------------------------------------------------------
    @property
    def Exth(self):
        # Q crosses the boundary at the stack temperature; Qloss at T0 and carries no exergy
        return self.Q*(1 - (T0/self.F1.T))

    @property
    def Exloss(self):
        return self.F0.Ex*self.F0.n + self.A0.Ex*self.A0.n - self.F1.Ex*self.F1.n - self.A1.Ex*self.A1.n + self.W + self.Exth

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n + self.A1.s*self.A1.n - self.F0.s*self.F0.n - self.A0.s*self.A0.n - self.Q/self.F1.T - self.Qloss/T0

    # ---- performance --------------------------------------------------------------------------------
    @property
    def RC(self):
        if self.I<0:
            val=-self.Ncells*self.I/(F_CONST*(self.F0.n*(2*self.F0.Xi[2] + 2*self.F0.Xi[4])))
        else:
            val=self.Ncells*self.I/(2*F_CONST*self.F0.n*(self.F0.Xi[0] + 4*self.F0.Xi[1] + self.F0.Xi[3]))
        return val

    @property
    def RC_EC(self):
        return (self.F0.n*(self.F0.Xi[2] + self.F0.Xi[4]) - self.F1.n*(self.F1.Xi[2] + self.F1.Xi[4]))/(
                self.F0.n*(self.F0.Xi[2] + self.F0.Xi[4]))

    @property
    def EnthalpyEfficiency(self):
        '''Change of the fuel's lower heating value flow per electrical power (LHV at 25 degC from the medium).'''
        return (self.F1.n*self.F1.LHV - self.F0.n*self.F0.LHV)/self.W

    def K_WGS(self):
        return eq.K_WGS(self.F1.T)

    def K_MSR(self):
        return eq.K_MSR(self.F1.T)

    def cform(self, side='out'):
        f = self.F1 if side == 'out' else self.F0
        return funs.carbonFraction(f.T, f.P, dict(zip(f.Medium.species, f.Xi)))

    # ---- equations and solution ---------------------------------------------------------------------
    def fuel_moles(self, lam=1.0, xi_WGS=None, xi_MSR=None):
        '''Fuel-side mole flows [mol/s] after a fraction lam of the charge (lam=1: outlet).'''
        xi_WGS = self.xi_WGS if xi_WGS is None else xi_WGS
        xi_MSR = self.xi_MSR if xi_MSR is None else xi_MSR
        return self.F0.n*np.asarray(self.F0.Xi, dtype=float) + np.array([lam*self.n_O, xi_WGS, xi_MSR]) @ self._nu

    def air_O2(self, lam=1.0):
        '''Air-side O2 and total flow [mol/s] after a fraction lam of the charge.'''
        n_O2 = self.A0.n*self.A0.Xi[0] - 0.5*lam*self.n_O
        return n_O2, self.A0.n - 0.5*lam*self.n_O

    @staticmethod
    def equilibrium_residuals(x, T, P):
        '''ln(Q/K) of the shift and reforming reactions for the fuel mole fractions x (H2, CH4, CO2, CO, H2O, N2).'''
        ln_x = np.log(np.maximum(x, 1e-300))
        H2, CH4, CO2, CO, H2O = ln_x[:5]
        return np.array([
            CO2 + H2 - CO - H2O - _lnK('WGS', T),                          # CO + H2O <-> CO2 + H2
            CO + 3*H2 + 2*np.log(P) - CH4 - H2O - _lnK('MSR', T),          # CH4 + H2O <-> CO + 3 H2
        ])

    def Eqs(self):
        '''
        Outlets from the extents and the outlet temperature, then the residuals:
            WGS equilibrium, MSR equilibrium (and the energy balance when Q is specified).
        The voltage enters only the energy balance, so it is updated here only for a specified Q
        (run_sol updates it once at the end otherwise).
        '''
        T1 = self.T_spec if self.isothermal else self.F1.T
        P = self.F0.P

        # fuel side
        n = self.fuel_moles()
        self.F1.T, self.F1.P = T1, P
        self.F1.n = float(n.sum())
        self.F1.Xi = np.maximum(n, 1e-14*self.F0.n)/np.maximum(n, 1e-14*self.F0.n).sum()

        # air side: O2 to (fuel cell) or from (electrolysis) the fuel side
        n_O2, n_air = self.air_O2()
        self.A0.P = P
        self.A1.T, self.A1.P = T1, P
        self.A1.n = n_air
        self.A1.Xi = np.array([n_O2/n_air, 1 - n_O2/n_air])

        Eq = list(self.equilibrium_residuals(n/n.sum(), T1, P))
        if not self.isothermal:
            self.update_voltage()
            Eq.append((self.Q - self.Q_spec)/self.F0.n)
        self.Eq = np.array(Eq)
        return self.Eq

    def f_sol(self, x):
        '''x = (xi_WGS, xi_MSR) for an isothermal stack, (xi_WGS, xi_MSR, T_out) for a specified heat duty.'''
        self.xi_WGS, self.xi_MSR = x[0], x[1]
        if not self.isothermal:
            self.F1.T = x[2]
        return self.Eqs()

    def run_sol(self, x0=None):
        '''
        Isothermal: solve the two equilibria (f_sol) for (xi_WGS, xi_MSR).
        Heat duty:  solve the energy balance for the outlet temperature; at each trial temperature the extents
                    are at equilibrium (nested: robust with trace species such as 1e-8 CH4).
        x0 = (xi_WGS, xi_MSR) is an optional starting point.
        '''
        xi0 = (self.xi_WGS, self.xi_MSR) if x0 is None else x0[:2]
        if self.isothermal:
            xi0 = self.local_equilibrium(1.0, self.T_spec, xi0)
            sol = solvers.solve(self.f_sol, xi0, bounds=self.extent_bounds(), tol=1e-9, name=self.name)
            self.f_sol(sol.x)
        else:
            def energy_balance(T1):
                xi = self.local_equilibrium(1.0, T1, (self.xi_WGS, self.xi_MSR))
                return self.f_sol([*xi, T1])[2]
            T_guess = self.F1.T if 400. < self.F1.T < 2000. else self.F0.T
            sol = solvers.solve_scalar(energy_balance, 400., 2000., x0=T_guess, tol=1e-10, name=self.name)
            energy_balance(sol.x[0])
            if np.max(np.abs(self.Eq)) > 1e-8:
                sol.success = False
                warnings.warn(f"{self.name}: not converged (max |residual| = {np.max(np.abs(self.Eq)):.3e})",
                              solvers.ConvergenceWarning, stacklevel=2)
        self.update_voltage()
        return sol

    def extent_bounds(self, lam=1.0):
        '''Range of xi_WGS and xi_MSR that keeps every fuel mole flow positive.'''
        b = self.fuel_moles(lam, 0.0, 0.0)
        cache = self.__dict__.setdefault('_bounds', {})              # the charge-average revisits the same lam
        if tuple(b) in cache:
            return cache[tuple(b)]
        lo, hi = np.empty(2), np.empty(2)
        for r in range(2):
            for sign, out in ((1.0, hi), (-1.0, lo)):
                c = np.zeros(2); c[r] = -sign
                res = spo.linprog(c, A_ub=-self._nu[1:].T, b_ub=b, bounds=[(None, None)]*2, method='highs')
                out[r] = sign*(-res.fun) if res.status == 0 else sign*np.inf
        span = np.where(np.isfinite(hi - lo), hi - lo, self.F0.n)
        cache[tuple(b)] = (lo + 1e-12*span, hi - 1e-12*span)
        return cache[tuple(b)]

    def local_equilibrium(self, lam, T, xi0):
        '''
        xi_WGS, xi_MSR at equilibrium after a fraction lam of the charge, at T and the inlet pressure.

        Newton's method on the two equilibrium residuals with the analytical Jacobian
            d ln(Q_r)/d xi_k = sum_i nu_ri nu_ki / n_i - dnu_r dnu_k / n_tot,
        each step shortened so that every mole flow stays positive. A generic solver is unreliable here:
        trace species (1e-8 CH4 in an electrolysis feed) make the residuals extremely steep near the bounds.
        '''
        nu = self._nu[1:]
        dnu = nu.sum(axis=1)                                         # change in moles per reaction (0, 2)

        def residuals(xi):
            n = self.fuel_moles(lam, *xi)
            return self.equilibrium_residuals(n/n.sum(), T, self.F0.P)

        xi = np.clip(xi0, *self.extent_bounds(lam))
        if np.any(self.fuel_moles(lam, *xi) <= 0):
            xi = 0.5*np.add(*self.extent_bounds(lam))
        r = residuals(xi)
        for _ in range(100):
            if np.max(np.abs(r)) < 1e-12:
                break
            n = self.fuel_moles(lam, *xi)
            J = (nu/n) @ nu.T - np.outer(dnu, dnu)/n.sum()
            step = np.linalg.solve(J, -r)
            dn = step @ nu
            t = min(1.0, 0.95*np.min(-n[dn < 0]/dn[dn < 0])) if np.any(dn < 0) else 1.0
            while np.linalg.norm(residuals(xi + t*step)) > np.linalg.norm(r) and t > 1e-8:
                t *= 0.5
            if t <= 1e-8:                                            # no further progress (round-off)
                break
            xi = xi + t*step
            r = residuals(xi)
        return xi


class AEL(ModelBase):
    '''
    Alkaline electrolyser model
    It determines the current density [A/cm2] (j) and outlet mole flow and composition for a given mole flow of water

    Parameters:
        name:   objects name
        Medium: media.Media_AEL() (H2O(L), H2, O2, H2O)
        T, P:   operating temperature [K] and pressure [bar]
        j:      initial guess of the current density [A/cm2], negative for electrolysis
        Ncells: number of cells
        A:      active cell area [cm2]

    Ports: F0 (water in) -> F1 (hydrogen side out), F2 (oxygen out)
    '''
    def __init__(self, name, Medium, T=298.15, P=1, j=-0.32, Ncells=None, A=None,
                 F0_n=1, F0_Xi=np.array([1, 0, 0, 0]), 
                 F1_n=1, F1_Xi=None, #['H2O(L)', 'H2', 'O2', 'H2O']
                 F2_n=1, F2_Xi=np.array([0, 0, 1, 0]) 
                ): 
                
        '''
        media.Media_AEL()
        '''
        super().__init__(locals())
        
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P) #H2
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=T, P=P) #O2
        self._j = j
        self.A = A
        
    def Eqs(self):
        y = np.array((
            [self.F0.Xi[0]+self.F0.Xi[3], self.F2.Xi[0]+self.F2.Xi[3], self.F1.Xi[0]+self.F1.Xi[3], 1], # H2O
            [self.F0.Xi[1], self.F2.Xi[1], self.F1.Xi[1], -1], # H2
            [self.F0.Xi[2], self.F2.Xi[2], self.F1.Xi[2], -0.5])) # O2
        
        n = np.array([self.F0.n, - self.F2.n, -self.F1.n, self.Ncells*self.I/2/F_CONST])

        self.Eq = np.concatenate((
            [
            np.sum(self.F1.Xi) - 1,
            ],
            np.dot(y, n),
        ))
            
        self.F1.P = self.F0.P
        self.F2.P = self.F0.P  
        
        return self.Eq

    @property
    def j(self):
        return self._j
    
    @j.setter
    def j(self, val):
        self._j = val
    
    @property
    def I(self):
        return self.j*self.A
    
    @property
    def Vop(self):
        return 1.481 + 2.9*1e-3*self.F0.P - self.j*1.05

    @property
    def W(self):
        return -self.I*self.Vop*self.Ncells*1e-3 # to kJ/s

    @property
    def Q(self):
        return self.F2.n*self.F2.h + self.F1.n*self.F1.h - self.F0.n*self.F0.h - self.W
    
    @property
    def Exth(self):
        return self.Q*(1 - 298.15/self.F1.T)
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex + self.Exth + self.W
    
    @property
    def Exeff(self):
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.max([0, self.Exth]) +  self.W)
        
    def f_sol(self, x):
        self.j = x[0]
        self.F1.n = x[1]
        self.F1.Xi = np.array([0, x[2], 0.001, 0.002])
        self.F2.n = x[3]
        return self.Eqs()
        
    def run_sol(self):
        x0 = np.array([self.j,self.F0.n,0.997,self.F0.n])  
        return solvers.solve(self.f_sol, x0, scale=self.F0.n, name=self.name)
