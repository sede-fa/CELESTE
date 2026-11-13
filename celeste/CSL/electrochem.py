'''
Electrochemical reactors
'''

from . import interfaces, ModelBase
from . import equilibrium as eq
from .exergy import T0
from ..jx import nasa as _nasa
from .. import solvers
import numpy as np
from celeste import functions as funs

F_CONST = 96485  #: Faraday constant used by the electrochemical models, C/mol


class SOC(ModelBase):
    r'''
    Solid oxide cell stack (electrolysis for I < 0, fuel cell for I > 0); the fuel side leaves at
    water-gas-shift and methane-reforming equilibrium.

    System of equations (``Eqs``):
        unknowns:  ln of the fuel outlet flows of H2, CH4, CO2, CO, H2O, N2   (+ T_out when Q is specified)
        equations: C, H, O, N balances, WGS and MSR equilibrium             (+ energy balance)
    The current fixes the oxygen exchanged with the air side, n_O = Ncells*I/(2F). Solving for the
    logarithms keeps every flow positive, also for trace species (1e-8 CH4 in an electrolysis feed).

    Energy specification (one of them; ``soc.T = ...`` / ``soc.Q = ...`` switch it on an existing model and
    select the matching ``Eqs`` / ``f_sol``):
        T:  isothermal stack at T [K] (default 1073.15); Q is computed
        Q:  heat added [kJ/s]; Q=0 is adiabatic and the outlet temperature is solved

    Ideal voltage (``V_model``, see docs/electrochemistry.md):
        'mean':    Nernst potential averaged over the charge passed (default)
        'outlet':  Nernst potential at the outlet
        'streams': Gibbs energy difference of the streams (the former model)

    Parameters:
        name, MediumF (media.Fuel_CH4()), MediumA (media.Air())
        I:         current per cell [A]; negative in electrolysis mode
        cellArea:  active cell area [cm2]
        Ncells:    number of cells
        f_ASR:     multiplier on the area specific resistance correlation
        Qloss:     heat exchanged with the surroundings [kJ/s], positive into the stack (a loss is negative)
        T, Q:      energy specification, see above
        V_model:   'mean', 'outlet' or 'streams'
        n_nodes:   quadrature points of the 'mean' voltage
    Deprecated: Tsoc0/T_soc0 and P_soc (set F0.T, A0.T, F0.P), Tsoc1/T_soc1 (use T), extent (no effect).

    Ports: F0 -> F1 (fuel side), A0 -> A1 (air side); outlets at the outlet temperature and inlet pressure.
    '''
    ELEMENTS = eq.FUEL_ELEMENTS                       # rows C, H, O, N

    def __init__(self, name, MediumF, MediumA, Tsoc0=None, Tsoc1=None, I:float=0.0, cellArea:float=100.,
                 Ncells:int|float=1, f_ASR:float=1., extent=None, Qloss:float=0.0, *, T:float|None=None,
                 Q:float|None=None, V_model:str='mean', n_nodes:int=12, T_soc0=None, T_soc1=None, P_soc=None):
        T, T_in, P_in = _soc_legacy_inputs(T, Q, Tsoc0, T_soc0, Tsoc1, T_soc1, P_soc, extent)
        if V_model not in ('mean', 'outlet', 'streams'):
            raise ValueError(f"{name}: V_model must be 'mean', 'outlet' or 'streams', not {V_model!r}")
        super().__init__(dict(name=name, I=I, cellArea=cellArea, Ncells=Ncells, f_ASR=f_ASR, Qloss=Qloss,
                              V_model=V_model, n_nodes=n_nodes))
        self.F0 = interfaces.Flange(name + '.F0', Medium=MediumF, T=T_in, P=P_in)
        self.F1 = interfaces.Flange(name + '.F1', Medium=MediumF, T=T or T_in, P=P_in)
        self.A0 = interfaces.Flange(name + '.A0', Medium=MediumA, T=T_in, P=P_in)
        self.A1 = interfaces.Flange(name + '.A1', Medium=MediumA, T=T or T_in, P=P_in)
        if Q is None:
            self.T = T
        else:
            self.Q = Q

    # ---- specification ------------------------------------------------------------------------------
    @property
    def isothermal(self) -> bool:
        return self.Q_spec is None

    @property
    def T(self):
        '''Stack (outlet) temperature: the specified T, or the solved outlet temperature.'''
        return self.T_spec if self.isothermal else self.F1.T

    @T.setter
    def T(self, value):
        '''Setting T makes the stack isothermal at T.'''
        self.T_spec, self.Q_spec = value, None
        self.Eqs, self.f_sol, self.n_unknowns = self.Eqs_T, self.f_sol_T, 6

    @property
    def Q(self):
        '''Heat added to the stack [kJ/s] (net of Qloss).'''
        return self.F1.n*self.F1.h + self.A1.n*self.A1.h - self.F0.n*self.F0.h - self.A0.n*self.A0.h - self.W - self.Qloss

    @Q.setter
    def Q(self, value):
        '''Setting Q (0 for adiabatic) makes the outlet temperature follow from the energy balance.'''
        self.Q_spec, self.T_spec = value, None
        self.Eqs, self.f_sol, self.n_unknowns = self.Eqs_Q, self.f_sol_Q, 7

    @property
    def P_soc(self):
        return self.F1.P

    @property
    def n_O(self):
        '''Oxygen ions transferred to the fuel side [mol/s] (negative in electrolysis).'''
        return self.Ncells*self.I/2/F_CONST

    @property
    def xi_WGS(self):
        '''Extent of the water-gas shift [mol/s] (CO2 formed).'''
        return self.F1.n*self.F1.Xi[2] - self.F0.n*self.F0.Xi[2]

    @property
    def xi_MSR(self):
        '''Extent of methane reforming [mol/s] (CH4 converted).'''
        return self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1]

    # ---- voltage ------------------------------------------------------------------------------------
    @staticmethod
    def nernst(T, P, x_H2, x_H2O, x_O2):
        '''Nernst potential of H2 + 1/2 O2 -> H2O [V] at T [K], P [bar] and the given mole fractions.'''
        dG0 = -_nasa.R*T*eq.ln_K('H2_ox', T)                                    # kJ/mol
        dG = dG0 + _nasa.R*T*(np.log(x_H2O) - np.log(x_H2) - 0.5*np.log(x_O2*P))
        return -dG*1e3/(2*F_CONST)

    @property
    def V_mean(self):
        '''
        Nernst potential averaged over the charge passed [V] (Gauss-Legendre in lam, the fraction of the
        charge), with the fuel at equilibrium and T linear in lam from F0.T to F1.T.
        '''
        if self.n_O == 0:
            return np.nan
        nodes, weights = np.polynomial.legendre.leggauss(int(self.n_nodes))
        E = 0.0
        ln_n = np.log(np.maximum(self.F0.n*np.asarray(self.F0.Xi), 1e-12*self.F0.n))
        for lam, w in zip(0.5*(nodes + 1), 0.5*weights):
            T = self.F0.T + lam*(self.F1.T - self.F0.T)
            ln_n = solvers.solve(lambda u: self.fuel_residuals(np.exp(u), T, lam), ln_n, tol=1e-9, name=self.name).x
            n = np.exp(ln_n)
            n_O2, n_air = self.air_O2(lam)
            E += w*self.nernst(T, self.F0.P, n[0]/n.sum(), n[4]/n.sum(), n_O2/n_air)
        return float(E)

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

    def ASR(self, T=None):
        '''Area specific resistance [Ohm cm2], M. Riedel 2020 DOI 10.1149/1945-7111/ab6820.'''
        T = self.F1.T if T is None else T
        return 7.28e2*np.exp(-8.29e-3*(T - 273.15))

    @property
    def ASR_mean(self):
        '''ASR averaged over the charge passed [Ohm cm2]: T is linear in the charge, so the average is analytic.'''
        b, dT = 8.29e-3, self.F1.T - self.F0.T
        return self.ASR(self.F0.T)*(1 - np.exp(-b*dT))/(b*dT) if abs(dT) > 1e-9 else self.ASR(self.F0.T)

    @property
    def j(self):
        return self.I/self.cellArea

    @property
    def Vop(self):
        return self.Vid - self.f_ASR*self.ASR_mean*self.j

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
    def air_O2(self, lam=1.0):
        '''Air-side O2 and total flow [mol/s] after a fraction lam of the charge.'''
        n_O2 = self.A0.n*self.A0.Xi[0] - 0.5*lam*self.n_O
        return n_O2, self.A0.n - 0.5*lam*self.n_O

    def fuel_residuals(self, n, T, lam=1.0):
        '''Fuel-side equations (``eq.fuel_residuals``) after a fraction lam of the charge; the oxygen comes from the air side.'''
        return eq.fuel_residuals(n, self.F0.n*np.asarray(self.F0.Xi, dtype=float), T, self.F0.P, lam*self.n_O)

    def Eqs_T(self):
        '''Isothermal: element balances, WGS and MSR equilibrium.'''
        self.Eq = self.fuel_residuals(self.F1.n*np.asarray(self.F1.Xi), self.F1.T)
        return self.Eq

    def Eqs_Q(self):
        '''Specified heat duty: element balances, WGS and MSR equilibrium, energy balance.'''
        self.Eq = np.append(self.Eqs_T(), (self.Q - self.Q_spec)/self.F0.n)
        return self.Eq

    def set_outlets(self, n, T):
        '''Fuel outlet from the mole flows n; air outlet from the oxygen exchanged; both at T and the inlet pressure.'''
        P = self.F0.P
        self.F1.n, self.F1.Xi, self.F1.T, self.F1.P = n.sum(), n/n.sum(), T, P
        n_O2, n_air = self.air_O2()
        self.A1.n, self.A1.Xi, self.A1.T, self.A1.P = n_air, np.array([n_O2/n_air, 1 - n_O2/n_air]), T, P

    def f_sol_T(self, x):
        '''Isothermal stack: x = ln of the fuel outlet flows.'''
        self.set_outlets(np.exp(x), self.T_spec)
        return self.Eqs_T()

    def f_sol_Q(self, x):
        '''Specified heat duty: x = (ln of the fuel outlet flows, outlet temperature).'''
        self.set_outlets(np.exp(x[:6]), x[6])
        return self.Eqs_Q()

    def run_sol(self, x0=None):
        '''Solve ``f_sol`` (chosen by T= / Q=), starting from the feed (and the inlet temperature).'''
        if x0 is None:
            ln_feed = np.log(np.maximum(self.F0.n*np.asarray(self.F0.Xi, dtype=float), 1e-12*self.F0.n))
            x0 = np.append(ln_feed, self.F0.T)[:self.n_unknowns]
        sol = solvers.solve(self.f_sol, x0, tol=1e-9, name=self.name)
        self.f_sol(sol.x)
        return sol


def _soc_legacy_inputs(T, Q, Tsoc0, T_soc0, Tsoc1, T_soc1, P_soc, extent):
    '''Energy specification and initial inlet state from the SOC arguments, including the deprecated ones.'''
    old = {'Tsoc0': Tsoc0, 'T_soc0': T_soc0, 'Tsoc1': Tsoc1, 'T_soc1': T_soc1, 'P_soc': P_soc, 'extent': extent}
    used = [k for k, v in old.items() if v is not None]
    if used:
        from .._deprecation import warn_deprecated
        warn_deprecated(f"SOC({', '.join(used)}=...) is deprecated: give the outlet temperature as T= (or a heat "
                        f"duty Q=) and set the inlet state on the flanges (unit.F0.T = ..., unit.F0.P = ...); "
                        f"extent has no effect", stacklevel=4)
    T_out = T_soc1 if T_soc1 is not None else Tsoc1
    if T_out is not None:
        T = T_out
    if T is not None and Q is not None:
        raise ValueError("SOC: give either T (isothermal) or Q (heat duty), not both")
    if T is None and Q is None:
        T = 1073.15
    T_in = T_soc0 if T_soc0 is not None else (Tsoc0 if Tsoc0 is not None else (T or 1073.15))
    return T, T_in, (1.0 if P_soc is None else P_soc)


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
