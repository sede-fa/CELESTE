'''
Electrochemical reactors
'''

from . import interfaces, ModelBase
from . import equilibrium as eq
from .exergy import T0
from .. import solvers
import numpy as np
import scipy.optimize as spo
from celeste import functions as funs

F_CONST = 96485  #: Faraday constant used by the electrochemical models, C/mol


class SOC(ModelBase):
    '''
    Solid oxide cell stack (electrolysis for I < 0, fuel cell for I > 0) with the
    fuel side at water-gas-shift and methane-reforming equilibrium.

    Parameters:
        name:      objects name
        MediumF:   fuel side medium, species order of media.Fuel_CH4() (H2, CH4, CO2, CO, H2O, N2)
        MediumA:   air side medium, media.Air() (O2, N2)
        Tsoc0:     inlet temperature in [K] (alias: T_soc0)
        Tsoc1:     outlet temperature in [K] (alias: T_soc1)
        P_soc:     optional operating pressure in [bar]; by default the fuel inlet pressure is used
        I:         current per cell in [A]; negative in electrolysis mode
        cellArea:  active cell area in [cm2]
        Ncells:    number of cells
        f_ASR:     multiplier on the area specific resistance correlation
        extent:    subtract the Gibbs energy of the internal reforming reaction from the ideal work
        Qloss:     heat exchanged with the surroundings in [kJ/s]; like Q and W positive into the stack, so a
                   loss is negative (Qloss = -5 means 5 kW lost)

    Ports: F0 -> F1 (fuel side), A0 -> A1 (air side)

    Properties:
        Vid, Vop:  reversible and operating cell voltage [V]
        j:         current density [A/cm2]
        W:         electrical power into the stack [kJ/s]
        Q:         heat into the stack [kJ/s]
        RC:        reactant conversion [-]
    '''
    def __init__(self, name, MediumF, MediumA, Tsoc0:float=1073.15, Tsoc1:float=1073.15, I:float=0.0, cellArea:float=100., Ncells:int|float=1,
                 f_ASR:float=1., extent:bool=False, Qloss:float=0.0, *, T_soc0:float|None=None, T_soc1:float|None=None,
                 P_soc:float|None=None):
        # T_soc0 / T_soc1 are accepted as alternative spellings of Tsoc0 / Tsoc1
        Tsoc0 = Tsoc0 if T_soc0 is None else T_soc0
        Tsoc1 = Tsoc1 if T_soc1 is None else T_soc1
        args = dict(locals())
        for key in ('T_soc0', 'T_soc1', 'P_soc'):
            args.pop(key)
        super().__init__(args)
        P0 = 1. if P_soc is None else P_soc
        self.F0 = interfaces.Flange(name + '.F0', Medium=MediumF, T=Tsoc0, P=P0)
        self.F1 = interfaces.Flange(name + '.F1', Medium=MediumF, T=Tsoc1, P=P0)
        self.A0 = interfaces.Flange(name + '.A0', Medium=MediumA, T=Tsoc0, P=P0)
        self.A1 = interfaces.Flange(name + '.A1', Medium=MediumA, T=Tsoc1, P=P0)

        self._N_H = np.array([2,4,0,0,2,0])
        self._N_C = np.array([0,1,1,1,0,0])
        self._N_O = np.array([0,0,2,1,1,0])
        self._N_N = np.array([0,0,0,0,0,2])

        self.Qloss = Qloss

    @property
    def P_soc(self):
        return self.F1.P
    
    @P_soc.setter
    def P_soc(self, P):
        self.F1.P = P
        self.A0.P = P
        self.A1.P = P

    def Eqs(self):
        self.P_soc=self.F0.P
        n_O = self.n_O

        self.Eq = np.array([
            # Fuel side MB
            self.F0.n*(np.dot(self.F0.Xi, self._N_C)) - self.F1.n*(np.dot(self.F1.Xi, self._N_C)), #
            self.F0.n*(np.dot(self.F0.Xi, self._N_H)) - self.F1.n*(np.dot(self.F1.Xi, self._N_H)), #
            self.F0.n*(np.dot(self.F0.Xi, self._N_O)) - self.F1.n*(np.dot(self.F1.Xi, self._N_O)) + n_O, #
            self.F0.n*(np.dot(self.F0.Xi, self._N_N)) - self.F1.n*(np.dot(self.F1.Xi, self._N_N)), #

            # Equil
            self.F1.Xi[3]*self.F1.Xi[4]*self.K_WGS() - self.F1.Xi[2]*self.F1.Xi[0], #
            self.F1.Xi[1]*self.F1.Xi[4]*self.K_MSR() - self.F1.Xi[3]*(self.F1.Xi[0]**3)*(self.F1.P**2)
        ]) 

        O2_out = self.A0.n*self.A0.Xi[0] - 0.5*n_O

        self.A1.n = self.A0.n - 0.5*n_O
        Xi_O2_out = O2_out/self.A1.n
        self.A1.Xi = np.array([Xi_O2_out, 1-Xi_O2_out], dtype=float)
        return self.Eq
    
    @property
    def n_O(self):
        return (self.Ncells*self.I/2/F_CONST)

    @property
    def Wchem(self):
        if self.extent:
            R = 8.314462618e-3
            gi_p = self.F1.T*R*np.log(self.F1.Xi*self.F1.P, where=(self.F1.Xi>=1e-6))
            return self.MSR_extent*(
                        4*(self.F1.Medium.g_i[0] + gi_p[0]) + (self.F1.Medium.g_i[2] + gi_p[2]) 
                        - 2*(self.F1.Medium.g_i[4] + gi_p[4]) - (self.F1.Medium.g_i[1] + gi_p[1]))
        else:
             return 0.
    
    @property
    def MSR_extent(self):
        return (self.F0.n*self.F0.Xi[1] - self.F1.n*self.F1.Xi[1])

    @property
    def Wid(self):
        return (self.F1.n*self.F1.g + self.A1.n*self.A1.g - self.F0.n*self.F0.g - self.A0.n*self.A0.g) - self.Wchem

    @property
    def Vid(self):
        return self.Wid*1e3/(-self.I*self.Ncells)

    @property
    def j(self):
        return self.I/self.cellArea

    @property
    def Vop(self):
        return self.Vid - self.f_ASR*self.ASR()*self.j

    @property
    def W(self):
        return self.Vop*(-self.I)*self.Ncells*1e-3
        
    @property
    def Q(self):
        return self.F1.n*self.F1.h + self.A1.n*self.A1.h - self.F0.n*self.F0.h - self.A0.n*self.A0.h - self.W - self.Qloss 
    
    def heat_and_work(self) -> float:
        """Q + W + Qloss: ``Q`` is defined net of ``Qloss``, so the energy balance needs all three."""
        return float(self.Q + self.W + self.Qloss)

    @property
    def Exth(self):
        # Qloss crosses the boundary at T0 and carries no exergy
        return self.Q*(1 - (298.15/self.F1.T))  
    
    @property
    def Exloss(self):
        return self.F0.Ex*self.F0.n + self.A0.Ex*self.A0.n - self.F1.Ex*self.F1.n - self.A1.Ex*self.A1.n + self.W + self.Exth 
        
    @property
    def RC(self):
        if self.I<0:
            val=-self.Ncells*self.I/(96485*(self.F0.n*(2*self.F0.Xi[2] + 2*self.F0.Xi[4]) - (2/self.RC_EC*self.MSR_extent if self.extent else 0) ) ) 
        else:
            val=self.Ncells*self.I/(2*96485*self.F0.n*(self.F0.Xi[0] + 4*self.F0.Xi[1] + self.F0.Xi[3]))
        return val
    
    @property
    def RC_EC(self):
        return (self.F0.n*(self.F0.Xi[2] + self.F0.Xi[4]) - self.F1.n*(self.F1.Xi[2] + self.F1.Xi[4]))/(
                self.F0.n*(self.F0.Xi[2] + self.F0.Xi[4]))
    
    def K_WGS(self):
        return eq.K_WGS(self.F1.T)

    def K_MSR(self):
        return eq.K_MSR(self.F1.T)
    
    def ASR(self):
        # M. Riedel 2020 DOI 10.1149/1945-7111/ab6820
        return 7.28e2*np.exp(-8.29e-3*((self.F1.T)-273.15))

    def cform(self, side='out'):
        if side == 'out':
            f = self.F1
        else:
            f = self.F0
        return funs.carbonFraction(f.T, f.P, dict(zip(f.Medium.species, f.Xi)))

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n + self.A1.s*self.A1.n - self.F0.s*self.F0.n - self.A0.s*self.A0.n - self.Q/self.F1.T - self.Qloss/T0
        
    def f_sol(self, x):
        x = np.abs(x)
        self.F1.Xi =  np.concatenate((x[:5], [1-np.sum(x[:5])]))
        self.F1.n = x[5]

        return self.Eqs()
        
    def run_sol(self, x0=None):
        if x0 is None:
            x0 = np.concatenate((self.F0.Xi[:5], [self.F0.n]))

        lower_bounds = [1e-8,1e-8,1e-8,1e-8,1e-8,1e-8*self.Ncells]
        upper_bounds = [0.9999,0.9999,0.9999,0.9999,0.9999,1e1*self.Ncells]
        x0_clipped = np.clip(np.abs(x0), lower_bounds, upper_bounds)

        sol = spo.least_squares(self.f_sol, x0_clipped, xtol=1e-15, bounds = (lower_bounds, upper_bounds)) 
        if np.linalg.norm(sol.fun)<=1e-7:
            sol = spo.root(self.f_sol, sol.x)
        if np.linalg.norm(sol.fun)>1e-7:
            sol = spo.root(self.f_sol, x0)
        res = self.f_sol(sol.x)
        sol.residual_norm = float(np.max(np.abs(res)))
        if not sol.success or sol.residual_norm > 1e-6*max(1.0, self.F0.n):
            sol.success = False
            import warnings
            warnings.warn(f'{self.name}: not converged (max |residual| = {sol.residual_norm:.3e})',
                          solvers.ConvergenceWarning, stacklevel=2)
        return sol  

    @property
    def EnthalpyEfficiency(self):
        LHV_H2 = self.F1.Medium.h_i[4] - self.F1.Medium.h_i[0] - 0.5*self.A1.Medium.h_i[0]
        LHV_CO = self.F1.Medium.h_i[2] - self.F1.Medium.h_i[3] - 0.5*self.A1.Medium.h_i[0]
        LHV_CH4 = self.F1.Medium.h_i[2] + 2*self.F1.Medium.h_i[4] - self.F1.Medium.h_i[3] - 2*self.A1.Medium.h_i[0]
        LHV_F0 = np.dot(-np.array([LHV_H2, LHV_CH4, 0, LHV_CO, 0, 0]), self.F0.Xi)*self.F0.n
        LHV_F1 = np.dot(-np.array([LHV_H2, LHV_CH4, 0, LHV_CO, 0, 0]), self.F1.Xi)*self.F1.n
        return (LHV_F1 - LHV_F0)/self.W


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
