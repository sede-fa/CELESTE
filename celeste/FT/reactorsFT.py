import numpy as np
import scipy.optimize as spo

from ..CSL import components, ModelBase
from . import interfacesFT as interFT, mediaFT

from numpy.typing import NDArray as npArray
from typing import Literal

class RStoic(ModelBase): #base.ModelBase
    '''
    Stoichiometric reactor for complete conversion of light gases (C2-C4) to methane
    3C2H6 + H2O -> 5CH4 + CO
    3C3H8 + 2H2O -> 7CH4 + 2CO
    3C4H10 + 3H2O -> 9CH4 + 3CO
    '''
    
    def __init__(self, name, Medium, T=500, P=23, Q=0,
                 F0_n=1e-5, F0_Xi=None, F1_n=1e-5, F1_Xi=None,
                ):
        
        super().__init__ (locals()) #name, Medium=Medium, T=T, P=P, F0_n=F0_n, F0_Xi=F0_Xi, F1_n=F1_n, F1_Xi=F1_Xi, Q=Q
        self.FH0 = interFT.Flange(name + '.FH0', Medium=mediaFT.Water(), n=1, Xi=np.array([1]), T=T, P=P, phase='Gas') # steam in
        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P, phase='Gas')
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P, phase='Gas')

    def Eqs(self):   
        index = np.concatenate(([0,1,4], np.arange(9,55)))
        
        self.Eq = np.concatenate((
            [(self.F0.n + self.FH0.n) - self.F1.n],
            self.F1.Xi[[3,6,7,8]] - 0, # no H20 and C2-C4 in the outlet stream
            self.F0.n*self.F0.Xi[index] - self.F1.n*self.F1.Xi[index], # H2,CO2,N2,C5-C50 
            [self.FH0.n - self.F0.n*(np.dot(self.F0.Xi[[6,7,8]], [1,2,3])/3)], # H2O 
            [(self.F0.n*self.F0.Xi[2] + self.FH0.n) - self.F1.n*self.F1.Xi[2]],  # CO
            [(self.F0.n*self.F0.Xi[5] + self.F0.n*(np.dot(self.F0.Xi[[6,7,8]], [5,7,9])/3)) - self.F1.n*self.F1.Xi[5]], # CH4    
            
        ))
        self.Q = self.F1.n*self.F1.h - (self.F0.n*self.F0.h + self.FH0.n*self.FH0.h)   # heat duty, kW
        
        return self.Eq
    
    def MBcheck(self):
        # Mass balance
        self.mH0 = self.FH0.n*self.FH0.mw 
        self.m0 = self.F0.n*self.F0.mw 
        self.m1 = self.F1.n*self.F1.mw
        MB = (self.mH0 + self.m0) - self.m1 
        print(f"{abs(((self.mH0 + self.m0)/(self.m1)) - 1)*100}% difference ")    
        return MB
    
    def ExBs(self):
        self.Exth = self.Q*(1 - (298.15/self.F0.T))
        self.Exloss = self.FH0.n*self.FH0.Ex + self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.Exth
        self.Exeff = 1 - self.Exloss/(self.F0.n*self.F0.Ex + self.FH0.n*self.FH0.Ex + np.maximum(0, self.Exth)) 
        
    def f_sol(self, x):
        
        self.FH0.n = x[0]
        self.F1.n = x[1]
        self.F1.Xi = x[2:57]
        
        return self.Eqs()
        
    def run_sol(self):
    
        x0 = np.concatenate(([self.F0.n] ,[self.F0.n], self.F0.Xi)) #, [self.F0.T+2]
        
        sol = spo.root(self.f_sol, x0)
        self.f_sol(sol.x)
        return sol
    
class FT(ModelBase):
    """
    Low temperature Fischer-Tropsch reactor assumimng cobalt catalyst -> CO2 doesn't react.
    Isothermal and isobaric. The temperature and pressure must be given.
    The reactor produces only linear and saturated alkanes, specifically hydrocarbons from C1 to C50 
    The model depends on the inlet CO conversion, the ASF distribution and the stochiometric of the reactions 
            CO + (2n+1)/n H2 -> 1/n CnH2n+2 + H2O 
    H2O (and CO2) should be removed from the inlet stream. Moreover, together with any other HC entering the reactor, they are all interts.

    Parameters:
        name: object's name
        Medium: media of the inlet stream. Defaults to mediaFT.Syncrude()
        alpha: alpha value in the ASF equation. Its goes from 0 to 1. Defaults to None 
        CO_conv: CO conversion. Its goes from 0 to 1. Defaults to None
        T: temperature in [K]. Defaults to 450
        P: pressure in [bar]. Defaults to 30
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: flange F1 mol flow in [mol/s]. Defaults to 1.0
        F2_n: flange F2 mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium. Defaults to None   
        F1_Xi: flange F1 mol fraction of every component in Medium. Defaults to None  
        F2_Xi: flange F2 mol fraction of every component in Medium. Defaults to None
        phase0: phase of the intlet stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Gas'
        phase1: phase of the flange F1 stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Gas'
        phase2: phase of the flange F2 stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Liquid'
    
    Attributes: 
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: flange F1 mass flow in [g/s]
        m2: flange F2 mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - flange F1 exergy flow in [kJ/s]
        F2.Ex: float - flange F2 exergy flow in [kJ/s]
        Exth: thermal exergy of the object in [kJ/s]
        Exloss: exergy loss in the object in [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Examples:
        FT = FT.reactorsFT('FT_example', Medium=mediaFT.Syncrude(), alpha=0.92, CO_conv=0.4,
        F0_n=100, F0_Xi=mediaFT.Syncrude()._reference_X, T=473.15, P=30.0)

    """
    
    def __init__(self, name:str, Medium:mediaFT.Syncrude=mediaFT.Syncrude(), alpha:float=None, CO_conv:float=None, 
                 F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, T:float=450, P:float=30, phase0:Literal['Gas', 'Liquid']='Gas',
                 F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, phase1:Literal['Gas', 'Liquid']='Gas', 
                 F2_n:float=1.0, F2_Xi:npArray[np.float64]|None=None, phase2:Literal['Gas', 'Liquid']='Liquid'):     
        super().__init__(locals())  
        
        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P, phase=phase0)
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P, phase=phase1) 
        self.F2 = interFT.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=T, P=P, phase=phase2) 
        self.alpha = alpha
        self.CO_conv = CO_conv
        self.__ASF = self.HC_dist() 
        self.F0.T = T
        self.F1.T = T
        self.F2.T = T
        self.F0.P = P
        self.F1.P = P
        self.F2.P = P
        
    def Eqs(self):
        
        # H2 balance check       
        H2_ = np.sum(self.F0.n*self.F0.Xi[2]*self.CO_conv*self.__ASF*(2 + 1/np.arange(1,51)))   
        if H2_ > self.F0.n*self.F0.Xi[0]:
            self.CO_conv_temp = self.F0.Xi[0]/self.F0.Xi[2]/np.sum(self.__ASF*(2 + 1/np.arange(1,51))) # total H2 conversion
        else:
            self.CO_conv_temp = self.CO_conv

        # F2 mol balance
        self.F2.n = np.sum(self.F0.n*self.F0.Xi[27:]) + np.sum(self.F0.n*self.F0.Xi[2]*self.CO_conv_temp*self.__ASF[22:]/np.arange(23,51)) 
        self.F2.Xi[0:27] = np.zeros(27) # H2 to C22
        self.F2.Xi[27:] = self.F0.n*(self.F0.Xi[27:] + self.F0.Xi[2]*self.CO_conv_temp*self.__ASF[22:]/np.arange(23,51))/self.F2.n # C23-C55
        # F1 mol balance
        self.F1.n = (self.F0.n*(self.F0.Xi[0] - self.F0.Xi[2]*self.CO_conv_temp*np.sum(self.__ASF*(2 + 1/np.arange(1,51)))) + self.F0.n*self.F0.Xi[1] + 
                     self.F0.n*self.F0.Xi[2]*(1 - self.CO_conv_temp) + (self.F0.n*(self.F0.Xi[3] + self.F0.Xi[2]*self.CO_conv_temp)) +
                     self.F0.n*self.F0.Xi[4] + np.sum(self.F0.n*(self.F0.Xi[5:27] + self.F0.Xi[2]*self.CO_conv_temp*self.__ASF[0:22]/np.arange(1,23))) 
                     )
        self.F1.Xi[0] = self.F0.n*(self.F0.Xi[0] - self.F0.Xi[2]*self.CO_conv_temp*np.sum(self.__ASF*(2 + 1/np.arange(1,51))) )/self.F1.n  # H2
        self.F1.Xi[1] = self.F0.n*self.F0.Xi[1]/self.F1.n # CO2  
        self.F1.Xi[2] = self.F0.n*self.F0.Xi[2]*(1 - self.CO_conv_temp)/self.F1.n #  CO
        self.F1.Xi[3] = self.F0.n*(self.F0.Xi[3] + self.F0.Xi[2]*self.CO_conv_temp)/self.F1.n # H2O 
        self.F1.Xi[4] = self.F0.n*self.F0.Xi[4]/self.F1.n # N2
        self.F1.Xi[5:27] = self.F0.n*(self.F0.Xi[5:27] + self.F0.Xi[2]*self.CO_conv_temp*self.__ASF[0:22]/np.arange(1,23))/self.F1.n # C1-C22
        self.F1.Xi[27:] = np.zeros(28) # C23-C55

        self.F1.Xi = self.F1.Xi
        self.F2.Xi = self.F2.Xi
    
    @property
    def Q(self):
        return (self.F1.n*self.F1.h + self.F2.n*self.F2.h) - self.F0.n*self.F0.h
  
    def HC_dist(self): 
        '''
        Returns the normalized ASF distribution of hydrocarbons: 
                ASF= (1-alpha)*alpha**(n°C - 1) 
        correction obtained from "Hamelinck_2004_Production of FT transportation fuels from biomass_technical options, process analysis"
        '''    
        C_distr = np.array([(1 - self.alpha)*self.alpha**(i - 1) for i in range(50)]) 
        norm_C = C_distr/np.sum(C_distr)
        ## C1-C4 correction
        C4_correc = np.sum(norm_C[0:4])*np.array([0.74, 0.16, 0.06, 0.04])
        ASF = np.concatenate((C4_correc, norm_C[4:50]))
        return ASF
    
    def MBcheck(self)->float:
         #Mass balance
        self.m0:float = self.F0.n*self.F0.mw
        self.m1:float = self.F1.n*self.F1.mw
        self.m2:float = self.F2.n*self.F2.mw
        MB:float = self.m0 - (self.m1 + self.m2)
        print(f"{self.CO_conv_temp*100} CO conversion ")
        print(f"{self.F0.Xi[0]/self.F0.Xi[2]} Syngas ratio ")
        print(f"{abs(self.m0/(self.m1 + self.m2) - 1)*100}% difference ")    
        return MB
        
    def ExBs(self)->None:
        self.Exth:float = self.Q*(1 - (298.15/self.F0.T))
        self.Exloss:float = self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex + self.Exth
        self.Exeff:float = 1 - self.Exloss/(self.F0.n*self.F0.Ex + np.maximum(0, self.Exth))      
        
    def run_sol(self):
        return self.Eqs()   

class Dist(components.GPUbase):
    def __init__(self, name, Medium, eff=1.,
                 F0_n=1, F0_Xi=np.zeros(55), 
                 F0_T=400, F0_P=1, phase0='Gas',
                 F1_T=400, F1_P=1, phase1='Liquid',
                 F2_T=400, F2_P=1, phase2='Liquid',
                 F3_T=400, F3_P=1, phase3='Liquid',
                ):       
        super().__init__(name, eff, locals())
        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P, phase=phase0)
        self.F1 = interFT.Flange(name + '.Naphtha', Medium=mediaFT.Naphtha(), n=1, Xi=np.zeros(6), T=F1_T, P=F1_P, phase=phase1) # naphtha C5-C10
        self.F2 = interFT.Flange(name + '.Kerosene', Medium=mediaFT.Kerosene(), n=1, Xi=np.zeros(4), T=F2_T, P=F2_P, phase=phase2) # kerosene
        self.F3 = interFT.Flange(name + '.Distillate', Medium=mediaFT.Distillate(), n=1, Xi=np.zeros(8), T=F3_T, P=F3_P, phase=phase3) #distillate

    def Eqs(self):
        self.Eq = np.concatenate((
            [self.F0.n - self.F1.n - self.F2.n - self.F3.n],
            self.F0.n*self.F0.Xi[self.F1.Medium.I1:self.F1.Medium.I2] - self.F1.n*self.F1.Xi,
            self.F0.n*self.F0.Xi[self.F2.Medium.I1:self.F2.Medium.I2] - self.F2.n*self.F2.Xi,
            self.F0.n*self.F0.Xi[self.F3.Medium.I1:self.F3.Medium.I2] - self.F3.n*self.F3.Xi,
            [np.sum(self.F1.Xi) - 1],
            #[np.sum(self.F2.Xi) - 1],
            [np.sum(self.F3.Xi) - 1],
            ))

        #self.Q = (self.F1.n*self.F1.h + self.F2.n*self.F2.h + self.F3.n*self.F3.h) - self.F0.n*self.F0.h 

        return self.Eq
    
    def MBcheck(self):
        # Mass balance
        self.m0 = self.F0.n*self.F0.mw 
        self.m1 = self.F1.n*self.F1.mw
        self.m2 = self.F2.n*self.F2.mw
        self.m3 = self.F3.n*self.F3.mw
        MB = self.m0 - (self.m1 + self.m2 + self.m3)
        print(f"{abs((self.m0/(self.m1 + self.m2 + self.m3)) - 1)*100}% difference ")    
        return MB
    
    def ExBs(self):
        self.Wid = self.F3.n*self.F3.Ex + self.F2.n*self.F2.Ex + self.F1.n*self.F1.Ex - self.F0.n*self.F0.Ex
        self.W = float(np.where(self.Wid > 0, self.Wid/self.eff, self.Wid*self.eff))
    
    def f_sol(self, x):

        self.F1.n = x[0]
        self.F1.Xi = x[1:7]
        self.F2.n = x[7]
        self.F2.Xi = x[8:12]
        self.F3.n = x[12]
        self.F3.Xi = x[13:21]
        
        return self.Eqs()
    
    def run_sol(self):
    
        x0 = np.concatenate(([self.F0.n/2], mediaFT.Naphtha()._reference_X, [self.F0.n/4], mediaFT.Kerosene()._reference_X, [self.F0.n/4], mediaFT.Distillate()._reference_X ))
        
        sol = spo.root(self.f_sol, x0)
        self.f_sol(sol.x)
        if not sol.success:
            print(f'{self.name} Failed')
        return sol


class Hydrocracker(ModelBase):
    """
    Wax conversion of 100% 
    The amount of H2 needed is calculated from H2 balance
    Yield reactor with: 
    25 wt%: lights(C1-C4) 2% + naphtha(C5-C10) 23%
    50 wt%: kerosene(C11-C14)
    25 wt%: Distillate (C15-C22)
    """
        
    def __init__(self, name, Medium, F0_Xi=None, 
                 F0_n=1, F0_T=400, F0_P=1, phase0='Liquid',
                 FH0_n=1, FH0_T=400, FH0_P=1, # Hydrogen in
                 F1_n=1, F1_T=400, F1_P=1, phase1='Gas', # light gases
                 F2_n=1, F2_T=400, F2_P=1, phase2='Gas',  # Naphtha
                 F3_n=1, F3_T=400, F3_P=1, phase3='Gas',  # Kerosene
                 F4_n=1, F4_T=400, F4_P=1, phase4='Gas'):  # Distillates
        super().__init__(locals())
        
        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P, phase=phase0) # change to wax      
        self.FH0 = interFT.Flange(name + '.FH0', Medium=mediaFT.Hydrogen(), n=FH0_n, Xi=np.array([1]), T=FH0_T, P=FH0_P, phase='Gas') # steam in
        self.F1 = interFT.Flange(name + '.LightGas', Medium=mediaFT.LightGas(), n=F1_n, Xi=self.equiMol(mediaFT.LightGas()), T=F1_T, P=F1_P, phase=phase1) # Light gases C1-C4
        self.F2 = interFT.Flange(name + '.Naphtha', Medium=mediaFT.Naphtha(), n=F2_n, Xi=self.equiMol(mediaFT.Naphtha()), T=F2_T, P=F2_P, phase=phase2) # naphtha 
        self.F3 = interFT.Flange(name + '.Kerosene', Medium=mediaFT.Kerosene(), n=F3_n, Xi=self.equiMol(mediaFT.Kerosene()), T=F3_T, P=F3_P, phase=phase3) # kerose
        self.F4 = interFT.Flange(name + '.Distillate', Medium=mediaFT.Distillate(), n=F4_n, Xi=self.equiMol(mediaFT.Distillate()), T=F4_T, P=F4_P, phase=phase4) #distillate     
        self.__MassFrac = np.array([0.02, 0.23, 0.5]) #, 0.25
        
    def Eqs(self): 
        
        # self.m0 = self.F0.n*self.F0.mw 
        # self.mH0 = self.FH0.n*self.FH0.mw
        # self.m1 = self.F1.n*self.F1.mw 
        # self.m2 = self.F2.n*self.F2.mw 
        # self.m3 = self.F3.n*self.F3.mw 
        # self.m4 = self.F4.n*self.F4.mw 
        H0 = self.F0.n*np.dot(self.F0.Xi, self.F0.Medium.HNumberi)
        H1 = self.F1.n*np.dot(self.F1.Xi, self.F1.Medium.HNumberi)
        H2 = self.F2.n*np.dot(self.F2.Xi, self.F2.Medium.HNumberi) 
        H3 = self.F3.n*np.dot(self.F3.Xi, self.F3.Medium.HNumberi)
        H4 = self.F4.n*np.dot(self.F4.Xi, self.F4.Medium.HNumberi)

        self.Eq = np.concatenate((
            # Total mass balance
            [(self.F0.m + self.FH0.m) - (self.F1.m + self.F2.m + self.F3.m + self.F4.m)],
            self.__MassFrac*(self.F0.m + self.FH0.m) - np.array([self.F1.m ,self.F2.m ,self.F3.m]),
            # Hydrogen balance           
            [H0 + self.FH0.n*2 - (H1 + H2 + H3 + H4)]
        ))

        return self.Eq
    
    @property
    def Q(self):
        return (self.F1.n*self.F1.h + self.F2.n*self.F2.h + self.F3.n*self.F3.h + self.F4.n*self.F4.h) - (self.F0.n*self.F0.h + self.FH0.n*self.FH0.h) 
    
    def equiMol(self, Medium):
        size = Medium.nX
        Xi = np.ones(size)/size
        Xi = Xi/np.sum(Xi)
        return Xi
        
    def MBcheck(self):
        # Mass balance
        self.m0 = self.F0.n*self.F0.mw 
        self.mH0 = self.FH0.n*self.FH0.mw
        self.m1 = self.F1.n*self.F1.mw
        self.m2 = self.F2.n*self.F2.mw
        self.m3 = self.F3.n*self.F3.mw
        self.m4 = self.F4.n*self.F4.mw
        MB = (self.F0.m + self.FH0.m) - (self.F1.m + self.F2.m + self.F3.m + self.F4.m)
        print(f"{abs(((self.F0.m + self.FH0.m)/(self.F1.m + self.F2.m + self.F3.m + self.F4.m)) - 1)*100}% difference ")    
        return MB
    
    def ExBs(self):
        self.Exth = self.Q*(1 - (298.15/self.F0.T))
        self.Exloss = (self.FH0.n*self.FH0.Ex + self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex 
                        - self.F2.n*self.F2.Ex - self.F3.n*self.F3.Ex - self.F4.n*self.F4.Ex + self.Exth)
        self.Exeff = 1 - self.Exloss/(self.FH0.n*self.FH0.Ex + self.F0.n*self.F0.Ex + np.maximum(0, self.Exth))        
   
    def f_sol(self, x):
        
        self.FH0.n = x[0]
        self.F1.n = x[1]
        self.F2.n = x[2]
        self.F3.n = x[3]
        self.F4.n = x[4]
      
        return self.Eqs()
        
    def run_sol(self):
        #guesses
        x0 = np.concatenate(([self.F0.n*1.25], [self.F0.n*0.22],[self.F0.n*0.88], [self.F0.n*1.16], [self.F0.n*0.4]))
        
        sol = spo.root(self.f_sol, x0)
        self.f_sol(sol.x)
        if not sol.success:
            print(f'{self.name} Failed')
        return sol




