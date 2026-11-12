import numpy as np
import scipy.optimize as spo

from ..CSL import interfaces, media, ModelBase, components as comp
from .. import solvers
from . import interfacesFT as interFT, mediaFT, thermoFT

from numpy.typing import NDArray as npArray
from typing import Literal

class GPU_1S(comp.GPUbase):
    """
    Black box object where the outlet conditions are prescribed and no reaction is occuring. 
    One stream is entering and one stream is coming out. 
    The outlet n, Xi, h, cp, g, s are determined. 
    The work of the GPU_1S object is calculated for a given efficiency.
    IMPORTANT: the inlet and outlet temperature, pressure and phase must be given manually.
    
    Parameters:
        name: objects name
        Medium1: media of the inlet stream
        Medium2: media of the outlet stream
        eff: work efficiency of the GPU_1S. Must be 0 < eff <= 1. Defaults to 1
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium 
        F1_Xi: outlet mol fraction of every component in Medium  
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: outlet temperature in [K]. Defaults to 298.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: outlet pressure in [bar]. Defaults to 1.0 
        phase0: phase of the intlet stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Gas'
        phase1: phase of the oulet stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Gas'
    
    Attributes: 
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        Wid: ideal GPU work in [kJ/s]
        W: real GPU work in [kJ/s]

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method

    Notes:  
        The size of F0_Xi and F1_Xi must be the same.
        Accept change of phase only for water.   
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.

    Examples:
        1) with change of phase
        GPU = componentsFT.GPU_1S('GPU_example', Medium=mediaFT.Syncrude(), eff=0.7, F0_n=100, F0_Xi=mediaFT.Syncrude()._reference_X, 
        F0_T=873.15, F0_P=3.0, phase0='Gas', F1_T=673.15, F1_P=1.0, phase1='Liquid')    
    """  
    def __init__(self, name:str, Medium:mediaFT.Medium_Syncrude, eff:float=1,
                 F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0, phase0:Literal['Gas', 'Liquid']='Gas',
                 F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, F1_T:float=298.15, F1_P:float=1.0, phase1:Literal['Gas', 'Liquid']='Gas'
                 ):
        super().__init__(name, eff, locals())

        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P, phase=phase0)
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P, phase=phase1)
        
    def Eqs(self)->None:
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
    
    def MBcheck(self)->float:
        # Mass balance
        self.m0 = self.F0.n*self.F0.mw 
        self.m1 = self.F1.n*self.F1.mw
        MB = self.m0 - self.m1 
        print(f"{abs((self.m0/(self.m1)) - 1)*100}% difference ")    
        return MB
    
    def ExBs(self)->None:
        self.Wid = self.F1.n*self.F1.Ex - self.F0.n*self.F0.Ex
        self.W = float(np.where(self.Wid > 0, self.Wid/self.eff, self.Wid*self.eff))
              
    def run_sol(self):
        return self.Eqs()
 
class GPU_2S(comp.GPUbase):
    """
    Black box object where the outlet conditions are prescribed and no reaction is occuring. 
    One stream is entering (flange F0) and two streams (flanges F1 and F2) are coming out. 
    For each outlet streams n, Xi, h, cp, g, s, mw are determined. 
    The work of the GPU_2S object is calculated for a given efficiency.
    IMPORTANT: the inlet and the two outlet temperatures, pressures, and the composition of the flange F2 
    must be given manually.
    
    Parameters:
        name: objects name
        Medium: media of the streams
        idxSep: index of the component that goes to the flange F2. Takes one value 0<=idxSep<=(Medium.species.size-1).
                Defaults to 3 (water index in media.Syncrude())
        eff: work efficiency of the GPU_2S. Must be 0 < eff <= 1. Defaults to 1
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: flange F1 mol flow in [mol/s]. Defaults to 1.0
        F2_n: flange F2 mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium. Defaults to None   
        F1_Xi: flange F1 mol fraction of every component in Medium. Defaults to None  
        F2_Xi: flange F2 mol fraction of every component in Medium. Defaults to np.concatenate((np.array([0,0,0,1]),np.zeros(51)))
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: flange F1 temperature in [K]. Defaults to 298.15
        F2_T: flange F2 temperature in [K]. Defaults to 298.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: flange F1 pressure in [bar]. Defaults to 1.0
        F2_P: flange F2 pressure in [bar]. Defaults to 1.0
        phase0: phase of the intlet stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Gas'
        phase1: phase of the flange F1 stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Gas'
        phase2: phase of the flange F2 stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Gas'
    
    Attributes: 
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: flange F1 mass flow in [g/s]
        m2: flange F2 mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - flange F1 exergy flow in [kJ/s]
        F2.Ex: float - flange F2 exergy flow in [kJ/s]
        Wid: ideal GPU work in [kJ/s]
        W: real GPU work in [kJ/s]

    Methods: 
        Eqs: the two outlet mol flows and Xi are determined. Returns: "Eq" attribute
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        f_sol: contains the variable to be optimized by the solver. Return: the "Eqs" method
        run_sol: contains the optimizer function and guesses values. Returns: the solution of the optimization

    Notes:  
        The size of F0_Xi, F1_Xi and F2_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.

    Examples:
        1) Separation of water
        GPU = componentsFT.GPU_2S('GPU_example', Medium=mediaFT.Syncrude(), idxSep=3, eff=0.7, F0_n=100, phase0='Gas',
        F0_Xi=mediaFT.Syncrude()._reference_X, F0_T=873.15, F0_P=3.0, F1_T=673.15, F1_P=1.0, phase1='Gas',
        F2_T=350.15, F2_P=5.0, F2_Xi=np.concatenate((np.array([0,0,0,1]),np.zeros(51))), phase2='Gas')
    """
    def __init__(self, name:str, Medium:mediaFT.Medium_Syncrude, idxSep:int=3, eff:float=1,
                 F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0, phase0:Literal['Gas', 'Liquid']='Gas', 
                 F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, F1_T:float=298.15, F1_P:float=1.0, phase1:Literal['Gas', 'Liquid']='Gas', 
                 F2_n:float=1.0, F2_Xi:npArray[np.float64]|None=np.concatenate((np.array([0,0,0,1]), np.zeros(51))) , 
                 F2_T:float=298.15, F2_P:float=1.0, phase2:str='Gas'
                 ):
        super().__init__(name, eff, locals())

        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P, phase=phase0)
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P, phase=phase1) 
        self.F2 = interFT.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=F2_T, P=F2_P, phase=phase2) 
        self.idxSep = idxSep
        self.Medium = Medium
          
    def Eqs(self)->None:
        self.F2.n = self.F0.Xi[self.idxSep]*self.F0.n
        self.F1.n = self.F0.n - self.F2.n 
        self.F1.Xi = (self.F0.Xi*self.F0.n - self.F2.Xi*self.F2.n)/self.F1.n 
        
    def MBcheck(self)->float:
        # Mass balance
        print(f"{abs((self.F0.m/(self.F1.m+self.F2.m)) - 1)*100}% difference ")    
        return self.F0.m - (self.F1.m + self.F2.m)

    @property
    def Wid(self):
        return self.F2.n*self.F2.Ex + self.F1.n*self.F1.Ex - self.F0.n*self.F0.Ex

    @property
    def W(self):
        return float(np.where(self.Wid > 0, self.Wid/self.eff, self.Wid*self.eff))
    
    @property
    def Exloss(self):
        return self.W - self.Wid
        
    def run_sol(self):
        return self.Eqs()  
class GPU_2SV2(comp.GPUbase): 
    # Important: this is a more general GPU where several components can be separeted and is not fixed to F2 pure water
    # to change to this from the version above, the Sep array should be given
    """
    Black box object where the outlet conditions are prescribed and no reaction is occuring. 
    One stream is entering (flange F0) and two streams (flanges F1 and F2) are coming out. 
    For each outlet streams n, Xi, h, cp, g, s, mw are determined. 
    The work of the GPU_2S object is calculated for a given efficiency.
    IMPORTANT: the inlet and the two outlet temperatures, pressures must be given manually. 
    
    Parameters:
        name: objects name
        Medium: media of the streams
        Sep: separation ratio of the components that goes to the flange F1. Takes an array of size Medium.species.size. 
            The individual components values can be a float from 0<=Sep<=1. Defaults to None
        eff: work efficiency of the GPU_2SV2. Must be 0 < eff <= 1. Defaults to 1
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: flange F1 mol flow in [mol/s]. Defaults to 1.0
        F2_n: flange F2 mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium. Defaults to None   
        F1_Xi: flange F1 mol fraction of every component in Medium. Defaults to None  
        F2_Xi: flange F2 mol fraction of every component in Medium. Defaults to np.array([0.,0.,0.,0.,1.,0.]) 
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: flange F1 temperature in [K]. Defaults to 298.15
        F2_T: flange F2 temperature in [K]. Defaults to 298.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: flange F1 pressure in [bar]. Defaults to 1.0
        F2_P: flange F2 pressure in [bar]. Defaults to 1.0
    
    Properties: 
        Wid: ideal GPU work in [kJ/s]
        W: real GPU work in [kJ/s]

    Methods: 
        Eqs: the two outlet mol flows and compositions are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        run_sol: empty method. Returns: the system of equations in the "Eqs" method

    Notes:  
        The size of F0_Xi, F1_Xi and F2_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.

    Examples:
        1) Separation of 100% H2, 100% CH4, 50% CO2, 40% CO, 0% H2O,  100% N2 to the F1. The difference goes to F2
        GPU = Components.GPU_2SV2('GPU_example', Medium=media.Fuel_CH4(), Sep=np.array([1,1,0.5,0.4,0,1]), eff=0.7, F0_n=100, 
        F0_Xi=np.array([0.1,0.3,0.05,0.05,0.5,1e-8]), F0_T=873.15, F0_P=3.0, F1_T=673.15, F1_P=1.0, 
        F2_T=350.15, F2_P=5.0)
    """
    def __init__(self, name:str, Medium:mediaFT.Medium_Syncrude, Sep:npArray[np.float64]|None=None, eff:float=1,
                 F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0,
                 F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, F1_T:float=298.15, F1_P:float=1.0,
                 F2_n:float=1.0, F2_Xi:npArray[np.float64]|None=None, F2_T:float=298.15, F2_P:float=1.0,
                 ):

        super().__init__(name, eff, locals())

        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P)
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P)
        self.F2 = interFT.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=F2_T, P=F2_P)
        self.Sep = Sep
          
    def Eqs(self):
        self.F1.n = np.sum(self.F0.n*self.F0.Xi*self.Sep)
        self.F1.Xi = self.F0.n*self.F0.Xi*self.Sep/self.F1.n
        self.F2.n = self.F0.n - self.F1.n 
        self.F2.Xi = self.F0.n*self.F0.Xi*(1-self.Sep)/self.F2.n
        
    def MBcheck(self)->float:
        # Mass balance
        print(f"{abs((self.F0.m/(self.F1.m+self.F2.m)) - 1)*100}% difference ")    
        return self.F0.m - (self.F1.m + self.F2.m)
        
    @property
    def Wid(self):
        return (self.F2.n*self.F2.Ex + self.F1.n*self.F1.Ex) - self.F0.n*self.F0.Ex
    
    @property
    def W(self):
        return float(np.where(self.Wid > 0, self.Wid/self.eff, self.Wid*self.eff))
    
    @property
    def Exloss(self):
        return self.W - self.Wid
      
    def run_sol(self):
        return self.Eqs()

class GPU_3S(comp.GPUbase):
    """
    Black box object where the outlet conditions are prescribed and no reaction is occuring. 
    One stream is entering (flange F0) and three streams (flanges F1, F2 and F3) are coming out. 
    For each outlet streams n, Xi, h, cp, g, s, mw are determined. 
    The work of the GPU_3S object is calculated for a given efficiency.
    IMPORTANT: the inlet and the three outlet temperatures, pressures and the phases must be given manually.
    
    Parameters:
        name: objects name
        Medium: media of the streams
        sep: separation ratio. Could hold any value between 0 and 1, and also -1. A value must be given for the each component in Medium,
            The fraction given goes to the flange F1, the difference goes to the flange F2. The flange F3 has only water and the value 
            must be equals to -1. Defaults to None.
            IMPORTANT -1 could only be assigned to one component. 
        eff: work efficiency of the GPU_3S. Must be 0 < eff <= 1. Defaults to 1
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: flange F1 mol flow in [mol/s]. Defaults to 1.0
        F2_n: flange F2 mol flow in [mol/s]. Defaults to 1.0
        F3_n: flange F3 mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium. Defaults to None   
        F1_Xi: flange F1 mol fraction of every component in Medium. Defaults to None  
        F2_Xi: flange F2 mol fraction of every component in Medium. Defaults to None
        F3_Xi: flange F3 mol fraction of every component in Medium. Defaults to 1
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: flange F1 temperature in [K]. Defaults to 298.15
        F2_T: flange F2 temperature in [K]. Defaults to 298.15
        F3_T: flange F3 temperature in [K]. Defaults to 298.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: flange F1 pressure in [bar]. Defaults to 1.0
        F2_P: flange F2 pressure in [bar]. Defaults to 1.0
        F3_P: flange F3 pressure in [bar]. Defaults to 1.0
        phase0: phase of the intlet stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Gas'
        phase1: phase of the flange F1 stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Gas'
        phase2: phase of the flange F2 stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Gas'
        phase3: phase of the flange F2 stream. It can be either 'Gas' or 'Liquid'. Defaults to 'Liquid'
    
    Attributes: 
        Eq: equation to be solved
        m0: inlet mass flow in [g/s]
        m1: flange F1 mass flow in [g/s]
        m2: flange F2 mass flow in [g/s]
        m3: flange F3 mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - flange F1 exergy flow in [kJ/s]
        F2.Ex: float - flange F2 exergy flow in [kJ/s]
        F3.Ex: float - flange F3 exergy flow in [kJ/s]
        Wid: ideal GPU work in [kJ/s]
        W: real GPU work in [kJ/s]

    Methods: 
        Eqs: the three outlet mol flows and, the flanges F1 and F2 composition are determined. Returns: "Eq" attribute
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        f_sol: contains the variable to be optimized by the solver. Return: the "Eqs" method
        run_sol: contains the optimizer function and guesses values. Returns: the solution of the optimization

    Notes:  
        The size of F0_Xi, F1_Xi, F2_Xi and F3_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.
        The flange F3 is fixed for water. The composition and the Medium is fixed. Therefore the water index in sep MUST be -1.
    
    Examples:
        1) Separation of H2+HC to flange F1, CH4 to flange F2 and water to flange F3
        GPU = componentsFT.GPU_3S('GPU_example', Medium=mediaFT.Syncrude4(), sep=np.concatenate((np.array([1,1,1,-1,1,0,1,1,1]), np.zeros(46))),
        eff=0.7, F0_n=100, F0_Xi=np.concatenate((np.array([0.1,0,0,0.7,0,0.2,0,0,0]), np.zeros(46))), F0_T=873.15, F0_P=3.0, F1_T=673.15, F1_P=1.0, 
        F2_T=350.15, F2_P=5.0, F3_T=673.15, F3_P=1.0, phase0='Gas', phase1='Gas', phase2='Gas', phase3='Liquid')
        
        2) how sep works --> if the value given for a specific component is equals to 0.9:
        0.9 of this component goes to flange F1
        (1 - 0.9)=0.1 of this component goes to flange F2
        -1: completely goes to flange F3  
    """
    def __init__(self, name:str, Medium:mediaFT.Medium_Syncrude, sep:npArray[np.float64]|None=None, eff:float=1,
                 F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0, phase0:Literal['Gas', 'Liquid']='Gas',
                 F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, F1_T:float=298.15, F1_P:float=1.0, phase1:Literal['Gas', 'Liquid']='Gas',
                 F2_n:float=1.0, F2_Xi:npArray[np.float64]|None=None, F2_T:float=298.15, F2_P:float=1.0, phase2:Literal['Gas', 'Liquid']='Gas',
                 F3_n:float=1.0, F3_T:float=298.15, F3_P:float=1.0, phase3:Literal['Gas', 'Liquid']='Liquid',
                ):       
        super().__init__(name, eff, locals())

        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P, phase=phase0)
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P, phase=phase1) 
        self.F2 = interFT.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=F2_T, P=F2_P, phase=phase2) 
        self.F3 = interFT.Flange(name + '.F3', Medium=mediaFT.Water(), n=F3_n, Xi=np.array([1]), T=F3_T, P=F3_P, phase=phase3) 
        self.Medium = Medium
        self.sep = sep
        
    def Eqs(self):
        idx = np.flatnonzero(np.asarray(self.sep) == -1)
        sep1:npArray[np.float64] = np.where(self.sep >= 0, self.sep, 0).astype(float)
        self.F3.n = float(self.F0.n*self.F0.Xi[idx[0]])
        self.F1.n = np.sum(self.F0.n*self.F0.Xi*sep1)
        self.F1.Xi = self.F0.n*self.F0.Xi*sep1/self.F1.n
        self.F2.n = self.F0.n - self.F1.n - self.F3.n
        sep1[idx] = 1
        self.Eq:npArray[np.float64] = (self.F0.n*self.F0.Xi*(1 - sep1)) - self.F2.n*self.F2.Xi
        return self.Eq
    
    def MBcheck(self)->float:
        # Mass balance
        self.m0:float = self.F0.n*self.F0.mw 
        self.m1:float = self.F1.n*self.F1.mw
        self.m2:float = self.F2.n*self.F2.mw
        self.m3:float = self.F3.n*self.F3.mw
        MB:float = self.m0 - (self.m1 + self.m2 + self.m3)
        print(f"{abs((self.m0/(self.m1 + self.m2 + self.m3)) - 1)*100}% difference ")    
        return MB
    
    def ExBs(self)->None:
        self.Wid:float = self.F3.n*self.F3.Ex + self.F2.n*self.F2.Ex + self.F1.n*self.F1.Ex - self.F0.n*self.F0.Ex
        self.W:float = float(np.where(self.Wid > 0, self.Wid/self.eff, self.Wid*self.eff))
    
    def f_sol(self, x):
        self.F2.Xi = x[0:1+self.F1.Medium.nX]
        return self.Eqs()
    
    def run_sol(self):
        x0 = self.F0.Medium._reference_X
        sol = spo.least_squares(self.f_sol, x0)
        return sol   
    
class ChgMedium(comp.ChgMedium):
    '''
    ChgMedium of the standard library with FT defaults: inlet media.Fuel_CH4(), outlet mediaFT.Syncrude().
    Give phase0 / phase1 to use FT flanges (with a phase) on that side.
    '''
    def __init__(self, name:str, Medium0:media.Medium_Base|None=None, Medium1:media.Medium_Base|None=None,
                 phase0:Literal['Gas', 'Liquid', None]=None, phase1:Literal['Gas', 'Liquid', None]=None):
        Medium0 = media.Fuel_CH4() if Medium0 is None else Medium0
        Medium1 = mediaFT.Syncrude() if Medium1 is None else Medium1
        super().__init__(name, Medium0=Medium0, Medium1=Medium1)
        self.phase0, self.phase1 = phase0, phase1
        if phase0 is not None:
            self.F0 = interFT.Flange(name + '.F0', Medium=Medium0, phase=phase0)
        if phase1 is not None:
            self.F1 = interFT.Flange(name + '.F1', Medium=Medium1, phase=phase1)

class CompIsent(comp.CompIsent):
    """
    Isentropic compressor extending from the CompressorFT Class. 
    One stream is entering and one stream is coming out. 
    The outlet n, Xi, h, cp, g, s, T are determined.
    The real work of the CompIsent is calculated for a given efficiency.  
    IMPORTANT: the inlet temperature and pressure, and the outlet pressure must be given manually.
    
    Parameters:
        name: objects name
        Medium: media of the streams
        eff: work efficiency of the CompIsent object. Must be 0 < eff <= 1. Defaults to 1
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium. Defaults to None   
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: outlet temperature in [K]. Defaults to 398.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: outlet pressure in [bar]. Defaults to 3.0
    
    Attributes: 
        R: ideal gas constant in [kJ/mol K]
        cpa: average cp between the inlet and outlet streams in [kJ/mol K]
        Eq: equation to be solved
        W: real pump work in [kJ/s]
        m0: inlet mass flow in [g/s]
        m1: flange F1 mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - flange F1 exergy flow in [kJ/s]
        Exloss: exergy loss in the object [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol flow and composition are determined. Returns: "Eq" attribute
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        ExBs: calls the exergy method on each flange. Returns: None
        f_sol: contains the variable to be optimized by the solver. Return: the "Eqs" method
        run_sol: contains the optimizer function and guesses values. Returns: the solution of the optimization

    Notes:  
        The size of F0_Xi and F1_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.

    Examples:
        1) 
        Compressor = componentsFT.CompIsent('Compressor_example', Medium=mediaFT.Syncrude(), eff=0.7, F0_n=100, 
        F0_Xi=mediaFT.Syncrude()._reference_X, F0_T=473.15, F0_P=3.0, F1_P=6.0)
    """
    def __init__(self, name:str, Medium:mediaFT.Medium_Syncrude, eff:float=1,
                F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0, 
                F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, F1_T:float=298.15, F1_P:float=3.0, 
                 ):
        super().__init__(name, Medium=Medium, eff=eff)   # FT flanges with a phase replace the CSL ones
        
        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P, phase='Gas')
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P, phase='Gas')  
        
 








class Flash(ModelBase):
    """
    Vapour-liquid flash drum for FT media with Raoult's law.

    The hydrocarbons C2+ form an ideal liquid solution (y_i P = x_i Psat_i), water condenses as a separate
    pure liquid (y_w P = Psat_w), and the permanent gases (H2, CO, CO2, N2, CH4) do not dissolve. The vapour
    pressures are the ones behind the FT liquid properties (``thermoFT.vapour_pressures``), so phase split
    and enthalpies are consistent. For dissolved gases or non-ideal liquids use ``CSL.separation.Flash``
    with real-fluid media.

    Parameters:
        name:   objects name
        Medium: FT medium (``mediaFT.*``)
        T:      drum temperature [K]; Q: heat duty [kJ/s] (Q=0 adiabatic). Neither: isothermal at the feed T
        P:      drum pressure [bar]; otherwise the feed pressure minus ``dP``

    Ports: F0 (feed) -> F1 (vapour, phase 'Gas'), F2 (liquid: hydrocarbons + water, phase 'Liquid')
    Properties: beta (vapour fraction), water_liquid (mol/s of liquid water), Q, Exloss, Sgen

    Example:
        drum = componentsFT.Flash('drum', mediaFT.Syncrude(), T=310.0, P=20.0)
        drum.F0.set(n=10.0, Xi=x_syncrude, T=500.0, P=20.0)
        drum.run_sol()
    """

    def __init__(self, name:str, Medium:mediaFT.Medium_Syncrude, T:float|None=None, P:float|None=None,
                 Q:float|None=None, dP:float=0.0):
        if T is not None and Q is not None:
            raise ValueError(f"{name}: give either the drum temperature T or the heat duty Q, not both")
        super().__init__(dict(name=name, Medium=Medium, dP=dP))
        self.T_spec, self.P_spec, self.Q_spec = T, P, Q
        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, phase='Gas')
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium, phase='Gas')
        self.F2 = interFT.Flange(name + '.F2', Medium=Medium, phase='Liquid', n=0.0)
        sp = [str(x) for x in Medium.species]
        self._water = sp.index('H2O') if 'H2O' in sp else None
        self.beta, self.water_liquid = 1.0, 0.0

    def _split(self, T:float, P:float) -> None:
        n = self.F0.n*np.maximum(np.asarray(self.F0.Xi, dtype=float), 0.0)
        psat = thermoFT.vapour_pressures(T, self.F0.Medium.species)
        v, liq, w = thermoFT.raoult_split(n, psat, P, self._water)
        liq = liq.copy()
        if self._water is not None:
            liq[self._water] += w
        nV, nL = float(v.sum()), float(liq.sum())
        self.beta, self.water_liquid = nV/max(nV + nL, 1e-300), w
        for f, flows, total in ((self.F1, v, nV), (self.F2, liq, nL)):
            f.n = total
            f.P, f.T = P, T
            if total > 0:
                f.Xi = flows/total

    @property
    def Q(self):
        return self.F1.n*self.F1.h + self.F2.n*self.F2.h - self.F0.n*self.F0.h

    @property
    def Exloss(self):
        from ..CSL.exergy import T0
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex + self.Q*(1 - T0/self.F1.T)

    @property
    def Sgen(self):
        return self.F1.n*self.F1.s + self.F2.n*self.F2.s - self.F0.n*self.F0.s - self.Q/self.F1.T

    def run_sol(self):
        P = self.P_spec if self.P_spec is not None else self.F0.P - self.dP
        if self.Q_spec is None:
            T = self.T_spec if self.T_spec is not None else self.F0.T
            self._split(T, P)
            return solvers.trivial_solution([T])

        def residual(T):
            self._split(T, P)
            return (self.Q - self.Q_spec)/max(self.F0.n, 1e-300)
        return solvers.solve_scalar(residual, 200.0, max(self.F0.T, 200.0) + 500.0, x0=self.F0.T, name=self.name)
