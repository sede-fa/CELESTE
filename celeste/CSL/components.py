import numpy as np
from numpy.typing import NDArray as npArray
from . import ModelBase, interfaces, media
from .. import solvers
from .._deprecation import deprecated

class GPUbase(ModelBase):
    def __init__(self, name, eff, args):
        super().__init__(locals())
        
class GPU_1S(GPUbase):
    """      
    Black box object where the outlet conditions are prescribed and no reaction is occuring. 
    One stream is entering and one stream is coming out. 
    The outlet n, Xi, h, cp, g, s are determined. 
    The work of the GPU_1S object is calculated for a given efficiency.
    IMPORTANT: the inlet and outlet temperatures and pressures must be given manually.
    
    Parameters:
        name: objects name
        Medium1: media of the inlet stream
        Medium2: media of the outlet stream
        eff: work efficiency of the GPU_1S. Must be 0 < eff <= 1. Defaults to 1
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium1. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium2. Defaults to None   
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: outlet temperature in [K]. Defaults to 298.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: outlet pressure in [bar]. Defaults to 1.0 
    
    Properties: 
        Wid: ideal GPU work in [kJ/s]
        W: real GPU work in [kJ/s]

    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Notes:  
        The size of F0_Xi and F1_Xi must be the same.
        Accept change of phase only for water.   
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.

    Examples:
        1) with water change of phase
        GPU = Components.GPU_1S('GPU_example', Medium1=media.Fuel_CH4(), Medium2=media.Fuel_CH4L(), eff=0.7, F0_n=100, 
        F0_Xi=np.array([0.1,1e-8,1e-8,1e-8,0.9,1e-8]), F0_T=873.15, F0_P=3.0, F1_T=673.15, F1_P=1.0)
    """
    def __init__(self, name:str, Medium1:media.Medium_Base, Medium2:media.Medium_Base, eff:float=0, 
                 F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0, 
                 F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, F1_T:float=298.15, F1_P:float=1.0,
                 ): 

        super().__init__(name, eff, locals())

        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium1, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P) 
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium2, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P) 
        
    def Eqs(self)->None:
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi  
    
    @property
    def Wid(self):
        return self.F1.Ex*self.F1.n - self.F0.n*self.F0.Ex
    
    @property
    def W(self):
        if self.eff>0:
            return float(np.where(self.Wid > 0, self.Wid/self.eff, self.Wid*self.eff))
        else:
            return 0.
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.Ex*self.F1.n #+ self.W
    
    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n
              
    def run_sol(self):
        return self.Eqs() 

class GPU_2S(GPUbase):
    """
    Black box object where the outlet conditions are prescribed and no reaction is occuring. 
    One stream is entering (flange F0) and two streams (flanges F1 and F2) are coming out. 
    For each outlet streams n, Xi, h, cp, g, s, mw are determined. 
    The work of the GPU_2S object is calculated for a given efficiency.
    
    Parameters:
        name: objects name
        Medium: media of the streams
        idxSep: index of the component that goes to the flange F2. Takes one value 0<=idxSep<=(Medium.species.size-1).
                Defaults to 4 (water index in media.Fuel_CH4())
        eff: work efficiency of the GPU_2S. Must be 0 < eff <= 1. Defaults to 1
    
    Properties: 
        Wid: ideal GPU work in [kJ/s]
        W: real GPU work in [kJ/s]

    Methods: 
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        run_sol: the two outlet mol flows and compositions are determined. Returns: None

    Notes:  
        The size of F0_Xi, F1_Xi and F2_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.

    Examples:
        1) Separation of water
        GPU = Components.GPU_2S('GPU_example', Medium=media.Fuel_CH4(), idxSep=4, eff=0.7)
    """
    def __init__(self, name, Medium, idxSep=4, eff=1):
        super().__init__(name, eff, locals())

        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium)
        self.idxSep = idxSep
        self.Medium = Medium

        X2 = np.zeros(self.Medium.nX)
        X2[self.idxSep] = 1.
        self.F2.Xi = X2

    def run_sol(self):
        self.F2.n = self.F0.Xi[self.idxSep]*self.F0.n
        self.F1.n = self.F0.n - self.F2.n
        
        self.F1.Xi = (self.F0.Xi*self.F0.n - self.F2.Xi*self.F2.n)/self.F1.n 

        self.F1.P = self.F0.P
        self.F2.P = self.F0.P

        self.F1.T = self.F0.T
        self.F2.T = self.F0.T 
    
    @property
    def Wid(self):
        return self.F2.n*self.F2.Ex + self.F1.n*self.F1.Ex - self.F0.n*self.F0.Ex

    @property
    def W(self):
        return float(np.where(self.Wid > 0, self.Wid/self.eff, self.Wid*self.eff))
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex
    
    @property
    def Sgen(self):
        return self.F2.n*self.F2.s + self.F1.n*self.F1.s - self.F0.n*self.F0.s

class GPU_2SV2(GPUbase): 
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
        run_sol: the two outlet mol flows and compositions are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]

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
    def __init__(self, name:str, Medium:media.Medium_Base, Sep:npArray[np.float64]|None=None, eff:float=1,
                 F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0
                 ):
        super().__init__(name, eff, locals())
        
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium)
        self.Sep = Sep
        self.Medium = Medium
          
    def run_sol(self):
        self.F1.n = np.sum(self.F0.n*self.F0.Xi*self.Sep)
        self.F1.Xi = self.F0.n*self.F0.Xi*self.Sep/self.F1.n if self.F1.n >=1e-10 else np.zeros(self.F0.Xi.size)
        self.F2.n = np.sum(self.F0.n*self.F0.Xi*(np.ones(self.Medium.nX) - self.Sep))
        self.F2.Xi =  self.F0.n*self.F0.Xi*(np.ones(self.Medium.nX) - self.Sep)/self.F2.n if self.F2.n >=1e-10 else np.zeros(self.F0.Xi.size)

        self.F1.P = self.F0.P
        self.F2.P = self.F0.P

        self.F1.T = self.F0.T
        self.F2.T = self.F0.T 
        
    @property
    def Wid(self):
        return (self.F2.n*self.F2.Ex + self.F1.n*self.F1.Ex) - self.F0.n*self.F0.Ex
    
    @property
    def W(self):
        if self.eff==0.:
            return 0.
        else:
            return float(np.where(self.Wid > 0, self.Wid/self.eff, self.Wid*self.eff))
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex
    
    @property
    def Sgen(self):
        return self.F1.s*self.F1.n + self.F2.s*self.F2.n - self.F0.s*self.F0.n

class GPU_3S(GPUbase):
    """
    Black box object where the outlet conditions are prescribed and no reaction is occuring. 
    One stream is entering (flange F0) and three streams (flanges F1, F2 and F3) are coming out. 
    For each outlet streams n, Xi, h, cp, g, s, mw are determined. 
    The work of the GPU_3S object is calculated for a given efficiency.
    IMPORTANT: the inlet and the three outlet temperatures, pressures, and the composition of the flange F3 
    must be given manually.
    
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
        F3_Xi: flange F3 mol fraction of every component in Medium. Defaults to np.array([0.,0.,0.,0.,1.,0.]) 
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: flange F1 temperature in [K]. Defaults to 298.15
        F2_T: flange F2 temperature in [K]. Defaults to 298.15
        F3_T: flange F3 temperature in [K]. Defaults to 298.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: flange F1 pressure in [bar]. Defaults to 1.0
        F2_P: flange F2 pressure in [bar]. Defaults to 1.0
        F3_P: flange F3 pressure in [bar]. Defaults to 1.0
    
    Attributes: 
        Eq: equation to be solved
        Wid: ideal GPU work in [kJ/s]
        W: real GPU work in [kJ/s]

    Methods: 
        Eqs: the three outlet mol flows and, the flanges F1 and F2 composition are determined. Returns: "Eq" attribute
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        f_sol: contains the variable to be optimized by the solver. Return: the "Eqs" method
        run_sol: contains the optimizer function and guesses values. Returns: the solution of the optimization
    
    Notes:  
        The size of F0_Xi, F1_Xi, F2_Xi and F3_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.
    
    Examples:
        1) Separation of H2 to flange F1, CH4 to flange F2 and water to flange F3
        GPU = Components.GPU_3S('GPU_example', Medium=media.Fuel_CH4(), sep=np.array([1,0,1,1,-1,1]), eff=0.7, F0_n=100, 
        F0_Xi=np.array([0.1,0.1, 1e-8,1e-8,0.8,1e-8]), F0_T=873.15, F0_P=3.0, F1_T=673.15, F1_P=1.0, 
        F2_T=350.15, F2_P=5.0, F3_Xi=np.array([0.,0.,0.,0.,1.,0.]), F3_T=673.15, F3_P=1.0,)
        
        2) how sep works --> if the value given for a specific component is equals to 0.9:
        0.9 of this component goes to flange F1
        (1 - 0.9)=0.1 of this component goes to flange F2
        -1: completely goes to flange F3
    """
    def __init__(self, name:str, Medium:media.Medium_Base, sep:npArray[np.float64]|None=None, eff:float=1.,
                 F0_n:float=1.0, F0_Xi:npArray[np.float64]|None=None, F0_T:float=298.15, F0_P:float=1.0, 
                 F1_n:float=1.0, F1_Xi:npArray[np.float64]|None=None, F1_T:float=298.15, F1_P:float=1.0,
                 F2_n:float=1.0, F2_Xi:npArray[np.float64]|None=None, F2_T:float=298.15, F2_P:float=1.0, 
                 F3_n:float=1.0, F3_Xi:npArray[np.float64]|None=np.array([0.,0.,0.,0.,1.,0.]), F3_T:float=298.15, F3_P:float=1.0, 
                ):       
        super().__init__(name, eff, locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P) 
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=F2_T, P=F2_P) 
        self.F3 = interfaces.Flange(name + '.F3', Medium=Medium, n=F3_n, Xi=F3_Xi, T=F3_T, P=F3_P) 
        self.Medium = Medium
        self.sep = sep
        
    def Eqs(self):
        sep = np.asarray(self.sep, dtype=float)
        idx = np.flatnonzero(sep == -1)
        if idx.size != 1:
            raise ValueError(f"{self.name}: 'sep' must mark exactly one component with -1 (the one sent to F3), got {sep}")
        sep1 = np.where(sep >= 0, sep, 0)
        self.F3.n = float(self.F0.n*self.F0.Xi[idx[0]])
        self.F1.n = np.sum(self.F0.n*self.F0.Xi*sep1)
        self.F1.Xi = self.F0.n*self.F0.Xi*sep1/self.F1.n
        self.F2.n = self.F0.n - self.F1.n - self.F3.n
        self.F2.Xi = (self.F0.n*self.F0.Xi - (self.F1.n*self.F1.Xi + self.F3.n*self.F3.Xi))/self.F2.n     
    
    @property
    def Wid(self):
        return (self.F3.Ex*self.F3.n + self.F2.n*self.F2.Ex + self.F1.n*self.F1.Ex) - self.F0.n*self.F0.Ex

    @property
    def W(self):
        return float(np.where(self.Wid > 0, self.Wid/self.eff, self.Wid*self.eff))
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex - self.F3.n*self.F3.Ex
    
    @property
    def Sgen(self):
        return self.F1.s*self.F1.n + self.F2.s*self.F2.n + self.F3.s*self.F3.n - self.F0.s*self.F0.n
    
    def run_sol(self):
        return self.Eqs()

class Compressor(ModelBase):
    """
    base.ModelBase object with one flange F0 in and one flange F1 out. 
    This is the base Compressor object from which other related objects could extent. No methods are included.
    
    Parameters:
        name: objects name
        Medium: media of the streams
        eff: work efficiency of the Compressor object. Must be 0 < eff <= 1. Defaults to 1
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium. Defaults to None   
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: outlet temperature in [K]. Defaults to 398.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: outlet pressure in [bar]. Defaults to 3.0 

    Attributes: 
        F0: flange F0
        F1: flange F1
    """
    _default_state = {'F0': dict(n=1.0, T=298.15, P=1.0), 'F1': dict(n=1.0, T=398.15, P=3.0)}

    def __init__(self, name:str, Medium:media.Medium_Base, eff:float=1.0, **legacy):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self._init_state(legacy)

class CompIsot(Compressor):
    """
    Isothermal compressor extending from the Compressor class. 
    One stream is entering and one stream is coming out. 
    The outlet n, Xi, h, cp, g, s are determined.
    The work of the CompIsot object is calculated isothermally.
    IMPORTANT: the inlet temperature and pressure, and the outlet pressure must be given manually.
    
    Parameters:
        name: objects name
        Medium: media of the streams
        eff: work efficiency of the CompIsot object. Must be 0 < eff <= 1. Defaults to 1
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
        W: work in [kJ/s]

    Methods: 
        Eqs: the outlet mol flow, composition, temperature, work and heat are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        run_sol: empty method. Returns: the system of equations in the "Eqs" method

    Notes:  
        The size of F0_Xi and F1_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.

    Examples:
        1) 
        Compressor = Components.CompIsot('Compressor_example', Medium=media.Fuel_CH4(), F0_n=100, 
        F0_Xi=np.array([0.1,1e-8,1e-8,1e-8,0.9,1e-8]), F0_T=473.15, F0_P=3.0, F1_P=6.0)

    """
    _default_state = {'F0': dict(n=1.0, T=298.15, P=1.0), 'F1': dict(n=1.0, T=298.15, P=3.0)}

    def __init__(self, name:str, Medium:media.Medium_Base, eff:float=1., **legacy):
        super().__init__(name, Medium=Medium, eff=eff, **legacy)
        self.R = 8.314462618e-3
        
    def Eqs(self)->None:
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        self.F1.T = self.F0.T
    
    @property
    def W(self):
        return self.F0.n*self.F0.T*self.R*np.log(self.F1.P/self.F0.P)/self.eff

    @property
    def Q(self):
        return -self.W

    @property
    def Exth(self):
        return self.Q*(1 - 298.15/self.F0.T)
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.W + self.Exth

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n - self.Q/self.F0.T
    
    def run_sol(self):
        return self.Eqs()
 
class CompIsent(Compressor):
    """
    Isentropic compressor extending from the Compressor Class. 
    One stream is entering and one stream is coming out. 
    The outlet n, Xi, h, cp, g, s, T are determined.
    The real work of the CompIsent object is calculated for a given efficiency.  
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
        W: real work in [kJ/s]
        Exloss: exergy loss in the object [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol flow and composition are determined. Returns: "Eq" attribute
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: float in [g/s]
        f_sol: contains the variable to be optimized by the solver. Return: the "Eqs" method
        run_sol: contains the optimizer function and guesses values. Returns: the solution of the optimization

    Notes:  
        The size of F0_Xi and F1_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.

    Examples:
        1) 
        Compressor = Components.CompIsent('Compressor_example', Medium=media.Fuel_CH4(), eff=0.7, F0_n=100, 
        F0_Xi=np.array([0.1,1e-8,1e-8,1e-8,0.9,1e-8]), F0_T=473.15, F0_P=3.0, F1_P=6.0)

    """
    def __init__(self, name:str, Medium:media.Medium_Base, eff:float=1, **legacy):
        super().__init__(name, Medium=Medium, eff=eff, **legacy)
        self.R = 8.314462618e-3
        
    def Eqs(self):
        cpa = np.average([self.F0.cp, self.F1.cp])
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        if self.F1.P > self.F0.P:
            eff_ratio = 1/self.eff
        else:
            eff_ratio = self.eff
        self.Eq = self.F1.T - (self.F0.T + (self.F0.T*(self.F1.P/self.F0.P)**(self.R/cpa) - self.F0.T)*eff_ratio )        
        return self.Eq

    @property
    def W(self):
        return self.F0.n*(self.F1.h - self.F0.h)
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.W

    @property
    def Exeff(self):
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + self.W)

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n
    
    def f_sol(self, x):
        self.F1.T = x[0]
        return self.Eqs()
        
    def run_sol(self):
        # ideal-gas isentropic estimate with the inlet heat capacity as initial guess
        x0 = self.F0.T*(self.F1.P/self.F0.P)**(self.R/self.F0.cp)
        return solvers.solve(self.f_sol, [x0], bounds=([1.0], [1e4]), name=self.name)

class ASU(ModelBase):
    """
    A simple air separation unit that seperates air into Nitrogen and Oxygen.
    Air stream enters and 2 streams exit, which are 100% Nitrogen and 100% Oxygen.
    Outlet stream n, h, cp, g, s, mw are calculated
    Work done in separation of N2 and O2 are calculated based on a paper, Morgan[2013],
    where an ASU consumes 108 kWh/tN2 @8 bar
    IMPORTANT: Inlet flow, temperature and pressure are to be specified.
    F1 is pure Nitrogen, F2 is pure Oxygen.
   
    Parameters:
        name: objects name
        Medium: media of inlet stream. Defaults to media.Air()
        n: Inlet mole flow in [mol/s]. Defaults to 1.0
        T: Inlet temperature in [K]. Defaults to 300
        P: Inlet pressure in [bar]. Defaults to 8 
        Power: Power consumption in [kWh/tN2]. Defaults to 108

    Attributes:
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        W: ASU work in [kJ/s]
    
    Methods:
        Eqs: the outlet mols, Xi and work are determined.
        MBCheck: Mass flows for every flnage and mass balance difference are calculated.
        ExBs: Calls the exergy method on each flange. 
        run_sol returns system of equation in "Eqs" mehod

    Examples:
        1) Separation of air
        ASU = airseparator.ASU('ASU_example', F0_n=1013.4, P=8, T=400)
    """
    _default_state = {'F0': dict(n=1.0, Xi=[0.21, 0.79], T=298.15, P=1.0),
                      'F1': dict(P=8.0), 'F2': dict(P=8.0)}

    def __init__(self, name:str, Medium:media.Medium_Base=media.Air(), Power:float=108, **legacy):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)  # air
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)  # nitrogen
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium)  # oxygen
        # former keywords: T, n, Xi of the feed; F1_P is the product pressure
        T, n, Xi = (legacy.pop(k, None) for k in ('T', 'n', 'Xi'))
        if 'F1_P' in legacy:
            legacy.setdefault('F2_P', legacy['F1_P'])
        self._init_state(self._spread(legacy, ['F0'], T=T, n=n, Xi=Xi))

    def Eqs(self):
        self.F1.T = self.F0.T
        self.F2.T = self.F0.T
        self.F2.P = self.F1.P
        self.F1.n = self.F0.n*self.F0.Xi[1] 
        self.F2.n = self.F0.n - self.F1.n 
        self.F1.Xi = np.array([0, 1])
        self.F2.Xi = np.array([1, 0])
        self.W = self.Power*self.F1.m*3600*1e-6 # kJ/s 
    
    @property
    def Q(self):
        return -self.W
        
    @deprecated('use MBcheck()')
    def MBCheck(self):
        return self.MBcheck()

    @property
    def Exth(self):
        return self.Q*(1 - (298.15/self.F1.T))
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex + self.W + self.Exth

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n + self.F2.s*self.F2.n - self.F0.s*self.F0.n - self.Q/self.F1.T

    def run_sol(self):
        return self.Eqs()

class Pump(ModelBase): 
    '''
    Isentropic pump (adiabatic). Used for water or methanol+water mixture
    One stream is entering and one stream is coming out. 
    The outlet n, Xi, h, cp, g, s, T are determined.
    The real work of the Pump object is calculated for a given efficiency.
    Important: when working with Fuel_CH4L pure water must be added
    when working with Fuel

    Parameters:
        name: objects name
        Medium: media of the streams
        n: inlet and outlet mol flow in [mol/s]. Defaults to 1.0
        Xi: inlet and outlet  mol fraction of every component in Medium. Defaults to np.array([0.,0.,0.,0.,1.,0.])
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: outlet temperature in [K]. Defaults to 368.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: outlet pressure in [bar]. Defaults to 3.0 
    
    Attributes: 
        R: ideal gas constant in [kJ/mol K]
        W: real pump work in [kJ/s]
        Eq: equation to be solved
        Exloss: exergy loss in the object [kJ/s]
        Exeff: exergy efficiency of the object. It goes between 0 and 1 

    Methods: 
        Eqs: the outlet mol flow and composition are determined. Returns: "Eq" attribute
        rho_water: determines the density of water. Returns: the density in [Kg/m3]
        rho_MeOH : determines density of Methanol. Returns: density in [Kg/m3]
        rho_avr : determines average density of the mixture. Returns: density in [Kg/m3]
        f_sol: contains the variable to be optimized by the solver. Return: the "Eqs" method
        run_sol: contains the optimizer function and guesses values. Returns: the solution of the optimization

    Notes:  
        The Pump object can work for other chemicals if the density of them are loaded (methanol in this case)

    Examples:
        1) 
        Pump = Components.Pump('pump_example', Medium=media.Fuel_CH4L(), F0_n=100, 
        F0_Xi=np.array([1e-8,1e-8,1e-8,1e-8,1.0,1e-8]), F0_T=298.15, F0_P=1.0, F1_P=2.0)
    '''
     
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_CH4L(), n:float=1.0, 
                 Xi=np.array([0.,0.,0.,0.,1.,0.]), F0_T:float=298.15, F0_P:float=1.0, 
                 F1_T:float=368.15, F1_P:float=3.0, eff:float=1
                ):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=n, Xi=Xi, T=F0_T, P=F0_P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=n, Xi=Xi, T=F1_T, P=F1_P)
        self.eff = eff
        self.R = 8.314462618e-3
        
    def Eqs(self):
        self.F1.n = self.F0.n 
        self.F1.Xi = self.F0.Xi
        self.Eq =  np.array([self.Q])
        return self.Eq

    @property
    def Q(self):
        return self.F1.n*self.F1.h - (self.W + self.F0.n*self.F0.h) 

    @property
    def W(self):
        if self.F1.P > self.F0.P:
            eff_ratio = 1/self.eff
        else:
            eff_ratio = self.eff
        return (self.F0.m*1e-1*(self.F1.P - self.F0.P)/self.F0.Medium.rho)*eff_ratio

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.W 

    @property
    def Exeff(self):
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + self.W)
    
    @property
    def Sgen(self):
        # taking out the pressure change in the entropy calculation
        return self.F1.n*self.F1.s- self.F0.n*self.F0.s 
    
    def f_sol(self, x):
        self.F1.T = x[0]
        return self.Eqs()
        
    def run_sol(self):
        return solvers.solve(self.f_sol, [self.F0.T], bounds=([1.0], [1e4]), name=self.name)
            
class Valve(ModelBase):
    """
    Isothermal valve model with no heat or work production
    """
    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_CH4(), n:float=1.0, 
                 Xi:npArray[np.float64]|None=np.array([0.,0.,0.,0.,1.,0.]), F0_T:float=298.15, F0_P:float=1.0, 
                 F1_T:float=368.15, F1_P:float=3.0,
                ):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=n, Xi=Xi, T=F0_T, P=F0_P)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=n, Xi=Xi, T=F1_T, P=F1_P)
        
    def Eqs(self):
        self.F1.n = self.F0.n 
        self.F1.Xi = self.F0.Xi
        self.F1.T = self.F0.T
        self.W = 0

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex

    @property
    def Exeff(self):
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex)

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n 
    
    def run_sol(self):
        return self.Eqs()
class DAC(ModelBase):
    """
    A Direct air capture unit that seperates CO2 from air. Assuming 100% capture rate.
    Air stream enters and 2 streams exit, which are 100% CO2 and other components of air(N2, O2). #Ar to be added later in the component list
    Outlet stream n, h, cp, g, s, mw are calculated
    Work done in separation of CO2 from air is calculated based on a paper by Climeworks, https://doi.org/10.3389/fclim.2019.00010 
    where a DAC consumes 2000 kWh/tCO2 @slightly above atmospheric pressure outlet CO2 , (1600 kWh/tCO2 Thermal and 400 kWh/tCO2 Electrical)
                         2447.8 kWh/tCO2 @15 Mpa according to Carbon Engineering, Keith, https://doi.org/10.1016/j.enconman.2023.117687 

    IMPORTANT: Inlet flow, temperature and pressure are to be specified.
    F1 is Air out, F2 is pure CO2.
   
    Parameters:
        name: objects name
        Medium: media of inlet stream. Defaults to media.Air()
        n: Inlet mole flow in [mol/s]. Defaults to 1.0
        T: Inlet temperature in [K]. Defaults to 298.15
        P: Inlet pressure in [bar]. Defaults to 1.0
        Power: Power consumption in [kWh/tCO2]. Defaults to 2000 

    Attributes:
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        W: DAC work in [kJ/s]
    
    Methods:
        Eqs: the outlet mols, Xi and work are updated based on new inputs.
        MBCheck: Mass flows for every flange and mass balance difference are calculated.
        run_sol updates system of equations in "Eqs" method based on new inputs

    Examples:
        1) Direct air capture to extract CO2
        DAC = components.DAC('DAC_example', F0_n=1013.4, P=1.2, T=400)
    """
    _default_state = {'F0': dict(n=1.0, Xi=[0.2114, 0.7883, 0.0003], T=298.15, P=1.0),
                      'F1': dict(P=1.0), 'F2': dict(P=1.0)}

    def __init__(self, name:str, Medium:media.Medium_Base=media.Atm_Air(), Power_el:float=400, Power_th:float=1600,
                 **legacy):  # 2447.8 @150 bar
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)  # air in
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)  # air out
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium)  # CO2
        T, n, Xi = (legacy.pop(k, None) for k in ('T', 'n', 'Xi'))
        if 'F1_P' in legacy:
            legacy.setdefault('F2_P', legacy['F1_P'])
        self._init_state(self._spread(legacy, ['F0'], T=T, n=n, Xi=Xi))

    def Eqs(self):
        self.F1.T = self.F0.T
        self.F2.T = self.F0.T
        self.F2.P = self.F1.P
        self.F2.n = self.F0.n*self.F0.Xi[2] 
        self.F1.n = self.F0.n - self.F2.n 
        self.F1.Xi = np.array([self.F0.Xi[0]/(1-self.F0.Xi[2]), self.F0.Xi[1]/(1-self.F0.Xi[2]), 0.])
        self.F2.Xi = np.array([0., 0., 1.0])

    @property
    def W(self): 
        return self.Power_el*self.F2.m*3600*1e-6 # kJ/s 
    
    @property
    def Q(self):
        return self.Power_th*self.F2.m*3600*1e-6
        
    @deprecated('use MBcheck()')
    def MBCheck(self):
        return self.MBcheck()

    @property
    def Exth(self):
        return self.Q*(1 - (298.15/self.F2.T))
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex + self.W + self.Exth

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n + self.F2.s*self.F2.n - self.F0.s*self.F0.n - self.Q/self.F2.T

    def run_sol(self):
        return self.Eqs()

class ChgMedium(ModelBase):
    """
    Hands a stream over from one medium to another (for example from a mixture
    medium to a pure-water medium after a separator). Flow, temperature and
    pressure are passed on; the composition is mapped by species name, where a
    species and its condensed form count as the same ('H2O' <-> 'H2O(L)').
    Species of the outlet medium that the inlet does not contain get zero, and
    the outlet composition is normalised.

    Parameters:
        name:    objects name
        Medium0: inlet medium. Defaults to media.Fuel_CH4()
        Medium1: outlet medium. Defaults to media.Water()

    Ports: F0 -> F1
    """
    def __init__(self, name:str, Medium0:media.Medium_Base|None=None, Medium1:media.Medium_Base|None=None):
        Medium0 = media.Fuel_CH4() if Medium0 is None else Medium0
        Medium1 = media.Water() if Medium1 is None else Medium1
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium0)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium1)

    @staticmethod
    def _base(species:str) -> str:
        return str(species)[:-3] if str(species).endswith('(L)') else str(species)

    def Eqs(self):
        spX0 = {}
        for sp, x in zip(self.F0.Medium.species, self.F0.Xi):
            spX0[self._base(sp)] = spX0.get(self._base(sp), 0.0) + (x if x > 1e-8 else 0.0)
        Xi = np.array([spX0.get(self._base(sp), 0.0) for sp in self.F1.Medium.species])
        if Xi.sum() <= 0:
            raise ValueError(f"{self.name}: the outlet medium has no species in common with the inlet")
        self.F1.n = self.F0.n
        self.F1.Xi = Xi/np.sum(Xi)
        self.F1.T = self.F0.T
        self.F1.P = self.F0.P

    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex

    @property
    def Sgen(self):
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n

    def run_sol(self):
        self.Eqs()
