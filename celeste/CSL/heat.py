'''
Heat Flow related objects i.e. sources, sinks and pipes
'''

from . import interfaces, ModelBase, media
from .. import solvers
import numpy as np
from numpy.typing import NDArray as npArray
from typing import Callable, override, Literal 

class HexBase(ModelBase):
    """      
    Base Heat exchanger where no reaction is occuring. 
    Two streams are entering and two streams are coming out (hot and cold). 
    The outlet n, Xi, h, cp, g, s are determined. 
    The outlet temperatures and the heat of the Hex object are calculated for a given efficiency.
    IMPORTANT: the inlet hot mol fraction H0_Xi and the inlet and outlet temperatures T must be given manually.
    
    Parameters:
        name: objects name
        MediumH: media of the hot stream
        MediumC: media of the cold stream
    
    Attributes: 
        H0.T: inlet hot temperature [k]
        H1.T: outlet hot temperature [k]
                
    Methods: 
        Eqs: the outlet hot temperature and heat are determined. 
        f_sol: contains the variable to be optimized by the solver. Return: the "Eqs" method
        run_sol: contains the optimizer function and guesses values. Returns: the solution of the optimization
    
    Notes:  
        The size of H0_Xi, H1_Xi and C0_Xi and C1_Xi must be the same.
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.
        Reference: ???????

    # Examples:
    #     1) H1 = Hex('H1', MediumH=media.Air(), MediumC=media.Air(),)

    """

    def __init__(self, name:str, MediumH0:media.Medium_Base, MediumC0:media.Medium_Base,
                 MediumH1:media.Medium_Base, MediumC1:media.Medium_Base):
        super().__init__(locals())

        self.H0: interfaces.Flange = interfaces.Flange(name + '.H0', Medium=MediumH0) #H=Hot
        self.H1: interfaces.Flange = interfaces.Flange(name + '.H1', Medium=MediumH1) #C=Cold
        self.C0: interfaces.Flange = interfaces.Flange(name + '.C0', Medium=MediumC0)
        self.C1: interfaces.Flange = interfaces.Flange(name + '.C1', Medium=MediumC1)

        self.__H0: interfaces.Flange = interfaces.Flange(name + '.__H0', Medium=MediumH1)
        self.__C0: interfaces.Flange = interfaces.Flange(name + '.__C0', Medium=MediumC1)

    def Eqs(self):
        self.H1.n = self.H0.n
        self.C1.n = self.C0.n

        self.H1.Xi = self.H0.Xi
        self.C1.Xi = self.C0.Xi
        
        self.H1.P = self.H0.P 
        self.C1.P = self.C0.P

        self.__H0.Medium.PTX = self.H0.P, self.C0.T, self.H0.Xi
        self.__C0.Medium.PTX = self.C0.P, self.H0.T, self.C0.Xi

        self.Eq: npArray[np.float64] = np.array([
            self.H0.h - self.H1.h - self.Q/self.H0.n,
            self.C0.h - self.C1.h + self.Q/self.C0.n,
        ])
        return self.Eq
    
    def heat_and_work(self) -> float:
        """Nothing crosses the unit boundary: ``Q`` is the duty passed from the hot to the cold side."""
        return 0.0

    @property
    def Q(self) -> float:
        return self.epsilon*self.Qmax
    
    @property
    def QmaxH(self) -> float:
        return self.H0.n*(self.H0.h - self.__H0.h)
    
    @property
    def QmaxC(self) -> float:
        return self.C0.n*(self.C0.h - self.__C0.h)

    @property
    def Qmax(self) -> float:
        return self.Cmin*(self.H0.T-self.C0.T)
    
    @property
    def Cmin(self) -> float :
        if np.isclose(np.abs(self.H0.T-self.C0.T),0., 1e-6):
            C=1.
        else: 
            C = np.min([np.abs(self.QmaxH),np.abs(self.QmaxC)])/np.abs(self.H0.T-self.C0.T)
        return C
    
    @property
    def Cmax(self) -> float :
        if np.isclose(np.abs(self.H0.T-self.C0.T),0., 1e-6):
            C=1.
        else:
            C=np.max([np.abs(self.QmaxH),np.abs(self.QmaxC)])/np.abs(self.H0.T-self.C0.T)
        return C
    
    @property
    def Cr(self) -> float:
        return self.Cmin/self.Cmax
    
    @property
    def epsilon(self) -> float:
        return -1.
    
    @property
    def Exloss(self) -> float:
        return self.H0.n*(self.H0.Ex - self.H1.Ex) + self.C0.n*(self.C0.Ex - self.C1.Ex)
    
    @property
    def Sgen(self) -> float:
        return self.H0.n*(self.H1.s - self.H0.s) + self.C0.n*(self.C1.s - self.C0.s)
    
    def f_sol(self, x:npArray[np.float64]) -> npArray[np.float64]:
        self.H1.T = x[0]
        self.C1.T = x[1]
        return self.Eqs()  

    def run_sol(self):
        x0=np.array([self.H0.T,self.C0.T])
        return solvers.solve(self.f_sol, x0, name=self.name)
   
class HexNTU(HexBase):
    """
    Two-stream heat exchanger rated with the effectiveness-NTU method.

    Parameters:
        name:    objects name
        MediumH: media of the hot stream (ports H0 -> H1)
        MediumC: media of the cold stream (ports C0 -> C1)
        UA:      overall conductance in W/K for 'counter-flow' and 'co-flow';
                 for any other HexType it is used directly as the effectiveness (0..1)
        HexType: 'counter-flow', 'co-flow' or anything else for a fixed effectiveness

    Properties:
        NTU, Cr, epsilon: number of transfer units, capacity ratio, effectiveness
        Q: heat transferred from the hot to the cold stream [kJ/s]

    Example:
        hx = HexNTU('hx', MediumH=media.Air(), MediumC=media.Air(), UA=50.)
        hx.H0.T, hx.C0.T = 800., 300.
        hx.run_sol()          # sets hx.H1.T and hx.C1.T
    """

    def __init__(self, name:str, MediumH:media.Medium_Base, MediumC:media.Medium_Base, UA:float=0.99, HexType:str='counter-flow'):
        
        super().__init__(name, MediumH0=MediumH, MediumC0=MediumC, MediumC1=MediumC, MediumH1=MediumH)
        # Inputs are stored as instance attributes for future use.
        self.name: str = name
        self.UA: float = UA
        self.HexType: str = HexType
        
        if HexType=='counter-flow':
            self.eta = self.etaCounter
        elif HexType=='co-flow':
            self.eta = self.etaCo
        else:
            self.eta = self.etaDummy

        # Declarations for the run_sol_fixOutletTemp() method.
        self._C1targ: interfaces.Flange
        self._H1targ: interfaces.Flange
        self.message: str # Variable for storing any non-critical warnings
        self.eps_targ: float

    @property
    def NTU(self) -> float:
        return self.UA*1e-3/self.Cmin
    
    @property
    @override
    def epsilon(self):
        return self.eta(self.Cr, self.NTU)
        
    def etaCounter(self, Cr:float, NTU:float) -> float:
        if Cr == 1.0:
            return NTU/(1+NTU)
        else:
            return (1 - np.exp(-NTU*(1-Cr)))/(1-Cr*np.exp(-NTU*(1-Cr)))
    
    def etaCo(self,Cr:float, NTU:float) -> float:
        return (1 - np.exp(-NTU*(1+Cr)))/(1+Cr)

    def etaDummy(self, Cr:float, NTU:float) -> float:
        return self.UA

    def run_sol_fixOutletTemp(self, Ttarg:float, stream: Literal['hot','cold'], eps_max:float=0.999):
        """
        Runs the HEX to get a desired outlet temperature (or as close to it as possible). Manipulates the UA parameter.
        """

        # Initialise the __H0 and __C0 outlet flanges for calculating Qmax, etc.
        _ = self.Eqs()

        # Check inputs
        if not (0 <= eps_max <= 0.999):
            raise ValueError(f'In {self.name}.fixOutletTemp(): invalid eps_max provided. Must be between 0 and 0.999')
        if stream not in ('cold','hot'):
            raise ValueError(f'In {self.name}.fixOutletTemp(): invalid stream provided. Must be "hot" or "cold"')

        # Calculate maximum effectiveness for co-flow (cannot approach 1).
        if self.HexType == 'co-flow':
            eps_max_co = 1 / (1+self.Cr)
            eps_max = min(eps_max, eps_max_co*eps_max) # eps_max_co cannot be reached either. So a value approaching that must be used. Here, eps_max itself is chosen as an arbitrary approach factor.

        ## Calculate the demanded (or max. possible) effectiveness    
        if stream == 'cold':
            if Ttarg <= self.C0.T:
                eps_targ = np.nan
                self.message = 'TCtarg =< TCin. No heat transferred.' # Can replace this variable with a warning output if desired.
            else:
                self._C1targ = interfaces.Flange(name=self.name+'.cold_targ_flange', Medium=self.C0.Medium, Xi=self.C0.Xi, n=self.C0.n, P=self.C0.P, T=Ttarg)
                eps_targ = self.C0.n*(self._C1targ.h-self.C0.h) / self.Qmax
        elif stream == 'hot':
            if Ttarg >= self.H0.T:
                eps_targ = np.nan
                self.message = 'THtarg >= THin. No heat transferred.' # Can replace this variable with a warning output if desired.
            else:
                self._H1targ = interfaces.Flange(name=self.name+'.hot_targ_flange', Medium=self.H0.Medium, Xi=self.H0.Xi, n=self.H0.n, P=self.H0.P, T=Ttarg)
                eps_targ = self.H0.n*(self.H0.h-self._H1targ.h) / self.Qmax

        if eps_targ > 0: # If heat transfer can take place
            eps_final = min(eps_targ, eps_max)
            self.message = f'Desired effectiveness = {eps_targ}, Max effectiveness = {eps_max} ---> Applied effectiveness = {eps_final}' # Can replace this variable with a warning output if desired.
        elif eps_targ <=0: # Why would this happen?????
            raise ValueError(f'In {self.name}: desired effectiveness is non-positive ({eps_targ}) despite corrections. Check for phase changes or extreme temperatures. Ttarg for {stream} outlet = {Ttarg}K.')
        else: # If heat transfer cannot take place (TCtarg <= TCin, or THtarg >= THin)
            eps_final = 0.

        self.eps_targ = eps_targ

        ## Calculate the appropriate self.UA for the calculated effectiveness
        if self.HexType == 'co-flow':
            self.UA = -self.Cmin * 1e3 * np.log(1 - eps_final*(1+self.Cr)) / (1+self.Cr)
        elif self.HexType == 'counter-flow':
            if self.Cr == 1.0:
                self.UA = self.Cmin*1e3*eps_final/(1-eps_final)
            else:
                self.UA = -self.Cmin * 1e3 * np.log((1-eps_final)/(1-eps_final*self.Cr)) / (1-self.Cr)
        else:    
            self.UA = eps_final

        return self.run_sol()
        


class Heater(ModelBase):
    """
    Heater where no reaction is occuring. 
    One stream is entering and one stream is coming out. 
    The outlet n, Xi, h, cp, g, s are determined. 
    The thermal exergy of the heater object is calculated.
    IMPORTANT: the inlet and outlet temperatures and pressures must be given manually.
    
    Parameters:
        name: objects name
        Medium0: media of the inlet stream
        Medium1: media of the outlet stream
        F0_n: inlet mol flow in [mol/s]. Defaults to 1.0
        F1_n: outlet mol flow in [mol/s]. Defaults to 1.0
        F0_Xi: inlet mol fraction of every component in Medium1. Defaults to None  
        F1_Xi: outlet mol fraction of every component in Medium2. Defaults to None   
        F0_T: inlet temperature in [K]. Defaults to 298.15 
        F1_T: outlet temperature in [K]. Defaults to 298.15
        F0_P: inlet pressure in [bar]. Defaults to 1.0
        F1_P: outlet pressure in [bar]. Defaults to 1.0 
    
    Attributes: 
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]    
        mw: Molar mass [g/mol]
        F0.Ex: inlet exergy flow in [kJ/s]
        F1.Ex: outlet exergy flow in [kJ/s]
        Exth: Thermal exergy [kJ/s]
        
    Methods: 
        Eqs: the outlet mol and Xi are determined. Returns: None
        MBcheck: the mass flows for every flange and the mass balance difference are calculated. Returns: in [g/s]  #We don't use float anymore rigth? 
        ExBs: calls the exergy method on each flange. The ideal and real work are calculated. Returns: None
        run_sol: empty method. Returns: the system of equations in the "Eqs" method
    
    Notes:  
        The size of F0_Xi and F1_Xi must be the same.
        Accept change of phase only for water.   
        If no composition is given then the object will take the reference composition in the given Medium. For more
        details go to the interfaces documentation.
        Reference: ???????

    Examples:
        1) C1 = Heater('C1', Medium0=media.Air(), Medium1=media.Air(),
                 F0_n=1, F0_Xi=np.array([0.60, 0.40]), F0_T=298.15, F0_P=1,
                 F1_n=1, F1_Xi=None, F1_T=800., F1_P=1,
                 )

        [in]    C1.Q
        [out]   15.52 [kj/s]
    
    """
    
    def __init__(self, name:str, Medium0:media.Medium_Base, Medium1:media.Medium_Base|None=None, **legacy):
        Medium1 = Medium0 if Medium1 is None else Medium1
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium0)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium1)
        self._init_state(legacy)
        
    def Eqs(self):
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        self.F1.P = self.F0.P

    @property
    def Q(self) -> float:
        return (self.F1.n*self.F1.h - self.F0.n*self.F0.h)
    
    @property
    def Exth(self) -> float:
        if self.Q < 0:
            th = self.Q*(1-298.15/self.F1.T) 
        else:
            th = self.Q # heaters as electrical work
        return th
    
    @property
    def Exloss(self) -> float:
        return self.F0.Ex*self.F0.n - self.F1.Ex*self.F1.n + self.Exth
    
    @property
    def Sgen(self) -> float:
        if self.Q < 0:
            sb = self.F1.s*self.F1.n - self.F0.s*self.F0.n - self.Q/self.F1.T 
        else:
            # Q in as electrical work
            sb = self.F1.s*self.F1.n - self.F0.s*self.F0.n
        return sb
    
    def run_sol(self):
        # forward run
        self.Eqs()


def heater_to(name: str, Medium: media.Medium_Base, T, P, MediumOut=None):
    """A ``Heater`` (or cooler) from ``Medium`` to ``MediumOut`` (default the same) whose outlet is at T [K] and P [bar]."""
    unit = Heater(name, Medium0=Medium, Medium1=MediumOut or Medium)
    unit.F1.T, unit.F1.P = T, P
    return unit
