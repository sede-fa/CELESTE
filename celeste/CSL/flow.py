'''
Flow related objects i.e. sources, sinks and pipes
'''
from . import interfaces, ModelBase
from .. import solvers
import numpy as np

import numpy.typing as npt


class SourceM(ModelBase):
    '''
    1+nSpecies unknowns
    '''
    def __init__(self, name, Medium, F1_n:float=1., F1_Xi:npt.NDArray[float] | None =None, F1_T:float=298.15, F1_P:float=1.):
        super().__init__(locals())
        self.name = f'{name}(({name}))'
            
        self.F1 = interfaces.Flange(self.name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P)        
        
class Sink(ModelBase):
    '''
    2*(1+nSpecies) unknowns
    '''
    def __init__(self, name, Medium, **legacy):
        super().__init__(locals())

        self.name = f'{name}[({name})]'
        self.F0 = interfaces.Flange(self.name + '.F0', Medium=Medium)
        self._init_state(legacy)
    
class Splitter(ModelBase):
    
    """Splits F0 into F1 (fraction ``Split``) and F2 (the rest) at the inlet state."""

    def __init__(self, name, Medium, Split=0.5, **legacy):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium)
        self._split = Split
        # former keywords T= and P= set all three ports
        T, P = legacy.pop('T', None), legacy.pop('P', None)
        self._init_state(self._spread(legacy, ['F0', 'F1', 'F2'], T=T, P=P))

    @property
    def split(self) -> float:
        return self._split

    @split.setter
    def split(self, val):
        self._split=val
        
    def Eqs(self):
        self.F1.Medium.PTX = self.F0.Medium.PTX  
        self.F2.Medium.PTX = self.F0.Medium.PTX 
        self.F1.n = self.split*self.F0.n
        self.F2.n = (1-self.split)*self.F0.n
    
    @property
    def Exloss(self):
        return self.F0.n*self.F0.Ex - self.F2.n*self.F2.Ex - self.F1.n*self.F1.Ex 
    
    @property
    def Sgen(self):
        return self.F2.s*self.F2.n + self.F1.s*self.F1.n - self.F0.s*self.F0.n
    
    def run_sol(self): 
        return self.Eqs()

    
class Mixer(ModelBase):
    _inlet_ports = ('F0', 'F1')   # two feeds, one product (F2)
    
    """Adiabatic mixer of the feeds F0 and F1 into F2 at the pressure of F0."""

    def __init__(self, name, Medium, **legacy):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium)
        # former keywords T= and P= set all three ports
        T, P = legacy.pop('T', None), legacy.pop('P', None)
        self._init_state(self._spread(legacy, ['F0', 'F1', 'F2'], T=T, P=P))
        
    def Eqs(self):
        self.F2.n = (self.F0.n + self.F1.n)
        self.F2.Xi = (self.F1.Xi*self.F1.n + self.F0.Xi*self.F0.n)/self.F2.n
        self.F2.P = self.F0.P 
        self.Eq = np.array([self.Q/self.F2.n])
    
        return self.Eq

    @property
    def Q(self):
        return self.F2.h*self.F2.n - (self.F1.h*self.F1.n + self.F0.h*self.F0.n)

    @property  
    def Exloss(self):
        return (self.F0.n*self.F0.Ex + self.F1.n*self.F1.Ex) - self.F2.n*self.F2.Ex 
    
    @property
    def Sgen(self):
        return self.F2.s*self.F2.n - self.F1.s*self.F1.n - self.F0.s*self.F0.n 
    
    def f_sol(self, x):
        self.F2.T = x[0]
        return self.Eqs()

    def run_sol(self):
        """Adiabatic mixing: the outlet temperature follows from the energy balance."""
        if np.isclose(self.F0.T, self.F1.T, rtol=1e-12, atol=0.0):   # a coarser test makes T_out jump
            self.f_sol([self.F0.T])
            return solvers.trivial_solution([self.F0.T])

        # the outlet temperature lies between the inlet temperatures (5 % margin for non-ideal cases)
        T_min = min(self.F0.T, self.F1.T) * 0.95
        T_max = max(self.F0.T, self.F1.T) * 1.05
        x0 = (self.F0.T * self.F0.n + self.F1.T * self.F1.n) / (self.F0.n + self.F1.n)
        return solvers.solve(self.f_sol, [x0], bounds=([T_min], [T_max]), tol=1e-5, name=self.name)
