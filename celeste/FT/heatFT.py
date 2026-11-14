from ..CSL import heat
from . import interfacesFT as interFT

class Heater(heat.Heater):
    def __init__(self, name,  Medium0, Medium1,   
                 F0_n=1, F0_Xi=None, F0_T=298.15, F0_P=1, phase0='Gas', 
                 F1_n=1, F1_Xi=None, F1_T=298.15, F1_P=1, phase1='Gas',
                 direction='F'): 
        super().__init__(name=name, Medium0=Medium0, Medium1=Medium1)   # FT flanges with a phase replace the CSL ones
        
        self.F0 = interFT.Flange(name + '.F0', Medium=Medium0, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P, phase=phase0)
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium1, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P, phase=phase1)
        self.direction = direction
        
    def Eqs(self): 
        if self.direction in ['B', 'b']: 
            self.F0.n = self.F1.n
            self.F0.Xi = self.F1.Xi
            self.F0.P = self.F1.P
        else:
            self.F1.n = self.F0.n
            self.F1.Xi = self.F0.Xi
            self.F1.P = self.F0.P
                    
    def run_sol(self):
        self.Eqs()
   