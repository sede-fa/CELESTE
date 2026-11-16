from ..CSL import heat
from . import interfacesFT as interFT

class Heater(heat.Heater):
    """Heater of the standard library with FT flanges (with a phase); ``direction='B'`` solves backwards (F1 -> F0).
    Deprecated: the state keywords F0_n, F0_Xi, F0_T, ..., F1_P (set the state on the flanges)."""

    def __init__(self, name, Medium0, Medium1, phase0='Gas', phase1='Gas', direction='F', **legacy):
        super().__init__(name=name, Medium0=Medium0, Medium1=Medium1)   # FT flanges with a phase replace the CSL ones
        self.F0 = interFT.Flange(name + '.F0', Medium=Medium0, phase=phase0)
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium1, phase=phase1)
        self.direction = direction
        self._init_state(legacy)

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
   