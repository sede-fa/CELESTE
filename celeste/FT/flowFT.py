from ..CSL import ModelBase, flow
from . import interfacesFT as interFT


class SourceM(flow.SourceM):  # could inherit from base source

    def __init__(self, name, Medium, F1_n=1, F1_Xi=None, F1_T=298.15, F1_P=1., phase='Gas'):
        super().__init__(name, Medium, F1_n, F1_Xi, F1_T, F1_P)
        self.F1 = interFT.Flange(self.name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=F1_T, P=F1_P, phase=phase)
        
        
class Sink(flow.Sink):  # could inherit from base sink

    def __init__(self, name, Medium, F0_n=1, F0_Xi=None, F0_T=298.15, F0_P=1., phase='Gas'):
        super().__init__(name, Medium, F0_n, F0_Xi, F0_T, F0_P)
        self.F0 = interFT.Flange(self.name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=F0_T, P=F0_P, phase=phase)
        
      
class Splitter(ModelBase):  # could inherit from base splitter
    
    def __init__(self, name, Medium, Split=0.5, T=298.15, P=1,
                 F0_n=1, F0_Xi=None,  phase0='Gas',
                 F1_n=1, F1_Xi=None,  phase1='Gas',
                 F2_n=1, F2_Xi=None,  phase2='Gas'              
                ):
        super().__init__(locals())
        
        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P, phase=phase0)
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P, phase=phase1)
        self.F2 = interFT.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=T, P=P, phase=phase2)
        
    def Eqs(self):
        self.F1.Medium.PTX = self.F0.Medium.PTX  
        self.F2.Medium.PTX = self.F0.Medium.PTX
        self.F1.n = self.Split*self.F0.n
        self.F2.n = (1-self.Split)*self.F0.n
    
    def ExBs(self):
        self.ExB = self.F2.n*self.F2.Ex + self.F1.n*self.F1.Ex - self.F0.n*self.F0.Ex
    
    def run_sol(self):
        return self.Eqs()
    
class Mixer(ModelBase):  # could inherit from base mixer
    _inlet_ports = ('F0', 'F1')   # two feeds, one product (F2)
    
    def __init__(self, name, Medium, T=298.15, P=1., 
                 F0_n=1, F0_Xi=None, phase0='Gas',
                 F1_n=1, F1_Xi=None, phase1='Gas',
                 F2_n=1, F2_Xi=None, phase2='Gas',                
                ):
        super().__init__(locals())
        
        self.F0 = interFT.Flange(name + '.F0', Medium=Medium, n=F0_n, Xi=F0_Xi, T=T, P=P, phase=phase0)
        self.F1 = interFT.Flange(name + '.F1', Medium=Medium, n=F1_n, Xi=F1_Xi, T=T, P=P, phase=phase1)
        self.F2 = interFT.Flange(name + '.F2', Medium=Medium, n=F2_n, Xi=F2_Xi, T=T, P=P, phase=phase2)
        
    def Eqs(self):
        self.F2.n = (self.F0.n + self.F1.n)
        self.F2.Medium.PTX = self.F0.P, self.F0.T, (self.F0.Xi*self.F0.n + self.F1.Xi*self.F1.n)/self.F2.n
       
    def ExBs(self):
        self.ExB = self.F2.n*self.F2.Ex - (self.F0.n*self.F0.Ex + self.F1.n*self.F1.Ex)
    
    def run_sol(self):
        
        return self.Eqs()