'''
System model Example with SOC system, with mass flows and current as inputs
'''


import numpy as np
from celeste.CSL import flow, media, electrochem, SysBase, heat, interfaces


class SOCBasicSys(SysBase):
    def __init__(self, name: str, nf0, Xf0, na0, Xa0, I, Ncells, T_soc: float = 1073.15, P_soc: int = 1, cellArea: float = 127.8):
        super().__init__(locals()) #name,
        
        self.sf1 = flow.SourceM(name + '.sf1', Medium=media.Fuel_CH4(), F1_n=nf0, F1_Xi=Xf0, F1_T=T_soc, F1_P=P_soc) 
        self.sa1 = flow.SourceM(name + '.sa1', Medium=media.Air(), F1_n=na0, F1_Xi=Xa0, F1_T=T_soc, F1_P=P_soc) 
        self.soc = electrochem.SOC(name + '.soc', MediumF=media.Fuel_CH4(), MediumA=media.Air(), T=T_soc, I=I, cellArea=cellArea, Ncells=Ncells) 
        self.sf2 = flow.Sink(name + '.sf2', Medium=media.Fuel_CH4()) 
        self.sa2 = flow.Sink(name + '.sa2', Medium=media.Air())  
        
        self.getObjs()
        
    def seq_sol(self, store: bool = False) -> None:
               
        self.connect_seq(self.sf1.F1, self.soc.F0, store=store)
        self.connect_seq(self.sa1.F1, self.soc.A0, store=store)
        
        self.solve(self.soc)
        
        self.connect_seq(self.soc.F1, self.sf2.F0, store=store)
        self.connect_seq(self.soc.A1, self.sa2.F0, store=store)
      
class SOCsysRecupe(SysBase):
    """
    SOC with recuperation: the feeds are heated by the outlets in two effectiveness-type exchangers (``etaHex``) and
    then by heaters to ``T_soc``. The exchangers need the outlet temperatures of the cell, so ``run_sol`` passes
    through the flowsheet twice.
    """
    def __init__(self, name: str, nf0: int = 1, na0: float = 1e-4, T_soc: float = 1073.15, I: float = 0., Ncells: int = 1000, etaHex: float = 0.99, cellArea: float = 100.):
        super().__init__(locals())
        fuel, air = media.Fuel_CH4(), media.Air()

        self.F0 = interfaces.Flange(name+'.F0', fuel, n=nf0, Xi=np.array([0.1, 1e-5, 1e-5, 1e-5, 0.9, 1e-5]), T=120+273.15, P=1.)
        self.A0 = interfaces.Flange(name+'.A0', air, n=na0, Xi=np.array([0.21, 0.79]), T=298.15, P=1.)

        self.HEX_F = heat.HexNTU(name+'.HEX_F', MediumC=fuel, MediumH=fuel, UA=etaHex, HexType='other')
        self.HT_F0 = heat.heater_to(name+'.HT_F1', fuel, T_soc, 1.0)
        self.HEX_A = heat.HexNTU(name+'.HEX_A', MediumC=air, MediumH=air, UA=etaHex, HexType='other')
        self.HT_A0 = heat.heater_to(name+'.HT_A1', air, T_soc, 1.0)
        self.soc = electrochem.SOC(name+'.soc', MediumF=fuel, MediumA=air, T=T_soc, I=I, Ncells=Ncells, cellArea=cellArea)

        self.F1 = interfaces.Flange(name+'.F1', fuel)
        self.A1 = interfaces.Flange(name+'.A1', air)
        self.getObjs()

    def seq_sol(self, store: bool = False) -> None:
        self.connect_seq(self.F0, self.HEX_F.C0, store=store)
        self.connect_seq(self.A0, self.HEX_A.C0, store=store)
        self.solve(self.HEX_F, self.HEX_A)

        self.connect_seq(self.HEX_F.C1, self.HT_F0.F0, store=store)
        self.connect_seq(self.HEX_A.C1, self.HT_A0.F0, store=store)
        self.solve(self.HT_F0, self.HT_A0)

        self.connect_seq(self.HT_F0.F1, self.soc.F0, store=store)
        self.connect_seq(self.HT_A0.F1, self.soc.A0, store=store)
        self.solve(self.soc)

        self.connect_seq(self.soc.F1, self.HEX_F.H0, store=store)
        self.connect_seq(self.soc.A1, self.HEX_A.H0, store=store)
        self.solve(self.HEX_F, self.HEX_A)
        self.connect_seq(self.HEX_F.H1, self.F1, store=store)
        self.connect_seq(self.HEX_A.H1, self.A1, store=store)

    def run_sol(self):
        self.seq_sol()
        self.seq_sol()
