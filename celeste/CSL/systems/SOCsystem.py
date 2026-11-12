'''
System model Example with SOC system, with mass flows and current as inputs
'''


import numpy as np
from celeste.CSL import flow, media, electrochem, SysBase, heat, interfaces


class SOCBasicSys(SysBase):
    def __init__(self, name, nf0, Xf0, na0, Xa0, I, Ncells, T_soc=1073.15, P_soc=1, cellArea=127.8):
        super().__init__(locals()) #name,
        
        self.sf1 = flow.SourceM(name + '.sf1', Medium=media.Fuel_CH4(), F1_n=nf0, F1_Xi=Xf0, F1_T=T_soc, F1_P=P_soc) 
        self.sa1 = flow.SourceM(name + '.sa1', Medium=media.Air(), F1_n=na0, F1_Xi=Xa0, F1_T=T_soc, F1_P=P_soc) 
        self.soc = electrochem.SOC(name + '.soc', MediumF=media.Fuel_CH4(), MediumA=media.Air(), T_soc0=T_soc, T_soc1=T_soc, P_soc=P_soc, I=I, cellArea=cellArea, Ncells=Ncells) 
        self.sf2 = flow.Sink(name + '.sf2', Medium=media.Fuel_CH4()) 
        self.sa2 = flow.Sink(name + '.sa2', Medium=media.Air())  
        
        self.getObjs()
        
    def seq_sol(self, store=False):
               
        self.connect_seq(self.sf1.F1, self.soc.F0, store=store)
        self.connect_seq(self.sa1.F1, self.soc.A0, store=store)
        
        self.soc._sol = self.soc.run_sol()
        
        self.connect_seq(self.soc.F1, self.sf2.F0, store=store)
        self.connect_seq(self.soc.A1, self.sa2.F0, store=store)
      
class SOCsysRecupe(SysBase):
    def __init__(self, name, nf0=1, na0=1e-4, T_soc=1073.15, I=0., Ncells=1000, etaHex=0.99, cellArea=100.):
        super().__init__(locals())

        self.F0 = interfaces.Flange(name+'.F0', media.Fuel_CH4(), n=nf0, Xi=np.array([0.1, 1e-5, 1e-5, 1e-5, 0.9, 1e-5]), T=120+273.15, P=1.)
        self.A0 = interfaces.Flange(name+'.A0', media.Air(), n=na0, Xi=np.array([0.21, 0.79]), T=298.15, P=1.)

        self.HEX_F = heat.HexNTU(name+'.HEX_F', MediumC=media.Fuel_CH4(), MediumH=media.Fuel_CH4(), UA=etaHex, HexType='other')
        self.HT_F0 = heat.Heater(name+'.HT_F1', Medium0=media.Fuel_CH4(), Medium1=media.Fuel_CH4())
        self.HT_F0.F1.set(T=T_soc)

        self.HEX_A = heat.HexNTU(name+'.HEX_A', MediumC=media.Air(), MediumH=media.Air(), UA=etaHex, HexType='other')
        self.HT_A0 = heat.Heater(name+'.HT_A1', Medium0=media.Air(), Medium1=media.Air())
        self.HT_A0.F1.set(T=T_soc)

        self.soc = electrochem.SOC(name+'.soc', MediumF=media.Fuel_CH4(), MediumA=media.Air(), T_soc0=T_soc, T_soc1=T_soc, I=I, Ncells=Ncells, cellArea=cellArea)

        self.F1 = interfaces.Flange(name+'.F1', media.Fuel_CH4())
        self.A1 = interfaces.Flange(name+'.A1', media.Air())
        
        self.getObjs()

    def seq_sol(self, store=False):
        self.connect_seq(self.F0, self.HEX_F.C0, store=store)
        self.HEX_F._sol = self.HEX_F.run_sol()

        self.connect_seq(self.A0, self.HEX_A.C0, store=store)
        self.HEX_A._sol = self.HEX_A.run_sol()

        self.connect_seq(self.HEX_F.C1, self.HT_F0.F0, store=store)
        self.HT_F0._sol = self.HT_F0.run_sol()

        self.connect_seq(self.HEX_A.C1, self.HT_A0.F0, store=store)
        self.HT_A0._sol = self.HT_A0.run_sol()

        self.connect_seq(self.HT_F0.F1, self.soc.F0, store=store)
        self.connect_seq(self.HT_A0.F1, self.soc.A0, store=store)
        self.soc._sol = self.soc.run_sol()

        self.connect_seq(self.soc.F1, self.HEX_F.H0, store=store)
        self.HEX_F._sol = self.HEX_F.run_sol()
        self.connect_seq(self.HEX_F.H1, self.F1, store=store)

        self.connect_seq(self.soc.A1, self.HEX_A.H0, store=store)
        self.HEX_A._sol = self.HEX_A.run_sol()
        self.connect_seq(self.HEX_A.H1, self.A1, store=store)

    def run_sol(self):
        self.seq_sol()
        self.seq_sol() # run twice to update the temperature in the HEX model (can be changed to a solver instead)
