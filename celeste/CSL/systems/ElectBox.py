import numpy as np
from celeste.CSL import media, interfaces, electrochem, heat, components, SysBase

'''
1. AEL system with heaters and separators
2. SOEC system with heaters and separators
3. SOEC system with separators. Only considers the heating for the outlet air but not for the outlet fuel stream
4. CO2-SOEC system with heaters and separators
'''

class AELSys(SysBase):
    '''
    alkaline electrolyser model
    the current density in A/cm^2 must be given. The inlet and outlet flow are being calculated
    '''
    
    def __init__(self, name, n=None, Xi=None, T0=298.15, P0=1., T_ael=351.15, P_ael=1., cellArea=800, Ncells=556, Sep=None): 
        super().__init__(locals())
        
        self.F1 = interfaces.Flange(name + '.F1', n=n, Xi=Xi, Medium=media.Media_AEL(), T=T_ael, P=P_ael)         
        
        self.ael = electrochem.AEL(name + '.ael', Medium=media.Media_AEL(), T=T_ael, P=P_ael, A=cellArea, Ncells=Ncells) 
        self.HT_O1 = heat.Heater(name + '.HT_O1', Medium0=media.Media_AEL(), Medium1=media.Media_AEL())
        self.HT_O1.F1.set(T=T0, P=P0)
        self.GPU = components.GPU_3S(name + '.GPU', Medium=media.Media_AEL(), sep=Sep)
        self.GPU.F1.set(T=T_ael, P=P_ael)
        self.GPU.F2.set(T=T_ael, P=P_ael)
        self.GPU.F3.set(Xi=np.array([0,0,0,1]), T=T_ael, P=P_ael)
        self.HT_O2 = heat.Heater(name + '.HT_O2', Medium0=media.Media_AEL(), Medium1=media.Media_AEL())
        self.HT_O2.F1.set(T=T0, P=P0)
        self.CD_W1 = heat.Heater(name + '.CD_W1', Medium0=media.Media_AEL(), Medium1=media.Media_AEL())
        self.CD_W1.F1.set(T=T0, P=P0)
        
        self.F2 = interfaces.Flange(name + '.F2', Medium=media.Media_AEL())
        self.O1 = interfaces.Flange(name + '.O1', Medium=media.Media_AEL()) # O2 from ael
        self.O2 = interfaces.Flange(name + '.O2', Medium=media.Media_AEL()) # O2 from GPU
        self.WWater = interfaces.Flange(name + '.WWater', Medium=media.Media_AEL()) #H2O from condenser
        
        self.getObjs()
        
    def seq_sol(self, store=False):
        
        self.connect_seq(self.F1, self.ael.F0, store=store)
        self.ael._sol = self.ael.run_sol()
        
        self.connect_seq(self.ael.F1, self.GPU.F0, store=store)
        self.connect_seq(self.ael.F2, self.HT_O1.F0, store=store) 
        
        self.GPU._sol = self.GPU.run_sol()
        self.HT_O1._sol = self.HT_O1.run_sol()
        self.connect_seq(self.HT_O1.F1, self.O1, store=store) # O2 from AEL to the system

        self.connect_seq(self.GPU.F1, self.F2, store=store) #  Dry H2 to the system
        self.connect_seq(self.GPU.F2, self.HT_O2.F0, store=store)
        self.connect_seq(self.GPU.F3, self.CD_W1.F0, store=store)
        
        self.HT_O2._sol = self.HT_O2.run_sol()
        self.CD_W1._sol = self.CD_W1.run_sol()
        
        self.connect_seq(self.HT_O2.F1, self.O2, store=store) # O2 from GPU to the system
        self.connect_seq(self.CD_W1.F1, self.WWater, store=store)  # Waste water to the system

    def run_sol(self):
        self.seq_sol()

class SOECSys(SysBase):
    def __init__(self, name, T0=298.15, P0=1, I=None, Ncells=None, T_soc=1073.15, P_soc=1, cellArea=127.8, 
                 ext=False, T_HT2=378.15): 
        super().__init__(locals())
                
        self.F1 = interfaces.Flange(name + '.F1', Medium=media.Fuel_CH4(), T=T_soc, P=P_soc) 
        self.A1 = interfaces.Flange(name + '.A1', Medium=media.Air(), T=T_soc, P=P_soc)      
        
        self.soc = electrochem.SOC(name + '.soc', MediumF=media.Fuel_CH4(), MediumA=media.Air(), T_soc0=T_soc, 
                                   T_soc1=T_soc, P_soc=P_soc, I=I, cellArea=cellArea, Ncells=Ncells, extent=ext) 
        self.HT_A2 = heat.Heater(name + '.HT_A2', Medium0=media.Air(), Medium1=media.Air())
        self.HT_A2.F1.set(T=T0, P=P_soc)
        self.HT_F2 = heat.Heater(name + '.HT_F2', Medium0=media.Fuel_CH4(), Medium1=media.Fuel_CH4())
        self.HT_F2.F1.set(T=T_HT2, P=P_soc)
        self.GPU1 = components.GPU_2S(name + '.GPU1', Medium=media.Fuel_CH4(), idxSep=4)  
        self.HT_W1 = heat.Heater(name + '.HT_W1', Medium0=media.Fuel_CH4(), Medium1=media.Fuel_CH4L())
        self.HT_W1.F1.set(T=T0, P=P_soc)
        self.Chm1 = components.ChgMedium(name + '.ChM_1', Medium0=media.Fuel_CH4L(), Medium1=media.Water())
        
        self.F2 = interfaces.Flange(name + '.F2', Medium=media.Fuel_CH4())
        self.A2 = interfaces.Flange(name + '.A2', Medium=media.Air())
        self.WWater = interfaces.Flange(name + '.WWater', Medium=media.Water())
        
        self.getObjs()
        
    def seq_sol(self, store=False):
        
        self.connect_seq(self.F1, self.soc.F0, store=store)
        self.connect_seq(self.A1, self.soc.A0, store=store)
        
        self.soc._sol = self.soc.run_sol()
        
        self.connect_seq(self.soc.A1, self.HT_A2.F0, store=store)
        
        self.HT_A2._sol = self.HT_A2.run_sol()
        self.connect_seq(self.HT_A2.F1, self.A2, store=store) # air to the system
        
        self.connect_seq(self.soc.F1, self.HT_F2.F0, store=store)
        
        self.HT_F2._sol = self.HT_F2.run_sol()     
        self.connect_seq(self.HT_F2.F1, self.GPU1.F0, store=store)
        
        self.GPU1._sol = self.GPU1.run_sol()
        self.connect_seq(self.GPU1.F1, self.F2, store=store) #  Dry syngas to the system
        self.connect_seq(self.GPU1.F2, self.HT_W1.F0, store=store)
        self.HT_W1._sol = self.HT_W1.run_sol()

        self.connect_seq(self.HT_W1.F1, self.Chm1.F0, store=store)  
        self.Chm1._sol = self.Chm1.run_sol()
        self.connect_seq(self.Chm1.F1, self.WWater, store=store) # Waste water to the system

    def run_sol(self):
        self.seq_sol()

class SOECSysV2(SysBase):
    def __init__(self, name, T0=298.15, P0=1., I=None, Ncells=None, T_soc=1073.15, P_soc=1., 
                 cellArea=127.8, ext=False): 
        super().__init__(locals())
        
        self.F1 = interfaces.Flange(name + '.F1', Medium=media.Fuel_CH4(), T=T_soc, P=P_soc) 
        self.A1 = interfaces.Flange(name + '.A1', Medium=media.Air(), T=T_soc, P=P_soc)       
        
        self.soc = electrochem.SOC(name + '.soc', MediumF=media.Fuel_CH4(), MediumA=media.Air(), T_soc0=T_soc, 
                                   T_soc1=T_soc, P_soc=P_soc, I=I, cellArea=cellArea, Ncells=Ncells, extent=ext)    
        self.HT_A2 = heat.Heater(name + '.HT_A2', Medium0=media.Air(), Medium1=media.Air())
        self.HT_A2.F1.set(T=T0, P=P_soc)
        self.GPU1 = components.GPU_2S(name + '.GPU1', Medium=media.Fuel_CH4(), idxSep=4) # separation of water
        self.HT_W1 = heat.Heater(name + '.HT_W1', Medium0=media.Fuel_CH4(), Medium1=media.Fuel_CH4L())  # at T0 and P_soc
        self.HT_W1.F1.set(T=T0, P=P_soc)
        self.Chm1 = components.ChgMedium(name + '.ChM_1', Medium0=media.Fuel_CH4L(), Medium1=media.Water())
        
        self.F2 = interfaces.Flange(name + '.F2', Medium=media.Fuel_CH4())
        self.A2 = interfaces.Flange(name + '.A2', Medium=media.Air())
        self.WWater = interfaces.Flange(name + '.WWater', Medium=media.Water()) 
        
        self.getObjs()
        
    def seq_sol(self, store=False):
        
        self.connect_seq(self.F1, self.soc.F0, store=store)
        self.connect_seq(self.A1, self.soc.A0, store=store)
        
        self.soc._sol = self.soc.run_sol()
        
        self.connect_seq(self.soc.A1, self.HT_A2.F0, store=store)
        
        self.HT_A2._sol = self.HT_A2.run_sol()
        self.connect_seq(self.HT_A2.F1, self.A2, store=store) # air to the system
        
        self.connect_seq(self.soc.F1, self.GPU1.F0, store=store)
        
        self.GPU1._sol = self.GPU1.run_sol()
        self.connect_seq(self.GPU1.F1, self.F2, store=store) #  Dry syngas to the system
        self.connect_seq(self.GPU1.F2, self.HT_W1.F0, store=store)
        self.HT_W1._sol = self.HT_W1.run_sol()

        self.connect_seq(self.HT_W1.F1, self.Chm1.F0, store=store)  
        self.Chm1._sol = self.Chm1.run_sol()
        self.connect_seq(self.Chm1.F1, self.WWater, store=store) # Waste water to the system

    def run_sol(self):
        self.seq_sol()

class CO2Sys(SysBase):
    '''
    CO2 SOEC system
    '''
    def __init__(self, name, T0=298.15, P0=1, I=None, Ncells=None, T_soc=1073.15, P_soc=1, cellArea=127.8, ext=False, T_HT2=298.15): 
        super().__init__(locals())
        
        self.F1 = interfaces.Flange(name + '.F1', Medium=media.Fuel_CH4(), T=T_soc, P=P_soc) # n=nf0, Xi=Xf0,  
        self.A1 = interfaces.Flange(name + '.A1', Medium=media.Air(), T=T_soc, P=P_soc) #n=na0, Xi=Xa0,        
        
        self.soc = electrochem.SOC(name + '.soc', MediumF=media.Fuel_CH4(), MediumA=media.Air(), T_soc0=T_soc, T_soc1=T_soc, P_soc=P_soc, I=I, cellArea=cellArea, Ncells=Ncells, extent=ext) 
        self.HT_A2 = heat.Heater(name + '.HT_A2', Medium0=media.Air(), Medium1=media.Air())
        self.HT_A2.F1.set(T=T0, P=P0)
        self.HT_F2 = heat.Heater(name + '.HT_F2', Medium0=media.Fuel_CH4(), Medium1=media.Fuel_CH4())  #changed to FuelCH4
        self.HT_F2.F1.set(T=T_HT2, P=P_soc)
        
        self.F2 = interfaces.Flange(name + '.F2', Medium=media.Fuel_CH4())
        self.A2 = interfaces.Flange(name + '.A2', Medium=media.Air())
        
        self.getObjs()
        
    def seq_sol(self, store=False):
        
        self.connect_seq(self.F1, self.soc.F0, store=store)
        self.connect_seq(self.A1, self.soc.A0, store=store)
        
        self.soc._sol = self.soc.run_sol()
        
        self.connect_seq(self.soc.A1, self.HT_A2.F0, store=store)
        
        self.HT_A2._sol = self.HT_A2.run_sol()
        self.connect_seq(self.HT_A2.F1, self.A2, store=store) # air to the system
        
        self.connect_seq(self.soc.F1, self.HT_F2.F0, store=store)
        
        self.HT_F2._sol = self.HT_F2.run_sol()     
        self.connect_seq(self.HT_F2.F1, self.F2, store=store) #  Dry syngas to the system
        
    def run_sol(self):
        self.seq_sol()

