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
    
    def __init__(self, name: str, n=None, Xi=None, T0: float = 298.15, P0: float = 1., T_ael: float = 351.15, P_ael: float = 1., cellArea: int = 800, Ncells: int = 556, Sep=None): 
        super().__init__(locals())
        
        self.F1 = interfaces.Flange(name + '.F1', n=n, Xi=Xi, Medium=media.Media_AEL(), T=T_ael, P=P_ael)         
        
        self.ael = electrochem.AEL(name + '.ael', Medium=media.Media_AEL(), T=T_ael, P=P_ael, A=cellArea, Ncells=Ncells) 
        self.HT_O1 = heat.heater_to(name + '.HT_O1', media.Media_AEL(), T0, P0)
        self.GPU = components.GPU_3S(name + '.GPU', Medium=media.Media_AEL(), sep=Sep)
        for outlet in (self.GPU.F1, self.GPU.F2, self.GPU.F3):
            outlet.T, outlet.P = T_ael, P_ael
        self.GPU.F3.Xi = np.array([0, 0, 0, 1])
        self.HT_O2 = heat.heater_to(name + '.HT_O2', media.Media_AEL(), T0, P0)
        self.CD_W1 = heat.heater_to(name + '.CD_W1', media.Media_AEL(), T0, P0)
        
        self.F2 = interfaces.Flange(name + '.F2', Medium=media.Media_AEL())
        self.O1 = interfaces.Flange(name + '.O1', Medium=media.Media_AEL()) # O2 from ael
        self.O2 = interfaces.Flange(name + '.O2', Medium=media.Media_AEL()) # O2 from GPU
        self.WWater = interfaces.Flange(name + '.WWater', Medium=media.Media_AEL()) #H2O from condenser
        
        self.getObjs()
        
    def seq_sol(self, store: bool = False) -> None:
        
        self.connect_seq(self.F1, self.ael.F0, store=store)
        self.solve(self.ael)
        
        self.connect_seq(self.ael.F1, self.GPU.F0, store=store)
        self.connect_seq(self.ael.F2, self.HT_O1.F0, store=store) 
        
        self.solve(self.GPU, self.HT_O1)
        self.connect_seq(self.HT_O1.F1, self.O1, store=store) # O2 from AEL to the system

        self.connect_seq(self.GPU.F1, self.F2, store=store) #  Dry H2 to the system
        self.connect_seq(self.GPU.F2, self.HT_O2.F0, store=store)
        self.connect_seq(self.GPU.F3, self.CD_W1.F0, store=store)
        
        self.solve(self.HT_O2, self.CD_W1)
        
        self.connect_seq(self.HT_O2.F1, self.O2, store=store) # O2 from GPU to the system
        self.connect_seq(self.CD_W1.F1, self.WWater, store=store)  # Waste water to the system

    def run_sol(self):
        self.seq_sol()

class _SOCPlant(SysBase):
    """
    SOC with its feed flanges F1 (fuel) and A1 (air) and an air cooler to T0; the air leaves at A2. The subclasses
    add the treatment of the fuel side. Run with ``seq_sol`` (the cell and the air side in ``solve_cell``).
    """
    def __init__(self, params, name: str, I, Ncells, T0, T_soc, P_soc, P_air, cellArea, ext):
        super().__init__(params)
        if ext:
            from ..._deprecation import warn_deprecated
            warn_deprecated(f"{type(self).__name__}(ext=True) is deprecated and has no effect: the SOC ideal "
                            "voltage excludes the reforming reactions", stacklevel=3)
        self.F1 = interfaces.Flange(name + '.F1', Medium=media.Fuel_CH4(), T=T_soc, P=P_soc)
        self.A1 = interfaces.Flange(name + '.A1', Medium=media.Air(), T=T_soc, P=P_soc)
        self.soc = electrochem.SOC(name + '.soc', MediumF=media.Fuel_CH4(), MediumA=media.Air(), T=T_soc, I=I,
                                   cellArea=cellArea, Ncells=Ncells)
        self.HT_A2 = heat.heater_to(name + '.HT_A2', media.Air(), T0, P_air)
        self.A2 = interfaces.Flange(name + '.A2', Medium=media.Air())

    def solve_cell(self, store: bool = False):
        """Feed the cell, solve it and cool the air to the system."""
        self.connect_seq(self.F1, self.soc.F0, store=store)
        self.connect_seq(self.A1, self.soc.A0, store=store)
        self.solve(self.soc)
        self.connect_seq(self.soc.A1, self.HT_A2.F0, store=store)
        self.solve(self.HT_A2)
        self.connect_seq(self.HT_A2.F1, self.A2, store=store)                   # air to the system

    def run_sol(self):
        self.seq_sol()


class SOECSys(_SOCPlant):
    """
    SOEC system: the fuel outlet is cooled to ``T_HT2``, the water is separated, condensed and leaves at
    WWater; the dry syngas leaves at F2.
    """
    fuel_cooler = True

    def __init__(self, name: str, T0: float = 298.15, P0: int = 1, I=None, Ncells=None, T_soc: float = 1073.15, P_soc: int = 1, cellArea: float = 127.8,
                 ext: bool = False, T_HT2: float = 378.15):
        super().__init__(locals(), name, I, Ncells, T0, T_soc, P_soc, P_soc, cellArea, ext)
        if self.fuel_cooler:
            self.HT_F2 = heat.heater_to(name + '.HT_F2', media.Fuel_CH4(), T_HT2, P_soc)
        self.GPU1 = components.GPU_2S(name + '.GPU1', Medium=media.Fuel_CH4(), idxSep=4)       # separation of water
        self.HT_W1 = heat.heater_to(name + '.HT_W1', media.Fuel_CH4(), T0, P_soc, MediumOut=media.Fuel_CH4L())
        self.Chm1 = components.ChgMedium(name + '.ChM_1', Medium0=media.Fuel_CH4L(), Medium1=media.Water())
        self.F2 = interfaces.Flange(name + '.F2', Medium=media.Fuel_CH4())
        self.WWater = interfaces.Flange(name + '.WWater', Medium=media.Water())
        self.getObjs()

    def seq_sol(self, store: bool = False) -> None:
        self.solve_cell(store)
        fuel_out = self.soc.F1
        if self.fuel_cooler:
            self.connect_seq(fuel_out, self.HT_F2.F0, store=store)
            self.solve(self.HT_F2)
            fuel_out = self.HT_F2.F1
        self.connect_seq(fuel_out, self.GPU1.F0, store=store)
        self.solve(self.GPU1)
        self.connect_seq(self.GPU1.F1, self.F2, store=store)                    # dry syngas to the system
        self.connect_seq(self.GPU1.F2, self.HT_W1.F0, store=store)
        self.solve(self.HT_W1)
        self.connect_seq(self.HT_W1.F1, self.Chm1.F0, store=store)
        self.solve(self.Chm1)
        self.connect_seq(self.Chm1.F1, self.WWater, store=store)                # waste water to the system


class SOECSysV2(SOECSys):
    """SOECSys without the cooler in front of the separator (the water is separated at the cell outlet temperature)."""
    fuel_cooler = False

    def __init__(self, name: str, T0: float = 298.15, P0: float = 1., I=None, Ncells=None, T_soc: float = 1073.15, P_soc: float = 1., cellArea: float = 127.8, ext: bool = False):
        super().__init__(name, T0=T0, P0=P0, I=I, Ncells=Ncells, T_soc=T_soc, P_soc=P_soc, cellArea=cellArea, ext=ext)


class CO2Sys(_SOCPlant):
    """CO2 SOEC system: the fuel outlet is cooled to ``T_HT2`` and leaves at F2."""
    def __init__(self, name: str, T0: float = 298.15, P0: int = 1, I=None, Ncells=None, T_soc: float = 1073.15, P_soc: int = 1, cellArea: float = 127.8, ext: bool = False,
                 T_HT2: float = 298.15):
        super().__init__(locals(), name, I, Ncells, T0, T_soc, P_soc, P0, cellArea, ext)
        self.HT_F2 = heat.heater_to(name + '.HT_F2', media.Fuel_CH4(), T_HT2, P_soc)
        self.F2 = interfaces.Flange(name + '.F2', Medium=media.Fuel_CH4())
        self.getObjs()

    def seq_sol(self, store: bool = False) -> None:
        self.solve_cell(store)
        self.connect_seq(self.soc.F1, self.HT_F2.F0, store=store)
        self.solve(self.HT_F2)
        self.connect_seq(self.HT_F2.F1, self.F2, store=store)                   # syngas to the system
