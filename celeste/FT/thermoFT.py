from functools import lru_cache

import numpy as np
import pandas as pd

from ..CSL import thermo as th
from ..CSL import exergy as _exergy
from ..jx import nasa as _nasa

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

'''
1. Thermo: calculation of MW, LHV cp, h, s, g of a single component or stream fir FT components
2. ThermoExergyFT: stream exergy for FT media (extends celeste.CSL.exergy.ThermoExergy)
'''

def _VapCalc(T, Tmin, Tmax, B, Tc, n, cpl):
    '''
    Enthalpy and entropy of vaporisation of the hydrocarbons, shape (2, N).

    T > Tmax          : 0
    Tmin <= T <= Tmax : B*(1 - T/Tc)*n
    T < Tmin          : B*(1 - Tmin/Tc)**n + cpl*(T - Tmin)

    NOTE: the in-range branch multiplies by ``n`` whereas the documented Watson
    correlation (see ``Thermo.calVAP``) and the branch below ``Tmin`` raise to
    the power ``n``. The original behaviour is kept unchanged here; see
    docs/known_issues.md before relying on liquid-phase hydrocarbon enthalpies.
    '''
    with np.errstate(invalid='ignore'):
        below = B*(1 - Tmin/Tc)**n + cpl*(T - Tmin)
    hvap = np.where(T > Tmax, 0.0, np.where(T >= Tmin, B*(1 - T/Tc)*n, below))
    return np.stack((hvap, hvap/T))

def _phaseVap(phaseID, T, Tmin, Tmax, B, Tc, n, cpl, NHC):
    if phaseID == 1:
        VAP = _VapCalc(T, Tmin, Tmax, B, Tc, n, cpl)
    else:
        VAP = np.zeros((2, NHC))
    return VAP

def _mtCalcStatic(m, c, thermArr, CNumberi):
    '''[cp, h, s] of the hydrocarbons as linear functions of the carbon number, shape (3, N)'''
    return np.outer(m @ thermArr, CNumberi) + (c @ thermArr)[:, None]

def _thermoArrStat(t):
    return np.array([t*t*t*t, t*t*t, t*t, t, 1.])

@lru_cache(maxsize=1)
def _hc_table():
    file_path = files("celeste.data").joinpath("Hydrocarbons_ChemProp.csv").as_posix()
    df = pd.read_csv(file_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    return df.set_index("Formula")


@lru_cache(maxsize=None)
def _hc_data(species: tuple, hc_species: tuple) -> dict:
    '''Property arrays for a species set (cached: table look-ups are slow compared to the property call).'''
    df = _hc_table()
    data = df.loc[list(species)]
    hc = df.loc[list(hc_species)]
    num = lambda frame, col: frame[col].to_numpy(dtype=float)
    out = dict(
        lhv=num(data, 'LHV_kJ/mol'), MW=num(data, 'MW'), H_N=num(data, 'H_N'),
        HVAP_B=num(hc, 'HVAP_B'), HVAP_n=num(hc, 'HVAP_n'), T_boil=num(hc, 'T_boil_K'), Tc=num(hc, 'Crit_T_K'),
        HVAP_Tmin=num(hc, 'HVAP_Tmin_K'), HVAP_Tmax=num(hc, 'HVAP_Tmax_K'), cpl=num(hc, 'cp_std_liq_kJ/mol*K'),
        C_N=num(hc, 'C_N'),
    )
    for v in out.values():
        v.setflags(write=False)
    return out


class Thermo:
    R = 8.314462618e-3

    @th._classproperty
    def df(cls):
        '''Hydrocarbon property table indexed by formula (loaded on first access).'''
        return _hc_table()

    _m = np.array([
        [1.60208009e-13, -3.85369069e-10, 2.84263828e-7, -2.1076165e-5, 0.0126558917],
        [-2.03518749e-11, 2.39474418e-8, 2.18318243e-5, 5.40208647e-3,-24.959112],
        [-3.21627106e-14, 5.28128653e-11,-4.63466461e-8, 9.28472733e-5, 0.0147243196]
    ])
    _c = np.array([
        [-3.32104578e-13, 7.15023059e-10, -4.77027078e-7, 1.21672696e-4, -0.00329707471],
        [2.12326032e-11,-1.2232199e-8,-6.26259659e-6, 1.17389703e-2,- 44.6550556],
        [-6.42129853e-14, 1.91117309e-10,-1.89411673e-7, 9.19937061e-5,0.136713671]
    ])

    def __init__(self, species, T, Xi, phase='Gas', P=1., T_unit='K'):

        self.species = species
        self._phase = phase

        self._phaseID = 1 if phase == 'liquid' or phase == 'Liquid' else 0

        # separate NASA and HC components
        self._spFuelCH4, self._spHC = self.getSpecies()

        if T_unit in ['C', 'c']:
            T = T + 273.15

        self._T = T

        self._Xi = Xi
        self._P = P
        self.__NFuelCH4 = len(self._spFuelCH4)
        self.__NHC = len(self._spHC)

        data = _hc_data(tuple(self.species), tuple(self._spHC))
        ## LHV at standard conditions
        '''
        from C1 to C4 --> gas phase
        from C5 to C15 --> liquid phase
        '''
        self._lhv_i = data['lhv']
        self._MWi = data['MW']
        self.HNumberi = data['H_N']

        ## HC thermo data
        self.__HVAP_B = data['HVAP_B']
        self.__HVAP_n = data['HVAP_n']
        self.__HVAP_Tb = data['T_boil']
        self.__HVAP_Tc = data['Tc']
        self.__HVAP_Tmin = data['HVAP_Tmin']
        self.__HVAP_Tmax = data['HVAP_Tmax']
        self.__cpl = data['cpl']
        self.CNumberi = data['C_N']

        self.__m = self._m
        self.__c = self._c

        self.thermoArr()

        if self.__NFuelCH4 > 0:
            self.__FuelCH4_Th = th.Thermo(self._spFuelCH4, self._T, np.zeros((self.__NFuelCH4)), self._P)
        else:
            self.__FuelCH4_Th = th.ThermoDum(self._T)

        if self.__NHC > 0:
            self.calcThermoHC()
        else: # temporary
            self.__HC_cp, self.__HC_h, self.__HC_s, self.__HC_g = np.array([]), np.array([]), np.array([]),np.array([])

        self.calcThermoSep()
        self.calcThermoMix()

    ## Methods 
    def getSpecies(self):
        hc = []
        fuelch4 = []
        for species in self.species:
            if species =='H2O' and self._phase == 'Liquid' or self._phase == 'liquid':
                fuelch4.append('H2O(L)')
            elif species in ['H2', 'CO2', 'H2O', 'CO', 'N2', 'H2O(L)', 'CH4', 'O2', 'NH3']: 
                fuelch4.append(species)
            else:                    
                hc.append(species)
        return fuelch4, hc
    
    def getMW(self): # data in HC
        return np.dot(self._MWi, self._Xi)
    
    def calVAP(self): # only for HC
        '''
        Method that returns the enthalpy of vaporization of hydrocarbons.
        - The n oefficient and Tc were taken from "Yaws, Carl. Thermophysical Properties of Chemicals and Hydrocarbons. 2nd Ed., 2014".
        - The HVAP_B coefficient is calculated with the HVAP and the boiling T as: HVAP_B = Hvap_boil/(1 - Tboil/Tc)**n. Hvap_boil was 
        obtained from the same reference.
        - The sensible heat when T < Tmin is accounted. Cp°liquid(cpl) was taken from 
        '''

        vap = _VapCalc(self._T, self.__HVAP_Tmin, self.__HVAP_Tmax, self.__HVAP_B, self.__HVAP_Tc, self.__HVAP_n, self.__cpl)
        return vap 

    def thermoArr(self): 
        self._thermArr = _thermoArrStat(self._T)

    def calcThermoHC(self): 

        VAP = _phaseVap(self._phaseID, self._T, self.__HVAP_Tmin, self.__HVAP_Tmax, self.__HVAP_B, self.__HVAP_Tc, self.__HVAP_n, self.__cpl, self.__NHC)
        self.__HVAP = VAP[0]
        self.__SVAP = VAP[1]

        self.__therm_sep = _mtCalcStatic(self.__m, self.__c, self._thermArr, self.CNumberi) 
        
        self.__HC_cp = self.__therm_sep[0]
        self.__HC_h = self.__therm_sep[1] - self.__HVAP 
        self.__HC_s = self.__therm_sep[2] - self.__SVAP
        self.__HC_g = self.__HC_h - self._T*self.__HC_s   
    
    def calcThermoSep(self):
        
        self.cp_i = np.concatenate((self.__FuelCH4_Th.cp_i, self.__HC_cp)) 
        self.h_i = np.concatenate((self.__FuelCH4_Th.h_i, self.__HC_h)) 
        self.s_i = np.concatenate((self.__FuelCH4_Th.s_i, self.__HC_s)) 
        self.g_i = np.concatenate((self.__FuelCH4_Th.g_i, self.__HC_g)) 
    
    def calcThermoMix(self):
        
        # species below 1e-6 are left out (this also masks hydrocarbons without liquid-phase data)
        X = np.asarray(self._Xi, dtype=float)
        on = X >= 1e-6
        Xa = X[on]
        self.cp = np.sum(Xa*self.cp_i[on])
        self.h = np.sum(Xa*self.h_i[on])
        self.s_T = np.sum(Xa*self.s_i[on])
        self.s = self.s_T - self.R*np.sum(Xa*np.log(Xa*self._P))
        self.g = self.h - self._T*self.s

    @property
    def LHVmix(self): # data in HC
        return np.dot(self._lhv_i, self._Xi)
        
    # LHV of a given specie
    def LHVi(self, specie):
        return self._lhv_i[list(self.species).index(specie)]
    
    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, ph):
        self._phase = ph
        if self.__NHC > 0:
            self.calcThermoHC()
        else: # temporary
            self.__HC_cp, self.__HC_h, self.__HC_s, self.__HC_g = np.array([]), np.array([]), np.array([]),np.array([])
        self.calcThermoSep()
        self.calcThermoMix()
      
    ## Poperties
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, t):
        self._T = t 
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def Xi(self):
        return self._Xi
    
    @Xi.setter
    def Xi(self, x):
        self._Xi = x
        self.calcThermoMix()
        
    @property
    def P(self):
        return self._P
        
    @P.setter
    def P(self, p):
        self._P = p     
        self.__FuelCH4_Th.P = self._P
        self.calcThermoSep()
        self.calcThermoMix()
    
    @property
    def TX(self):
        return self._T, self._Xi
        
    @TX.setter
    def TX(self, tx):
        t, x = tx
        
        self._T = t 
        self._Xi = x
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def TcX(self):
        return self._T-273.15, self._Xi
    
    @TcX.setter
    def TcX(self, tx):
        t,x = tx
            
        self._T = t + 273.15
        self._Xi = x
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def Tc(self):
        return self._T - 273.15
        
    @Tc.setter
    def Tc(self, t):
        self._T = t + 273.15
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def PTX(self):
        return self._P, self._T, self._Xi
        
    @PTX.setter
    def PTX(self, ptx):
        p,t,x = ptx
        
        self._T = t 
        self._P = p
        self._Xi = x
        self.__FuelCH4_Th.P = self._P
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()
        
    @property
    def PTcX(self):
        return self._P, self._T - 273.15, self._Xi
        
    @PTcX.setter
    def PTcX(self, ptx):
        p,t,x = ptx
        
        self._T = t + 273.15
        self._P = p
        self._Xi = x
        self.__FuelCH4_Th.P = self._P
        self.__FuelCH4_Th.T = self._T
        self.thermoArr()
        self.calcThermoHC()
        self.calcThermoSep()
        self.calcThermoMix()

''' 
    @classmethod
    def getPsat(cls, T, species):   
        # returns the saturation pressure in bar. T in Kelvin
        A = cls.df[1]['A'][species]
        B = cls.df[1]['B'][species]
        C = cls.df[1]['C'][species]
        T_C = T - 273.15 # in celcius
        ## hypothetical vapor pressure for non-condensables from "Sun, Chenghao PhD Thesis 2017: Direct syngas-to-fuel:integration of FT
                                                                 synthesis and hydrocraking in micro-structured reactors, page 41"
        if species in ['H2', 'CO', 'H2O', 'N2', 'CH4', 'C2H6', 'C3H8', 'C4H10', 'C5H12']:
            P_sat = A + B*np.exp(-T/C)
        else:
            ## from "Yaws Handbook of vapor pressure Antoine Coeff" 2nd Ed., 2015
            P_sat = (10**(A-B/(T_C+C)))*0.00133322
        return P_sat
    
    @classmethod
    def Psat(cls, T, speciesList):
        Psat = np.zeros(len(speciesList))
        for i, var in enumerate(speciesList):
            Psat[i] = cls.getPsat(T, var)
        return Psat
    
    @classmethod
    def Pbubble(cls, T, Xi, speciesList):
        Pbubble = 0
        for i, var in enumerate(speciesList):
            Pbubble += np.where(Xi[i] > 1e-6,  Xi[i]*cls.getPsat(T, var), 0)
        return Pbubble
    
    @classmethod
    def Pdew(cls, T, Xi, speciesList):
        Pdew = 0
        for i, var in enumerate(speciesList):
            Pdew +=np.where(Xi[i] > 1e-6, Xi[i]/cls.getPsat(T, var), 0)      
        return 1/Pdew
''' 
    
@lru_cache(maxsize=None)
def _reference_ft(species: tuple, T0: float, P0: float, phase: str):
    '''(h_i, s_i) at the dead-state temperature from the FT property model.'''
    ref = Thermo(species=list(species), T=T0, Xi=np.full(len(species), 1.0/len(species)), P=P0, phase=phase)
    return ref.h_i, ref.s_i


class ThermoExergyFT(_exergy.ThermoExergy):
    '''
    Exergy of streams described by the FT property model (``Thermo`` of this module).

    Extends ``celeste.CSL.exergy.ThermoExergy``: only the dead-state species
    properties come from the FT correlations instead of the NASA polynomials.
    '''
    def __init__(self, Medium, Xi=None, T=None, P=None, phase='Gas'):
        self.phase = phase
        super().__init__(Medium, Xi, T, P)

    def reference_props(self, species: tuple):
        return _reference_ft(species, self.T0, self.P0, self.phase)


class ThermoExergy(ThermoExergyFT):
    '''
    Former location of the exergy model, kept for existing code: returns a
    ``ThermoExergyFT`` for FT media and a ``celeste.CSL.exergy.ThermoExergy`` otherwise.
    '''
    def __new__(cls, Medium, Xi=None, T=None, P=None, phase='Gas'):
        if isinstance(Medium, Thermo):
            return ThermoExergyFT(Medium, Xi, T, P, phase)
        return _exergy.ThermoExergy(Medium, Xi, T, P)


_exergy_table = _exergy.exergy_table
standard_chemical_exergy = _exergy.standard_chemical_exergy

# FT media use the FT exergy model
Thermo.ExergyModel = ThermoExergyFT
