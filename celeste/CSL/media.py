'''
Medium objects to store species info
'''
import numpy as np
from . import thermo

from numpy.typing import NDArray as npArray

# 1. Medium_Base      - Base object -> hold the structure
# 2. Medium_NASAbase  - ideal gas / ideal mixture media from NASA polynomials (Air, Fuel_CH4, Steam, ...)
# 3. Medium_CPbase    - CoolProp media (optional dependency)
# 4. Medium_PRbase    - Peng-Robinson real-fluid media with vapour-liquid equilibrium (JAX)



class Medium_Base:
    def __init__(self, name, species:list[str]= [], reference_X=None):
        self.name = name
        self.species = species #empty list
        self.nX = len(self.species) 
        if reference_X is not None or not hasattr(self, '_reference_X'):
            self._reference_X = reference_X
    
    def __repr__(self):
        return f'{self.name}(species={self.species})' 
    
    def get_Idx(self, species:str) -> int:
        """Index of a species in the medium."""
        return list(self.species).index(species)
    

#%% NASA Media
def _rho_ideal_gas(medium):
    """Ideal gas density in kg/m³"""
    return medium.P*1e5*medium.getMW()*1e-3/(medium.R*1e3)/medium.T


def _rho_water_rackett(T):
    """Saturated liquid water density in kg/m³ (Rackett type correlation, Yaws)"""
    return 0.3471*0.274**-(1 - (T/647.13))**0.28571*1e3


def _rho_methanol_rackett(T):
    """Saturated liquid methanol density in kg/m³ (Rackett type correlation, Yaws)"""
    return 0.27197*0.27192**-(1 - (T/512.58))**0.23310*1e3


class Medium_NASAbase(Medium_Base, thermo.Thermo):
    """
    NASA polynomial-based medium.

    Sub-classes only declare ``_default_species`` and ``_default_Xi`` (and
    ``_liquid = True`` for media whose density is that of liquid water):

        class Syngas(Medium_NASAbase):
            _default_species = ['H2', 'CO', 'CO2']
            _default_Xi = np.array([0.6, 0.3, 0.1])
    """
    
    _default_species = None
    _default_Xi = None
    _liquid = False
    
    def __init__(self, name:str|None=None, species=None, T=298.15, Xi:npArray[np.float64]|None=None, P=1.):
        name = type(self).__name__ if name is None else name
        species = species if species is not None else self._default_species
        Xi = Xi if Xi is not None else self._default_Xi
        if Xi is None:
            Xi = getattr(self, '_reference_X', None)
        if species is None or Xi is None:
            raise TypeError(f"{type(self).__name__} needs 'species' and 'Xi' (or class attributes "
                            f"_default_species and _default_Xi)")
        Medium_Base.__init__(self, name, species, reference_X=Xi) 
        thermo.Thermo.__init__(self, species=species, T=T, Xi=Xi, P=P)
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        if cls.__dict__.get('_default_species') is not None:
            species_str = ', '.join(cls._default_species)
            Xi_str = np.array2string(cls._default_Xi, precision=2, separator=', ') if cls._default_Xi is not None else 'None'
            
            base_doc = cls.__doc__ or ''
            cls.__doc__ = f"""{base_doc}
    
            Default Species: {species_str}
            Default Composition: {Xi_str}
            """

    def __copy__(self):
        """A fresh medium of the same kind at its default state (flanges never share a medium object)."""
        Xi = np.array(self._reference_X, dtype=float)
        try:   # keep a species list given to the constructor
            return type(self)(name=self.name, species=list(self.species), Xi=Xi)
        except TypeError:   # sub-class with another constructor signature
            return Medium_NASAbase(self.name, species=self.species, Xi=np.array(self._reference_X, dtype=float))
    
    @property
    def rho(self):
        """Density in kg/m³: ideal gas, or saturated liquid water for ``_liquid`` media"""
        return _rho_water_rackett(self.T) if self._liquid else _rho_ideal_gas(self)

class Fuel_CH4(Medium_NASAbase): #extending medium
    _default_species = ['H2','CH4','CO2','CO','H2O','N2']
    _default_Xi = np.array([0.1, 1e-8, 1e-8, 1e-8, 0.9, 1e-8])
        
class Fuel_CH4L(Medium_NASAbase): #extending medium
    _default_species = ['H2','CH4','CO2','CO','H2O(L)','N2']
    _default_Xi = np.array([0.1, 1e-8, 1e-8, 1e-8, 0.9, 1e-8])
    _liquid = True

class Steam(Medium_NASAbase): #extending medium
    _default_species = ['H2O']
    _default_Xi = np.array([1.0])
        
class Air(Medium_NASAbase): 
    _default_species = ['O2', 'N2']
    _default_Xi = np.array([0.21, 0.79])
        
class Water(Medium_NASAbase):
    _default_species = ['H2O(L)']
    _default_Xi = np.array([1.0])
    _liquid = True
        
class HB(Medium_NASAbase): #extending medium
    _default_species = ['H2','N2','NH3']
    _default_Xi = np.array([0.1, 0.9, 1e-8])

    def __init__(self, name='H_B', **kwargs):
        super().__init__(name, **kwargs)
             
class Steam_Water(Medium_NASAbase): #extending medium
    _default_species = ['H2O', 'H2O(L)']
    _default_Xi = np.array([0.97, 0.03])
    _liquid = True

    def __init__(self, name='Steam', **kwargs):
        super().__init__(name, **kwargs)
        
class Fuel_MeOH(Medium_NASAbase): #extending medium
    _default_species = ['H2','CH4','CO2','CO','H2O','N2', 'CH3OH']
    _default_Xi = np.array([0.2, 1e-8, 0.05, 0.05, 0.3, 1e-8, 0.4])

class Fuel_MeOHL(Medium_NASAbase): #extending medium 
    _default_species = ['H2','CH4','CO2','CO','H2O(L)','N2', 'CH3OH(L)']
    _default_Xi = np.array([0.2, 1e-8, 0.05, 0.05, 0.3, 1e-8, 0.4])

    @property
    def rho(self):
        return 1/(self.Yi[4]/self.rho_water() + self.Yi[6]/self.rho_MeOH())
    
    def rho_water(self):
        # https://booksite.elsevier.com/9780750683661/Appendix_C.pdf, The Yaws handbook of physical properties; T in K
        return _rho_water_rackett(self.T)  # Kg/m3
    
    def rho_MeOH(self):
        return _rho_methanol_rackett(self.T)  # Kg/m3

class Media_AEL(Medium_NASAbase):
    _default_species = ['H2O(L)', 'H2', 'O2', 'H2O']
    _default_Xi = np.array([0.997, 1e-8, 1e-8, 0.003])
    _liquid = True

class media_C3(Medium_NASAbase):
    _default_species = ['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2', 'C3H8']
    _default_Xi = np.array([0.1, 1e-5, 1e-5, 1e-5, 0.9, 1e-5, 1e-5])

class Atm_Air(Medium_NASAbase): 
    _default_species = ['O2', 'N2', 'CO2']
    _default_Xi = np.array([0.2114, 0.7883, 0.0003])

#%% CP Media
class Medium_CPbase(Medium_Base, thermo.Thermo_CP):
    def __init__(self, name, species=[], T=298.15, Xi=None, P=1., phase='gas', EOS='HEOS'):
        Medium_Base.__init__(self, name, species, Xi) 
        thermo.Thermo_CP.__init__(self, species=species, T=T, Xi=Xi, P=P, EOS=EOS, phase=phase)
        self.phase = phase
        self.EOS = EOS

    def __copy__(self):
        return type(self)(self.name, self.phase, self.EOS)

class Fuel_CH4_CP(Medium_CPbase): #extending medium
    def __init__(self, name = 'Fuel_CH4_CP', phase='gas', EOS='HEOS'):
        super().__init__(name, species=['H2','CH4','CO2','CO','H2O','N2'], Xi=np.array([0.1, 1e-8, 1e-8, 1e-8, 0.9, 1e-8]), phase=phase, EOS=EOS) 


#%% Peng-Robinson media (real gas, liquid and two-phase streams)
class Medium_PRbase(Medium_Base, thermo.Thermo_PR):
    """
    Real-fluid medium based on the Peng-Robinson equation of state (JAX).

    A flange with such a medium can hold a gas, a liquid or a two-phase stream:
    with ``phase='auto'`` the phase split is found by a flash whenever a
    property is requested, so e.g. a ``Heater`` cooling below the dew point
    becomes a partial condenser without further changes.

        syngas = Medium_PRbase('syngas', species=['H2', 'CO', 'CO2', 'H2O'], Xi=np.array([0.5, 0.2, 0.1, 0.2]))

    or by sub-classing with ``_default_species`` / ``_default_Xi`` like the NASA media.
    """
    _default_species = None
    _default_Xi = None

    def __init__(self, name:str|None=None, species=None, T=298.15, Xi:npArray[np.float64]|None=None, P=1.,
                 phase='auto', kij=None):
        name = type(self).__name__ if name is None else name
        species = species if species is not None else self._default_species
        Xi = Xi if Xi is not None else self._default_Xi
        if species is None or Xi is None:
            raise TypeError(f"{type(self).__name__} needs 'species' and 'Xi' (or class attributes "
                            f"_default_species and _default_Xi)")
        Medium_Base.__init__(self, name, list(species), reference_X=np.array(Xi, dtype=float))
        thermo.Thermo_PR.__init__(self, species=list(species), T=T, Xi=np.array(Xi, dtype=float), P=P, phase=phase, kij=kij)

    def __copy__(self):
        """A fresh medium of the same kind (species, k_ij and phase specification) at its reference state."""
        return self.with_phase(self.phase)

    def with_phase(self, phase):
        """Copy of this medium with another phase specification ('auto', 'vapour' or 'liquid')."""
        new = Medium_PRbase.__new__(type(self))
        Medium_PRbase.__init__(new, self.name, species=self.species, Xi=self._reference_X, phase=phase, kij=self._kij)
        return new

class Fuel_CH4_PR(Medium_PRbase):
    """Real-fluid counterpart of Fuel_CH4"""
    _default_species = ['H2','CH4','CO2','CO','H2O','N2']
    _default_Xi = np.array([0.1, 1e-8, 1e-8, 1e-8, 0.9, 1e-8])

class Fuel_MeOH_PR(Medium_PRbase):
    """Real-fluid counterpart of Fuel_MeOH"""
    _default_species = ['H2','CH4','CO2','CO','H2O','N2', 'CH3OH']
    _default_Xi = np.array([0.2, 1e-8, 0.05, 0.05, 0.3, 1e-8, 0.4])

class HB_PR(Medium_PRbase):
    """Real-fluid counterpart of HB (ammonia synthesis loop)"""
    _default_species = ['H2','N2','NH3']
    _default_Xi = np.array([0.1, 0.9, 1e-8])

class Air_PR(Medium_PRbase):
    """Real-fluid counterpart of Air"""
    _default_species = ['O2', 'N2']
    _default_Xi = np.array([0.21, 0.79])

class Water_PR(Medium_PRbase):
    """Water / steam as a real fluid (liquid, vapour or two-phase)"""
    _default_species = ['H2O']
    _default_Xi = np.array([1.0])
