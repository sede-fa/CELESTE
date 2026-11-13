'''
Thermodynamic property objects

1. Thermo:    cp, h, s, g of a single component or ideal mixture from NASA-9 polynomials
2. ThermoDum: dummy thermo object. Returns empty arrays for cp, h, s and g
3. Thermo_CP: CoolProp based properties for single components or mixtures using HEOS, PR or SRK
4. Thermo_PR: Peng-Robinson real-fluid properties (gas and liquid) evaluated with JAX

The numerical kernels live in :mod:`celeste.jx` and are shared with the JAX /
Equinox layer; the classes below are the stateful wrappers used by ``Flange``.
'''
import math

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..jx import nasa as _nasa
from .._deprecation import deprecated


class _classproperty:
    """Read-only, lazily evaluated class attribute."""
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner)

# Transport properties (viscosity, thermal conductivity, Prandtl number) of the NASA media:
#   - pure-species low-pressure viscosity: Chung et al. (1984/1988), with the data and constants of
#     Modelica.Media.IdealGases (FluidData.mo); like the MSL, the association correction kappa is not applied
#   - pure-species conductivity: Eucken (condMethod=1) or modified Eucken (condMethod=2) on the species' NASA cp(T)
#   - mixtures: Wilke (viscosity) and Mason-Saxena (conductivity), as in the MSL
#   - the liquid pseudo-species 'H2O(L)' and 'CH3OH(L)' use simple literature correlations

R_UNIV = 8.314462618  # J/(mol.K)

# Critical/polar data, identical to Modelica.Media.IdealGases.Common.FluidData.
# Tc: critical temperature [K]; Pc: critical pressure [Pa] (needed by the Mason-Saxena
# mixture conductivity); Vc: critical molar volume [cm3/mol]; w: acentric factor [-];
# mu: dipole moment [Debye]; kappa: Chung's association correction. The MSL never passes
# kappa to dynamicViscosityLowPressure (default k = 0), so it is 0 here for every species.
# For a physically better (but non-MSL) water/methanol viscosity use kappa=0.076 / 0.215.

FLUID_DATA = {
    'H2':    dict(Tc=33.25,   Pc=12.97e5,  Vc=65.00,  w=-0.216, mu=0.0, kappa=0.0),
    'CH4':   dict(Tc=190.56,  Pc=45.99e5,  Vc=98.60,  w=0.011,  mu=0.0, kappa=0.0),
    'CO2':   dict(Tc=304.12,  Pc=73.74e5,  Vc=94.07,  w=0.225,  mu=0.0, kappa=0.0),
    'CO':    dict(Tc=132.85,  Pc=34.94e5,  Vc=93.10,  w=0.045,  mu=0.1, kappa=0.0),
    'H2O':   dict(Tc=647.096, Pc=220.64e5, Vc=55.95,  w=0.344,  mu=1.8, kappa=0.0),
    'N2':    dict(Tc=126.20,  Pc=33.98e5,  Vc=90.10,  w=0.037,  mu=0.0, kappa=0.0),
    'O2':    dict(Tc=154.58,  Pc=50.43e5,  Vc=73.37,  w=0.022,  mu=0.0, kappa=0.0),
    'NH3':   dict(Tc=405.40,  Pc=113.53e5, Vc=72.47,  w=0.257,  mu=1.5, kappa=0.0),
    'CH3OH': dict(Tc=512.64,  Pc=80.97e5,  Vc=118.00, w=0.565,  mu=1.7, kappa=0.0),
    'C3H8':  dict(Tc=369.83,  Pc=42.48e5,  Vc=200.00, w=0.152,  mu=0.0, kappa=0.0),
}

# Species handled through liquid-state engineering correlations instead of the
# low-pressure-gas theory above (see _waterLiquid*/_methanolLiquid* below).
_LIQUID_TRANSPORT_SPECIES = ('H2O(L)', 'CH3OH(L)')


def _omegaV(Tstar):
    '''Neufeld collision integral, Omega_v(T*), used by the Chung viscosity correlation.'''
    return (1.16145*Tstar**-0.14874 + 0.52487*np.exp(-0.77320*Tstar) + 2.16178*np.exp(-2.43787*Tstar))


def _chungViscosityArr(T, Mi, Tci, Vci, wi, mui, kappai):
    '''Pure-species low-pressure dynamic viscosity / Pa.s (Chung et al. 1984/1988).

    T: gas temperature, K. Mi: molar mass, g/mol. Tci: critical temperature, K.
    Vci: critical molar volume, cm3/mol. wi: acentric factor. mui: dipole moment, Debye.
    kappai: empirical association correction.
    '''
    mur = 131.3*mui/np.sqrt(Vci*Tci)
    Fc = 1. - 0.2756*wi + 0.059035*mur**4 + kappai
    eta_muP = 40.785*Fc*np.sqrt(Mi*T)/(Vci**(2./3.)*_omegaV(1.2593*T/Tci))   # micropoise
    return eta_muP*1e-7  # Pa.s


def _euckenConductivityArr(eta_i, cp_i, R_i, method):
    '''Pure-species thermal conductivity, W/(m.K), from the Eucken (method=1) or
    modified-Eucken (method=2) correlation. eta_i: viscosity, Pa.s. cp_i, R_i: mass-specific
    cp and gas constant, J/(kg.K).
    '''
    if method == 1:
        return eta_i*(cp_i - R_i + 2.25*R_i)
    return eta_i*(1.32*(cp_i - R_i) + 1.77*R_i)


def _wilkePhi(eta_i, Mi):
    '''Wilke interaction parameters phi_ij.'''
    return (1. + np.sqrt(eta_i[:, None]/eta_i[None, :])*(Mi[None, :]/Mi[:, None])**0.25)**2 / np.sqrt(8.*(1. + Mi[:, None]/Mi[None, :]))


def _wilkeMixViscosity(Xi, eta_i, Mi):
    '''Mixture viscosity, Pa.s, from Wilke's method (species below 1e-6 mole fraction are left out).'''
    m = Xi >= 1e-6
    phi = _wilkePhi(eta_i[m], Mi[m])
    return float(np.sum(Xi[m]*eta_i[m]/(phi @ Xi[m])))


def _masonSaxenaMixConductivity(Xi, k_i, Tci, Pci, Mi, T):
    '''Mixture thermal conductivity, W/(m.K): Mason & Saxena as in Modelica's
    lowPressureThermalConductivity (epsilon = 1.0).

    Xi: mole fractions. k_i: pure-species conductivities, W/(m.K). Tci: critical
    temperatures, K. Pci: critical pressures, Pa. Mi: molar masses (any unit, only
    ratios enter). T: gas temperature, K.
    '''
    gam = 210.*(Tci*Mi**3/Pci**4)**(1./6.)
    Tr = T/Tci
    f = np.exp(0.0464*Tr) - np.exp(-0.2412*Tr)
    ratio = gam[None, :]*f[:, None]/(gam[:, None]*f[None, :])       # translational-conductivity ratio lambda_i/lambda_j
    A = (1. + np.sqrt(ratio)*(Mi[:, None]/Mi[None, :])**0.25)**2/np.sqrt(8.*(1. + Mi[:, None]/Mi[None, :]))
    return float(np.sum(Xi*k_i/(A @ Xi)))


def _waterLiquidViscosity(T):
    '''Liquid water dynamic viscosity, Pa.s. Engineering correlation, valid roughly
    273-600 K (same accuracy class as the Yaws-handbook density fits already used
    elsewhere in this package for H2O(L)).'''
    return 2.414e-5*10**(247.8/(T - 140.))


def _waterLiquidConductivity(T):
    '''Liquid water thermal conductivity, W/(m.K). Quadratic engineering fit,
    valid roughly 273-600 K.'''
    return -0.5752 + 6.397e-3*T - 8.151e-6*T**2


def _methanolLiquidViscosity(T):
    '''Liquid methanol dynamic viscosity, Pa.s. Andrade-type fit to literature data,
    valid roughly 250-350 K.'''
    return np.exp(-11.772 + 1272.7/T)


def _methanolLiquidConductivity(T):
    '''Liquid methanol thermal conductivity, W/(m.K). Linear fit to literature data,
    valid roughly 260-330 K.'''
    return 0.202 - 4.0e-4*(T - 293.15)



def _as_float(v):
    """Scalar float from a python number, numpy scalar or 1-element array."""
    if type(v) is float:
        return v
    if isinstance(v, np.ndarray) and v.ndim:
        return float(v.reshape(-1)[0])
    return float(v)


class Thermo:
    '''
    Ideal-gas / ideal-mixture properties from NASA-9 polynomials.

    Inputs:
        species: sequence of species names (see ``Thermo.df.index`` for what is available)
        T:       temperature, K (or degC with ``T_unit='C'``)
        Xi:      mole fractions, np.array[len=nSpecies]
        P:       pressure, bar

    Attrs (recomputed automatically when T, P or Xi are set):
        cp_i, h_i, s_i, g_i:  species values at T and the standard pressure of 1 bar
        cp, h, s, g:          mixture values; kJ/mol/K, kJ/mol, kJ/mol/K, kJ/mol
        s_T:                  mixture entropy without the mixing / pressure term

    Notes:
        * Outside the fitted temperature range of a species its heat capacity is
          frozen at the range limit, so properties stay finite and continuous
          (``in_range`` tells whether the current state is inside the fit).
        * ``H2O(L)`` is treated as a separate condensed phase: it carries no
          mixing or pressure entropy.
    '''
    R = _nasa.R

    # conductivity of the pure species: 1 = Eucken (Modelica's default), 2 = modified Eucken (Poling et al.);
    # can be overridden per instance, e.g. ``medium.condMethod = 1``
    condMethod = 2
    s_lhv = pd.Series(_nasa.LHV_TABLE)

    @_classproperty
    def df(cls):
        '''NASA database as a DataFrame indexed by species name (loaded on first access).'''
        return _nasa.species_table()

    def __init__(self, species, T: float, Xi, P: float = 1., T_unit: str = 'K'):

        self.species = species  # nSpecies
        self.Ns = len(self.species)

        if T_unit in ['C', 'c']:
            T = T + 273.15

        data = _nasa.nasa_data(species)
        self._data = data
        self._a = data.a
        self._T_break = data.T_break
        self._T_fit_lo = float(data.T_lo.max()) if self.Ns else 0.0
        self._T_fit_hi = float(data.T_hi.min()) if self.Ns else math.inf
        self._MWi = data.mw
        self._lhv_i = data.lhv
        self._no_mix = np.array([1.0 if str(s) == 'H2O(L)' else 0.0 for s in species])
        self._gas = 1.0 - self._no_mix
        self.maxTint = data.a.shape[1]

        self._T = _as_float(T)
        self._Xi = Xi  # nSpecies
        self._P = P

        self.calcThermoSep()
        self.calcThermoMix()
        self.setupTransport()

    ## Methods
    def getMW(self):
        return np.dot(self._MWi, self._Xi)

    @property
    def in_range(self) -> bool:
        '''True when T lies inside the fitted range of every species.'''
        return self._T_fit_lo <= self._T <= self._T_fit_hi

    def calcThermoSep(self):
        '''Species properties at the current T: sets cp_i, h_i, s_i, g_i (each of shape nSpecies).'''
        T = self._T
        if self._T_fit_lo <= T <= self._T_fit_hi:
            M = _nasa.pure_props_scalar(T, self._a, self._T_break)
        else:
            M = np.stack(_nasa.pure_props(np, T, self._a, self._T_break, self._data.T_lo, self._data.T_hi))
        self._M = M                                   # (3, nSpecies): one product gives all mixture sums
        self.cp_i, self.h_i, self.s_i = M
        self.g_i = self.h_i - T*self.s_i

    def calcThermoMix(self):
        '''Mixture properties at the current T, P, Xi: sets cp, h, s_T, s, g.'''
        X = self._Xi if type(self._Xi) is np.ndarray and self._Xi.dtype == np.float64 else np.asarray(self._Xi, dtype=float)
        self.cp, self.h, self.s_T = self._M @ X
        if X.size and X.min() > 0.0:
            s_mix = (X*np.log(X*self._P)) @ self._gas
        else:
            pos = X > 0.0
            s_mix = np.sum(np.where(pos, X*np.log(np.where(pos, X, 1.0)*self._P), 0.0)*self._gas)
        self.s = self.s_T - self.R*s_mix
        self.g = self.h - self._T*self.s

    @deprecated('interval selection is handled inside calcThermoSep()', removal=None)
    def updateBoolArr(self):
        return None

    @deprecated('the temperature basis is built inside calcThermoSep()', removal=None)
    def thermoArr(self):
        return None

    @deprecated('coefficients are cached per species set in celeste.jx.nasa.nasa_data()', removal=None)
    def getCoeffs(self):
        return None


    def setupTransport(self):
        '''Build the per-species transport-property tables (critical/polar data for the
        gas-phase species, flags for the liquid pseudo-species). Called once at
        construction, since self.species never changes over the life of the object.'''

        n = self.Ns
        self._Tc_i = np.ones(n)
        self._Pc_i = np.ones(n)
        self._Vc_i = np.ones(n)
        self._w_i = np.zeros(n)
        self._dipole_i = np.zeros(n)
        self._kappa_i = np.zeros(n)
        self._isGasTransport = np.zeros(n, dtype=bool)
        self._isLiqTransport = np.zeros(n, dtype=bool)

        unknown = []
        for i, sp in enumerate(self.species):
            if sp in FLUID_DATA:
                d = FLUID_DATA[sp]
                self._Tc_i[i] = d['Tc']
                self._Pc_i[i] = d['Pc']
                self._Vc_i[i] = d['Vc']
                self._w_i[i] = d['w']
                self._dipole_i[i] = d['mu']
                self._kappa_i[i] = d['kappa']
                self._isGasTransport[i] = True
            elif sp in _LIQUID_TRANSPORT_SPECIES:
                self._isLiqTransport[i] = True
                base = FLUID_DATA[sp[:-3]]
                self._Tc_i[i], self._Pc_i[i] = base['Tc'], base['Pc']
            else:
                unknown.append(sp)

        self._hasTransportData = self._isGasTransport | self._isLiqTransport

        if unknown:
            print(f"Warning: no transport (viscosity/thermal conductivity) data for "
                  f"species {unknown}; mu and k will exclude {'them' if len(unknown)>1 else 'it'} "
                  f"from the mixing rule (as if not present).")

    def calcTransport(self):
        '''Pure-species dynamic viscosity (mu_i, Pa.s) and thermal conductivity
        (k_i, W/(m.K)) at the current temperature. Gas species use the Chung
        low-pressure-gas viscosity correlation and the Eucken/modified-Eucken thermal
        conductivity correlation; the liquid pseudo-species use dedicated correlations.'''

        self.mu_i = np.zeros(self.Ns)
        self.k_i = np.zeros(self.Ns)

        if np.any(self._isGasTransport):
            g = self._isGasTransport
            self.mu_i[g] = _chungViscosityArr(self._T, self._MWi[g], self._Tc_i[g],
                                               self._Vc_i[g], self._w_i[g],
                                               self._dipole_i[g], self._kappa_i[g])
            cp_mass_g = self.cp_i[g]*1e6/self._MWi[g]   # J/(kg.K)
            R_mass_g = self.R*1e6/self._MWi[g]          # J/(kg.K)
            self.k_i[g] = _euckenConductivityArr(self.mu_i[g], cp_mass_g, R_mass_g, self.condMethod)

        for i in np.nonzero(self._isLiqTransport)[0]:
            sp = self.species[i]
            if sp == 'H2O(L)':
                self.mu_i[i] = _waterLiquidViscosity(self._T)
                self.k_i[i] = _waterLiquidConductivity(self._T)
            elif sp == 'CH3OH(L)':
                self.mu_i[i] = _methanolLiquidViscosity(self._T)
                self.k_i[i] = _methanolLiquidConductivity(self._T)

        # species with no transport data get a tiny, non-zero placeholder viscosity so the
        # Wilke/Mason-Saxena interaction parameters stay finite; they are still excluded
        # from the actual mixture average via the Xi mask applied in mu/k below.
        noData = ~self._hasTransportData
        if np.any(noData):
            self.mu_i[noData] = 1e-12
            self.k_i[noData] = 1e-12
            
    @property
    def mu(self) -> float:
        '''
        Dynamic viscosity of the mixture / Pa.s
        (low-pressure-gas assumption; Chung pure-species values combined with Wilke's mixing rule). Species without
        transport data (see setupTransport) are excluded, as if not present.
        '''
        self.calcTransport()
        Xi_eff = np.where(self._hasTransportData, self._Xi, 0.)
        return _wilkeMixViscosity(Xi_eff, self.mu_i, self._MWi)
    
    @property
    def k(self) -> float:
        '''
        Thermal conductivity of the mixture / W/(m.K)
        (low-pressure-gas assumption; Eucken/modified-Eucken pure-species values combined with the Mason-Saxena
        mixing rule). Species without transport data (see setupTransport) are excluded,
        as if not present.
        '''
        self.calcTransport()
        Xi_eff = np.where(self._hasTransportData, self._Xi, 0.)
        return _masonSaxenaMixConductivity(Xi_eff, self.k_i, self._Tc_i, self._Pc_i, self._MWi, self._T)
    
    @property
    def cp_mass(self) -> float:
        '''Mixture specific heat capacity at constant pressure, mass basis / J/(kg.K)'''
        return self.cp*1e6/self.getMW()
    
    @property
    def Pr(self) -> float:
        '''Mixture Prandtl number / -'''
        return self.mu*self.cp_mass/self.k
    
    # Calculate LHV of mixture
    @property
    def LHVmix(self):
        return np.dot(self._lhv_i, self.Xi)

    # LHV of a given specie
    @property
    def LHVi(self):
        return pd.Series(self._lhv_i, index=list(self.species), name='lhv')

    ## Poperties
    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, t):
        self._T = _as_float(t)
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
    def Yi(self):
        return self._Xi*self._MWi/self.getMW()

    @Yi.setter
    def Yi(self, y):
        self.Xi = (y/self._MWi)/np.sum(y/self._MWi)

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, p):
        self._P = p
        self.calcThermoMix()

    @property
    def TX(self):
        return self._T, self._Xi

    @TX.setter
    def TX(self, tx):
        t, x = tx
        self._T = _as_float(t)
        self._Xi = x
        self.calcThermoSep()
        self.calcThermoMix()

    @property
    def TcX(self):
        return self._T-273.15, self._Xi

    @TcX.setter
    def TcX(self, tx):
        t, x = tx
        self.TX = t + 273.15, x

    @property
    def Tc(self):
        return self._T - 273.15

    @Tc.setter
    def Tc(self, t):
        self.T = t + 273.15

    @property
    def PTX(self):
        return self._P, self._T, self._Xi

    @PTX.setter
    def PTX(self, ptx):
        p, t, x = ptx
        t = _as_float(t)
        self._P = p
        self._Xi = x
        if t != self._T:
            self._T = t
            self.calcThermoSep()
        self.calcThermoMix()

    @property
    def PTcX(self):
        return self._P, self._T - 273.15, self._Xi

    @PTcX.setter
    def PTcX(self, ptx):
        p, t, x = ptx
        self.PTX = p, t + 273.15, x

    def __copy__(self):  # makes completely new object when copy is called
        return Thermo(self.species, self._T, self._Xi, self._P)


class ThermoDum():
    def __init__(self, T):
        self._T = T
        self.P = None
        self.cp_i = np.array([])
        self.h_i = np.array([])
        self.s_i = np.array([])
        self.g_i = np.array([])

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, T):
        self.__dict__['T'] = T
        self._T = T
        self.cp_i = np.array([])
        self.h_i = np.array([])
        self.s_i = np.array([])
        self.g_i = np.array([])


def _coolprop():
    try:
        import CoolProp.CoolProp as CP
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("Thermo_CP needs the optional dependency CoolProp: pip install 'celeste[coolprop]'") from err
    return CP


class Thermo_CP:
    '''
    CoolProp Based Thermo Object

    Inputs:
        species:    array of species; np.array[len=nSpecies, dtype=str]
        T:          Temperature     ; float
        Xi:         mole fractions  ; np.array[len=nSpecies, dtype=float]
        P:          Pressure        ; float
        EOS:        Equation of state;str {'HEOS', 'PR', 'SRK'}; default: 'HEOS'
        phase:      fluid phase     ; str {'gas', 'liquid'} | int {cp.iphase_}; default; 'gas'

    Attrs:
        T:  Temperature, K      ; float
        P:  Pressure, bar       ; float
        Xi: Mole Fraction, -    ; np.Array[len=nSpecies, dtype=float]
        h:  Enthalpy, kJ/mol    ; float                 ; Difference from Standard Conditions
        s:  Entropy, kJ/mol/K   ; float                 ; Difference from Standard Conditions
        g:  Gibbs Energy, kJ/mol; float                 ; H - TS
        cp: Specific Heat Cap.  ; float
        Mixed: TP, TPX          ; Tuple(T, P), Tuple(T, P, Xi)

    Private Attrs:
        _Thermo_CP__AS: CoolProp AbstractState with current conditions. Can use the Thermo_CP object to access methods in this object.
        _Thermo_CP__REF:CoolProp AbstractState with reference conditions. Used to calculate H and S difference

    CoolProp is an optional dependency and is imported when the first object is created.
    '''
    R = _nasa.R
    _PHASES = ('liquid', 'supercritical_liquid', 'supercritical_gas', 'gas', 'supercritical', 'twophase', 'unknown')

    @_classproperty
    def phase_dict(cls):
        '''Dictionary to convert a CoolProp phase index into a phase string.'''
        CP = _coolprop()
        return {getattr(CP, f'iphase_{name}'): name for name in cls._PHASES}

    @classmethod
    def _phase_index(cls, phase):
        '''CoolProp phase index for a phase name; anything else is passed through unchanged.'''
        if isinstance(phase, str) and phase.lower() in cls._PHASES[:-1]:
            return getattr(_coolprop(), f'iphase_{phase.lower()}')
        return phase

    def __init__(self, species, T, Xi, P, EOS='HEOS', phase=None):
        #with EOS as HEOS, true phase at the state conditions can be retrieved but not for PR.
        #HEOS doesnt work for mixtures due to binary interaction error.
        CP = _coolprop()
        self._CP = CP
        self.species = species
        self._T = T
        self._P = P
        self._Xi = Xi
        self.EOS = EOS
        self._phase = phase
        self.__CP_phase = self._phase_index(phase)

        self.__REF = CP.AbstractState(EOS, '&'.join(species))
        self.__REF.set_mole_fractions(Xi)
        self.__update_ref()

        self.__REF_Cel = Thermo(species, 298.15, Xi, 1.)

        self.__AS = CP.AbstractState(EOS, '&'.join(species))
        self.__AS.set_mole_fractions(Xi)

        try:
            # Attempt to specify the phase
            self.__AS.specify_phase(self.__CP_phase)
            # Attempt to update the state with the specified P and T
            self.__update_AS()
        except ValueError as e:
            # Print the warning only if there's an actual ValueError
            print('Warning: Invalid phase input at the given Temperature and Pressure')
            print(f"Error Details: {e}")
            return  # Exit the method to avoid further processing

    def __update_ref(self):  # will be called loads of times
        self.__REF.update(self._CP.PT_INPUTS, 101325, 298.15)

    def __update_AS(self):
        self.__AS.update(self._CP.PT_INPUTS, self._P * 1e5, self._T)

    def getMW(self):
        return self.__AS.molar_mass() * 1e3  # g/mol

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, t):
        self._T = t
        self.__update_AS()

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, p):
        self._P = p
        self.__update_AS()

    @property
    def TP(self):
        return self._T, self._P

    @TP.setter
    def TP(self, tp):
        self._T, self._P = tp
        self.__update_AS()

    @property
    def Xi(self):
        return self._Xi

    @Xi.setter
    def Xi(self, xi):
        self._Xi = xi
        self.__AS.set_mole_fractions(xi)
        self.__update_AS()
        self.__REF.set_mole_fractions(xi)
        self.__update_ref()
        self.__REF_Cel.Xi = xi

    @property
    def PTX(self):
        return self._P, self._T, self._Xi

    @PTX.setter
    def PTX(self, ptx):
        self._P, self._T = ptx[:2]
        self.Xi = ptx[2]

    @property
    def h(self):
        return self.__REF_Cel.h + (self.__AS.hmolar() - self.__REF.hmolar()) * 1e-3

    @property
    def s(self):
        return self.__REF_Cel.s + (self.__AS.smolar() - self.__REF.smolar()) * 1e-3

    @property
    def g(self):
        return self.h - self.T * self.s

    @property
    def cp(self):
        '''Heat capacity, kJ/mol/K'''
        return self.__AS.cpmolar() * 1e-3

    @property
    def rho(self):
        '''Density from the equation of state, kg/m3'''
        return self.__AS.rhomass()

    @property
    def LHVmix(self):
        return self.__REF_Cel.LHVmix

    @property
    def mu(self) -> float:
        '''Dynamic viscosity of the mixture / Pa.s'''
        try:
            return self.__AS.viscosity()
        except Exception as e:
            print(f"Warning: CoolProp could not compute viscosity for this state/EOS ({self.EOS}): {e}. Returning NaN.")
            return np.nan

    @property
    def k(self) -> float:
        '''Thermal conductivity of the mixture / W/(m.K)'''
        try:
            return self.__AS.conductivity()
        except Exception as e:
            print(f"Warning: CoolProp could not compute thermal conductivity for this state/EOS ({self.EOS}): {e}. Returning NaN.")
            return np.nan

    @property
    def Pr(self) -> float:
        '''Mixture Prandtl number / -'''
        return self.mu*self.cp*1e6/self.getMW()/self.k

    @property
    def Yi(self):
        return self.__AS.get_mass_fractions()

    @Yi.setter
    def Yi(self, y):
        self.__AS.set_mass_fractions(y)
        self.Xi = np.array(self.__AS.get_mole_fractions())

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, ph):
        self._phase = ph
        self.__CP_phase = self._phase_index(ph)
        self.__AS.specify_phase(self.__CP_phase)

    def __copy__(self):  # makes completely new object when copy is called
        return Thermo_CP(self.species, self._T, self._Xi, self._P, self.EOS, self._phase)


_PHASE_CODES = {'liquid': 0, 'l': 0, 'vapour': 1, 'vapor': 1, 'gas': 1, 'v': 1, 'g': 1, 'auto': 3, None: 3}


class _PRModel:
    """Compiled Peng-Robinson property functions of one species set (shared between all objects using it)."""
    _cache: dict = {}

    def __init__(self, species, kij):
        import jax
        from ..jx import eos, flash, fluid
        self.eos = eos
        self.fluid = fluid.fluid(species, kij)
        fl = self.fluid
        self.state = jax.jit(lambda T, P, z, phase: fluid.state_pt(T, P, z, fl, phase))
        self.cp = jax.jit(jax.jacfwd(lambda T, P, z, phase: fluid.state_pt(T, P, z, fl, phase).h))
        self.flash = jax.jit(lambda T, P, z: flash.flash_pt(T, P, z, fl.pr))
        nasa_names = tuple(eos.nasa_names(species))
        d = _nasa.nasa_data(nasa_names)
        self.mw = d.mw
        self.lhv = d.lhv
        _, h0, s0 = _nasa.pure_props(np, 298.15, d.a, d.T_break, d.T_lo, d.T_hi)
        self.g0_i = h0 - 298.15*s0          # standard Gibbs energy of the ideal gases at T0, P0

    @classmethod
    def get(cls, species, kij):
        from ..jx import eos
        names = tuple(eos.eos_name(s) for s in species)
        k = eos.default_kij(names) if kij is None else kij
        if isinstance(k, dict):
            key = (names, tuple(sorted((tuple(sorted(pair)), float(v)) for pair, v in k.items())))
        else:
            key = (names, np.asarray(k, dtype=float).tobytes())
        if key not in cls._cache:
            cls._cache[key] = cls(names, kij)
        return cls._cache[key]


class Thermo_PR:
    '''
    Real-fluid properties from the Peng-Robinson equation of state (JAX), with
    the NASA polynomials as ideal-gas reference. The JAX-native alternative to
    Thermo_CP: no external property library, liquids and vapour-liquid
    equilibrium included.

    Inputs:
        species: sequence of species names (see celeste.jx.eos.species_available())
        T:       temperature, K
        Xi:      overall mole fractions
        P:       pressure, bar
        phase:   'auto' (default) - the state is flashed and may be two-phase;
                 'vapour' / 'liquid' - evaluate that phase without a flash
        kij:     binary interaction parameters: None (built-in values), a matrix or {('CO2', 'H2O'): 0.12}

    Attrs (evaluated lazily, on first access after T, P, Xi or phase changed):
        h, s, g:    molar enthalpy [kJ/mol], entropy [kJ/mol/K], Gibbs energy [kJ/mol] of the whole stream
        cp:         dh/dT at constant P and overall composition [kJ/mol/K]; includes the latent
                    contribution inside the two-phase region
        beta:       vapour fraction [mol/mol];  state: 'liquid', 'vapour' or 'two-phase'
        x_liquid, y_vapour: phase compositions
        v, rho, Z:  molar volume [m3/mol], density [kg/m3], compressibility factor

    Enthalpies share the reference of the NASA media (elements at 298.15 K), so
    energy balances can mix both kinds of media.
    '''
    R = _nasa.R
    T0 = 298.15
    _STATE_NAMES = ('liquid', 'vapour', 'two-phase')

    def __init__(self, species, T: float, Xi, P: float = 1., phase='auto', kij=None):
        self.species = species
        self.Ns = len(species)
        self._kij = kij
        self._model = _PRModel.get(species, kij)
        self._MWi = self._model.mw
        self._T = _as_float(T)
        self._P = P
        self._Xi = Xi
        self._phase_code = self._code(phase)
        self._phase = phase
        self._st = None
        self._cp = None

    @staticmethod
    def _code(phase):
        key = phase.lower() if isinstance(phase, str) else phase
        if key not in _PHASE_CODES:
            raise ValueError(f"Unknown phase {phase!r}: use 'auto', 'vapour' or 'liquid'")
        return _PHASE_CODES[key]

    def _dirty(self):
        self._st = None
        self._cp = None

    @property
    def _state(self):
        if self._st is None:
            st = self._model.state(self._T, float(self._P), np.asarray(self._Xi, dtype=float), self._phase_code)
            self._st = type(st)(*(np.asarray(v) for v in st))
        return self._st

    def getMW(self):
        return np.dot(self._MWi, self._Xi)

    ## state variables
    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, t):
        self._T = _as_float(t)
        self._dirty()

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, p):
        self._P = p
        self._dirty()

    @property
    def Xi(self):
        return self._Xi

    @Xi.setter
    def Xi(self, x):
        self._Xi = x
        self._dirty()

    @property
    def Yi(self):
        return self._Xi*self._MWi/self.getMW()

    @Yi.setter
    def Yi(self, y):
        self.Xi = (y/self._MWi)/np.sum(y/self._MWi)

    @property
    def PTX(self):
        return self._P, self._T, self._Xi

    @PTX.setter
    def PTX(self, ptx):
        self._P, t, self._Xi = ptx
        self._T = _as_float(t)
        self._dirty()

    @property
    def phase(self):
        """Phase specification: 'auto', 'vapour' or 'liquid'."""
        return self._phase

    @phase.setter
    def phase(self, ph):
        self._phase_code = self._code(ph)
        self._phase = ph
        self._dirty()

    ## properties
    @property
    def h(self):
        return float(self._state.h)

    @property
    def s(self):
        return float(self._state.s)

    @property
    def g(self):
        return self.h - self._T*self.s

    @property
    def cp(self):
        if self._cp is None:
            self._cp = float(self._model.cp(self._T, float(self._P), np.asarray(self._Xi, dtype=float), self._phase_code))
        return self._cp

    @property
    def beta(self):
        """Vapour fraction, mol vapour / mol stream"""
        return float(self._state.beta)

    @property
    def state(self) -> str:
        """Phase state found at the current conditions: 'liquid', 'vapour' or 'two-phase'"""
        return self._STATE_NAMES[int(self._state.phase)]

    @property
    def x_liquid(self):
        return self._state.x

    @property
    def y_vapour(self):
        return self._state.y

    @property
    def converged(self) -> bool:
        return bool(self._state.converged)

    @property
    def v(self):
        """Molar volume, m3/mol"""
        return float(self._state.v)

    @property
    def Z(self):
        return self._P*self.v/(8.314462618e-5*self._T)

    @property
    def rho(self):
        """Density, kg/m3"""
        return self.getMW()*1e-3/self.v

    @property
    def LHVmix(self):
        return np.dot(self._model.lhv, self._Xi)

    def exergy(self, T0: float | None = None):
        '''
        Molar exergy (physical + chemical) in kJ/mol:

            ex = h - T0*s - sum_i z_i*(g0_i - ex0_i)

        with g0_i the Gibbs energy of the pure ideal gas i at T0, 1 bar and ex0_i its
        standard chemical exergy. Phase changes (also condensation at the dead
        state) are contained in h and s of the real fluid.
        '''
        from .exergy import standard_chemical_exergy
        if T0 is not None and T0 != self.T0:
            raise NotImplementedError("the standard chemical exergies are tabulated for T0 = 298.15 K only")
        ex0 = standard_chemical_exergy(tuple(self._model.eos.eos_name(s) for s in self.species))
        z = np.asarray(self._Xi, dtype=float)
        return self.h - self.T0*self.s - float(z @ (self._model.g0_i - ex0))

    def exergy_parts(self):
        '''
        (physical, chemical) molar exergy in kJ/mol. The chemical part is the exergy of the
        stream's composition as an ideal gas mixture at T0, P0; the physical part is the rest.
        '''
        from .exergy import standard_chemical_exergy
        ex0 = standard_chemical_exergy(tuple(self._model.eos.eos_name(s) for s in self.species))
        z = np.asarray(self._Xi, dtype=float)
        pos = z > 0
        ex_ch = float(z @ ex0 + self.R*self.T0*np.sum(np.where(pos, z*np.log(np.where(pos, z, 1.0)), 0.0)))
        return self.exergy() - ex_ch, ex_ch

    def __copy__(self):
        return Thermo_PR(self.species, self._T, self._Xi, self._P, self._phase, self._kij)
