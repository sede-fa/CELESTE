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
import pandas as pd

from ..jx import nasa as _nasa
from .._deprecation import deprecated


class _classproperty:
    """Read-only, lazily evaluated class attribute."""
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner)


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
            self.cp_i, self.h_i, self.s_i = _nasa.pure_props_scalar(T, self._a, self._T_break)
        else:
            self.cp_i, self.h_i, self.s_i = _nasa.pure_props(np, T, self._a, self._T_break,
                                                             self._data.T_lo, self._data.T_hi)
        self.g_i = self.h_i - T*self.s_i

    def calcThermoMix(self):
        '''Mixture properties at the current T, P, Xi: sets cp, h, s_T, s, g.'''
        X = np.asarray(self._Xi, dtype=float)
        self.cp = X @ self.cp_i
        self.h = X @ self.h_i
        self.s_T = X @ self.s_i
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
        from ..FT.thermoFT import standard_chemical_exergy
        if T0 is not None and T0 != self.T0:
            raise NotImplementedError("the standard chemical exergies are tabulated for T0 = 298.15 K only")
        ex0 = standard_chemical_exergy(tuple(self._model.eos.eos_name(s) for s in self.species))
        z = np.asarray(self._Xi, dtype=float)
        return self.h - self.T0*self.s - float(z @ (self._model.g0_i - ex0))

    def __copy__(self):
        return Thermo_PR(self.species, self._T, self._Xi, self._P, self._phase, self._kij)
