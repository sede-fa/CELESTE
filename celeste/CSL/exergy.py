'''
Stream exergy

1. standard_chemical_exergy: tabulated standard chemical exergies (celeste/data/Chem_exergies.csv)
2. ThermoExergy:             molar exergy of a stream (physical + chemical) for NASA media

The dead state is T0 = 298.15 K, P0 = 1 bar; water above its saturation
fraction condenses at the dead state. Property models other than the NASA
polynomials extend ``ThermoExergy`` by overriding ``reference_props`` (the
species properties at the dead state) and, if needed, ``condensed`` (see
``celeste.FT.thermoFT.ThermoExergyFT``). A medium selects its exergy model with
the class attribute ``ExergyModel``.

Definitions (per mol of stream, x the mole fractions):

    Ex_ph = sum_i x_i [(h_i - h0_i) - T0 (s_i - s0_i - R ln P)]           gas species
    Ex_ch = sum_i x_i [ex0_i + R T0 ln x_i]                               gas species
    H2O(L): no pressure term and no mixing term (a separate liquid phase)

so that Ex = h - T0 s + (terms that depend on the composition only), which
makes the exergy destruction of a unit equal to T0 times its entropy
generation (Gouy-Stodola) whenever no species changes phase at the dead state.
'''
from functools import lru_cache
from importlib.resources import files

import numpy as np
import pandas as pd

from ..jx import nasa as _nasa

__all__ = ["T0", "P0", "standard_chemical_exergy", "exergy_table", "ThermoExergy"]

T0 = 298.15     #: dead-state temperature, K
P0 = 1.0        #: dead-state pressure, bar


@lru_cache(maxsize=1)
def exergy_table() -> pd.DataFrame:
    '''Table of standard chemical exergies indexed by species name (loaded once).'''
    file_path = files("celeste.data").joinpath("Chem_exergies.csv").as_posix()
    df = pd.read_csv(file_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    return df.set_index("Name")


@lru_cache(maxsize=None)
def standard_chemical_exergy(species: tuple) -> np.ndarray:
    '''Standard chemical exergy [kJ/mol] at T0, P0 for a tuple of species names (cached).'''
    col = exergy_table()['Standard Chem Exergy kJ/mol']
    missing = [sp for sp in species if sp not in col.index]
    if missing:
        raise KeyError(f"No standard chemical exergy tabulated for {missing} (see celeste/data/Chem_exergies.csv)")
    out = np.array(col[list(species)], dtype=float)
    out.setflags(write=False)
    return out


@lru_cache(maxsize=None)
def _nasa_reference(species: tuple, T_ref: float):
    '''(h_i, s_i) of every species at T_ref from the NASA polynomials.'''
    d = _nasa.nasa_data(species)
    _, h_i, s_i = _nasa.pure_props(np, T_ref, d.a, d.T_break, d.T_lo, d.T_hi)
    for v in (h_i, s_i):
        v.setflags(write=False)
    return h_i, s_i


class ThermoExergy:
    '''
    Molar exergy [kJ/mol] of a stream: physical + chemical part.

    Parameters:
        Medium: medium providing ``species``, ``h_i`` and ``s_i`` (species values at the stream temperature)
        Xi:     mole fractions
        T, P:   stream temperature [K] and pressure [bar]

    Attributes / methods:
        Ex:                  total molar exergy
        Ex_ph, Ex_ch:        physical and chemical part
        streamPhysExergy(), streamChemExergy(): the same, as methods (former interface)

    The medium is only read. Its temperature is set to ``T`` if it differs.
    Mole fractions below 1e-6 are treated as 1e-12 inside the calculation.
    '''
    Ps = 0.03169        #: saturation pressure of water at T0, bar
    T0 = T0
    P0 = P0
    R = _nasa.R
    WATER, LIQUID_WATER = 'H2O', 'H2O(L)'

    class _Table:
        def __get__(self, obj, owner):
            return exergy_table()

    df = _Table()       #: standard chemical exergy table (class attribute, loaded on first access)

    # ---- hooks for other property models --------------------------------------------------------
    def reference_props(self, species: tuple):
        '''(h0_i, s0_i): species enthalpy and entropy at the dead-state temperature.'''
        return _nasa_reference(species, self.T0)

    def condensed(self, species: tuple) -> np.ndarray:
        '''
        Species that carry no pressure and no mixing term. Follows the convention of
        the property model, so that exergy and entropy balances agree: NASA media
        treat ``H2O(L)`` as a separate liquid phase (``Thermo._no_mix``).
        '''
        no_mix = getattr(self.Medium, '_no_mix', None)
        if no_mix is not None and len(no_mix) == len(species):
            return np.asarray(no_mix) > 0
        return np.array([sp == self.LIQUID_WATER for sp in species])

    # ---------------------------------------------------------------------------------------------
    def __init__(self, Medium, Xi=None, T=None, P=None):
        self.Medium = Medium
        Xi = np.asarray(Xi, dtype=float)
        self.Xi = np.where(Xi >= 1e-6, Xi, 1e-12)
        self.T = T
        self.P = P
        species = tuple(str(sp) for sp in Medium.species)
        self.species = species

        if Medium.T != T:
            Medium.T = T
        self._h_i = np.asarray(Medium.h_i, dtype=float)
        self._s_i = np.asarray(Medium.s_i, dtype=float)        # species entropy at P0 (no pressure / mixing term)
        self._h0_i, self._s0_i = self.reference_props(species)
        h0_w, s0_w = _nasa_reference((self.LIQUID_WATER,), self.T0)
        self._h0_w, self._s0_w = h0_w[0], s0_w[0]
        self._liq = np.asarray(self.condensed(species), dtype=bool)
        self._ex0 = standard_chemical_exergy(species)
        self._ex0_w = standard_chemical_exergy((self.LIQUID_WATER,))[0]
        self._iw = species.index(self.WATER) if self.WATER in species else None
        self._x_w0 = self.Ps/self.P0                           # water fraction of a saturated gas at the dead state

    # ---------------------------------------------------------------------------------------------
    @property
    def condenses(self) -> bool:
        '''True if water vapour in the stream condenses partly at the dead state.'''
        return self._iw is not None and self.Xi[self._iw] > self._x_w0

    def _dead_state_split(self):
        '''Gas composition x_0 at the dead state and phase split (n_V, n_L, n_H2O) per mol of stream.'''
        iw, xw0 = self._iw, self._x_w0
        if self.Xi[iw] == 1:
            x_0 = self.Xi.copy()
        else:
            x_0 = self.Xi*(1 - xw0)/(1 - self.Xi[iw])
            x_0[iw] = xw0
        n_V = (1 - self.Xi[iw])/(1 - xw0)       # mol of gas at the dead state
        n_L = 1 - n_V                           # mol of condensate
        n_H2O = n_L + n_V*x_0[iw]
        return x_0, n_V, n_L, n_H2O

    def _ln(self, x):
        return np.log(x, where=(x >= 1e-6), out=np.zeros_like(x))

    @property
    def Ex_ph(self) -> float:
        '''Physical exergy, kJ/mol'''
        R, T0, gas = self.R, self.T0, ~self._liq
        dH_i = self._h_i - self._h0_i
        dS_i = self._s_i - self._s0_i - np.where(gas, R*np.log(self.P), 0.0)
        if self.condenses:
            iw = self._iw
            x_0, n_V, n_L, n_H2O = self._dead_state_split()
            dS_i = self._s_i - self._s0_i - np.where(gas, R*self._ln_ratio(x_0), 0.0)
            dH_i[iw] = (n_H2O*self._h_i[iw] - n_L*self._h0_w - x_0[iw]*n_V*self._h0_i[iw])/n_H2O
            dS_i[iw] = (n_H2O*(self._s_i[iw] - R*np.log(n_H2O*self.P)) - n_L*self._s0_w
                        - x_0[iw]*n_V*(self._s0_i[iw] - R*np.log(x_0[iw])))/n_H2O
        return float(self.Xi @ dH_i - T0*(self.Xi @ dS_i))

    def _ln_ratio(self, x_0):
        return np.log(self.Xi*self.P/x_0, where=(x_0 >= 1e-6), out=np.zeros_like(self.Xi))

    @property
    def Ex_ch(self) -> float:
        '''Chemical exergy, kJ/mol'''
        R, T0, gas = self.R, self.T0, ~self._liq
        if self.condenses:
            iw = self._iw
            x_0, n_V, n_L, n_H2O = self._dead_state_split()
            ex_i = self._ex0 + np.where(gas, R*T0*self._ln(x_0), 0.0)
            ex_i[iw] = (x_0[iw]*n_V*(self._ex0[iw] + R*T0*np.log(x_0[iw])) + n_L*self._ex0_w)/n_H2O
        else:
            ex_i = self._ex0 + np.where(gas, R*T0*self._ln(self.Xi), 0.0)
        return float(self.Xi @ ex_i)

    @property
    def Ex(self) -> float:
        '''Total molar exergy (physical + chemical), kJ/mol'''
        return self.Ex_ph + self.Ex_ch

    # former method names
    def streamPhysExergy(self):
        return self.Ex_ph

    def streamChemExergy(self):
        return self.Ex_ch
