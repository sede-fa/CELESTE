"""
NASA-9 polynomial ideal-gas thermodynamics.

The property kernels in this module are written once against a generic array
namespace ``xp`` and are instantiated twice:

* with :mod:`numpy` for the stateful, one-state-at-a-time objects of the
  CELESTE Standard Library (``celeste.CSL.thermo.Thermo``) where a plain
  eager call is the fastest option, and
* with :mod:`jax.numpy` (``jit``/``vmap``/``grad`` ready) for batched
  evaluation, automatic differentiation and the Equinox based models in
  :mod:`celeste.jx`.

Units follow the rest of CELESTE: ``T`` in K, ``P`` in bar, molar enthalpy and
Gibbs energy in kJ/mol, entropy and heat capacity in kJ/mol/K.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib.resources import files
from typing import NamedTuple

import numpy as np

__all__ = [
    "R", "P_REF", "T_REF", "NasaData", "nasa_data", "species_names", "species_table",
    "pure_props", "pure_props_scalar", "mix_props", "props", "elements",
]

R = 8.314462618e-3      #: molar gas constant, kJ/mol/K
P_REF = 1.0             #: standard pressure of the NASA data, bar
T_REF = 298.15          #: reference temperature, K

#: Lower heating values at 298.15 K, kJ/mol (kept from the original framework).
LHV_TABLE = {
    'H2': 241.809, 'CH4': 802.262, 'CO2': 0.0, 'CO': 282.98, 'H2O': 0.0, 'N2': 0.0, 'H2O(L)': 0.0, 'O2': 0.0,
    'CH3OH': 637.6358 + 28.38906, 'CH3OH(L)': 637.6358, 'NH3': 316.836, 'C3H8': 2044.2, 'C(gr)': 393.5,
}


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _records() -> dict:
    """Raw NASA-9 records keyed by species name (loaded once, on first use)."""
    with files("celeste.data").joinpath("coeffsData.json").open("r") as f:
        data = json.load(f)
    return {rec["speciesName"]: rec for rec in data}


def species_names() -> list[str]:
    """Names of every species available in the NASA database."""
    return list(_records())


@lru_cache(maxsize=1)
def species_table():
    """The NASA database as a :class:`pandas.DataFrame` indexed by species name."""
    import pandas as pd
    return pd.json_normalize(list(_records().values())).set_index("speciesName")


def elements(species: str) -> dict[str, float]:
    """Elemental composition of a species, e.g. ``elements('CH4') == {'C': 1.0, 'H': 4.0}``."""
    raw = _records()[species]["Formula"]
    out = {}
    for k in range(0, len(raw), 8):
        sym, num = raw[k:k + 2].strip(), raw[k + 2:k + 8].strip()
        if sym and num and float(num) != 0.0:
            sym = sym[0].upper() + sym[1:].lower()
            out[sym] = out.get(sym, 0.0) + float(num)
    return out


class NasaData(NamedTuple):
    """Coefficient arrays for an ordered set of species (plain numpy arrays)."""
    species: tuple
    a: np.ndarray           #: (Ns, K, 9) polynomial coefficients, a1..a7, b1, b2
    T_break: np.ndarray     #: (Ns, K-1) upper bound of every interval but the last (+inf padded)
    T_lo: np.ndarray        #: (Ns,) lower validity limit
    T_hi: np.ndarray        #: (Ns,) upper validity limit
    mw: np.ndarray          #: (Ns,) molar mass, g/mol
    condensed: np.ndarray   #: (Ns,) 1.0 for condensed species (no pressure / mixing entropy), else 0.0
    lhv: np.ndarray         #: (Ns,) lower heating value, kJ/mol (nan when unknown)

    @property
    def n_species(self) -> int:
        return len(self.species)


def _lhv(name: str) -> float:
    """Tabulated LHV, otherwise computed from formation enthalpies for C/H/O/N species."""
    if name in LHV_TABLE:
        return LHV_TABLE[name]
    try:
        el = elements(name)
    except KeyError:
        return float("nan")
    if "C" not in el and "H" not in el:
        return 0.0                      # nothing to oxidise (inert gases, O2, N2, ...)
    if not set(el) <= {"C", "H", "O", "N"}:
        return float("nan")
    rec = _records()
    hf = lambda s: rec[s]["HeatOfFormation"] * 1e-3
    c, h = el.get("C", 0.0), el.get("H", 0.0)
    return float(hf(name) - c * hf("CO2") - 0.5 * h * hf("H2O"))


@lru_cache(maxsize=None)
def _nasa_data(species: tuple) -> NasaData:
    rec = _records()
    missing = [s for s in species if s not in rec]
    if missing:
        raise KeyError(f"Species not found in the NASA database: {missing}")
    ns = len(species)
    k = max(rec[s]["nTIntervals"] for s in species) if ns else 1
    a = np.zeros((ns, k, 9))
    t_break = np.full((ns, max(k - 1, 1)), np.inf)
    t_lo, t_hi = np.zeros(ns), np.zeros(ns)
    for i, s in enumerate(species):
        rng = rec[s]["TemperatureRange"]
        n = rec[s]["nTIntervals"]
        for j in range(k):
            r = rng[min(j, n - 1)]          # pad by repeating the last interval
            a[i, j, :7] = r["a"]
            a[i, j, 7:] = r["b"]
        for j in range(n - 1):
            t_break[i, j] = rng[j]["T_max"]
        t_lo[i], t_hi[i] = rng[0]["T_min"], rng[n - 1]["T_max"]
    mw = np.array([rec[s]["MolecularWeight"] for s in species], dtype=float)
    condensed = np.array([1.0 if rec[s]["phase"] != 0 else 0.0 for s in species])
    lhv = np.array([_lhv(s) for s in species], dtype=float)
    for arr in (a, t_break, t_lo, t_hi, mw, condensed, lhv):
        arr.setflags(write=False)
    return NasaData(species, a, t_break, t_lo, t_hi, mw, condensed, lhv)


def nasa_data(species) -> NasaData:
    """Coefficient arrays for ``species`` (cached per species tuple)."""
    return _nasa_data(tuple(str(s) for s in species))


# --------------------------------------------------------------------------
# Kernels (generic in the array namespace ``xp``)
# --------------------------------------------------------------------------
def pure_props(xp, T, a, T_break, T_lo, T_hi):
    """
    Standard-state cp, h and s of every species at temperature ``T``.

    Inside the fitted range of a species the NASA-9 polynomial is used as is.
    Outside of it the property is continued with the heat capacity frozen at the
    range limit (``h`` linear, ``s`` logarithmic in ``T``), which keeps all
    properties finite, continuous and monotonic for solver iterates that leave
    the range instead of returning zeros or diverging polynomial values.

    Returns
    -------
    (cp_i, h_i, s_i) : arrays of shape (Ns,), in kJ/mol/K, kJ/mol, kJ/mol/K
    """
    Tc = xp.clip(T, T_lo, T_hi)                                         # (Ns,)
    idx = xp.sum(Tc[:, None] > T_break, axis=1)                         # (Ns,)
    c = xp.take_along_axis(a, idx[:, None, None], axis=1)[:, 0, :]      # (Ns, 9)
    T2 = Tc * Tc
    T3 = T2 * Tc
    T4 = T3 * Tc
    iT = 1.0 / Tc
    lnT = xp.log(Tc)
    cp = c[:, 0] * iT * iT + c[:, 1] * iT + c[:, 2] + c[:, 3] * Tc + c[:, 4] * T2 + c[:, 5] * T3 + c[:, 6] * T4
    h = (-c[:, 0] * iT + c[:, 1] * lnT + c[:, 2] * Tc + c[:, 3] * (0.5 * T2) + c[:, 4] * (T3 / 3.0)
         + c[:, 5] * (0.25 * T4) + c[:, 6] * (0.2 * T4 * Tc) + c[:, 7])
    s = (-0.5 * c[:, 0] * iT * iT - c[:, 1] * iT + c[:, 2] * lnT + c[:, 3] * Tc + c[:, 4] * (0.5 * T2)
         + c[:, 5] * (T3 / 3.0) + c[:, 6] * (0.25 * T4) + c[:, 8])
    h = h + cp * (T - Tc)
    s = s + cp * xp.log(T / Tc)
    return R * cp, R * h, R * s


def pure_props_scalar(T, a, T_break):
    """
    NumPy fast path of :func:`pure_props` for a scalar ``T`` that lies inside the
    fitted range of *every* species (the caller checks this). Same numbers,
    a fraction of the call overhead; used by the stateful CSL objects.
    """
    idx = (T > T_break).sum(axis=1)
    c = a[np.arange(a.shape[0]), idx]                                   # (Ns, 9)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    iT = 1.0 / T
    lnT = math.log(T)
    basis = np.array((
        (iT * iT, iT, 1.0, T, T2, T3, T4, 0.0, 0.0),
        (-iT, lnT, T, 0.5 * T2, T3 / 3.0, 0.25 * T4, 0.2 * T4 * T, 1.0, 0.0),
        (-0.5 * iT * iT, -iT, lnT, T, 0.5 * T2, T3 / 3.0, 0.25 * T4, 0.0, 1.0),
    ))
    out = (R * basis) @ c.T                                             # (3, Ns)
    return out[0], out[1], out[2]


def mix_props(xp, X, P, cp_i, h_i, s_i, condensed):
    """
    Ideal mixture cp, h, s from the species values.

    The mixing/pressure term ``-R x ln(x P)`` is applied to gas species only and
    vanishes smoothly for ``x -> 0`` (non-positive fractions contribute nothing).

    Returns
    -------
    (cp, h, s_T, s) : ``s_T`` is the composition-weighted standard entropy, ``s`` the mixture entropy.
    """
    cp = xp.sum(X * cp_i)
    h = xp.sum(X * h_i)
    s_T = xp.sum(X * s_i)
    pos = X > 0.0
    xs = xp.where(pos, X, 1.0)
    s_mix = xp.sum(xp.where(pos, X * xp.log(xs * P), 0.0) * (1.0 - condensed))
    return cp, h, s_T, s_T - R * s_mix


def props(xp, T, P, X, a, T_break, T_lo, T_hi, condensed):
    """All pure and mixture properties in one call; see :func:`pure_props` and :func:`mix_props`."""
    cp_i, h_i, s_i = pure_props(xp, T, a, T_break, T_lo, T_hi)
    cp, h, s_T, s = mix_props(xp, X, P, cp_i, h_i, s_i, condensed)
    return cp_i, h_i, s_i, cp, h, s_T, s
