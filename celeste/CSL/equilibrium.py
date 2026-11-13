"""
Equilibrium constants shared by the reactor and electrochemical models.

The constants are computed from the standard Gibbs energies of the NASA-9
polynomials - the same data that give ``h`` and ``s`` of the NASA media - so
that equilibrium compositions, heat duties and exergy balances are
thermodynamically consistent over the whole temperature range:

    K(T) = exp(-sum_i nu_i g°_i(T) / (R T))       (ideal gas, standard pressure 1 bar)

All functions take the temperature in K and work on floats, NumPy arrays and
(scalar) JAX values. Pressures in the corresponding mass action laws are in bar.

The earlier empirical fits are kept for reference (``*_fit``, ``*_Graaf``); they
deviate from the NASA values by up to 36 % (water-gas shift at 450 K).
"""
from functools import lru_cache

import numpy as np

from ..jx import nasa as _nasa

__all__ = ["K_eq", "ln_K", "fuel_residuals", "FUEL_SPECIES", "FUEL_ELEMENTS", "K_WGS", "K_rWGS", "K_MSR", "K_PSR", "K_NH3", "K_MeOH_CO2", "K_MeOH_CO",
           "K_WGS_fit", "K_MSR_fit", "K_rWGS_Graaf", "K_MeOH_CO2_Graaf", "K_MeOH_CO_Graaf"]

_R_FIT = 8.314  # J/mol/K, value used when the fits were made

#: reactions as {species: stoichiometric coefficient} (products positive)
REACTIONS = {
    'WGS':      {'CO': -1, 'H2O': -1, 'CO2': 1, 'H2': 1},           # CO + H2O <-> CO2 + H2
    'MSR':      {'CH4': -1, 'H2O': -1, 'CO': 1, 'H2': 3},           # CH4 + H2O <-> CO + 3 H2
    'MeOH_CO2': {'CO2': -1, 'H2': -3, 'CH3OH': 1, 'H2O': 1},        # CO2 + 3 H2 <-> CH3OH + H2O
    'MeOH_CO':  {'CO': -1, 'H2': -2, 'CH3OH': 1},                   # CO + 2 H2 <-> CH3OH
    'NH3':      {'N2': -0.5, 'H2': -1.5, 'NH3': 1},                 # 1/2 N2 + 3/2 H2 <-> NH3
    'PSR':      {'C3H8': -1, 'H2O': -3, 'CO': 3, 'H2': 7},          # C3H8 + 3 H2O <-> 3 CO + 7 H2
    'H2_ox':    {'H2': -1, 'O2': -0.5, 'H2O': 1},                   # H2 + 1/2 O2 <-> H2O
}


@lru_cache(maxsize=None)
def _reaction_data(stoich: tuple):
    species = tuple(sp for sp, _ in stoich)
    nu = np.array([v for _, v in stoich], dtype=float)
    return _nasa.nasa_data(species), nu


def _xp_of(exp):
    if 'jax' in getattr(exp, '__module__', '') or 'jax' in type(exp).__module__:
        import jax.numpy as jnp
        return jnp
    return np


def K_eq(T, reaction, exp=np.exp):
    """
    Equilibrium constant of ``reaction`` at ``T`` [K] from the NASA Gibbs energies.

    ``reaction`` is a key of ``REACTIONS`` or a dict {species: coefficient}.
    The unit is bar^(sum nu) for gas-phase reactions.
    """
    stoich = REACTIONS[reaction] if isinstance(reaction, str) else reaction
    d, nu = _reaction_data(tuple(stoich.items()))
    xp = _xp_of(exp)
    if xp is np and np.ndim(T) > 0:
        return np.vectorize(lambda t: K_eq(float(t), stoich))(T)
    _, h, s = _nasa.pure_props(xp, T, xp.asarray(d.a), xp.asarray(d.T_break),
                               xp.asarray(d.T_lo), xp.asarray(d.T_hi))
    dG = xp.sum(xp.asarray(nu)*(h - T*s))
    return exp(-dG/(_nasa.R*T))


@lru_cache(maxsize=4096)
def ln_K(reaction: str, T: float) -> float:
    """ln K of a named reaction at T [K] (cached: solvers revisit the same temperature)."""
    return float(np.log(K_eq(float(T), reaction)))


# ---- fuel-side equilibrium of the shift and reforming reactions (MR, SOC) -----------------------------
FUEL_SPECIES = ['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2']       #: species order of media.Fuel_CH4()
#: atoms per mol of species (rows C, H, O, N)
FUEL_ELEMENTS = np.array([[0, 1, 1, 1, 0, 0],
                          [2, 4, 0, 0, 2, 0],
                          [0, 0, 2, 1, 1, 0],
                          [0, 0, 0, 0, 0, 2]])


def fuel_residuals(n, n0, T, P, O_in=0.0):
    """
    Equations of a fuel-side stream (H2, CH4, CO2, CO, H2O, N2) at shift and reforming equilibrium:
    the C, H, O, N balances between the inlet flows ``n0`` and the outlet flows ``n`` (relative to the
    element inflow; ``O_in`` is oxygen added from elsewhere, mol/s), then the water-gas shift and
    methane reforming equilibria as ln(Q/K), at T [K] and P [bar].
    """
    n, n0 = np.asarray(n, dtype=float), np.asarray(n0, dtype=float)
    balances = (FUEL_ELEMENTS @ n - FUEL_ELEMENTS @ n0 - np.array([0, 0, O_in, 0]))/np.maximum(FUEL_ELEMENTS @ n0, 1e-30)
    H2, CH4, CO2, CO, H2O = np.log(n[:5]/n.sum())
    return np.concatenate((balances, [
        CO2 + H2 - CO - H2O - ln_K('WGS', T),                          # CO + H2O <-> CO2 + H2
        CO + 3*H2 + 2*np.log(P) - CH4 - H2O - ln_K('MSR', T),          # CH4 + H2O <-> CO + 3 H2
    ]))


def K_WGS(T, exp=np.exp):
    """Water-gas shift, CO + H2O <-> CO2 + H2 [-]."""
    return K_eq(T, 'WGS', exp)


def K_rWGS(T, exp=np.exp):
    """Reverse water-gas shift, CO2 + H2 <-> CO + H2O [-]."""
    return 1.0/K_eq(T, 'WGS', exp)


def K_MSR(T, exp=np.exp):
    """Methane steam reforming, CH4 + H2O <-> CO + 3 H2 [bar^2]."""
    return K_eq(T, 'MSR', exp)


def K_PSR(T, exp=np.exp):
    """Propane steam reforming, C3H8 + 3 H2O <-> 3 CO + 7 H2 [bar^6]."""
    return K_eq(T, 'PSR', exp)


def K_NH3(T, exp=np.exp):
    """Ammonia synthesis, 1/2 N2 + 3/2 H2 <-> NH3 [bar^-1]."""
    return K_eq(T, 'NH3', exp)


def K_MeOH_CO2(T, exp=np.exp):
    """CO2 hydrogenation, CO2 + 3 H2 <-> CH3OH + H2O [bar^-2]."""
    return K_eq(T, 'MeOH_CO2', exp)


def K_MeOH_CO(T, exp=np.exp):
    """CO hydrogenation, CO + 2 H2 <-> CH3OH [bar^-2]."""
    return K_eq(T, 'MeOH_CO', exp)


# ---- former empirical correlations (reference only) ---------------------------------------------
def K_WGS_fit(T, exp=np.exp):
    """Linear Gibbs-energy fit of the water-gas shift used before (accurate at SOC temperatures)."""
    return exp(-(32.1153*T - 3.5211e4)/(_R_FIT*T))


def K_MSR_fit(T, exp=np.exp):
    """Linear Gibbs-energy fit of methane steam reforming used before."""
    return exp(-(225215.699 - 252.643*T)/(_R_FIT*T))


def K_rWGS_Graaf(T):
    """Reverse water-gas shift, CO2 + H2 <-> CO + H2O [-] (Graaf et al., 1986)."""
    return 10**((-2073/T) + 2.029)


def K_MeOH_CO2_Graaf(T):
    """CO2 hydrogenation, CO2 + 3 H2 <-> CH3OH + H2O [bar^-2] (Graaf et al., 1986)."""
    return 10**((3066/T) - 10.592)


def K_MeOH_CO_Graaf(T):
    """CO hydrogenation, CO + 2 H2 <-> CH3OH [bar^-2] (Graaf et al., 1986)."""
    return 10**((5139/T) - 12.621)
