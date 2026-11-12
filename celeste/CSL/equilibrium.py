"""
Equilibrium constants shared by the reactor and electrochemical models.

All functions take the temperature in K and work on floats, NumPy and JAX arrays.
Pressures in the corresponding equilibrium expressions are in bar.
"""
import numpy as np

__all__ = ["K_WGS", "K_MSR", "K_rWGS_Graaf", "K_MeOH_CO2_Graaf", "K_MeOH_CO_Graaf"]

_R = 8.314  # J/mol/K, value used when the correlations were fitted


def K_WGS(T, exp=np.exp):
    """Water-gas shift, CO + H2O <-> CO2 + H2 [-]."""
    return exp(-(32.1153*T - 3.5211e4)/(_R*T))


def K_MSR(T, exp=np.exp):
    """Methane steam reforming, CH4 + H2O <-> CO + 3 H2 [bar^2]."""
    return exp(-(225215.699 - 252.643*T)/(_R*T))


def K_rWGS_Graaf(T):
    """Reverse water-gas shift, CO2 + H2 <-> CO + H2O [-] (Graaf et al., 1986)."""
    return 10**((-2073/T) + 2.029)


def K_MeOH_CO2_Graaf(T):
    """CO2 hydrogenation, CO2 + 3 H2 <-> CH3OH + H2O [bar^-2] (Graaf et al., 1986)."""
    return 10**((3066/T) - 10.592)


def K_MeOH_CO_Graaf(T):
    """CO hydrogenation, CO + 2 H2 <-> CH3OH [bar^-2] (Graaf et al., 1986)."""
    return 10**((5139/T) - 12.621)
