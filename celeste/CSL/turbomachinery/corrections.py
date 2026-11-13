"""
Gas-turbine corrections and scaling laws.

Pure functions (no CELESTE dependencies). Reference conditions are ICAO sea
level: Tref = 288.15 K, Pref = 101325 Pa.
"""
import numpy as np

TREF = 288.15       # K
PREF = 101325.0     # Pa
R_UNIVERSAL = 8.314462618   # J/mol/K


def theta(T, Tref=TREF):
    """Temperature ratio ``T / Tref`` (T in K)."""
    return T / Tref


def delta(p, Pref=PREF):
    """Pressure ratio ``p / Pref`` (p in Pa)."""
    return p / Pref


def Wcorr_from_mf(W, T, P, Tref=TREF, Pref=PREF):
    """Corrected mass flow from real mass flow.

    .. math:: W_{corr} = W \\sqrt{\\theta} / \\delta

    Parameters
    ----------
    W : float
        Real mass flow / kg/s.
    T : float
        Total temperature / K.
    P : float
        Total pressure / Pa.
    """
    return W * np.sqrt(theta(T, Tref)) / delta(P, Pref)


def mf_from_Wcorr(Wc, T, P, Tref=TREF, Pref=PREF):
    """Real mass flow from corrected mass flow (inverse of :func:`Wcorr_from_mf`)."""
    return Wc * delta(P, Pref) / np.sqrt(theta(T, Tref))


def sutherland_mu(T):
    """Air dynamic viscosity / Pa*s from Sutherland's law (170 K < T < 1900 K)."""
    mu0 = 1.716e-5    # Pa*s
    T0 = 273.15       # K
    S = 110.4         # K
    return mu0 * (T / T0)**1.5 * (T0 + S) / (T + S)


def reynolds_number_index(Pt, Tt, Pt_ref=PREF, Tt_ref=TREF):
    """GasTurb-style Reynolds Number Index.

    ``RNI = (mu_ref / mu) * (Pt / Pt_ref) * (Tt_ref / Tt)``
    """
    return (sutherland_mu(Tt_ref) / sutherland_mu(Tt)) * (Pt / Pt_ref) * (Tt_ref / Tt)


def rel_error(value, target):
    """Relative error ``(value - target) / |target|`` for solver residuals."""
    return (value - target) / abs(target)
