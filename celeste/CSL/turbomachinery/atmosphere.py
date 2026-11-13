"""
ISA atmosphere helper for flight-condition inputs.

Needs the optional dependency ``ambiance`` (``pip install celeste[flight]``).
It is imported lazily, so the rest of the subpackage works without it.
"""
import numpy as np

GAMMA = 1.4
R_AIR = 287.05  # J/(kg K)


def isa_conditions(altitude: float, dT_ISA: float = 0.0, mach: float | None = None):
    """
    ISA atmosphere at a given altitude, with an optional temperature deviation.

    Parameters
    ----------
    altitude : float
        Altitude / m.
    dT_ISA : float
        ISA temperature deviation / K (e.g. 15 for "ISA+15").
    mach : float, optional
        If given, the true airspeed is also returned.

    Returns
    -------
    T : float
        Static temperature / K.
    P : float
        Static pressure / bar.
    rho : float
        Air density / kg/m^3 (ideal gas, consistent with ``dT_ISA``).
    V : float
        True airspeed / m/s (only if ``mach`` is given).
    """
    try:
        from ambiance import Atmosphere
    except ImportError as e:
        raise ImportError("isa_conditions needs the optional dependency 'ambiance' "
                          "(pip install celeste[flight])") from e

    atm = Atmosphere(altitude)
    T = float(atm.temperature[0]) + dT_ISA
    P_pa = float(atm.pressure[0])
    rho = P_pa / (R_AIR * T)

    if mach is None:
        return T, P_pa * 1e-5, rho
    return T, P_pa * 1e-5, rho, mach * np.sqrt(GAMMA * R_AIR * T)
