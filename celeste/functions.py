import base64
import functools

import numpy as np
import scipy.constants


def _cantera():
    try:
        import cantera as ct
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("carbonFraction() needs the optional dependency cantera: pip install 'celeste[cantera]'") from err
    return ct


def toSolve(x, sys, *args):
    
    eqs = sys(x, *args).Eqs #'sys', 
    
    return eqs

def rgetattr(obj, attr, *args):
    '''
    Get attributes of an object while giving dotted string as input
    '''
    
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def carbonFraction(T, P, x_dict):
    '''
    Calculate vapor and graphite fractions at equilibrium using Cantera
    
    Parameters
    ----------
    T : float
        Temperature [K]
    P : float
        Pressure [bar]
    x : dict
        Molar composition 
        Must sum to 1.0 or will be normalized
    
    Returns
    -------
    gasphase : float
        Vapor phase mole fraction (0 to 1)
    graphite : float
        Solid carbon mole fraction (0 to 1)
    '''
    
    ct = _cantera()
    T_K = T
    P_Pa = P * 1e5
    
    # Normalize composition if needed
    total = sum(x_dict.values())
    if not np.isclose(total, 1.0):
        x_dict_normalized = {species: xi / total for species, xi in x_dict.items()}
    else:
        x_dict_normalized = x_dict.copy()
    
    # Remove species with zero or negligible mole fraction
    x_dict_normalized = {species: xi for species, xi in x_dict_normalized.items() 
                         if xi > 1e-10}
    
    # Create gas phase with composition
    Fuel_gas = ct.Solution('gri30.yaml')
    # Set initial state
    Fuel_gas.TPX = T_K, P_Pa, x_dict_normalized
    
    # Create carbon phase
    carbon = ct.Solution('graphite.yaml')
    carbon.TP = T_K, P_Pa
    
    # Create mixture (start with 1 mole of gas, 0 moles of carbon)
    mix = ct.Mixture([(Fuel_gas, 1.0), (carbon, 0.0)])
    
    # Equilibrate at constant T and P
    mix.equilibrate('TP', solver='auto', max_steps=5000)
    
    # Get phase moles
    phase_moles = mix.phase_moles()
    
    # Calculate fractions
    total_moles = sum(phase_moles)
    if total_moles > 1e-10:
        gasphase = phase_moles[0] / total_moles
        graphite = phase_moles[1] / total_moles
    else:
        gasphase = 1.0
        graphite = 0.0
    
    return gasphase, graphite
    
def CForm(T, P, x):
    ''' 
    function for checking the possibility of carbon formation with the help of the alpha and betha parameters and the equilibrium constants calculation.
    
    T:     Temperature [°C]
    P:     Pressure [bar]
    x:     np.array with the compositions in this order H2, CH4, CO2, CO, H2O, N2  (H2O and N2 are inerts)
       
    The reactions that are being considered are:
    (1)CH4 -> C + 2H2  dH° = 75 kJ/mol   methane decomposition
    (2)2CO -> C + CO2   dH° = -172 kJ/mol   disproportionation of carbon monoxide (or reverse Boudouard rx)
    
    Po = 1e5 Pa
    alpha = p_H2**2/(p_CH4*Po) : p are the partial pressures of the components
    betha = p_CO2*Po/p_CO**2  
        
    returns K_1, K_2 (equilibrium constants for each rx), alpha, betha, and the corresponding ratios
    if K_1/alpha > 1 -> likely to have carbon deposition
    if K_2/betha > 1 -> is likely to have carbon deposition
    '''
    
    R = scipy.constants.R # J/mol*K
    T_K = T + 273.15
    Po = 1e5 
    P_Pa = P*Po 
    # partial pressures 
    p_i = P_Pa*x 
    # Gibbs free energy
    dG_1 = 58886.79 + 270.55*T_K + 0.0311*T_K**2 - 3e-6*T_K**3 + 291405.7/T_K - 54.598*T_K*np.log(T_K) # J/mol
    dG_2 = -188030.19 + 402.82*T_K - 0.00524*T_K**2 + 828509.9/T_K - 32.026*T_K*np.log(T_K) # J/mol

    # Equilibrium constants
    K_1 = np.exp(-dG_1/R/T_K)
    K_2 = np.exp(-dG_2/R/T_K)
    # alpha and betha
    alpha = p_i[0]**2/p_i[1]/Po
    betha = p_i[2]*Po/(p_i[3]**2)
    #print(f"{'Warning!: Carbon deposition by the' if K_1/alpha > 1 else 'No carbon deposition by the'} CH4 reaction. K_CH4/alpha = {K_1/alpha} ")
    #print(f"{'Warning: Carbon deposition by the' if K_2/betha > 1 else 'No carbon deposition by the'} CO2 reaction. K_CO2/betha = {K_2/betha} ")
    
    variables = {'K_1':K_1, 'K_2':K_2, 'alpha':alpha, 'betha':betha, 'K_CH4/alpha':K_1/alpha, 'K_CO2/betha':K_2/betha}
    return variables

def mm(graph):
    """Render a Mermaid graph in a notebook through the mermaid.ink web service (needs IPython and internet access)."""
    from urllib.request import Request, urlopen
    from IPython.display import display_svg
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url="https://mermaid.ink/svg/" + base64_string
    req=Request(url, headers={'User-Agent': 'IPython/Notebook'})
    display_svg(urlopen(req).read().decode(), raw=True)
