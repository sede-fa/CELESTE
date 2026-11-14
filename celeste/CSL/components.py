import numpy as np
from .._types import Array
from numpy.typing import NDArray as npArray
from . import ModelBase, interfaces, media
from .. import solvers
from .._deprecation import deprecated

class GPUbase(ModelBase):
    """
    Black-box separator ("gas processing unit"): the outlet compositions are prescribed, the work follows
    from the exergy change and an efficiency.

    Shared by all GPU models:

    * ``Wid`` = exergy out - exergy in (ideal work); ``W`` = Wid/eff when work is needed (Wid > 0),
      Wid*eff when it is recovered; ``eff = 0`` means no work is accounted (W = 0).
    * The work ends up as heat: ``Q_amb`` = ΔH - W is exchanged with the surroundings at T0 so that the
      energy balance closes (``heat_and_work()`` = W + Q_amb). It is not reported as process heat.
    * ``Exloss`` = exergy in + W - exergy out, ``Sgen`` = ΔS - Q_amb/T0, so Exloss = T0·Sgen.
    """
    def __init__(self, name: str, eff, args):
        params = {k: v for k, v in dict(args).items() if k not in ('self', '__class__', 'legacy')}
        super().__init__({**params, 'name': name, 'eff': eff})

    @property
    def Wid(self) -> float:
        return sum(f.n*f.Ex for f in self.outlets) - sum(f.n*f.Ex for f in self.inlets)

    @property
    def W(self) -> float:
        if self.eff == 0:
            return 0.
        Wid = self.Wid
        return float(Wid/self.eff if Wid > 0 else Wid*self.eff)

    @property
    def Q_amb(self) -> float:
        """Heat exchanged with the surroundings at T0 (positive into the unit), kJ/s."""
        dH = sum(f.n*f.h for f in self.outlets) - sum(f.n*f.h for f in self.inlets)
        return dH - self.W

    def heat_and_work(self) -> float:
        return float(self.W + self.Q_amb)

    @property
    def Exloss(self) -> float:
        return sum(f.n*f.Ex for f in self.inlets) + self.W - sum(f.n*f.Ex for f in self.outlets)

    @property
    def Sgen(self) -> float:
        from .exergy import T0
        dS = sum(f.n*f.s for f in self.outlets) - sum(f.n*f.s for f in self.inlets)
        return dS - self.Q_amb/T0

    @staticmethod
    def _split(n, Xi, frac):
        """Molar flow and composition of the part ``frac`` (per component) of a stream n, Xi."""
        flows = n*np.asarray(Xi, dtype=float)*frac
        total = float(flows.sum())
        if total >= 1e-10:
            return total, flows/total
        w = np.asarray(frac, dtype=float)*(np.asarray(Xi) > 0)   # empty stream: keep a valid composition
        w = w if w.sum() > 0 else np.asarray(frac, dtype=float)
        return 0.0, (w/w.sum() if w.sum() > 0 else np.asarray(Xi, dtype=float))


class GPU_1S(GPUbase):
    """
    Black box with one inlet and one outlet whose state is prescribed (for example a condenser that
    changes the medium from gas to liquid water). Flow and composition pass through.

    Parameters:
        name:    objects name
        Medium1: medium of the inlet (F0)
        Medium2: medium of the outlet (F1)
        eff:     efficiency; 0 (default) accounts no work

    Operating state: feed on ``F0``; outlet temperature and pressure set on ``F1`` (they are kept).

    Example:
        gpu = components.GPU_1S('cond', media.Fuel_CH4(), media.Fuel_CH4L(), eff=0.7)
        gpu.F0.n = 100
        gpu.F0.Xi = np.array([0.1, 1e-8, 1e-8, 1e-8, 0.9, 1e-8])
        gpu.F0.T = 873.15
        gpu.F0.P = 3.0
        gpu.F1.T = 673.15
        gpu.F1.P = 1.0
        gpu.run_sol()
    """
    def __init__(self, name:str, Medium1:media.Medium_Base, Medium2:media.Medium_Base, eff:float=0, **legacy):
        super().__init__(name, eff, locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium1)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium2)
        self._init_state(legacy)

    def Eqs(self)->None:
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi

    def run_sol(self):
        return self.Eqs()


class GPU_2SV2(GPUbase):
    """
    Black-box separator with one inlet (F0) and two outlets: the fraction ``Sep[i]`` of every component
    goes to F1, the rest to F2. Both outlets leave at the inlet temperature and pressure.

    Parameters:
        name:   objects name
        Medium: medium of all streams
        Sep:    array (one value 0..1 per species) of the fraction sent to F1
        eff:    efficiency of the separation work (0 < eff <= 1; 0 accounts no work). Defaults to 1

    Operating state: feed on ``F0``.

    Example (H2, CH4, N2 and parts of CO2 and CO to F1; water to F2):
        gpu = components.GPU_2SV2('gpu', media.Fuel_CH4(), Sep=np.array([1, 1, 0.5, 0.4, 0, 1]), eff=0.7)
        gpu.F0.n = 100
        gpu.F0.Xi = np.array([0.1, 0.3, 0.05, 0.05, 0.5, 1e-8])
        gpu.F0.T = 873.15
        gpu.F0.P = 3.0
        gpu.run_sol()
    """
    def __init__(self, name:str, Medium:media.Medium_Base, Sep:npArray[np.float64]|None=None, eff:float=1, **legacy):
        if Sep is None:
            raise ValueError(f"{name}: give the split fractions Sep (one value per species)")
        super().__init__(name, eff, locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium)
        self.Sep = np.asarray(Sep, dtype=float)
        if self.Sep.shape != (len(Medium.species),) or np.any((self.Sep < 0) | (self.Sep > 1)):
            raise ValueError(f"{name}: Sep needs one fraction 0..1 per species of {Medium.species}")
        self.Medium = Medium
        self._init_state(legacy)

    def Eqs(self):
        self.F1.n, self.F1.Xi = self._split(self.F0.n, self.F0.Xi, self.Sep)
        self.F2.n, self.F2.Xi = self._split(self.F0.n, self.F0.Xi, 1.0 - self.Sep)
        for f in (self.F1, self.F2):
            f.P = self.F0.P
            f.T = self.F0.T

    def run_sol(self):
        return self.Eqs()


class GPU_2S(GPU_2SV2):
    """
    Black-box separator that removes one component completely: species ``idxSep`` goes to F2, everything
    else to F1 (``GPU_2SV2`` with Sep = 1 except 0 at ``idxSep``).

    Parameters:
        name:   objects name
        Medium: medium of all streams
        idxSep: index of the separated species. Defaults to 4 (water in media.Fuel_CH4())
        eff:    efficiency of the separation work. Defaults to 1

    Example (water removal):
        gpu = components.GPU_2S('dryer', media.Fuel_CH4(), idxSep=4, eff=0.7)
    """
    def __init__(self, name: str, Medium: media.Medium_Base, idxSep: int = 4, eff: int = 1, **legacy):
        Sep = np.ones(len(Medium.species))
        Sep[idxSep] = 0.0
        super().__init__(name, Medium, Sep=Sep, eff=eff, **legacy)
        self.idxSep = idxSep


class GPU_3S(GPUbase):
    """
    Black-box separator with one inlet (F0) and three outlets. The species marked -1 in ``sep`` goes
    entirely to F3; of every other species the fraction ``sep[i]`` goes to F1 and the rest to F2.

    Parameters:
        name:   objects name
        Medium: medium of all streams
        sep:    array, one value per species: 0..1 (fraction to F1) or -1 (exactly one species, to F3)
        eff:    efficiency of the separation work. Defaults to 1

    Operating state: feed on ``F0``; outlet temperatures and pressures set on F1, F2, F3 (they are kept).

    Example (H2 to F1, CH4 to F2, water to F3):
        gpu = components.GPU_3S('gpu', media.Fuel_CH4(), sep=np.array([1, 0, 1, 1, -1, 1]), eff=0.7)
        gpu.F0.n = 100
        gpu.F0.Xi = np.array([0.1, 0.1, 1e-8, 1e-8, 0.8, 1e-8])
        gpu.F0.T = 873.15
        gpu.F0.P = 3.0
        gpu.F1.T = 673.15
        gpu.F1.P = 1.0
        gpu.run_sol()
    """
    def __init__(self, name:str, Medium:media.Medium_Base, sep:npArray[np.float64]|None=None, eff:float=1.,
                 **legacy):
        super().__init__(name, eff, locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium)
        self.F3 = interfaces.Flange(name + '.F3', Medium=Medium)
        self.Medium = Medium
        self.sep = sep
        legacy.pop('F3_Xi', None)                 # F3 is always the pure species marked -1
        self._init_state(legacy)

    def Eqs(self):
        sep = np.asarray(self.sep, dtype=float)
        idx = np.flatnonzero(sep == -1)
        if idx.size != 1:
            raise ValueError(f"{self.name}: 'sep' must mark exactly one component with -1 (the one sent to F3), got {sep}")
        pure = np.zeros(sep.size); pure[idx[0]] = 1.0
        self.F3.n, self.F3.Xi = float(self.F0.n*self.F0.Xi[idx[0]]), pure
        frac1 = np.where(sep >= 0, sep, 0.0)
        self.F1.n, self.F1.Xi = self._split(self.F0.n, self.F0.Xi, frac1)
        self.F2.n, self.F2.Xi = self._split(self.F0.n, self.F0.Xi, np.where(sep >= 0, 1.0 - sep, 0.0))

    def run_sol(self):
        return self.Eqs()

class ASU(ModelBase):
    """
    A simple air separation unit that seperates air into Nitrogen and Oxygen.
    Air stream enters and 2 streams exit, which are 100% Nitrogen and 100% Oxygen.
    Outlet stream n, h, cp, g, s, mw are calculated
    Work done in separation of N2 and O2 are calculated based on a paper, Morgan[2013],
    where an ASU consumes 108 kWh/tN2 @8 bar
    IMPORTANT: Inlet flow, temperature and pressure are to be specified.
    F1 is pure Nitrogen, F2 is pure Oxygen.
   
    Parameters:
        name: objects name
        Medium: media of inlet stream. Defaults to media.Air()
        n: Inlet mole flow in [mol/s]. Defaults to 1.0
        T: Inlet temperature in [K]. Defaults to 300
        P: Inlet pressure in [bar]. Defaults to 8 
        Power: Power consumption in [kWh/tN2]. Defaults to 108

    Attributes:
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        W: ASU work in [kJ/s]
    
    Methods:
        Eqs: the outlet mols, Xi and work are determined.
        MBCheck: Mass flows for every flnage and mass balance difference are calculated.
        ExBs: Calls the exergy method on each flange. 
        run_sol returns system of equation in "Eqs" mehod

    Examples:
        1) Separation of air
        ASU = airseparator.ASU('ASU_example', F0_n=1013.4, P=8, T=400)
    """
    _default_state = {'F0': dict(n=1.0, Xi=[0.21, 0.79], T=298.15, P=1.0),
                      'F1': dict(P=8.0), 'F2': dict(P=8.0)}

    def __init__(self, name:str, Medium:media.Medium_Base=media.Air(), Power:float=108, **legacy):
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)  # air
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)  # nitrogen
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium)  # oxygen
        # former keywords: T, n, Xi of the feed; F1_P is the product pressure
        T, n, Xi = (legacy.pop(k, None) for k in ('T', 'n', 'Xi'))
        if 'F1_P' in legacy:
            legacy.setdefault('F2_P', legacy['F1_P'])
        self._init_state(self._spread(legacy, ['F0'], T=T, n=n, Xi=Xi))

    def Eqs(self):
        self.F1.T = self.F0.T
        self.F2.T = self.F0.T
        self.F2.P = self.F1.P
        self.F1.n = self.F0.n*self.F0.Xi[1] 
        self.F2.n = self.F0.n - self.F1.n 
        self.F1.Xi = np.array([0, 1])
        self.F2.Xi = np.array([1, 0])
        self.W = self.Power*self.F1.m*3600*1e-6 # kJ/s 
    
    @property
    def Q(self) -> float:
        return -self.W
        
    @deprecated('use MBcheck()')
    def MBCheck(self):
        return self.MBcheck()

    @property
    def Exth(self) -> float:
        return self.Q*(1 - (298.15/self.F1.T))
    
    @property
    def Exloss(self) -> float:
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex + self.W + self.Exth

    @property
    def Sgen(self) -> float:
        return self.F1.s*self.F1.n + self.F2.s*self.F2.n - self.F0.s*self.F0.n - self.Q/self.F1.T

    def run_sol(self):
        return self.Eqs()

class Pump(ModelBase):
    '''
    Adiabatic liquid pump (or hydraulic turbine when the pressure falls).

    Work for an incompressible liquid, W = V̇·ΔP/eff (pump) or V̇·ΔP·eff (turbine),
    with the density of the inlet medium; the outlet temperature follows from the
    energy balance h1 = h0 + W/n.

    Parameters:
        name:   objects name
        Medium: medium of the stream. Defaults to media.Fuel_CH4L() (pure water feed)
        eff:    efficiency, 0 < eff <= 1. Defaults to 1
        P_out:  outlet pressure [bar]
        dP:     pressure rise [bar] (P1 = P0 + dP)

    With neither ``P_out`` nor ``dP`` the pressure set on ``F1`` is the specification.

    Ports: F0 -> F1
    Properties: W (kJ/s, added), Q (= 0 after run_sol), Exloss, Exeff, Sgen

    Example:
        pump = components.Pump('pump', media.Fuel_CH4L(), eff=0.75, P_out=30.0)
        pump.F0.n = 100.0
        pump.F0.T = 298.15
        pump.F0.P = 1.0
        pump.run_sol()
    '''
     
    _default_state = {'F0': dict(n=1.0, Xi=[0., 0., 0., 0., 1., 0.], T=298.15, P=1.0),
                      'F1': dict(n=1.0, T=298.15, P=3.0)}

    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_CH4L(), eff:float=1.0,
                 P_out:float|None=None, dP:float|None=None, **legacy):
        if P_out is not None and dP is not None:
            raise ValueError(f"{name}: give either the outlet pressure P_out or the pressure rise dP, not both")
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        n, Xi = legacy.pop('n', None), legacy.pop('Xi', None)   # former keywords for the feed
        self._init_state(self._spread(legacy, ['F0'], n=n, Xi=Xi))
        self.R = 8.314462618e-3

    def _set_outlet_pressure(self):
        """Outlet pressure from ``P_out`` or ``dP`` (rise); with neither, the pressure set on ``F1`` is the spec."""
        if self.P_out is not None:
            self.F1.P = self.P_out
        elif self.dP is not None:
            self.F1.P = self.F0.P + self.dP

    def Eqs(self):
        self._set_outlet_pressure()
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi
        self.Eq = np.array([self.Q])
        return self.Eq

    @property
    def Q(self) -> float:
        return self.F1.n*self.F1.h - (self.W + self.F0.n*self.F0.h) 

    @property
    def W(self) -> float:
        if self.F1.P > self.F0.P:
            eff_ratio = 1/self.eff
        else:
            eff_ratio = self.eff
        return (self.F0.m*1e-1*(self.F1.P - self.F0.P)/self.F0.Medium.rho)*eff_ratio

    @property
    def Exloss(self) -> float:
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.W 

    @property
    def Exeff(self) -> float:
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex + self.W)
    
    @property
    def Sgen(self) -> float:
        # taking out the pressure change in the entropy calculation
        return self.F1.n*self.F1.s- self.F0.n*self.F0.s 
    
    def f_sol(self, x: Array) -> Array:
        self.F1.T = x[0]
        return self.Eqs()
        
    def run_sol(self):
        return solvers.solve(self.f_sol, [self.F0.T], bounds=([1.0], [1e4]), name=self.name)
            
class Valve(ModelBase):
    """
    Throttle valve: adiabatic, no work, so the outlet has the inlet enthalpy (isenthalpic).

    The outlet temperature follows from h(T1, P1) = h(T0, P0). For ideal-gas media
    the enthalpy does not depend on pressure and T1 = T0; real-fluid media
    (``*_PR``) show the Joule–Thomson effect.

    Parameters:
        name:   objects name
        Medium: medium of the stream. Defaults to media.Fuel_CH4()
        P_out:  outlet pressure [bar]
        dP:     pressure drop [bar] (P1 = P0 - dP)

    With neither ``P_out`` nor ``dP`` the pressure set on ``F1`` is the specification.

    Ports: F0 -> F1
    Properties: Exloss, Exeff, Sgen (W = Q = 0)
    """
    _default_state = {'F0': dict(n=1.0, T=298.15, P=3.0), 'F1': dict(n=1.0, T=298.15, P=1.0)}
    W = 0.0
    Q = 0.0

    def __init__(self, name:str, Medium:media.Medium_Base=media.Fuel_CH4(), P_out:float|None=None,
                 dP:float|None=None, **legacy):
        if P_out is not None and dP is not None:
            raise ValueError(f"{name}: give either the outlet pressure P_out or the pressure drop dP, not both")
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)
        n, Xi = legacy.pop('n', None), legacy.pop('Xi', None)   # former keywords for the feed
        self._init_state(self._spread(legacy, ['F0'], n=n, Xi=Xi))

    def Eqs(self):
        if self.P_out is not None:
            self.F1.P = self.P_out
        elif self.dP is not None:
            self.F1.P = self.F0.P - self.dP
        self.F1.n = self.F0.n
        self.F1.Xi = self.F0.Xi

    def _dh(self, T):
        self.F1.T = T
        return self.F1.h - self.F0.h

    def run_sol(self):
        self.Eqs()
        h0, T0 = self.F0.h, self.F0.T
        if abs(self._dh(T0)) <= 1e-9*max(1.0, abs(h0)):
            return solvers.trivial_solution([T0])      # ideal gas (or no pressure change): T1 = T0
        return solvers.solve_scalar(self._dh, max(1.0, T0 - 150.0), T0 + 150.0, x0=T0, name=self.name)

    @property
    def Exloss(self) -> float:
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex

    @property
    def Exeff(self) -> float:
        return 1 - self.Exloss/(self.F0.n*self.F0.Ex)

    @property
    def Sgen(self) -> float:
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n


class DAC(ModelBase):
    """
    A Direct air capture unit that seperates CO2 from air. Assuming 100% capture rate.
    Air stream enters and 2 streams exit, which are 100% CO2 and other components of air(N2, O2). #Ar to be added later in the component list
    Outlet stream n, h, cp, g, s, mw are calculated
    Work done in separation of CO2 from air is calculated based on a paper by Climeworks, https://doi.org/10.3389/fclim.2019.00010 
    where a DAC consumes 2000 kWh/tCO2 @slightly above atmospheric pressure outlet CO2 , (1600 kWh/tCO2 Thermal and 400 kWh/tCO2 Electrical)
                         2447.8 kWh/tCO2 @15 Mpa according to Carbon Engineering, Keith, https://doi.org/10.1016/j.enconman.2023.117687 

    IMPORTANT: Inlet flow, temperature and pressure are to be specified.
    F1 is Air out, F2 is pure CO2.
   
    Parameters:
        name: objects name
        Medium: media of inlet stream. Defaults to media.Air()
        n: Inlet mole flow in [mol/s]. Defaults to 1.0
        T: Inlet temperature in [K]. Defaults to 298.15
        P: Inlet pressure in [bar]. Defaults to 1.0
        Power: Power consumption in [kWh/tCO2]. Defaults to 2000 

    Attributes:
        m0: inlet mass flow in [g/s]
        m1: outlet mass flow in [g/s]
        F0.Ex: float - inlet exergy flow in [kJ/s]
        F1.Ex: float - outlet exergy flow in [kJ/s]
        W: DAC work in [kJ/s]
    
    Methods:
        Eqs: the outlet mols, Xi and work are updated based on new inputs.
        MBCheck: Mass flows for every flange and mass balance difference are calculated.
        run_sol updates system of equations in "Eqs" method based on new inputs

    Examples:
        1) Direct air capture to extract CO2
        DAC = components.DAC('DAC_example', F0_n=1013.4, P=1.2, T=400)
    """
    _default_state = {'F0': dict(n=1.0, Xi=[0.2114, 0.7883, 0.0003], T=298.15, P=1.0),
                      'F1': dict(P=1.0), 'F2': dict(P=1.0)}

    def __init__(self, name:str, Medium:media.Medium_Base=media.Atm_Air(), Power_el:float=400, Power_th:float=1600,
                 **legacy):  # 2447.8 @150 bar
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)  # air in
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium)  # air out
        self.F2 = interfaces.Flange(name + '.F2', Medium=Medium)  # CO2
        T, n, Xi = (legacy.pop(k, None) for k in ('T', 'n', 'Xi'))
        if 'F1_P' in legacy:
            legacy.setdefault('F2_P', legacy['F1_P'])
        self._init_state(self._spread(legacy, ['F0'], T=T, n=n, Xi=Xi))

    def Eqs(self):
        self.F1.T = self.F0.T
        self.F2.T = self.F0.T
        self.F2.P = self.F1.P
        self.F2.n = self.F0.n*self.F0.Xi[2] 
        self.F1.n = self.F0.n - self.F2.n 
        self.F1.Xi = np.array([self.F0.Xi[0]/(1-self.F0.Xi[2]), self.F0.Xi[1]/(1-self.F0.Xi[2]), 0.])
        self.F2.Xi = np.array([0., 0., 1.0])

    @property
    def W(self) -> float: 
        return self.Power_el*self.F2.m*3600*1e-6 # kJ/s 
    
    @property
    def Q(self) -> float:
        return self.Power_th*self.F2.m*3600*1e-6
        
    @deprecated('use MBcheck()')
    def MBCheck(self):
        return self.MBcheck()

    @property
    def Exth(self) -> float:
        return self.Q*(1 - (298.15/self.F2.T))
    
    @property
    def Exloss(self) -> float:
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex - self.F2.n*self.F2.Ex + self.W + self.Exth

    @property
    def Sgen(self) -> float:
        return self.F1.s*self.F1.n + self.F2.s*self.F2.n - self.F0.s*self.F0.n - self.Q/self.F2.T

    def run_sol(self):
        return self.Eqs()

class ChgMedium(ModelBase):
    """
    Hands a stream over from one medium to another (for example from a mixture
    medium to a pure-water medium after a separator). Flow, temperature and
    pressure are passed on; the composition is mapped by species name, where a
    species and its condensed form count as the same ('H2O' <-> 'H2O(L)').
    Species of the outlet medium that the inlet does not contain get zero, and
    the outlet composition is normalised.

    Parameters:
        name:    objects name
        Medium0: inlet medium. Defaults to media.Fuel_CH4()
        Medium1: outlet medium. Defaults to media.Water()

    Ports: F0 -> F1
    """
    def __init__(self, name:str, Medium0:media.Medium_Base|None=None, Medium1:media.Medium_Base|None=None):
        Medium0 = media.Fuel_CH4() if Medium0 is None else Medium0
        Medium1 = media.Water() if Medium1 is None else Medium1
        super().__init__(locals())
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium0)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium1)

    @staticmethod
    def _base(species:str) -> str:
        return str(species)[:-3] if str(species).endswith('(L)') else str(species)

    def Eqs(self):
        spX0 = {}
        for sp, x in zip(self.F0.Medium.species, self.F0.Xi):
            spX0[self._base(sp)] = spX0.get(self._base(sp), 0.0) + (x if x > 1e-8 else 0.0)
        Xi = np.array([spX0.get(self._base(sp), 0.0) for sp in self.F1.Medium.species])
        if Xi.sum() <= 0:
            raise ValueError(f"{self.name}: the outlet medium has no species in common with the inlet")
        self.F1.n = self.F0.n
        self.F1.Xi = Xi/np.sum(Xi)
        self.F1.T = self.F0.T
        self.F1.P = self.F0.P

    @property
    def Exloss(self) -> float:
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex

    @property
    def Sgen(self) -> float:
        return self.F1.s*self.F1.n - self.F0.s*self.F0.n

    def run_sol(self):
        self.Eqs()


def __getattr__(name: str):
    # the compressor models live in celeste.CSL.turbomachinery; the old names keep working
    if name in ('Compressor', 'CompIsot', 'CompIsent', 'CompPolytropic'):
        from . import turbomachinery
        return getattr(turbomachinery, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
