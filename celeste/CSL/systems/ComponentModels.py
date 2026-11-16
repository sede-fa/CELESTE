"""
Small reusable systems built from the component models.

1. SimpleHeaterSys:     source -> heater -> sink
2. SimpleCompressorSys: source -> isentropic compressor -> sink
3. Ncomp:               N-stage compression with inter- and after-cooling (a reusable "box")
"""
import numpy as np
from collections.abc import Sequence

from celeste.CSL import interfaces, media, heat, flow, components, turbomachinery, SysBase

class SimpleHeaterSys(SysBase):
    """
    Minimal system: an air stream is heated (or cooled) to ``T_out``.

    Example:
        sys = SimpleHeaterSys('sys', flow_in=1., Xi_in=None, T_in=300., P_in=1., T_out=500.)
        sys.run_sol()
        sys.heater.Q        # heat duty in kJ/s
    """
    def __init__(self, name: str, flow_in, Xi_in, T_in, P_in, T_out):
        super().__init__(locals())
        ## initialise components with fixed parameters
        self.source = flow.SourceM(name + '.source', media.Air(), flow_in, Xi_in, T_in, P_in)
        self.heater = heat.Heater(name + '.heater', media.Air(), media.Air())
        self.heater.F1.T = T_out
        self.sink = flow.Sink(name + '.sink', media.Air())

    def seq_sol(self, store: bool = False) -> None:
        self.connect_seq(self.source.F1, self.heater.F0, store=store) # feed Heater inlet flow
        self.solve(self.heater)
        self.connect_seq(self.heater.F1, self.sink.F0, store=store) # feed sink inlet flow

class SimpleCompressorSys(SysBase):
    """
    Minimal system: an air stream is compressed (isentropic efficiency ``eff``, or any other compressor class
    given as ``compressor``) to ``P_out``.

    Example:
        sys = SimpleCompressorSys('sys', flow_in=1., Xi_in=None, T_in=300., P_in=1., P_out=5.)
        sys.run_sol()
        sys.compressor.W    # shaft work in kJ/s
    """
    def __init__(self, name: str, flow_in, Xi_in, T_in, P_in, P_out, eff: float = 1., compressor=None):
        super().__init__(locals())
        ## initialise components with fixed parameters
        self.source = flow.SourceM(name + '.source', media.Air(), flow_in, Xi_in, T_in, P_in)
        self.compressor = (compressor or turbomachinery.CompIsent)(name + '.compressor', media.Air(), eff=eff)
        self.compressor.F1.P = P_out
        self.sink = flow.Sink(name + '.sink', media.Air())

    def seq_sol(self, store: bool = False) -> None:
        self.connect_seq(self.source.F1, self.compressor.F0, store=store) # feed Compressor inlet flow
        self.solve(self.compressor)
        self.connect_seq(self.compressor.F1, self.sink.F0, store=store) # feed sink inlet flow

class Ncomp(SysBase):
    '''
    N compression stages with intercooling: every stage is a compressor followed by a cooler. F0 is the inlet,
    F1 the outlet of the last cooler. At least 2 stages.

    Parameters:
        name, Medium, n, Xi, T_in, P_in:  feed
        N_stage:     number of stages
        eff_comp:    efficiency of the compressors, one value or a list of N_stage values
        P_comp:      outlet pressure of every stage [bar], list of N_stage values
        P_out:       final pressure [bar]; if given, the stages share one pressure ratio and P_comp is not used
        T_HT:        outlet temperature of every cooler [K], one value or a list of N_stage values
        compressor:  compressor model: a class (default ``turbomachinery.CompIsent``; ``CompPolytropic``, ``CompIsot``)
                     or any callable ``(name, Medium, eff) -> compressor``, one for all stages or a list per stage
                     (a mapped compressor: ``lambda name, Medium, eff: CompressorMapped(name, cmap, Medium, eta=eff)``)
    '''

    def __init__(self, name: str, Medium: media.Medium_Base, n: float = 1., Xi=None, T_in: int = 400, P_in: float = 1., N_stage: int = 2, eff_comp: float | Sequence[float] = 0.7, P_comp: Sequence[float] = (2., 3.),
                 T_HT: float | Sequence[float] = 350., compressor=None, P_out: float | None = None):
        super().__init__(dict(name=name, Medium=Medium, n=n, Xi=Xi, T_in=T_in, P_in=P_in, N_stage=N_stage,
                              eff_comp=eff_comp, P_comp=P_comp, T_HT=T_HT, P_out=P_out))
        if N_stage < 2:
            raise ValueError("Minimum 2 compression stages required for valid configuration")
        if P_out is not None:
            P_comp = P_in*(P_out/P_in)**(np.arange(1, N_stage + 1)/N_stage)
        eff, T_HT, comp_cls = (self._per_stage(v, N_stage, label) for v, label in
                               ((eff_comp, 'eff_comp'), (T_HT, 'T_HT'), (compressor or turbomachinery.CompIsent, 'compressor')))
        if len(P_comp) != N_stage:
            raise ValueError("P_comp list length must match N_stage")
        self.N = 0

        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium, n=n, Xi=Xi, T=T_in, P=P_in)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium, n=n, Xi=Xi, T=298, P=1)

        self.compressors = []
        self.heaters = []
        for i in range(N_stage):
            self.add_stage(eff[i], P_comp[i], T_HT[i], comp_cls[i])

    @staticmethod
    def _per_stage(value, N: int, label: str) -> list:
        '''One value for every stage, or a list with one value per stage.'''
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != N:
                raise ValueError(f"{label} list length must match N_stage")
            return list(value)
        return [value]*N

    def add_stage(self, eff, pressure, temp, compressor=None):
        """Add a compression stage (compressor to ``pressure``, then a cooler to ``temp``) at the end."""
        self.N += 1
        comp = (compressor or turbomachinery.CompIsent)(f"{self.name}.comp{self.N}", Medium=self.Medium, eff=eff)
        comp.F1.P = pressure
        heater = heat.Heater(f"{self.name}.HT{self.N}", Medium0=self.Medium, Medium1=self.Medium)
        heater.F1.T = temp
        heater.F1.P = pressure
        self.compressors.append(comp)
        self.heaters.append(heater)
        setattr(self, f"comp{self.N}", comp)
        setattr(self, f"HT{self.N}", heater)
        self.getObjs()

    def seq_sol(self, store: bool = False) -> None:
        current_flow = self.F0
        for idx, (comp, heater) in enumerate(zip(self.compressors, self.heaters)):
            self.connect_seq(current_flow, comp.F0, store=store)
            self.solve(comp)
            
            self.connect_seq(comp.F1, heater.F0, store=store)
            self.solve(heater)
            
            current_flow = heater.F1
            if idx == self.N - 1:  # Final connection
                self.connect_seq(current_flow, self.F1, store=store)
    
    def performance_report(self):
        """Generate system performance summary"""
        return {
            "total_pressure_ratio": self.F1.P / self.F0.P,
            "stages": [
                {
                    "compressor": comp.name,
                    "pressure_ratio": comp.F1.P / comp.F0.P,
                    "efficiency": comp.eff,
                    "cooler_temp_drop": heater.F0.T - heater.F1.T
                }
                for comp, heater in zip(self.compressors, self.heaters)
            ]
        }

    def run_sol(self):
        self.seq_sol()
 
