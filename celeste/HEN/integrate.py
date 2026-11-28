"""
Build a synthesised heat exchanger network into a copy of the system and solve it, to verify the network.

    plant_hen = plant.integrate(hen)               # duties of the network (mode='duty'), or
    plant_hen = plant.integrate(hen, mode='area')  # areas of the network: the duties follow (rating)
    plant_hen.hen_report                           # network against the simulated system

Every heater or cooler of the stream table (``pinch_stream_table()``) is replaced in the copy by a chain:
the sides of its exchangers in the order of the stages, then a trim heater or cooler to the original outlet
temperature, which takes the place of the utility. As in the network synthesis the stream splits of one stage
mix at one temperature, so every stage of a stream is one ``ExchangerSide``; a stage shared by several
exchangers carries the sum of their duties. The copy keeps the declared streams of the system: the
original flanges of each heater become the ports of its chain.

``mode='duty'``  each stage takes the duties of ``hen.network``: the real property models give the
                 temperatures, the report gives the approach temperatures and the trim duties.
``mode='area'``  each exchanger has the UA of the network (U * area); its duty follows from the
                 effectiveness-NTU method (counter-current) at the simulated inlet temperatures and heat
                 capacity flows, iterated with the whole system until the duties settle.
"""
import copy

import numpy as np
import pandas as pd

from ..CSL import heat, interfaces
from ..CSL._base import ModelBase, SysBase


class ExchangerSide(ModelBase):
    """
    One side of a process-process heat exchanger: the stream F0 -> F1 takes the heat ``duty`` (kW,
    negative on the hot side). Pressure, flow and composition pass through, the outlet temperature follows
    from the duty.

    Second law: the heat crosses at ``T_heat``, the logarithmic mean temperature of the cold side of the
    exchanger, so the hot side carries the destruction of the temperature difference and the cold side
    none; the heat stays in the process (``Exloss = 0``).
    """
    pfd_kind = 'heat'

    def __init__(self, name: str, Medium0, Medium1=None, duty: float = 0.0):
        super().__init__(dict(name=name))
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium0)
        self.F1 = interfaces.Flange(name + '.F1', Medium=Medium1 or Medium0)
        self.duty = float(duty)
        self.cold_side = self                              # the side whose temperature the heat crosses at

    def run_sol(self):
        """Outlet at the inlet flow, composition and pressure; its temperature from n (h1 - h0) = duty (secant)."""
        f0, f1 = self.F0, self.F1
        f1.n, f1.Xi, f1.P = f0.n, f0.Xi, f0.P
        target = f0.h + self.duty/f0.n if f0.n > 0 else f0.h
        T, f1.T = f0.T, f0.T
        r = f1.h - target
        T_new = T - r/max(f0.cp, 1e-12)
        for _ in range(50):
            f1.T = T_new
            r_new = f1.h - target
            if abs(r_new) <= 1e-10*max(1.0, abs(target)):
                break
            slope = (r_new - r)/(T_new - T) if T_new != T else f0.cp
            T, r = T_new, r_new
            T_new = T - r/slope
        return None

    def heat_and_work(self) -> float:
        return self.duty

    @property
    def T_lm(self) -> float:
        """Logarithmic mean of the inlet and outlet temperature, K."""
        a, b = self.F0.T, self.F1.T
        return a if abs(a - b) < 1e-9 else (a - b)/np.log(a/b)

    @property
    def T_heat(self) -> float:
        return self.cold_side.T_lm

    @property
    def Sgen(self) -> float:
        return self.F1.n*self.F1.s - self.F0.n*self.F0.s - self.duty/self.T_heat

    @property
    def Exdest_balance(self) -> float:
        from ..CSL.exergy import T0
        return self.F0.n*self.F0.Ex - self.F1.n*self.F1.Ex + self.duty*(1 - T0/self.T_heat)

    @property
    def Exloss(self) -> float:
        return 0.0


class StreamChain(SysBase):
    """
    A heater or cooler replaced by the sides of its exchangers (one per stage, in flow order) and a trim
    heater or cooler to the original outlet temperature. Ports F0 and F1 are the flanges of the replaced unit.
    """
    def __init__(self, heater: heat.Heater, stages: list):
        super().__init__(dict(name=heater.name))
        self.F0, self.F1 = heater.F0, heater.F1
        self.T_target = float(heater.F1.T)
        M0, M1 = heater.F0.Medium, heater.F1.Medium
        self.sides = {}
        previous = self.F0
        for k in stages:
            side = ExchangerSide(f'{heater.name}.stage{k}', _medium_like(M0))
            self.__dict__[f'stage{k}'] = side
            self.sides[k] = side
            self.link(previous, side.F0)
            previous = side.F1
        self.trim = heat.Heater(f'{heater.name}.trim', _medium_like(M0), _medium_like(M1))
        self.trim.F1.T = self.T_target
        self.link(previous, self.trim.F0)
        self.link(self.trim.F1, self.F1)

    def seq_sol(self, store: bool = False) -> None:
        self.trim.F1.T = self.T_target
        self.auto_sol(store=store)

    def run_sol(self):
        return self.seq_sol()

    @property
    def Q(self) -> float:
        """Heat of the trim heater or cooler (the utility), kJ/s."""
        return self.trim.Q


def _medium_like(medium):
    """An independent medium of the same kind and species (flanges must not share their state)."""
    return copy.deepcopy(medium)


def _replace(plant: SysBase, path: str, unit) -> None:
    parent = plant.rgetattr(path.rsplit('.', 1)[0]) if '.' in path else plant
    parent.__dict__[path.rsplit('.', 1)[-1]] = unit
    parent.__dict__.pop('_plan', None)


def _ntu_duty(UA: float, Th: float, Tc: float, Ch: float, Cc: float) -> float:
    """Heat of a counter-current exchanger, kJ/s, from its UA (kW/K), inlet temperatures and heat capacity flows."""
    if Th <= Tc or UA <= 0:
        return 0.0
    Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)
    NTU, Cr = UA/Cmin, Cmin/Cmax
    eps = NTU/(1 + NTU) if abs(1 - Cr) < 1e-9 else (1 - np.exp(-NTU*(1 - Cr)))/(1 - Cr*np.exp(-NTU*(1 - Cr)))
    return eps*Cmin*(Th - Tc)


def _capacity(side: ExchangerSide) -> float:
    """Heat capacity flow over the stage, kW/K (the inlet n*cp when the stage carries no heat)."""
    dT = abs(side.F0.T - side.F1.T)
    return abs(side.duty)/dT if dT > 1e-6 and abs(side.duty) > 1e-9 else side.F0.n*side.F0.cp


def integrate(system: SysBase, hen, mode: str = 'duty', relax: float = 0.5, tol: float = 1e-6, max_iter: int = 100):
    """
    A solved copy of ``system`` with the network ``hen`` built in (see the module description). The report
    is kept as ``hen_report`` on the copy: one row per exchanger and per utility, network against simulation.
    """
    if mode not in ('duty', 'area'):
        raise ValueError("mode must be 'duty' or 'area'")
    if hen.network is None:
        raise ValueError('integrate: the HEN has not been solved (HENSynthesiser.solve())')
    net = hen.network.reset_index(drop=True)
    plant = copy.deepcopy(system)

    # the chains: hot streams run through the stages 1..K, cold streams K..1
    hot_names, cold_names = list(dict.fromkeys(net.hot)), list(dict.fromkeys(net.cold))
    chains = {}
    for name, reverse in [(n, False) for n in hot_names] + [(n, True) for n in cold_names]:
        try:
            unit = plant.rgetattr(name)
        except AttributeError as err:
            raise ValueError(f'integrate: {name} is not a unit of {system.name}') from err
        if not isinstance(unit, heat.Heater):
            raise ValueError(f'integrate: {name} is not a heater or cooler of {system.name}')
        stages = sorted(set(net.stage[(net.hot == name) | (net.cold == name)]), reverse=reverse)
        chains[name] = StreamChain(unit, stages)
        _replace(plant, name, chains[name])
    for _, r in net.iterrows():
        chains[r.hot].sides[r.stage].cold_side = chains[r.cold].sides[r.stage]

    def set_duties(q):
        for side in (s for c in chains.values() for s in c.sides.values()):
            side.duty = 0.0
        for (_, r), qi in zip(net.iterrows(), q):
            chains[r.hot].sides[r.stage].duty -= qi
            chains[r.cold].sides[r.stage].duty += qi

    run = plant.run_sol
    q = net.Q.to_numpy(dtype=float).copy()
    set_duties(q)
    run()
    iterations = 0
    if mode == 'area':
        UA = (net.U*net.area).to_numpy(dtype=float)
        for iterations in range(1, max_iter + 1):
            q_new = np.zeros_like(q)
            for n, r in net.iterrows():
                hs, cs = chains[r.hot].sides[r.stage], chains[r.cold].sides[r.stage]
                share = lambda side, mask: q[n]/max(q[mask].sum(), 1e-12) if q[mask].sum() > 1e-12 else 1.0
                Ch = _capacity(hs)*share(hs, ((net.hot == r.hot) & (net.stage == r.stage)).to_numpy())
                Cc = _capacity(cs)*share(cs, ((net.cold == r.cold) & (net.stage == r.stage)).to_numpy())
                q_new[n] = _ntu_duty(UA[n], hs.F0.T, cs.F0.T, Ch, Cc)
            change = np.abs(q_new - q).max()
            q = (1 - relax)*q + relax*q_new
            set_duties(q)
            run()
            if change <= tol*max(1.0, np.abs(q).max()):
                break

    rows = []
    for (_, r), qi in zip(net.iterrows(), q):
        hs, cs = chains[r.hot].sides[r.stage], chains[r.cold].sides[r.stage]
        rows.append(dict(item=f'{r.hot} -> {r.cold}', kind='exchanger', stage=r.stage, Q_hen=r.Q, Q_sim=qi,
                         dT_hot_end_hen=r.dT_hot_end, dT_hot_end_sim=hs.F0.T - cs.F1.T,
                         dT_cold_end_hen=r.dT_cold_end, dT_cold_end_sim=hs.F1.T - cs.F0.T))
    util = hen.utilities.set_index('stream') if len(hen.utilities) else pd.DataFrame(columns=['type', 'Q'])
    for name, chain in chains.items():
        hot = name in hot_names
        q_hen = float(util.Q.get(name, 0.0)) if name in util.index else 0.0
        rows.append(dict(item=name, kind='cooler' if hot else 'heater', stage=np.nan, Q_hen=q_hen,
                         Q_sim=-chain.Q if hot else chain.Q))
    report = pd.DataFrame(rows)
    report['dQ'] = report.Q_sim - report.Q_hen
    ok = report[['dT_hot_end_sim', 'dT_cold_end_sim']].min(axis=1) >= hen.dt_min - 1e-6
    report['approach_ok'] = ok.astype(object).where(report.kind == 'exchanger', None)   # None for utilities
    ex = report[report.kind == 'exchanger']
    report.attrs.update(mode=mode, iterations=iterations, dt_min=hen.dt_min,
                        min_approach=float(ex[['dT_hot_end_sim', 'dT_cold_end_sim']].min().min()) if len(ex) else np.nan)
    plant.__dict__['hen_report'] = report
    return plant
