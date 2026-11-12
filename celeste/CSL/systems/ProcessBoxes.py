"""
Reusable process sections ("boxes") and example systems built on the flash model.

1. CondenserBox:   cooler + knock-out drum; ports F0 -> F1 (gas), F2 (condensate)
2. MethanolLoop:   methanol synthesis loop with product condensation, purge and recycle

A box is a ``SysBase`` with its own inlet / outlet flanges, so it can be placed
inside a larger system like a single unit:

    self.cond = CondenserBox(name + '.cond', media.Fuel_CH4(), T_cond=303.15)
    ...
    self.connect_seq(self.reactor.F1, self.cond.F0)
    self.cond.run_sol()
    self.connect_seq(self.cond.F1, self.next_unit.F0)
"""
import numpy as np

from celeste.CSL import SysBase, flow, heat, interfaces, media, reactors, separation


class CondenserBox(SysBase):
    """
    Cools a stream to ``T_cond`` and separates the condensate in a knock-out drum.

    Parameters:
        name:    objects name
        Medium:  medium of the feed and the gas outlet (NASA or Peng-Robinson medium)
        T_cond:  temperature after the cooler and in the drum [K]
        dP:      pressure drop of the drum [bar]

    Ports: F0 (feed) -> F1 (gas), F2 (liquid)
    Units: ``cooler`` (heat.Heater), ``drum`` (separation.Flash)
    """
    def __init__(self, name, Medium, T_cond=303.15, dP=0.0):
        super().__init__(locals())
        self.cooler = heat.Heater(name + '.cooler', Medium0=Medium, Medium1=Medium)
        self.cooler.F1.set(T=T_cond)
        self.drum = separation.Flash(name + '.drum', Medium=Medium, dP=dP)
        self.F0 = interfaces.Flange(name + '.F0', Medium=Medium)
        self.F1 = interfaces.Flange(name + '.F1', Medium=self.drum.F1.Medium)
        self.F2 = interfaces.Flange(name + '.F2', Medium=self.drum.F2.Medium)

    def seq_sol(self, store=False):
        self.connect_seq(self.F0, self.cooler.F0, store=store)
        self.cooler.F1.T = self.T_cond
        self.cooler._sol = self.cooler.run_sol()
        self.connect_seq(self.cooler.F1, self.drum.F0, store=store)
        self.drum._sol = self.drum.run_sol()
        self.connect_seq(self.drum.F1, self.F1, store=store)
        self.connect_seq(self.drum.F2, self.F2, store=store)

    @property
    def Q(self):
        """Total heat removed by cooler and drum [kJ/s] (negative)"""
        return self.cooler.Q + self.drum.Q


class MethanolLoop(SysBase):
    """
    Methanol synthesis loop: make-up gas and recycle are mixed, heated and
    converted in an isothermal reactor; the product is condensed and separated,
    a part of the remaining gas is purged, the rest is recycled.

        feed --> mix --> preheat --> reactor --> cond (cooler + drum) --> crude methanol
                  ^                                  |
                  +------------ recycle <-- split <--+--> purge

    Parameters:
        name:     objects name
        n_feed:   make-up gas flow [mol/s]
        X_feed:   make-up gas composition in the species order of media.Fuel_MeOH
                  (H2, CH4, CO2, CO, H2O, N2, CH3OH)
        T_reac:   reactor temperature [K];  P: loop pressure [bar]
        T_cond:   condensation temperature [K]
        purge:    fraction of the drum gas that is purged [-]
        reactor:  model of the isothermal reactors.MeOH: 'equilibrium' or 'yield'
        Yield:    per-pass methanol yield for reactor='yield'

    Solve with ``run_sol()``; results: ``carbon_efficiency``, ``recycle_ratio``,
    ``product`` (flange of the crude methanol) and all the SysBase reports.
    """
    def __init__(self, name, n_feed=100.0, X_feed=None, T_reac=523.15, P=50.0, T_cond=313.15, purge=0.02,
                 reactor='equilibrium', Yield=0.35, T_feed=313.15):
        super().__init__(locals())
        gas = media.Fuel_MeOH()
        if X_feed is None:
            X_feed = np.array([0.72, 1e-8, 0.20, 0.06, 1e-8, 0.02, 1e-8])     # stoichiometric number ~ 2
        self.feed = flow.SourceM(name + '.feed', gas, F1_n=n_feed, F1_Xi=np.asarray(X_feed, dtype=float), F1_T=T_feed, F1_P=P)
        self.mix = flow.Mixer(name + '.mix', gas)
        self.mix.F0.set(T=T_feed, P=P)
        self.mix.F1.set(n=1e-6*n_feed, Xi=np.asarray(X_feed, dtype=float), T=T_feed, P=P)
        self.mix.F2.set(T=T_feed, P=P)
        self.preheat = heat.Heater(name + '.preheat', gas, gas)
        self.preheat.F1.set(T=T_reac, P=P)
        if reactor not in ('equilibrium', 'yield'):
            raise ValueError("reactor must be 'equilibrium' or 'yield'")
        self.reactor = reactors.MeOH(name + '.reactor', gas, model=reactor, Yield=Yield, T=T_reac)
        self.reactor.F0.set(T=T_reac, P=P)
        self.cond = CondenserBox(name + '.cond', gas, T_cond=T_cond)
        self.split = flow.Splitter(name + '.split', gas, Split=1.0 - purge)
        self.purge_out = flow.Sink(name + '.purge_out', gas)
        self.product = flow.Sink(name + '.product', self.cond.F2.Medium)

    def seq_sol(self, store=False):
        self.connect_seq(self.feed.F1, self.mix.F0, store=store)
        self.mix._sol = self.mix.run_sol()
        self.connect_seq(self.mix.F2, self.preheat.F0, store=store)
        self.preheat._sol = self.preheat.run_sol()
        self.connect_seq(self.preheat.F1, self.reactor.F0, store=store)
        self.reactor._sol = self.reactor.run_sol()
        self.connect_seq(self.reactor.F1, self.cond.F0, store=store)
        self.cond.seq_sol(store=store)
        self.connect_seq(self.cond.F1, self.split.F0, store=store)
        self.split._sol = self.split.run_sol()
        self.connect_seq(self.split.F2, self.purge_out.F0, store=store)
        self.connect_seq(self.cond.F2, self.product.F0, store=store)
        if store:   # the recycle closes the loop (shown in the graph; converged by run_sol)
            self.connect_seq(self.split.F1, self.mix.F1, store=True)

    def run_sol(self, eps=1e-9, max_iter=300):
        """Converge the recycle (tear stream: mixer inlet F1). Returns (passes, relative error of the tear stream)."""
        return self.Recycle_Tear(self.split.F1, self.mix.F1, flow_method='seq_sol', eps=eps, max_iter=max_iter)

    @property
    def recycle_ratio(self):
        """Recycle flow per make-up gas flow [mol/mol]"""
        return self.split.F1.n/self.feed.F1.n

    @property
    def carbon_efficiency(self):
        """Carbon in the methanol of the crude product per carbon (CO + CO2) in the make-up gas [-]"""
        f, p = self.feed.F1, self.product.F0
        return p.n*p.Xi[6]/(f.n*(f.Xi[2] + f.Xi[3]))
