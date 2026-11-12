"""Reusable process boxes and the tear-stream recycle solver."""
import numpy as np
import pytest

pytest.importorskip('jax')

from celeste.CSL import SysBase, flow, heat, media, reactors  # noqa: E402
from celeste.CSL.systems import ProcessBoxes  # noqa: E402
from celeste.solvers import ConvergenceWarning  # noqa: E402
from test_systems import RecycleLoop  # noqa: E402


def test_condenser_box(syngas_x):
    box = ProcessBoxes.CondenserBox('c', media.Fuel_CH4(), T_cond=303.15)
    box.F0.n, box.F0.Xi, box.F0.T, box.F0.P = 10.0, syngas_x, 600.0, 20.0
    box.run_sol()
    assert box.F1.n == pytest.approx(8.0188, abs=1e-3) and box.F2.n == pytest.approx(1.9812, abs=1e-3)
    assert box.F1.T == box.F2.T == 303.15 and box.F2.Xi[4] > 0.9999
    assert box.Q == pytest.approx(box.F1.n*box.F1.h + box.F2.n*box.F2.h - box.F0.n*box.F0.h)
    assert box.mass_balance() == pytest.approx(0.0, abs=1e-5)
    assert box.r_get() == ['cooler', 'drum'] and box.get_success() == {'drum': True}
    assert box.exdest_df().loc['Separation', 'Ex. destruction / kW'] != 0


@pytest.mark.parametrize('kind, efficiency, ratio', [('equilibrium', 0.908687, 5.1638), ('yield', 0.948888, 3.2842)])
def test_methanol_loop(kind, efficiency, ratio):
    loop = ProcessBoxes.MethanolLoop('loop', reactor=kind)
    passes, error = loop.run_sol()
    assert error < 1e-9 and passes < 100                 # flow-only acceleration needed > 500 passes here
    assert loop.carbon_efficiency == pytest.approx(efficiency, abs=2e-5)
    assert loop.recycle_ratio == pytest.approx(ratio, abs=2e-3)
    assert loop.MBcheck() == pytest.approx(0.0, abs=1e-4)
    assert loop.EBcheck() == pytest.approx(0.0, abs=1e-6)
    boundary = loop.boundary_flanges()                   # make-up in; liquid product and purge out
    assert boundary['feeds'] == [loop.mix.F0] and boundary['products'] == [loop.cond.drum.F2, loop.split.F2]
    assert all(loop.get_success().values())
    product = loop.product.F0
    assert product.Xi[6] > 0.5 and product.Xi[4] > 0.4 and product.Xi[0] < 1e-3        # crude methanol: CH3OH + H2O
    # carbon balance over the loop: make-up = product + purge
    carbon = lambda f: f.n*(f.Xi[1] + f.Xi[2] + f.Xi[3] + f.Xi[6])
    assert carbon(loop.feed.F1) == pytest.approx(carbon(product) + carbon(loop.purge_out.F0), rel=1e-6)
    streams = loop.pinch_stream_table()
    assert set(streams.index) == {'cond.cooler', 'preheat'}
    assert 'loop.split --F1_F1--> loop.mix' in loop.create_graph('mermaid', save=False)
    with pytest.raises(ValueError):
        ProcessBoxes.MethanolLoop('x', reactor='plug-flow')


@pytest.mark.parametrize('method', ['anderson', 'wegstein', 'aitken', 'direct'])
def test_tear_stream_solver(method):
    loop = RecycleLoop('loop')
    passes, error = loop.Recycle_Tear(loop.split.F1, loop.mix.F1, method=method, eps=1e-10)
    assert error < 1e-10
    assert loop.split.F1.n == pytest.approx(1.5, rel=1e-8) and loop.split.F1.T == pytest.approx(400.0, abs=1e-6)
    assert loop.MBcheck() == pytest.approx(0.0, abs=1e-7)
    if method == 'anderson':
        assert passes < 25


def test_tear_stream_solver_reports_failure():
    loop = RecycleLoop('loop', ratio=0.99)
    with pytest.warns(ConvergenceWarning, match='recycle not converged'):
        loop.Recycle_Tear(loop.split.F1, loop.mix.F1, method='direct', max_iter=5)
    with pytest.raises(ValueError):
        loop.Recycle_Tear(loop.split.F1, loop.mix.F1, method='newton')


@pytest.mark.parametrize('method, max_passes', [('anderson', 30), ('aitken', 120), ('wegstein', 260)])
def test_methanol_loop_tear_methods(method, max_passes):
    """All accelerated methods reach the same steady state; Anderson needs the fewest passes."""
    loop = ProcessBoxes.MethanolLoop('loop', reactor='equilibrium')
    passes, error = loop.run_sol(method=method, max_iter=500)
    assert error < 1e-9 and passes <= max_passes
    assert loop.carbon_efficiency == pytest.approx(0.908687, abs=2e-6)


def test_tear_options():
    loop = RecycleLoop('loop')
    passes, error = loop.Recycle_Tear(loop.split.F1, loop.mix.F1, method='wegstein', q=-0.5, n_direct=3)
    assert error < 1e-8 and loop.split.F1.n == pytest.approx(1.5, rel=1e-7)
    loop = RecycleLoop('loop')
    assert loop.Recycle_Tear(loop.split.F1, loop.mix.F1, method='steffensen')[1] < 1e-8   # alias of 'aitken'


class FEHELoop(SysBase):
    """Adiabatic ammonia bed with a feed-effluent heat exchanger: the hot inlet of the exchanger is the tear."""
    def __init__(self, name, UA=40.0, Vol=0.05):
        super().__init__(locals())
        self.feed = flow.SourceM(name + '.feed', media.HB(), F1_n=1.0, F1_Xi=[0.74, 0.25, 0.01], F1_T=450.0, F1_P=200.0)
        self.hx = heat.HexNTU(name + '.hx', media.HB(), media.HB(), UA=UA)
        self.reactor = reactors.HB(name + '.reactor', media.HB(), Vol=Vol, Q=0.0)
        self.product = flow.Sink(name + '.product', media.HB())

    def seq_sol(self, store=False):
        self.connect_seq(self.feed.F1, self.hx.C0, store=store)
        self.hx._sol = self.hx.run_sol()
        self.connect_seq(self.hx.C1, self.reactor.F0, store=store)
        self.reactor._sol = self.reactor.run_sol()
        self.connect_seq(self.hx.H1, self.product.F0, store=store)

    def run_sol(self, T_guess=800.0, **kw):
        self.hx.H0.set(n=1.0, Xi=[0.74, 0.25, 0.01], T=T_guess, P=200.0)
        return self.Recycle_Tear(self.reactor.F1, self.hx.H0, **kw)


@pytest.mark.parametrize('T_guess', [460.0, 900.0])
def test_feed_effluent_heat_exchanger(T_guess):
    """A heat-recovery loop converges like a material recycle; both first guesses reach the same state."""
    s = FEHELoop('s')
    passes, error = s.run_sol(T_guess=T_guess)
    assert error < 1e-8 and passes < 20
    assert s.reactor.F0.T == pytest.approx(670.9, abs=0.2) and s.reactor.F1.T == pytest.approx(837.2, abs=0.2)
    assert s.hx.C1.T == s.reactor.F0.T and s.hx.H0.T == pytest.approx(s.reactor.F1.T, abs=1e-5)
    assert s.EBcheck() == pytest.approx(0.0, abs=1e-6)
    assert s.boundary_flanges()['feeds'] == [s.hx.C0] and s.boundary_flanges()['products'] == [s.hx.H1]
