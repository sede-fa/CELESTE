"""Reusable process boxes and the tear-stream recycle solver."""
import numpy as np
import pytest

pytest.importorskip('jax')

from celeste.CSL import media  # noqa: E402
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


@pytest.mark.parametrize('method', ['anderson', 'wegstein', 'direct'])
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
