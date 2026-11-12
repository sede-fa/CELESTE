"""Pinch analysis and heat exchanger network synthesis."""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from celeste.HEN import HENCosts, HENSynthesiser, PinchAnalyser, Utility


def table(rows):
    return pd.DataFrame([dict(segment_id=n, segment_type='sensible', side=s, T_in=a, T_out=b, CP=cp, T_phase=np.nan, Q=np.nan)
                         for n, s, a, b, cp in rows])


@pytest.fixture
def four_streams():
    """Linnhoff & Hindmarsh (1983): with dt_min = 20 K the targets are 107.5 kW hot and 40 kW cold utility."""
    return table([('H1', 'hot', 150, 60, 2.0), ('H2', 'hot', 90, 60, 8.0), ('C1', 'cold', 20, 125, 2.5), ('C2', 'cold', 25, 100, 3.0)])


def test_pinch_targets(four_streams):
    pa = PinchAnalyser(four_streams, 20).solve()
    s = pa.summary()
    assert s.hot_utility == pytest.approx(107.5) and s.cold_utility == pytest.approx(40.0)
    assert s.pinch_hot == [90.0] and s.pinch_cold == [70.0]
    assert s.heat_recovery == pytest.approx(180 + 240 - 40)
    assert pa.grand_composite_curve.H.min() == pytest.approx(0.0)
    sweep = PinchAnalyser.sweep_dt_min(four_streams, [10, 20, 30])
    assert sweep.hot_utility.is_monotonic_increasing
    fig, _ = pa.plot_summary()
    assert len(fig.axes) == 2
    with pytest.raises(RuntimeError):
        PinchAnalyser(four_streams, 20).summary()


def test_network_reaches_the_pinch_targets_when_energy_dominates(four_streams):
    costs = HENCosts(hot_utility=1e4, cold_utility=1e4, unit_fixed=1.0, area_coeff=0.0)
    hen = HENSynthesiser(four_streams, 20, n_stages=4, costs=costs).solve(refine=False)
    assert hen.hot_utility_load == pytest.approx(107.5, abs=1e-6)
    assert hen.cold_utility_load == pytest.approx(40.0, abs=1e-6)
    assert hen.check().feasible and hen.milp_status['optimal']


@pytest.mark.parametrize('refine', [False, True])
def test_network_is_consistent(four_streams, refine):
    hen = HENSynthesiser(four_streams, 20).solve(refine=refine)
    s, chk = hen.summary(), hen.check()
    assert chk.feasible and chk.min_approach >= 20 - 1e-6
    assert s.hot_utility >= s.hot_utility_target - 1e-6
    # first law: recovered heat + utilities close every stream
    assert s.heat_recovery + s.cold_utility == pytest.approx(180 + 240)
    assert s.heat_recovery + s.hot_utility == pytest.approx(262.5 + 225)
    assert s.hot_utility - s.cold_utility == pytest.approx(107.5 - 40.0)
    net = hen.network
    # a stream may be split inside a stage: the branch duties add up to the temperature change of the whole stream
    for (name, _), grp in net.groupby(['hot', 'stage']):
        assert grp.Q.sum() == pytest.approx(hen.streams.CP[name]*(grp.T_hot_in.iloc[0] - grp.T_hot_out.iloc[0]))
    np.testing.assert_allclose(net.area, net.Q/(net.U*net.LMTD))
    assert (net.T_hot_in > net.T_cold_out).all() and (net.T_hot_out > net.T_cold_in).all()
    assert s.total_annual_cost == pytest.approx(net.cost.sum() + hen.utilities.cost.sum() + 100*s.hot_utility + 15*s.cold_utility)
    assert s.n_exchangers == len(net) and s.n_heaters + s.n_coolers == len(hen.utilities)


def test_refinement_does_not_increase_the_cost():
    streams = table([('H1', 'hot', 175, 45, 10.), ('H2', 'hot', 125, 65, 40.), ('C1', 'cold', 20, 155, 20.), ('C2', 'cold', 40, 112, 15.)])
    raw = HENSynthesiser(streams, 10).solve(refine=False).summary()
    fine = HENSynthesiser(streams, 10).solve(refine=True)
    assert fine.summary().total_annual_cost <= raw.total_annual_cost*(1 + 1e-9)
    assert fine.check().feasible


def test_expensive_exchangers_lead_to_fewer_units(four_streams):
    cheap = HENSynthesiser(four_streams, 20, costs=HENCosts(unit_fixed=10.0)).solve().summary()
    dear = HENSynthesiser(four_streams, 20, costs=HENCosts(unit_fixed=1e6)).solve().summary()
    assert dear.n_exchangers < cheap.n_exchangers
    assert dear.hot_utility > cheap.hot_utility


def test_latent_segments_and_film_coefficients(four_streams):
    latent = pd.DataFrame([dict(segment_id='condenser', segment_type='latent', side='hot', T_in=np.nan, T_out=np.nan,
                                CP=np.nan, T_phase=100.0, Q=60.0)])
    streams = pd.concat([four_streams, latent], ignore_index=True)
    streams['h'] = [0.5, 2.0, 0.8, np.nan, 5.0]
    hen = HENSynthesiser(streams, 20).solve()
    assert hen.streams.loc['condenser', 'CP'] == pytest.approx(60.0) and hen.streams.loc['C2', 'h'] == 1.0
    assert hen.check().feasible
    assert hen.summary().hot_utility >= hen.targets().hot_utility - 1e-6
    fig, _ = hen.plot_grid_diagram()
    assert fig is not None


def test_stream_table_from_a_system_model():
    from celeste.CSL import media
    from celeste.CSL.systems import ComponentModels
    train = ComponentModels.Ncomp('train', media.Air(), n=20.0, P_comp=[3.0, 9.0])
    train.run_sol()
    streams = train.pinch_stream_table(min_duty=0.0)             # indexed by the unit path, as for the pinch analysis
    cold = pd.DataFrame([dict(segment_type='sensible', side='cold', T_in=20.0, T_out=110.0, CP=0.8, T_phase=np.nan, Q=np.nan)],
                        index=['feedwater'])
    hen = HENSynthesiser(pd.concat([streams, cold]), dt_min=10).solve()
    assert hen.check().feasible and hen.heat_recovery > 0
    assert set(hen.network.hot) <= {'HT1', 'HT2'} and set(hen.network.cold) == {'feedwater'}


def test_only_one_kind_of_stream():
    hen = HENSynthesiser(table([('H1', 'hot', 150, 60, 2.0)]), 10).solve()
    assert hen.network.empty and hen.cold_utility_load == pytest.approx(180.0) and hen.hot_utility_load == 0.0


def test_input_validation(four_streams):
    with pytest.raises(ValueError, match='dt_min'):
        HENSynthesiser(four_streams, 0)
    with pytest.raises(ValueError, match='hot utility'):
        HENSynthesiser(four_streams, 20, hot_utility=Utility(120.0, 119.0))
    with pytest.raises(ValueError, match='unique'):
        HENSynthesiser(pd.concat([four_streams, four_streams]), 20)
    with pytest.raises(RuntimeError):
        HENSynthesiser(four_streams, 20).summary()
