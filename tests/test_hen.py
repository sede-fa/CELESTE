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
    indexed = PinchAnalyser(four_streams.set_index('segment_id'), 20).solve()       # names as index work too
    assert indexed.hot_utility == pytest.approx(107.5)


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


def test_objective_mer_reaches_the_pinch_targets(four_streams):
    """'mer' reaches the energy targets whatever the costs, with the fewest exchangers that can."""
    dear_units = HENCosts(unit_fixed=1e6)
    hen = HENSynthesiser(four_streams, 20, n_stages=4, costs=dear_units).solve(objective='mer')
    s = hen.summary()
    assert s.objective == 'mer' and hen.check().feasible
    assert s.hot_utility == pytest.approx(107.5, abs=1e-5) and s.cold_utility == pytest.approx(40.0, abs=1e-5)
    assert s.n_exchangers + s.n_heaters + s.n_coolers == 7      # 4 exchangers, 2 heaters, 1 cooler


def test_objective_min_units(four_streams):
    """'min_units' uses no more units than any other objective and can need more utility."""
    few = HENSynthesiser(four_streams, 20).solve(objective='min_units').summary()
    mer = HENSynthesiser(four_streams, 20, n_stages=4).solve(objective='mer').summary()
    cost = HENSynthesiser(four_streams, 20).solve().summary()
    units = lambda s: s.n_exchangers + s.n_heaters + s.n_coolers
    assert units(few) <= min(units(mer), units(cost))
    assert few.hot_utility >= mer.hot_utility - 1e-6
    assert units(few) == 4 and few.hot_utility == pytest.approx(262.5)   # C1 entirely on hot utility


def test_objective_is_validated(four_streams):
    with pytest.raises(ValueError, match="objective"):
        HENSynthesiser(four_streams, 20).solve(objective='area')


def test_composite_curves_start_at_the_stream_temperatures(four_streams):
    """Each composite curve spans exactly the temperatures of its own streams; the cold one starts at Q_C."""
    pa = PinchAnalyser(four_streams, 20).solve()
    hot, cold = pa.composite_curves['hot'], pa.composite_curves['cold']
    assert (hot['T'].iloc[0], hot['T'].iloc[-1]) == (60.0, 150.0) and hot.H.iloc[0] == 0.0
    assert (cold['T'].iloc[0], cold['T'].iloc[-1]) == (20.0, 125.0)
    assert cold.H.iloc[0] == pytest.approx(pa.cold_utility)
    assert hot.H.iloc[-1] == pytest.approx(180 + 240) and cold.H.iloc[-1] - cold.H.iloc[0] == pytest.approx(262.5 + 225)
    gcc = pa.grand_composite_curve                       # straight lines between the levels: one point per level
    assert gcc.H.iloc[0] == pytest.approx(pa.hot_utility) and gcc.H.iloc[-1] == pytest.approx(pa.cold_utility)
    assert len(gcc) == len(pa.problem_table) + 1


def test_latent_segments_match_a_narrow_glide(four_streams):
    """A latent segment gives the targets of a sensible stream with a very small temperature glide."""
    latent = pd.DataFrame([dict(segment_id='cond', segment_type='latent', side='hot', T_phase=100.0, Q=60.0),
                           dict(segment_id='boil', segment_type='latent', side='cold', T_phase=50.0, Q=40.0)])
    pa = PinchAnalyser(pd.concat([four_streams, latent], ignore_index=True), 20).solve()
    d = 1e-4
    glide = table([('cond', 'hot', 100 + d/2, 100 - d/2, 60/d), ('boil', 'cold', 50 - d/2, 50 + d/2, 40/d)])
    ref = PinchAnalyser(pd.concat([four_streams, glide], ignore_index=True), 20).solve()
    assert pa.hot_utility == pytest.approx(ref.hot_utility, abs=1e-3) and pa.hot_utility == pytest.approx(72.5)
    assert pa.cold_utility - pa.hot_utility == pytest.approx(60 - 40 + 180 + 240 - 262.5 - 225)   # energy balance
    cond = pa.composite_curves['hot']
    assert ((cond['T'] == 100.0).sum() == 2) and cond.H[cond['T'] == 100.0].diff().iloc[-1] == pytest.approx(60.0)
def test_integrated_exergy_efficiency():
    """Heat recovery in the methanol loop: the balance stays closed and the efficiency rises."""
    from celeste.CSL.systems.ProcessBoxes import MethanolLoop
    loop = MethanolLoop('loop', n_feed=100.0, T_reac=523.15, P=50.0, T_cond=313.15, purge=0.02)
    loop.run_sol()
    hen = HENSynthesiser(loop.pinch_stream_table(), dt_min=10).solve()
    base = loop.exergy_summary()
    s = loop.exergy_summary(hen=hen)
    ex = loop.hen_exergy(hen)
    assert s.Q_recovered == pytest.approx(hen.network.Q.sum()) and s.Q_recovered > 0
    # the balance residual is the data inconsistency of the units (entropy against exergy), unchanged by the network
    assert s.Ex_balance == pytest.approx(loop.exergy_df().Exbalance_error.sum(), abs=1e-4)
    assert s.Ex_balance == pytest.approx(base.Ex_balance)
    assert (ex.Exdest >= 0).all() and (ex.Ex_hot >= ex.Ex_cold).all()
    assert s.efficiency_no_integration == pytest.approx(base.efficiency)
    assert s.efficiency > base.efficiency and s.Ex_in < base.Ex_in and s.Exloss < base.Exloss
    assert s.Exdest == pytest.approx(base.Exdest + ex.attrs['d_dest'])
    assert s.Ex_product == pytest.approx(base.Ex_product)


def test_integrated_exergy_needs_the_streams_of_the_system():
    from celeste.CSL import media
    from celeste.CSL.systems import ComponentModels
    train = ComponentModels.Ncomp('train', media.Air(), n=20.0, P_comp=[3.0, 9.0])
    train.run_sol()
    cold = pd.DataFrame([dict(segment_type='sensible', side='cold', T_in=20.0, T_out=110.0, CP=0.8, T_phase=np.nan, Q=np.nan)],
                        index=['feedwater'])
    hen = HENSynthesiser(pd.concat([train.pinch_stream_table(min_duty=0.0), cold]), dt_min=10).solve()
    with pytest.raises(ValueError, match='feedwater'):
        train.exergy_summary(hen=hen)
