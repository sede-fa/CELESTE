"""System models, the reports generated from them and the recycle solvers."""
import numpy as np
import pytest

from celeste.CSL import SysBase, components, flow, heat, interfaces, media
from celeste.CSL.systems import ComponentModels, ElectBox, SOCsystem


def test_simple_systems():
    hs = ComponentModels.SimpleHeaterSys('s', 1.0, None, 300.0, 1.0, 500.0)
    hs.run_sol()
    assert hs.heater.Q == pytest.approx(5.8934622, rel=1e-7) and hs.sink.F0.T == 500.0
    assert hs.MBcheck() == pytest.approx(0.0, abs=1e-12)
    assert hs.EBcheck() == pytest.approx(0.0, abs=1e-4)
    cs = ComponentModels.SimpleCompressorSys('s', 1.0, None, 300.0, 1.0, 5.0)
    cs.run_sol()
    assert cs.compressor.W == pytest.approx(5.0881399, rel=1e-6)
    assert cs.EBcheck() == pytest.approx(0.0, abs=1e-4)


@pytest.fixture
def train():
    sys = ComponentModels.Ncomp('train', media.Air())
    sys.run_sol()
    return sys


def test_compression_train(train):
    assert (train.F1.T, train.F1.P) == (350.0, 3.0)
    assert train.tot_Exloss() == pytest.approx(1.9776524, rel=1e-6)
    assert train.r_get() == ['HT1', 'HT2', 'comp1', 'comp2']
    assert train.get_success() == {}                  # the compressors are explicit: nothing to converge
    report = train.performance_report()
    assert report['total_pressure_ratio'] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        ComponentModels.Ncomp('bad', media.Air(), N_stage=1)
    train.add_stage(0.75, 6.0, 340.0)
    assert train.r_get() == ['HT1', 'HT2', 'HT3', 'comp1', 'comp2', 'comp3']


def test_reports(train):
    heaters = train.heater_df()
    assert list(heaters.index) == ['train.HT1', 'train.HT2'] and (heaters.Q < 0).all()
    with pytest.warns(UserWarning, match='no heat flows'):
        assert train.heat_df().empty
    assert train.tot_heat_in() == 0
    work = train.work_df()
    assert work.name == 'W' and list(work.index) == ['train.comp1', 'train.comp2']
    assert train.tot_work_in() == pytest.approx(work.sum()) and train.tot_work_out() == 0
    assert train.tot_work_comp() == pytest.approx(work.sum())
    assert train.tot_heaters_out() == pytest.approx(heaters.Q.sum()) and train.tot_heaters_in() == 0
    assert train.tot_Qloss() == 0
    ex = train.exdest_df()
    assert ex.loc['Total', 'Ex. destruction / kW'] == pytest.approx(train.tot_Exloss())
    assert ex.loc[['Compressors', 'Coolers'], '%'].sum() == pytest.approx(100.0)
    assert train.exloss_df().shape == (4, 1) and train.SBcheck().shape == (4, 2)
    assert train.ViewAllFlange().shape[1] == 8
    assert train.tot_exth_out() < 0


def test_pinch_stream_table(train):
    table = train.pinch_stream_table(min_duty=0.0)
    assert list(table.index) == ['HT1', 'HT2'] and (table.side == 'hot').all()
    assert table.loc['HT1', 'CP']*(table.loc['HT1', 'T_in'] - table.loc['HT1', 'T_out']) == pytest.approx(-train.HT1.Q)
    assert train.rgetattr(table.index[0]) is train.HT1
    assert list(train.pinch_stream_table(min_duty=0.0, include=['HT2']).index) == ['HT2']
    with pytest.raises(ValueError):
        train.pinch_stream_table(min_duty=1e6)


def test_graph_system_flanges_are_nodes():
    from celeste.CSL.systems.SOCsystem import SOCsysRecupe
    text = SOCsysRecupe('socsys').create_graph('mermaid', save=False)
    assert '\tF0 --C0--> HEX_F' in text and '\tHEX_F --H1--> F1' in text and 'socsys' not in text


def test_graph(train, tmp_path):
    text = train.create_graph('mermaid', save=False)
    assert text.startswith('flowchart LR') and 'comp1 --F1_F0--> HT1' in text and 'train.comp1' not in text
    target = tmp_path / 'g.mmd'
    train.create_graph('mermaid', save=str(target))
    assert target.read_text() == text
    with pytest.raises(ValueError):
        train.create_graph('dot')


@pytest.mark.parametrize('cls', [ElectBox.SOECSys, ElectBox.SOECSysV2, ElectBox.CO2Sys])
def test_soec_boxes(cls):
    box = cls('box', I=-50.0, Ncells=100)
    box.run_sol()
    assert box.soc._sol.success
    assert box.soc.Vop == pytest.approx(1.22128, abs=1e-4)
    assert box.F2.n > 0 and box.A2.T == 298.15
    assert np.isfinite(box.tot_Exloss()) and not box.exdest_df().isna().any().any()
    if hasattr(box, 'WWater'):
        assert box.WWater.Xi == pytest.approx([1.0]) and box.WWater.n > 0


def test_ael_box():
    box = ElectBox.AELSys('box', n=1.0, Sep=np.array([0, 1, 0, -1]))
    box.run_sol()
    assert box.ael._sol.success and box.ael.W == pytest.approx(373.32639, rel=1e-6)
    assert box.F2.n > 0 and np.isfinite(box.tot_Exloss())


def test_soc_example_systems():
    basic = SOCsystem.SOCBasicSys('s', 1.0, None, 1.0, None, -50.0, 100)
    basic.run_sol()
    assert basic.soc.Vop == pytest.approx(1.22128, abs=1e-4)
    assert basic.MBcheck() == pytest.approx(0.0, abs=1e-8)
    assert set(basic.get_Sources()) == {'sa1', 'sf1'} and set(basic.get_Sinks()) == {'sa2', 'sf2'}
    assert basic.flow_df().shape == (4, 6)
    recup = SOCsystem.SOCsysRecupe('s', I=-50.0)
    recup.run_sol()
    assert recup.soc.Vop == pytest.approx(1.39789, abs=1e-4) and recup.F1.T == pytest.approx(400.488, abs=1e-2)
    assert all(recup.get_success().values())


class RecycleLoop(SysBase):
    """Feed + recycle -> mixer -> splitter (purge / recycle): the recycle converges to feed*r/(1 - r)."""
    def __init__(self, name, ratio=0.6):
        super().__init__(locals())
        self.src = flow.SourceM(name + '.src', media.Air(), F1_n=1.0, F1_T=400.0)
        self.mix = flow.Mixer(name + '.mix', media.Air())
        self.mix.F1.n = 0.1
        self.split = flow.Splitter(name + '.split', media.Air(), Split=ratio)
        self.snk = flow.Sink(name + '.snk', media.Air())

    def seq_sol(self, store=False):
        self.connect_seq(self.src.F1, self.mix.F0, store=store)
        self.mix._sol = self.mix.run_sol()
        self.connect_seq(self.mix.F2, self.split.F0, store=store)
        self.split.run_sol()
        self.connect_seq(self.split.F2, self.snk.F0, store=store)


@pytest.mark.parametrize('method, kwargs', [
    ('Recycle_FPI', {}), ('Recycle_Wegstein_q', {'q': -1.0}), ('Recycle_Wegstein_Generic', {}),
    ('Recycle_Steffensen', {}), ('Recycle_Aitken', {}), ('Recycle_Hybrid_FPI_Aitken', {}),
])
def test_former_recycle_solvers(method, kwargs, capsys):
    """The former methods are wrappers of Recycle_Tear and still converge."""
    from celeste._deprecation import CelesteDeprecationWarning
    loop = RecycleLoop('loop')
    with pytest.warns(CelesteDeprecationWarning, match="Recycle_Tear"):
        count, error = getattr(loop, method)(loop.split.F1, loop.mix.F1, print_err=True, **kwargs)
    capsys.readouterr()
    assert error < 1e-5 and count < 1000
    assert loop.split.F1.n == pytest.approx(1.5, rel=1e-4)        # 1 * 0.6 / 0.4
    assert loop.MBcheck() == pytest.approx(0.0, abs=1e-3)


def test_connection_checks():
    sys = RecycleLoop('loop')
    with pytest.raises(ValueError, match='Incorrect connection'):
        sys.connect_seq(sys.src.F1, interfaces.Flange('x', media.Steam()))
    a, b = interfaces.Flange('a', media.Air(), n=2.0, T=500.0), interfaces.Flange('b', media.Air())
    sys.connect_seq(a, b, direction='B')
    assert a.T == 298.15 and a.n == 1
    with pytest.raises(NotImplementedError):
        SysBase({'name': 'empty'}).run_sol()


def test_lazy_subpackages():
    import celeste
    assert celeste.CSL.media.Air().nX == 2
    assert celeste.CSL.flow.SourceM is flow.SourceM
    with pytest.raises(AttributeError):
        celeste.CSL.does_not_exist


def test_boundary_flanges_follow_the_connections():
    loop = RecycleLoop('loop')
    loop.Recycle_Tear(loop.split.F1, loop.mix.F1, eps=1e-10)
    b = loop.boundary_flanges()
    assert b['feeds'] == [loop.mix.F0] and b['products'] == [loop.split.F2]   # the recycle is internal
    assert loop.MBcheck() == pytest.approx(0.0, abs=1e-10)
    assert loop.EBcheck() == pytest.approx(0.0, abs=1e-10)


def test_system_mass_balance_is_in_minus_out():
    hs = ComponentModels.SimpleHeaterSys('s', 1.0, None, 300.0, 1.0, 500.0)
    hs.run_sol()
    hs.heater.F1.n = 0.5                                   # break the balance on purpose
    assert hs.MBcheck() == pytest.approx(0.5*hs.heater.F0.mw)


@pytest.mark.parametrize('make', [
    lambda: ComponentModels.SimpleHeaterSys('s', 1.0, None, 300.0, 1.0, 500.0),
    lambda: ComponentModels.Ncomp('train', media.Air()),
    lambda: ElectBox.CO2Sys('box', I=-50.0, Ncells=100),
    lambda: ElectBox.SOECSys('box', I=-50.0, Ncells=100),     # GPU work dissipated to the surroundings
    lambda: ElectBox.AELSys('box', n=1.0, Sep=np.array([0, 1, 0, -1])),
    lambda: SOCsystem.SOCBasicSys('s', 1.0, None, 1.0, None, -50.0, 100),
    lambda: SOCsystem.SOCsysRecupe('s', I=-50.0),             # heat exchangers: internal duty only
])
def test_energy_balance_closes_over_the_boundary(make):
    s = make()
    s.run_sol()
    residual = s.EBcheck()
    assert residual == pytest.approx(0.0, abs=1e-8*max(1.0, abs(s.q_in), abs(s.q_out)))
    balances = s.unit_balances()
    assert (balances.energy.abs() < 1e-6).all() and (balances.mass.abs() < 1e-6).all()


def test_soc_heat_loss_follows_the_sign_convention():
    """Qloss is heat into the stack like Q and W: a loss is negative and must be made up by Q."""
    def run(qloss):
        s = SOCsystem.SOCBasicSys('s', 1.0, None, 1.0, None, -50.0, 100)
        s.soc.Qloss = qloss
        s.run_sol()
        return s
    ref, lossy = run(0.0), run(-5.0)
    assert lossy.soc.Q == pytest.approx(ref.soc.Q + 5.0, rel=1e-9)
    assert lossy.EBcheck() == pytest.approx(0.0, abs=1e-9)
    # the loss leaves at T0: no exergy with it, so it shows up as destruction and as entropy generation
    assert lossy.soc.Exloss - 298.15*lossy.soc.Sgen == pytest.approx(ref.soc.Exloss - 298.15*ref.soc.Sgen, abs=1e-9)


def test_compression_train_with_other_compressors_and_pressure_ratio():
    from celeste.CSL import turbomachinery
    iso = ComponentModels.Ncomp('t', media.Air(), n=2.0, N_stage=3, P_out=8.0, eff_comp=0.8, T_HT=320.0)
    iso.run_sol()
    assert [c.F1.P for c in iso.compressors] == pytest.approx([2.0, 4.0, 8.0]) and iso.F1.P == pytest.approx(8.0)
    assert iso.F1.T == 320.0 and all(h.F1.T == 320.0 for h in iso.heaters)
    poly = ComponentModels.Ncomp('t', media.Air(), n=2.0, N_stage=3, P_out=8.0, eff_comp=0.8, T_HT=320.0,
                                 compressor=turbomachinery.CompPolytropic)
    poly.run_sol()
    assert all(isinstance(c, turbomachinery.CompPolytropic) for c in poly.compressors)
    assert poly.tot_work_comp() > iso.tot_work_comp()                       # polytropic < isentropic efficiency for the same value
    mixed = ComponentModels.Ncomp('t', media.Air(), N_stage=2, P_comp=[2.0, 4.0], eff_comp=[0.8, 0.7],
                                  compressor=[turbomachinery.CompIsent, turbomachinery.CompIsot])
    mixed.run_sol()
    assert isinstance(mixed.comp2, turbomachinery.CompIsot) and mixed.comp2.eff == 0.7
    mapped = ComponentModels.Ncomp('t', media.Air(), N_stage=2, P_comp=[2.0, 4.0], eff_comp=0.8,
                                   compressor=lambda name, Medium, eff: turbomachinery.CompressorMapped(
                                       name, None, Medium, PR=2.0, eta=eff, use_map=False))
    mapped.run_sol()
    assert mapped.F1.P == pytest.approx(4.0)
    mixed.add_stage(0.75, 8.0, 330.0, turbomachinery.CompPolytropic)
    mixed.run_sol()
    assert mixed.N == 3 and mixed.F1.T == 330.0
    cs = ComponentModels.SimpleCompressorSys('s', 1.0, None, 300.0, 1.0, 5.0, eff=0.8, compressor=turbomachinery.CompPolytropic)
    cs.run_sol()
    ref = ComponentModels.SimpleCompressorSys('s', 1.0, None, 300.0, 1.0, 5.0, eff=0.8)
    ref.run_sol()
    assert cs.compressor.F1.T > ref.compressor.F1.T


# ---- flowsheets declared by their streams --------------------------------------------------------------

class _TwoRecycles(SysBase):
    """feed -> mix1 -> mix2 -> heater -> split2 -> split1 -> product; split2.F1 -> mix2, split1.F1 -> mix1."""
    def __init__(self, name, declared=True):
        super().__init__(locals())
        gas = media.Air()
        self.feed = flow.SourceM(name + '.feed', gas, F1_n=1.0, F1_Xi=np.array([0.21, 0.79]), F1_T=300.0, F1_P=1.0)
        self.mix1 = flow.Mixer(name + '.mix1', gas)
        self.mix2 = flow.Mixer(name + '.mix2', gas)
        self.heater = heat.Heater(name + '.heater', gas)
        self.heater.F1.T = 500.0
        self.split2 = flow.Splitter(name + '.split2', gas, Split=0.4)
        self.split1 = flow.Splitter(name + '.split1', gas, Split=0.3)
        self.out = flow.Sink(name + '.out', gas)
        for f in (self.mix1.F1, self.mix2.F1):
            f.n, f.T = 1e-6, 300.0
        if declared:
            self.link(self.feed.F1, self.mix1.F0)
            self.link(self.mix1.F2, self.mix2.F0)
            self.link(self.mix2.F2, self.heater.F0)
            self.link(self.heater.F1, self.split2.F0)
            self.link(self.split2.F2, self.split1.F0)
            self.link(self.split2.F1, self.mix2.F1)
            self.link(self.split1.F1, self.mix1.F1)
            self.link(self.split1.F2, self.out.F0)

    def hand_pass(self):
        self.connect_seq(self.feed.F1, self.mix1.F0)
        self.solve(self.mix1)
        self.connect_seq(self.mix1.F2, self.mix2.F0)
        self.solve(self.mix2)
        self.connect_seq(self.mix2.F2, self.heater.F0)
        self.solve(self.heater)
        self.connect_seq(self.heater.F1, self.split2.F0)
        self.solve(self.split2)
        self.connect_seq(self.split2.F2, self.split1.F0)
        self.solve(self.split1)
        self.connect_seq(self.split1.F2, self.out.F0)


def test_declared_flowsheet_generates_order_and_tears():
    sys_ = _TwoRecycles('t')
    plan = sys_.solve_order()
    assert [len(step.units) for step in plan] == [1, 5, 1]
    assert len(plan[1].tears) == 1 and 'loop' in sys_.print_solve_order()
    sys_.run_sol()
    # every recycle closed: out = feed, and the heater carries feed/(0.7*0.6) ... checked against a hand solution
    ref = _TwoRecycles('r', declared=False)
    ref.Recycle_Tear([ref.split2.F1, ref.split1.F1], [ref.mix2.F1, ref.mix1.F1], flow_method='hand_pass', eps=1e-10)
    assert sys_.out.F0.n == pytest.approx(1.0, rel=1e-8) and sys_.heater.F0.n == pytest.approx(ref.heater.F0.n, rel=1e-8)
    assert sys_.heater.F0.n == pytest.approx(1.0/(0.6*0.7), rel=1e-7)
    assert sys_.EBcheck() == pytest.approx(0.0, abs=1e-6)
    assert 'split1 --F1_F1--> mix1' in sys_.create_graph('mermaid', save=False)


def test_links_from_a_hand_written_run_give_the_declared_flowsheet():
    ref = _TwoRecycles('r', declared=False)
    ref.Recycle_Tear([ref.split2.F1, ref.split1.F1], [ref.mix2.F1, ref.mix1.F1], flow_method='hand_pass', eps=1e-10)
    pairs = ref.links_from_run()
    assert len(pairs) == 8 and len(ref.streams) == 8
    declared = _TwoRecycles('d')
    assert [len(step.units) for step in ref.solve_order()] == [len(step.units) for step in declared.solve_order()]
    ref.mix1.F1.n, ref.mix2.F1.n = 1e-6, 1e-6
    ref.run_sol()                                          # generated order and tears from the recorded streams
    declared.run_sol()
    assert ref.heater.F0.n == pytest.approx(declared.heater.F0.n, rel=1e-8)


def test_iterate_flange_of_the_system_is_not_a_stream_of_its_own():
    class Chain(SysBase):
        def __init__(self, name):
            super().__init__(locals())
            gas = media.Air()
            self.a = heat.Heater(name + '.a', gas)
            self.b = heat.Heater(name + '.b', gas)
            self.guess = interfaces.Flange(name + '.guess', gas)          # iterate of a tear stream: a.F1 -> guess -> b.F0
            self.getObjs()

    chain = Chain('c')
    chain.connect_seq(chain.a.F1, chain.guess)
    chain.connect_seq(chain.guess, chain.b.F0)
    assert chain.links_from_run() == [(chain.a.F1, chain.b.F0)]


def test_declared_flowsheet_rejects_foreign_flanges():
    sys_ = _TwoRecycles('t')
    other = flow.Sink('x', media.Air())
    sys_.link(sys_.split1.F2, other.F0)
    with pytest.raises(ValueError, match='does not belong'):
        sys_.solve_order()


def test_design_specs():
    from celeste.CSL import reactors
    from celeste.CSL.systems import ProcessBoxes

    class Reforming(SysBase):
        def __init__(self, name):
            super().__init__(locals())
            gas = media.Fuel_CH4()
            self.source1 = flow.SourceM('source1', gas, 1, np.array([0.01, 0.3, 0.2, 1e-5, 0.5, 1e-5]), 973.15, 5.)
            self.reformer = reactors.MR('reformer', gas, T=973.15)
            self.sink1 = flow.Sink('sink1', gas)
            self.link(self.source1.F1, self.reformer.F0)
            self.link(self.reformer.F1, self.sink1.F0)

    r = Reforming('r')
    r.add_spec('sink1.F0.n', 2.0, vary='source1.F1.n', bounds=(0.1, 10.0))
    assert r.solve_specs().success and r.sink1.F0.n == pytest.approx(2.0, rel=1e-8)
    assert list(r.specs['vary']) == ['source1.F1.n'] and r.specs['current'][0] == pytest.approx(2.0)
    # two specifications at once, one of them a function of the system
    r.add_spec(lambda s: s.sink1.F0.Xi[0], 0.45, vary='reformer.T', bounds=(800.0, 1200.0), name='H2 fraction')
    assert r.solve_specs().success
    assert r.sink1.F0.n == pytest.approx(2.0, rel=1e-7) and r.sink1.F0.Xi[0] == pytest.approx(0.45, rel=1e-7)
    r.clear_specs()
    with pytest.raises(ValueError):
        r.solve_specs()
    with pytest.raises(AttributeError):
        r.add_spec('sink1.F0.n', 1.0, vary='source1.nothing')
    # a recycle loop: purge fraction for a given carbon efficiency
    loop = ProcessBoxes.MethanolLoop('m')
    loop.add_spec('carbon_efficiency', 0.92, vary='split.split', bounds=(0.9, 0.999))
    assert loop.solve_specs().success and loop.carbon_efficiency == pytest.approx(0.92, abs=1e-7)


def test_pfd(tmp_path):
    import json
    import re
    from celeste.CSL.systems import ProcessBoxes
    loop = ProcessBoxes.MethanolLoop('m')
    drawn = loop.pfd()                                   # declared flowsheet: drawn without solving
    assert '8 units, 8 streams; 1 back stream' in drawn.html and 'class="stream recycle"' in drawn.html
    loop.run_sol()
    path = loop.pfd(str(tmp_path/'m.html')).save(str(tmp_path/'m2.html'))
    page = open(path, encoding='utf-8').read()
    tips = json.loads(re.search(r'const TIPS = (\{.*?\});\n', page).group(1))
    stream = next(v for v in tips.values() if 'split.F1' in v and 'mix.F1' in v)
    assert 'mole fractions' in stream and 'CH3OH' in stream
    assert any('Ex loss' in v and 'MeOH' in v for v in tips.values())
    assert '<iframe srcdoc=' in drawn._repr_html_()
    soc = SOCsystem.SOCsysRecupe('s', I=-50.0)           # hand-written seq_sol: drawn from the last run
    with pytest.raises(ValueError, match='nothing to draw'):
        soc.pfd()
    soc.run_sol()
    assert '2 back stream' in soc.pfd().html


def test_equations_listing():
    from celeste.CSL.systems.SOCsystem import SOCsysRecupe
    s = SOCsysRecupe('socsys', I=10.0, Ncells=1000, na0=0.5)
    s.run_sol()
    df = s.equations().set_index('unit')
    assert df.loc['soc', 'solved_as'] == 'equations' and df.loc['soc', 'equations'] == 2 and df.loc['soc', 'residual'] < 1e-8
    assert 'WGS and MSR equilibrium' in df.loc['soc', 'equation']
    assert df.loc['HT_F0', 'solved_as'] == 'algorithmic'
    text = s.print_equations()
    assert 'soc (SOC): 2 equations in 2 unknowns' in text and 'HT_F0 (Heater)' in text


# ---- equation-oriented solve on top of the declared streams --------------------------------------------

def test_eo_solve_matches_the_sequential_solve_on_recycles():
    sys_ = _TwoRecycles('t')
    eo = sys_.eo_compile()
    assert eo.n_unknowns == eo.n_equations and len(eo.streams) == 8
    res = sys_.eo_sol(sweep_kw=dict(eps=1e-2))                # a loose sweep is enough to start from
    assert res.success and res.residual_norm < 1e-9
    assert sys_.heater.F0.n == pytest.approx(1.0/(0.6*0.7), rel=1e-8)
    assert sys_.out.F0.n == pytest.approx(1.0, rel=1e-8)
    assert 'unknown' in eo.variables.columns and eo.variables['value'].notna().all()


def test_eo_solve_of_the_soc_system_converges_from_a_perturbed_start():
    from celeste.CSL.systems.SOCsystem import SOCsysRecupe
    s = SOCsysRecupe('socsys', I=10., Ncells=1000, na0=0.5)
    s.run_sol()
    ref = (s.soc.F1.T, s.soc.F1.n, s.HEX_A.C1.T)
    s.links_from_run()
    eo = s.eo_compile()
    assert 'unknowns' in repr(eo) and len(eo.solved) == 3     # SOC extents and the two heat exchangers
    eo.initialise()
    eo.x = eo.x*(1 + 0.05*np.random.default_rng(1).standard_normal(eo.x.size))
    res = eo.solve()
    assert res.success
    assert s.soc.F1.T == pytest.approx(ref[0], abs=1e-6) and s.soc.F1.n == pytest.approx(ref[1], rel=1e-8)
    assert s.HEX_A.C1.T == pytest.approx(ref[2], abs=1e-6)


def test_design_specs_with_the_equation_oriented_solve():
    from celeste.CSL import reactors

    class Reforming(SysBase):
        def __init__(self, name):
            super().__init__(locals())
            gas = media.Fuel_CH4()
            self.source1 = flow.SourceM('source1', gas, 1, np.array([0.01, 0.3, 0.2, 1e-5, 0.5, 1e-5]), 973.15, 5.)
            self.reformer = reactors.MR('reformer', gas, T=973.15)
            self.sink1 = flow.Sink('sink1', gas)
            self.link(self.source1.F1, self.reformer.F0)
            self.link(self.reformer.F1, self.sink1.F0)

    r = Reforming('r')
    r.add_spec('sink1.F0.n', 2.0, vary='source1.F1.n', bounds=(0.1, 10.0))
    assert r.solve_specs(eo=True).success and r.sink1.F0.n == pytest.approx(2.0, rel=1e-7)
    r.add_spec(lambda s: s.sink1.F0.Xi[0], 0.45, vary='reformer.T', bounds=(800.0, 1200.0), name='H2 fraction')
    assert r.solve_specs(eo=True).success
    assert r.sink1.F0.n == pytest.approx(2.0, rel=1e-6) and r.sink1.F0.Xi[0] == pytest.approx(0.45, rel=1e-6)
