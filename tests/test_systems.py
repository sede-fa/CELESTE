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
    assert cs.compressor.W == pytest.approx(5.0717949, rel=1e-6)
    assert cs.EBcheck() == pytest.approx(0.0, abs=1e-4)


@pytest.fixture
def train():
    sys = ComponentModels.Ncomp('train', media.Air())
    sys.run_sol()
    return sys


def test_compression_train(train):
    assert (train.F1.T, train.F1.P) == (350.0, 3.0)
    assert train.tot_Exloss() == pytest.approx(1.972713, rel=1e-6)
    assert train.r_get() == ['HT1', 'HT2', 'comp1', 'comp2']
    assert train.get_success() == {'comp1': True, 'comp2': True}
    report = train.performance_report()
    assert report['total_pressure_ratio'] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        ComponentModels.Ncomp('bad', media.Air(), N_stage=1)
    train.add_stage(0.75, 6.0, 340.0)
    assert train.r_get() == ['HT1', 'HT2', 'HT3', 'comp1', 'comp2', 'comp3']


def test_reports(train):
    heaters = train.heater_df()
    assert list(heaters.index) == ['train.HT1', 'train.HT2'] and (heaters.Q < 0).all()
    assert train.heat_df().empty and train.tot_heat_in() == 0
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


def test_graph(train, tmp_path):
    text = train.create_graph('mermaid', save=False)
    assert text.startswith('flowchart LR') and 'train.comp1 --F1_F0--> train.HT1' in text
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
        self.mix.F1.set(n=0.1)
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
def test_recycle_solvers(method, kwargs, capsys):
    loop = RecycleLoop('loop')
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
