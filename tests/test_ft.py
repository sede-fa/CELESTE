"""Fischer-Tropsch module: properties match the original implementation, the units run."""
import numpy as np
import pytest

from celeste.FT import componentsFT, flowFT, heatFT, interfacesFT, mediaFT, reactorsFT

FT_MEDIA = ['Syncrude', 'Wax', 'Naphtha', 'LightGas', 'Kerosene', 'Distillate', 'Water', 'Hydrogen', 'Alkanes']


@pytest.mark.parametrize('name', FT_MEDIA)
def test_ft_properties_match_original(golden, name):
    for row in golden['ft'][name]:
        f = interfacesFT.Flange('f', getattr(mediaFT, name)(), T=row['T'], P=row['P'], phase=row['phase'])
        assert f.cp == pytest.approx(row['cp'], rel=1e-10)
        assert f.h == pytest.approx(row['h'], rel=1e-10)
        assert f.s == pytest.approx(row['s'], rel=1e-10)
        assert f.LHV == pytest.approx(row['lhv'], rel=1e-12)
        assert f.Ex == pytest.approx(row['ex'], rel=1e-10)


def test_reference_composition_is_kept():
    """Medium_Base used to overwrite the reference composition of the FT media with None."""
    assert mediaFT.Wax()._reference_X.shape == (28,)
    f = interfacesFT.Flange('f', mediaFT.Syncrude(), phase='Liquid')
    assert f.Xi.shape == (55,) and f.phase == 'Liquid'
    f.Ex
    assert f.phase == 'Liquid'          # evaluating the exergy no longer relabels the stream


def test_ft_reactor_and_separation():
    x = np.zeros(55)
    x[[0, 1, 2, 4]] = 0.62, 0.05, 0.30, 0.03
    r = reactorsFT.FT('r', alpha=0.85, CO_conv=0.6, F0_Xi=x)
    r.run_sol()
    assert r.Q == pytest.approx(-32.627621, rel=1e-6)
    assert r.F0.m == pytest.approx(r.F1.m + r.F2.m, rel=1e-5)
    r.ExBs()
    assert r.Exloss > 0 and 0 < r.Exeff < 1
    g = componentsFT.GPU_2S('g', mediaFT.Syncrude(), F0_Xi=r.F1.Xi)
    g.run_sol()
    assert g.F2.n == pytest.approx(r.F1.Xi[3]) and np.isfinite(g.W)


def test_ft_units_run():
    h = reactorsFT.Hydrocracker('h', mediaFT.Wax())
    assert h.run_sol().success and h.mass_balance() == pytest.approx(0.0, abs=1e-6)
    h.ExBs()
    assert np.isfinite(h.Exloss)
    c = componentsFT.CompIsent('c', mediaFT.LightGas(), eff=0.8)
    assert c.run_sol().success and c.F1.T == pytest.approx(350.848, abs=1e-2)
    m = flowFT.Mixer('m', mediaFT.LightGas())
    m.run_sol()
    assert m.F2.n == 2 and m.mass_balance() == pytest.approx(0.0, abs=1e-9)
    heater = heatFT.Heater('ht', mediaFT.Naphtha(), mediaFT.Naphtha(), F1_T=400.0)
    heater.run_sol()
    assert heater.Q == pytest.approx(20.470005, rel=1e-6)
    chg = componentsFT.ChgMedium('chg')
    chg.run_sol()
    assert chg.F1.Xi.sum() == pytest.approx(1.0)
