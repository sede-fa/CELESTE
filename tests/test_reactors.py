"""Reactor and electrochemical models: element balances, equilibrium, physical solutions."""
import numpy as np
import pytest

from celeste.CSL import electrochem, equilibrium, media, reactors
from conftest import element_flow

MEOH_FEED = np.array([0.75, 0.05, 0.1, 0.1, 1e-8, 1e-8, 1e-8])


def assert_elements_conserved(unit, elements='CHON', rel=1e-7):
    for el in elements:
        n_in = sum(element_flow(f, el) for f in unit.inlets)
        n_out = sum(element_flow(f, el) for f in unit.outlets)
        assert n_out == pytest.approx(n_in, rel=rel, abs=1e-9), el


def assert_physical(flange):
    assert flange.n > 0 and np.all(flange.Xi >= 0) and np.sum(flange.Xi) == pytest.approx(1.0, abs=1e-7)


def test_methanation_equilibrium():
    r = reactors.MR('r', F0_Xi=np.array([0.6, 0.01, 0.2, 0.1, 0.08, 0.01]), T=600.0, P=10.0)
    sol = r.run_sol()
    assert sol.success and sol.residual_norm < 1e-7
    np.testing.assert_allclose(r.F1.Xi, [0.0136, 0.2792, 0.1942, 0.0002, 0.4975, 0.0153], atol=5e-5)
    assert r.Q == pytest.approx(-34.751232, rel=1e-6)
    assert_elements_conserved(r); assert_physical(r.F1)
    x = r.F1.Xi
    assert x[2]*x[0]/(x[3]*x[4]) == pytest.approx(equilibrium.K_WGS(600.0), rel=1e-5)
    assert x[3]*x[0]**3*10.0**2/(x[1]*x[4]) == pytest.approx(equilibrium.K_MSR(600.0), rel=1e-5)
    assert r.Exloss > 0 and r.Sgen > 0 and 0 < r.Exeff < 1
    import scipy.optimize as spo
    assert r.run_sol(method=spo.least_squares).cost < 1e-10      # user supplied solver still accepted


def test_haber_bosch():
    r = reactors.HB('r', F0_Xi=np.array([0.74, 0.25, 0.01]))
    assert r.run_sol().success
    np.testing.assert_allclose(r.F1.Xi, [0.4722, 0.1619, 0.3658], atol=5e-5)
    assert_elements_conserved(r, 'HN'); assert_physical(r.F1)

    a = reactors.HB_Adiabatic('a', F0_Xi=np.array([0.74, 0.25, 0.01]))
    assert a.run_sol().success
    np.testing.assert_allclose(a.F1.Xi, [0.6627, 0.2246, 0.1127], atol=5e-5)
    assert a.F1.T == pytest.approx(838.505, abs=1e-2)
    assert a.F0.n*a.F0.h == pytest.approx(a.F1.n*a.F1.h, abs=1e-6)       # adiabatic
    assert 0 < a.conversion < 1 and a.Sgen > 0
    assert a.for_sol.__func__ is a.f_sol.__func__                           # former method name still works


@pytest.mark.parametrize('cls, x_ref', [
    (reactors.MeOH_Equil, [0.6404, 0.0655, 0.0986, 0.0087, 0.0323, 0.0, 0.1546]),
    (reactors.MeOH_Yield, [0.7014, 0.0581, 0.1084, 0.0428, 0.0079, 0.0, 0.0814]),
])
def test_methanol_isothermal_reactors(cls, x_ref):
    r = cls('r', F0_n=100, F0_Xi=MEOH_FEED)
    sol = r.run_sol()
    assert sol.success
    np.testing.assert_allclose(r.F1.Xi, x_ref, atol=5e-5)
    assert_elements_conserved(r); assert_physical(r.F1)
    assert r.Q < 0 and r.Exloss > 0 and 0 < r.Exeff < 1          # available without calling ExBs()
    from celeste._deprecation import CelesteDeprecationWarning
    with pytest.warns(CelesteDeprecationWarning):
        r.ExBs()


def test_methanol_kinetic_reactor_returns_a_physical_state():
    """The unbounded solve used to converge to negative water and methanol fractions."""
    r = reactors.MeOH_Kinetic('r', M_cat=1000, F0_n=100, F0_Xi=MEOH_FEED)
    sol = r.run_sol()
    assert sol.success
    assert_physical(r.F1); assert_elements_conserved(r)
    assert r.F1.Xi[6] > 0.005 and r.F1.Xi[4] > 0.0
    k_b = r._K_b
    r.T = 500.0
    assert r._K_b != k_b                                           # adsorption constants follow the temperature


def test_methanol_adiabatic_reactor():
    r = reactors.MeOH_WGS('r', F0_n=100, F0_Xi=MEOH_FEED)
    assert r.run_sol().success
    np.testing.assert_allclose(r.F1.Xi, [0.666, 0.0581, 0.0729, 0.0782, 0.0433, 0.0, 0.0814], atol=5e-5)
    assert r.F1.T == pytest.approx(649.082, abs=5e-3) and r.Q == pytest.approx(0.0, abs=1e-5)
    assert_elements_conserved(r)
    assert set(r.conversion) == {'H2 Conversion', 'CO2 Conversion', 'Carbon Conversion'}


def test_reformer():
    r = reactors.Reformer('r', media.media_C3(), F0_Xi=np.array([1e-5, 0.2, 1e-5, 1e-5, 0.7, 1e-5, 0.1]), T=900.0)
    sol = r.run_sol()
    assert sol.success and np.isfinite(r.F1.T)
    line = reactors.Reformer_1D('l', media.media_C3(), F0_Xi=np.array([1e-5, 0.2, 1e-5, 1e-5, 0.7, 1e-5, 0.1]), T=900.0, N=3)
    line.run_sol()
    assert line.getArrAttr('F1.T').shape == (3,) and line.Xd.shape == (3, 7)


def test_solid_oxide_cell():
    soc = electrochem.SOC('s', media.Fuel_CH4(), media.Air(), I=-50.0, Ncells=100)
    sol = soc.run_sol()
    assert sol.success
    assert soc.F1.Xi[0] == pytest.approx(0.1259, abs=1e-4) and soc.Vop == pytest.approx(1.3256, abs=1e-4)
    assert soc.W == pytest.approx(6.628, abs=1e-3)
    assert soc.mass_balance() == pytest.approx(0.0, abs=1e-8)
    assert soc.F0.n*np.dot(soc.F0.Xi, soc._N_H) == pytest.approx(soc.F1.n*np.dot(soc.F1.Xi, soc._N_H), rel=1e-8)
    assert soc.Exloss > 0 and soc.j == -0.5 and 0 < soc.RC < 1
    # alternative spellings used by the system models
    alt = electrochem.SOC('s', media.Fuel_CH4(), media.Air(), T_soc0=1000.0, T_soc1=1050.0, P_soc=3.0, I=-10.0, Ncells=10)
    assert (alt.F0.T, alt.F1.T, alt.F0.P, alt.A1.P) == (1000.0, 1050.0, 3.0, 3.0) and alt.Tsoc0 == 1000.0


def test_alkaline_electrolyser():
    ael = electrochem.AEL('a', media.Media_AEL(), Ncells=100, A=800, T=350.0)
    sol = ael.run_sol()
    assert sol.success
    assert ael.j == pytest.approx(-2.4072959, rel=1e-6) and ael.W == pytest.approx(772.5611, rel=1e-6)
    assert ael.mass_balance() == pytest.approx(0.0, abs=1e-6)
