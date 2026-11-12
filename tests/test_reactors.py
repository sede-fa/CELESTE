"""Reactor and electrochemical models: element balances, equilibrium, physical solutions."""
import numpy as np
import pytest

from celeste import solvers
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
    np.testing.assert_allclose(r.F1.Xi, [0.0151, 0.2786, 0.1944, 0.0002, 0.4964, 0.0153], atol=5e-5)
    assert r.Q == pytest.approx(-34.707014, rel=1e-6)            # K from NASA Gibbs energies (was -34.751)
    assert_elements_conserved(r); assert_physical(r.F1)
    x = r.F1.Xi
    assert x[2]*x[0]/(x[3]*x[4]) == pytest.approx(equilibrium.K_WGS(600.0), rel=1e-5)
    assert x[3]*x[0]**3*10.0**2/(x[1]*x[4]) == pytest.approx(equilibrium.K_MSR(600.0), rel=1e-5)
    assert r.Exloss > 0 and r.Sgen > 0 and 0 < r.Exeff < 1
    import scipy.optimize as spo
    assert r.run_sol(method=spo.least_squares).cost < 1e-10      # user supplied solver still accepted


def test_equilibrium_constants_from_nasa_gibbs_energies():
    """K = exp(-dG/RT) from the NASA data; matches the former fits where they were fitted."""
    assert equilibrium.K_WGS(1073.15) == pytest.approx(equilibrium.K_WGS_fit(1073.15), rel=6e-3)     # SOC range
    assert equilibrium.K_rWGS(500.0) == pytest.approx(equilibrium.K_rWGS_Graaf(500.0), rel=5e-2)     # methanol range
    assert equilibrium.K_MeOH_CO2(500.0) == pytest.approx(equilibrium.K_MeOH_CO2_Graaf(500.0), rel=0.25)  # Graaf: 22 % lower
    assert equilibrium.K_rWGS(700.0)*equilibrium.K_WGS(700.0) == pytest.approx(1.0)
    # CO + 2 H2 -> CH3OH is CO2 + 3 H2 -> CH3OH + H2O plus the reverse shift
    assert equilibrium.K_MeOH_CO(550.0) == pytest.approx(equilibrium.K_MeOH_CO2(550.0)*equilibrium.K_WGS(550.0), rel=1e-10)
    import jax, jax.numpy as jnp
    assert float(jax.jit(lambda t: equilibrium.K_WGS(t, jnp.exp))(800.0)) == pytest.approx(equilibrium.K_WGS(800.0), rel=1e-5)


HB_FEED = np.array([0.74, 0.25, 0.01])


def test_haber_bosch_isothermal_and_adiabatic():
    r = reactors.HB('r', T=673.15)
    r.F0.set(n=1.0, Xi=HB_FEED, P=200.0)
    assert r.run_sol().success
    np.testing.assert_allclose(r.F1.Xi, [0.4709, 0.1615, 0.3677], atol=5e-5)
    assert r.F1.T == 673.15 and r.Q < 0
    assert_elements_conserved(r, 'HN'); assert_physical(r.F1)
    assert r.xi[0] == pytest.approx(r.F1.n*r.F1.Xi[2] - r.F0.n*r.F0.Xi[2], rel=1e-9)

    a = reactors.HB('a', Q=0.0)                                   # same model, adiabatic
    assert a.run_sol().success
    np.testing.assert_allclose(a.F1.Xi, [0.6627, 0.2246, 0.1127], atol=5e-5)
    assert a.F1.T == pytest.approx(838.505, abs=1e-2)             # ignited steady state
    assert a.Q == pytest.approx(0.0, abs=1e-6) and 0 < a.conversion < 1 and a.Sgen > 0

    c = reactors.HB('c', Q=-5.0)                                  # cooled bed
    assert c.run_sol().success and c.Q == pytest.approx(-5.0, rel=1e-8) and a.F1.T > c.F1.T > 673.15
    with pytest.warns(solvers.ConvergenceWarning):
        assert not reactors.HB('d', Q=-50.0).run_sol().success     # duty cannot be met
    with pytest.raises(ValueError):
        reactors.HB('e', T=600.0, Q=0.0)


def test_haber_bosch_former_interfaces():
    from celeste._deprecation import CelesteDeprecationWarning
    with pytest.warns(CelesteDeprecationWarning):
        old = reactors.HB_Adiabatic('o', F0_Xi=HB_FEED)
    assert isinstance(old, reactors.HB) and old.run_sol().success
    assert old.F1.T == pytest.approx(838.505, abs=1e-2)
    assert old.for_sol.__func__ is old.f_sol.__func__
    with pytest.warns(CelesteDeprecationWarning):
        legacy = reactors.HB('l', F0_n=2.0, F0_Xi=HB_FEED, P=150.0, F1_n=2.0)
    assert (legacy.F0.n, legacy.F0.P) == (2.0, 150.0)


def test_haber_bosch_plug_flow_bed():
    bed = reactors.HB_1D('b', Vol=1.0)                               # adiabatic bed
    assert bed.run_sol().success
    assert bed.Q == pytest.approx(0.0, abs=1e-6)                     # exact energy balance
    assert bed.F1.T == pytest.approx(838.5, abs=0.5)                 # approaches the adiabatic equilibrium
    assert np.all(np.diff(bed.profile['T']) >= -1e-9) and np.all(np.diff(bed.profile['x_NH3']) >= -1e-12)
    assert_elements_conserved(bed, 'HN')
    small = reactors.HB_1D('s', Vol=1e-4)
    small.run_sol()
    assert 673.15 < small.F1.T < bed.F1.T
    cooled = reactors.HB_1D('c', Vol=1.0, UA_V=0.5, T_wall=600.0)
    cooled.run_sol()
    assert cooled.Q == pytest.approx(cooled.Q_wall, rel=1e-6) and cooled.Q < 0 and cooled.F1.T < bed.F1.T
    assert cooled.Exloss > 0


def _meoh(**kw):
    r = reactors.MeOH('r', **kw)
    r.F0.set(n=100, Xi=MEOH_FEED, T=473.15, P=50.0)
    return r


@pytest.mark.parametrize('kw, x_ref', [
    (dict(model='equilibrium', T=473.15), [0.6342, 0.0660, 0.0965, 0.0073, 0.0356, 0.0, 0.1604]),
    (dict(model='yield', T=473.15), [0.7017, 0.0581, 0.1087, 0.0425, 0.0076, 0.0, 0.0814]),
    (dict(model='yield', Q=0.0), [0.6674, 0.0581, 0.0744, 0.0768, 0.0419, 0.0, 0.0814]),   # former MeOH_WGS
])
def test_methanol_reactor(kw, x_ref):
    r = _meoh(**kw)
    sol = r.run_sol()
    assert sol.success
    np.testing.assert_allclose(r.F1.Xi, x_ref, atol=5e-5)
    assert_elements_conserved(r); assert_physical(r.F1)
    assert r.Exloss > 0 and 0 < r.Exeff < 1 and r.Sgen > 0
    if 'T' in kw:
        assert r.F1.T == kw['T'] and r.Q < 0
    else:
        assert r.Q == pytest.approx(0.0, abs=1e-6) and r.F1.T == pytest.approx(650.648, abs=5e-3)
    assert set(r.conversion) == {'H2 Conversion', 'CO2 Conversion', 'Carbon Conversion'}


def test_methanol_equilibrium_satisfies_both_mass_action_laws():
    r = _meoh(model='equilibrium', Q=0.0)
    assert r.run_sol().success
    x, T, P = r.F1.Xi, r.F1.T, r.F1.P
    assert x[6]*x[4]/(x[2]*x[0]**3*P**2) == pytest.approx(equilibrium.K_MeOH_CO2(T), rel=1e-6)
    assert x[3]*x[4]/(x[2]*x[0]) == pytest.approx(equilibrium.K_rWGS(T), rel=1e-6)
    assert 473.15 < T < 650.0 and r.Q == pytest.approx(0.0, abs=1e-6)


def test_methanol_kinetic_reactor_returns_a_physical_state():
    """The unbounded solve used to converge to negative water and methanol fractions."""
    r = _meoh(model='kinetic', M_cat=1000, T=473.15)
    assert r.run_sol().success
    assert_physical(r.F1); assert_elements_conserved(r)
    assert r.F1.Xi[6] > 0.005 and r.F1.Xi[4] > 0.0
    k_b = r._K_b
    r.T = 500.0
    assert r._K_b != k_b                                           # adsorption constants follow the temperature
    big = _meoh(model='kinetic', M_cat=1e5, T=473.15)              # a large bed approaches equilibrium
    big.run_sol()
    eq_ = _meoh(model='equilibrium', T=473.15)
    eq_.run_sol()
    assert eq_.F1.Xi[6] > big.F1.Xi[6] > r.F1.Xi[6]
    adiabatic = _meoh(model='kinetic', M_cat=1000, Q=0.0)          # ignited steady state
    assert adiabatic.run_sol().success and adiabatic.F1.T > 540.0


def test_methanol_former_classes():
    from celeste._deprecation import CelesteDeprecationWarning
    for cls, kw in [(reactors.MeOH_Equil, {}), (reactors.MeOH_Yield, {}), (reactors.MeOH_Kinetic, dict(M_cat=1000)),
                    (reactors.MeOH_WGS, {})]:
        with pytest.warns(CelesteDeprecationWarning):
            r = cls('r', F0_n=100, F0_Xi=MEOH_FEED, **kw)
        assert isinstance(r, reactors.MeOH) and r.run_sol().success
    assert r.F1.T == pytest.approx(650.648, abs=5e-3)                  # MeOH_WGS is adiabatic
    with pytest.warns(CelesteDeprecationWarning):
        r.ExBs()


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
