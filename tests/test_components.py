"""Unit models of the standard library: balances, second law and reference values."""
import numpy as np
import pytest

from celeste.CSL import components, flow, heat, interfaces, media

AIR = np.array([0.21, 0.79])


def test_flange_basics():
    f = interfaces.Flange('f', media.Air(), n=2.0, Xi=AIR, T=300.0, P=1.0)
    assert f.m == pytest.approx(2.0*f.mw)
    assert f.h == pytest.approx(0.0539808, rel=1e-5) and f.Ex == pytest.approx(0.10489, rel=1e-4)
    f.m = 100.0
    assert f.n == pytest.approx(100.0/f.mw)
    table = f.get_nXTP()
    assert table['T /K'] == 300.0 and 'Xi_O2' in table.index
    assert 'n /mol/s' in repr(f)
    g = interfaces.Flange('g', media.Air())
    assert f == g and f != interfaces.Flange('s', media.Steam())


def test_flange_temperature_from_enthalpy_and_entropy():
    f = interfaces.Flange('f', media.Air(), Xi=AIR, T=700.0, P=4.0)
    h, s = f.h, f.s
    f.T = 300.0
    assert f.T_hx(h, AIR).success and f.T == pytest.approx(700.0, abs=1e-7)
    f.T = 1500.0
    f.T_sPX(s, 4.0, AIR)
    assert f.T == pytest.approx(700.0, abs=1e-6)


def test_heater():
    h = heat.Heater('h', media.Air(), media.Air())
    h.F0.set(n=1, Xi=np.array([0.6, 0.4]))
    h.F1.set(T=800.0)
    h.run_sol()
    assert h.Q == pytest.approx(15.5211664, rel=1e-8)            # value of the docstring example
    assert h.Q == pytest.approx(h.F1.n*h.F1.h - h.F0.n*h.F0.h)
    assert h.mass_balance() == pytest.approx(0.0, abs=1e-12)
    assert h.Sgen > 0 and h.Exloss > 0


def test_mixer_energy_and_mass_balance():
    m = flow.Mixer('m', media.Air())
    m.F0.set(n=1)
    m.F1.set(n=2)
    m.F0.T, m.F1.T = 300.0, 600.0
    sol = m.run_sol()
    assert sol.success
    assert m.F2.T == pytest.approx(501.638728, abs=1e-5)
    assert m.Q == pytest.approx(0.0, abs=1e-8) and m.mass_balance() == pytest.approx(0.0, abs=1e-10)
    assert m.Sgen > 0 and m.Exloss > 0
    m.F1.T = 300.0
    assert m.run_sol().success and m.F2.T == pytest.approx(300.0)     # equal temperatures: explicit result


def test_splitter():
    s = flow.Splitter('s', media.Air(), Split=0.3)
    s.F0.set(n=5.0)
    s.run_sol()
    assert (s.F1.n, s.F2.n) == pytest.approx((1.5, 3.5))
    assert s.Exloss == pytest.approx(0.0, abs=1e-12) and s.MBcheck() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize('eff', [1.0, 0.8])
def test_isentropic_compressor(eff):
    c = components.CompIsent('c', media.Air(), eff=eff)
    c.F1.set(P=5.0)
    sol = c.run_sol()
    assert sol.success
    assert c.W == pytest.approx(c.F0.n*(c.F1.h - c.F0.h))
    if eff == 1.0:
        assert c.Sgen == pytest.approx(0.0, abs=1e-4)        # mean-cp model: reversible up to the cp variation
    else:
        assert c.F1.T == pytest.approx(511.4323892, abs=1e-5) and c.W == pytest.approx(6.289629, rel=1e-6)
        assert c.Sgen > 0 and c.Exloss > 0 and 0 < c.Exeff < 1


def test_turbine_delivers_work():
    t = components.CompIsent('t', media.Air(), eff=0.85)
    t.F0.set(T=900.0, P=8.0)
    t.F1.set(P=1.0)
    assert t.run_sol().success
    assert t.W < 0 and t.F1.T < 900.0 and t.Sgen > 0


def test_isothermal_compressor():
    c = components.CompIsot('c', media.Air(), eff=0.8)
    c.F1.set(P=5.0)
    c.run_sol()
    assert c.F1.T == c.F0.T and c.W == pytest.approx(4.987159283, rel=1e-9) and c.Q == -c.W
    assert c.Exloss == pytest.approx(0.2*c.W, rel=1e-6)


def test_pump():
    p = components.Pump('p')
    sol = p.run_sol()
    assert sol.success
    assert p.W == pytest.approx(0.0035067845, rel=1e-6) and p.F1.T == pytest.approx(298.19654, abs=1e-4)
    assert p.Q == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= p.Exloss <= 1.01*p.W                   # incompressible liquid model: bounded by the pump work


def test_valve():
    v = components.Valve('v', P_out=2.0)
    v.F0.set(Xi=np.array([0.3, 0.1, 0.1, 0.1, 0.3, 0.1]), T=500.0, P=10.0)
    v.run_sol()
    assert v.F1.P == 2.0 and v.F1.T == 500.0 and v.Exloss > 0 and v.Sgen > 0
    assert v.W == 0 and v.Q == 0


@pytest.mark.parametrize("spec", [dict(P_out=2.0), dict(dP=8.0), dict()])
def test_valve_pressure_specification(spec):
    v = components.Valve('v', media.Air(), **spec)
    v.F0.set(Xi=AIR, T=400.0, P=10.0)
    if not spec:
        v.F1.set(P=2.0)                                 # spec on the flange
    v.run_sol()
    assert v.F1.P == pytest.approx(2.0) and v.F1.h == pytest.approx(v.F0.h, abs=1e-12)


def test_valve_is_isenthalpic_for_real_fluids():
    """Throttling a real gas cools it (Joule–Thomson); enthalpy is conserved."""
    v = components.Valve('v', media.Fuel_CH4_PR(), P_out=5.0)
    v.F0.set(n=1.0, Xi=np.array([0.1, 0.8, 0.05, 0.03, 0.01, 0.01]), T=300.0, P=80.0)
    sol = v.run_sol()
    assert sol.success
    assert v.F1.h == pytest.approx(v.F0.h, abs=1e-8)
    assert 260.0 < v.F1.T < 280.0                       # methane: about 0.4 K/bar (JT coefficient)


def test_valve_and_pump_reject_two_pressure_specs():
    with pytest.raises(ValueError):
        components.Valve('v', P_out=1.0, dP=1.0)
    with pytest.raises(ValueError):
        components.Pump('p', P_out=1.0, dP=1.0)


@pytest.mark.parametrize("spec", [dict(P_out=3.0), dict(dP=2.0)])
def test_pump_pressure_specification(spec):
    p = components.Pump('p', eff=0.8, **spec)
    p.run_sol()
    assert p.F1.P == pytest.approx(3.0) and p.W > 0 and p.F1.T > p.F0.T
    assert p.F1.n*p.F1.h - p.F0.n*p.F0.h == pytest.approx(p.W, rel=1e-6)


def test_pump_as_turbine_recovers_work():
    p = components.Pump('t', eff=0.8, P_out=1.0)
    p.F0.set(P=30.0)
    p.run_sol()
    assert p.W < 0


def test_black_box_separators(syngas_x):
    g1 = components.GPU_1S('g', media.Fuel_CH4(), media.Fuel_CH4L(), eff=0.7, F0_n=100, F0_T=873.15, F0_P=3.0, F1_T=350.0)
    g1.run_sol()
    assert np.isfinite(g1.W) and g1.mass_balance() == pytest.approx(0.0, abs=1e-9)

    g2 = components.GPU_2S('g', media.Fuel_CH4())
    g2.run_sol()
    assert (g2.F1.n, g2.F2.n) == pytest.approx((0.1, 0.9)) and g2.W == pytest.approx(0.8057289, rel=1e-6)

    g = components.GPU_2SV2('g', media.Fuel_CH4(), Sep=np.array([1, 1, .5, .4, 0, 1]), F0_Xi=np.array([0.1, 0.3, 0.05, 0.05, 0.5, 1e-8]))
    g.run_sol()
    assert g.F1.n == pytest.approx(0.44500001) and g.W == pytest.approx(1.5333083, rel=1e-6)
    assert g.mass_balance() == pytest.approx(0.0, abs=1e-9) and g.Exloss < 0 < g.W

    g3 = components.GPU_3S('g', media.Fuel_CH4(), sep=np.array([1, 0, 1, 1, -1, 1]), F0_Xi=np.array([0.1, 0.1, 1e-8, 1e-8, 0.8, 1e-8]))
    g3.run_sol()
    assert (g3.F1.n, g3.F2.n, g3.F3.n) == pytest.approx((0.1, 0.1, 0.8), abs=1e-6)
    assert g3.mass_balance() == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match='exactly one'):
        bad = components.GPU_3S('g', media.Fuel_CH4(), sep=np.array([1, 0, 1, 1, 1, 1]))
        bad.run_sol()


def test_asu_and_dac_outlets_follow_the_inlet_temperature():
    a = components.ASU('a')
    a.F0.set(T=310.0)
    a.run_sol()
    assert a.F1.T == a.F2.T == 310.0
    assert a.F1.n == pytest.approx(0.79) and a.W == pytest.approx(8.6043718, rel=1e-7)
    assert a.MBcheck() == pytest.approx(0.0, abs=1e-9)
    d = components.DAC('d')
    d.F0.set(T=290.0)
    d.F0.T = 305.0
    d.run_sol()
    assert d.F0.T == 305.0 and d.F1.T == d.F2.T == 305.0      # the inlet used to be overwritten
    assert d.F2.n == pytest.approx(3e-4) and d.W == pytest.approx(0.019012104, rel=1e-7)


def test_heat_exchanger_ntu():
    hx = heat.HexNTU('hx', media.Air(), media.Air(), UA=50.0)
    hx.H0.T, hx.C0.T = 800.0, 300.0
    sol = hx.run_sol()
    assert sol.success
    assert (hx.H1.T, hx.C1.T) == pytest.approx((494.251066, 617.096598), abs=1e-4)
    assert hx.Q == pytest.approx(hx.H0.n*(hx.H0.h - hx.H1.h)) == pytest.approx(hx.C0.n*(hx.C1.h - hx.C0.h))
    assert 0 < hx.epsilon < 1 and hx.Sgen > 0 and hx.Exloss > 0
    fixed = heat.HexNTU('hx', media.Air(), media.Air(), UA=0.9, HexType='effectiveness')
    fixed.H0.T, fixed.C0.T = 800.0, 300.0
    fixed.run_sol()
    assert fixed.epsilon == 0.9


def test_ports_and_generic_mass_balance(capsys):
    hx = heat.HexNTU('hx', media.Air(), media.Air())
    assert {f.name for f in hx.inlets} == {'hx.H0', 'hx.C0'} and {f.name for f in hx.outlets} == {'hx.H1', 'hx.C1'}
    assert len(hx.get_Flanges()) == 6                      # including the two internal helper flanges
    assert hx.MBcheck() == pytest.approx(0.0)
    assert '% difference' in capsys.readouterr().out
    assert hx.ViewFlange(hx).shape[1] == 6


def test_serialisation():
    h = heat.Heater('h', media.Air(), media.Air())
    h.F1.set(T=500.0)
    d = h.to_dict()
    assert d['name'] == 'h' and '__class__' not in d and d['F1']['T'] == 500.0
    assert h.to_pandas().shape[0] == 1
    assert h.rgetattr('F1.T') == 500.0
    h.rsetattr('F1.T', 600.0)
    assert h.F1.T == 600.0


def test_change_of_medium():
    c = components.ChgMedium('c', media.Fuel_CH4L(), media.Water())
    c.F0.n, c.F0.Xi, c.F0.T = 2.0, np.array([1e-9, 0, 0, 0, 1.0, 0]), 320.0
    c.run_sol()
    assert c.F1.n == 2.0 and c.F1.Xi == pytest.approx([1.0]) and c.F1.T == 320.0
    s = components.ChgMedium('s', media.Fuel_CH4L(), media.Steam())       # H2O(L) maps onto H2O
    s.F0.Xi = np.array([0, 0, 0, 0, 1.0, 0])
    s.run_sol()
    assert s.F1.Xi == pytest.approx([1.0])
    bad = components.ChgMedium('b', media.Air(), media.Water())
    with pytest.raises(ValueError, match='no species in common'):
        bad.run_sol()


@pytest.mark.parametrize("make, port, var, value", [
    (lambda **kw: heat.Heater('h', media.Air(), **kw), 'F1', 'T', 800.0),
    (lambda **kw: components.CompIsent('c', media.Air(), **kw), 'F1', 'P', 5.0),
    (lambda **kw: components.CompIsot('c', media.Air(), **kw), 'F0', 'T', 350.0),
    (lambda **kw: flow.Mixer('m', media.Air(), **kw), 'F1', 'n', 2.0),
    (lambda **kw: flow.Sink('s', media.Air(), **kw), 'F0', 'P', 4.0),
])
def test_former_state_keywords_still_work(make, port, var, value):
    """F*_n/Xi/T/P keywords are deprecated but still land on the flange."""
    from celeste._deprecation import CelesteDeprecationWarning
    with pytest.warns(CelesteDeprecationWarning, match="set the operating state on the flanges"):
        unit = make(**{f'{port}_{var}': value})
    assert getattr(getattr(unit, port), var) == pytest.approx(value)


def test_state_keyword_warning_points_at_caller():
    from celeste._deprecation import CelesteDeprecationWarning
    with pytest.warns(CelesteDeprecationWarning) as rec:
        components.CompIsent('c', media.Air(), F1_P=5.0)
    assert rec[0].filename == __file__


def test_unknown_keyword_is_an_error():
    with pytest.raises(TypeError, match="unexpected keyword argument 'F1_Q'"):
        heat.Heater('h', media.Air(), F1_Q=1.0)


def test_mixer_former_T_P_set_all_ports():
    from celeste._deprecation import CelesteDeprecationWarning
    with pytest.warns(CelesteDeprecationWarning):
        m = flow.Mixer('m', media.Air(), T=400.0, P=3.0)
    assert [f.T for f in (m.F0, m.F1, m.F2)] == [400.0] * 3
    assert [f.P for f in (m.F0, m.F1, m.F2)] == [3.0] * 3
