"""Equinox units reproduce the stateful CSL units and can be transformed by JAX."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
pytest.importorskip('equinox')
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from celeste.CSL import components, flow, heat, media, reactors  # noqa: E402
from celeste.jx import ideal, units  # noqa: E402

AIR = ideal.ideal_gas(['O2', 'N2'])
X_AIR = np.array([0.21, 0.79])
FUEL = ideal.ideal_gas(['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'])


def test_heater_matches_csl():
    ref = heat.Heater('h', media.Air(), media.Air())
    ref.F0.n = 2.0
    ref.F0.Xi = X_AIR
    ref.F0.T = 300.0
    ref.F1.T = 800.0
    ref.run_sol()
    out, Q = units.Heater(AIR, 800.0)(units.Stream(2.0, 300.0, 1.0, X_AIR))
    assert float(Q) == pytest.approx(ref.Q, rel=1e-10) and float(out.T) == 800.0


def test_mixer_matches_csl():
    ref = flow.Mixer('m', media.Air())
    ref.F0.n = 1
    ref.F1.n = 2
    ref.F0.T, ref.F1.T = 300.0, 600.0
    ref.run_sol()
    out = units.Mixer(AIR)(units.Stream(1.0, 300.0, 1.0, X_AIR), units.Stream(2.0, 600.0, 1.0, X_AIR))
    assert float(out.T) == pytest.approx(ref.F2.T, abs=1e-5) and float(out.n) == 3.0


def test_compressor():
    feed = units.Stream(1.0, 298.15, 1.0, X_AIR)
    ideal_out, _ = units.Compressor(AIR, 5.0, 1.0)(feed)
    assert float(ideal_out.s(AIR)) == pytest.approx(float(feed.s(AIR)), abs=1e-12)        # exactly isentropic
    out, W = units.Compressor(AIR, 5.0, 0.8)(feed)
    ref = components.CompIsent('c', media.Air(), eff=0.8)
    ref.F1.P = 5.0
    ref.run_sol()
    assert float(W) == pytest.approx(ref.W, rel=1e-2)           # the CSL model uses a mean heat capacity
    assert float(out.T) == pytest.approx(ref.F1.T, abs=1.5)
    expander = units.Compressor(AIR, 1.0, 0.85)
    back, W_exp = expander(out)
    assert float(W_exp) < 0 and float(back.T) < float(out.T)


@pytest.mark.parametrize('x0, T, P', [
    ([0.6, 0.01, 0.2, 0.1, 0.08, 0.01], 600.0, 10.0),
    ([0.1, 1e-8, 1e-8, 1e-8, 0.9, 1e-8], 1073.15, 1.0),
    ([0.0, 0.25, 0.0, 0.0, 0.75, 0.0], 1000.0, 5.0),            # species absent from the feed are formed
    ([0.7, 0.0, 0.25, 0.0, 0.0, 0.05], 550.0, 30.0),
])
def test_equilibrium_reactor_matches_csl(x0, T, P):
    x0 = np.array(x0)
    out, Q = jax.jit(lambda t, p: units.ShiftReformingEquilibrium(FUEL, t, p)(units.Stream(1.0, t, p, x0)))(T, P)
    ref = reactors.MR('r', T=T)
    ref.F0.Xi, ref.F0.P = np.maximum(x0, 1e-10), P
    assert ref.run_sol().success
    np.testing.assert_allclose(out.x, ref.F1.Xi, atol=2e-6)
    assert float(out.n) == pytest.approx(ref.F1.n, rel=1e-6)
    assert float(Q) == pytest.approx(ref.Q, abs=1e-4)


def test_vmap_and_grad_through_a_flowsheet():
    x0 = jnp.array([0.6, 0.01, 0.2, 0.1, 0.08, 0.01])

    def methane_out(T, P):
        feed = units.Stream(1.0, 500.0, P, x0)
        hot, _ = units.Heater(FUEL, T)(feed)
        out, _ = units.ShiftReformingEquilibrium(FUEL, T, P)(hot)
        return out.n*out.x[1]

    T = jnp.linspace(500.0, 1000.0, 26)
    ch4 = jax.jit(jax.vmap(methane_out, (0, None)))(T, 10.0)
    assert np.all(np.diff(np.asarray(ch4)) < 0)                 # methanation is favoured at low temperature
    d_dT = jax.grad(methane_out)(700.0, 10.0)
    fd = (float(methane_out(700.05, 10.0)) - float(methane_out(699.95, 10.0)))/0.1
    assert float(d_dT) == pytest.approx(fd, rel=1e-5)
    d_dP = jax.grad(methane_out, argnums=1)(700.0, 10.0)
    assert float(d_dP) > 0                                      # and at high pressure


def test_stream_replace():
    feed = units.Stream(n=1.0, T=500.0, P=10.0, x=X_AIR)
    hot = feed.replace(T=600.0)
    assert float(hot.T) == 600.0 and float(feed.T) == 500.0 and float(hot.P) == 10.0   # original unchanged
    # usable inside jit and grad
    dH = jax.grad(lambda T: feed.replace(T=T).H(AIR))(700.0)
    assert float(dH) == pytest.approx(float(feed.n*ideal.heat_capacity(700.0, X_AIR, AIR)), rel=1e-10)
    with pytest.raises(TypeError, match="no field"):
        feed.replace(Q=1.0)


# ---- SOC and the SOEC flowsheet ------------------------------------------------------------------------
import equinox as eqx  # noqa: E402

from celeste.CSL import electrochem, equilibrium  # noqa: E402


@pytest.mark.parametrize('I, spec, X0', [
    (-50.0, dict(T=1073.15), None),                                                  # electrolysis
    (20.0, dict(T=1073.15), [0.5, 0.05, 0.05, 0.05, 0.35, 1e-5]),                    # fuel cell, with reforming
    (-50.0, dict(Q=0.0), None),                                                      # adiabatic
])
def test_soc_matches_csl(I, spec, X0):
    ref = electrochem.SOC('s', media.Fuel_CH4(), media.Air(), I=I, Ncells=100, **spec)
    if X0 is not None:
        ref.F0.Xi = np.array(X0)
    ref.run_sol()
    fuel = units.Stream(ref.F0.n, ref.F0.T, ref.F0.P, np.asarray(ref.F0.Xi))
    air = units.Stream(ref.A0.n, ref.A0.T, ref.A0.P, np.asarray(ref.A0.Xi))
    soc = units.SOC(FUEL, AIR, I=I, Ncells=100.0, **spec)
    f_out, a_out, W, Q = eqx.filter_jit(lambda s, f, a: s(f, a))(soc, fuel, air)
    assert float(W) == pytest.approx(ref.W, rel=1e-10) and float(f_out.T) == pytest.approx(ref.F1.T, rel=1e-12)
    assert float(Q) == pytest.approx(ref.Q, abs=1e-9)
    assert np.allclose(np.asarray(f_out.x), ref.F1.Xi, atol=1e-12)
    assert np.max(np.abs(np.asarray(soc.residuals(fuel, air, f_out, a_out)))) < 1e-10


def test_fuel_residuals_numpy_and_jax_agree():
    n, n0 = np.array([0.3, 0.01, 0.1, 0.05, 0.5, 0.04]), np.array([0.1, 0.02, 0.1, 0.05, 0.7, 0.04])
    a = equilibrium.fuel_residuals(n, n0, 1073.15, 2.0, O_in=-0.1)
    b = equilibrium.fuel_residuals(n, n0, 1073.15, 2.0, O_in=-0.1, xp=jnp)
    assert np.allclose(a, np.asarray(b), atol=1e-13)


def test_units_residuals_vanish_at_their_solution():
    feed = units.Stream(1.0, 600.0, 1.0, np.array([0.3, 0.05, 0.1, 0.05, 0.45, 0.05]))
    cold = units.Stream(0.5, 350.0, 1.0, np.array([0.1, 0.0, 0.0, 0.0, 0.9, 0.0]))
    sep = units.Separator(FUEL, jnp.array([1, 1, 1, 1, 0.1, 1.0]), 320.0)
    o1, o2, _ = sep(feed)
    assert np.max(np.abs(np.asarray(sep.residuals(feed, o1, o2)))) < 1e-14
    spl = units.Splitter(0.3)
    assert np.max(np.abs(np.asarray(spl.residuals(feed, *spl(feed))))) < 1e-14
    hx = units.HeatExchanger(FUEL, FUEL, 0.8)
    h_out, c_out, Q = hx(feed, cold)
    assert np.max(np.abs(np.asarray(hx.residuals(feed, cold, h_out, c_out)))) < 1e-10
    assert float(h_out.H(FUEL)) == pytest.approx(float(feed.H(FUEL) - Q), abs=1e-9)


def _soec_example():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).parents[1]/'examples'/'05_soec_flowsheet.py'
    spec = importlib.util.spec_from_file_location('soec_flowsheet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_soec_flowsheet_sequential_and_equation_oriented_agree():
    ex = _soec_example()
    plant = ex.SOECPlant()
    assert plant._plan()[1] == ['soc.fuel_out', 'spl.out1']                     # tears generated from the links
    assert plant.products == ['soc.air_out', 'ko.out2', 'spl.out2']
    s = eqx.filter_jit(lambda p: p.sequential())(plant)
    H2 = float(units.flows(s['spl.out2'])[0])
    assert H2 == pytest.approx(-float(plant.soc.n_O), rel=1e-8)          # all hydrogen made leaves as product
    x0 = plant.pack(s)
    assert np.max(np.abs(np.asarray(plant.residuals(x0, s)))) < 1e-8      # SM solution satisfies the EO equations
    s_eo = eqx.filter_jit(lambda p, s: p.solve(s))(plant, s)
    assert np.max(np.abs(np.asarray(plant.residuals(plant.pack(s_eo), s)))) < 1e-12
    _, I = eqx.filter_jit(lambda p, s: ex.solve_with_spec(p, s, 0.09))(plant, s)
    assert float(I) == pytest.approx(-0.09*2*96485.0/300.0, rel=1e-8)
    assert 'soc (SOC): ht.out, air -> soc.fuel_out, soc.air_out' in plant.print_equations()


def test_jx_flowsheet_pfd_and_link_checks():
    from celeste.jx.flowsheet import Flowsheet
    ex = _soec_example()
    plant = ex.SOECPlant()
    s = plant.sequential()
    page = plant.pfd(s, species=ex.SPECIES).html
    assert '6 units, 12 streams; 2 back stream' in page and 'spl.out1 \\u2192 mix.b' in page and 'H2O' in page
    assert 'electrochemical' in page                                      # kind of the jx SOC

    class Broken(Flowsheet):
        mix: units.Mixer

        def __init__(self):
            self.mix = units.Mixer(ex.FUEL)
            self.link(self.mix.out, self.mix.out)
    with pytest.raises(ValueError, match='not an inlet'):
        Broken()
