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
    ref = reactors.MR('r', F0_Xi=np.maximum(x0, 1e-10), T=T, P=P)
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
