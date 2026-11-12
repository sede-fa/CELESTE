"""Peng-Robinson equation of state and the vapour-liquid flash (celeste.jx)."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402

from celeste.jx import eos, flash, fluid, ideal  # noqa: E402

SPECIES = ['H2', 'CO2', 'CO', 'H2O', 'CH3OH', 'CH4', 'N2']
X = jnp.array([0.5, 0.15, 0.1, 0.1, 0.08, 0.05, 0.02])


@pytest.fixture(scope='module')
def params():
    return eos.pr_params(SPECIES)


def _n_gibbs_departure(n, T, P, p, phase):
    x = n/jnp.sum(n)
    Z, A, B, *_ = eos._state(T, P, x, p, phase)
    return jnp.sum(n)*eos._gibbs_departure_rt(Z, A, B)


@pytest.mark.parametrize('T, P, phase', [(480., 60., eos.VAPOUR), (300., 60., eos.LIQUID), (900., 200., eos.VAPOUR),
                                         (350., 5., eos.LIQUID), (320., 150., eos.STABLE)])
def test_thermodynamic_consistency(params, T, P, phase):
    """Fugacity coefficients and departure functions are derivatives of one Gibbs energy (checked by autodiff)."""
    ln_phi = eos.ln_phi(T, P, X, params, phase)
    np.testing.assert_allclose(ln_phi, jax.grad(_n_gibbs_departure)(X, T, P, params, phase), atol=1e-12)
    h_dep, s_dep, Z = eos.departure(T, P, X, params, phase)
    dg_dT = jax.grad(lambda t: _n_gibbs_departure(X, t, P, params, phase))(T)
    assert float(h_dep) == pytest.approx(float(-eos.R_KJ*T*T*dg_dT), rel=1e-9, abs=1e-12)     # Gibbs-Helmholtz
    g_dep = eos.R_KJ*T*float(jnp.sum(X*ln_phi))
    assert float(s_dep) == pytest.approx((float(h_dep) - g_dep)/T, rel=1e-9, abs=1e-12)
    assert float(Z) > 0


def test_ideal_gas_limit(params):
    pp = eos.phase_props(600.0, 1e-4, X, params)
    assert float(pp.Z) == pytest.approx(1.0, abs=1e-6)
    assert float(pp.h_dep) == pytest.approx(0.0, abs=1e-6) and np.abs(np.asarray(pp.ln_phi)).max() < 1e-6


def test_cubic_roots():
    z_l, z_v = eos.z_roots(0.3, 0.03)              # three real roots
    for z in (z_l, z_v):
        assert float(eos._cubic_residual(z, 0.3, 0.03)) == pytest.approx(0.0, abs=1e-13)
    assert float(z_l) < float(z_v)
    z_l, z_v = eos.z_roots(0.1, 0.2)               # a single root
    assert float(z_l) == pytest.approx(float(z_v))
    dz = jax.grad(lambda A: eos.z_roots(A, 0.03)[1])(0.3)
    fd = (float(eos.z_roots(0.3 + 1e-6, 0.03)[1]) - float(eos.z_roots(0.3 - 1e-6, 0.03)[1]))/2e-6
    assert float(dz) == pytest.approx(fd, rel=1e-6)


def _psat(species, T, lo, hi):
    from scipy.optimize import brentq
    p = eos.pr_params([species])
    one = jnp.array([1.0])
    gap = jax.jit(lambda P: (eos.ln_phi(T, P, one, p, eos.LIQUID) - eos.ln_phi(T, P, one, p, eos.VAPOUR))[0])
    ps = brentq(lambda P: float(gap(P)), lo, hi)
    liq, vap = eos.phase_props(T, ps, one, p, eos.LIQUID), eos.phase_props(T, ps, one, p, eos.VAPOUR)
    return ps, float(vap.h_dep - liq.h_dep)


@pytest.mark.parametrize('species, T, p_ref, hvap_ref', [
    ('H2O', 298.15, 0.031699, 43.99), ('H2O', 373.15, 1.01418, 40.65), ('H2O', 473.15, 15.549, 34.94),
    ('CH3OH', 337.8, 1.0200, 35.27), ('NH3', 300.0, 10.617, 19.8), ('CO2', 280.0, 41.607, 9.67),
])
def test_saturation_of_pure_fluids(species, T, p_ref, hvap_ref):
    """Reference values: IAPWS-95 and the reference equations of state of the other fluids."""
    ps, hvap = _psat(species, T, 0.8*p_ref, 1.2*p_ref)
    assert ps == pytest.approx(p_ref, rel=0.01)
    assert hvap == pytest.approx(hvap_ref, rel=0.08)


def test_parameters_and_kij():
    assert 'CH3OH' in eos.species_available()
    p = eos.pr_params(['CO2', 'H2O(L)', 'CH4'], kij={('CH4', 'CO2'): 0.1})
    assert float(p.kij[0, 2]) == float(p.kij[2, 0]) == 0.1 and float(p.kij[0, 1]) == 0.12
    assert float(eos.pr_params(['CO2', 'CH4'], kij=np.zeros((2, 2))).kij.sum()) == 0.0
    with pytest.raises(KeyError, match='No Peng-Robinson parameters'):
        eos.pr_params(['C60'])
    with pytest.raises(ValueError):
        eos.pr_params(['CO2', 'CH4'], kij=np.zeros((3, 3)))
    # the alpha function is continuous and monotonic across the critical temperature
    T = jnp.linspace(100.0, 3000.0, 400)
    a = jax.vmap(lambda t: eos.alpha(t, eos.pr_params(['N2', 'H2O'])))(T)
    assert np.all(np.diff(np.asarray(a), axis=0) < 0)


def test_rachford_rice():
    z = jnp.array([0.3, 0.3, 0.4])
    K = jnp.array([3.0, 1.2, 0.2])
    beta = flash.rachford_rice(z, K)
    assert float(jnp.sum(z*(K - 1)/(1 + beta*(K - 1)))) == pytest.approx(0.0, abs=1e-13)
    assert 0 < float(beta) < 1
    assert float(flash.rachford_rice(z, K*10)) > 1 and float(flash.rachford_rice(z, K/50)) < 0


def _check_equilibrium(r, T, P, z, p):
    z = np.asarray(z)/np.sum(z)
    np.testing.assert_allclose(r.beta*r.y + (1 - r.beta)*r.x, z, atol=1e-12)
    f_l = jnp.log(r.x) + eos.ln_phi(T, P, r.x, p, eos.LIQUID)
    f_v = jnp.log(r.y) + eos.ln_phi(T, P, r.y, p, eos.VAPOUR)
    np.testing.assert_allclose(f_l, f_v, atol=1e-8)


@pytest.mark.parametrize('species, z, T, P, beta_ref', [
    (['H2O', 'CH3OH'], [0.5, 0.5], 350.0, 1.013, None),
    (['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'], [0.6, 0.05, 0.1, 0.05, 0.2, 0.0], 303.15, 20.0, 0.8019),
    (['CH4', 'C3H8'], [0.5, 0.5], 300.0, 80.0, None),            # close to the critical point of the mixture
    (['N2', 'O2'], [0.79, 0.21], 80.0, 1.0, None),
    (['H2', 'N2', 'NH3'], [0.6, 0.2, 0.2], 250.0, 150.0, None),
])
def test_two_phase_flash(species, z, T, P, beta_ref):
    p = eos.pr_params(species)
    r = flash.flash_pt(T, P, jnp.array(z), p)
    assert int(r.phase) == flash.TWO_PHASE and bool(r.converged) and float(r.tpd) < 0
    _check_equilibrium(r, T, P, z, p)
    if beta_ref is not None:
        assert float(r.beta) == pytest.approx(beta_ref, abs=5e-4)


@pytest.mark.parametrize('species, z, T, P, phase', [
    (['H2O'], [1.0], 300.0, 1.0, flash.LIQUID), (['H2O'], [1.0], 400.0, 1.0, flash.VAPOUR),
    (['H2O'], [1.0], 700.0, 300.0, flash.VAPOUR), (['H2O', 'CH3OH'], [0.5, 0.5], 340.0, 1.013, flash.LIQUID),
    (['H2O', 'CH3OH'], [0.5, 0.5], 365.0, 1.013, flash.VAPOUR),
    (['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'], [0.6, 0.05, 0.1, 0.05, 0.2, 0.0], 1073.15, 1.0, flash.VAPOUR),
])
def test_single_phase_flash(species, z, T, P, phase):
    r = flash.flash_pt(T, P, jnp.array(z), eos.pr_params(species))
    assert int(r.phase) == phase and float(r.beta) == float(phase)
    np.testing.assert_allclose(r.x, r.y)


def test_flash_against_independent_implementation():
    """Standard Peng-Robinson (Soave alpha, k_ij = 0): values computed with the `thermo` library."""
    cases = [  # Tc [K], Pc [bar], omega, z, T, P, vapour fraction, x, y
        ([369.89, 425.125], [42.512, 37.96], [0.1521, 0.201], [0.5, 0.5], 320.0, 8.0, 0.670906,
         [0.333562, 0.666438], [0.581642, 0.418358]),
    ]
    for Tc, Pc, om, z, T, P, vf, x, y in cases:
        om = np.array(om)
        kappa = 0.37464 + 1.54226*om - 0.26992*om**2
        p = eos.PRParams(jnp.array(Tc), jnp.array(Pc), jnp.stack([jnp.array(kappa), jnp.zeros(2), jnp.zeros(2)], axis=1),
                         jnp.zeros((2, 2)))
        r = flash.flash_pt(T, P, jnp.array(z), p)
        assert float(r.beta) == pytest.approx(vf, abs=1e-5)
        np.testing.assert_allclose(r.x, x, atol=2e-6)
        np.testing.assert_allclose(r.y, y, atol=2e-6)


def test_flash_is_robust_over_a_wide_range(params):
    """Random compositions, 250-650 K, 0.3-300 bar: always converged, balanced and in equilibrium."""
    rng = np.random.default_rng(7)
    n = 300
    T, P = rng.uniform(250, 650, n), 10**rng.uniform(-0.5, 2.5, n)
    Z = rng.dirichlet(np.ones(len(SPECIES))*0.5, n)
    r = jax.jit(jax.vmap(lambda t, p_, z: flash.flash_pt(t, p_, z, params)))(T, P, Z)
    assert bool(np.all(np.asarray(r.converged))) and np.all(np.isfinite(np.asarray(r.beta)))
    beta = np.asarray(r.beta)[:, None]
    np.testing.assert_allclose(beta*np.asarray(r.y) + (1 - beta)*np.asarray(r.x), Z, atol=1e-10)
    two = np.asarray(r.phase) == flash.TWO_PHASE
    assert 20 < two.sum() < n - 20
    gap = jax.vmap(lambda t, p_, x, y: jnp.max(jnp.abs(jnp.where((x > 1e-12) & (y > 1e-12),
                   jnp.log(x) + eos.ln_phi(t, p_, x, params, eos.LIQUID) - jnp.log(y) - eos.ln_phi(t, p_, y, params, eos.VAPOUR), 0.0))))
    assert float(jnp.max(gap(T[two], P[two], np.asarray(r.x)[two], np.asarray(r.y)[two]))) < 1e-8
    # single-phase answers really are stable
    tpd = jax.vmap(lambda t, p_, z: flash.stability(t, p_, z, params).tpd)(T[~two], P[~two], Z[~two])
    assert float(jnp.min(tpd)) > -1e-8


def test_vapour_fraction_is_monotonic_and_differentiable(params):
    z = jnp.array([0.55, 0.1, 0.05, 0.15, 0.1, 0.03, 0.02])
    T = jnp.linspace(280.0, 480.0, 81)
    beta = np.asarray(jax.vmap(lambda t: flash.flash_pt(t, 20.0, z, params).beta)(T))
    assert np.all(np.diff(beta) >= -1e-10) and beta[0] < 0.85 and beta[-1] == 1.0
    d_beta = jax.jacfwd(lambda t: flash.flash_pt(t, 20.0, z, params).beta)(320.0)
    fd = (float(flash.flash_pt(320.01, 20.0, z, params).beta) - float(flash.flash_pt(319.99, 20.0, z, params).beta))/0.02
    assert float(d_beta) == pytest.approx(fd, rel=1e-5)


def test_real_fluid_state():
    fl = fluid.fluid(['H2O'])
    one = jnp.array([1.0])
    liquid, steam = fluid.state_pt(298.15, 1.0, one, fl), fluid.state_pt(600.0, 1.0, one, fl)
    assert int(liquid.phase) == fluid.LIQUID and int(steam.phase) == fluid.VAPOUR
    assert float(liquid.h) == pytest.approx(-285.83, abs=0.5)           # enthalpy of formation of liquid water
    assert float(steam.h) == pytest.approx(float(ideal.enthalpy(600.0, one, fl.gas)), abs=0.05)
    assert float(liquid.v) < 3e-5 and float(steam.v) == pytest.approx(8.314462618e-5*600.0, rel=0.01)
    forced = fluid.state_pt(298.15, 1.0, one, fl, fluid.VAPOUR)        # metastable vapour on request
    assert float(forced.h) > float(liquid.h) + 40.0
    # crossing the two-phase region: the apparent heat capacity contains the heat of condensation
    wet = fluid.fluid(['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'])
    z = jnp.array([0.6, 0.05, 0.1, 0.05, 0.2, 0.0])
    cp = jax.jacfwd(lambda t: fluid.enthalpy_pt(t, 20.0, z, wet))
    assert float(cp(380.0)) > 3*float(cp(450.0))
