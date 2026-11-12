"""Ideal-gas property kernel: parity with the original implementation, both array backends, edge cases."""
import copy

import numpy as np
import pytest

from celeste.CSL import interfaces, media, thermo
from celeste.jx import nasa

R = 8.314462618e-3
NASA_MEDIA = ['Air', 'Atm_Air', 'Fuel_CH4', 'Fuel_CH4L', 'Fuel_MeOH', 'Fuel_MeOHL', 'HB', 'Media_AEL', 'Steam',
              'Steam_Water', 'Water', 'media_C3']
# media whose first species is H2O(L): the original code applied the gas mixing entropy to the liquid there
LIQUID_FIRST = {'Water': 0, 'Media_AEL': 0}


def _flange(name, row):
    return interfaces.Flange('f', getattr(media, name)(), n=1.0, Xi=np.array(row['X']), T=row['T'], P=row['P'])


@pytest.mark.parametrize('name', NASA_MEDIA)
def test_species_properties_match_original(golden, name):
    checked = 0
    for row in golden['media'][name]:
        f = _flange(name, row)
        if not f.Medium.in_range:
            continue        # the original returned zeros outside the fit; covered by test_out_of_range_*
        # (the summation order differs from the original loops: agreement to round-off, not bit for bit)
        np.testing.assert_allclose(f.Medium.cp_i, row['cp_i'], rtol=1e-9)
        np.testing.assert_allclose(f.Medium.h_i, row['h_i'], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(f.Medium.s_i, row['s_i'], rtol=1e-9)
        np.testing.assert_allclose(f.mw, row['mw'], rtol=1e-13)
        np.testing.assert_allclose(f.LHV, row['lhv'], rtol=1e-13)
        checked += 1
    assert checked >= 6


@pytest.mark.parametrize('name', NASA_MEDIA)
def test_mixture_properties_match_original(golden, name):
    for row in golden['media'][name]:
        f = _flange(name, row)
        if not f.Medium.in_range:
            continue
        x = np.array(row['X'])
        # the original dropped species below 1e-6 from the mixture sums; the new kernel is smooth in x
        np.testing.assert_allclose(f.cp, row['cp'], atol=1e-8)
        np.testing.assert_allclose(f.h, row['h'], atol=1e-5)
        s_expected = row['s']
        if name in LIQUID_FIRST:
            xw = x[LIQUID_FIRST[name]]
            s_expected += R*xw*np.log(xw*row['P'])      # liquid water carries no mixing / pressure entropy
        np.testing.assert_allclose(f.s, s_expected, atol=1e-7)
        np.testing.assert_allclose(f.g, f.h - row['T']*f.s, rtol=1e-13)


def test_numpy_and_jax_kernels_agree():
    import jax
    from celeste.jx import ideal
    species = ['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2', 'CH3OH', 'H2O(L)']
    d = nasa.nasa_data(species)
    gas = ideal.ideal_gas(species)
    x = np.array([0.3, 0.05, 0.1, 0.05, 0.2, 0.1, 0.1, 0.1])
    jit_props = jax.jit(lambda T, P, X: nasa.props(jax.numpy, T, P, X, gas.a, gas.T_break, gas.T_lo, gas.T_hi, gas.condensed))
    for T in (150.0, 250.0, 400.0, 1000.0, 1000.5, 3000.0, 7000.0):
        ref = nasa.props(np, T, 12.0, x, d.a, d.T_break, d.T_lo, d.T_hi, d.condensed)
        out = jit_props(T, 12.0, x)
        for a, b in zip(ref, out):
            np.testing.assert_allclose(np.asarray(b), a, rtol=1e-8)


def test_scalar_fast_path_equals_generic_kernel():
    d = nasa.nasa_data(['H2', 'CH4', 'CO2', 'CO', 'H2O', 'N2'])
    for T in (200.0, 200.5, 999.999, 1000.0, 1000.001, 5999.0, 6000.0):
        fast = nasa.pure_props_scalar(T, d.a, d.T_break)
        generic = nasa.pure_props(np, T, d.a, d.T_break, d.T_lo, d.T_hi)
        for a, b in zip(fast, generic):
            np.testing.assert_allclose(a, b, rtol=1e-13)


def test_jax_vmap_and_grad():
    import jax
    import jax.numpy as jnp
    from celeste.jx import ideal
    gas = ideal.ideal_gas(['O2', 'N2'])
    x = jnp.array([0.21, 0.79])
    T = jnp.linspace(300.0, 1500.0, 50)
    h = jax.vmap(ideal.enthalpy, (0, None, None))(T, x, gas)
    ref = media.Air()
    for t, hv in zip(np.asarray(T)[::7], np.asarray(h)[::7]):
        ref.T = float(t)
        np.testing.assert_allclose(hv, ref.h, rtol=1e-12)
    # cp is the exact temperature derivative of h
    cp = jax.grad(ideal.enthalpy)(800.0, x, gas)
    ref.T = 800.0
    np.testing.assert_allclose(float(cp), ref.cp, rtol=1e-10)


@pytest.mark.parametrize('species, T_edge', [(['H2O(L)'], 600.0), (['H2O(L)'], 273.15), (['N2'], 200.0)])
def test_out_of_range_is_finite_and_continuous(species, T_edge):
    """Beyond the fitted range cp is frozen: h and s continue without a jump (the original returned zeros)."""
    th = thermo.Thermo(species, T_edge, np.array([1.0]))
    inside = (th.cp, th.h, th.s)
    for dT in (-1e-6, 1e-6):
        th.T = T_edge + dT
        np.testing.assert_allclose((th.cp, th.h, th.s), inside, rtol=1e-6)
    th.T = T_edge + (200.0 if T_edge > 500 else -100.0)
    assert not th.in_range
    assert np.isfinite([th.cp, th.h, th.s]).all()
    assert th.cp == pytest.approx(inside[0])                         # frozen heat capacity
    assert th.h == pytest.approx(inside[1] + inside[0]*(th.T - T_edge))


def test_entropy_is_smooth_at_vanishing_fractions():
    th = media.Fuel_CH4()
    values = []
    for x_ch4 in (0.0, 1e-12, 9.9e-7, 1.01e-6, 1e-5):
        x = np.array([0.5, x_ch4, 0.1, 0.1, 0.2, 0.1 - x_ch4])
        th.Xi = x
        values.append((th.h, th.s))
    values = np.array(values)
    assert np.all(np.isfinite(values))
    assert np.all(np.abs(np.diff(values[:, 0])) < 1e-3)     # no jump at the former 1e-6 threshold
    th.Xi = np.array([0.5, -1e-9, 0.1, 0.1, 0.2, 0.1])       # slightly negative iterate of a solver
    assert np.isfinite(th.s)


def test_setters_keep_properties_consistent():
    a = media.Fuel_CH4()
    x = np.array([0.3, 0.1, 0.1, 0.1, 0.3, 0.1])
    a.PTX = 12.0, 850.0, x
    b = thermo.Thermo(a.species, 850.0, x, 12.0)
    for attr in ('cp', 'h', 's', 'g'):
        assert getattr(a, attr) == pytest.approx(getattr(b, attr), rel=1e-14)
    a.T = 400.0; a.P = 2.0; a.Xi = x[::-1].copy()
    c = thermo.Thermo(a.species, 400.0, x[::-1].copy(), 2.0)
    assert a.s == pytest.approx(c.s, rel=1e-14)
    a.Tc = 100.0
    assert a.T == pytest.approx(373.15)
    a.Yi = np.full(6, 1/6)
    assert a.Yi == pytest.approx(np.full(6, 1/6))
    assert a.TX[0] == a.T and a.PTcX[1] == pytest.approx(100.0)


def test_scalar_inputs_of_any_flavour():
    f = interfaces.Flange('f', media.Air(), T=500.0)
    h = f.h
    for T in (np.float64(500.0), np.array(500.0), np.array([500.0]), 500):
        f.T = T
        assert f.h == pytest.approx(h, rel=1e-15)


def test_lhv():
    th = thermo.Thermo(['H2', 'CH4', 'C2H6', 'Ar', 'CO'], 300.0, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    assert th.LHVi['H2'] == pytest.approx(241.809)
    assert th.LHVi['C2H6'] == pytest.approx(1428.6, abs=0.5)        # computed from formation enthalpies
    assert th.LHVi['Ar'] == 0.0
    assert th.LHVmix == pytest.approx(np.dot(th.LHVi.values, th.Xi))


def test_database_access():
    assert thermo.Thermo.df.index.size > 1800
    assert nasa.elements('CH3OH') == {'C': 1.0, 'H': 4.0, 'O': 1.0}
    with pytest.raises(KeyError, match='NotASpecies'):
        thermo.Thermo(['NotASpecies'], 300.0, np.array([1.0]))


def test_media_copy_and_custom_media():
    class MediaExample(media.Medium_NASAbase):          # the pattern used in the tutorial
        def __init__(self, name='MediaExample'):
            self._reference_X = np.array([0.25, 0.5, 0.25])
            super().__init__(name, species=np.array(['CO2', 'CO', 'CH4']), Xi=self._reference_X)

    class Declarative(media.Medium_NASAbase):
        _default_species = ['H2', 'CO']
        _default_Xi = np.array([0.5, 0.5])

    f = interfaces.Flange('f', MediaExample())
    assert f.Xi == pytest.approx([0.25, 0.5, 0.25]) and f.Medium.get_Idx('CO') == 1
    m = Declarative()
    m.T = 900.0
    c = copy.copy(m)
    assert c is not m and c.T == 298.15 and c.name == 'Declarative'
    assert 'Default Species: H2, CO' in Declarative.__doc__
    assert media.Air().rho == pytest.approx(1.164, abs=0.01)
    assert media.Water().rho == pytest.approx(1000.0, abs=40.0)      # Rackett type correlation


def test_deprecated_shims_warn():
    from celeste._deprecation import CelesteDeprecationWarning
    th = media.Air()
    with pytest.warns(CelesteDeprecationWarning):
        th.updateBoolArr()


def test_coolprop_medium():
    pytest.importorskip('CoolProp')
    m = media.Fuel_CH4_CP(EOS='PR')
    ref = media.Fuel_CH4()
    dry = np.array([0.5, 0.1, 0.1, 0.1, 1e-6, 0.2 - 1e-6])
    m.PTX = 1.0, 500.0, dry
    ref.PTX = 1.0, 500.0, dry
    assert m.h == pytest.approx(ref.h, abs=0.5)
    assert m.cp == pytest.approx(ref.cp, rel=0.05)       # kJ/mol/K, like every other medium
    assert m.rho > 0
