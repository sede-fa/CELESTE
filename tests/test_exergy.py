"""Stream exergy: parity with the original implementation, the fixed defects, absence of side effects."""
import numpy as np
import pytest

from celeste.CSL import interfaces, media
from celeste.CSL import exergy
from celeste.FT import thermoFT


def _flange(name, row):
    return interfaces.Flange('f', getattr(media, name)(), n=1.0, Xi=np.array(row['X']), T=row['T'], P=row['P'])


@pytest.mark.parametrize('name', ['Air', 'Atm_Air', 'Fuel_CH4', 'Fuel_CH4L', 'HB', 'Steam'])
def test_exergy_matches_original(golden, name):
    for row in golden['media'][name]:
        f = _flange(name, row)
        if f.Medium.in_range:
            assert f.Ex == pytest.approx(row['ex'], abs=1e-8)


def test_exergy_reference_state_uses_the_property_model_of_the_medium(golden):
    """Propane used a different correlation for the dead state than for the stream: small, consistent shift."""
    for row in golden['media']['media_C3']:
        f = _flange('media_C3', row)
        assert f.Ex == pytest.approx(row['ex'], abs=0.05)


@pytest.mark.parametrize('name', ['Fuel_MeOH', 'Fuel_MeOHL'])
def test_methanol_media_have_an_exergy(name):
    """The original returned nan (Fuel_MeOH) or raised (Fuel_MeOHL)."""
    f = interfaces.Flange('f', getattr(media, name)(), T=400.0, P=10.0)
    assert np.isfinite(f.Ex) and f.Ex > 0


def test_liquid_water_exergy_does_not_depend_on_pressure(golden):
    low = interfaces.Flange('f', media.Water(), T=298.15, P=1.0)
    high = interfaces.Flange('f', media.Water(), T=298.15, P=60.0)
    assert low.Ex == pytest.approx(0.9, abs=1e-6)           # standard chemical exergy of liquid water
    assert high.Ex == pytest.approx(low.Ex, abs=1e-9)       # was 11.05 kJ/mol: ideal-gas pressure term on a liquid
    row = next(r for r in golden['media']['Water'] if r['P'] == 1.0 and r['T'] == 400.0)
    assert _flange('Water', row).Ex == pytest.approx(row['ex'], abs=1e-8)


def test_exergy_does_not_modify_the_stream():
    x = np.array([0.5, 5e-7, 0.1, 0.1, 0.2, 0.1 - 5e-7])
    f = interfaces.Flange('f', media.Fuel_CH4(), Xi=x.copy(), T=700.0, P=3.0)
    h = f.h
    f.Ex
    np.testing.assert_array_equal(f.Xi, x)      # fractions below 1e-6 used to be overwritten with 1e-12
    assert f.h == h and f.T == 700.0 and f.P == 3.0


@pytest.mark.parametrize('name', ['Media_AEL', 'Steam_Water'])
def test_liquid_water_next_to_steam(golden, name):
    """Media with H2O and H2O(L): the liquid no longer gets a gas pressure / mixing term in the exergy."""
    worst = 0.0
    for row in golden['media'][name]:
        f = _flange(name, row)
        if f.Medium.in_range:
            worst = max(worst, abs(f.Ex - row['ex']))
    assert 0 < worst < 8.314462618e-3*298.15*np.log(60.0) + 0.1      # at most the old pressure term at 60 bar


@pytest.mark.parametrize('name', ['Air', 'Fuel_CH4', 'Fuel_CH4L', 'Fuel_MeOH', 'HB', 'Media_AEL', 'Steam_Water', 'media_C3'])
def test_exergy_and_entropy_are_consistent(name):
    """Gouy-Stodola: between two states of equal composition, dEx = dh - T0 ds (no condensation at the dead state)."""
    rng = np.random.default_rng(3)
    cls = getattr(media, name)
    for _ in range(5):
        x = rng.dirichlet(np.ones(cls().nX))
        if 'H2O' in cls().species:
            x[cls().species.index('H2O')] = 0.01
            x /= x.sum()
        a = interfaces.Flange('a', cls(), Xi=x, T=rng.uniform(300, 590), P=rng.uniform(1, 50))
        b = interfaces.Flange('b', cls(), Xi=x, T=rng.uniform(300, 590), P=rng.uniform(1, 50))
        if a.ExergyObj.condenses:
            continue
        assert b.Ex - a.Ex == pytest.approx(b.h - a.h - exergy.T0*(b.s - a.s), abs=1e-9)


def test_exergy_destruction_equals_T0_times_entropy_generation():
    from celeste.CSL import flow, components
    m = flow.Mixer('m', media.Fuel_CH4())
    m.F0.n = 1.0
    m.F0.Xi = np.array([0.5, 0.2, 0.1, 0.1, 0.01, 0.09])
    m.F1.n = 2.0
    m.F1.Xi = np.array([0.1, 0.1, 0.3, 0.3, 0.01, 0.19])
    m.F0.T, m.F1.T = 400.0, 700.0
    m.run_sol()
    assert m.Exloss == pytest.approx(exergy.T0*m.Sgen, rel=1e-6)
    v = components.Valve('v', media.Air(), P_out=2.0)
    v.F0.Xi = np.array([0.21, 0.79])
    v.F0.T = 400.0
    v.F0.P = 10.0
    v.run_sol()
    assert v.Exloss == pytest.approx(exergy.T0*v.Sgen, rel=1e-9)


def test_exergy_breakdown():
    x = np.array([0.5, 0.1, 0.1, 0.1, 0.01, 0.19])          # dry enough not to condense at the dead state
    f = interfaces.Flange('f', media.Fuel_CH4(), n=2.0, Xi=x, T=800.0, P=5.0)
    assert f.Ex == pytest.approx(f.Ex_ph + f.Ex_ch) and f.EX == pytest.approx(2.0*f.Ex)
    assert f.Ex_ph > 0 and f.Ex_ch > 0
    dead = interfaces.Flange('d', media.Air(), T=298.15, P=1.0)
    assert dead.Ex_ph == pytest.approx(0.0, abs=1e-12)
    pr = interfaces.Flange('p', media.Fuel_CH4_PR(), Xi=x, T=800.0, P=5.0)
    assert pr.Ex_ch == pytest.approx(f.Ex_ch, abs=1e-6) and pr.Ex_ph == pytest.approx(f.Ex_ph, abs=0.02)


def test_base_package_does_not_need_the_ft_module():
    import subprocess, sys
    code = ("import sys, numpy as np; from celeste.CSL import media, interfaces, separation, reactors; "
            "from celeste.CSL.systems import ElectBox, ComponentModels, SOCsystem, ProcessBoxes; "
            "interfaces.Flange('f', media.Fuel_CH4()).Ex; "
            "print(any(m.startswith('celeste.FT') for m in sys.modules))")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'False'


def test_ft_media_use_the_ft_exergy_model():
    from celeste.FT import interfacesFT, mediaFT
    f = interfacesFT.Flange('f', mediaFT.Naphtha())
    assert isinstance(f.ExergyObj, thermoFT.ThermoExergyFT) and isinstance(f.ExergyObj, exergy.ThermoExergy)


def test_exergy_object_api():
    m = media.Fuel_CH4()
    x = np.array([0.3, 0.1, 0.1, 0.1, 0.3, 0.1])
    ex = thermoFT.ThermoExergy(m, x, 600.0, 5.0)
    assert isinstance(ex, exergy.ThermoExergy)
    assert ex.Ex == pytest.approx(ex.streamPhysExergy() + ex.streamChemExergy())
    assert m.T == 600.0                                      # a foreign temperature is applied to the medium
    assert float(thermoFT.ThermoExergy.df.loc['H2', 'Standard Chem Exergy kJ/mol']) == pytest.approx(236.09)
    with pytest.raises(KeyError, match='standard chemical exergy'):
        thermoFT.standard_chemical_exergy(('NoSuchSpecies',))


def test_dead_state_has_only_chemical_exergy():
    f = interfaces.Flange('f', media.Air(), T=298.15, P=1.0)
    obj = f.ExergyObj
    assert obj.streamPhysExergy() == pytest.approx(0.0, abs=1e-10)
    assert f.Ex == pytest.approx(obj.streamChemExergy())
