"""Stream exergy: parity with the original implementation, the fixed defects, absence of side effects."""
import numpy as np
import pytest

from celeste.CSL import interfaces, media
from celeste.FT import thermoFT


def _flange(name, row):
    return interfaces.Flange('f', getattr(media, name)(), n=1.0, Xi=np.array(row['X']), T=row['T'], P=row['P'])


@pytest.mark.parametrize('name', ['Air', 'Atm_Air', 'Fuel_CH4', 'Fuel_CH4L', 'HB', 'Media_AEL', 'Steam', 'Steam_Water'])
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


def test_exergy_object_api():
    m = media.Fuel_CH4()
    x = np.array([0.3, 0.1, 0.1, 0.1, 0.3, 0.1])
    ex = thermoFT.ThermoExergy(m, x, 600.0, 5.0)
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
