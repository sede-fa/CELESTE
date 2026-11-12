"""Flash drum unit model and the Peng-Robinson media."""
import copy

import numpy as np
import pytest

pytest.importorskip('jax')

from celeste.CSL import heat, interfaces, media, separation  # noqa: E402
from conftest import element_flow  # noqa: E402

MEOH_WATER = np.array([1e-8, 1e-8, 0.02, 1e-8, 0.48, 1e-8, 0.5])


def test_knock_out_drum_with_nasa_media(syngas_x):
    ko = separation.Flash('ko', Medium=media.Fuel_CH4(), F0_n=10.0, F0_Xi=syngas_x, F0_T=303.15, F0_P=20.0)
    sol = ko.run_sol()
    assert sol.success and ko.phase == 'two-phase'
    assert ko.beta == pytest.approx(0.80188, abs=2e-4)
    assert ko.F2.Medium.species == ['H2', 'CH4', 'CO2', 'CO', 'H2O(L)', 'N2']
    assert ko.F2.Xi[4] > 0.9999 and ko.F1.Xi[4] == pytest.approx(0.00235, abs=1e-4)
    assert ko.F1.n + ko.F2.n == pytest.approx(10.0) and ko.mass_balance() == pytest.approx(0.0, abs=1e-5)
    assert ko.recovery[4] == pytest.approx(0.9906, abs=1e-3) and ko.recovery[0] < 1e-4
    assert ko.Q < 0                                         # heat of condensation leaves the drum
    assert ko.T == 303.15 and ko.P == 20.0 and ko.K[0] > 1e3 > 1 > ko.K[4]
    assert np.isfinite([ko.Exloss, ko.Sgen, ko.Exth]).all()


def test_single_phase_feed_leaves_through_one_outlet(syngas_x):
    hot = separation.Flash('f', Medium=media.Fuel_CH4(), F0_n=3.0, F0_Xi=syngas_x, F0_T=500.0, F0_P=20.0)
    assert hot.run_sol().success
    assert hot.phase == 'vapour' and hot.F1.n == pytest.approx(3.0) and hot.F2.n == 0.0 and hot.Q == pytest.approx(0.0, abs=1e-9)
    cold = separation.Flash('f', Medium=media.Steam(), T=300.0, F0_n=3.0, F0_T=500.0, F0_P=1.0)
    cold.run_sol()
    assert cold.phase == 'liquid' and cold.F2.n == pytest.approx(3.0) and cold.F1.n == 0.0
    assert cold.Q == pytest.approx(3.0*(cold.F2.h - cold.F0.h))


def test_specifications(syngas_x):
    iso = separation.Flash('f', Medium=media.Fuel_CH4(), T=320.0, P=10.0, F0_n=1.0, F0_Xi=syngas_x, F0_T=600.0, F0_P=20.0)
    iso.run_sol()
    assert (iso.T, iso.P, iso.F2.T, iso.F1.P) == (320.0, 10.0, 320.0, 10.0)
    drop = separation.Flash('f', Medium=media.Fuel_CH4(), dP=0.5, F0_Xi=syngas_x, F0_T=310.0, F0_P=20.0)
    drop.run_sol()
    assert drop.P == 19.5
    duty = separation.Flash('f', Medium=media.Fuel_CH4(), Q=-40.0, F0_n=10.0, F0_Xi=syngas_x, F0_T=400.0, F0_P=20.0)
    sol = duty.run_sol()
    assert sol.success and duty.Q == pytest.approx(-40.0, abs=1e-6) and duty.phase == 'two-phase'
    assert 303.15 < duty.T < 400.0
    with pytest.raises(ValueError, match='either'):
        separation.Flash('f', Medium=media.Fuel_CH4(), T=300.0, Q=0.0)


def test_adiabatic_flash_of_a_real_fluid():
    fl = separation.Flash('fl', Medium=media.Fuel_MeOH_PR(), P=1.2, Q=0.0, F0_n=5.0, F0_Xi=MEOH_WATER, F0_T=420.0, F0_P=30.0)
    assert fl.F0.Medium.state == 'two-phase' or fl.F0.Medium.state == 'liquid'
    sol = fl.run_sol()
    assert sol.success and fl.Q == pytest.approx(0.0, abs=1e-6)
    assert fl.T == pytest.approx(349.59, abs=0.05) and fl.beta == pytest.approx(0.1987, abs=1e-3)
    assert fl.F1.Xi[6] > fl.F0.Xi[6] > fl.F2.Xi[6]                # methanol enriches in the vapour
    assert fl.mass_balance() == pytest.approx(0.0, abs=1e-9)
    assert fl.Sgen > 0 and fl.Exloss > 0                          # throttling is irreversible
    assert fl.Exloss == pytest.approx(298.15*fl.Sgen, rel=1e-6)   # Gouy-Stodola, exergy and entropy are consistent
    for el in 'CHO':
        assert element_flow(fl.F0, el) == pytest.approx(element_flow(fl.F1, el) + element_flow(fl.F2, el), rel=1e-9)
    assert fl.F1.Medium.phase == 'vapour' and fl.F2.Medium.phase == 'liquid'


def test_real_fluid_medium_in_the_ideal_gas_limit():
    for x, T in [(np.array([0.3, 0.1, 0.1, 0.1, 0.3, 0.1]), 800.0), (np.array([0.5, 0.1, 0.1, 0.1, 0.01, 0.19]), 400.0)]:
        a = interfaces.Flange('a', media.Fuel_CH4(), Xi=x, T=T, P=1.0)
        b = interfaces.Flange('b', media.Fuel_CH4_PR(), Xi=x, T=T, P=1.0)
        assert b.Medium.state == 'vapour'
        assert b.h == pytest.approx(a.h, abs=5e-3) and b.s == pytest.approx(a.s, abs=2e-5)
        assert b.cp == pytest.approx(a.cp, rel=2e-3) and b.mw == pytest.approx(a.mw)
        assert b.Ex == pytest.approx(a.Ex, abs=0.02)               # independent exergy formulations agree
        assert b.LHV == pytest.approx(a.LHV) and b.Medium.Z == pytest.approx(1.0, abs=2e-3)


def test_real_fluid_medium_handles_phase_change(syngas_x):
    water = interfaces.Flange('w', media.Water_PR(), T=298.15, P=1.0)
    assert water.Medium.state == 'liquid' and water.Ex == pytest.approx(0.9, abs=0.1)
    assert water.Medium.rho == pytest.approx(997.0, rel=0.2)       # cubic equations underpredict liquid density
    water.T = 400.0
    assert water.Medium.state == 'vapour' and water.Medium.beta == 1.0
    # a cooler with a real-fluid medium is a partial condenser
    cooler = heat.Heater('c', media.Fuel_CH4_PR(), media.Fuel_CH4_PR(), F0_n=10.0, F0_Xi=syngas_x, F0_T=600.0, F0_P=20.0,
                         F1_T=303.15, F1_P=20.0)
    cooler.run_sol()
    assert cooler.F1.Medium.state == 'two-phase' and cooler.F1.Medium.beta == pytest.approx(0.80188, abs=2e-4)
    dry = heat.Heater('c', media.Fuel_CH4(), media.Fuel_CH4(), F0_n=10.0, F0_Xi=syngas_x, F0_T=600.0, F0_P=20.0,
                      F1_T=303.15, F1_P=20.0)
    dry.run_sol()
    assert cooler.Q < dry.Q - 70.0                                 # the latent heat is part of the duty
    assert cooler.Exloss > 0


def test_real_fluid_medium_api():
    m = media.Medium_PRbase('mix', species=['CO2', 'H2O'], Xi=np.array([0.5, 0.5]), kij={('CO2', 'H2O'): 0.1})
    c = copy.copy(m)
    assert type(c) is media.Medium_PRbase and c.phase == 'auto' and c.species == ['CO2', 'H2O']
    liquid = m.with_phase('liquid')
    assert liquid.phase == 'liquid' and liquid.state == 'liquid'
    f, g = interfaces.Flange('f', m), interfaces.Flange('g', media.Fuel_CH4_PR())
    assert f != g
    with pytest.raises(ValueError, match='Unknown phase'):
        media.Water_PR(phase='plasma')
    with pytest.raises(TypeError):
        media.Medium_PRbase('x')
    m.Yi = np.array([0.5, 0.5])
    assert m.Yi == pytest.approx([0.5, 0.5]) and m.converged
