"""Fischer-Tropsch module: properties match the original implementation, the units run."""
import numpy as np
import pytest

from celeste.FT import componentsFT, flowFT, heatFT, interfacesFT, mediaFT, reactorsFT

FT_MEDIA = ['Syncrude', 'Wax', 'Naphtha', 'LightGas', 'Kerosene', 'Distillate', 'Water', 'Hydrogen', 'Alkanes']


LIGHT = {'LightGas', 'Water', 'Hydrogen'}           # no C5+ species: dead state unchanged


@pytest.mark.parametrize('name', FT_MEDIA)
def test_ft_gas_properties_match_original(golden, name):
    """Gas streams: cp, h, s, LHV as in the original implementation."""
    for row in golden['ft'][name]:
        if row['phase'] != 'Gas':
            continue
        f = interfacesFT.Flange('f', getattr(mediaFT, name)(), T=row['T'], P=row['P'], phase=row['phase'])
        assert f.cp == pytest.approx(row['cp'], rel=1e-10)
        assert f.h == pytest.approx(row['h'], rel=1e-10)
        assert f.s == pytest.approx(row['s'], rel=1e-10)
        assert f.LHV == pytest.approx(row['lhv'], rel=1e-12)
        if name in LIGHT:
            assert f.Ex == pytest.approx(row['ex'], rel=1e-10)
        else:
            # C5+ have a liquid standard chemical exergy: the vapour now carries RT0 ln(P0/Psat) on top
            assert f.Ex > row['ex']


@pytest.mark.parametrize('name', ['Naphtha', 'Kerosene', 'Distillate', 'Wax'])
def test_liquid_at_the_dead_state_has_the_tabulated_chemical_exergy(golden, name):
    """
    A liquid C5+ stream at T0, P0 has exactly sum x_i ex0_i + R T0 sum x_i ln x_i (the tabulated values are for the
    liquid); the original implementation gave this number for every phase.
    """
    row = next(r for r in golden['ft'][name] if r['T'] == 298.15 and r['P'] == 1.0)
    f = interfacesFT.Flange('f', getattr(mediaFT, name)(), T=298.15, P=1.0, phase='Liquid')
    assert f.Ex == pytest.approx(row['ex'], abs=0.5)


def test_exergy_of_vaporisation():
    """Pure decane at T0, P0: Ex(gas) - Ex(liquid) = g_gas - g_liquid = -R T0 ln(Psat(T0)/P0)."""
    x = np.zeros(50)
    x[9] = 1.0                                                            # C10H22 in mediaFT.Alkanes (C1-C50)
    gas = interfacesFT.Flange('g', mediaFT.Alkanes(), Xi=x, T=298.15, P=1.0, phase='Gas')
    liq = interfacesFT.Flange('l', mediaFT.Alkanes(), Xi=x, T=298.15, P=1.0, phase='Liquid')
    hvap, svap, _ = liq.Medium.calVAP()
    lnPsat = (svap[8] - hvap[8]/298.15)/thermoFT.Thermo.R                  # hydrocarbon list starts at C2
    assert gas.Ex - liq.Ex == pytest.approx(-thermoFT.Thermo.R*298.15*lnPsat, rel=1e-9)
    assert np.exp(lnPsat) == pytest.approx(0.00180, rel=0.15)              # decane: 1.8 mbar at 25 °C


@pytest.mark.parametrize('name', ['Syncrude', 'Naphtha', 'Wax', 'Water'])
def test_liquid_exergy_obeys_gouy_stodola(name):
    """
    Between two states of the same composition: dEx = dh - T0 ds (phase change and pressure included). Not for a
    gas whose water partly condenses at the dead state (Syncrude), where the exergy model splits the phases.
    """
    a = interfacesFT.Flange('a', getattr(mediaFT, name)(), T=320.0, P=1.0, phase='Liquid')
    states = [(450.0, 1.0, 'Liquid'), (450.0, 30.0, 'Liquid'), (500.0, 5.0, 'Gas'), (320.0, 1.0, 'Gas')]
    for T, P, phase in (states[:2] if name == 'Syncrude' else states):
        b = interfacesFT.Flange('b', getattr(mediaFT, name)(), T=T, P=P, phase=phase)
        assert b.Ex - a.Ex == pytest.approx((b.h - a.h) - 298.15*(b.s - a.s), abs=1e-8)


def test_flange_phase_switches_the_properties():
    """Setting the phase on an FT flange used to have no effect: liquid streams returned gas properties."""
    f = interfacesFT.Flange('f', mediaFT.Water(), T=298.15, P=1.0, phase='Gas')
    gas_h, gas_s = f.h, f.s
    f.phase = 'Liquid'
    assert f.Medium.property_species == ('H2O(L)',)
    assert f.h == pytest.approx(-285.83, abs=0.01)                          # liquid water
    assert gas_h - f.h == pytest.approx(44.0, abs=0.1)                      # heat of vaporisation at 25 °C
    f.phase = 'gas'
    assert (f.h, f.s) == (gas_h, gas_s) and f.phase == 'Gas'
    with pytest.raises(ValueError):
        f.Medium.phase = 'vapour-ish'


def test_liquid_phase_spelling_only_changes_water():
    """'liquid' used to send every species to the NASA list (operator precedence in getSpecies)."""
    upper = interfacesFT.Flange('u', mediaFT.Syncrude(), phase='Liquid')
    lower = interfacesFT.Flange('l', mediaFT.Syncrude(), phase='liquid')
    assert upper.Medium.property_species == lower.Medium.property_species
    assert upper.Medium.property_species[3] == 'H2O(L)' and 'C10H22' in upper.Medium.property_species
    assert lower.h == upper.h


def test_liquid_entropy_has_no_pressure_term():
    lo = interfacesFT.Flange('lo', mediaFT.Kerosene(), T=350.0, P=1.0, phase='Liquid')
    hi = interfacesFT.Flange('hi', mediaFT.Kerosene(), T=350.0, P=50.0, phase='Liquid')
    assert hi.s == pytest.approx(lo.s, abs=1e-12)
    x = lo.Xi
    assert lo.s == pytest.approx(np.dot(x, lo.Medium.s_i) - thermoFT.Thermo.R*np.sum(x*np.log(x)), rel=1e-12)


def test_reference_composition_is_kept():
    """Medium_Base used to overwrite the reference composition of the FT media with None."""
    assert mediaFT.Wax()._reference_X.shape == (28,)
    f = interfacesFT.Flange('f', mediaFT.Syncrude(), phase='Liquid')
    assert f.Xi.shape == (55,) and f.phase == 'Liquid'
    f.Ex
    assert f.phase == 'Liquid'          # evaluating the exergy no longer relabels the stream


def test_ft_reactor_and_separation():
    x = np.zeros(55)
    x[[0, 1, 2, 4]] = 0.62, 0.05, 0.30, 0.03
    r = reactorsFT.FT('r', alpha=0.85, CO_conv=0.6, F0_Xi=x)
    r.run_sol()
    assert r.Q == pytest.approx(-32.643397, rel=1e-6)          # F2 is liquid (was evaluated as gas: -32.627621)
    assert r.F0.m == pytest.approx(r.F1.m + r.F2.m, rel=1e-5)
    r.ExBs()
    assert r.Exloss > 0 and 0 < r.Exeff < 1
    g = componentsFT.GPU_2S('g', mediaFT.Syncrude(), F0_Xi=r.F1.Xi)
    g.run_sol()
    assert g.F2.n == pytest.approx(r.F1.Xi[3]) and np.isfinite(g.W)


def test_ft_units_run():
    h = reactorsFT.Hydrocracker('h', mediaFT.Wax())
    assert h.run_sol().success and h.mass_balance() == pytest.approx(0.0, abs=1e-6)
    h.ExBs()
    assert np.isfinite(h.Exloss)
    c = componentsFT.CompIsent('c', mediaFT.LightGas(), eff=0.8)
    assert c.run_sol().success and c.F1.T == pytest.approx(350.9861, abs=1e-3)
    m = flowFT.Mixer('m', mediaFT.LightGas())
    m.run_sol()
    assert m.F2.n == 2 and m.mass_balance() == pytest.approx(0.0, abs=1e-9)
    heater = heatFT.Heater('ht', mediaFT.Naphtha(), mediaFT.Naphtha())
    heater.F1.T = 400.0
    heater.run_sol()
    assert heater.Q == pytest.approx(20.470005, rel=1e-6)
    chg = componentsFT.ChgMedium('chg')
    chg.run_sol()
    assert chg.F1.Xi.sum() == pytest.approx(1.0)


# ---- liquid hydrocarbons: vaporisation -------------------------------------------------------------
from celeste.FT import thermoFT


def _liquid(species, T):
    return thermoFT.Thermo(list(species), T, np.full(len(species), 1.0/len(species)), phase='Liquid')


@pytest.mark.parametrize('sp', ['C6H14', 'C10H22', 'C16H34', 'C30H62'])
def test_vaporisation_at_the_normal_boiling_point(sp):
    """Watson correlation reproduces the tabulated heat of vaporisation; Psat(Tb) = 1 atm."""
    row = thermoFT._hc_table().loc[sp]
    hvap, svap, _ = _liquid([sp], row['T_boil_K']).calVAP()
    assert hvap[0] == pytest.approx(row['HVAP_boil_kJ/mol'], rel=2e-3)
    lnP = (svap[0] - hvap[0]/row['T_boil_K'])/thermoFT.Thermo.R
    assert np.exp(lnP) == pytest.approx(1.01325, rel=1e-9)


@pytest.mark.parametrize('sp, T', [('C6H14', 300.0), ('C6H14', 350.0), ('C10H22', 400.0), ('C10H22', 450.0),
                                   ('C16H34', 450.0), ('C16H34', 550.0)])
def test_vapour_pressure_agrees_with_antoine(sp, T):
    """Clausius-Clapeyron over the Watson enthalpy vs. the tabulated Antoine equation (inside its range)."""
    row = thermoFT._hc_table().loc[sp]
    hvap, svap, _ = _liquid([sp], T).calVAP()
    psat = np.exp((svap[0] - hvap[0]/T)/thermoFT.Thermo.R)
    antoine = 10**(row['A'] - row['B']/(T - 273.15 + row['C']))*0.00133322
    assert psat == pytest.approx(antoine, rel=0.15)


@pytest.mark.parametrize('sp, T', [('C10H22', 300.0), ('C10H22', 500.0), ('C20H42', 350.0), ('C3H8', 250.0)])
def test_liquid_is_thermodynamically_consistent(sp, T):
    """
    Liquid minus ideal gas obeys d(dh)/dT = dcp and d(dh) = T d(ds) (checked on the difference, so that
    the small inconsistency of the ideal-gas fit itself drops out).
    """
    dT = 1e-3

    def delta(T_):
        liq = _liquid([sp], T_)
        gas = thermoFT.Thermo([sp], T_, np.array([1.0]), phase='Gas')
        return liq.h - gas.h, liq.s - gas.s, liq.cp - gas.cp

    (h_lo, s_lo, _), (_, _, cp_mid), (h_hi, s_hi, _) = delta(T - dT), delta(T), delta(T + dT)
    dh = (h_hi - h_lo)/(2*dT)
    assert cp_mid == pytest.approx(dh, rel=1e-5, abs=1e-8)
    assert T*(s_hi - s_lo)/(2*dT) == pytest.approx(dh, rel=1e-5, abs=1e-8)


def test_subcooled_liquid_uses_the_liquid_heat_capacity():
    """Below Tmin: cp = dh/dT = cpl of the table."""
    row = thermoFT._hc_table().loc['C20H42']
    T, dT = row['HVAP_Tmin_K'] - 10.0, 1e-3
    lo, mid, hi = _liquid(['C20H42'], T - dT), _liquid(['C20H42'], T), _liquid(['C20H42'], T + dT)
    assert mid.cp == pytest.approx(row['cp_std_liq_kJ/mol*K'], rel=1e-12)
    assert (hi.h - lo.h)/(2*dT) == pytest.approx(mid.cp, rel=1e-6)
    assert T*(hi.s - lo.s)/(2*dT) == pytest.approx(mid.cp, rel=3e-3)    # up to the ideal-gas fit (dh vs T ds)


def test_liquid_enthalpy_is_continuous_at_tmin_and_vanishes_above_tc():
    row = thermoFT._hc_table().loc['C20H42']
    Tmin = row['HVAP_Tmin_K']
    below, above = _liquid(['C20H42'], Tmin - 1e-6), _liquid(['C20H42'], Tmin + 1e-6)
    assert below.h == pytest.approx(above.h, abs=1e-5)
    assert below.s == pytest.approx(above.s, abs=1e-7)
    hot = _liquid(['C6H14'], 600.0)                                   # above Tc = 507.6 K
    assert np.all(hot.calVAP() == 0.0)
    gas = thermoFT.Thermo(['C6H14'], 600.0, np.array([1.0]), phase='Gas')
    assert hot.h == pytest.approx(gas.h)


def test_rstoic_heat_duty_is_the_enthalpy_balance():
    """RStoic used Q = n1*h1 - (n0*n0 + nH*nH) (flow squared); now flow times enthalpy."""
    x = np.zeros(55)
    x[[0, 2, 5, 6, 7, 8, 10]] = 0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05
    r = reactorsFT.RStoic('r', mediaFT.Syncrude(), F0_n=1.0, F0_Xi=x)
    assert r.run_sol().success
    assert r.FH0.n == pytest.approx((0.1*1 + 0.1*2 + 0.05*3)/3)            # steam for C2-C4 -> CH4 + CO
    assert r.Q == pytest.approx(r.F1.n*r.F1.h - r.F0.n*r.F0.h - r.FH0.n*r.FH0.h, abs=1e-12)
    assert r.Q == pytest.approx(3.732064, rel=1e-5)                          # endothermic steam reforming
    r.ExBs()
    assert r.Exloss > 0 and 0 < r.Exeff < 1


@pytest.fixture
def ft_effluent():
    med = mediaFT.Syncrude()
    sp = [str(s) for s in med.species]
    x = np.full(len(sp), 1e-12)
    for k, v in {'H2': 0.3, 'CO': 0.15, 'CO2': 0.05, 'H2O': 0.25, 'CH4': 0.05, 'C3H8': 0.03, 'C6H14': 0.05,
                 'C10H22': 0.05, 'C16H34': 0.05, 'C25H52': 0.02}.items():
        x[sp.index(k)] = v
    return med, sp, x/x.sum()


def test_vapour_pressures():
    p = thermoFT.vapour_pressures(309.2, ['H2', 'CO', 'H2O', 'CH4', 'C5H12', 'C6H14'])
    assert np.isinf(p[[0, 1, 3]]).all()                               # permanent gases do not dissolve
    assert p[4] == pytest.approx(1.01325, rel=2e-3)                   # pentane at its normal boiling point
    assert p[5] == pytest.approx(0.33, rel=0.05)                       # hexane (Yaws: 0.33 bar)
    assert thermoFT.psat_water(373.124) == pytest.approx(1.01325, rel=1e-4)
    assert thermoFT.psat_water(298.15) == pytest.approx(0.031699, rel=1e-4)


def test_raoult_flash(ft_effluent):
    med, sp, x = ft_effluent
    d = componentsFT.Flash('d', med, T=320.0, P=20.0)
    d.F0.n = 10.0
    d.F0.Xi = x
    d.F0.T = 500.0
    d.F0.P = 20.0
    d.run_sol()
    assert 0 < d.beta < 1 and d.water_liquid > 0
    assert d.mass_balance() == pytest.approx(0.0, abs=1e-9)
    for gas in ('H2', 'CO', 'CO2', 'CH4'):
        assert d.F2.Xi[sp.index(gas)] == 0.0                            # no dissolved gases
    # equilibrium: Raoult for the organics, pure liquid water at its vapour pressure
    psat = thermoFT.vapour_pressures(320.0, sp)
    w, org_liq = sp.index('H2O'), d.F2.n*d.F2.Xi
    n_org = org_liq.sum() - org_liq[w]
    for k in ('C6H14', 'C10H22', 'C16H34'):
        i = sp.index(k)
        assert d.F1.Xi[i]*20.0 == pytest.approx(org_liq[i]/n_org*psat[i], rel=1e-8)
    assert d.F1.Xi[w]*20.0 == pytest.approx(psat[w], rel=1e-8)
    assert d.F2.phase == 'Liquid' and d.F1.phase == 'Gas'
    # second law; not exactly T0*Sgen because the wet feed condenses at the dead state (exergy reference)
    assert d.Sgen > 0 and d.Exloss == pytest.approx(298.15*d.Sgen, rel=1e-3)


def test_raoult_flash_above_the_dew_point_and_adiabatic(ft_effluent):
    med, sp, x = ft_effluent
    hot = componentsFT.Flash('h', med, T=700.0, P=2.0)
    hot.F0.n = 1.0
    hot.F0.Xi = x
    hot.F0.T = 700.0
    hot.F0.P = 2.0
    hot.run_sol()
    assert hot.beta == 1.0 and hot.F2.n == 0.0
    adi = componentsFT.Flash('a', med, Q=0.0, P=5.0)
    adi.F0.n = 10.0
    adi.F0.Xi = x
    adi.F0.T = 500.0
    adi.F0.P = 20.0
    assert adi.run_sol().success and adi.Q == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        componentsFT.Flash('x', med, T=300.0, Q=0.0)
