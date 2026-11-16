# Vapour–liquid flash

`celeste.CSL.separation.Flash` splits a feed into a vapour and a liquid stream
in thermodynamic equilibrium. The split is computed, not prescribed – in
contrast to the black-box separators `GPU_2S`, `GPU_2SV2`, `GPU_3S`.

    F0 (feed)  ──►  ┌───────┐ ──►  F1 (vapour)
                    │ Flash │
                    └───────┘ ──►  F2 (liquid)

## Usage

```python
from celeste.CSL import media, separation

drum = separation.Flash('drum', Medium=media.Fuel_CH4())
```

| Specification | Arguments | Result |
|---|---|---|
| phase separator at feed conditions | none | `Q` is the heat that has to be removed to reach equilibrium |
| isothermal flash | `T=...` (and optionally `P=...`) | `Q` |
| adiabatic flash | `Q=0.0`, `P=...` | drum temperature `T` |
| flash with a given duty | `Q=...` | drum temperature `T` |

`P` defaults to the feed pressure minus `dP`.

After `run_sol()`:

| Attribute | Meaning |
|---|---|
| `beta` | vapour fraction, mol vapour per mol feed |
| `phase` | `'two-phase'`, `'vapour'` or `'liquid'` |
| `K` | equilibrium ratios y/x |
| `recovery` | fraction of every component that leaves with the liquid |
| `F1`, `F2` | outlet flanges; an outlet without flow has `n = 0` |
| `Q`, `Exth`, `Exloss`, `Sgen` | as for the other units |

In a system the drum is used like any other unit:

```python
self.connect_seq(self.cooler.F1, self.drum.F0, store=store)
self.drum._sol = self.drum.run_sol()
self.connect_seq(self.drum.F1, self.compressor.F0, store=store)   # gas
self.connect_seq(self.drum.F2, self.water_sink.F0, store=store)   # condensate
```

`CSL.systems.ProcessBoxes.CondenserBox` packages a cooler and a drum.

## Two ways to use it

**With the ideal-gas (NASA) media** – `Flash('d', Medium=media.Fuel_MeOH())`.
The K-values come from the Peng–Robinson equation, the stream properties from
the media: `F1` uses `Medium`, `F2` uses its liquid counterpart (`H2O` →
`H2O(L)`, `CH3OH` → `CH3OH(L)`; dissolved gases keep their gas species). Pass
`MediumL=` to choose the liquid medium yourself. This fits into existing
flowsheets without changing any other unit.

The feed is a gas-phase medium, so its enthalpy is that of a gas even below the
dew point; the heat of condensation appears in the `Q` of the drum. Because the
split and the properties come from two models, the exergy balance of the drum
is closed only approximately (a few per cent of the drum's own contribution).

**With real-fluid media** – `Flash('d', Medium=media.Fuel_MeOH_PR())`. Feed and
products use the same Peng–Robinson model, so energy, entropy and exergy
balances are consistent, and upstream units already know the phase state of
their streams (a cooler with such a medium condenses).

## Algorithm

`celeste.jx.flash.flash_pt(T, P, z, params)`:

1. **Stability test** (Michelsen's tangent plane criterion) from a vapour-like
   and a liquid-like trial phase. The feed is split only if a trial phase
   lowers the Gibbs energy. A single-phase feed therefore never yields a
   spurious two-phase answer, and a two-phase feed is not missed because of a
   poor first guess.
2. **Phase split** by successive substitution on ln K, started from the
   K-values of the stability test and accelerated every fifth step.
3. **Rachford–Rice** equation solved by a bracketed Newton iteration between
   its asymptotes.

The adiabatic flash wraps this in a bracketing search for the temperature; the
product enthalpy is monotonic in temperature.

Everything runs in fixed-shape JAX loops: the flash can be compiled, vectorised
(`jax.vmap`) and differentiated in forward mode (`jax.jacfwd`).

## Validation

| Check | Result |
|---|---|
| fugacity coefficients and departure functions against derivatives of the Gibbs energy (automatic differentiation) | agree to 1e-12 |
| saturation pressure of water, methanol, ammonia, CO2 against reference equations | < 1 % |
| flash against an independent Peng–Robinson implementation (`thermo` library) | 6 digits |
| 4000 random flashes, 7 components, 250–650 K, 0.3–300 bar | all converged; component balance 1e-16; fugacity equality 1e-11 |
| single-phase results | pass the stability test |
| vapour fraction along an isobar | monotonic |

## Limits

- **Two phases at most.** Mixtures that form a second liquid (for example
  liquid CO2 next to water at low temperature and high pressure) are returned
  as vapour + one liquid. In the random scan this affected about 1 % of the
  two-phase results. `celeste.jx.flash.stability` applied to the outlet
  compositions reveals such cases.
- Electrolytes, association and solids are not described.
- Peng–Robinson with one interaction parameter is approximate for strongly
  non-ideal liquids (water–alcohol): compositions of the liquid phase should
  be checked against data if they matter.
- Compilation takes a few seconds for every new set of species.

## Fischer–Tropsch streams: Raoult flash

`celeste.FT.componentsFT.Flash` is a simpler flash for FT media (`mediaFT.*`),
where Peng–Robinson data for 50 hydrocarbons would be needed otherwise:

- The hydrocarbons C2+ form an ideal liquid solution: y_i P = x_i Psat_i.
- Water condenses as a separate, pure liquid: y_w P = Psat_w (water and
  hydrocarbons are practically immiscible). The liquid outlet carries both
  liquids; the FT liquid properties already treat water as a separate phase.
- H2, CO, CO2, N2, CH4 and every species above its critical temperature do not
  dissolve.

The hydrocarbon vapour pressures come from the same Clausius–Clapeyron
integral that gives the FT liquid entropy (`thermoFT.vapour_pressures`), so
the split and the stream enthalpies are consistent; water uses the
Wagner–Pruss equation. Pentane boils at 1.013 bar at 309.2 K, hexane has
0.33 bar at 309 K (Yaws).

```python
drum = componentsFT.Flash('drum', mediaFT.Syncrude(), T=310.0, P=20.0)   # or Q=0.0 for adiabatic
drum.F0.n = 10.0
drum.F0.Xi = x_effluent
drum.F0.T = 500.0
drum.F0.P = 20.0
drum.run_sol()
drum.beta, drum.water_liquid, drum.F1, drum.F2
```

Use `CSL.separation.Flash` with real-fluid media when dissolved gases or
non-ideal liquids matter (high-pressure separators, light-end recovery).
