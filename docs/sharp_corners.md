# Sharp corners

Behaviour that is intended but can surprise a new user. None of these are bugs;
open problems are listed in [known_issues.md](known_issues.md).

## Flanges and connections

- **Each flange owns a copy of its medium.** `Flange(name, Medium)` copies the
  medium, so two flanges never share state. Changing `unit.F0.Medium` after
  construction does not affect `unit.F1`.
- **Connections are checked with `Flange.compatible()`.** Two flanges can be
  connected when they are the same port class, carry the same medium class and
  list the same species in the same order. `connect` and `connect_seq` raise a
  `ValueError` that names the reason (`flange.incompatibility(other)` returns it
  as text). To pass a stream between different media, put a
  `components.ChgMedium` in between.
- **Flanges compare by identity.** `f == g` is true only for the same object,
  so flanges can be used in sets and as dictionary keys. Use
  `f.compatible(g)` to ask whether two flanges could be connected.
- **Ports ending in `0` are inlets.** Every other port is an outlet. `Mixer`
  is the exception (`F0` and `F1` in, `F2` out) and declares `_inlet_ports`.
- **Outlet values can be specifications.** A unit does not compute every
  outlet value: the outlet temperature of a `Heater` and the outlet pressure of
  a compressor are what you set on `F1`. Inlet values are overwritten as soon
  as an upstream unit is connected with `connect_seq`.

## Thermodynamics

- **Ideal-gas media have no Joule–Thomson effect.** An isenthalpic `Valve`
  keeps the temperature of a NASA (ideal-gas) medium; use a `*_PR` medium to
  see throttling cooling.
- **`Pump` treats the liquid as incompressible** with the density of the inlet
  medium; the temperature rise comes from the energy balance.
- **Mixed-model flash.** `Flash` with NASA media takes the phase split from
  Peng–Robinson and the stream properties from the media. Its exergy balance is
  therefore approximate (in the methanol loop example the drum shows −0.3 % of
  the total exergy destruction). Use real-fluid media where this matters.
- **`Thermo_CP` with a liquid-forming default composition.** The reference
  state is evaluated without a phase specification while the stream state uses
  the given phase. For `Fuel_CH4_CP()` at its default composition (90 % water,
  298 K) the enthalpy differs from the NASA value by the heat of vaporisation.

## Fischer–Tropsch module

- **The FT model does not determine the phase of a stream**; it evaluates the
  phase set on the flange. A liquid flange above the bubble point is still
  liquid.
- **Reference composition of `mediaFT.Syncrude`.** Its first two entries (H2,
  CO2) are zero, so an `FT` reactor created without a feed composition has no
  hydrogen and no conversion. Set the feed on `F0`.
