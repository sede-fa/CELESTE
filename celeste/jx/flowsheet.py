"""
Flowsheets of Equinox units declared by their streams, as in CSL with ``link``.

    class Plant(Flowsheet):
        def __init__(self):
            self.feed = Stream(...)                  # feeds are streams of the plant
            self.mix = units.Mixer(GAS)
            self.heat = units.Heater(GAS, 600.0)
            self.spl = units.Splitter(0.2)
            self.link(self.feed, self.mix.a)
            self.link(self.mix.out, self.heat.feed)
            self.link(self.heat.out, self.spl.feed)
            self.link(self.spl.out1, self.mix.b)     # recycle; spl.out2 is not linked: a product

    s = plant.sequential()      # one sequential-modular solve: streams by name, 'mix.out', 'spl.out2', ...
    s = plant.solve(s)          # every equation together (Newton, exact Jacobian), from s
    plant.pfd()                 # process flow diagram

A stream is named after the outlet it leaves (``'mix.out'``) or the feed (``'feed'``). An outlet that is not
linked is a product. The unknowns of the equation-oriented solve are the species flows and the temperature of
every stream that leaves a unit; their pressures are those of the starting streams. The equations are the
``residuals`` of the units, in the order the units are declared - written out by ``print_equations()``.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from ..tearing import tear_edges
from .units import Port, Stream, Unit, newton, stream_from_vector, stream_vector


class Flowsheet(eqx.Module):
    """Base of a plant built from units and streams (see the module description)."""
    links: tuple = eqx.field(static=True, default=())

    # ---- declaration -------------------------------------------------------------------------------
    def link(self, source, inlet: Port) -> None:
        """Declare a stream from ``source`` (an outlet ``unit.port`` or a feed stream of the plant) to ``inlet``."""
        if not isinstance(inlet, Port) or inlet.name not in type(inlet.unit).inlets:
            raise ValueError(f"link: {getattr(inlet, 'name', inlet)} is not an inlet port such as mix.a ({type(getattr(inlet, 'unit', None)).__name__})")
        if isinstance(source, Port):
            if source.name not in type(source.unit).outlets:
                raise ValueError(f"link: {source.name} is not an outlet of {type(source.unit).__name__}")
            src = f'{self._field(source.unit)}.{source.name}'
        elif isinstance(source, Stream):
            src = self._field(source)
        else:
            raise ValueError("link: the source must be an outlet port (mix.out) or a feed stream of the plant")
        self.links = self.links + ((src, f'{self._field(inlet.unit)}.{inlet.name}'),)

    def _field(self, obj) -> str:
        name = next((k for k, v in vars(self).items() if v is obj), None)
        if name is None:
            raise ValueError("link: assign the unit or feed to the plant (self.name = ...) before linking it")
        return name

    # ---- structure (all from the declared links) ---------------------------------------------------
    @property
    def units(self) -> dict:
        """{name: unit} of the units in the flowsheet, in declaration order."""
        used = {name.split('.')[0] for link in self.links for name in link}
        return {k: v for k, v in vars(self).items() if isinstance(v, Unit) and k in used}

    @property
    def feeds(self) -> dict:
        """{name: stream} of the feeds."""
        return {src: getattr(self, src) for src, _ in self.links if '.' not in src}

    @property
    def products(self) -> list:
        """Names of the outlets that are not linked."""
        linked = {src for src, _ in self.links}
        return [f'{u}.{p}' for u, unit in self.units.items() for p in type(unit).outlets if f'{u}.{p}' not in linked]

    @property
    def stream_names(self) -> list:
        """Every stream that leaves a unit (the unknowns of the equation-oriented solve), in unit order."""
        return [f'{u}.{p}' for u, unit in self.units.items() for p in type(unit).outlets]

    def _source_of(self) -> dict:
        """inlet 'unit.port' -> name of the stream that feeds it."""
        into = {dst: src for src, dst in self.links}
        for u, unit in self.units.items():
            for p in type(unit).inlets:
                if f'{u}.{p}' not in into:
                    raise ValueError(f"{type(self).__name__}: inlet {u}.{p} has no stream (link)")
        return into

    def _plan(self):
        """Unit order and tear streams: the same tearing as the CSL flowsheets (fewest tears, back streams first)."""
        G = nx.MultiDiGraph()
        names = list(self.units)
        G.add_nodes_from(names)
        for src, dst in self.links:
            if '.' in src:
                G.add_edge(src.split('.')[0], dst.split('.')[0], stream=src)
        order = {u: i for i, u in enumerate(names)}
        tears = []
        for comp in nx.strongly_connected_components(G):
            sub = G.subgraph(comp).copy()
            if len(comp) > 1 or sub.number_of_edges():
                tears += tear_edges(sub, order)
        H = G.copy()
        H.remove_edges_from(tears)
        return list(nx.lexicographical_topological_sort(H, key=order.get)), sorted({G.edges[e]['stream'] for e in tears})

    # ---- sequential modular ------------------------------------------------------------------------
    def evaluate(self, streams: dict) -> dict:
        """Call every unit once in the solve order with the given streams (feeds and tear guesses); all streams."""
        order, _ = self._plan()
        into = self._source_of()
        s = dict(self.feeds, **streams)
        for u in order:
            unit = getattr(self, u)
            out = unit(*[s[into[f'{u}.{p}']] for p in type(unit).inlets])
            out = out if isinstance(out, tuple) else (out,)
            s.update({f'{u}.{p}': o for p, o in zip(type(unit).outlets, out)})
        return s

    def sequential(self, guesses: dict | None = None, tol: float = 1e-10, max_iter: int = 200) -> dict:
        """
        Sequential-modular solve: the units in the generated order, the tear streams converged by successive
        substitution (a ``lax.while_loop``, so it compiles and vectorises). ``guesses`` = {tear stream: Stream};
        a tear without a guess starts from the first feed with as many species. Returns {name: Stream}.
        """
        _, tears = self._plan()
        guesses = dict(guesses or {})
        for t in tears:
            if t not in guesses:
                guesses[t] = self._default_guess(t, guesses)
        if not tears:
            return self.evaluate({})
        P = {t: guesses[t].P for t in tears}

        def run(x):
            parts = jnp.split(x, sizes[:-1])
            return self.evaluate({t: stream_from_vector(v, P[t]) for t, v in zip(tears, parts)})

        def pack(s):
            return jnp.concatenate([stream_vector(s[t]) for t in tears])

        x0 = jnp.concatenate([stream_vector(guesses[t]) for t in tears])
        sizes = list(np.cumsum([stream_vector(guesses[t]).size for t in tears]))

        def body(c):
            x, _, k = c
            x_new = pack(run(x))
            return x_new, jnp.max(jnp.abs(x_new - x)/(jnp.abs(x) + 1e-3)), k + 1

        x, _, _ = jax.lax.while_loop(lambda c: (c[1] > tol) & (c[2] < max_iter), body, (x0, jnp.inf, 0))
        return run(x)

    def _default_guess(self, tear: str, guesses: dict) -> Stream:
        """The first feed whose species count matches the tear stream (found from one trial pass)."""
        feeds = list(self.feeds.values())
        _, tears = self._plan()
        trial = {t: guesses.get(t, feeds[0]) for t in tears}
        size = jax.eval_shape(lambda: self.evaluate(trial)[tear].x).shape[0]
        return next((f for f in feeds if f.x.shape[0] == size), feeds[0])

    # ---- equation oriented -------------------------------------------------------------------------
    def pack(self, s: dict) -> jax.Array:
        """The unknowns: species flows and temperature of every stream that leaves a unit."""
        return jnp.concatenate([stream_vector(s[k]) for k in self.stream_names])

    def unpack(self, x, ref: dict) -> dict:
        """Streams from the unknowns ``x``; sizes and pressures from the streams ``ref`` (a starting solution)."""
        s, i = dict(self.feeds), 0
        for k in self.stream_names:
            m = ref[k].x.shape[0] + 1
            s[k] = stream_from_vector(x[i:i + m], ref[k].P)
            i += m
        return s

    def residuals(self, x, ref: dict) -> jax.Array:
        """Equations of all units, ``unit.residuals(*inlets, *outlets)``, in declaration order."""
        s = self.unpack(x, ref)
        into = self._source_of()
        out = []
        for u, unit in self.units.items():
            ins = [s[into[f'{u}.{p}']] for p in type(unit).inlets]
            outs = [s[f'{u}.{p}'] for p in type(unit).outlets]
            out.append(unit.residuals(*ins, *outs))
        return jnp.concatenate(out)

    def solve(self, start: dict | None = None) -> dict:
        """Equation-oriented solve of all units together from ``start`` (default: ``sequential()``)."""
        start = self.sequential() if start is None else start
        return self.unpack(newton(lambda x: self.residuals(x, start), self.pack(start)), start)

    def print_equations(self) -> str:
        """The equation system: every unit with its equations, in and out streams (also printed)."""
        into = self._source_of()
        lines = [f'{type(self).__name__}: {len(self.units)} units, {len(self.stream_names)} streams as unknowns '
                 f'(species flows and T), feeds {", ".join(self.feeds)}; products {", ".join(self.products)}']
        for u, unit in self.units.items():
            doc = (type(unit).residuals.__doc__ or '').strip().split('\n')[0] or f'see {type(unit).__name__}.residuals'
            ins = ', '.join(into[f'{u}.{p}'] for p in type(unit).inlets)
            outs = ', '.join(f'{u}.{p}' for p in type(unit).outlets)
            lines.append(f'  {u} ({type(unit).__name__}): {ins} -> {outs}\n      {doc}')
        text = '\n'.join(lines)
        print(text)
        return text

    # ---- drawing -----------------------------------------------------------------------------------
    def pfd(self, streams: dict | None = None, species: dict | None = None, path: str | None = None, views=None):
        """
        Process flow diagram (:mod:`celeste.pfd`). ``streams`` = a solution ({name: Stream}) shows the stream
        states on the lines; ``species`` = {number of species: names} labels the compositions.
        """
        from ..pfd import StreamView, Terminal, draw
        species = species or {}
        streams = streams or {}

        class _Line:
            def __init__(self, s):
                self.s = s

            def pfd_view(self):
                s = self.s
                if s is None:
                    return StreamView()
                x = [float(v) for v in s.x]
                names = species.get(len(x), [f'x[{i}]' for i in range(len(x))])
                view = StreamView().add('n', float(s.n), 'mol/s').add('T', float(s.T), 'K').add('P', float(s.P), 'bar')
                view.composition = sorted(((n, v) for n, v in zip(names, x) if v > 1e-6), key=lambda t: -t[1])
                return view

        lines = []
        for src, dst in self.links:
            a = src.split('.')[0] if '.' in src else Terminal(src, 'feed')
            lines.append((a, dst.split('.')[0], _Line(streams.get(src)), f'{src} → {dst}'))
        for p in self.products:
            lines.append((p.split('.')[0], Terminal(p, 'product'), _Line(streams.get(p)), p))
        return draw(type(self).__name__, self.units, lines, views=views, path=path)
