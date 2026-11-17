"""
Flowsheets declared by their topology: the solve order and the tear streams are generated.

A system lists its streams once with ``self.link(outlet, inlet)``. ``solve_order()`` builds the unit graph,
finds the recycle loops (strongly connected components), picks the fewest tear streams that break every loop
and orders the units; ``seq_sol`` then runs that plan, converging each loop with ``Recycle_Tear`` on its tear
streams. A system that writes its own ``seq_sol`` is not affected.
"""
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx


@dataclass
class Step:
    """One step of a solve plan: a unit, or a recycle loop of units converged on its tear streams."""
    units: list
    links: list = field(default_factory=list)       #: links into the units, connected before each unit runs
    tears: list = field(default_factory=list)       #: (outlet, inlet) pairs iterated by Recycle_Tear

    @property
    def is_loop(self) -> bool:
        return bool(self.tears)


class FlowsheetMixin:
    """Declared streams (``link``), generated solve order (``solve_order``) and its execution (``auto_sol``)."""

    def link(self, outlet, inlet) -> None:
        """
        Declare a stream from ``outlet`` (a flange of a unit, or a feed flange of the system) to ``inlet``
        (a flange of a unit, or a product flange of the system). Declare every stream once; the order of the
        declarations does not matter, except that the units' own declaration order breaks ties when tear
        streams are chosen (a stream that runs back to an earlier unit is torn first).
        """
        self._check_connection(outlet, inlet)
        if '_streams' not in self.__dict__:
            self.__dict__['_streams'] = []
        self._streams.append((outlet, inlet))
        self.__dict__.pop('_plan', None)

    def _run_streams(self) -> list:
        """
        The recorded connections between units of this system. A flange of the system that a run uses as the
        iterate of a tear stream (stack outlet -> ``guess`` -> next unit) is not a stream of its own: the two
        recorded connections through it are joined into one (stack outlet -> next unit).
        """
        owners = self._owner_map()
        own = {id(f) for f in self.get_Flanges()}
        pairs = [(a, b) for a, b in self.__dict__.get('_links', {}) if id(a) in owners and id(b) in owners]
        into = {id(b): a for a, b in pairs if id(b) in own and id(a) not in own}
        out_of = {id(a) for a, b in pairs if id(a) in own and id(b) not in own}
        through = set(into) & out_of
        streams = []
        for a, b in pairs:
            if id(b) in through:
                continue
            streams.append((into[id(a)] if id(a) in through else a, b))
        return streams

    def links_from_run(self, declare: bool = True) -> list:
        """
        The streams made by ``connect_seq`` (and by ``Recycle_Tear``) in the runs so far, as (outlet, inlet) flange
        pairs between units of this system, in the order they were made. With ``declare=True`` they replace the
        declared streams, so a hand-written ``seq_sol`` run once gives the declared flowsheet (``solve_order``,
        ``auto_sol``, tear streams, ``pfd``) without writing the ``link`` calls again. Connections made inside a
        subsystem (a stage of a cascade, say) belong to that subsystem: call it there.
        """
        pairs = self._run_streams()
        if declare:
            self.__dict__['_streams'] = []
            self.__dict__.pop('_plan', None)
            for outlet, inlet in pairs:
                self.link(outlet, inlet)
        return pairs

    @property
    def streams(self) -> list:
        """Declared streams as (outlet, inlet) flange pairs."""
        return list(self.__dict__.get('_streams', []))

    # ---- graph ----------------------------------------------------------------------------------------
    def _owner_map(self) -> dict:
        """id(flange) -> unit for the flanges of the units of this system (the system for its own flanges)."""
        owners = {id(f): self for f in self.get_Flanges()}
        for unit in self.getObjs():
            for f in unit.get_Flanges():
                owners[id(f)] = unit
        return owners

    def unit_graph(self) -> nx.MultiDiGraph:
        """Units as nodes, declared streams as edges (attribute ``flanges`` = (outlet, inlet))."""
        owners = self._owner_map()
        G = nx.MultiDiGraph()
        units = [u for u in self.getObjs()]
        G.add_nodes_from(units)
        for outlet, inlet in self.streams:
            src, dst = owners.get(id(outlet)), owners.get(id(inlet))
            if src is None or dst is None:
                missing = outlet if src is None else inlet
                raise ValueError(f"{self.name}: flange {missing.name} does not belong to a unit of this system")
            if src is not self and dst is not self:
                G.add_edge(src, dst, flanges=(outlet, inlet))
        return G

    @staticmethod
    def _tear_edges(G: nx.MultiDiGraph, order: dict) -> list:
        """
        Fewest edges whose removal leaves G acyclic; among equally small sets the one with most edges running
        back in the flow ``order``. Exhaustive up to three tears, greedy beyond.
        """
        edges = list(G.edges(keys=True))

        def acyclic_without(tears):
            H = G.copy()
            H.remove_edges_from(tears)
            return nx.is_directed_acyclic_graph(H)

        def backwardness(tears):
            return sum(order[u] >= order[v] for u, v, _ in tears)

        for k in range(1, min(3, len(edges)) + 1):
            sets = [c for c in combinations(edges, k) if acyclic_without(c)]
            if sets:
                return list(max(sets, key=backwardness))
        # greedy: tear the edge that lies on most simple cycles, prefer back edges, until acyclic
        tears, H = [], G.copy()
        while not nx.is_directed_acyclic_graph(H):
            counts = {}
            for cycle in nx.simple_cycles(nx.DiGraph(H)):
                for u, v in zip(cycle, cycle[1:] + cycle[:1]):
                    counts[(u, v)] = counts.get((u, v), 0) + 1
            u, v = max(counts, key=lambda e: (counts[e], order[e[0]] >= order[e[1]]))
            key = next(iter(H.get_edge_data(u, v)))
            H.remove_edge(u, v, key)
            tears.append((u, v, key))
        return tears

    def _flow_order(self, G) -> dict:
        """
        Position of every unit along the flow: its distance (in streams) from the feeds - units without an
        inlet stream and units fed by the system - then the declaration order. A stream that runs to a unit
        no further along than its source is a back stream: the preferred tear.
        """
        owners = self._owner_map()
        fed = {owners[id(inlet)] for outlet, inlet in self.streams if owners[id(outlet)] is self}
        starts = [u for u in G if G.in_degree(u) == 0 or u in fed]
        depth = nx.multi_source_dijkstra_path_length(G, starts, weight=lambda u, v, d: 1) if starts else {}
        index = {u: i for i, u in enumerate(self.getObjs())}
        return {u: (depth.get(u, len(G)), index[u]) for u in G}

    def solve_order(self) -> list:
        """
        The solve plan generated from the declared streams: a list of :class:`Step` (a single unit, or a recycle
        loop with its units in order and its tear streams). Printing it with ``print_solve_order`` shows it.
        """
        if '_plan' in self.__dict__:
            return self._plan
        G = self.unit_graph()
        order = self._flow_order(G)
        C = nx.condensation(G)
        plan = []
        for c in nx.lexicographical_topological_sort(C, key=lambda c: min(order[u] for u in C.nodes[c]['members'])):
            members = C.nodes[c]['members']
            sub = G.subgraph(members).copy()
            tears = self._tear_edges(sub, order) if len(members) > 1 or sub.number_of_edges() else []
            sub.remove_edges_from(tears)
            units = list(nx.lexicographical_topological_sort(sub, key=lambda u: order[u]))
            torn = {sub_e for sub_e in tears}
            links = [d['flanges'] for u, v, k, d in G.in_edges(members, keys=True, data=True)
                     if (u, v, k) not in torn]
            plan.append(Step(units, links, [G.edges[e]['flanges'] for e in tears]))
        self.__dict__['_plan'] = plan
        return plan

    def print_solve_order(self) -> str:
        """Text of the solve plan (also printed)."""
        lines = []
        for i, step in enumerate(self.solve_order(), 1):
            names = ', '.join(u.name for u in step.units)
            if step.is_loop:
                tears = '; '.join(f'{a.name} -> {b.name}' for a, b in step.tears)
                lines.append(f'{i}. loop [{names}]  tear: {tears}')
            else:
                lines.append(f'{i}. {names}')
        text = '\n'.join(lines)
        print(text)
        return text

    # ---- execution ------------------------------------------------------------------------------------
    def _run_units(self, step: Step, store: bool) -> None:
        for unit in step.units:
            for outlet, inlet in step.links:
                if any(inlet is f for f in unit.get_Flanges()):
                    self.connect_seq(outlet, inlet, store=store)
            if hasattr(unit, 'run_sol'):
                self.solve(unit)

    def auto_sol(self, store: bool = False, eps: float = 1e-9, max_iter: int = 300, method: str = 'anderson') -> dict:
        """
        Run the generated plan: feeds into the units, units in order, recycle loops converged with
        ``Recycle_Tear`` on their tear streams, products to the system's outlet flanges.
        Returns {loop number: (passes, error)} of the converged loops.
        """
        owners = self._owner_map()
        for outlet, inlet in self.streams:                       # system feeds
            if owners[id(outlet)] is self:
                self.connect_seq(outlet, inlet, store=store)
        result = {}
        for i, step in enumerate(self.solve_order(), 1):
            if step.is_loop:
                outlets, inlets = [a for a, _ in step.tears], [b for _, b in step.tears]
                result[i] = self.Recycle_Tear(outlets, inlets, flow_method=lambda s=step: self._run_units(s, False),
                                              eps=eps, max_iter=max_iter, method=method)
                for outlet, inlet in (step.links + step.tears) if store else []:     # recorded for the graphs
                    self._store_connection(outlet, inlet)
            else:
                self._run_units(step, store)
        for outlet, inlet in self.streams:                       # system products
            if owners[id(inlet)] is self:
                self.connect_seq(outlet, inlet, store=store)
        return result
