"""
Tear streams of a flowsheet graph: the fewest edges whose removal breaks every loop. Used by the solve order of
declared flowsheets and by the process flow diagrams (``celeste.pfd``), so it depends on networkx only.
"""
from itertools import combinations

import networkx as nx


def tear_edges(G: nx.MultiDiGraph, order: dict) -> list:
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
