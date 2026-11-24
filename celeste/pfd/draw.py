"""
Drawing of a process flow diagram from units and streams, independent of the modelling framework.

    pfd = draw('plant', units={'mix': mix, 'reactor': reactor, ...},
               streams=[('mix', 'reactor', mix.F2, 'mix.F2 -> reactor.F0'), (Terminal('feed'), 'mix', feed, None), ...])

Each unit and stream is asked for its view (``pfd_view()``, see :mod:`celeste.pfd.views`).

Layout (the "layered" method of Sugiyama et al., as in Graphviz dot and ELK layered): recycle streams are torn
(fewest tears, back streams preferred), every unit is placed in the layer of its longest path from the feeds,
the order within each layer is improved by barycentre sweeps, and the streams are routed orthogonally - forward
streams through the gap before their destination, recycles along lanes below the units.
"""
import html as _html
import json
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..tearing import tear_edges
from .page import _PAGE
from .views import KINDS, Row, StreamView, UnitView, view_of

_NODE_H, _GAP_X, _GAP_Y, _MARGIN, _LANE = 46.0, 96.0, 34.0, 28.0, 14.0


@dataclass(frozen=True)
class Terminal:
    """A feed or product of the drawn flowsheet (a stream end outside every unit), drawn as a small circle."""
    label: str
    note: str = ''


class PFD:
    """A drawn flowsheet: ``html`` (the page), ``save(path)``, displayed inline in Jupyter."""

    def __init__(self, html: str, width: float, height: float, name: str):
        self.html, self.width, self.height, self.name = html, width, height, name

    def save(self, path: str | None = None) -> str:
        path = path or f'{self.name}_pfd.html'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.html)
        return path

    def _repr_html_(self) -> str:
        return (f'<iframe srcdoc="{_html.escape(self.html, quote=True)}" style="width:100%;'
                f'height:{int(self.height) + 140}px;border:0"></iframe>')


# ---- hover tables -----------------------------------------------------------------------------------------
def _num(value, digits=5):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return f'{v:.{digits}g}'


def _cell(value):
    if isinstance(value, (bool, np.bool_)):
        return 'yes' if value else '<b class="bad">no</b>'
    if isinstance(value, str):
        return _html.escape(value)
    return _num(value)


def _table(title, subtitle, rows, composition=None) -> str:
    out = [f'<div class="tt-title">{_html.escape(title)}</div>']
    if subtitle:
        out.append(f'<div class="tt-sub">{_html.escape(subtitle)}</div>')
    body = ''
    for r in rows:
        r = r if isinstance(r, Row) else Row(*r)
        cell = _cell(r.value)
        if cell is not None:
            label = f'{r.label}  {r.unit}' if r.unit else r.label
            body += f'<tr><th>{_html.escape(label)}</th><td>{cell}</td></tr>'
    out.append(f'<table>{body}</table>')
    if composition:
        comp = ''.join(f'<tr><th>{_html.escape(str(sp))}</th><td>{x:.4g}</td></tr>' for sp, x in composition)
        out.append(f'<div class="tt-sub">mole fractions</div><table>{comp}</table>')
    return ''.join(out)


# ---- drawing ----------------------------------------------------------------------------------------------
def draw(name: str, units: dict, streams: list, views: dict | None = None, path: str | None = None) -> PFD:
    """
    Draw a flowsheet.

    Parameters:
        name:     title of the diagram
        units:    {label: unit object}
        streams:  [(source, destination, stream object, title)]: source and destination are unit labels or
                  :class:`Terminal` (a feed or product); ``title`` defaults to 'source -> destination'
        views:    {label or class: function(object) -> view} replacing ``pfd_view()`` of those units
        path:     also write the HTML file
    """
    views = views or {}

    def override(label, obj):
        if label in views:
            return views[label]
        return next((f for k, f in views.items() if isinstance(k, type) and isinstance(obj, k)), None)

    G = nx.MultiDiGraph()
    G.add_nodes_from(units)
    for src, dst, stream, title in streams:
        G.add_edge(src, dst, stream=stream, title=title or f'{_label(src)} → {_label(dst)}')

    # 1. tear the recycles; 2. layers by longest path; 3. order within layers
    index = {u: i for i, u in enumerate(G.nodes)}
    starts = [u for u in G if G.in_degree(u) == 0]
    depth = nx.multi_source_dijkstra_path_length(G, starts, weight=lambda u, v, d: 1) if starts else {}
    order = {u: (depth.get(u, len(G)), index[u]) for u in G}
    tears = set()
    for comp in nx.strongly_connected_components(G):
        sub = G.subgraph(comp).copy()
        if len(comp) > 1 or sub.number_of_edges():
            tears.update(tear_edges(sub, order))
    H = G.copy()
    H.remove_edges_from(tears)
    rank = {}
    for u in nx.topological_sort(H):
        rank[u] = max((rank[p] + 1 for p in H.predecessors(u)), default=0)
    layers = {}
    for u in sorted(G, key=lambda u: order[u]):
        layers.setdefault(rank[u], []).append(u)
    for _ in range(4):                                      # barycentre sweeps, down and up
        for r in sorted(layers)[1:]:
            pos = {u: i for l in layers.values() for i, u in enumerate(l)}
            layers[r].sort(key=lambda u: np.mean([pos[p] for p in H.predecessors(u)]) if any(True for _ in H.predecessors(u)) else pos[u])
        for r in sorted(layers, reverse=True)[1:]:
            pos = {u: i for l in layers.values() for i, u in enumerate(l)}
            layers[r].sort(key=lambda u: np.mean([pos[s] for s in H.successors(u)]) if any(True for _ in H.successors(u)) else pos[u])

    # 4. coordinates
    def width(u):
        return 22.0 if isinstance(u, Terminal) else max(84.0, 7.4*len(_label(u)) + 26)

    col_w = {r: max(width(u) for u in layers[r]) for r in layers}
    x, X = _MARGIN, {}
    for r in sorted(layers):
        X[r] = x
        x += col_w[r] + _GAP_X
    tallest = max(len(l) for l in layers.values())
    box = {}
    for r, l in layers.items():
        offset = (tallest - len(l))*(_NODE_H + _GAP_Y)/2
        for i, u in enumerate(l):
            w = width(u)
            h = 22.0 if isinstance(u, Terminal) else _NODE_H
            cx = X[r] + col_w[r]/2
            cy = _MARGIN + 30 + offset + i*(_NODE_H + _GAP_Y) + _NODE_H/2
            box[u] = (cx - w/2, cy - h/2, w, h)
    bottom = max(b[1] + b[3] for b in box.values())
    W = x - _GAP_X + _MARGIN
    Hgt = bottom + _MARGIN + (len(tears) + 1)*_LANE + 10

    # 5. ports: spread the streams over the sides of each box
    out_edges = {u: sorted(G.out_edges(u, keys=True), key=lambda e: box[e[1]][1]) for u in G}
    in_edges = {u: sorted(G.in_edges(u, keys=True), key=lambda e: box[e[0]][1]) for u in G}

    def port(u, e, side):
        edges = out_edges[u] if side == 'out' else in_edges[u]
        bx, by, bw, bh = box[u]
        k, n = edges.index(e), len(edges)
        y = by + bh*(k + 1)/(n + 1)
        return (bx + bw, y) if side == 'out' else (bx, y)

    # 6. routes: forward streams turn in the gap before their destination, recycles run along lanes below
    gap_use, paths, tips, lane = {}, [], {}, 0
    for e in G.edges(keys=True):
        u, v, _ = e
        (x1, y1), (x2, y2) = port(u, e, 'out'), port(v, e, 'in')
        back = e in tears or rank[v] <= rank[u]
        if not back:
            k = gap_use.get(rank[v], 0)
            gap_use[rank[v]] = k + 1
            xm = X[rank[v]] - 14 - (k % 6)*9                 # in the gap before the destination column
            d = f'M{x1:.1f},{y1:.1f} H{xm:.1f} V{y2:.1f} H{x2:.1f}'
        else:
            yl = bottom + _LANE*(lane + 1)
            lane += 1
            xa, xb = x1 + 10 + lane*8, X[rank[v]] - 10 - lane*8
            d = f'M{x1:.1f},{y1:.1f} H{xa:.1f} V{yl:.1f} H{xb:.1f} V{y2:.1f} H{x2:.1f}'
        sid = f's{len(paths)}'
        data = G.edges[e]
        sv = view_of(data['stream']) or StreamView()
        tips[sid] = _table(data['title'], sv.subtitle, sv.rows, sv.composition)
        paths.append((sid, d, back))

    # 7. SVG
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {Hgt:.0f}" width="{W:.0f}" '
             f'height="{Hgt:.0f}" role="img" aria-label="Process flow diagram of {_html.escape(name)}">',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>']
    for sid, d, back in paths:
        parts.append(f'<g class="stream{" recycle" if back else ""}" data-tip="{sid}">'
                     f'<path class="hit" d="{d}"/><path class="line" d="{d}" marker-end="url(#arrow)"/></g>')
    used = set()
    for u, (bx, by, bw, bh) in box.items():
        uid = f'u{len(tips)}'
        if isinstance(u, Terminal):
            tips[uid] = _table(u.label, u.note, [])
            parts.append(f'<g class="terminal" data-tip="{uid}"><circle cx="{bx + bw/2:.1f}" cy="{by + bh/2:.1f}" '
                         f'r="{bw/2:.1f}"/><text x="{bx + bw/2:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
                         f'class="small">{_html.escape(u.label)}</text></g>')
            continue
        obj = units[u]
        uv = view_of(obj, override(u, obj)) or UnitView()
        kind = uv.kind if uv.kind in KINDS else 'other'
        used.add(kind)
        subtitle = uv.subtitle or type(obj).__name__
        tips[uid] = _table(u, subtitle, uv.rows)
        parts.append(f'<g class="unit {kind}" data-tip="{uid}"><rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                     f'height="{bh:.1f}" rx="7"/><text x="{bx + bw/2:.1f}" y="{by + 19:.1f}" text-anchor="middle">'
                     f'{_html.escape(u)}</text><text x="{bx + bw/2:.1f}" y="{by + 35:.1f}" '
                     f'text-anchor="middle" class="small">{_html.escape(subtitle)}</text></g>')
    parts.append('</svg>')
    legend = ''.join(f'<span class="key unit {k}"><i></i>{KINDS[k]}</span>' for k in KINDS if k in used)
    page = _PAGE.format(title=_html.escape(name), legend=legend, svg=''.join(parts),
                        tips=json.dumps(tips).replace('</', '<\\/'), n_units=len(units), n_streams=len(paths),
                        n_tears=len(tears))
    result = PFD(page, W, Hgt, name)
    if path:
        result.save(path)
    return result


def _label(node) -> str:
    return node.label if isinstance(node, Terminal) else str(node)
