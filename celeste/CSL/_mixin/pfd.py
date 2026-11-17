"""
Process flow diagram of a system: layered left-to-right layout with orthogonal streams, drawn as one
self-contained HTML file (SVG and a few lines of JavaScript, no downloads). Hovering a stream shows its
state and composition, hovering a unit its duties and exergy loss; a click pins the table.

    pfd = plant.pfd()          # after run_sol(), or for a flowsheet declared with link()
    pfd.save('plant.html')     # in Jupyter the object displays itself

Layout (the "layered" method of Sugiyama et al., as in Graphviz dot and ELK layered): recycle streams are torn
(fewest tears, back streams preferred), every unit is placed in the layer of its longest path from the feeds,
the order within each layer is improved by barycentre sweeps, and the streams are routed orthogonally - forward
streams through the gap before their destination, recycles along lanes below the units.
"""
import html as _html
import json
import math

import networkx as nx
import numpy as np

_CATEGORIES = [   # module or class name fragment -> (category, legend label)
    ('electrochem', 'electrochem', 'electrochemical'),
    ('reactors', 'reaction', 'reactor'),
    ('turbomachinery', 'pressure', 'pressure change'),
    ('heat', 'heat', 'heat exchange'),
    ('separation', 'separation', 'separation'),
    ('GPU', 'separation', 'separation'),
    ('Dist', 'separation', 'separation'),
    ('Comp', 'pressure', 'pressure change'),
    ('Pump', 'pressure', 'pressure change'),
    ('Valve', 'pressure', 'pressure change'),
    ('Source', 'terminal', 'source / sink'),
    ('Sink', 'terminal', 'source / sink'),
    ('flow', 'flow', 'mixer / splitter'),
]

_NODE_H, _GAP_X, _GAP_Y, _MARGIN, _LANE = 46.0, 96.0, 34.0, 28.0, 14.0


def _category(unit) -> str:
    if getattr(unit, '_is_system', False):
        return 'system'
    names = [type(unit).__module__, type(unit).__name__] + [c.__name__ for c in type(unit).__mro__]
    for key, cat, _ in _CATEGORIES:
        if any(key in n for n in names):
            return cat
    return 'other'


def _num(value, digits=5):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return f'{v:.{digits}g}'


def _table(title, subtitle, rows, composition=None) -> str:
    out = [f'<div class="tt-title">{_html.escape(title)}</div>']
    if subtitle:
        out.append(f'<div class="tt-sub">{_html.escape(subtitle)}</div>')
    body = ''.join(f'<tr><th>{_html.escape(k)}</th><td>{v}</td></tr>' for k, v in rows if v is not None)
    out.append(f'<table>{body}</table>')
    if composition:
        comp = ''.join(f'<tr><th>{_html.escape(sp)}</th><td>{x}</td></tr>' for sp, x in composition)
        out.append(f'<div class="tt-sub">mole fractions</div><table>{comp}</table>')
    return ''.join(out)


def _stream_table(outlet, inlet) -> str:
    f = outlet
    rows = [('n  mol/s', _num(getattr(f, 'n', None))), ('m  g/s', _num(_safe(f, 'm'))),
            ('T  K', _num(getattr(f, 'T', None))), ('P  bar', _num(getattr(f, 'P', None))),
            ('h  kJ/mol', _num(_safe(f, 'h'))), ('Ex  kJ/mol', _num(_safe(f, 'Ex')))]
    phase = getattr(f, 'phase', None)
    if isinstance(phase, str):
        rows.append(('phase', _html.escape(phase)))
    comp = []
    try:
        Xi = np.asarray(f.Xi, dtype=float)
        species = [str(s) for s in f.Medium.species]
        comp = [(sp, f'{x:.4g}') for x, sp in sorted(zip(Xi, species), reverse=True) if x > 1e-6]
    except Exception:
        pass
    return _table(f'{_short(outlet.name)} → {_short(inlet.name)}', type(f.Medium).__name__, rows, comp)


def _short(name: str) -> str:
    """'plant.mix.F2' -> 'mix.F2' (drop the system prefix, and the decorations of sources and sinks)."""
    parts = name.replace('((', '.').replace('[(', '.').split('.')
    parts = [p.strip('()[]') for p in parts if p.strip('()[]')]
    return '.'.join(parts[-2:]) if len(parts) > 1 else name


def _safe(obj, attr):
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def _unit_table(label, unit) -> str:
    rows = [('Q  kW', _num(_safe(unit, 'Q'))), ('W  kW', _num(_safe(unit, 'W'))),
            ('Ex loss  kW', _num(_safe(unit, 'Exloss')))]
    sol = unit.__dict__.get('_sol')
    if sol is not None and hasattr(sol, 'success'):
        rows.append(('converged', 'yes' if sol.success else '<b class="bad">no</b>'))
    return _table(label, type(unit).__name__, rows)


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


class PFDMixin:
    """``pfd()``: interactive process flow diagram of the system."""

    def _drawn_streams(self) -> list:
        if self.streams:
            return self.streams
        links = self._run_streams()
        if not links:
            raise ValueError(f"{self.name}: nothing to draw - run the system once, or declare its streams with link()")
        return links

    def pfd(self, path: str | None = None) -> PFD:
        """
        Draw the flowsheet (declared streams, or the connections made by the last run). ``path`` also writes
        the HTML file. Returns a :class:`PFD`.
        """
        streams = self._drawn_streams()
        owners = self._owner_map()
        labels = {id(v): k for k, v in self.__dict__.items() if hasattr(v, 'get_Flanges') and v is not self}
        units = [u for u in self.getObjs()]
        G = nx.MultiDiGraph()
        G.add_nodes_from(units)
        terminals = {}
        for outlet, inlet in streams:
            src, dst = owners.get(id(outlet)), owners.get(id(inlet))
            if src is None or dst is None:
                continue
            if src is self:                                     # feed of the system
                src = terminals.setdefault(id(outlet), ('in', outlet))
            if dst is self:                                     # product of the system
                dst = terminals.setdefault(id(inlet), ('out', inlet))
            G.add_edge(src, dst, flanges=(outlet, inlet))

        # 1. tear the recycles; 2. layers by longest path; 3. order within layers
        index = {u: i for i, u in enumerate(G.nodes)}
        starts = [u for u in G if G.in_degree(u) == 0]
        depth = nx.multi_source_dijkstra_path_length(G, starts, weight=lambda u, v, d: 1) if starts else {}
        order = {u: (depth.get(u, len(G)), index[u]) for u in G}
        tears = set()
        for comp in nx.strongly_connected_components(G):
            sub = G.subgraph(comp).copy()
            if len(comp) > 1 or sub.number_of_edges():
                tears.update(self._tear_edges(sub, order))
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
        def label(u):
            if isinstance(u, tuple):
                return _short(u[1].name)
            return labels.get(id(u), _short(u.name))

        def width(u):
            return 22.0 if isinstance(u, tuple) else max(84.0, 7.4*len(label(u)) + 26)

        n_lanes = len(tears)
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
                h = 22.0 if isinstance(u, tuple) else _NODE_H
                cx = X[r] + col_w[r]/2
                cy = _MARGIN + 30 + offset + i*(_NODE_H + _GAP_Y) + _NODE_H/2
                box[u] = (cx - w/2, cy - h/2, w, h)
        bottom = max(b[1] + b[3] for b in box.values())
        W = x - _GAP_X + _MARGIN
        Hgt = bottom + _MARGIN + (n_lanes + 1)*_LANE + 10

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
        gap_use = {}
        paths, tips = [], {}
        lane = 0
        for e in G.edges(keys=True):
            u, v, _ = e
            (x1, y1), (x2, y2) = port(u, e, 'out'), port(v, e, 'in')
            outlet, inlet = G.edges[e]['flanges']
            back = e in tears or rank[v] <= rank[u]
            if not back:
                gap = rank[v]
                k = gap_use.get(gap, 0)
                gap_use[gap] = k + 1
                xm = X[gap] - 14 - (k % 6)*9                     # in the gap before the destination column
                d = f'M{x1:.1f},{y1:.1f} H{xm:.1f} V{y2:.1f} H{x2:.1f}'
            else:
                yl = bottom + _LANE*(lane + 1)
                lane += 1
                xa, xb = x1 + 10 + lane*8, X[rank[v]] - 10 - lane*8
                d = f'M{x1:.1f},{y1:.1f} H{xa:.1f} V{yl:.1f} H{xb:.1f} V{y2:.1f} H{x2:.1f}'
            sid = f's{len(paths)}'
            tips[sid] = _stream_table(outlet, inlet)
            paths.append((sid, d, back))

        # 7. SVG
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {Hgt:.0f}" width="{W:.0f}" '
                 f'height="{Hgt:.0f}" role="img" aria-label="Process flow diagram of {_html.escape(self.name)}">',
                 '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                 'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>']
        for sid, d, back in paths:
            parts.append(f'<g class="stream{" recycle" if back else ""}" data-tip="{sid}">'
                         f'<path class="hit" d="{d}"/><path class="line" d="{d}" marker-end="url(#arrow)"/></g>')
        for u, (bx, by, bw, bh) in box.items():
            uid = f'u{len(tips)}'
            if isinstance(u, tuple):
                kind, f = u
                tips[uid] = _table(_short(f.name), 'system ' + ('feed' if kind == 'in' else 'product'), [])
                parts.append(f'<g class="terminal" data-tip="{uid}"><circle cx="{bx + bw/2:.1f}" cy="{by + bh/2:.1f}" '
                             f'r="{bw/2:.1f}"/><text x="{bx + bw/2:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
                             f'class="small">{_html.escape(label(u))}</text></g>')
                continue
            cat = _category(u)
            tips[uid] = _unit_table(label(u), u)
            parts.append(f'<g class="unit {cat}" data-tip="{uid}"><rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                         f'height="{bh:.1f}" rx="7"/><text x="{bx + bw/2:.1f}" y="{by + 19:.1f}" text-anchor="middle">'
                         f'{_html.escape(label(u))}</text><text x="{bx + bw/2:.1f}" y="{by + 35:.1f}" '
                         f'text-anchor="middle" class="small">{_html.escape(type(u).__name__)}</text></g>')
        parts.append('</svg>')
        used = sorted({_category(u) for u in G if not isinstance(u, tuple)})
        names = {cat: text for _, cat, text in _CATEGORIES}
        names.update(system='subsystem', other='other')
        legend = ''.join(f'<span class="key unit {c}"><i></i>{names[c]}</span>' for c in used)
        page = _PAGE.format(title=_html.escape(self.name), legend=legend, svg=''.join(parts),
                            tips=json.dumps(tips).replace('</', '<\\/'), n_units=len(units), n_streams=len(paths),
                            n_tears=len(tears))
        result = PFD(page, W, Hgt, self.name)
        if path:
            result.save(path)
        return result


_PAGE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} flowsheet</title>
<style>
:root {{ --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --line:#4b5563; --recycle:#b45309; --card:#ffffff;
  --border:#d0d7de; --heat:#fde7cf; --reaction:#f6d6dd; --electrochem:#d5f0dc; --separation:#d6e6fb;
  --pressure:#d3efef; --flow:#eceef1; --terminal:#c9ced6; --system:#efe6fb; --other:#f2f2f2; --hl:#2563eb; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1;
  --line:#9ca3af; --recycle:#f59e0b; --card:#161b22; --border:#3d444d; --heat:#4a3420; --reaction:#4a2530;
  --electrochem:#1f3d29; --separation:#1f3350; --pressure:#1c3c3d; --flow:#2a2f36; --terminal:#3d444d;
  --system:#33284a; --other:#2a2f36; --hl:#60a5fa; }} }}
html, body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }}
header {{ padding:14px 16px 4px; }} h1 {{ font-size:17px; margin:0 0 4px; }}
.meta {{ color:var(--muted); font-size:12.5px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 14px; padding:6px 16px 0; font-size:12.5px; color:var(--muted); }}
.key i {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:5px; vertical-align:-1px;
  border:1px solid var(--border); background:var(--c); }}
.wrap {{ overflow:auto; padding:8px 16px 16px; }}
svg {{ max-width:none; display:block; }}
svg text {{ fill:var(--fg); font-size:13px; pointer-events:none; }} svg text.small {{ fill:var(--muted); font-size:11px; }}
.unit {{ --c:var(--other); }} .unit.heat {{ --c:var(--heat); }} .unit.reaction {{ --c:var(--reaction); }}
.unit.electrochem {{ --c:var(--electrochem); }} .unit.separation {{ --c:var(--separation); }}
.unit.pressure {{ --c:var(--pressure); }} .unit.flow {{ --c:var(--flow); }} .unit.terminal {{ --c:var(--terminal); }}
.unit.system {{ --c:var(--system); }}
g.unit rect {{ fill:var(--c); stroke:var(--border); stroke-width:1.2; cursor:pointer; }}
g.unit.system rect {{ stroke-dasharray:5 3; }}
g.unit:hover rect, g.unit.pinned rect {{ stroke:var(--hl); stroke-width:2; }}
g.terminal circle {{ fill:var(--card); stroke:var(--line); stroke-width:1.5; cursor:pointer; }}
g.stream .line {{ fill:none; stroke:var(--line); stroke-width:1.6; }}
g.stream.recycle .line {{ stroke:var(--recycle); stroke-dasharray:6 4; }}
g.stream .hit {{ fill:none; stroke:transparent; stroke-width:12; cursor:pointer; }}
g.stream:hover .line, g.stream.pinned .line {{ stroke:var(--hl); stroke-width:2.6; }}
.arrowhead {{ fill:var(--line); }}
#tip {{ position:fixed; z-index:10; display:none; background:var(--card); color:var(--fg); border:1px solid var(--border);
  border-radius:8px; padding:8px 10px; box-shadow:0 6px 24px rgba(0,0,0,.18); font-size:12.5px; max-width:300px; }}
#tip table {{ border-collapse:collapse; margin-top:2px; }} #tip th {{ text-align:left; font-weight:500; color:var(--muted);
  padding:1px 12px 1px 0; }} #tip td {{ text-align:right; font-variant-numeric:tabular-nums; }}
.tt-title {{ font-weight:600; }} .tt-sub {{ color:var(--muted); font-size:11.5px; margin-top:4px; }}
.bad {{ color:#dc2626; }}
</style></head>
<body>
<header><h1>{title}</h1><div class="meta">{n_units} units, {n_streams} streams, {n_tears} recycle(s) dashed.
Hover a unit or a stream for its data; click to pin.</div></header>
<div class="legend">{legend}</div>
<div class="wrap">{svg}</div>
<div id="tip" role="tooltip"></div>
<script>
const TIPS = {tips};
const tip = document.getElementById('tip');
let pinned = null;
function place(ev) {{
  const pad = 14, r = tip.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = Math.max(8, innerHeight - r.height - 8);
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}}
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mouseenter', ev => {{ if (pinned) return; tip.innerHTML = TIPS[el.dataset.tip];
    tip.style.display = 'block'; place(ev); }});
  el.addEventListener('mousemove', ev => {{ if (!pinned) place(ev); }});
  el.addEventListener('mouseleave', () => {{ if (!pinned) tip.style.display = 'none'; }});
  el.addEventListener('click', ev => {{ ev.stopPropagation();
    if (pinned) pinned.classList.remove('pinned');
    if (pinned === el) {{ pinned = null; tip.style.display = 'none'; return; }}
    pinned = el; el.classList.add('pinned'); tip.innerHTML = TIPS[el.dataset.tip]; tip.style.display = 'block'; place(ev); }});
}});
document.addEventListener('click', () => {{ if (pinned) {{ pinned.classList.remove('pinned'); pinned = null; tip.style.display = 'none'; }} }});
</script>
</body></html>
'''
