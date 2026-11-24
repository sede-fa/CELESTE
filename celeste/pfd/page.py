"""The HTML page of a process flow diagram: SVG plus a few lines of JavaScript for the hover tables."""

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
<header><h1>{title}</h1><div class="meta">{n_units} units, {n_streams} streams; {n_tears} back stream(s) closing a loop (recycle or heat integration) dashed.
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
