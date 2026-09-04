"""One HTML page, one row per frame, for the perception check.

Every aggregate this project has produced has hidden the structure that
mattered.  A 4% detection rate near the gripper was invisible inside a 71%
episode mean; a placement rate that decays threefold within an episode is a
single healthy-looking number when averaged.  So the rule here is the same one
that has already caught three of these: the page shows every frame, and the
summary is a header on top of the frames rather than a substitute for them.

Three outlines per frame, and only outlines -- filled overlays hide the pixels
the page exists to compare:

    green   what the renderer says the target is
    orange  what the depth segmenter and TargetTracker produced
    blue    what SAM2.1 produced
"""

from __future__ import annotations

import json
import pathlib

CSS = """
:root{
  --bg:#faf9f7; --panel:#ffffff; --ink:#1b1a17; --dim:#6c6862;
  --line:#e2ded7; --truth:#2fae4a; --depth:#ff8c28; --sam:#3c78ff;
  --bad:#d4351c; --ok:#2fae4a;
}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#151412; --panel:#1e1d1a; --ink:#f0ede7; --dim:#a39e95;
    --line:#33302b;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:18px 22px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0 0 4px;font-size:17px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:12.5px}
.wrap{max-width:1500px;margin:0 auto;padding:18px 22px 60px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12.5px}
th,td{padding:4px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:550}
.cards{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0 22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:12px 16px;overflow-x:auto}
.card h2{margin:0 0 8px;font-size:12px;text-transform:uppercase;
  letter-spacing:.07em;color:var(--dim);font-weight:600}
.strip{position:relative;height:74px;background:var(--panel);
  border:1px solid var(--line);border-radius:8px;margin:10px 0;cursor:crosshair}
.strip canvas{display:block;width:100%;height:100%;border-radius:8px}
.cursor{position:absolute;top:0;bottom:0;width:1px;background:var(--ink);
  pointer-events:none}
.viewer{display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap}
.viewer img{image-rendering:auto;border-radius:6px;border:1px solid var(--line);
  width:min(700px,100%);background:#000}
.viewer img.zoom{width:330px;flex:none;image-rendering:pixelated}
.meta{min-width:270px;flex:1}
.k{color:var(--dim)}
.legend span{display:inline-flex;align-items:center;gap:6px;margin-right:14px}
.sw{width:14px;height:3px;border-radius:2px;display:inline-block}
.ctl{display:flex;gap:8px;align-items:center;margin:12px 0}
button{font:inherit;padding:5px 12px;border-radius:6px;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--dim)}
input[type=range]{flex:1}
.bad{color:var(--bad);font-weight:600}
.ok{color:var(--ok)}
.note{color:var(--dim);font-size:12.5px;max-width:70ch;margin:10px 0 0}
"""

JS = """
const F = DATA.frames, S = DATA.summary, V = DATA.variants;
const COL = {depth:'#ff8c28', sam:'#3c78ff', deploy:'#c05a00', 'deploy+sam':'#1f4fd0'};
let i = 0, playing = null;

function drawStrip(){
  const c = document.getElementById('strip'), g = c.getContext('2d');
  const w = c.width = c.clientWidth * devicePixelRatio;
  const h = c.height = c.clientHeight * devicePixelRatio;
  g.clearRect(0,0,w,h);
  const n = F.length, bw = w / n, lanes = V.length;
  const lh = h / (lanes + 1);
  F.forEach((f, k) => {
    // phase band on top
    g.fillStyle = f.phase === 'holding' ? 'rgba(120,120,120,.45)' : 'rgba(120,120,120,.13)';
    g.fillRect(k*bw, 0, Math.max(bw,1), lh*0.55);
    V.forEach((v, j) => {
      const d = f[v];
      const y = lh*(j+1);
      g.fillStyle = !d.det ? 'rgba(212,53,28,.85)'
                  : d.iou < 0.10 ? 'rgba(212,53,28,.45)'
                  : `rgba(47,174,74,${0.25 + 0.75*Math.min(1,d.iou/0.6)})`;
      g.fillRect(k*bw, y, Math.max(bw,1), lh*0.8);
    });
  });
  g.font = `${11*devicePixelRatio}px ui-sans-serif`;
  g.fillStyle = getComputedStyle(document.body).getPropertyValue('--dim');
  g.fillText('phase', 4*devicePixelRatio, lh*0.45);
  V.forEach((v,j)=> g.fillText(v, 4*devicePixelRatio, lh*(j+1)+lh*0.6));
}

function show(k){
  i = Math.max(0, Math.min(F.length-1, k));
  const f = F[i];
  document.getElementById('img').src = 'data:image/jpeg;base64,' + f.img;
  document.getElementById('zoom').src = 'data:image/jpeg;base64,' + (f.zoom || f.img);
  document.getElementById('rng').value = i;
  document.querySelector('.cursor').style.left = (100*i/(F.length-1)) + '%';
  const rows = V.map(v => {
    const d = f[v];
    const cls = !d.det ? 'bad' : (d.iou >= 0.3 ? 'ok' : '');
    return `<tr><td>${v}</td><td class="${cls}">${d.det ? 'yes' : 'EMPTY'}</td>
            <td>${d.iou.toFixed(3)}</td><td>${d.px}</td></tr>`;
  }).join('');
  document.getElementById('meta').innerHTML = `
    <table><tr><th>frame</th><td>${f.step}</td></tr>
    <tr><th>phase</th><td>${f.phase}</td></tr>
    <tr><th>gripper&nbsp;&rarr;&nbsp;object</th><td>${f.grip_dist_mm.toFixed(0)} mm</td></tr>
    <tr><th>tracker label</th><td>${f.label || '&mdash;'}</td></tr>
    <tr><th>truth px (policy grid)</th><td>${f.truth_px}</td></tr>
    <tr><th>sam state</th><td>${f.sam_state}${f.sam_reason ? ' &mdash; ' + f.sam_reason : ''}</td></tr>
    </table>
    <table style="margin-top:12px"><tr><th>variant</th><th>mask</th><th>IoU</th><th>px</th></tr>
    ${rows}</table>`;
}

addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') { show(i+1); e.preventDefault(); }
  if (e.key === 'ArrowLeft')  { show(i-1); e.preventDefault(); }
  if (e.key === ' ') { toggle(); e.preventDefault(); }
});
function toggle(){
  if (playing) { clearInterval(playing); playing = null; }
  else playing = setInterval(() => show(i+1 >= F.length ? 0 : i+1), 60);
  document.getElementById('play').textContent = playing ? 'pause' : 'play';
}
addEventListener('resize', drawStrip);
"""


def render(path, rows, summary, args) -> None:
  """Write the page.  ``rows`` carry base64 JPEGs already."""
  variants = [v for v in ("depth", "deploy", "sam_raw", "sam", "deploy+sam")
              if any(v in r for r in rows)]
  data = {"frames": rows, "summary": summary, "variants": variants}
  head = []
  for title, tbl in (summary.get("by_phase") or {}).items():
    body = "".join(
      f"<tr><td>{v}</td><td>{100*d['detected']:.1f}%</td>"
      f"<td>{d['iou_mean']:.3f}</td><td>{d['iou_median']:.3f}</td>"
      f"<td>{100*d['wrong_target']:.1f}%</td></tr>"
      for v, d in tbl.items())
    n = next(iter(tbl.values()))["n"] if tbl else 0
    head.append(
      f"<div class='card'><h2>{title} &middot; {n} frames</h2><table>"
      f"<tr><th>variant</th><th>detected</th><th>IoU mean</th>"
      f"<th>IoU p50</th><th>wrong target</th></tr>{body}</table></div>")
  dist = summary.get("by_distance") or {}
  if dist:
    cols = "".join(f"<th>{v}</th>" for v in variants)
    body = "".join(
      f"<tr><td>{k} mm</td>"
      + "".join(f"<td>{100*d.get(v, float('nan')):.0f}%</td>" for v in variants)
      + "</tr>" for k, d in dist.items())
    head.append(f"<div class='card'><h2>detected, by gripper&ndash;object "
                f"distance</h2><table><tr><th>range</th>{cols}</tr>"
                f"{body}</table></div>")

  html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>perception check &middot; {pathlib.Path(args.get('policy') or 'no-policy').name}</title>
<style>{CSS}</style></head><body>
<header>
  <h1>Deployment perception, scored against the renderer</h1>
  <div class="sub">{args.get('steps')} steps &middot; seed {args.get('seed')}
    &middot; policy <code>{args.get('policy')}</code>
    &middot; SAM2.1 {'on' if args.get('sam') else 'off'}
    &middot; lifecycle {'on' if args.get('lifecycle') else 'off'}</div>
</header>
<div class="wrap">
  <div class="cards">{''.join(head)}</div>
  <p class="legend">
    <span><i class="sw" style="background:var(--truth)"></i>renderer truth</span>
    <span><i class="sw" style="background:var(--depth)"></i>depth segmenter</span>
    <span><i class="sw" style="background:var(--sam)"></i>SAM2.1</span>
  </p>
  <div class="strip"><canvas id="strip"></canvas><div class="cursor"></div></div>
  <p class="note">Each lane is one variant, one column per frame: red = no mask
  at all, faded red = a mask that is not the target, green = correct, brighter
  with IoU. The top band is dark while the object is held.</p>
  <div class="ctl">
    <button onclick="show(i-1)">&larr;</button>
    <button id="play" onclick="toggle()">play</button>
    <button onclick="show(i+1)">&rarr;</button>
    <input id="rng" type="range" min="0" max="{max(len(rows)-1,0)}" value="0"
           oninput="show(+this.value)">
  </div>
  <div class="viewer">
    <img id="img" alt="frame">
    <img id="zoom" class="zoom" alt="target, 3x">
    <div class="meta" id="meta"></div>
  </div>
</div>
<script>const DATA = {json.dumps(data)};</script>
<script>{JS}</script>
<script>drawStrip(); show(0);
document.querySelector('.strip').addEventListener('click', e => {{
  const r = e.currentTarget.getBoundingClientRect();
  show(Math.round((e.clientX - r.left) / r.width * (F.length - 1)));
}});</script>
</body></html>"""
  pathlib.Path(path).write_text(html)
