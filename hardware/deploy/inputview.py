"""The same frame, segmented from three different pictures of it.

The question this answers is which image a segmenter should be looking at.
Three answers are on disk or derivable from it, and they are genuinely
different inputs rather than three renderings of one:

* **gray** -- the D455's colour stream, converted to grayscale and warped into
  the depth grid.  This is appearance, and it is what SAM3 has been running on.
  It is black wherever the depth dropped, which is 14% of the frame and
  concentrated on the edges an object is made of.
* **shape** -- height above the fitted table plane, from the FoundationStereo
  depth this session was recorded with.  No colour, no texture, nothing that
  moves when the lighting does.  Written by ``shaperender.py``.
* **shape, connected components** -- the same height field, segmented the way
  the deployment segments it today: threshold and label.  It is here because
  otherwise the comparison silently credits the instance model with the
  change from appearance to shape, when half of it may be the change from
  connected components to an instance model.

There is no RGB pane and its absence is the point: no recording in this
repository holds one.  ``run.py`` writes ``gray`` and the colour is converted
away at capture, so a real RGB comparison needs new data, not a new script.
The gray pane is the closest thing that exists -- the same stream, decoloured.

    python -m hardware.deploy.inputview recordings/session
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys

import cv2
import numpy as np

from . import mask, overlay, proprio, segbench


def _draw(canvas, seg_out, label, decimate, tag, site_px):
  img = canvas.copy()
  lab = seg_out.labels
  if decimate > 1:
    lab = np.repeat(np.repeat(lab, decimate, 0), decimate, 1)
  lab = lab[:img.shape[0], :img.shape[1]]
  for inst in seg_out.instances:
    hit = inst.label == label
    colour = (80, 235, 80) if hit else (235, 170, 60)
    cs, _ = cv2.findContours((lab == inst.label).astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, colour, 3 if hit else 1)
  if np.isfinite(site_px).all():
    cv2.drawMarker(img, (int(site_px[0]), int(site_px[1])), (0, 210, 255),
                   cv2.MARKER_CROSS, 16, 2)
  cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                (80, 200, 80) if label else (60, 60, 190), 3)
  cv2.putText(img, tag, (9, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
              (255, 255, 255), 2, cv2.LINE_AA)
  if not label:
    cv2.putText(img, "NO TARGET", (9, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 245), 2, cv2.LINE_AA)
  return img


def _chosen(seg_out, xy, tol=0.006):
  """Which instance this frame's replay chose, matched by where it was.

  The three replays have already been run end to end by ``segbench``; their
  per-frame answers are in the ``.jsonl`` beside this, and the tracker's state
  at frame 1500 depended on the 1499 before it.  Re-running all of that to draw
  a hundred pictures would mean decompressing 2893 mask files three times over
  for nothing.  Instead the recorded centroid identifies the instance, which is
  exact: it is the position that replay reported for that very frame.
  """
  if xy is None:
    return 0
  best, best_d = 0, tol
  for inst in seg_out.instances:
    c = np.asarray(inst.centroid_base, dtype=np.float64)
    if not np.isfinite(c).all():
      continue
    d = float(np.hypot(c[0] - xy[0], c[1] - xy[1]))
    if d < best_d:
      best, best_d = inst.label, d
  return best


def _rows_of(session, name):
  f = segbench.RESULTS / session.name / f"{name}.jsonl"
  if not f.exists():
    raise FileNotFoundError(
      f"{f} is missing -- run `segbench` for {name} before drawing it")
  return {r["i"]: r for r in (json.loads(l) for l in f.read_text().splitlines() if l)}


def build(session, gray_store, shape_store, shape_dir, stride, quality, scale,
          names):
  rig, reproj, meta, files = segbench._session_setup(session)
  kin = proprio.Kinematics()
  depth_seg = mask.DepthSegmenter(rig, reproj)
  rec = [_rows_of(session, n) for n in names]

  wanted = [f for k, f in enumerate(files) if k % stride == 0]
  panes, rows = [[], [], []], []
  with segbench.precomputed_segmenter(rig, reproj, gray_store) as (g_seg, g_det), \
       segbench.precomputed_segmenter(rig, reproj, shape_store) as (s_seg, s_det):
    for f in wanted:
      key = pathlib.Path(f).stem
      i = int(key)
      m = meta.get(i)
      if m is None or any(i not in r for r in rec):
        continue
      q = np.asarray(m["joint_pos"], dtype=np.float64)
      gj = float(q[6]) if q.size > 6 else 0.05
      kin.update(np.array([*q[:6], gj, -gj]))
      arm = kin.link_spheres()
      z = np.load(f)
      depth = z["depth"].astype(np.float32) / 10000.0
      shape_img = cv2.imread(str(shape_dir / f"{key}.jpg"))
      if shape_img is None:
        continue

      g_det.seek(key)
      out_g = g_seg(depth, rgb=z["gray"], arm=arm)
      s_det.seek(key)
      out_s = s_seg(depth, rgb=z["gray"], arm=arm)
      out_d = depth_seg(depth, rgb=z["gray"], arm=arm)
      outs = (out_g, out_s, out_d)
      labs = [_chosen(o, rec[j][i].get("xy")) for j, o in enumerate(outs)]

      gray_img = cv2.cvtColor(z["gray"], cv2.COLOR_GRAY2BGR)
      sp = overlay.project(kin.site_pos[None], rig, gray_img.shape)[0]
      for idx, (canvas, out, lab, dec, tag) in enumerate((
          (gray_img, out_g, labs[0], g_seg.decimate, "gray  -  SAM3"),
          (shape_img, out_s, labs[1], s_seg.decimate, "shape  -  SAM3"),
          (shape_img, out_d, labs[2], depth_seg.decimate,
           "shape  -  connected components"))):
        img = _draw(canvas, out, lab, dec, tag, sp)
        if scale != 1.0:
          img = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img,
                               [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        panes[idx].append(base64.b64encode(buf).decode() if ok else "")

      def tel(j):
        r = rec[j][i]
        return {"n": r["instances"], "px": r.get("n_px_full") or r["n_px"],
                "top": r["top_mm"], "gap": r["gap_mm"]}
      rows.append({"i": i, "a": tel(0), "b": tel(1), "c": tel(2)})
  return panes, rows, session.name


PAGE = """<title>__SESSION__ — which picture to segment</title>
<style>
:root{--ground:#eef1f4;--surface:#fff;--sunk:#e4e9ee;--line:#d3dbe3;--ink:#121a21;
 --muted:#5b6976;--faint:#8b97a3;--accent:#0d7f99;--ok:#2f7d55;--bad:#a8402f;
 --mono:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
 --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 --ground:#0c1015;--surface:#141a21;--sunk:#0a0e12;--line:#232e38;--ink:#dbe4eb;
 --muted:#808f9e;--faint:#5f6d7a;--accent:#4cc0dc;--ok:#4fae76;--bad:#e07565}}
:root[data-theme=dark]{--ground:#0c1015;--surface:#141a21;--sunk:#0a0e12;
 --line:#232e38;--ink:#dbe4eb;--muted:#808f9e;--faint:#5f6d7a;--accent:#4cc0dc;
 --ok:#4fae76;--bad:#e07565}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.55}
.wrap{max-width:1700px;margin:0 auto;padding:clamp(18px,2.5vw,30px) clamp(12px,2vw,20px) 44px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
 color:var(--accent);margin:0 0 9px}
h1{font-family:var(--mono);font-size:clamp(19px,2.3vw,25px);font-weight:600;margin:0 0 12px}
.lede{margin:0 0 8px;color:var(--muted);max-width:80ch;font-size:14.5px}
.lede b{color:var(--ink)}
.lede code{font-family:var(--mono);font-size:12.5px;background:var(--sunk);padding:1px 5px;border-radius:2px;color:var(--ink)}
.missing{border-left:2px solid var(--bad);padding:2px 0 2px 14px;margin:0 0 18px;
 color:var(--muted);font-size:14px;max-width:76ch}
.missing b{color:var(--ink)}
.panes{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:0 0 12px}
@media (max-width:1180px){.panes{grid-template-columns:1fr}}
.pane{background:var(--surface);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.pane img{display:block;width:100%;height:auto;background:var(--sunk)}
.tel{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line)}
.tel>div{padding:8px 10px;border-right:1px solid var(--line)}
.tel>div:last-child{border-right:none}
.t-k{font-family:var(--mono);font-size:9px;letter-spacing:.11em;text-transform:uppercase;
 color:var(--faint);margin:0 0 3px}
.t-v{font-family:var(--mono);font-size:13px;margin:0;font-variant-numeric:tabular-nums}
.t-v.dim{color:var(--faint)} .t-v.bad{color:var(--bad)} .t-v.ok{color:var(--ok)}
.bar{background:var(--surface);border:1px solid var(--line);border-radius:4px;display:flex;
 align-items:center;gap:14px;padding:11px 15px;flex-wrap:wrap;margin:0 0 12px}
button{font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;
 background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:3px;
 padding:6px 13px;cursor:pointer;min-width:74px}
button:hover{border-color:var(--accent);color:var(--accent)}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
input[type=range]{flex:1;min-width:220px;accent-color:var(--accent)}
.fno{font-family:var(--mono);font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.key{display:flex;flex-wrap:wrap;gap:18px;font-family:var(--mono);font-size:11.5px;
 color:var(--muted);margin:0}
.key span{display:flex;align-items:center;gap:7px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.ramp{width:120px;height:10px;border-radius:2px;display:inline-block;
 background:linear-gradient(90deg,#30123b,#4662d8,#36aab8,#a2fc3c,#f9ba38,#c42503)}
</style>
<div class="wrap">
<p class="eyebrow">segmentation input &middot; __SESSION__</p>
<h1>Which picture to segment</h1>
<p class="lede">The same recorded frame, segmented from two different images of
it. <b>gray</b> is the D455's colour stream de-coloured and warped into the depth
grid &mdash; appearance, black wherever the depth dropped. <b>shape</b> is height
above the fitted table plane, from the FoundationStereo depth this session ran on,
shaded as a surface lit from above and left: the same information the depth
backend thresholds, drawn as a picture. Panes one and two are SAM3 on those two
images. Pane three is the same height field segmented the way the deployment does
it today &mdash; threshold and label connected components &mdash; so the
comparison cannot quietly credit the model for the change of input, or the input
for the change of model.</p>
<p class="lede"><b>The result is a clear negative, and it is visible frame by
frame.</b> SAM3 reads the grayscale well (segmenter recall 0.86) and the shaded
height field badly (0.24), while plain connected components on that same height
field reach 0.69. A colour-mapped version of the field was worse still, at 0.07.
Rendering depth into a picture does not make it a photograph, and SAM3 is a
photograph model &mdash; the shape route needs a model trained on shape, not a
zero-shot one pointed at a rendering of it.</p>
<p class="missing"><b>There is no RGB pane, and that is a finding rather than an
omission.</b> No recording in this repository holds colour: all 47 hardware
sessions store <code>depth</code> and a single-channel <code>gray</code>, and
<code>run.py</code> converts the colour away at capture. A real RGB comparison
needs a new recording, not a new script. The gray pane is the same stream with
its colour discarded.</p>
<div class="key">
  <span><i class="dot" style="background:rgb(80,235,80)"></i> chosen target</span>
  <span><i class="dot" style="background:rgb(60,170,235)"></i> other instance</span>
  <span><i class="dot" style="background:rgb(0,210,255)"></i> gripper site</span>
  <span>height <i class="ramp"></i> 0 &rarr; 120 mm, black = no depth</span>
</div>
<div class="panes">
  <div class="pane"><img id="i0" alt="segmented from the grayscale" /><div class="tel" id="t0"></div></div>
  <div class="pane"><img id="i1" alt="segmented from the height field" /><div class="tel" id="t1"></div></div>
  <div class="pane"><img id="i2" alt="height field, connected components" /><div class="tel" id="t2"></div></div>
</div>
<div class="bar">
  <button id="play">Play</button>
  <input type="range" id="sl" min="0" max="__LAST__" value="0" aria-label="frame" />
  <span class="fno" id="fno"></span>
</div>
</div>
<script>
const P=__PANES__, S=__ROWS__;
const im=[0,1,2].map(k=>document.getElementById('i'+k));
const te=[0,1,2].map(k=>document.getElementById('t'+k));
const sl=document.getElementById('sl'),fno=document.getElementById('fno'),
      play=document.getElementById('play');
let i=0,timer=null;
function cells(t){
  const px=t.px==null?'&mdash;':t.px+' px', top=t.top==null?'&mdash;':t.top.toFixed(0)+' mm',
        gap=t.gap==null?'&mdash;':t.gap.toFixed(0)+' mm';
  return `<div><p class="t-k">inst</p><p class="t-v">${t.n}</p></div>`+
   `<div><p class="t-k">target</p><p class="t-v ${t.px==null?'bad':'ok'}">${t.px==null?'lost':px}</p></div>`+
   `<div><p class="t-k">height</p><p class="t-v ${t.top==null?'dim':''}">${top}</p></div>`+
   `<div><p class="t-k">to hand</p><p class="t-v ${t.gap==null?'dim':''}">${gap}</p></div>`;
}
function draw(k){
  if(!P[0].length) return;
  i=Math.max(0,Math.min(k,P[0].length-1));
  for(let j=0;j<3;j++) im[j].src='data:image/jpeg;base64,'+P[j][i];
  const s=S[i]; sl.value=i;
  fno.textContent=`frame ${s.i} \\u00b7 ${i+1} / ${P[0].length}`;
  te[0].innerHTML=cells(s.a); te[1].innerHTML=cells(s.b); te[2].innerHTML=cells(s.c);
}
sl.addEventListener('input',e=>draw(+e.target.value));
play.addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;play.textContent='Play';return;}
 play.textContent='Pause';timer=setInterval(()=>draw((i+1)%P[0].length),220);});
addEventListener('keydown',e=>{if(e.key==='ArrowRight'){draw(i+1);e.preventDefault();}
 if(e.key==='ArrowLeft'){draw(i-1);e.preventDefault();}});
draw(0);
</script>"""


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("session", type=pathlib.Path)
  p.add_argument("--gray-masks", type=pathlib.Path, default=None)
  p.add_argument("--shape-masks", type=pathlib.Path, default=None)
  p.add_argument("--shape-dir", type=pathlib.Path, default=None)
  p.add_argument("--gray-name", default="sam3_dart")
  p.add_argument("--shape-name", default="sam3_relief")
  p.add_argument("--depth-name", default="depth")
  p.add_argument("--stride", type=int, default=28)
  p.add_argument("--quality", type=int, default=60)
  p.add_argument("--scale", type=float, default=0.62)
  p.add_argument("--out", type=pathlib.Path, default=None)
  a = p.parse_args()

  root = segbench.RESULTS / a.session.name
  gray = a.gray_masks or (root / "masks_sam3_dart")
  shape = a.shape_masks or (root / "masks_sam3_relief")
  sdir = a.shape_dir or (root / "relief")
  panes, rows, name = build(a.session, gray, shape, sdir, a.stride,
                            a.quality, a.scale,
                            (a.gray_name, a.shape_name, a.depth_name))
  if not panes[0]:
    print("no frames rendered", file=sys.stderr)
    return 1
  out = a.out or (root / "inputview.html")
  out.parent.mkdir(parents=True, exist_ok=True)
  page = PAGE
  for k, v in (("__SESSION__", name), ("__LAST__", str(len(panes[0]) - 1)),
               ("__PANES__", json.dumps(panes)), ("__ROWS__", json.dumps(rows))):
    page = page.replace(k, v)
  out.write_text(page)
  print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(panes[0])} frames)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
