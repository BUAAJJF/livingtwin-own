"""Two segmentation backends over the same recorded frames, drawn side by side.

``segbench.py`` produces the numbers.  This produces the picture, because the
numbers cannot settle the question on their own: there is no ground truth for a
recorded session, so "found a target in 40% of frames" is only worth something
once somebody has looked at what was found.  Every frame here is the same frame
twice, with the same task geometry drawn on it, segmented two ways.

Both backends are replayed over the *whole* session even when only a window is
drawn.  ``mask.TargetTracker`` confirms an instance across consecutive frames
and holds a target until it is gone, so its state at frame 1500 depends on
frames 0-1499; rendering a window without replaying up to it would show a
tracker that had just started.

    python -m hardware.deploy.segreport recordings/session \
        --masks results/segbench/session/masks_sam3 --name sam3_dart
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import glob
import json
import os
import pathlib
import sys

import cv2
import numpy as np

from . import config, mask, overlay, proprio, rectify, segbench


def _panel(gray: np.ndarray, rig, seg, label, decimate, caption, site):
  img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
  overlay.annotate(img, rig, seg, label, decimate, caption)
  p = overlay.project(np.asarray(site)[None], rig, img.shape)[0]
  if np.isfinite(p).all():
    cv2.drawMarker(img, (int(p[0]), int(p[1])), (0, 190, 255),
                   cv2.MARKER_CROSS, 18, 2)
  return img


def build(session: pathlib.Path, store: pathlib.Path, name: str,
          stride: int, window, quality: int, scale: float) -> dict:
  rig, reproj, meta, files = segbench._session_setup(session)
  kin = proprio.Kinematics()
  depth_seg = mask.DepthSegmenter(rig, reproj)
  depth_trk = mask.TargetTracker()
  other_trk = mask.TargetTracker()

  stamps = [float(v["frame_stamp"]) for v in meta.values()
            if v.get("frame_stamp")]
  t_zero = min(stamps) if stamps else 0.0

  frames, others, rows = [], [], []
  with segbench.precomputed_segmenter(rig, reproj, store) as (other_seg, det):
    for k, f in enumerate(files):
      key = os.path.basename(f).split(".")[0]
      m = meta.get(int(key))
      if m is None:
        continue
      q = np.asarray(m["joint_pos"], dtype=np.float64)
      g = float(q[6]) if q.size > 6 else 0.05
      kin.update(np.array([*q[:6], g, -g]))
      arm = kin.link_spheres()
      z = np.load(f)
      d = z["depth"].astype(np.float32) / 10000.0

      a_out = depth_seg(d, rgb=z["gray"], arm=arm)
      a_lab = depth_trk.update(a_out, kin.site_pos)
      det.seek(key)
      b_out = other_seg(d, rgb=z["gray"], arm=arm)
      b_lab = other_trk.update(b_out, kin.site_pos)

      t_s = ((float(m["frame_stamp"]) - t_zero)
             if m.get("frame_stamp") else float(k) / 30.0)
      if window is not None:
        if not (window[0] <= t_s <= window[1]):
          continue
      elif k % stride:
        continue

      a_hit = next((i for i in a_out.instances if i.label == a_lab), None)
      b_hit = next((i for i in b_out.instances if i.label == b_lab), None)
      # Two images, not one composite.  A single wide strip has to shrink to
      # the narrower of the two panes on any screen that cannot hold 1700
      # pixels, and the objects here are 25 px across -- half of that is
      # nothing.  Separate panes let the layout stack them instead.
      panes = []
      for out, lab, dec, tag in ((a_out, a_lab, depth_seg.decimate, "depth"),
                                 (b_out, b_lab, other_seg.decimate, name)):
        img = _panel(z["gray"], rig, out, lab, dec, tag, kin.site_pos)
        if scale != 1.0:
          img = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img,
                               [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
          break
        panes.append(base64.b64encode(buf).decode())
      if len(panes) != 2:
        continue
      frames.append(panes[0])
      others.append(panes[1])

      def tel(out, hit):
        return {
          "n": len(out.instances),
          "px": None if hit is None else int(hit.n_px),
          "top": (None if hit is None or not np.isfinite(hit.top_z)
                  else round(float(hit.top_z) * 1000, 1)),
          "gap": (None if hit is None else round(float(np.linalg.norm(
            kin.site_pos - np.asarray(hit.centroid_base))) * 1000, 1)),
        }
      # Where a zoom should centre.  Both panes are the same underlying frame,
      # so they share one origin or the two views stop showing the same thing.
      # The chosen target when there is one -- it is what the reader is trying
      # to see -- and the hand when there is not.
      focus = kin.site_pos
      for h in (a_hit, b_hit):
        if h is not None and np.isfinite(h.centroid_base).all():
          focus = np.asarray(h.centroid_base)
          break
      fp = overlay.project(np.asarray(focus)[None], rig, (480, 848, 3))[0]
      zx, zy = ((float(np.clip(fp[0] / 848.0, 0, 1)),
                 float(np.clip(fp[1] / 480.0, 0, 1)))
                if np.isfinite(fp).all() else (0.5, 0.5))
      rows.append({"i": int(m["i"]), "t": round(t_s, 2),
                   "a": tel(a_out, a_hit), "b": tel(b_out, b_hit),
                   "zx": round(zx, 4), "zy": round(zy, 4)})
  return {"left": frames, "right": others, "rows": rows, "name": name,
          "session": session.name}


PAGE = """<title>__SESSION__ — depth vs __NAME__</title>
<style>
:root{
  --ground:#eef1f4; --surface:#ffffff; --sunk:#e4e9ee; --line:#d3dbe3;
  --ink:#121a21; --muted:#5b6976; --faint:#8b97a3;
  --accent:#0d7f99; --ok:#2f7d55; --bad:#a8402f;
  --mono:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --ground:#0c1015; --surface:#141a21; --sunk:#0a0e12; --line:#232e38;
  --ink:#dbe4eb; --muted:#808f9e; --faint:#5f6d7a;
  --accent:#4cc0dc; --ok:#4fae76; --bad:#e07565;
}}
:root[data-theme=dark]{
  --ground:#0c1015; --surface:#141a21; --sunk:#0a0e12; --line:#232e38;
  --ink:#dbe4eb; --muted:#808f9e; --faint:#5f6d7a;
  --accent:#4cc0dc; --ok:#4fae76; --bad:#e07565;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:var(--sans);line-height:1.55}
.wrap{max-width:1560px;margin:0 auto;padding:clamp(20px,3vw,34px) clamp(14px,3vw,24px) 48px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--accent);margin:0 0 9px}
h1{font-family:var(--mono);font-size:clamp(19px,2.4vw,24px);font-weight:600;margin:0 0 12px;
  letter-spacing:-.015em}
.lede{margin:0 0 20px;color:var(--muted);max-width:74ch;font-size:14.5px}
.lede code{font-family:var(--mono);font-size:12.5px;background:var(--sunk);
  padding:1px 5px;border-radius:2px;color:var(--ink)}

.panes{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 12px}
@media (max-width:900px){.panes{grid-template-columns:1fr}}
.pane{background:var(--surface);border:1px solid var(--line);border-radius:4px;
  overflow:hidden;display:flex;flex-direction:column}
.pane.win{border-color:var(--accent)}
.pane-hd{display:flex;align-items:baseline;gap:10px;padding:10px 14px;
  border-bottom:1px solid var(--line)}
.pane-nm{font-family:var(--mono);font-size:13px;font-weight:600;letter-spacing:.02em}
.pane-tag{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--faint);margin-left:auto}
.view{position:relative;overflow:hidden;background:var(--sunk);line-height:0}
.view img{display:block;width:100%;height:auto;transition:transform .18s ease}
.pane-tel{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);
  margin-top:auto}
.pane-tel>div{padding:9px 12px;border-right:1px solid var(--line)}
.pane-tel>div:last-child{border-right:none}
.t-k{font-family:var(--mono);font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--faint);margin:0 0 4px}
.t-v{font-family:var(--mono);font-size:14px;margin:0;font-variant-numeric:tabular-nums}
.t-v.bad{color:var(--bad)} .t-v.ok{color:var(--ok)} .t-v.dim{color:var(--faint)}

.bar{background:var(--surface);border:1px solid var(--line);border-radius:4px;
  display:flex;align-items:center;gap:14px;padding:11px 15px;flex-wrap:wrap;margin:0 0 14px}
button{font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;
  background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:3px;
  padding:6px 13px;cursor:pointer;min-width:70px}
button:hover{border-color:var(--accent);color:var(--accent)}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button[aria-pressed=true]{border-color:var(--accent);color:var(--accent);background:var(--sunk)}
input[type=range]{flex:1;min-width:220px;accent-color:var(--accent)}
.fno{font-family:var(--mono);font-size:12px;color:var(--muted);
  font-variant-numeric:tabular-nums;white-space:nowrap}
.key{display:flex;flex-wrap:wrap;gap:18px;font-family:var(--mono);font-size:11.5px;
  color:var(--muted);margin:0}
.key span{display:flex;align-items:center;gap:7px}
.sw{width:14px;height:3px;border-radius:2px;display:inline-block}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
@media (prefers-reduced-motion:reduce){.view img{transition:none}}
</style>

<div class="wrap">
<p class="eyebrow">segmentation comparison &middot; __SESSION__</p>
<h1>depth vs __NAME__</h1>
<p class="lede">The same recorded frame, segmented two ways. Left is the depth
backend the run actually used; right is __NAME__. Both were replayed over every
frame in order and both pass through the same workspace, arm, bin, height,
footprint and elongation filters and the same three-frames-in-five confirmation
&mdash; only the instance masks came from somewhere different. The pane with a
confirmed target is outlined. <code>&larr;</code> <code>&rarr;</code> step frames.</p>

<div class="panes">
  <div class="pane" id="paneA">
    <div class="pane-hd"><span class="pane-nm">depth</span>
      <span class="pane-tag">DepthSegmenter</span></div>
    <div class="view"><img id="imgA" alt="frame segmented by the depth backend" /></div>
    <div class="pane-tel" id="telA"></div>
  </div>
  <div class="pane" id="paneB">
    <div class="pane-hd"><span class="pane-nm">__NAME__</span>
      <span class="pane-tag">__TAG__</span></div>
    <div class="view"><img id="imgB" alt="frame segmented by __NAME__" /></div>
    <div class="pane-tel" id="telB"></div>
  </div>
</div>

<div class="bar">
  <button id="play">Play</button>
  <button id="zoom" aria-pressed="false">Zoom 1&times;</button>
  <input type="range" id="sl" min="0" max="__LAST__" value="0" aria-label="frame" />
  <span class="fno" id="fno"></span>
</div>

<div class="key">
  <span><i class="dot" style="background:rgb(80,235,80)"></i> chosen target</span>
  <span><i class="sw" style="background:rgb(235,80,80)"></i> other instance</span>
  <span><i class="dot" style="background:rgb(0,190,255)"></i> gripper</span>
  <span><i class="sw" style="background:rgb(60,165,215)"></i> accepted sector</span>
  <span><i class="sw" style="background:rgb(90,220,90)"></i> spawn wedge</span>
  <span><i class="sw" style="background:rgb(150,150,150)"></i> bin</span>
</div>
</div>

<script>
const L=__LEFT__, R=__RIGHT__, S=__ROWS__, NAME=__NAMEJSON__;
const imgA=document.getElementById('imgA'), imgB=document.getElementById('imgB'),
      telA=document.getElementById('telA'), telB=document.getElementById('telB'),
      paneA=document.getElementById('paneA'), paneB=document.getElementById('paneB'),
      sl=document.getElementById('sl'), fno=document.getElementById('fno'),
      play=document.getElementById('play'), zoomBtn=document.getElementById('zoom');
const ZOOMS=[1,2,3];
let i=0, timer=null, zi=0;

function cells(t){
  const px = t.px==null ? '&mdash;' : t.px+' px';
  const top = t.top==null ? '&mdash;' : t.top.toFixed(0)+' mm';
  const gap = t.gap==null ? '&mdash;' : t.gap.toFixed(0)+' mm';
  // 24-90 mm is what the task's objects are; outside it the chosen thing is a
  // fragment or something that is not an object.
  const hCls = t.top==null ? 'dim' : ((t.top<24||t.top>90) ? 'bad' : 'ok');
  return `<div><p class="t-k">instances</p><p class="t-v">${t.n}</p></div>`+
         `<div><p class="t-k">target size</p><p class="t-v ${t.px==null?'dim':''}">${px}</p></div>`+
         `<div><p class="t-k">target height</p><p class="t-v ${hCls}">${top}</p></div>`+
         `<div><p class="t-k">to hand</p><p class="t-v ${t.gap==null?'dim':''}">${gap}</p></div>`;
}
function applyZoom(s){
  const z=ZOOMS[zi];
  const ox=(s.zx==null?0.5:s.zx)*100, oy=(s.zy==null?0.5:s.zy)*100;
  for(const el of [imgA,imgB]){
    el.style.transformOrigin = ox+'% '+oy+'%';
    el.style.transform = 'scale('+z+')';
  }
}
function draw(k){
  if(!L.length) return;
  i=Math.max(0,Math.min(k,L.length-1));
  imgA.src='data:image/jpeg;base64,'+L[i];
  imgB.src='data:image/jpeg;base64,'+R[i];
  sl.value=i;
  const s=S[i]||{a:{n:0},b:{n:0}};
  fno.textContent=`frame ${s.i} \u00b7 t=${s.t}s \u00b7 ${i+1} / ${L.length}`;
  telA.innerHTML=cells(s.a); telB.innerHTML=cells(s.b);
  paneA.classList.toggle('win', s.a.px!=null);
  paneB.classList.toggle('win', s.b.px!=null);
  applyZoom(s);
}
sl.addEventListener('input',e=>draw(+e.target.value));
play.addEventListener('click',()=>{
  if(timer){clearInterval(timer);timer=null;play.textContent='Play';return;}
  play.textContent='Pause';timer=setInterval(()=>draw((i+1)%L.length),200);
});
zoomBtn.addEventListener('click',()=>{
  zi=(zi+1)%ZOOMS.length;
  zoomBtn.textContent='Zoom '+ZOOMS[zi]+'\u00d7';
  zoomBtn.setAttribute('aria-pressed', zi>0 ? 'true':'false');
  applyZoom(S[i]||{});
});
addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'){draw(i+1);e.preventDefault();}
  if(e.key==='ArrowLeft'){draw(i-1);e.preventDefault();}
});
draw(0);
</script>"""


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("session", type=pathlib.Path)
  p.add_argument("--masks", type=pathlib.Path, required=True)
  p.add_argument("--name", default="sam3_dart")
  p.add_argument("--tag", default="SAM3 + DART",
                 help="what the right pane's model is, shown beside its name")
  p.add_argument("--stride", type=int, default=24)
  p.add_argument("--window", type=float, nargs=2, default=None,
                 metavar=("T0", "T1"), help="seconds from the first frame")
  p.add_argument("--quality", type=int, default=62)
  p.add_argument("--scale", type=float, default=0.78)
  p.add_argument("--out", type=pathlib.Path, default=None)
  a = p.parse_args()

  d = build(a.session, a.masks, a.name, a.stride, a.window, a.quality, a.scale)
  if not d["left"]:
    print("no frames rendered", file=sys.stderr)
    return 1
  out = a.out or (segbench.RESULTS / a.session.name / f"compare_{a.name}.html")
  out.parent.mkdir(parents=True, exist_ok=True)
  page = PAGE
  for k, v in (("__SESSION__", d["session"]), ("__NAME__", d["name"]),
               ("__TAG__", a.tag), ("__LAST__", str(len(d["left"]) - 1)),
               ("__LEFT__", json.dumps(d["left"])),
               ("__RIGHT__", json.dumps(d["right"])),
               ("__ROWS__", json.dumps(d["rows"])),
               ("__NAMEJSON__", json.dumps(d["name"]))):
    page = page.replace(k, v)
  out.write_text(page)
  print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(d['left'])} frames)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
