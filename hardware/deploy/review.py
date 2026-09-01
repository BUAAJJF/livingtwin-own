"""Replay a recorded session with the task's geometry drawn on it.

``run.py --record`` preserves the depth, the grayscale and the joint angles, so
a rehearsal can be re-segmented afterwards with the geometry the operator has
to satisfy drawn over it: where an object may be, where training puts them, and
where the bin is.

Deliberately offline.  It re-runs the segmenter and the tracker over the frames
in the order they were recorded, which is what reproduces the live behaviour --
sampling every Nth frame does not, because the tracker confirms an instance
across consecutive frames and a strided replay hands it a scene that jumps.

    python -m hardware.deploy.review recordings/session --out review.html
"""

from __future__ import annotations

import argparse
import base64
import collections
import glob
import json
import math
import os
import pathlib
import sys

import cv2
import numpy as np

from . import config, mask, obs as obs_mod, overlay, proprio, rectify


def _turbo(depth: np.ndarray, lo=0.35, hi=1.20) -> np.ndarray:
  v = np.clip((depth - lo) / (hi - lo), 0, 1)
  img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
  img[depth <= 0] = (32, 32, 32)
  return img


def run(session: pathlib.Path, stride: int, quality: int,
        window: "tuple[float, float] | None" = None) -> dict:
  """Segment and track every frame; render a subset of them.

  ``stride`` and ``window`` only thin the pictures -- every frame is still
  segmented, tracked and counted, so the numbers do not change with them.

  The two exist because they trade against each other and against the page.
  A whole 60 s session at every frame is 2900 pictures and about half a
  gigabyte, which no browser will open; at ``stride=12`` it is 45 MB and
  1.25 pictures a second, which is too coarse to see a grasp happen.  A
  ``window`` of a few seconds at ``stride=1`` is the third option and it is
  usually the one wanted: full rate over the moment in question.
  """
  rig_file = session / "rig.json"
  rig = config.Rig.load(rig_file if rig_file.exists() else config.RIG_FILE)
  reproj = rectify.Reprojector(rig, device="cpu")
  seg = mask.DepthSegmenter(rig, reproj)
  tracker = mask.TargetTracker()
  kin = proprio.Kinematics()

  # If the run flattened the scene onto the table plane, the review has to as
  # well, or the pictures show a different observation from the one the policy
  # was given.
  flatten_plane = None
  rj = session / "run.json"
  if rj.exists():
    try:
      if json.loads(rj.read_text()).get("args", {}).get("flatten_scene"):
        flatten_plane = reproj.ground_plane_depth(rig)
    except Exception:
      pass

  meta = {m["i"]: m for m in json.loads((session / "meta.json").read_text())
          if "i" in m and "joint_pos" in m}
  files = sorted(glob.glob(str(session / "*.npz")))
  if not files:
    # ValueError, not SystemExit: this is a library function, and SystemExit
    # walks straight through the ``except Exception`` its callers guard with.
    # A --dry-run session legitimately has no frames, and the exit-time review
    # ended the process over it.
    raise ValueError(f"no frames in {session}")

  stamps = [float(v["frame_stamp"]) for v in meta.values()
            if v.get("frame_stamp")]
  t_zero = min(stamps) if stamps else 0.0
  frames, rows, shown, chosen = [], [], [], collections.Counter()
  (rlo, rhi), (alo, ahi), _ = config.WORKSPACE_SECTOR
  for k, f in enumerate(files):
    m = meta.get(int(os.path.basename(f).split(".")[0]))
    if m is None:
      continue
    q = np.asarray(m["joint_pos"], dtype=np.float64)
    g = float(q[6]) if q.size > 6 else 0.05
    kin.update(np.array([*q[:6], g, -g]))
    z = np.load(f)
    depth = z["depth"].astype(np.float32) / 10000.0
    seg_out = seg(depth, rgb=z["gray"], arm=kin.link_spheres())
    label = tracker.update(seg_out, kin.site_pos)
    hit = next((i for i in seg_out.instances if i.label == label), None)
    payload = (mask.full_mask(seg_out, label, seg.decimate) if label else None)
    if hit is not None:
      c = hit.centroid_base
      chosen[(round(float(c[0]), 2), round(float(c[1]), 2))] += 1
    rows.append({
      "i": int(m["i"]),
      "site_mm": [round(float(x) * 1000, 1) for x in kin.site_pos],
      "gap_mm": (None if hit is None else round(float(np.linalg.norm(
        kin.site_pos - np.asarray(hit.centroid_base))) * 1000, 1)),
      "grip_mm": round(g * 2000, 1),
      "grip_target_mm": (None if "target" not in m else
                         round(float(m["target"][6]) * 2000, 1)),
      "effort": (None if m.get("gripper_effort") is None else
                 round(float(m["gripper_effort"]), 3)),
      "instances": len(seg_out.instances),
      "confirmed": len(tracker.confirmed_labels),
      "fill": round(float((depth > 0).mean()), 3),
      "top_mm": None if hit is None else round(hit.top_z * 1000, 1),
      "n_px": None if hit is None else int(hit.n_px),
      "r": None if hit is None else round(
        float(math.hypot(hit.centroid_base[0], hit.centroid_base[1])), 3),
      "a_deg": None if hit is None else round(math.degrees(math.atan2(
        hit.centroid_base[1], hit.centroid_base[0])), 1),
    })
    if window is not None:
      # Seconds from the first recorded frame, using the recorded stamps
      # rather than the frame index: with --depth-source stereo the recording
      # is at the perception rate, not the camera's, so an index is not a time.
      t_s = (float(m["frame_stamp"]) - t_zero) if m.get("frame_stamp") else None
      if t_s is None or not (window[0] <= t_s <= window[1]):
        continue
    elif k % stride:
      continue
    gray = cv2.cvtColor(z["gray"], cv2.COLOR_GRAY2BGR)
    site = kin.site_pos.copy()
    overlay.annotate(gray, rig, seg_out, label, seg.decimate,
                     caption="t=%.1fs  gripper %.0f mm  green=chosen target"
                             % (k / float(config.D405_FPS), site[2] * 1000))
    # Where the gripper is, through the same calibration.  Without it the
    # picture says what the policy saw but not where it put the hand, and the
    # question on the rig is always the distance between those two.
    gp = overlay.project(site[None], rig, gray.shape)[0]
    if np.isfinite(gp).all():
      cv2.drawMarker(gray, (int(gp[0]), int(gp[1])), (0, 190, 255),
                     cv2.MARKER_CROSS, 26, 3)
      if hit is not None:
        tp = overlay.project(np.asarray(hit.centroid_base)[None], rig,
                             gray.shape)[0]
        if np.isfinite(tp).all():
          cv2.line(gray, (int(gp[0]), int(gp[1])), (int(tp[0]), int(tp[1])),
                   (0, 190, 255), 1, cv2.LINE_AA)
          mid = ((gp + tp) / 2).astype(int)
          cv2.putText(gray, "%.0f mm" % (np.linalg.norm(
            site - np.asarray(hit.centroid_base)) * 1000),
            (mid[0] + 6, mid[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 190, 255), 1, cv2.LINE_AA)
    dep = _turbo(depth)
    overlay.draw(dep, rig, labels=False)

    # What the policy actually receives, rebuilt exactly as run.py builds it --
    # scene depth, target mask, and the two crossed.  The sensor views above say
    # what was on the table; these say what the network was shown, and the two
    # are not the same picture: the mask is the one channel with no sensor
    # behind it and the depth is resampled through the calibration.
    dpi, dvalid, dtgt = reproj(depth, payload=payload)
    if flatten_plane is not None:
      dpi, dvalid = obs_mod.flatten_scene(dpi, dvalid, flatten_plane)
    if dtgt is None:
      dtgt = np.zeros_like(dvalid)
    cam = obs_mod.camera_obs(dpi, dvalid, dtgt > 0)
    tiles = []
    for ch, tint, name in ((0, (1.0, 1.0, 1.0), "ch0 scene depth"),
                           (1, (0.45, 1.0, 0.45), "ch1 target mask"),
                           (2, (1.0, 0.80, 0.35), "ch2 masked depth")):
      v = (np.clip(cam[ch], 0, 1) * 255).astype(np.uint8)
      t = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR).astype(np.float32)
      t = (t * np.asarray(tint, np.float32)[None, None, :]).clip(0, 255).astype(np.uint8)
      t = cv2.resize(t, (gray.shape[1] // 3, gray.shape[0] // 3),
                     interpolation=cv2.INTER_NEAREST)
      cv2.putText(t, name, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                  (235, 235, 235), 1, cv2.LINE_AA)
      tiles.append(t)
    strip = np.hstack(tiles)
    strip = np.pad(strip, ((0, 0), (0, 2 * gray.shape[1] - strip.shape[1]),
                           (0, 0)))
    comp = np.vstack([np.hstack([gray, dep]), strip])
    ok, buf = cv2.imencode(".jpg", comp,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
      frames.append(base64.b64encode(buf).decode())
      shown.append(rows[-1])

  ok_rows = [r for r in rows if r["top_mm"] is not None]
  inside = [r for r in ok_rows if rlo < r["r"] < rhi
            and math.degrees(alo) < r["a_deg"] < math.degrees(ahi)]
  return {
    "session": session.name,
    "frames": frames,
    # One entry per rendered image, so a viewer can put the numbers beside the
    # picture they came from.  ``rows`` holds every frame; this is the subset
    # that was drawn.
    #
    # Recorded while rendering rather than recomputed from ``stride``.  The
    # recomputed version counted from the start of the session and was wrong
    # the moment ``--window`` existed: the pictures came from the window and
    # the numbers beside them came from the first seconds of the run, so the
    # page showed a caption that belonged to a different moment.
    "shown": shown,
    "rows": rows,
    "n_frames": len(rows),
    "with_target": len(ok_rows),
    "target_in_sector": len(inside),
    "instances_mean": round(float(np.mean([r["instances"] for r in rows])), 2),
    "fill_mean": round(float(np.mean([r["fill"] for r in rows])), 3),
    "top_mm": (None if not ok_rows else {
      "min": min(r["top_mm"] for r in ok_rows),
      "median": float(np.median([r["top_mm"] for r in ok_rows])),
      "max": max(r["top_mm"] for r in ok_rows)}),
    "n_px": (None if not ok_rows else {
      "min": min(r["n_px"] for r in ok_rows),
      "median": float(np.median([r["n_px"] for r in ok_rows])),
      "max": max(r["n_px"] for r in ok_rows)}),
    "chosen": [{"xy": list(k), "n": v,
                "r": round(math.hypot(*k), 3),
                "a_deg": round(math.degrees(math.atan2(k[1], k[0])), 1)}
               for k, v in chosen.most_common(8)],
    "sector": {"r": [rlo, rhi],
               "a_deg": [round(math.degrees(alo), 1), round(math.degrees(ahi), 1)]},
  }


PAGE_CSS = """
:root{--ground:#eef1f4;--surface:#fff;--sunk:#e4e9ee;--line:#d3dbe3;--ink:#121a21;
 --muted:#5b6976;--faint:#8b97a3;--accent:#0d7f99;--ok:#2f7d55;--bad:#a8402f;
 --mono:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
 --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 --ground:#0c1015;--surface:#141a21;--sunk:#0a0e12;--line:#232e38;--ink:#dbe4eb;
 --muted:#808f9e;--faint:#5f6d7a;--accent:#4cc0dc;--ok:#4fae76;--bad:#e07565;}}
:root[data-theme=dark]{--ground:#0c1015;--surface:#141a21;--sunk:#0a0e12;--line:#232e38;
 --ink:#dbe4eb;--muted:#808f9e;--faint:#5f6d7a;--accent:#4cc0dc;--ok:#4fae76;--bad:#e07565;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.55}
.wrap{max-width:1020px;margin:0 auto;padding:clamp(20px,4vw,40px) clamp(16px,4vw,28px) 56px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
 color:var(--accent);margin:0 0 10px}
h1{font-family:var(--mono);font-size:clamp(20px,3vw,27px);font-weight:600;margin:0 0 12px;
 letter-spacing:-.015em;text-wrap:balance}
.lede{margin:0 0 20px;color:var(--muted);max-width:62ch;font-size:14.5px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:4px;
 overflow:hidden;margin:0 0 12px;box-shadow:0 1px 2px rgba(18,26,33,.06),0 8px 24px rgba(18,26,33,.05)}
.panel img{display:block;width:100%;height:auto;background:var(--sunk)}
.ctl{display:flex;align-items:center;gap:14px;padding:12px 16px;border-top:1px solid var(--line);flex-wrap:wrap}
button{font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;
 background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:3px;
 padding:6px 14px;cursor:pointer;min-width:74px}
button:hover{border-color:var(--accent);color:var(--accent)}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
input[type=range]{flex:1;min-width:170px;accent-color:var(--accent)}
.fno{font-family:var(--mono);font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.tele{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));border-top:1px solid var(--line)}
.tele>div{padding:11px 15px;border-right:1px solid var(--line)}
.tele>div:last-child{border-right:none}
.t-k{font-family:var(--mono);font-size:10px;letter-spacing:.13em;text-transform:uppercase;
 color:var(--faint);margin:0 0 5px}
.t-v{font-family:var(--mono);font-size:16px;margin:0;font-variant-numeric:tabular-nums}
.t-v.bad{color:var(--bad)} .t-v.ok{color:var(--ok)}
.key{display:flex;flex-wrap:wrap;gap:16px;font-family:var(--mono);font-size:11.5px;
 color:var(--muted);margin:0 0 24px;padding-left:2px}
.key span{display:flex;align-items:center;gap:7px}
.sw{width:15px;height:3px;border-radius:2px;display:inline-block}
h2{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--faint);font-weight:500;margin:28px 0 12px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:13px;
 font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:8px 14px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td.v{text-align:left;color:var(--faint);font-size:12px}
th{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);font-weight:500}
.tbl{background:var(--surface);border:1px solid var(--line);border-radius:4px;overflow-x:auto;margin:0 0 8px}
.note{font-size:13.5px;color:var(--muted);max-width:66ch}
.note code{font-family:var(--mono);font-size:12.5px;color:var(--ink);background:var(--sunk);
 padding:1px 5px;border-radius:2px}
"""


def build_page(d: dict, meta: dict | None = None) -> str:
  """The session, as one self-contained page.

  Written beside the recording so that the numbers, the images and the settings
  that produced them cannot drift apart.  A run whose evidence lives in three
  places gets read as three separate runs a week later.
  """
  meta = meta or {}
  F, S = d.get("frames", []), d.get("shown", [])
  n = max(0, len(F) - 1)

  def cell(k, v, cls=""):
    return f'<tr class="{cls}"><td>{k}</td><td>{v}</td></tr>'

  rates = meta.get("rates", {})
  head = "".join([
    cell("stop reason", meta.get("stop_reason", "—")),
    cell("control steps", rates.get("steps", "—")),
    cell("overruns", rates.get("overruns", "—"),
         "bad" if rates.get("overruns") else "ok"),
    cell("no target", rates.get("no_target", "—")),
    cell("held-over mask", rates.get("blind", "—")),
    cell("guard holds", rates.get("guard_holds", "—")),
    cell("stale frames", rates.get("stale", "—")),
  ])

  perc = "".join([
    cell("frames", d.get("n_frames", "—")),
    cell("instances / frame", d.get("instances_mean", "—")),
    cell("depth fill", f"{d.get('fill_mean', 0) * 100:.1f}%"),
    cell("frames with a target",
         f"{d.get('with_target', 0)} ({100 * d.get('with_target', 0) / max(1, d.get('n_frames', 1)):.0f}%)"),
    cell("target inside the sector",
         f"{100 * d.get('target_in_sector', 0) / max(1, d.get('with_target', 1)):.0f}%"),
  ])
  if d.get("top_mm"):
    perc += cell("chosen height",
                 "%.0f – %.0f mm, median %.0f  (objects are 24–90)"
                 % (d["top_mm"]["min"], d["top_mm"]["max"], d["top_mm"]["median"]))
    perc += cell("chosen size",
                 "%d – %d px, median %d  (~180 expected)"
                 % (d["n_px"]["min"], d["n_px"]["max"], d["n_px"]["median"]))

  chosen = "".join(
    f'<tr><td>({c["xy"][0]}, {c["xy"][1]})</td><td>{c["n"]}</td>'
    f'<td>{c["r"]:.2f} m</td><td>{c["a_deg"]:.0f}&deg;</td></tr>'
    for c in d.get("chosen", [])[:6])

  args = meta.get("args", {})
  keep = ("policy", "camera", "mask", "device", "seconds", "min_grasp_height",
          "min_table_clearance", "guard_mode", "max_blind_steps",
          "command_rate_scale", "command_accel_limit", "gripper_accel_limit",
          "max_joint_speed_fraction", "policy_device")
  settings = "".join(
    f"<tr><td>{k.replace('_', ' ')}</td><td>{args[k]}</td></tr>"
    for k in keep if k in args and args[k] is not None)

  return f"""<title>{d.get('session', 'session')} — deployment review</title>
<style>{PAGE_CSS}</style>
<div class="wrap">
<p class="eyebrow">deployment session &middot; {d.get('session', '')}</p>
<h1>{d.get('session', 'session')}</h1>
<p class="lede">Written automatically when the run ended. Every recorded frame
was re-segmented in the order it arrived, with the workspace sector, the bin
and the gripper projected through the calibration this run used.</p>

<div class="panel">
  <img id="fr" alt="recorded frame with the task geometry drawn" />
  <div class="ctl">
    <button id="play">Play</button>
    <input type="range" id="sl" min="0" max="{n}" value="0" aria-label="frame" />
    <span class="fno" id="fno"></span>
  </div>
  <div class="tele" id="tele"></div>
</div>
<div class="key">
  <span><i class="sw" style="background:rgb(90,220,90)"></i> spawn wedge</span>
  <span><i class="sw" style="background:rgb(60,165,215)"></i> accepted sector</span>
  <span><i class="sw" style="background:rgb(150,150,150)"></i> bin</span>
  <span><i class="sw" style="background:rgb(0,190,255);height:9px;width:9px"></i> gripper</span>
  <span><i class="sw" style="background:rgb(80,235,80);height:9px;width:9px;border-radius:50%"></i> chosen target</span>
</div>

<h2>The run</h2>
<div class="tbl"><table>{head}</table></div>

<h2>Perception</h2>
<div class="tbl"><table>{perc}</table></div>

<h2>What the tracker chose</h2>
<div class="tbl"><table>
<tr><th>position (x, y)</th><th>frames</th><th>r</th><th>azimuth</th></tr>
{chosen or '<tr><td colspan="4">no target was ever confirmed</td></tr>'}
</table></div>

<h2>Settings</h2>
<div class="tbl"><table>{settings}</table></div>
<p class="note">Everything on this page comes from the files beside it:
<code>run.json</code>, <code>control.json</code>, <code>meta.json</code> and the
per-frame <code>.npz</code>. Re-make it with
<code>python -m hardware.deploy.review {d.get('session', '')}</code>.</p>
</div>
<script>
const F={json.dumps(F)}, S={json.dumps(S)};
const img=document.getElementById('fr'),sl=document.getElementById('sl'),
      fno=document.getElementById('fno'),play=document.getElementById('play'),
      tele=document.getElementById('tele');
let i=0,timer=null;
function draw(k){{
  if(!F.length) return;
  i=Math.max(0,Math.min(k,F.length-1));
  img.src='data:image/jpeg;base64,'+F[i];
  sl.value=i;
  const s=S[i]||{{}};
  fno.textContent=`frame ${{String(i+1).padStart(2,'0')}} / ${{F.length}}`;
  const z=s.site_mm?s.site_mm[2]:null;
  tele.innerHTML=[
    ['gripper z',(z!=null?z.toFixed(0):'—')+' mm',z!=null&&z<10?'bad':''],
    ['to target',(s.gap_mm!=null?s.gap_mm.toFixed(0):'—')+' mm',s.gap_mm>100?'bad':'ok'],
    ['target height',(s.top_mm!=null?s.top_mm.toFixed(0):'—')+' mm',''],
    ['target size',(s.n_px!=null?s.n_px:'—')+' px',''],
    ['gripper',(s.grip_mm!=null?s.grip_mm.toFixed(0):'—')+' mm',''],
    ['commanded',(s.grip_target_mm!=null?s.grip_target_mm.toFixed(0):'—')+' mm',
      s.grip_target_mm!=null&&s.grip_mm!=null&&s.grip_target_mm<s.grip_mm-2?'ok':''],
    ['gripper load',(s.effort!=null?Math.abs(s.effort).toFixed(2):'—'),
      s.effort!=null&&Math.abs(s.effort)>0.15?'ok':''],
    ['instances',(s.instances!=null?s.instances:'—'),''],
  ].map(([a,b,c])=>`<div><p class="t-k">${{a}}</p><p class="t-v ${{c}}">${{b}}</p></div>`).join('');
}}
sl.addEventListener('input',e=>draw(+e.target.value));
play.addEventListener('click',()=>{{if(timer){{clearInterval(timer);timer=null;play.textContent='Play';return;}}
 play.textContent='Pause';timer=setInterval(()=>draw((i+1)%F.length),200);}});
addEventListener('keydown',e=>{{if(e.key==='ArrowRight')draw(i+1);if(e.key==='ArrowLeft')draw(i-1);}});
draw(0);
</script>
"""


def summary(d: dict) -> str:
  lines = [
    f"{d['session']}: {d['n_frames']} frames",
    f"  instances/frame     {d['instances_mean']}",
    f"  depth fill          {d['fill_mean'] * 100:.1f}%",
    f"  frames with target  {d['with_target']} "
    f"({100 * d['with_target'] / max(1, d['n_frames']):.0f}%)",
    f"  target in sector    {d['target_in_sector']} "
    f"({100 * d['target_in_sector'] / max(1, d['with_target']):.0f}% of those)",
  ]
  if d["top_mm"]:
    lines.append("  chosen height mm    min %.1f  median %.1f  max %.1f  "
                 "(objects are 24-90)"
                 % (d["top_mm"]["min"], d["top_mm"]["median"], d["top_mm"]["max"]))
    lines.append("  chosen size px      min %d  median %d  max %d  (~180 expected)"
                 % (d["n_px"]["min"], d["n_px"]["median"], d["n_px"]["max"]))
  lines.append("  sector              r %.2f-%.2f m, %.0f-%.0f deg"
               % (*d["sector"]["r"], *d["sector"]["a_deg"]))
  lines.append("  what was chosen:")
  for c in d["chosen"]:
    inside = (d["sector"]["r"][0] < c["r"] < d["sector"]["r"][1]
              and d["sector"]["a_deg"][0] < c["a_deg"] < d["sector"]["a_deg"][1])
    lines.append("    %-16s %5d frames  r %.2f  %6.1f deg  %s"
                 % (str(c["xy"]), c["n"], c["r"], c["a_deg"],
                    "in sector" if inside else "OUTSIDE"))
  return "\n".join(lines)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("session")
  p.add_argument("--stride", type=int, default=100,
                 help="frames between rendered images.  Every frame is still "
                      "segmented and tracked; this only thins the pictures")
  p.add_argument("--window", type=float, nargs=2, metavar=("FROM_S", "TO_S"),
                 default=None,
                 help="render every frame between these two times instead, in "
                      "seconds from the start of the recording.  This is how "
                      "you look at a grasp: the whole-session page is strided "
                      "down to about one picture a second because at full rate "
                      "it is half a gigabyte, and a few seconds at --stride 1 "
                      "is small enough to open and fine enough to see")
  p.add_argument("--quality", type=int, default=86)
  p.add_argument("--json", default=None, help="write the numbers here")
  p.add_argument("--frames-out", default=None,
                 help="write the rendered frames here as base64 JPEG JSON")
  p.add_argument("--html", default=None,
                 help="write the self-contained review page here")
  a = p.parse_args()

  d = run(pathlib.Path(a.session), max(1, a.stride), a.quality,
          window=(tuple(a.window) if a.window else None))
  print(summary(d))
  if a.json:
    pathlib.Path(a.json).write_text(json.dumps(
      {k: v for k, v in d.items() if k not in ("frames", "rows", "shown")},
      indent=1) + "\n")
    print(f"wrote {a.json}")
  if a.frames_out:
    pathlib.Path(a.frames_out).write_text(json.dumps(d))
    print(f"wrote {a.frames_out} ({len(d['frames'])} images)")
  if a.html:
    run_meta = {}
    rj = pathlib.Path(a.session) / "run.json"
    if rj.exists():
      run_meta = json.loads(rj.read_text())
    pathlib.Path(a.html).write_text(build_page(d, run_meta))
    print(f"wrote {a.html}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
