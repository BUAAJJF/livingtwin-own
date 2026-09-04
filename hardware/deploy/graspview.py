"""Watch the target being deleted, one frame at a time.

The numbers say the object is not occluded when the gripper arrives: at every
frame where the depth backend lost its target with the hand inside 80 mm, the
object's points were still in the frame, a median of one of them was inside the
arm's own geometry, and a median of 52 were inside the 20 mm margin
``arm_mask`` adds around it.  That is a claim about a picture, so this draws the
picture.

Each pane is the recorded frame with the exclusion painted on it:

* **deep red** is the arm's own sphere cover -- the robot itself.
* **orange** is the 20 mm clearance margin and nothing else.  This is the layer
  the argument is about.  An object that vanishes under orange while the deep
  red is somewhere else was not hidden by the gripper; it was removed by the
  margin around it.
* **blue outlines** are the instances that survived, **green** is the one the
  tracker chose, and a pane with no green says NO TARGET.

Left is the deployment as it stands.  Right is the same frame with the already
confirmed target exempted from the exclusion, height-capped so the exemption
cannot also spare the gripper standing over the object.

Frames are chosen around the moments the target was lost with the hand close,
because that is when the two panes differ, and they are cropped to the hand so
that a 25-pixel object is not a 25-pixel object on screen.

    python -m hardware.deploy.graspview recordings/v4_stereo_repro_scene2
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import pathlib
import sys

import cv2
import numpy as np

from . import config, mask, overlay, proprio, segbench

CROP = (520, 380)
"""Width and height of the window kept around the hand.  Big enough to hold the
gripper, the object and enough table to see where they are; small enough that
the object is not a speck."""


def _layers(seg, rig, depth, arm, protect=None):
  """The two exclusion layers and the instance labels, on the sensor grid.

  Recomputed here rather than read out of ``DepthSegmenter`` because the class
  does not keep them -- and recomputing costs a few milliseconds against a
  rendering pass that is already writing JPEGs.
  """
  d = seg.reproj.source(depth)
  h, w = d.shape
  pts = seg.reproj.points_base(d, rig)
  src = seg.reproj.last_src
  in_box = mask.workspace_mask(pts)
  body = mask.arm_mask(pts, arm, 0.0, within=in_box)
  full = mask.arm_mask(pts, arm, seg.cfg.arm_clearance_m, within=in_box)
  lay = np.zeros(h * w, dtype=np.uint8)
  lay[src[in_box & full & ~body]] = 1        # the margin only
  lay[src[in_box & body]] = 2                # the arm itself
  # What the exemption handed back.  ``mask.arm_mask`` is patched while the
  # protected pass runs, so ``full`` above is already post-exemption; the set
  # that was given back is stashed by the patch itself.
  ex = None if protect is None else protect.get("_exempted")
  if ex is not None and ex.any():
    lay[src[ex]] = 3
  return lay.reshape(h, w)


PERMISSIVE = dict(min_area_px=400, max_area_px=10_000_000, max_top_z_m=10.0,
                  min_top_z_floor_m=0.0, min_top_z_sigmas=0.0,
                  width_range_m=(0.0, 100.0), max_elongation=1e6)
"""Every gate opened, so a second segmenter's instances ARE the components the
real one had to judge.  Drawing those is the difference between a pane that
says "nothing was found" and one that shows the single arm-shaped blob that was
found and thrown away -- which is the whole argument about what the exclusion
does.  Built by opening the real config rather than by reimplementing the
thresholding, so the two cannot drift apart.

``min_area_px`` is opened to 400 and not to nothing: at the real 150 the point
is moot, and below about a hundred the sensor's own scatter produces thousands
of components a frame, each of which costs a percentile over its points.  That
is minutes per session spent drawing specks nobody can see."""


def _sphere_px(rig, protect, radius_m):
  """The exemption sphere projected into the image: centre and radius in px.

  The radius is taken from two projected points rather than from a focal
  length, so it stays right without this file needing to know the intrinsics.
  """
  c = protect.get("centre")
  if c is None:
    return None
  c = np.asarray(c, dtype=np.float64)
  pc = overlay.project(c[None], rig, (480, 848, 3))[0]
  pe = overlay.project((c + np.array([radius_m, 0.0, 0.0]))[None], rig,
                       (480, 848, 3))[0]
  if not (np.isfinite(pc).all() and np.isfinite(pe).all()):
    return None
  return (float(pc[0]), float(pc[1])), float(np.linalg.norm(pe - pc))


def _paint(gray, lay, decimate, seg_out, label, tag, focus_px, lost,
           raw_out=None, rect=None, frame_id=None, sphere=None):
  img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
  up = np.repeat(np.repeat(lay, decimate, 0), decimate, 1)
  up = up[:img.shape[0], :img.shape[1]]
  # Tinted rather than replaced: the object has to stay visible underneath, or
  # the picture cannot show that it is still there.
  for value, colour in ((1, (40, 140, 235)), (2, (60, 60, 190)),
                        (3, (90, 230, 90))):
    m = up == value
    if m.any():
      img[m] = (0.45 * img[m] + 0.55 * np.array(colour, np.float32)).astype(np.uint8)
  # What was found before the filters, underneath what survived them.
  if raw_out is not None:
    rl = raw_out.labels
    if decimate > 1:
      rl = np.repeat(np.repeat(rl, decimate, 0), decimate, 1)
    rl = rl[:img.shape[0], :img.shape[1]]
    for inst in raw_out.instances:
      cs, _ = cv2.findContours((rl == inst.label).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cv2.drawContours(img, cs, -1, (170, 170, 170), 1)
      x, y, _w, _h = inst.bbox
      cv2.putText(img, f"{inst.top_z * 1000:.0f}mm {inst.n_px * decimate ** 2}px",
                  (int(x * decimate), max(12, int(y * decimate) - 4)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
  if seg_out is not None:
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
  # Where the exemption applies, as the segmenter sees it: the circle is the
  # sphere's silhouette at the target's own range, so it is what the pixels
  # inside it were spared by.
  if sphere is not None:
    (sx, sy), sr = sphere
    if np.isfinite(sx) and np.isfinite(sy) and sr > 1:
      cv2.circle(img, (int(sx), int(sy)), int(sr), (90, 230, 90), 1, cv2.LINE_AA)
  if np.isfinite(focus_px).all():
    cv2.drawMarker(img, (int(focus_px[0]), int(focus_px[1])), (255, 210, 0),
                   cv2.MARKER_CROSS, 20, 2)
  # A window that does not move.  Following the target keeps it centred and
  # makes the whole scene appear to swim: over one grasp the follow point
  # travelled 153 px across and 144 down, stepping up to 9 px between frames
  # and jumping outright whenever the target was lost and the crop fell back
  # to the hand.  The camera itself is bolted down and measures under half a
  # pixel of drift over the same 98 frames, so all of that motion was the
  # crop's.  One rectangle per grasp, sized to hold the whole approach.
  if rect is not None:
    x0, y0, x1, y1 = rect
  else:
    cw, ch = CROP
    cx = int(np.clip(focus_px[0] if np.isfinite(focus_px[0]) else img.shape[1] / 2,
                     cw // 2, img.shape[1] - cw // 2))
    cy = int(np.clip(focus_px[1] if np.isfinite(focus_px[1]) else img.shape[0] / 2,
                     ch // 2, img.shape[0] - ch // 2))
    x0, y0, x1, y1 = cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2
  img = img[y0:y1, x0:x1]
  cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                (60, 60, 190) if lost else (80, 200, 80), 3)
  cv2.putText(img, tag, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
              (255, 255, 255), 2, cv2.LINE_AA)
  if lost:
    cv2.putText(img, "NO TARGET", (10, img.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (90, 90, 245), 2, cv2.LINE_AA)
  # The frame id, burned in.  It is in the readout below the panes too, but a
  # number on the picture is what survives a screenshot and a scroll-back, and
  # this page exists to be argued over frame by frame.
  if frame_id is not None:
    text = f"#{frame_id}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    x = img.shape[1] - tw - 14
    cv2.rectangle(img, (x - 8, 8), (x + tw + 8, 18 + th + 8), (20, 20, 20), -1)
    cv2.putText(img, text, (x, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (0, 235, 255), 2, cv2.LINE_AA)
  return img


def _pass(session, rig, reproj, meta, files, protect_r, protect_cap, render_set,
          quality, no_arm=False, tag=None, width_max=None, confirm=3,
          rects=None):
  """One replay.  Records every frame; renders only ``render_set``."""
  cfg = (None if width_max is None else dataclasses.replace(
    mask.SegmenterCfg(),
    width_range_m=(mask.SegmenterCfg().width_range_m[0], float(width_max))))
  seg = mask.DepthSegmenter(rig, reproj, cfg=cfg)
  raw = mask.DepthSegmenter(
    rig, reproj, cfg=dataclasses.replace(mask.SegmenterCfg(), **PERMISSIVE))
  tracker = mask.TargetTracker(confirm=int(confirm))
  kin = proprio.Kinematics()
  protect = {"centre": None, "ceiling": None}
  rows, shots = [], {}
  with segbench.target_protection(protect_r, protect):
    for f in files:
      key = int(pathlib.Path(f).stem)
      m = meta.get(key)
      if m is None:
        continue
      q = np.asarray(m["joint_pos"], dtype=np.float64)
      g = float(q[6]) if q.size > 6 else 0.05
      kin.update(np.array([*q[:6], g, -g]))
      arm = kin.link_spheres()
      z = np.load(f)
      depth = z["depth"].astype(np.float32) / 10000.0
      protect["centre"] = getattr(tracker, "_centroid", None)
      protect["ceiling"] = (None if protect["centre"] is None or protect_cap <= 0
                            else float(protect["centre"][2]) + protect_cap)
      out = seg(depth, rgb=z["gray"], arm=(None if no_arm else arm))
      label = tracker.update(out, kin.site_pos)
      hit = next((i for i in out.instances if i.label == label), None)
      gap = (None if hit is None else
             float(np.linalg.norm(kin.site_pos - np.asarray(hit.centroid_base))) * 1000)
      rows.append({"i": key, "gap": gap, "has": hit is not None})
      if key in render_set:
        focus = (np.asarray(hit.centroid_base) if hit is not None
                 else kin.site_pos)
        fp = overlay.project(np.asarray(focus)[None], rig, (480, 848, 3))[0]
        lay = _layers(seg, rig, depth, arm,
                      protect if protect_r > 0 else None)
        label_text = tag or ("as deployed" if protect_r <= 0
                             else "target exempted")
        # With the exclusion off there is nothing to paint, and painting it
        # anyway would show a boundary the segmenter did not use.
        raw_out = raw(depth, rgb=z["gray"],
                      arm=(None if no_arm else arm))
        img = _paint(z["gray"], np.zeros_like(lay) if no_arm else lay,
                     seg.decimate, out, label, label_text, fp, hit is None,
                     raw_out, (rects or {}).get(key), key,
                     _sphere_px(rig, protect, protect_r) if protect_r > 0
                     else None)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
          shots[key] = dict(b64=base64.b64encode(buf).decode(),
                            n=len(out.instances),
                            px=(None if hit is None else int(hit.n_px) * seg.decimate ** 2),
                            top=(None if hit is None or not np.isfinite(hit.top_z)
                                 else round(float(hit.top_z) * 1000, 1)),
                            gap=(None if gap is None else round(gap, 1)))
  return rows, shots


def windows(rig, meta, ids, spans, pad=110, min_wh=(360, 280)):
  """One crop rectangle per span, covering everywhere the hand goes in it.

  Built from the projected grasp site rather than from the chosen target: the
  site exists on every frame, including the ones where the target was lost,
  so the rectangle does not depend on the thing the two panes disagree about.
  """
  kin = proprio.Kinematics()
  pos = {}
  for i in ids:
    m = meta.get(i)
    if m is None:
      continue
    q = np.asarray(m["joint_pos"], dtype=np.float64)
    g = float(q[6]) if q.size > 6 else 0.05
    kin.update(np.array([*q[:6], g, -g]))
    fp = overlay.project(kin.site_pos[None], rig, (480, 848, 3))[0]
    if np.isfinite(fp).all():
      pos[i] = fp
  H, W = 480, 848
  rects = {}
  for sp in spans:
    p = np.array([v for i, v in pos.items() if sp["from"] <= i <= sp["to"]])
    if p.size == 0:
      continue
    x0, y0 = p.min(axis=0) - pad
    x1, y1 = p.max(axis=0) + pad
    # Grown about its own centre to the minimum, then slid inside the frame.
    def fit(lo, hi, need, limit):
      if hi - lo < need:
        c = 0.5 * (lo + hi)
        lo, hi = c - need / 2, c + need / 2
      lo, hi = int(round(lo)), int(round(hi))
      if lo < 0:
        hi, lo = hi - lo, 0
      if hi > limit:
        lo, hi = max(0, lo - (hi - limit)), limit
      return lo, hi
    x0, x1 = fit(x0, x1, min_wh[0], W)
    y0, y1 = fit(y0, y1, min_wh[1], H)
    for i in ids:
      if sp["from"] <= i <= sp["to"]:
        rects[i] = (x0, y0, x1, y1)
  return rects


def choose_grasps(meta, rows, before, after, budget, effort=0.7,
                  jaw_mm=60.0, stride=1):
  """Frames around each grasp: the approach into it and the carry out of it.

  The loss-event selection below finds where the target was dropped, which is
  the right window for asking why it was dropped.  It is the wrong window for
  showing a fix whose largest effect is after the jaws close, and it does not
  contain a single carrying frame on this session.  A grasp is where the
  gripper and the object become one component, which is the moment the
  footprint ceiling decides.

  A grasp is identified from the effort trace, not from the segmenter:
  ``gripper_effort.json`` measures 0.085 closing on nothing against 1.043
  closing on an object, so the threshold separates them with room to spare.
  """
  index = {r["i"]: k for k, r in enumerate(rows)}
  carrying = [r["i"] for r in rows
              if (m := meta.get(r["i"])) is not None
              and m.get("gripper_effort") is not None
              and abs(float(m["gripper_effort"])) > effort
              and len(m.get("joint_pos", [])) > 6
              and float(m["joint_pos"][6]) * 2000 < jaw_mm]
  runs, cur = [], []
  for i in sorted(carrying):
    if cur and i - cur[-1] > 5:
      runs.append(cur)
      cur = []
    cur.append(i)
  if cur:
    runs.append(cur)
  runs = [r for r in runs if len(r) >= 4]
  runs.sort(key=len, reverse=True)
  keep, spans = set(), []
  for run in runs:
    lo = max(0, index[run[0]] - before)
    hi = min(len(rows) - 1, index[run[-1]] + after)
    ids = [rows[j]["i"] for j in range(lo, hi + 1, max(1, int(stride)))]
    if len(keep) + len(ids) > budget:
      continue
    spans.append({"loss": run[0], "gap": float(len(run)), "from": ids[0],
                  "to": ids[-1]})
    keep.update(ids)
  return sorted(keep), spans


def choose(rows, near_mm, before, after, budget):
  """Frames worth drawing: the approach that ends in a loss, and its recovery.

  A loss with the hand far away is a different failure and is not what these
  panes are about, so the events are ranked by how close the hand was.
  """
  events = []
  for a, b in zip(rows[:-1], rows[1:]):
    if a["has"] and not b["has"] and a["gap"] is not None and a["gap"] < near_mm:
      events.append((a["gap"], b["i"]))
  events.sort()
  index = {r["i"]: k for k, r in enumerate(rows)}
  keep, spans = set(), []
  for gap, i in events:
    k = index[i]
    lo, hi = max(0, k - before), min(len(rows) - 1, k + after)
    ids = [rows[j]["i"] for j in range(lo, hi + 1)]
    if len(keep) + len(ids) > budget:
      break
    spans.append({"loss": i, "gap": round(gap, 1), "from": ids[0], "to": ids[-1]})
    keep.update(ids)
  return sorted(keep), spans


LEDE_FIXED = """<p class="lede">The same recorded frame, twice, cropped to the hand.
<b>Left is the deployment as it stands.</b> <b>Right changes two constants and
nothing else</b>: the footprint ceiling <code>SegmenterCfg.width_range_m</code>
from 200&nbsp;mm to 300&nbsp;mm, and <code>TargetTracker.confirm</code> from 3
frames to 2. No new model, no new sensor, no change to the arm exclusion.</p>
<p class="lede">The grey outlines are components the segmenter found and then
<b>rejected</b>, labelled with the height and area it measured. They are the
point of this page. Watch the moment the jaws close: the gripper and the object
merge into one component, that component measures a little over 200&nbsp;mm
across, and the ceiling throws it away &mdash; on frame 1335 of this session by
6.6&nbsp;mm. The track dies, and the three-frame confirmation window then keeps
the target away for three more frames after the component comes back.</p>
<p class="lede"><b>Measured over the session:</b> a target while carrying rises
from 44.6% of frames to 95.4%, recall against the hand-labelled frames from 0.55
to 0.76, and segmenter recall from 0.69 to 0.90 &mdash; past SAM3's 0.86, at
5.8&nbsp;ms instead of 148. False alarms are unchanged at 0.25. A red border and
<b>NO TARGET</b> mean the tracker had nothing that frame.</p>"""

LEDE_PROTECT = """<p class="lede">The same recorded frame, twice, cropped to a
window that does not move. <b>Left is the deployment as it stands. Right exempts
the already confirmed target from the arm exclusion.</b></p>
<p class="lede"><b>How the exemption works.</b> Every frame, the tracker is
already holding a 3-D position for the target it chose last frame. A sphere of
40&nbsp;mm is placed there and the arm exclusion is switched off inside it,
capped 20&nbsp;mm above the target's own centroid so the exemption cannot also
spare the gripper standing over the object. Nothing else changes: an
unconfirmed blob on the arm is still removed, and the exclusion still covers the
robot everywhere else. The <span style="color:rgb(46,140,68)"><b>green
circle</b></span> on the right pane is that sphere, and the
<span style="color:rgb(46,140,68)"><b>green pixels</b></span> are exactly what
it handed back &mdash; points the exclusion had deleted and the segmenter now
gets to see.</p>
<p class="lede">The exclusion itself is painted on both panes: <b>deep red is
the robot's own sphere cover</b>, <b>orange is the 20&nbsp;mm clearance margin
around it</b>. At the cover's median 50&nbsp;mm radius that margin triples each
sphere's volume, and it is what covers the object as the hand arrives &mdash; on
frame 2375 of this session it deletes 500 of the object's 612 pixels, leaving
112 against a 150&nbsp;px floor, so no component forms at all. Grey outlines are
components found and then rejected, labelled with the height and area measured
for them.</p>
<p class="lede"><b>Measured:</b> the close-range loss hazard falls from 18.3% to
6.2% and the number of frames holding a target with the hand inside 60&nbsp;mm
nearly triples, 93 to 274. <b>It is not free:</b> false alarms on the labelled
empty frames rise from 6/24 to 9/24, because the exclusion had been implicitly
signalling "the object has been picked up" and the exemption removes that
signal. A red border and <b>NO TARGET</b> mean the tracker had nothing.</p>"""

LEDE_NOARM = """<p class="lede">The same recorded frame, twice, cropped to the hand.
<b>Left keeps the arm exclusion</b> and paints it on: deep red is the robot's own
sphere cover, orange is the 20&nbsp;mm margin around it. <b>Right turns the
exclusion off entirely</b> &mdash; nothing is painted there because nothing is
removed.</p>
<p class="lede">The intuition is that switching it off should make the object in
the gripper visible again. <b>Measured, it does the opposite.</b> Detection while
carrying falls from 44.6% of frames to 24.6%, segmenter recall from 0.69 to 0.52,
and instances per frame from 0.83 to 0.38. Watch why: with the exclusion gone the
object does not become a separate thing, it becomes <b>part of the arm</b>. The
outline swallows the gripper, the wrist and the forearm, and the merged component
is 300&nbsp;mm tall against the 110&nbsp;mm a task object may be, so it is thrown
away as arm-shaped. The rejection reason flips to <code>height</code>, 1.03 per
frame.</p>
<p class="lede">Both sides of the dilemma are on screen. <b>Delete the arm and the
object goes with it; keep the arm and the object merges into it.</b> From one
fixed viewpoint a held object is depth-continuous with the hand holding it, and
no threshold on that image separates them. A red border and <b>NO TARGET</b>
mean the tracker had nothing.</p>"""

PAGE = """<title>__SESSION__ — the target being deleted</title>
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
.wrap{max-width:1320px;margin:0 auto;padding:clamp(20px,3vw,34px) clamp(14px,3vw,22px) 46px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
 color:var(--accent);margin:0 0 9px}
h1{font-family:var(--mono);font-size:clamp(19px,2.4vw,25px);font-weight:600;margin:0 0 12px}
.lede{margin:0 0 18px;color:var(--muted);max-width:76ch;font-size:14.5px}
.lede b{color:var(--ink)}
.panes{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 12px}
@media (max-width:940px){.panes{grid-template-columns:1fr}}
.pane{background:var(--surface);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.pane img{display:block;width:100%;height:auto;background:var(--sunk)}
.tel{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line)}
.tel>div{padding:9px 12px;border-right:1px solid var(--line)}
.tel>div:last-child{border-right:none}
.t-k{font-family:var(--mono);font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;
 color:var(--faint);margin:0 0 4px}
.t-v{font-family:var(--mono);font-size:14px;margin:0;font-variant-numeric:tabular-nums}
.t-v.dim{color:var(--faint)} .t-v.bad{color:var(--bad)} .t-v.ok{color:var(--ok)}
.bar{background:var(--surface);border:1px solid var(--line);border-radius:4px;display:flex;
 align-items:center;gap:14px;padding:11px 15px;flex-wrap:wrap;margin:0 0 12px}
button{font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;
 background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:3px;
 padding:6px 13px;cursor:pointer;min-width:74px}
button:hover{border-color:var(--accent);color:var(--accent)}
button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
select{font-family:var(--mono);font-size:12px;background:var(--surface);color:var(--ink);
 border:1px solid var(--line);border-radius:3px;padding:6px 9px}
input[type=range]{flex:1;min-width:200px;accent-color:var(--accent)}
.fno{font-family:var(--mono);font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;
 white-space:nowrap}
.key{display:flex;flex-wrap:wrap;gap:18px;font-family:var(--mono);font-size:11.5px;
 color:var(--muted);margin:0 0 4px}
.key span{display:flex;align-items:center;gap:7px}
.sw{width:15px;height:11px;border-radius:2px;display:inline-block;border:1px solid rgba(0,0,0,.15)}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
</style>
<div class="wrap">
<p class="eyebrow">grasp approach &middot; __SESSION__</p>
<h1>__H1__</h1>
__LEDE__
<div class="key">
  <span><i class="sw" style="background:rgb(190,60,60)"></i> the arm itself</span>
  <span><i class="sw" style="background:rgb(235,140,40)"></i> the 20 mm margin only</span>
  <span><i class="dot" style="background:rgb(80,235,80)"></i> chosen target</span>
  <span><i class="dot" style="background:rgb(60,170,235)"></i> other instance</span>
  <span><i class="dot" style="background:rgb(0,210,255)"></i> gripper site</span>
  <span><i class="sw" style="background:rgb(170,170,170)"></i> found, then rejected &mdash; labelled with its measured height</span>
  <span><i class="sw" style="background:rgb(90,230,90)"></i> exempted &mdash; pixels the arm exclusion would have deleted</span>
</div>
<div class="panes">
  <div class="pane"><img id="imgA" alt="the frame as the deployment segments it" />
    <div class="tel" id="telA"></div></div>
  <div class="pane"><img id="imgB" alt="the same frame with the target exempted" />
    <div class="tel" id="telB"></div></div>
</div>
<div class="bar">
  <button id="play">Play</button>
  <select id="ev" aria-label="approach"></select>
  <input type="range" id="sl" min="0" max="__LAST__" value="0" aria-label="frame" />
  <span class="fno" id="fno"></span>
</div>
</div>
<script>
const A=__A__, B=__B__, IDS=__IDS__, SPANS=__SPANS__;
const imgA=document.getElementById('imgA'),imgB=document.getElementById('imgB'),
      telA=document.getElementById('telA'),telB=document.getElementById('telB'),
      sl=document.getElementById('sl'),fno=document.getElementById('fno'),
      play=document.getElementById('play'),ev=document.getElementById('ev');
let i=0,timer=null;
SPANS.forEach((s,k)=>{const o=document.createElement('option');
  o.value=k;o.textContent=`approach ${k+1} — lost at frame ${s.loss}, hand ${s.gap} mm`;
  ev.appendChild(o);});
function cells(t){
  if(!t) return '';
  const px=t.px==null?'&mdash;':t.px+' px', top=t.top==null?'&mdash;':t.top.toFixed(0)+' mm',
        gap=t.gap==null?'&mdash;':t.gap.toFixed(0)+' mm';
  return `<div><p class="t-k">instances</p><p class="t-v">${t.n}</p></div>`+
   `<div><p class="t-k">target</p><p class="t-v ${t.px==null?'bad':'ok'}">${t.px==null?'lost':px}</p></div>`+
   `<div><p class="t-k">height</p><p class="t-v ${t.top==null?'dim':''}">${top}</p></div>`+
   `<div><p class="t-k">to hand</p><p class="t-v ${t.gap==null?'dim':''}">${gap}</p></div>`;
}
function draw(k){
  if(!IDS.length) return;
  i=Math.max(0,Math.min(k,IDS.length-1));
  const id=IDS[i];
  if(A[id]) imgA.src='data:image/jpeg;base64,'+A[id].b64;
  if(B[id]) imgB.src='data:image/jpeg;base64,'+B[id].b64;
  sl.value=i;
  fno.textContent=`frame ${id} \\u00b7 ${i+1} / ${IDS.length}`;
  telA.innerHTML=cells(A[id]); telB.innerHTML=cells(B[id]);
}
sl.addEventListener('input',e=>draw(+e.target.value));
ev.addEventListener('change',e=>{const s=SPANS[+e.target.value];
  const k=IDS.indexOf(s.from); if(k>=0) draw(k);});
play.addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;play.textContent='Play';return;}
 play.textContent='Pause';timer=setInterval(()=>draw((i+1)%IDS.length),220);});
addEventListener('keydown',e=>{if(e.key==='ArrowRight'){draw(i+1);e.preventDefault();}
 if(e.key==='ArrowLeft'){draw(i-1);e.preventDefault();}});
draw(0);
</script>"""


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("session", type=pathlib.Path)
  p.add_argument("--near", type=float, default=90.0,
                 help="a loss counts when the hand was this close, in mm")
  p.add_argument("--before", type=int, default=14)
  p.add_argument("--after", type=int, default=10)
  p.add_argument("--budget", type=int, default=150,
                 help="most frames to draw")
  p.add_argument("--select", choices=("loss", "grasp"), default="loss",
                 help="windows around lost targets, or around grasps")
  p.add_argument("--effort", type=float, default=0.7,
                 help="|gripper_effort| above which the jaws are on something. "
                      "gripper_effort.json measures 0.085 free, 1.043 held; a "
                      "light object loads the drive less, so a lower bar finds "
                      "grasps a 0.7 bar misses")
  p.add_argument("--jaw", type=float, default=60.0, metavar="MM")
  p.add_argument("--select-stride", type=int, default=1,
                 help="thin the chosen windows -- the scene moves slowly and a "
                      "whole grasp does not fit in one page otherwise")
  p.add_argument("--variant", choices=("protect", "noarm", "fixed"),
                 default="protect",
                 help="what the right pane shows against the deployment")
  p.add_argument("--protect", type=float, default=0.04)
  p.add_argument("--width-max", type=float, default=0.30)
  p.add_argument("--confirm", type=int, default=2)
  p.add_argument("--cap", type=float, default=0.020)
  p.add_argument("--quality", type=int, default=68)
  p.add_argument("--out", type=pathlib.Path, default=None)
  a = p.parse_args()

  rig, reproj, meta, files = segbench._session_setup(a.session)
  rows, _ = _pass(a.session, rig, reproj, meta, files, 0.0, 0.0, set(), a.quality)
  if a.select == "grasp":
    ids, spans = choose_grasps(meta, rows, a.before, a.after, a.budget,
                               a.effort, a.jaw, a.select_stride)
    what = "grasps"
  else:
    ids, spans = choose(rows, a.near, a.before, a.after, a.budget)
    what = "approaches"
  if not ids:
    print(f"no {what} to draw", file=sys.stderr)
    return 1
  print(f"{len(spans)} {what}, {len(ids)} frames", flush=True)
  keep = set(ids)
  rects = windows(rig, meta, ids, spans)
  print("fixed windows: " + ", ".join(
    f"{sp['from']}-{sp['to']} -> {rects.get(sp['from'])}" for sp in spans),
    flush=True)
  _, shots_a = _pass(a.session, rig, reproj, meta, files, 0.0, 0.0, keep,
                     a.quality,
                     tag=("as deployed  (200 mm, confirm 3)"
                          if a.variant == "fixed"
                          else "arm exclusion ON  (as deployed)"),
                     rects=rects)
  if a.variant == "noarm":
    _, shots_b = _pass(a.session, rig, reproj, meta, files, 0.0, 0.0, keep,
                       a.quality, no_arm=True, tag="arm exclusion OFF",
                       rects=rects)
  elif a.variant == "fixed":
    _, shots_b = _pass(a.session, rig, reproj, meta, files, 0.0, 0.0, keep,
                       a.quality, width_max=a.width_max, confirm=a.confirm,
                       tag=f"FIXED  ({a.width_max * 1000:.0f} mm, confirm {a.confirm})",
                       rects=rects)
  else:
    _, shots_b = _pass(a.session, rig, reproj, meta, files, a.protect, a.cap,
                       keep, a.quality, tag="target exempted", rects=rects)

  out = a.out or (segbench.RESULTS / a.session.name /
                  (f"graspview_{a.variant}.html"))
  out.parent.mkdir(parents=True, exist_ok=True)
  h1 = {"protect": "The target being deleted",
        "noarm": "What happens with the exclusion off",
        "fixed": "Two constants"}[a.variant]
  lede = {"protect": LEDE_PROTECT, "noarm": LEDE_NOARM,
          "fixed": LEDE_FIXED}[a.variant]
  page = PAGE
  for k, v in (("__SESSION__", a.session.name), ("__LAST__", str(len(ids) - 1)),
               ("__H1__", h1), ("__LEDE__", lede),
               ("__A__", json.dumps(shots_a)), ("__B__", json.dumps(shots_b)),
               ("__IDS__", json.dumps(ids)), ("__SPANS__", json.dumps(spans))):
    page = page.replace(k, v)
  out.write_text(page)
  print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(ids)} frames)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
