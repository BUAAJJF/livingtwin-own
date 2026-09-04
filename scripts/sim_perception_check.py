"""Score the DEPLOYMENT's perception stack against simulated ground truth.

Every mask number this project has -- detection rate, occlusion, identity
swaps -- came from one of two places that never met.  On the robot the
segmenter's own output was the only thing available, so "is it right" could
only be answered by looking.  In simulation the target mask came straight from
the renderer's segmentation buffer, so it was right by construction and the
deployment's segmenter was never exercised at all.

The consequences are on the record.  A 160 mm filter constant discarded a
164 mm blob for an entire campaign.  An occlusion metric traced the wrong
geometry for a day.  A dropout model was fitted to the worst of twenty
sessions.  None of those are subtle, and all of them survived because there
was no place where the deployment's perception could be scored against a
truth.

This is that place.  ``simsource.SimSource`` renders the scene at the real
sensor's resolution and intrinsics; the pipeline below is imported from
``hardware.deploy`` and is the same code the arm runs -- segmenter, tracker,
lifecycle, reprojector.  What changes between this and a real run is where the
pixels come from and what the actions reach.  Nothing else.

    python scripts/sim_perception_check.py --steps 600 \\
        --policy checkpoints/v7_teachers/strong_teacher.pt --sam --html out.html

Four masks are scored side by side, all on the **policy's** 224x168 grid,
because that is the only grid the policy ever sees:

  depth        the segmenter and ``TargetTracker`` alone -- raw perception
  deploy       depth plus the two fallbacks ``run.py`` really ships: re-using
               the last mask while the arm stands in front of the object, and
               rebuilding it at the grasp site during a carry
  sam_raw      whatever SAM2.1 returned, watchdog ignored -- the upper bound,
               and the only way to tell a strict watchdog from a bad tracker
  sam          sam_raw after the watchdog, which is what may be published
  deploy+sam   sam plus the same two fallbacks

Reported per phase, and per distance from the gripper to the object, which is
where the interesting failure lives.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hardware.deploy import lifecycle as lc_mod  # noqa: E402
from hardware.deploy import mask, proprio, rectify  # noqa: E402
from hardware.deploy.simsource import SimSource  # noqa: E402
from hardware.deploy.target_mask import TargetMask  # noqa: E402

VARIANTS = ("depth", "deploy", "sam_raw", "sam", "deploy+sam")


def load_teacher(path, task, env, device):
  """The state teacher, to drive the arm the way a policy really would.

  Zero actions leave the arm parked and every frame identical, which measures
  the segmenter on one pose.  The interesting failures are all motion:
  the arm crossing the object, the object entering the gripper, the hand
  filling the frame.
  """
  import torch  # noqa: F401
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

  agent = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
    wrapped, asdict(agent), None, device)
  from piper_push import evalcfg
  return wrapped, evalcfg.load_policy(runner, path, device)


def iou(a, b) -> float:
  u = float((a | b).sum())
  return float((a & b).sum()) / u if u else 0.0


def _jpeg(img, quality=72) -> str:
  import cv2
  ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
  return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def thumb(rgb, truth, depth_m, sam_m, scale=2, win=110):
  """One frame as two JPEGs: the whole view, and a crop around the target.

  Outlines rather than fills -- three filled overlays hide the thing they are
  describing, and the question this page exists to answer is *which* pixels
  each stack claimed.

  The crop is not decoration.  At this stand-off the object is about 25 sensor
  pixels across; on a half-scale 424x240 thumbnail that is 12, and three
  one-pixel outlines around a 12-pixel blob are indistinguishable by eye.
  Every visual check this project has run on the full frame has been a check
  that the object is roughly in the right place, which is not the question.
  """
  import cv2
  img = np.ascontiguousarray(rgb[..., ::-1].copy())          # to BGR
  # BGR, because that is what the array is after the flip above.  Written the
  # other way round the first time, which drew the depth mask in the colour
  # the legend gives SAM and vice versa -- a page whose whole purpose is
  # telling two masks apart.
  for m, colour, thick in ((truth, (60, 220, 60), 2),        # green
                           (depth_m, (40, 140, 255), 1),     # orange
                           (sam_m, (255, 120, 60), 1)):      # blue
    if m is None or not m.any():
      continue
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, colour, thick)
  anchor = truth if (truth is not None and truth.any()) else (
    sam_m if (sam_m is not None and sam_m.any()) else depth_m)
  H, W = img.shape[:2]
  if anchor is not None and anchor.any():
    ys, xs = np.nonzero(anchor)
    cy, cx = int(ys.mean()), int(xs.mean())
  else:
    cy, cx = H // 2, W // 2
  y0 = int(np.clip(cy - win, 0, max(H - 2 * win, 0)))
  x0 = int(np.clip(cx - win, 0, max(W - 2 * win, 0)))
  crop = img[y0:y0 + 2 * win, x0:x0 + 2 * win]
  crop = cv2.resize(crop, (330, 330), interpolation=cv2.INTER_NEAREST)
  small = cv2.resize(img, (W // scale, H // scale))
  return _jpeg(small), _jpeg(crop, 82)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Robust")
  p.add_argument("--policy", default=None,
                 help="state teacher to drive the arm; without it the arm "
                      "does not move and only one pose is measured")
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--num-envs", type=int, default=1)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=20260903)
  p.add_argument("--lifecycle", action="store_true",
                 help="run the target lifecycle, as --target-lifecycle does "
                      "on the arm")
  p.add_argument("--sam", action="store_true",
                 help="run SAM2.1 as the target carrier as well")
  p.add_argument("--sam-reanchor", type=int, default=30,
                 help="minimum frames between re-anchors on depth agreement")
  p.add_argument("--sam-reanchor-iou", type=float, default=0.5)
  p.add_argument("--held-radius", type=float, default=0.06,
                 help="the deployment's --held-target-radius")
  p.add_argument("--iou-wrong", type=float, default=0.10,
                 help="a nonempty mask below this IoU is called wrong-target "
                      "rather than merely poor")
  p.add_argument("--html", default=None)
  p.add_argument("--out", default=None)
  a = p.parse_args()

  src = SimSource(task=a.task, device=a.device, num_envs=a.num_envs,
                  seed=a.seed)
  chk = src.self_check()
  print("source self-check:", json.dumps(chk))
  if not chk.get("ok"):
    print("REFUSING to score a stack against a source that does not agree "
          "with its own geometry", file=sys.stderr)
    return 2

  # The deployment's own objects, built the way run.py builds them.
  reproj = rectify.Reprojector(src.rig, device=a.device)
  segmenter = mask.DepthSegmenter(src.rig, reproj)
  tracker = mask.TargetTracker()
  kin = proprio.Kinematics()
  lc = lc_mod.TargetLifecycle() if a.lifecycle else None

  sam = sam_pred = None
  if a.sam:
    from hardware.deploy.sam2_predictor import Sam2StreamingPredictor
    from hardware.deploy.sam_tracker import SamTargetTracker
    sam_pred = Sam2StreamingPredictor(device=a.device)
    sam = SamTargetTracker(sam_pred)
    print(f"sam2.1 loaded in {sam_pred.load_s:.1f}s")

  # The deployment's own object, not a second implementation of it.
  fb = {v: TargetMask(a.held_radius, reproj, src.rig)
        for v in ("deploy", "deploy+sam")}

  policy = wrapped = None
  if a.policy:
    wrapped, policy = load_teacher(a.policy, a.task, src.env, a.device)
    obs = wrapped.get_observations()
    if isinstance(obs, tuple):
      obs = obs[0]

  import torch
  rows = []
  # Which gate threw the target away, when it did.  The segmenter records
  # this and nothing has ever read it; a detection rate without it says
  # "something failed" and leaves the next person to guess which constant.
  rejects = {"approach": Counter(), "holding": Counter()}
  swaps = 0
  last_label = 0

  for step in range(a.steps):
    if policy is not None:
      with torch.inference_mode():
        action = policy(obs)
      obs, _, dones, _ = wrapped.step(action)
      reset_recurrent(policy, dones)
      src._index += 1
    else:
      src.step()
    b = src.latest()

    kin.update(b.joints)
    arm = kin.link_spheres()
    was_holding = lc.holding if lc is not None else False
    seg = segmenter(b.frame.depth, rgb=b.frame.gray, arm=arm)
    label = tracker.update(seg, kin.site_pos)
    if a.lifecycle:
      jaws_closed = float(b.joints[6]) < 0.045
      label = lc.update(label, jaws_closed, b.grasped)
    if label and last_label and label != last_label:
      swaps += 1
    last_label = label or last_label

    phase = "holding" if b.grasped else "approach"
    if not label:
      # The tracker withholds a label for its own reasons -- confirmation
      # frames after a loss, or the stale window -- and those are not the
      # segmenter failing.  Charging them to the segmenter is how a pipeline
      # number becomes a wrong conclusion about a constant.
      if seg.instances:
        rejects[phase][f"segmenter found {len(seg.instances)}, tracker "
                       f"withheld"] += 1
      else:
        for why, _v in (getattr(segmenter, "rejected", None) or []):
          rejects[phase][f"gate: {why}"] += 1
        if not (getattr(segmenter, "rejected", None) or []):
          rejects[phase]["nothing survived to be gated"] += 1

    depth_full = (mask.full_mask(seg, label, segmenter.decimate).astype(bool)
                  if label else np.zeros_like(b.truth_mask))

    # -- SAM, on the sensor grid, carrying what the depth stack chose --------
    sam_full = np.zeros_like(b.truth_mask)
    sam_raw = np.zeros_like(b.truth_mask)
    sam_state, sam_reason = "off", ""
    if sam is not None:
      # ``carry`` is the deployment's own call, anchor policy included -- see
      # ``sam_tracker.SamTargetTracker.carry``.  Reimplementing it here is how
      # the checker would end up scoring code the arm does not run.
      # Reset only on a lifecycle edge -- see the note in ``run.py``.  Keyed
      # on the tracker's window instead, this cleared SAM on every frame of a
      # carry and took its held detection from 100% to 1.2%.
      if lc is not None and was_holding and not lc.holding:
        sam.reset("placed")
      am = segmenter.arm_image_mask(b.frame.depth, arm, kin.site_pos)
      rep = sam.carry(b.frame.rgb, depth_full, b.frame.depth > 0, am,
                      reanchor_frames=a.sam_reanchor,
                      reanchor_iou=a.sam_reanchor_iou)
      sam_state, sam_reason = rep.state.value, rep.reason
      if rep.mask is not None:
        sam_full = rep.mask.astype(bool)
      sam_raw = (rep.raw.astype(bool) if rep.raw is not None
                 else sam_full.copy())

    # -- everything onto the policy's grid, through the same resampler -------
    d_pol, valid, t_depth = reproj(b.frame.depth,
                                   payload=depth_full.astype(np.int32))
    _, _, t_truth = reproj(b.frame.depth, payload=b.truth_mask.astype(np.int32))
    _, _, t_sam = reproj(b.frame.depth, payload=sam_full.astype(np.int32))
    _, _, t_raw = reproj(b.frame.depth, payload=sam_raw.astype(np.int32))
    t_depth = (t_depth > 0) if t_depth is not None else np.zeros_like(valid)
    t_truth = (t_truth > 0) if t_truth is not None else np.zeros_like(valid)
    t_sam = (t_sam > 0) if t_sam is not None else np.zeros_like(valid)
    t_raw = (t_raw > 0) if t_raw is not None else np.zeros_like(valid)

    site = np.asarray(kin.site_pos, dtype=np.float64)
    carrying = (lc.holding if lc is not None else bool(b.grasped))
    masks = {
      "depth": t_depth,
      "sam_raw": t_raw,
      "sam": t_sam,
      "deploy": fb["deploy"](t_depth, label, tracker.has_target, carrying,
                             d_pol, valid, site)[0],
      "deploy+sam": fb["deploy+sam"](t_sam, label,
                                     tracker.has_target, carrying,
                                     d_pol, valid, site)[0],
    }

    obj = src.cmd._object_pos_local()[src.env_id].cpu().numpy().astype(np.float64)
    row = {"step": step, "phase": phase, "label": int(label),
           "grip_dist_mm": float(np.linalg.norm(site - obj)) * 1000.0,
           "truth_px": int(t_truth.sum()),
           "sam_state": sam_state, "sam_reason": sam_reason,
           "sensor_truth_px": int(b.truth_mask.sum())}
    for v, m in masks.items():
      row[v] = {"det": bool(m.any()), "iou": iou(m, t_truth),
                "px": int(m.sum())}
    if a.html:
      row["img"], row["zoom"] = thumb(
        b.frame.rgb, b.truth_mask, depth_full,
        sam_raw if sam_raw.any() else sam_full)
    rows.append(row)

  src.close()

  # -- report --------------------------------------------------------------
  out = {"self_check": chk, "steps": a.steps, "swaps": swaps,
         "lifecycle": bool(a.lifecycle), "sam": bool(a.sam),
         "policy": a.policy, "seed": a.seed, "by_phase": {}}
  if sam_pred is not None:
    out["sam_report"] = sam_pred.report()
    out["sam_rejections"] = dict(sam.rejections)

  def block(title, subset, key):
    if not subset:
      return
    print(f"\n{title}   (n={len(subset)})")
    print(f"{'variant':<12}{'detected':>10}{'IoU mean':>10}{'IoU p50':>9}"
          f"{'wrong target':>14}{'empty':>8}")
    for v in VARIANTS:
      if "sam" in v and not a.sam:
        continue
      det = np.array([r[v]["det"] for r in subset])
      io_ = np.array([r[v]["iou"] for r in subset])
      wrong = det & (io_ < a.iou_wrong)
      out.setdefault(key, {}).setdefault(title, {})[v] = {
        "n": len(subset), "detected": float(det.mean()),
        "iou_mean": float(io_.mean()), "iou_median": float(np.median(io_)),
        "wrong_target": float(wrong.mean()), "empty": float(1 - det.mean())}
      print(f"{v:<12}{100 * det.mean():>9.1f}%{io_.mean():>10.3f}"
            f"{np.median(io_):>9.3f}{100 * wrong.mean():>13.1f}%"
            f"{100 * (1 - det.mean()):>7.1f}%")

  for ph in ("approach", "holding"):
    block(ph, [r for r in rows if r["phase"] == ph], "by_phase")

  # The near field is where the depth segmenter was measured to collapse.
  # Binning by gripper-to-object distance is the only way to see it: averaged
  # over an episode it is a few percent of frames and disappears.
  edges = [(0, 30), (30, 50), (50, 80), (80, 150), (150, 10_000)]
  print("\nby gripper-to-object distance (approach only)")
  print(f"{'mm':<12}{'n':>6}" + "".join(f"{v:>13}" for v in VARIANTS
                                        if a.sam or "sam" not in v))
  for lo, hi in edges:
    sub = [r for r in rows
           if r["phase"] == "approach" and lo <= r["grip_dist_mm"] < hi]
    if not sub:
      continue
    cells = []
    for v in VARIANTS:
      if "sam" in v and not a.sam:
        continue
      det = np.mean([r[v]["det"] for r in sub])
      cells.append(f"{100 * det:>12.0f}%")
      out.setdefault("by_distance", {}).setdefault(f"{lo}-{hi}", {})[v] = \
        float(det)
    print(f"{f'{lo}-{hi}':<12}{len(sub):>6}" + "".join(cells))

  print("\nwhen the depth stack produced nothing, the gate that rejected it:")
  for ph, c in rejects.items():
    if not c:
      continue
    tot = sum(c.values())
    print(f"  {ph:<10} " + "   ".join(f"{k} {100*v/tot:.0f}%"
                                      for k, v in c.most_common(6)))
    out.setdefault("rejects", {})[ph] = dict(c)
  print(f"\nidentity swaps: {swaps}"
        + ("  (lifecycle on)" if a.lifecycle else "  (lifecycle off)"))
  if sam is not None:
    print(sam.summary())
    print("sam timing (this machine, inside a sim loop -- NOT a deployment "
          f"latency): {json.dumps(sam_pred.report()['timing'])}")
  print("fallbacks used (hold-over, grasp-site rebuild): "
        + "  ".join(f"{k} ({v.holdovers}, {v.rebuilds})" for k, v in fb.items()))
  out["fallbacks"] = {k: {"holdover": v.holdovers, "rebuild": v.rebuilds}
                      for k, v in fb.items()}

  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")
  if a.html:
    write_html(a.html, rows, out, a)
    print(f"wrote {a.html}")
  return 0


def write_html(path, rows, summary, a) -> None:
  from html_frames import render                                # noqa: E402
  render(path, rows, summary, vars(a))


if __name__ == "__main__":
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
  raise SystemExit(main())
