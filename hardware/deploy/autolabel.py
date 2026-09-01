"""Turn recorded sessions into a YOLO segmentation dataset, with no hand labels.

The depth segmenter already solves the problem most of the time.  What it
cannot do is work where there is no depth, and the bench measured exactly how
often that is: on a blank white surface the D405 fills 88% of pixels on average
and 42% in the worst shot, and almost every silhouette is holed.  The colour
image is unaffected by any of that.

So the good frames pay for the bad ones.  Run the depth segmenter over a
recording, keep only the frames where it was confident, and write its output as
ground truth for a model that reads colour.  Nobody labels anything.

What "confident" means here is the whole design, because a dataset labelled by
an unreliable teacher teaches unreliability:

* the instance has to have been **confirmed by the tracker** -- seen in three
  of the last five frames -- so the phantoms the noise creates are excluded by
  the same mechanism that stops the robot reaching for them;
* the frame's **fill rate** has to be high, because a frame where the depth
  segmenter is already struggling is a frame where its labels are guesses, and
  those are precisely the frames the trained model is supposed to fix;
* the instance's own **pixels** have to be mostly valid, for the same reason at
  the level of the object rather than the frame.

The result is a model trained only on the easy cases and asked to generalise to
the hard ones.  That is the right way round: the hard cases are hard because
the depth is missing, not because the object looks different.

    python -m hardware.deploy.autolabel recordings/2026-08-24 --out yolo/data
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import cv2
import numpy as np

from . import config, mask, proprio, rectify


def polygons(binary: np.ndarray, epsilon_px: float = 1.5) -> list[np.ndarray]:
  """Outlines of a mask, as YOLO wants them.

  YOLO's segmentation format is a polygon per instance, not a bitmap, so the
  mask has to be traced.  Holes are dropped -- ``RETR_EXTERNAL`` -- because the
  format has no way to express them and a hole traced as a second outline
  becomes a second object.
  """
  cnts, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
  out = []
  for c in cnts:
    if cv2.contourArea(c) < 30:
      continue
    a = cv2.approxPolyDP(c, epsilon_px, True).reshape(-1, 2)
    if a.shape[0] >= 3:
      out.append(a)
  return out


def label_session(session: pathlib.Path, rig: "config.Rig", out: pathlib.Path,
                  min_fill: float | None = None,
                  min_instance_fill: float = 0.80,
                  stride: int = 1) -> dict:
  meta = json.loads((session / "meta.json").read_text())
  reproj = rectify.Reprojector(rig)
  segmenter = mask.DepthSegmenter(rig, reproj)
  tracker = mask.TargetTracker()
  kin = proprio.Kinematics()

  if min_fill is None:
    # Absolute whole-image fill is a property of the camera pose as much as of
    # sensor confidence: this D455 view legitimately contains only about 65%
    # valid depth while the old close-range D405 threshold was 90%.  Select the
    # better two thirds of each recording as the teacher frames, then apply the
    # stricter per-instance validity gate below.
    fills = []
    for rec in meta:
      f = np.load(session / f"{rec['i']:06d}.npz")
      fills.append(float((f["depth"] > 0).mean()))
    min_fill = float(np.quantile(fills, 0.35)) if fills else 1.0
    print(f"{session}: adaptive frame fill threshold {min_fill:.3f} "
          f"(35th percentile)")

  kept, seen, reasons = [], 0, {"fill": 0, "unconfirmed": 0, "instance": 0,
                                "no gray": 0}
  for rec in meta:
    seen += 1
    f = np.load(session / f"{rec['i']:06d}.npz")
    depth = f["depth"].astype(np.float32) / 10000.0
    gray = f["gray"]
    if gray.ndim != 2 or gray.shape != depth.shape:
      reasons["no gray"] += 1
      continue

    kin.update(np.asarray(rec["joint_pos"], dtype=np.float64))
    seg = segmenter(depth, arm=kin.link_spheres())
    label = tracker.update(seg, kin.site_pos)

    if rec["i"] % stride:
      continue
    fill = float((depth > 0).mean())
    if fill < min_fill:
      reasons["fill"] += 1
      continue
    if not tracker.confirmed_labels:
      reasons["unconfirmed"] += 1
      continue

    # Every *confirmed* instance is labelled, not only the target, and not
    # every instance.  The class is "a thing on the table" -- which one is the
    # target is the tracker's job at run time and is not a property of the
    # image -- but an instance the tracker has not confirmed is most likely a
    # blob of correlated depth noise, and labelling those would teach the model
    # to hallucinate precisely what the confirmation exists to reject.  On a
    # one-object synthetic session this cut the labels from 8.3 per frame
    # to 1.4.
    polys = []
    for inst in seg.instances:
      if inst.label not in tracker.confirmed_labels:
        continue
      m = mask.full_mask(seg, inst.label, segmenter.decimate) > 0
      if float((depth[m] > 0).mean()) < min_instance_fill:
        reasons["instance"] += 1
        continue
      polys.extend(polygons(m))
    if not polys:
      continue
    kept.append((session, rec["i"], gray, polys))
    del label

  return {"kept": kept, "seen": seen, "reasons": reasons}


def write_dataset(items, out: pathlib.Path, val_fraction: float = 0.15,
                  seed: int = 0) -> None:
  rng = random.Random(seed)
  sessions: dict[str, list[int]] = {}
  for i, (session, *_rest) in enumerate(items):
    sessions.setdefault(str(session.resolve()), []).append(i)
  split = {}
  if len(sessions) >= 2:
    # Adjacent video frames are near duplicates.  A random frame split leaks
    # the same scene into train and validation and reports a flattering mAP.
    # Hold out complete recording sessions instead.
    names = sorted(sessions)
    rng.shuffle(names)
    n_val_sessions = min(len(names) - 1,
                         max(1, int(round(len(names) * val_fraction))))
    val_sessions = set(names[:n_val_sessions])
    for name, indices in sessions.items():
      for i in indices:
        split[i] = "val" if name in val_sessions else "train"
  else:
    # One session cannot test environmental generalisation, but a contiguous
    # time block at least avoids putting adjacent frames on both sides.
    indices = next(iter(sessions.values()))
    indices = sorted(indices, key=lambda i: int(items[i][1]))
    n_val = min(len(indices) - 1, max(1, int(round(len(indices) * val_fraction))))
    val = set(indices[-n_val:])
    split = {i: ("val" if i in val else "train") for i in indices}

  for sub in ("train", "val"):
    (out / "images" / sub).mkdir(parents=True, exist_ok=True)
    (out / "labels" / sub).mkdir(parents=True, exist_ok=True)

  for k, (session, i, gray, polys) in enumerate(items):
    sub = split[k]
    stem = f"{session.name}_{i:06d}"
    cv2.imwrite(str(out / "images" / sub / f"{stem}.png"), gray)
    h, w = gray.shape
    lines = []
    for p in polys:
      flat = (p.astype(np.float64) / np.array([w, h])).reshape(-1)
      lines.append("0 " + " ".join(f"{v:.6f}" for v in np.clip(flat, 0, 1)))
    (out / "labels" / sub / f"{stem}.txt").write_text("\n".join(lines) + "\n")

  (out / "data.yaml").write_text(
    f"path: {out.resolve()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: object\n"
    "# One class.  Everything on the table goes in the bin, so there is nothing\n"
    "# to recognise -- the model is here to find things where the depth cannot,\n"
    "# not to tell them apart.\n"
  )
  n_train = sum(v == "train" for v in split.values())
  print(f"split by recording session: {n_train} train, "
        f"{len(items) - n_train} val frame(s) from {len(sessions)} session(s)")


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("sessions", nargs="+", help="directories written by run.py --record")
  p.add_argument("--out", default=str(
    pathlib.Path(__file__).resolve().parent / "yolo" / "data"))
  p.add_argument("--rig", default=None,
                 help="the rig file the session was recorded under.  Defaults "
                      "to the current one, which is right for a session from "
                      "this rig and wrong for an older one -- the extrinsic "
                      "decides where every point lands, so labelling an old "
                      "session with a new calibration is labelling a different "
                      "scene.  run.py --record should be pointed at a "
                      "directory that keeps its own copy.")
  p.add_argument("--min-fill", type=float, default=None,
                 help="absolute whole-frame depth fill; default adapts to the "
                      "35th percentile of each session")
  p.add_argument("--min-instance-fill", type=float, default=0.80)
  p.add_argument("--stride", type=int, default=2,
                 help="frames at 50 Hz are nearly duplicates; every other one "
                      "is already more than the model needs")
  a = p.parse_args()

  session_rigs = [pathlib.Path(x) / "rig.json" for x in a.sessions]
  if a.rig:
    rig = config.Rig.load(a.rig)
  elif session_rigs[0].exists():
    rig = config.Rig.load(session_rigs[0])
    print(f"using the rig recorded beside the session: {session_rigs[0]}")
  else:
    rig = config.Rig.load()
  out = pathlib.Path(a.out)
  items, seen, reasons = [], 0, {}
  for s in a.sessions:
    r = label_session(pathlib.Path(s), rig, out, a.min_fill,
                      a.min_instance_fill, a.stride)
    items.extend(r["kept"])
    seen += r["seen"]
    for k, v in r["reasons"].items():
      reasons[k] = reasons.get(k, 0) + v

  if not items:
    print(f"nothing survived out of {seen} frames: {reasons}")
    return 1
  write_dataset(items, out)
  print(f"{len(items)} labelled frames from {seen} seen; rejected {reasons}")
  print(f"wrote {out}/data.yaml")
  return 0


if __name__ == "__main__":
  sys.exit(main())
