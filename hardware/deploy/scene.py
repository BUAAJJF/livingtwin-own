"""What is on the table right now, and does it match a run you want to repeat?

Two runs of this deployment with byte-identical arguments produced very
different results, and the reason was not in ``run.json``: one had two objects
on the table and the other had three, of a different size. Nothing recorded
that, and nothing could be repeated without it.

So this reads the scene back the way the policy sees it -- same camera, same
segmenter, same tracker -- and prints where the objects are in the robot's base
frame. Given a previous recording it also matches what it sees against what
that recording started with, and says which object to move and by how much.

The arm is read only. CAN is opened to read joint angles, because the segmenter
subtracts the arm from the cloud and needs to know where it is; the drives are
never enabled and nothing is ever commanded.

    # what is on the table
    python -m hardware.deploy.scene

    # place it back the way a recording started
    python -m hardware.deploy.scene --like recordings/v4_stereo_try3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np

from . import config, mask, proprio, rectify


def instances_now(reader, rig, kin, seg, frames: int):
  """The most-populated segmentation of several frames.

  Most-populated rather than the last: the tracker needs a few frames to
  confirm anything, and a single frame that happened to drop one object would
  silently change the answer this is asked to compare against.
  """
  best: list = []
  for _ in range(frames):
    frame = reader.latest()
    if frame is None:
      continue
    out = seg(frame.depth, rgb=frame.gray, arm=kin.link_spheres())
    if len(out.instances) > len(best):
      best = list(out.instances)
  return best


def instances_from_recording(session: pathlib.Path, frames: int = 12):
  """What a recorded session started with, segmented from its own frames."""
  rigf = session / "rig.json"
  rig = config.Rig.load(rigf if rigf.exists() else config.RIG_FILE)
  reproj = rectify.Reprojector(rig, device="cpu")
  seg = mask.DepthSegmenter(rig, reproj)
  kin = proprio.Kinematics()
  meta = {m["i"]: m for m in json.loads((session / "meta.json").read_text())
          if "i" in m and "joint_pos" in m}
  best: list = []
  for f in sorted(glob.glob(str(session / "0*.npz")))[:frames]:
    m = meta.get(int(os.path.basename(f).split(".")[0]))
    if m is None:
      continue
    q = np.asarray(m["joint_pos"], dtype=np.float64)
    g = float(q[6]) if q.size > 6 else 0.05
    kin.update(np.array([*q[:6], g, -g]))
    z = np.load(f)
    out = seg(z["depth"].astype(np.float32) / 10000.0, rgb=z["gray"],
              arm=kin.link_spheres())
    if len(out.instances) > len(best):
      best = list(out.instances)
  return best


def _describe(items) -> list[str]:
  rows = []
  for i in sorted(items, key=lambda x: -x.n_px):
    c = np.asarray(i.centroid_base) * 1000.0
    rows.append(f"    [{c[0]:+7.1f}, {c[1]:+7.1f}] mm   "
                f"top {1000 * float(i.top_z):5.1f} mm   {int(i.n_px):5d} px")
  return rows


def _match(now, want, tol_m: float):
  """Pair what is there with what is wanted, nearest first.

  Greedy on distance rather than optimal assignment: with three or four
  objects the two agree, and a reader can check a greedy pairing by eye.
  """
  now = sorted(now, key=lambda x: -x.n_px)
  want = sorted(want, key=lambda x: -x.n_px)
  used, pairs = set(), []
  for w in want:
    wc = np.asarray(w.centroid_base)
    best, bd = None, None
    for k, n in enumerate(now):
      if k in used:
        continue
      d = float(np.linalg.norm(np.asarray(n.centroid_base)[:2] - wc[:2]))
      if bd is None or d < bd:
        best, bd = k, d
    if best is None:
      pairs.append((w, None, None))
    else:
      used.add(best)
      pairs.append((w, now[best], bd))
  extra = [n for k, n in enumerate(now) if k not in used]
  return pairs, extra


def main() -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--camera", choices=("d405", "d455"), default="d455")
  p.add_argument("--rig-file", default=None)
  p.add_argument("--serial", default=None)
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--frames", type=int, default=12)
  p.add_argument("--like", default=None,
                 help="a recording whose opening scene should be matched")
  p.add_argument("--tol-mm", type=float, default=20.0,
                 help="how close counts as placed")
  p.add_argument("--json", default=None)
  a = p.parse_args()

  want = None
  if a.like:
    want = instances_from_recording(pathlib.Path(a.like), a.frames)
    if not want:
      print(f"no objects found in the opening frames of {a.like}",
            file=sys.stderr)
      return 2

  suffix = "" if a.camera == "d405" else f"_{a.camera}"
  rig_path = (pathlib.Path(a.rig_file) if a.rig_file
              else pathlib.Path(config.RIG_FILE).with_name(f"rig{suffix}.json"))
  rig = config.Rig.load(rig_path)

  from . import robot, sensor
  from .run import _wait_for_feedback

  print("The arm is read only -- the drives are not enabled and nothing is "
        "commanded.")
  reader = sensor.Reader(serial=a.serial or rig.serial, backend=a.camera)
  arm = robot.PiperArm(a.can)
  try:
    reader.wait_for_first()
    if rig.serial and reader.serial and str(rig.serial) != str(reader.serial):
      raise SystemExit(f"{rig_path} belongs to camera {rig.serial}, "
                       f"connected {reader.serial}")
    rig.K = reader.K
    arm.connect()
    st = _wait_for_feedback(arm)
    kin = proprio.Kinematics()
    kin.update(np.array([*st.q, st.gripper, -st.gripper]))
    seg = mask.DepthSegmenter(rig, rectify.Reprojector(rig, device="cpu"))
    now = instances_now(reader, rig, kin, seg, a.frames)
  finally:
    try:
      arm.disconnect()
    except Exception:
      pass
    reader.close()

  print(f"\non the table now: {len(now)} object(s)")
  for line in _describe(now):
    print(line)

  ok = True
  if want is not None:
    print(f"\n{a.like} started with: {len(want)} object(s)")
    for line in _describe(want):
      print(line)
    tol = a.tol_mm / 1000.0
    pairs, extra = _match(now, want, tol)
    print(f"\nto match it (tolerance {a.tol_mm:.0f} mm):")
    for w, n, d in pairs:
      wc = np.asarray(w.centroid_base) * 1000.0
      if n is None:
        print(f"    ADD an object at [{wc[0]:+7.1f}, {wc[1]:+7.1f}] mm")
        ok = False
        continue
      nc = np.asarray(n.centroid_base) * 1000.0
      if d is not None and d <= tol:
        print(f"    ok    [{nc[0]:+7.1f}, {nc[1]:+7.1f}] is within "
              f"{1000 * d:.0f} mm of [{wc[0]:+7.1f}, {wc[1]:+7.1f}]")
      else:
        print(f"    MOVE  [{nc[0]:+7.1f}, {nc[1]:+7.1f}] by "
              f"[{wc[0] - nc[0]:+6.1f}, {wc[1] - nc[1]:+6.1f}] mm  "
              f"-> [{wc[0]:+7.1f}, {wc[1]:+7.1f}]")
        ok = False
      # Size is part of the scene and it is not adjustable by moving anything,
      # so it is reported rather than instructed.
      if w.n_px and abs(int(n.n_px) - int(w.n_px)) > 0.4 * int(w.n_px):
        print(f"          note: {int(n.n_px)} px against {int(w.n_px)} px -- "
              f"a different object, not a different place")
        ok = False
    for n in extra:
      nc = np.asarray(n.centroid_base) * 1000.0
      print(f"    REMOVE the object at [{nc[0]:+7.1f}, {nc[1]:+7.1f}] mm")
      ok = False
    print(f"\n{'scene matches' if ok else 'scene does NOT match yet'}")

  if a.json:
    pathlib.Path(a.json).write_text(json.dumps({
      "now": [{"centroid_base_m": list(map(float, i.centroid_base)),
               "top_z_m": float(i.top_z), "n_px": int(i.n_px)} for i in now],
      "like": a.like,
      "matches": bool(ok) if want is not None else None,
    }, indent=2))
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
