#!/usr/bin/env python3
"""Calibrate the Odin 1's colour-to-lidar extrinsic, once, against the target.

    python calibrate_odin1.py

Hold the printed sheet in front of the sensor and move it: nearer, further,
and above all *tilted differently* each time.  The script captures a pose
whenever the view is new and still, and stops when it has enough.

Why this exists: the factory ``Tcl`` in ``calib.yaml`` is written against a
frame the vendor does not define, and no composition of it with an axis
permutation gets the board's plane closer than eight degrees to where the lidar
sees it.  Eight degrees at 0.7 m is centimetres, and centimetres is the whole
question.

**The reason tilt matters.**  The parameters solved for here are a rotation, a
translation, *and a constant depth bias*, because a translation along the line
of sight and a depth bias produce the identical residual on a sheet held
square-on.  They separate only when the sheet is seen at different angles: the
error from a translation ``t`` on a plane with normal ``n`` scales as ``n.t``
and changes with the tilt, and a bias does not.  Calibrate this on four
fronto-parallel views and the fit will quietly move the sensor's bias into the
extrinsic, after which the bench measures a bias of zero on a camera that has
one.  The script refuses to write a result whose tilt spread is too small to
tell them apart.

The fitted bias is *not* applied to anything -- it is reported, and it is a
property of the sensor.  Only the rotation and translation are stored.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics as M
from capture import odin1

MIN_TILT_SPREAD_DEG = 25.0
"""Below this the bias and the along-axis translation are not separable, and a
result written anyway would look like a successful calibration."""


def collect(args, board, spec) -> tuple[list, object]:
  stream = odin1.Stream(args)
  shots, prev_small, still = [], None, 0
  print(f"\n  collecting {args.poses} poses — move the sheet between each one, "
        f"and change its TILT, not just its distance\n")
  t_end = time.time() + args.timeout
  while len(shots) < args.poses and time.time() < t_end:
    depth, gray = stream.read()
    small = cv2.resize(gray, None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    motion = (float(np.mean(np.abs(small - prev_small)))
              if prev_small is not None else 1e9)
    prev_small = small
    still = still + 1 if motion < args.still else 0
    if still < 8:
      continue
    try:
      pose = M.detect_pose(gray, stream.K, stream.dist, board)
    except RuntimeError:
      continue
    new = all(
      np.linalg.norm(pose["tvec"].ravel() - s["pose"]["tvec"].ravel()) > 0.05 or
      np.degrees(np.arccos(np.clip(
        (np.trace(s["pose"]["R"].T @ pose["R"]) - 1) / 2, -1, 1))) > 12.0
      for s in shots)
    if not new:
      continue
    frames = [depth] + [stream.read()[0] for _ in range(7)]
    shots.append({"pose": pose, "depth": np.median(np.stack(frames), axis=0)})
    still = 0
    print(f"    {len(shots)}/{args.poses}  "
          f"{np.linalg.norm(pose['tvec']) * 1000:4.0f} mm  "
          f"tilt {pose['tilt_deg']:4.1f} deg  "
          f"reproj {pose['reproj_rms_px']:.2f} px")
  return shots, stream


def solve(shots, rays, T0, spec, region="charuco"):
  """Least squares for (rotation, translation, bias) over every captured pose."""

  def unpack(x):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(x[:3])[0]
    T[:3, 3] = x[3:6]
    return T, x[6]

  def gather(T, bias, sel=None):
    res, sets = [], []
    for j, s in enumerate(shots):
      pose = M.transform_pose(s["pose"], T)
      _, pb = M.board_coords(rays, pose)
      mask = M.region_mask(pb, spec["regions"][region]) & (s["depth"] > 0.05)
      if sel is not None:
        mask &= sel[j]
      pts = (rays * (s["depth"] - bias)[..., None])[mask]
      if len(pts) < 12:
        sets.append(mask)
        continue
      n = pose["normal"]
      res.append((pts - pose["tvec"].ravel()) @ n)
      sets.append(mask)
    return (np.concatenate(res) if res else np.zeros(1)), sets

  x0 = np.concatenate([cv2.Rodrigues(T0[:3, :3])[0].ravel(), T0[:3, 3], [0.0]])
  sel = None
  for _ in range(2):  # re-select the pixels once the pose has improved
    T, b = unpack(x0)
    _, sets = gather(T, b, sel)
    # keep pixels that are plausibly on the sheet, so a mask that is a few
    # pixels off does not drag the desk into the fit
    keep = []
    for j, s in enumerate(shots):
      pose = M.transform_pose(s["pose"], T)
      gt = M.plane_depth(rays, pose)
      keep.append(sets[j] & (np.abs(np.nan_to_num(s["depth"] - gt, nan=9)) < 0.06))
    sel = keep
    out = least_squares(lambda x: gather(*unpack(x), sel)[0], x0,
                        loss="soft_l1", f_scale=0.005, max_nfev=200)
    x0 = out.x
  T, bias = unpack(x0)
  r = gather(T, bias, sel)[0]
  return T, bias, float(np.sqrt(np.mean(r**2))), int(r.size)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--poses", type=int, default=8)
  ap.add_argument("--still", type=float, default=3.0)
  ap.add_argument("--timeout", type=float, default=300.0)
  ap.add_argument("--target", type=Path, default=HERE / "targets" / "target_a4.json")
  ap.add_argument("--force", action="store_true",
                  help="write the result even if the tilt spread cannot separate "
                       "bias from translation (it will be wrong; don't)")
  odin1.add_args(ap)
  args = ap.parse_args()

  board, spec = M.load_target(args.target)
  shots, stream = collect(args, board, spec)
  try:
    if len(shots) < 4:
      raise SystemExit(f"only {len(shots)} poses; need at least 4")
    tilts = np.array([s["pose"]["tilt_deg"] for s in shots])
    dists = np.array([float(np.linalg.norm(s["pose"]["tvec"])) for s in shots])
    spread = float(tilts.max() - tilts.min())
    print(f"\n  {len(shots)} poses: {dists.min()*1000:.0f}–{dists.max()*1000:.0f} mm, "
          f"tilt {tilts.min():.0f}–{tilts.max():.0f} deg (spread {spread:.0f} deg)")

    T0 = stream.T_dg
    T, bias, rms, n = solve(shots, stream.rays, T0, spec)
    dR = np.degrees(np.arccos(np.clip(
      (np.trace(T0[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
    print(f"\n  solved on {n} points")
    print(f"    rotation moved   {dR:.2f} deg from the factory value")
    print(f"    translation      {np.round(T[:3, 3] * 1000, 2)} mm "
          f"(was {np.round(T0[:3, 3] * 1000, 2)})")
    print(f"    sensor bias      {bias * 1000:+.2f} mm   <- a property of the "
          f"sensor, not applied to anything")
    print(f"    residual rms     {rms * 1000:.2f} mm")

    if spread < MIN_TILT_SPREAD_DEG and not args.force:
      raise SystemExit(
        f"\n  NOT WRITTEN: tilt spread {spread:.0f} deg is under "
        f"{MIN_TILT_SPREAD_DEG:.0f}, so the bias and the along-axis translation "
        "are not separable and this fit has probably absorbed one into the "
        "other. Re-run and tilt the sheet more.")
    np.savez(odin1.EXTRINSIC_CACHE, T_dg=T, bias_m=bias, residual_rms_m=rms,
             n_poses=len(shots), tilt_spread_deg=spread)
    print(f"\n  -> {odin1.EXTRINSIC_CACHE}")
    print("     bias is now measurable on this sensor; re-run measure.py.")
  finally:
    stream.close()


if __name__ == "__main__":
  main()
