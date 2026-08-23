#!/usr/bin/env python3
"""Calibrate the Odin 1's colour-to-lidar extrinsic, once, against the target.

    python calibrate_odin1.py

**Hold the printed sheet up in the air, not flat on the desk**, and move it
between captures: nearer, further, left, right, and above all *tilted
differently* each time.  The script captures a pose whenever the view is new
and still, and stops when it has enough.

Why in the air: with the sheet lying flat on a desk, an error in the transform
puts the region somewhere else *on the same desk*, at a similar distance and
with an identical surface normal, so neither a plane test nor a depth test can
see it.  Held up at several distances, the board is the only thing at its
distance and there is nowhere for an error to hide.

The starting point is the vendor's own composition -- ``Tcl`` from calib.yaml
with the raw-to-lidar mapping from ``rawCloudRender.cpp`` -- which is an answer
rather than a guess.  What is left for this script is the couple of centimetres
it still leaves against the ChArUco plane, and splitting that into extrinsic
residual and sensor bias, which is the whole reason the two are solved
together.

**The reason tilt matters.**  The parameters solved for are a rotation, a
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
  print(f"\n  collecting {args.poses} poses — hold the sheet UP IN THE AIR, move it "
        f"between each one,\n  and change its TILT, not just its distance\n")
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
  """Least squares for (rotation correction, translation, bias) over all poses.

  The pixel set is chosen once per round and held fixed while the optimiser
  runs.  It has to be: the region a candidate transform selects changes with
  the candidate, so recomputing it inside the objective hands ``least_squares``
  a residual vector whose length moves between iterations, which it cannot
  take.  Two rounds, so the set is re-chosen once the pose has improved.
  """
  R0 = T0[:3, :3]

  def unpack(x):
    """A small proper rotation applied on top of the vendor's transform.

    Parameterised as a correction rather than as the matrix itself because the
    vendor's transform is a *reflection* -- det -1 -- and a rotation vector
    cannot represent one.  This keeps the determinant exact and solves for the
    couple of degrees actually in question.
    """
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(x[:3])[0] @ R0
    T[:3, 3] = x[3:6]
    return T, x[6]

  def select(T):
    """Pixels plausibly on the sheet under the current transform."""
    sets = []
    for s in shots:
      pose = M.transform_pose(s["pose"], T)
      gt, pb = M.board_coords(rays, pose)
      m = M.region_mask(pb, spec["regions"][region]) & (s["depth"] > 0.05)
      # keeps the desk out when the region lands a few pixels off the sheet
      m &= np.abs(np.nan_to_num(s["depth"] - gt, nan=9.0)) < 0.06
      sets.append(m)
    return sets

  def residual(x, sets):
    T, bias = unpack(x)
    out = []
    for s, m in zip(shots, sets):
      if not m.any():
        continue
      pose = M.transform_pose(s["pose"], T)
      pts = (rays * (s["depth"] - bias)[..., None])[m]
      out.append((pts - pose["tvec"].ravel()) @ pose["normal"])
    return np.concatenate(out) if out else np.zeros(1)

  x0 = np.concatenate([np.zeros(3), T0[:3, 3], [0.0]])
  sets = None
  for _ in range(2):
    sets = select(unpack(x0)[0])
    if sum(int(m.sum()) for m in sets) < 50:
      raise SystemExit("the region landed on too few lidar pixels to fit; "
                       "was the sheet in both cameras' view every time?")
    x0 = least_squares(lambda x: residual(x, sets), x0,
                       loss="soft_l1", f_scale=0.005, max_nfev=300).x
  T, bias = unpack(x0)
  r = residual(x0, sets)
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
    print(f"    rotation moved   {dR:.2f} deg from the vendor value")
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
