#!/usr/bin/env python3
"""Find the Odin 1's colour-to-lidar rotation by watching something move.

    python odin1_align.py          # then wave a hand slowly around the view

The factory extrinsic is unusable as shipped and every indirect test of it has
been ambiguous: on a board lying flat on a desk, a 180 degree error puts the
region somewhere else *on the same plane*, so a plane-normal test cannot see
it; the infrared intensity image saturates, so an albedo test cannot see it
either.  Both agreed with the wrong answer.

A moving hand does not have that problem.  It appears at one place in the
colour image and at one place in the depth image, and those two places are the
same physical direction seen through two different lenses.  Each wave is a
correspondence between a colour ray and a lidar ray, and enough of them
spread over the field of view determine the rotation outright -- Wahba's
problem, solved in closed form.  Nothing here is a guess between 24 options.

Wave slowly and cover the whole view: corners as well as the middle.  Pairs
are only accepted when the motion is a single compact blob in both images at
once, so a hand crossing the frame contributes and someone walking past behind
does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capture import odin1


def blob(diff: np.ndarray, min_frac: float = 0.35) -> tuple[float, float] | None:
  """Centroid of the dominant moving region, or None if it is not dominant.

  The compactness test is what keeps this honest: a global brightness change,
  an exposure step or a person walking behind the target all move many pixels
  in many places, and averaging those gives a centroid near the middle of the
  frame every time -- which would look like a perfectly consistent
  correspondence and quietly fit the identity rotation.
  """
  d = cv2.GaussianBlur(diff, (0, 0), 2.0)
  hi = np.percentile(d, 99.5)
  if hi <= 1e-6:
    return None
  mask = (d > 0.5 * hi).astype(np.uint8)
  n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
  if n <= 1:
    return None
  areas = stats[1:, cv2.CC_STAT_AREA]
  k = int(np.argmax(areas)) + 1
  if areas.sum() == 0 or areas.max() / areas.sum() < min_frac:
    return None
  if areas.max() > 0.25 * mask.size:  # the whole frame changed: not a hand
    return None
  m = lab == k
  w = d * m
  tot = w.sum()
  if tot <= 1e-6:
    return None
  ys, xs = np.nonzero(m)
  return (float((w[ys, xs] * xs).sum() / tot), float((w[ys, xs] * ys).sum() / tot))


def wahba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Rotation R minimising the angle between R @ a_i and b_i."""
  H = b.T @ a
  U, _, Vt = np.linalg.svd(H)
  d = np.sign(np.linalg.det(U @ Vt))
  return U @ np.diag([1.0, 1.0, d]) @ Vt


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--pairs", type=int, default=120)
  ap.add_argument("--timeout", type=float, default=240.0)
  ap.add_argument("--write", action="store_true",
                  help="store the rotation (with the factory translation) as the "
                       "extrinsic; run calibrate_odin1.py afterwards to refine it")
  odin1.add_args(ap)
  args = ap.parse_args()

  s = odin1.Stream(args)
  print("\n  wave a hand slowly across the whole field of view — corners too\n")
  A, B, prev_g, prev_d = [], [], None, None
  import time
  t_end = time.time() + args.timeout
  try:
    while len(A) < args.pairs and time.time() < t_end:
      depth, gray = s.read()
      g = cv2.resize(gray, (320, 256)).astype(np.float32)
      d = np.where(depth > 0.05, depth, np.nan).astype(np.float32)
      if prev_g is not None:
        cg = blob(np.abs(g - prev_g))
        dd = np.abs(np.nan_to_num(d - prev_d, nan=0.0))
        cd = blob(dd)
        if cg is not None and cd is not None:
          # colour pixel -> ray, using the pinhole the colour was rectified to
          sx = gray.shape[1] / 320.0
          sy = gray.shape[0] / 256.0
          u, v = cg[0] * sx, cg[1] * sy
          ray_c = np.array([(u - s.K[0, 2]) / s.K[0, 0],
                            (v - s.K[1, 2]) / s.K[1, 1], 1.0])
          # depth pixel -> ray, straight out of the measured table
          ray_d = s.rays[int(round(cd[1])), int(round(cd[0]))]
          A.append(ray_c / np.linalg.norm(ray_c))
          B.append(ray_d / np.linalg.norm(ray_d))
          if len(A) % 10 == 0:
            print(f"    {len(A)}/{args.pairs} pairs")
      prev_g, prev_d = g, d
  finally:
    pass

  if len(A) < 20:
    s.close()
    raise SystemExit(f"only {len(A)} usable pairs; wave more, and more slowly")

  A, B = np.array(A), np.array(B)
  R = wahba(A, B)
  ang = np.degrees(np.arccos(np.clip(np.sum((A @ R.T) * B, axis=1), -1, 1)))
  keep = ang < np.percentile(ang, 80)      # drop the mismatched pairs, refit
  R = wahba(A[keep], B[keep])
  ang = np.degrees(np.arccos(np.clip(np.sum((A[keep] @ R.T) * B[keep], axis=1), -1, 1)))

  print(f"\n  {len(A)} pairs, {keep.sum()} kept")
  print(f"  residual angle: median {np.median(ang):.2f} deg, p90 {np.percentile(ang, 90):.2f} deg")
  print(f"\n  R (colour -> dtof) =\n{np.round(R, 5)}")
  for name, C in (("identity", np.eye(3)), ("180 deg about z", np.diag([-1.0, -1.0, 1.0]))):
    a = np.degrees(np.arccos(np.clip((np.trace(C.T @ R) - 1) / 2, -1, 1)))
    print(f"    {a:7.2f} deg away from {name}")

  if np.median(ang) > 8.0:
    print("\n  residual too large to trust — the pairs were probably not a single "
          "moving object. Re-run and wave one hand, slowly, nothing else moving.")
  elif args.write:
    T = s.T_dg.copy()
    T[:3, :3] = R
    np.savez(odin1.EXTRINSIC_CACHE, T_dg=T, bias_m=0.0,
             residual_rms_m=float("nan"), n_poses=0, tilt_spread_deg=0.0,
             source="odin1_align rotation only")
    print(f"\n  -> {odin1.EXTRINSIC_CACHE} (rotation only)")
    print("     bias stays untrustworthy until calibrate_odin1.py has run.")
  else:
    print("\n  re-run with --write to store it.")
  s.close()


if __name__ == "__main__":
  main()
