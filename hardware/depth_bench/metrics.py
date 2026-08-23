"""What the bench measures, and why each number is there.

The target sheet is planar, so once its pose is known every pixel inside it has
a ground truth depth that is exact up to the pose fit.  That turns a depth image
into an error image, and the error image is what the three cameras get compared
on.

The plane comes from the ChArUco pose rather than from fitting a plane to the
depth itself.  Fitting the depth would absorb exactly the thing we want to see:
a camera with a scale error or a constant offset produces a beautifully flat
self-fitted plane and a wrong distance.  ``bias`` below only means something
because the reference is independent of the depth.

Five numbers per region, and each is here because it maps onto something the
policy or the simulator does:

  ``fill``          fraction of pixels with any measurement.  ``camera.py``
                    calls this ``1 - DEPTH_DROPOUT`` and currently guesses 0.98
                    uniformly over the image.
  ``stable_fill``   fraction valid in *every* frame of a static scene.  A pixel
                    that flickers is worse for a recurrent policy at 50 Hz than
                    one that is honestly always missing, and ``fill`` alone
                    cannot tell them apart.
  ``bias``          median signed error against the ChArUco plane.  Systematic,
                    survives averaging, and is what a wrong baseline or a wrong
                    depth unit looks like.
  ``spatial_rms``   RMS error after removing that bias: the flatness of a flat
                    thing.  This is the closest analogue to ``DEPTH_NOISE_M``.
  ``temporal_std``  per-pixel standard deviation over the frame stack, median
                    over pixels.  Distinct from ``spatial_rms``: fixed-pattern
                    error is invisible here and dominant there.

``p95_abs`` is carried alongside because a mean sim-to-real story that ignores
the tail is how a policy meets an edge it never saw.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_target(path: Path) -> tuple[cv2.aruco.CharucoBoard, dict]:
  spec = json.loads(Path(path).read_text())
  dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
  board = cv2.aruco.CharucoBoard(
    (spec["squares_x"], spec["squares_y"]),
    spec["square_m"], spec["marker_m"], dictionary,
  )
  return board, spec


def detect_pose(gray, K, dist, board) -> dict:
  """Locate the sheet.  Returns the pose and the evidence for trusting it."""
  detector = cv2.aruco.CharucoDetector(board)
  corners, ids, _, _ = detector.detectBoard(gray)
  if ids is None or len(ids) < 6:
    raise RuntimeError(
      f"only {0 if ids is None else len(ids)} ChArUco corners found. "
      "Is the whole sheet in frame, lit, and not glossy?"
    )
  obj, img = board.matchImagePoints(corners, ids)
  ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
  if not ok:
    raise RuntimeError("solvePnP failed on the ChArUco corners")
  proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
  reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2,
                                        axis=1))))
  R, _ = cv2.Rodrigues(rvec)
  normal = R[:, 2] / np.linalg.norm(R[:, 2])
  # Angle between the sheet normal and the optical axis: stereo noise grows with
  # obliquity, so a comparison across cameras is only fair at similar tilt.
  tilt = float(np.degrees(np.arccos(min(1.0, abs(float(normal @ np.array([0, 0, 1.0])))))))
  return {
    "rvec": rvec, "tvec": tvec, "R": R, "normal": normal,
    "n_corners": int(len(ids)), "reproj_rms_px": reproj, "tilt_deg": tilt,
  }


def plane_depth(K, pose, shape) -> np.ndarray:
  """Ground-truth Z of the sheet's plane at every pixel.

  Depth images store Z along the optical axis, not radial distance, so the ray
  is normalised to z = 1 and the intersection parameter *is* the depth.
  """
  h, w = shape
  u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
  d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
  n = pose["normal"]
  num = float(n @ pose["tvec"].ravel())
  den = d @ n
  with np.errstate(divide="ignore", invalid="ignore"):
    z = np.where(np.abs(den) > 1e-9, num / den, np.nan)
  return z


def region_mask(K, dist, pose, region: dict, shape) -> np.ndarray:
  """Pixels covered by a rectangle of the sheet, in the sheet's own frame."""
  x0, x1 = region["x_min_m"], region["x_max_m"]
  y0, y1 = region["y_min_m"], region["y_max_m"]
  quad = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
  proj, _ = cv2.projectPoints(quad, pose["rvec"], pose["tvec"], K, dist)
  mask = np.zeros(shape, np.uint8)
  cv2.fillConvexPoly(mask, np.round(proj.reshape(-1, 2)).astype(np.int32), 1)
  return mask.astype(bool)


def region_metrics(depth: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                   outlier_m: float = 0.05) -> dict:
  """Reduce a frame stack over one region to the numbers described above.

  ``outlier_m`` discards samples further than 5 cm from the sheet.  Those are
  not sensor noise: they are the background showing through a hole, or a stereo
  mismatch onto a repeated pattern, and averaging them in would report a metre
  of "noise" for a camera whose valid pixels are excellent.  They are not
  swept under the rug either -- what they are is *not filled*, and ``fill``
  counts them as missing.
  """
  n_pix = int(mask.sum())
  if n_pix == 0:
    return {"n_pixels": 0}

  valid = (depth > 0) & mask & np.isfinite(gt)
  err = np.where(valid, depth - gt, np.nan)
  inlier = valid & (np.abs(np.nan_to_num(err, nan=1e9)) < outlier_m)

  fill = float(inlier.sum()) / (n_pix * depth.shape[0])
  gross = float((valid & ~inlier).sum()) / (n_pix * depth.shape[0])
  stable = inlier.all(axis=0) & mask
  stable_fill = float(stable.sum()) / n_pix

  e = err[inlier]
  bias = float(np.median(e)) if e.size else float("nan")
  spatial_rms = float(np.sqrt(np.mean((e - bias) ** 2))) if e.size else float("nan")
  p95 = float(np.percentile(np.abs(e), 95)) if e.size else float("nan")

  if stable.sum() >= 16 and depth.shape[0] >= 3:
    per_pix = np.nanstd(np.where(inlier, depth, np.nan), axis=0)
    temporal = float(np.median(per_pix[stable]))
  else:
    temporal = float("nan")

  return {
    "n_pixels": n_pix,
    "range_m": float(np.nanmean(gt[mask])),
    "fill": fill,
    "gross_outlier_frac": gross,
    "stable_fill": stable_fill,
    "bias_m": bias,
    "spatial_rms_m": spatial_rms,
    "temporal_std_m": temporal,
    "p95_abs_m": p95,
  }


def evaluate(cap, board, spec, outlier_m: float = 0.05) -> dict:
  pose = detect_pose(cap.gray, cap.K, cap.dist, board)
  gt = plane_depth(cap.K, pose, cap.gray.shape)
  out = {
    "pose": {
      "distance_m": float(np.linalg.norm(pose["tvec"])),
      "n_corners": pose["n_corners"],
      "reproj_rms_px": pose["reproj_rms_px"],
      "tilt_deg": pose["tilt_deg"],
    },
    "regions": {},
  }
  for name, region in spec["regions"].items():
    mask = region_mask(cap.K, cap.dist, pose, region, cap.gray.shape)
    out["regions"][name] = region_metrics(cap.depth, gt, mask, outlier_m)
  return out
