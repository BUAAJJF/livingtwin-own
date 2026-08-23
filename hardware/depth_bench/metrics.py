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


def _detector(board) -> cv2.aruco.CharucoDetector:
  """Subpixel corner refinement, which is not the default and is worth 0.2 mm.

  Measured on the synthetic capture in ``selftest.py``, where the true pose is
  known: plain detection puts the board 0.92 mm too far away, refined 0.71 mm.
  Almost all of the residual is along the optical axis -- it is a scale error,
  which is what a planar target seen at low pixel span always gives.
  """
  dp = cv2.aruco.DetectorParameters()
  dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
  cp = cv2.aruco.CharucoParameters()
  cp.tryRefineMarkers = True
  return cv2.aruco.CharucoDetector(board, cp, dp)


def detect_pose(gray, K, dist, board) -> dict:
  """Locate the sheet.  Returns the pose and the evidence for trusting it."""
  detector = _detector(board)
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
  # How much the plane itself could be wrong, in metres, which sets the floor
  # under any bias number computed against it.  A residual of `reproj` pixels
  # spread over a board spanning `span` pixels is a relative scale error of
  # reproj/span, and a scale error on a planar target lands almost entirely on
  # the distance.  Checked against the known-truth capture in selftest.py: this
  # predicts 0.5 mm where the actual pose error is 0.66 mm.
  pts = img.reshape(-1, 2)
  span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
  # Perpendicular distance from the camera to the plane, not the distance to
  # the board's origin corner: the corner is off-axis, and quoting it makes a
  # sheet placed at 250 mm report 288 mm.
  dist_m = abs(float(normal @ tvec.ravel()))
  return {
    "rvec": rvec, "tvec": tvec, "R": R, "normal": normal,
    "n_corners": int(len(ids)), "reproj_rms_px": reproj, "tilt_deg": tilt,
    "board_span_px": span,
    "plane_distance_m": dist_m,
    "plane_uncertainty_m": dist_m * reproj / span if span > 0 else float("nan"),
  }


def transform_pose(pose: dict, T: np.ndarray) -> dict:
  """Carry a pose from the camera it was found in into the depth frame.

  Needed because the two are not always the same camera.  On the Odin 1 the
  board is found in the 1600x1296 colour image -- in the 256x192 lidar image a
  25 mm marker is under two pixels across and no detector will read it -- and
  the pose then has to travel into the lidar frame across the factory
  extrinsic.  That extrinsic's error lands in ``bias`` and is *not* covered by
  ``plane_uncertainty_m``, which only knows about the fit in the image it saw.
  """
  R = T[:3, :3] @ pose["R"]
  t = (T[:3, :3] @ pose["tvec"].ravel() + T[:3, 3]).reshape(3, 1)
  normal = R[:, 2] / np.linalg.norm(R[:, 2])
  out = dict(pose)
  out["R"], out["tvec"] = R, t
  # The Odin 1's colour-to-depth transform is a reflection, so this can arrive
  # with det = -1.  Points, planes and normals all transform correctly through
  # it; a rotation vector does not exist for it.  Nothing downstream of the
  # depth frame needs one -- the region outline on the greyscale panel is drawn
  # with the *untransformed* pose -- so this is left absent rather than filled
  # with whatever Rodrigues returns for a matrix that is not a rotation.
  out["rvec"] = cv2.Rodrigues(R)[0] if np.linalg.det(R) > 0 else None
  out["normal"] = normal
  out["plane_distance_m"] = abs(float(normal @ t.ravel()))
  out["tilt_deg"] = float(np.degrees(np.arccos(
    min(1.0, abs(float(normal @ np.array([0, 0, 1.0])))))))
  return out


def plane_depth(rays: np.ndarray, pose: dict) -> np.ndarray:
  """Ground-truth depth of the sheet's plane along every pixel's ray.

  The rays are z-normalised, so the intersection parameter *is* the depth in
  the same sense the sensor reports it -- Z along the optical axis, not radial
  range.  Rays that run parallel to the sheet, or meet it behind the camera,
  come back NaN rather than as a very large number that would quietly pass a
  distance check.
  """
  n = pose["normal"]
  num = float(n @ pose["tvec"].ravel())
  den = rays @ n
  with np.errstate(divide="ignore", invalid="ignore"):
    z = np.where(np.abs(den) > 1e-9, num / den, np.nan)
  return np.where(z > 0, z, np.nan)


def board_coords(rays: np.ndarray, pose: dict) -> tuple[np.ndarray, np.ndarray]:
  """Where each pixel's ray lands on the sheet, in the sheet's own frame.

  Regions are selected here rather than by projecting their corners into the
  image, which is what this used to do.  Two reasons, and the second is why it
  had to change: intersecting the ray is exact where rasterising a projected
  quadrilateral is not, and projection needs a forward camera model that a
  lidar does not have.  A ray table is enough for this, and every sensor on the
  bench has one.
  """
  z = plane_depth(rays, pose)
  p_cam = rays * z[..., None]
  p_board = np.einsum("ij,hwj->hwi", pose["R"].T,
                      p_cam - pose["tvec"].reshape(1, 1, 3))
  return z, p_board


def region_mask(p_board: np.ndarray, region: dict) -> np.ndarray:
  """Pixels whose ray lands inside a rectangle of the sheet."""
  x, y = p_board[..., 0], p_board[..., 1]
  with np.errstate(invalid="ignore"):
    return ((x >= region["x_min_m"]) & (x <= region["x_max_m"]) &
            (y >= region["y_min_m"]) & (y <= region["y_max_m"]) &
            np.isfinite(x) & np.isfinite(y))


def fit_plane(rays: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> dict | None:
  """A plane through the sensor's *own* points inside a region.

  Only ever used as a fallback, and it costs exactly one number: a self-fitted
  plane absorbs any constant depth error, so ``bias`` against it is identically
  zero and meaningless.  Flatness, fill and temporal noise survive, because
  none of them is a statement about where the plane is.

  It exists for the Odin 1, whose colour-to-lidar extrinsic is not usable as
  the vendor ships it.  Rather than quote a bias measured across a transform
  known to be eight degrees out, the bench withholds that one number and keeps
  the rest.
  """
  P = (rays * depth[..., None])[mask & (depth > 0)]
  if len(P) < 12:
    return None
  w = np.ones(len(P))
  for _ in range(4):  # IRLS: a few outlying pixels must not tilt the reference
    c = np.average(P, axis=0, weights=w)
    _, _, Vt = np.linalg.svd((P - c) * w[:, None])
    n = Vt[2] / np.linalg.norm(Vt[2])
    r = (P - c) @ n
    sigma = 1.4826 * np.median(np.abs(r)) + 1e-9
    w = 1.0 / (1.0 + (r / (3 * sigma)) ** 2)
  return {"normal": n, "tvec": np.asarray(c).reshape(3, 1)}


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

  # Temporal noise is measured on pixels seen often enough to have a variance,
  # not on pixels seen every single time.  Requiring every frame sounds
  # stricter and is in fact useless exactly where it matters: at 55% dropout,
  # no pixel out of 24 frames survives, and the region that most needs a noise
  # number reports none.
  n_frames = depth.shape[0]
  need = max(3, int(np.ceil(0.4 * n_frames)))
  counts = inlier.sum(axis=0)
  usable = (counts >= need) & mask
  if usable.sum() >= 16 and n_frames >= 3:
    samples = np.where(inlier, depth, np.nan)
    per_pix = np.nanstd(samples, axis=0, ddof=1)
    temporal = float(np.median(per_pix[usable]))
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
    "temporal_n_pixels": int(usable.sum()),
    "p95_abs_m": p95,
  }


def evaluate(cap, board, spec, outlier_m: float = 0.05,
             reference: str | None = None) -> dict:
  """Reduce a capture to the bench's numbers.

  ``reference`` picks what depth is compared against.  ``"target"`` is the
  ChArUco plane and is the only one that makes ``bias`` mean anything.
  ``"self"`` fits a plane to the sensor's own points and reports no bias; it is
  chosen automatically for a sensor whose capture declares that its pose has to
  cross an uncalibrated extrinsic to reach the depth frame.
  """
  if reference is None:
    reference = "target" if cap.meta.get("bias_trustworthy", True) else "self"
  if reference not in ("target", "self"):
    raise ValueError(f"reference must be 'target' or 'self', got {reference!r}")

  pose_img = detect_pose(cap.gray, cap.K, cap.dist, board)
  pose = transform_pose(pose_img, cap.T_dg)
  gt_target, p_board = board_coords(cap.rays, pose)
  out = {
    "pose": {
      "distance_m": pose["plane_distance_m"],
      "n_corners": pose["n_corners"],
      "reproj_rms_px": pose["reproj_rms_px"],
      "tilt_deg": pose["tilt_deg"],
      "board_span_px": pose["board_span_px"],
      "plane_uncertainty_m": pose["plane_uncertainty_m"],
      "cross_frame": bool(not np.allclose(cap.T_dg, np.eye(4))),
    },
    "reference": reference,
    "regions": {},
  }
  median = np.median(cap.depth, axis=0)
  for name, region in spec["regions"].items():
    mask = region_mask(p_board, region)
    gt = gt_target
    if reference == "self":
      plane = fit_plane(cap.rays, median, mask)
      gt = plane_depth(cap.rays, plane) if plane is not None else gt_target
    m = region_metrics(cap.depth, gt, mask, outlier_m)
    if reference == "self":
      m["bias_m"] = float("nan")  # a self-fitted plane cannot see a bias
    out["regions"][name] = m
  return out
