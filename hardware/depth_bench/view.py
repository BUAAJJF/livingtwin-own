"""The one place that draws a camera's two panels.

``measure.py`` and ``live.py`` both need the same picture -- the image the
board was found in, the depth beside it, and the three regions marked on both
-- and they used to draw it twice, slightly differently.  That stopped being
survivable once the Odin 1 arrived, because on that sensor the two panels are
not the same grid, not the same size, and the regions cannot be drawn the same
way on each.

On the greyscale panel the regions are projected: that image is a pinhole by
construction, since a fisheye is rectified before the board is looked for.  On
the depth panel they cannot be, because a lidar has no forward projection to
use -- so the region *mask* is computed by intersecting each pixel's ray with
the board plane, and its outline is drawn.  Both routes mark the same physical
patch of paper, which is the point of drawing it at all: it is the check that
the numbers are about the pixels you think they are.
"""

from __future__ import annotations

import cv2
import numpy as np

import metrics as M

COLOURS = {"charuco": (0, 255, 0), "white": (255, 128, 0), "black": (0, 200, 255)}


def depth_panel(depth: np.ndarray, near: float, far: float) -> np.ndarray:
  dn = np.clip((depth - near) / max(far - near, 1e-6), 0, 1)
  vis = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
  vis[depth <= 0] = (0, 0, 0)  # invalid stays black, not "far away"
  return vis


def compose(gray, depth, K, dist, rays, T_dg, spec, board=None, pose_img=None,
            near=0.10, far=1.50, width=480, banner=None) -> np.ndarray:
  """Greyscale and depth side by side, regions marked on both."""
  vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
  dv = depth_panel(depth, near, far)

  if pose_img is None and board is not None:
    try:
      pose_img = M.detect_pose(gray, K, dist, board)
    except RuntimeError as e:
      banner = banner or str(e)
      pose_img = None

  if pose_img is not None:
    pose = M.transform_pose(pose_img, T_dg)
    _, p_board = M.board_coords(rays, pose)
    for name, region in spec["regions"].items():
      c = COLOURS.get(name, (255, 255, 255))
      quad = np.array([
        [region["x_min_m"], region["y_min_m"], 0.0],
        [region["x_max_m"], region["y_min_m"], 0.0],
        [region["x_max_m"], region["y_max_m"], 0.0],
        [region["x_min_m"], region["y_max_m"], 0.0]])
      proj, _ = cv2.projectPoints(quad, pose_img["rvec"], pose_img["tvec"], K, dist)
      pts = np.round(proj.reshape(-1, 2)).astype(np.int32)
      cv2.polylines(vis, [pts], True, c, 2)
      cv2.putText(vis, name, tuple(pts[0] + np.array([4, -6])),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

      mask = M.region_mask(p_board, region).astype(np.uint8)
      cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cv2.drawContours(dv, cnts, -1, c, 1)
    banner = banner or (
      f"{pose['plane_distance_m'] * 1000:.0f} mm  tilt {pose['tilt_deg']:.0f} deg  "
      f"{pose_img['n_corners']} corners  reproj {pose_img['reproj_rms_px']:.2f} px")

  # The panels are different grids on a lidar, so match heights before joining
  # rather than assuming they already agree.
  h = max(vis.shape[0], dv.shape[0])
  def fit(img):
    s = h / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * s)), h),
                      interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
  out = np.hstack([fit(vis), fit(dv)])
  if banner:
    ok = pose_img is not None
    cv2.putText(out, banner[:96], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
  scale = width * 2 / out.shape[1]
  return cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
