"""Draw the task's own geometry onto a camera image.

The workspace is an annular sector about the robot base and the bin is a
rectangle in the base frame.  Both are things the operator has to satisfy
physically -- an object outside the sector is invisible to the policy however
well the segmenter finds it -- and neither is visible in a depth image.  So
they are projected through the calibration and drawn.

This is a *calibration-dependent* overlay, and that is the point.  If the arcs
do not sit where the table's usable area actually is, either the rig moved or
the extrinsic is wrong, and the drawing says so before a run does.

Nothing here decides anything.  ``mask.workspace_mask`` remains the one place
that tests whether a point is inside; this module reads the same constants and
draws them, so the two cannot disagree about the shape while disagreeing about
where it is.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import config

# Base-frame BGR, chosen to stay legible over a grey infrared image and over
# the turbo depth ramp, and to mean the same thing in every view.
SPAWN_BGR = (90, 220, 90)
"""Where training puts objects.  The wedge the operator should aim for."""
ALLOWED_BGR = (215, 165, 60)
"""What the segmenter accepts.  Wider than spawn, and the actual filter."""
BIN_BGR = (150, 150, 150)
LABEL_BGR = (240, 240, 240)


def project(points_base: np.ndarray, rig, shape) -> np.ndarray:
  """Base-frame points to pixel coordinates, NaN where behind the camera.

  ``rig.T_base_cam`` is camera-to-base in OpenCV convention, so the inverse
  takes a base-frame point into the optical frame and the intrinsics finish
  the job.  Points at or behind the image plane have no projection and are
  returned as NaN rather than as a wrapped-around pixel, which is what a naive
  divide produces and what puts a stray line across an otherwise correct
  drawing.
  """
  p = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
  T = np.asarray(rig.T_base_cam, dtype=np.float64)
  R, t = T[:3, :3], T[:3, 3]
  cam = (p - t) @ R                      # R^T (p - t), the inverse rigid map
  K = np.asarray(rig.K, dtype=np.float64).reshape(3, 3)
  h, w = shape[:2]
  # The stored intrinsics are for the deployment resolution; a preview at a
  # different width has to scale them or every drawing lands in the corner.
  sx = w / float(config.D405_WIDTH)
  sy = h / float(config.D405_HEIGHT)
  z = cam[:, 2]
  out = np.full((p.shape[0], 2), np.nan)
  ok = z > 1e-6
  out[ok, 0] = (K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2]) * sx
  out[ok, 1] = (K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]) * sy
  return out


def _polyline(img, pts, colour, thickness=2, closed=False):
  """Draw only the runs that projected, so a partly visible shape still draws."""
  run = []
  for p in pts:
    if np.isfinite(p).all():
      run.append(p)
      continue
    if len(run) > 1:
      cv2.polylines(img, [np.asarray(run, np.int32)], False, colour,
                    thickness, cv2.LINE_AA)
    run = []
  if len(run) > 1:
    cv2.polylines(img, [np.asarray(run, np.int32)], closed, colour,
                  thickness, cv2.LINE_AA)


def sector_points(radius, angle, z: float, n: int = 96) -> np.ndarray:
  """The closed outline of an annular sector, at one height."""
  rlo, rhi = radius
  alo, ahi = angle
  a = np.linspace(alo, ahi, n)
  outer = np.stack([rhi * np.cos(a), rhi * np.sin(a), np.full(n, z)], axis=1)
  inner = np.stack([rlo * np.cos(a[::-1]), rlo * np.sin(a[::-1]),
                    np.full(n, z)], axis=1)
  return np.concatenate([outer, inner, outer[:1]])


def draw(img: np.ndarray, rig, *, table_z: float | None = None,
         spawn: bool = True, allowed: bool = True, bin_: bool = True,
         labels: bool = True) -> np.ndarray:
  """Draw the sector, the spawn wedge and the bin onto ``img`` in place.

  ``table_z`` defaults to the rig's measured plane, so the outline lies on the
  table rather than floating at the base frame's zero -- on this rig those
  differ by about 4 mm, which is a couple of pixels at the far edge and enough
  to look like a calibration error when it is not.
  """
  from piper_push.tasks.pick_place import env_cfg as task

  z = float(rig.table_z if table_z is None else table_z)
  shape = img.shape

  if allowed:
    (r, a, _) = config.WORKSPACE_SECTOR
    pts = project(sector_points(r, a, z), rig, shape)
    _polyline(img, pts, ALLOWED_BGR, 2)
    if labels:
      _label(img, rig, r[1], 0.5 * (a[0] + a[1]), z,
             "accepted  r %.2f-%.2f" % r, ALLOWED_BGR, dy=-8)

  if spawn:
    pts = project(sector_points(task.SPAWN_RADIUS, task.SPAWN_ANGLE, z),
                  rig, shape)
    _polyline(img, pts, SPAWN_BGR, 2)
    if labels:
      _label(img, rig, task.SPAWN_RADIUS[1],
             0.5 * (task.SPAWN_ANGLE[0] + task.SPAWN_ANGLE[1]), z,
             "put objects here", SPAWN_BGR, dy=-8)

  if bin_:
    (bx, by), (hx, hy) = config.BIN_CENTER, config.BIN_OUTER
    corners = np.array([[bx - hx, by - hy, z], [bx + hx, by - hy, z],
                        [bx + hx, by + hy, z], [bx - hx, by + hy, z],
                        [bx - hx, by - hy, z]])
    _polyline(img, project(corners, rig, shape), BIN_BGR, 2)
    if labels:
      p = project(np.array([[bx, by + hy, z]]), rig, shape)[0]
      if np.isfinite(p).all():
        cv2.putText(img, "bin", (int(p[0]) - 12, int(p[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, BIN_BGR, 1, cv2.LINE_AA)
  return img


def _label(img, rig, r, a, z, text, colour, dy=0):
  p = project(np.array([[r * math.cos(a), r * math.sin(a), z]]), rig,
              img.shape)[0]
  if not np.isfinite(p).all():
    return
  x, y = int(p[0]), int(p[1]) + dy
  if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
    return
  cv2.putText(img, text, (x - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
              colour, 1, cv2.LINE_AA)


def annotate(img, rig, seg=None, label: int = 0, decimate: int = 1,
             caption: str | None = None) -> np.ndarray:
  """The full rehearsal view: geometry, then whatever the segmenter found.

  Instances are drawn after the geometry so a component sits on top of the
  wedge it is inside, and each carries the two numbers that decide whether it
  is an object at all -- its height above the fitted table and its area.
  """
  draw(img, rig)
  if seg is not None:
    up = max(1, int(decimate))
    lab = (seg.labels if up == 1 else
           np.repeat(np.repeat(seg.labels, up, 0), up, 1))
    lab = lab[:img.shape[0], :img.shape[1]]
    for inst in seg.instances:
      hit = inst.label == label
      colour = (80, 235, 80) if hit else (80, 80, 235)
      cs, _ = cv2.findContours((lab == inst.label).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cv2.drawContours(img, cs, -1, colour, 2 if hit else 1)
      x, y, _w, _h = inst.bbox
      cv2.putText(img, "%.0fmm %dpx" % (inst.top_z * 1000, inst.n_px),
                  (int(x * up), max(12, int(y * up) - 5)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
  if caption:
    cv2.putText(img, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                LABEL_BGR, 1, cv2.LINE_AA)
  return img
