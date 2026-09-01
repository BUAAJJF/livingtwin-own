"""The task layout's yaw about the robot base.

The calibrated camera is fixed by the physical mount.  Rotating that
extrinsic to make an old screenshot look familiar would make the geometry
wrong.  The rig instead places the usable table sector ninety degrees
counter-clockwise from the original S0 layout, so every base-frame task
quantity is derived from this one angle.
"""

from __future__ import annotations

import math


WORKSPACE_YAW_RAD = math.pi / 2.0
"""Counter-clockwise task rotation in the base-frame x-y plane."""


def rotate_xy(xy: tuple[float, float]) -> tuple[float, float]:
  """Rotate one base-frame point by :data:`WORKSPACE_YAW_RAD`."""
  x, y = xy
  c, s = math.cos(WORKSPACE_YAW_RAD), math.sin(WORKSPACE_YAW_RAD)
  return (c * x - s * y, s * x + c * y)


def rotate_angle_range(angles: tuple[float, float]) -> tuple[float, float]:
  """Rotate a non-wrapping polar-angle interval."""
  return tuple(a + WORKSPACE_YAW_RAD for a in angles)


def rotate_aabb_xy(
  bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
  """Axis-aligned bounds of a rotated planar box.

  Computing this from all four corners keeps deployment filters tied to the
  same layout definition as training and remains correct if the selected yaw
  is changed later.
  """
  (xlo, xhi), (ylo, yhi) = bounds
  corners = tuple(rotate_xy((x, y)) for x in (xlo, xhi) for y in (ylo, yhi))
  xs, ys = zip(*corners, strict=True)
  return ((min(xs), max(xs)), (min(ys), max(ys)))


def rotate_half_extents(half: tuple[float, float]) -> tuple[float, float]:
  """Base-frame AABB half-extents of a rotated rectangle."""
  hx, hy = half
  c, s = abs(math.cos(WORKSPACE_YAW_RAD)), abs(math.sin(WORKSPACE_YAW_RAD))
  return (c * hx + s * hy, s * hx + c * hy)
