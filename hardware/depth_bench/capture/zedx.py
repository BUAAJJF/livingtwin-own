"""Stereolabs ZED X backend -- not written yet, the unit has a hardware fault.

Left here so the shape of the comparison is visible: when the camera arrives,
this file has to produce a depth stack, a grayscale image on the depth grid and
that grid's intrinsics, and nothing else in the bench changes.

Notes for whoever writes it: the ZED SDK returns MEASURE.DEPTH already in the
left rectified image's frame, so the grayscale can be the left view with no
alignment step at all.  Record NEURAL vs ULTRA depth mode in ``meta`` -- they
are different sensors as far as these numbers are concerned.
"""

from __future__ import annotations

NAME = "zedx"


def available() -> list[dict]:
  """Nothing to find until the backend exists."""
  return []


def add_args(ap) -> None:
  ap.add_argument("--depth-mode", default="neural")


def grab(args, n_frames: int, warmup: int = 30):
  raise SystemExit(
    "zedx backend not implemented -- the unit is out for a hardware fault. "
    "Implement capture/zedx.py against the same Capture contract as d405.py."
  )
