"""Manifold Tech Odin 1 backend -- not written yet.

Left here so the shape of the comparison is visible: it has to produce a depth
stack, a grayscale image on the depth grid and that grid's intrinsics, and
nothing else in the bench changes.

The Odin 1 is a lidar-plus-camera unit rather than a stereo camera, so two
things need care that do not arise for the other two.  Its depth is sparse and
on a different grid from any image, which means the grayscale-registered
contract here is a rendering choice, not a passthrough; and its per-point
timestamps differ across a sweep, so a "static scene" assumption that is free
for a global-shutter stereo pair has to be checked rather than assumed.
"""

from __future__ import annotations

NAME = "odin1"


def add_args(ap) -> None:
  pass


def grab(args, n_frames: int, warmup: int = 30):
  raise SystemExit(
    "odin1 backend not implemented. Implement capture/odin1.py against the "
    "same Capture contract as d405.py."
  )
