"""Manifold Tech Odin 1 backend -- not written yet, and the contract has to
change before it can be.

The SDK is installed and the ABI is verified; see ``../odin1_sdk_notes.md`` for
the install, the API and the one remaining root-only step.  What that work
turned up is that this device does not fit the interface the other two share,
and pretending it does would quietly invent numbers.

**It is a lidar, not a depth camera.**  Depth arrives as XYZ, float32, metres,
on a fixed 256x192 grid at 10/14.5/29 Hz, and the dTOF has *no pinhole
intrinsics* -- there is no (fx, fy, cx, cy) for the depth grid, because nothing
about a lidar promises its rays form a pinhole bundle.  ``Capture`` currently
requires a K for the depth grid, and ``metrics.plane_depth`` builds its rays
from that K.  Two consequences:

  * The ChArUco has to be found in the **colour** image, which does have
    intrinsics, and the pose carried into the lidar frame through the 4x4
    camera<-lidar extrinsic from ``lidar_get_calibration``.  The registered-
    greyscale-on-the-depth-grid contract is a rendering choice here, not a
    passthrough.
  * ``fill`` needs to know the ray of a pixel that returned *nothing*, and an
    invalid lidar pixel carries no direction.  The fix is a ray table averaged
    over many frames -- dropouts move, the geometry does not -- built once per
    device and stored beside the backend.  Without it, dropout is unmeasurable
    on this sensor, which would be a convenient place to lose the single number
    the black patch exists to produce.

Its per-point timestamps also differ across a sweep, so "static scene" is an
assumption to check rather than one that is free, as it is for a global-shutter
stereo pair.

**Two things to decide before it is worth measuring at all.**  Its field of
view is 120 x 90 degrees against the simulator camera's 66 x 52, so of its
256x192 grid only about 141x111 lands on the policy's field -- roughly 40% of
the pixel count the policy is trained on.  And it is built for 70 m; the task
happens at 0.7 m, closer to the near end of a sensor whose specification is
written about the far one.
"""

from __future__ import annotations

NAME = "odin1"


def available() -> list[dict]:
  """Nothing to find until the backend exists."""
  return []


def add_args(ap) -> None:
  pass


def grab(args, n_frames: int, warmup: int = 30):
  raise SystemExit(
    "odin1 backend not implemented; see hardware/depth_bench/odin1_sdk_notes.md. "
    "The Capture contract needs a ray table for this device -- it has no pinhole "
    "intrinsics on the depth grid."
  )
