"""Route names, and what each one shows the actor.  Shared by the task configs and the deployment.

A route is a name for one actor observation.  The four first-generation
routes are P0 (metric depth), P1A (cloud, PointNet), P1B (cloud, point-patch
transformer) and P2 (analytic grasp candidates).  The second generation adds
two P1B variants that differ ONLY in a fifth per-point column:

  P1BZ   the column is always zero.  Same input width and the same network as
         P1BT, so the two are a fair pair; deployable (the robot feeds zeros).
  P1BT   the column is the oracle target flag: 1 on a sampled point whose pixel
         the renderer labelled as the commanded object, 0 otherwise.  Only the
         points the cloud already contains are labelled; nothing is added,
         moved or resampled.  A simulation diagnostic and nothing else -- the
         robot has no such signal -- so bundling and every deployment entry
         point refuse it.

Kept free of mjlab imports so ``hardware/deploy`` can read it without
registering tasks.
"""

from __future__ import annotations

ROUTES = ("P0", "P1A", "P1B", "P2", "P1BZ", "P1BT", "P1BZ6")
_BASE = {"P1BZ": "P1B", "P1BT": "P1B", "P1BZ6": "P1B"}
_TARGET_CHANNEL = {"P1BZ": "zero", "P1BT": "oracle", "P1BZ6": "zero"}
ORACLE_ROUTES = ("P1BT",)
DEFAULT_CROP_Z_MIN = 0.010
_CROP_Z_MIN = {"P1BZ6": 0.006}
"""P1BZ6 = P1BZ with the height-above-table cut at 6 mm instead of 10 mm.  Measured 2026-09-07
(results/pc/gen3/crop): at 10 mm the shortest objects in the distribution contribute no point on
83 % of frames and accumulate on the table over a long run; at 6 mm that is 56 % for 3.5 % more
table-noise pixels.  The deployment applies the same cut above the calibrated plane."""


def base_route(route: str) -> str:
  """The first-generation route a variant is built on (itself for the originals)."""
  return _BASE.get(route, route)


def target_channel(route: str) -> str:
  """``none`` (4 columns), ``zero`` or ``oracle`` (5 columns)."""
  return _TARGET_CHANNEL.get(route, "none")


def crop_z_min(route: str) -> float:
  """Height above the table plane below which a point is not shown to the policy (metres)."""
  return _CROP_Z_MIN.get(route, DEFAULT_CROP_Z_MIN)


def is_oracle(route: str) -> bool:
  return route in ORACLE_ROUTES


def check_deployable(route: str) -> None:
  """Raise for a route that must never run on the robot."""
  if route not in ROUTES:
    raise ValueError(f"unknown route {route!r}; one of {ROUTES}")
  if is_oracle(route):
    raise ValueError(
      f"route {route} is oracle-only: its observation carries the renderer's target labels, "
      "which the robot cannot produce.  It exists for simulation diagnostics and is never "
      "bundled or deployed.")
