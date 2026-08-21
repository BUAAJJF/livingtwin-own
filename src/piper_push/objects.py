"""The things being tidied, and the bin they go into.

Every dimension here sits inside the envelope S0 measured on this gripper, and
the reasons are recorded next to the numbers: a shape outside it is not a hard
task, it is an impossible one, and a policy cannot tell the difference.

MuJoCo-Warp shares one model topology across every parallel environment, so a
geom's *type* cannot vary per environment but its *size* can.  Shape variety
therefore comes from randomising the half-extents of a fixed set of boxes --
which spans cubes, bars and slabs from one geom, and stepped and L-shaped
bodies from two.  Cylinders and capsules need their own entity and arrive with
the multi-object stage.
"""

from __future__ import annotations

import mujoco

# ---------------------------------------------------------------------------
# Graspable object
# ---------------------------------------------------------------------------

# S0, measured on this gripper (report: artifact 8ea7b8a6):
#   width across the jaws  20-50 mm   (>=55 mm rolls out of the 29.6 mm pads)
#   height                 >=24 mm    (the fingertip hangs 12.3 mm below the
#                                      grasp site, and the site cannot go below
#                                      ~20 mm without the arm touching the table)
#   mass                   <=600 g    (10 N of grip against lift and sweep)
# The first training distribution is deliberately inside all three.
OBJECT_HALF_EXTENT_RANGE = ((0.0125, 0.0225), (0.0125, 0.0225), (0.015, 0.045))
"""Per-axis half-extent bounds: 25-45 mm wide either way, 30-90 mm tall."""

OBJECT_MASS_RANGE = (0.05, 0.40)
OBJECT_FRICTION_RANGE = (0.4, 1.0)
"""Slide friction against the TABLE.  The grasp does not use this -- the pads
outrank the object, so pad friction owns the object-pad contact.  See
``robot.PAD_PRIORITY``."""

OBJECT_PRIORITY = 2
"""Above the terrain's 0 so the object owns how it slides, below the pads' 3 so
it does not also own how it is held.  Equal priorities would make MuJoCo mix
friction by elementwise max, which masks both knobs at once (measured in S0)."""

DEFAULT_HALF_SIZE = (0.020, 0.020, 0.025)
DEFAULT_MASS = 0.15
DEFAULT_FRICTION = 0.6


def get_object_spec(
    half_size: tuple[float, float, float] = DEFAULT_HALF_SIZE,
    mass: float = DEFAULT_MASS,
    slide_friction: float = DEFAULT_FRICTION,
    rgba: tuple[float, float, float, float] = (0.85, 0.30, 0.22, 1.0),
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_joint")
    geom = body.add_geom(
        name="object_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half_size,
        mass=mass,
        rgba=rgba,
        group=0,
    )
    geom.condim = 3
    geom.friction[0] = slide_friction
    geom.priority = OBJECT_PRIORITY
    return spec


# ---------------------------------------------------------------------------
# Bin
# ---------------------------------------------------------------------------

# S0 checked five placements by IK; this one clears every one of them with the
# most margin.  Radius 0.372 m and azimuth -36 deg put it inside the reachable
# band at every height the release needs, and outside the object spawn sector.
BIN_CENTER = (0.30, -0.22)
BIN_INNER = (0.080, 0.070)
"""Inner half-extents: a 160 x 140 mm opening, room for several objects."""
BIN_WALL_HEIGHT = 0.060
"""Rim height.  Release happens 55 mm above it, at z = 115 mm, which S0 solved
to 1.5 mm and 0.3 deg.  A taller rim eats into the straight-down envelope,
which runs out at 150 mm."""
BIN_WALL_THICKNESS = 0.008
"""Thick enough that a 400 g object at transport speed cannot tunnel it; thin
walls are the classic way objects leave a bin without ever going over the rim."""


def get_bin_spec(
    inner: tuple[float, float] = BIN_INNER,
    height: float = BIN_WALL_HEIGHT,
    thickness: float = BIN_WALL_THICKNESS,
    rgba: tuple[float, float, float, float] = (0.35, 0.45, 0.62, 1.0),
) -> mujoco.MjSpec:
    """Four walls standing on the terrain.  The terrain is the floor: one fewer
    geom, one fewer contact pair, and nothing can slip between floor and wall."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="bin")
    # Mocap, not plain static: a fixed-base entity that is not a mocap body has
    # no way to be positioned per environment, so every parallel bin would
    # stack at the world origin while the arms spread out on their grid.
    body.mocap = True
    ix, iy = inner
    half_h = height / 2.0
    walls = (
        ("bin_x_pos", (ix + thickness, 0.0, half_h), (thickness, iy + 2 * thickness, half_h)),
        ("bin_x_neg", (-ix - thickness, 0.0, half_h), (thickness, iy + 2 * thickness, half_h)),
        ("bin_y_pos", (0.0, iy + thickness, half_h), (ix, thickness, half_h)),
        ("bin_y_neg", (0.0, -iy - thickness, half_h), (ix, thickness, half_h)),
    )
    for name, pos, size in walls:
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=rgba,
            group=0,
        )
        geom.condim = 3
        geom.friction[0] = 0.6
        # Left at priority 0 so an object keeps owning its own contacts here
        # too: the same friction that decides how it slides on the table
        # decides how it settles in the bin.
    return spec
