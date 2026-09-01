"""The things being tidied, and the bin they go into.

Every dimension here sits inside the envelope S0 measured on this gripper, and
the reasons are recorded next to the numbers: a shape outside it is not a hard
task, it is an impossible one, and a policy cannot tell the difference.

MuJoCo-Warp shares one model topology across every parallel environment, so a
geom's *type* cannot vary per environment.  Shape variety therefore comes from
giving the object body one geom of each type it will ever need and varying
their sizes and offsets per environment: a part that is not wanted shrinks to a
millimetre and hides inside the part that is.  That buys genuine cylinders and
genuine two-part bodies out of a single shared topology.
"""

from __future__ import annotations

import mujoco

from piper_push import layout

# ---------------------------------------------------------------------------
# Graspable object
# ---------------------------------------------------------------------------

# S0, measured on this gripper (report: artifact 8ea7b8a6):
#   width across the jaws  20-50 mm   (>=55 mm rolls out of the 29.6 mm pads)
#   height                 >=24 mm    (the fingertip hangs 12.3 mm below the
#                                      grasp site, and the site cannot go below
#                                      ~20 mm without the arm touching the table)
#   mass                   <=600 g    (10 N of grip against lift and sweep)
OBJECT_WIDTH_RANGE = (0.025, 0.045)
"""Bounding width across the jaws, in metres.  The hard ceiling is 50 mm."""

OBJECT_ASPECT_RANGE = (0.55, 2.2)
"""Height divided by mean width.

Sampling this directly is the whole point.  The first distribution drew each
half-extent independently from its own range, which sounds equivalent and is
not: with height in [30, 90] mm and width in [25, 45] mm, a shape flat enough
to count as a slab needed height at its floor *and* width at its ceiling at the
same time.  That intersection is 0.4% of the draws, so the shape curriculum
never actually contained a flat object and the acceptance number that came out
of it described boxes of varying proportion, not shape generality.
"""

OBJECT_ANISOTROPY_RANGE = (0.7, 1.4)
"""Ratio of the two horizontal half-extents, so bars appear as well as squares."""

OBJECT_MAX_HALF_WIDTH = 0.025
"""Hard ceiling across the jaws.  Anisotropy multiplies the sampled width, so
without this a 45 mm draw at the top of the anisotropy range comes out 53 mm
and rolls straight out of the 29.6 mm pads."""

OBJECT_MAX_HALF_HEIGHT = 0.045
"""Hard ceiling on height.  A 98 mm object on a 25 mm base tips before the pads
close on it, and the release already happens at z = 115 mm."""

OBJECT_HEIGHT_FLOOR = 0.024
"""S0's grasp floor.  Below this the fingertip reaches the table before the pad
reaches the object, so a flatter object is not a harder task, it is one this
gripper cannot do."""

OBJECT_MASS_RANGE = (0.05, 0.40)
OBJECT_FRICTION_RANGE = (0.4, 1.0)
"""Slide friction against the TABLE.  The grasp does not use this -- the pads
outrank the object, so pad friction owns the object-pad contact.  See
``robot.PAD_PRIORITY``."""

OBJECT_PRIORITY = 2
"""Above the terrain's 0 so the object owns how it slides, below the pads' 3 so
it does not also own how it is held.  Equal priorities would make MuJoCo mix
friction by elementwise max, which masks both knobs at once (measured in S0)."""

# A 1.4 m/s release travels 2.8 mm in one 2 ms step, so the contact has to be
# stiff enough to stop the object inside that.  The default 0.02 s time
# constant is four times the old physics step and let objects sink 3.6 mm into
# the table (p95) and 18 mm at worst; 0.008 s with the step at 2 ms measured
# 0.58 mm and 7.8 mm for the same policy, and ran no slower.
OBJECT_SOLREF = (0.008, 1.0)
OBJECT_SOLIMP = (0.95, 0.99, 0.001)

# The three parts every object is built from.  ``core`` is the box that carries
# the shape in every class; ``barrel`` is the cylinder; ``stub`` is the second
# box that makes a body two-part.  Unused parts shrink to COLLAPSED_HALF.
CORE_GEOM = "object_core"
BARREL_GEOM = "object_barrel"
STUB_GEOM = "object_stub"
OBJECT_GEOMS = (CORE_GEOM, BARREL_GEOM, STUB_GEOM)

COLLAPSED_HALF = 0.001
"""A part that is not in use is shrunk to this and parked inside a part that
is, where nothing outside the body can reach it."""

# Shape classes, and how often each is drawn.  Cylinders and two-part bodies
# are what make "arbitrary rigid object" more than a slogan; the plain box
# stays the plurality because it is the one every earlier result is comparable
# against.
SHAPE_CLASSES = ("box", "cylinder", "stepped", "l_shape", "capped")
SHAPE_WEIGHTS = (0.34, 0.22, 0.16, 0.16, 0.12)

DEFAULT_HALF_SIZE = (0.020, 0.020, 0.025)
DEFAULT_MASS = 0.15
DEFAULT_FRICTION = 0.6


def get_object_spec(
  half_size: tuple[float, float, float] = DEFAULT_HALF_SIZE,
  mass: float = DEFAULT_MASS,
  slide_friction: float = DEFAULT_FRICTION,
  rgba: tuple[float, float, float, float] = (0.85, 0.30, 0.22, 1.0),
) -> mujoco.MjSpec:
  """One body carrying every part any shape class needs.

  The compiled sizes are placeholders: ``randomize_object_shape`` overwrites
  size, offset, mass and inertia together on every reset.  What is fixed here
  is only what cannot vary per environment -- how many geoms there are and what
  type each one is.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_joint")

  parts = (
    (CORE_GEOM, mujoco.mjtGeom.mjGEOM_BOX, half_size),
    (BARREL_GEOM, mujoco.mjtGeom.mjGEOM_CYLINDER, (half_size[0], half_size[2], 0.0)),
    (STUB_GEOM, mujoco.mjtGeom.mjGEOM_BOX, (COLLAPSED_HALF,) * 3),
  )
  for name, geom_type, size in parts:
    geom = body.add_geom(
      name=name,
      type=geom_type,
      size=size,
      rgba=rgba,
      group=0,
    )
    geom.condim = 3
    geom.friction[0] = slide_friction
    geom.priority = OBJECT_PRIORITY
    geom.solref[:2] = OBJECT_SOLREF
    geom.solimp[:3] = OBJECT_SOLIMP

  # Mass is written per environment from the sampled volume, but the compiled
  # body still needs one, and a body whose geoms are all placeholders would
  # otherwise take its density default.
  body.mass = mass
  return spec


# ---------------------------------------------------------------------------
# Bin
# ---------------------------------------------------------------------------

# S0 checked five placements by IK; the unrotated point (0.30, -0.22) clears
# every one of them with the most margin.  The installed rig's whole task
# layout is +90 degrees about the base, including the bin footprint.
BIN_CENTER = layout.rotate_xy((0.30, -0.22))
BIN_INNER = layout.rotate_half_extents((0.080, 0.070))
"""Rotated base-frame half-extents: a 140 x 160 mm axis-aligned footprint.

It is the same physical 160 x 140 mm opening as before, turned with the rest
of the task, and has room for several objects.
"""
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
    # A thrown object arrives at the wall as fast as it arrives at the floor,
    # so the walls get the same stiffened contact.
    geom.solref[:2] = OBJECT_SOLREF
    geom.solimp[:3] = OBJECT_SOLIMP
    # Left at priority 0 so an object keeps owning its own contacts here
    # too: the same friction that decides how it slides on the table
    # decides how it settles in the bin.
  return spec
