"""The 50 mm cube that gets pushed."""

from __future__ import annotations

import mujoco

CUBE_HALF_SIZE = 0.025  # MuJoCo box size is a half-extent, so this is 50 mm.
CUBE_MASS = 0.10  # 800 kg/m^3 -- plausible for a plastic/wood block.
CUBE_SLIDE_FRICTION = 0.4


def get_cube_spec(
    half_size: float = CUBE_HALF_SIZE,
    mass: float = CUBE_MASS,
    slide_friction: float = CUBE_SLIDE_FRICTION,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    geom = body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_size,) * 3,
        mass=mass,
        rgba=(0.85, 0.25, 0.20, 1.0),
    )
    geom.condim = 3
    geom.friction[0] = slide_friction
    # mjlab's plane terrain takes MuJoCo's default mu = 1.0 and priority 0, and
    # equal priorities mix friction by elementwise max -- so a low-friction cube
    # alone would change nothing.  Priority 1 makes the cube win that mix
    # against the ground; against the pads (also priority 1) the mix falls back
    # to max, keeping the pusher grippy.
    geom.priority = 1
    return spec
