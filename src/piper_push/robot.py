"""AgileX PiPER-X configured as a closed-gripper pusher.

Ported from ``piper_mjlab.robots.piper_x_constants`` in the pinned
``piperx-mjlab`` submodule.  The URDF and meshes still come from the submodule;
only the mjlab configuration lives here, so this repository owns its own
mjlab-version compatibility and its own "gripper is a paddle" semantics.

Three deliberate differences from the upstream constants:

* the home keyframe closes the gripper (``gripper_joint1 = 0``), matching the
  ``ctrl = 0`` that the un-actuated gripper column holds forever;
* ``ARM_ACTION_SCALE`` / ``ARM_TARGET_CLIP`` carry arm joints only, because
  mjlab raises if a scale/clip key matches none of the controlled joints;
* ``joint6``'s target clip is the real URDF limit (+/-2.0944), not the wider
  +/-2.8715 upstream carries.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_URDF = (
    _REPO_ROOT / "piperx-mjlab" / "src" / "piper_mjlab" / "assets" / "agilex_piper_x" / "piper_x.urdf"
)
PIPER_X_URDF = Path(os.environ.get("PIPER_X_URDF", _DEFAULT_URDF))
if not PIPER_X_URDF.exists():
    raise FileNotFoundError(
        f"PiPER-X URDF not found at {PIPER_X_URDF}. "
        "Run `git submodule update --init` in the repository root, or set $PIPER_X_URDF."
    )

# Grasp point on gripper_base, at the finger-pad centre height.  With the
# fingers shut this is the pushing paddle's reference frame.
GRASP_SITE_POS = (0.0, 0.0, 0.1265)

# Finger pad boxes: the finger mesh hulls are disabled below and these flat
# pads carry every contact.  Closed (joint = 0) the two pads meet at y = 0 and
# present a single flat face to the cube.
_PAD_POS = (0.0, -0.0115, 0.003)
_PAD_SIZE = (0.0148, 0.0115, 0.003)

ARM_JOINT_EXPR = ("joint[1-6]",)
GRASP_SITE = ("grasp_site",)
FINGER_PADS = ("[lr]f_pad",)
EE_BODY = "link6"
VIEWER_BODY = "base_link"
# Links whose collision geoms are disabled below; a reward term keeps them
# above the table instead of paying for the contacts.
GHOST_LINKS = ("link[2-5]",)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PIPER_X_URDF))

    # The URDF importer leaves geoms unnamed and collision in group 0.
    for body in spec.bodies:
        if not body.name or body.name == "world":
            continue
        for geom in body.geoms:
            if geom.contype == 0 and geom.conaffinity == 0:
                geom.name = f"{body.name}_visual"
                geom.group = 2
            else:
                geom.name = f"{body.name}_collision"
                geom.group = 3

    for body_name, pad_name in (("gripper_link1", "lf_pad"), ("gripper_link2", "rf_pad")):
        spec.body(body_name).add_geom(
            name=pad_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=_PAD_POS,
            size=_PAD_SIZE,
            group=3,
        )

    spec.body("gripper_base").add_site(
        name="grasp_site", pos=GRASP_SITE_POS, size=[0.005, 0.005, 0.005], group=5
    )

    # Finger coupling: the URDF mimic tag is dropped by the importer.  A stiff
    # solref keeps the 10 N actuator from tearing the coupling apart.
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_JOINT
    eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
    eq.name1 = "gripper_joint2"
    eq.name2 = "gripper_joint1"
    eq.data[:5] = [0.0, -1.0, 0.0, 0.0, 0.0]
    eq.solref = [0.005, 1.0]

    # Shut fingers can momentarily overlap; letting their pads collide wedges
    # them crossed and locks the gripper permanently.
    ex = spec.add_exclude()
    ex.bodyname1 = "gripper_link1"
    ex.bodyname2 = "gripper_link2"

    for joint in spec.joints:
        if joint.name.startswith("joint"):
            joint.frictionloss = 0.3
            joint.armature = 0.005
        elif joint.name.startswith("gripper_joint"):
            joint.armature = 0.005
            joint.damping[:] = 1.0
            joint.solref_limit = [0.005, 1.0]
    return spec


# Position actuators (the converted URDF ships none).  Only gripper_joint1 is
# actuated; joint2 follows through the equality.  Its target is never written
# by the action term, so it stays at ctrl = 0, i.e. fully closed.
ACTUATORS = (
    BuiltinPositionActuatorCfg(
        target_names_expr=("joint[1-3]",), stiffness=80, damping=5, effort_limit=100
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=("joint4",), stiffness=40, damping=5, effort_limit=100
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=("joint[5-6]",), stiffness=10, damping=1.5, effort_limit=100
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=("gripper_joint1",), stiffness=40, damping=5, effort_limit=10
    ),
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=ACTUATORS, soft_joint_pos_limit_factor=0.9
)

# Gripper at 0.0 = shut.  qpos and ctrl agree at t = 0, so there is no slam
# transient at every reset.
PUSH_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "joint2": 1.57,
        "joint3": -1.35,
        "gripper_joint1": 0.0,
        "gripper_joint2": 0.0,
    },
    joint_vel={".*": 0.0},
)

# Gripper-only collisions: wrist housing plus the two pads.  Arm links and the
# finger mesh hulls are off, which keeps the contact set (and therefore the
# Warp solve) small.  Every structural dict covers `.*_collision` because
# mjlab 1.6 requires full coverage of `geom_names_expr`.
COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision", "[lr]f_pad"),
    contype={
        "(link6|gripper_base)_collision": 1,
        "[lr]f_pad": 1,
        ".*_collision": 0,
    },
    conaffinity={
        "(link6|gripper_base)_collision": 1,
        "[lr]f_pad": 1,
        ".*_collision": 0,
    },
    condim={
        "[lr]f_pad": 6,
        ".*_collision": 3,
    },
    friction={
        "[lr]f_pad": (1.0, 5e-3, 5e-4),
        ".*_collision": (0.6,),
    },
    solref={
        "[lr]f_pad": (0.01, 1.0),
    },
    priority={
        "[lr]f_pad": 1,
        ".*_collision": 0,
    },
)


def get_push_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=PUSH_HOME_KEYFRAME,
        collisions=(COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


# Action targets resolve to JOINT names, and mjlab raises if a key matches
# nothing -- so these must not mention the gripper.
ARM_ACTION_SCALE: dict[str, float] = {
    "joint[1-3]": 0.3,
    "joint[4-6]": 0.5,
}

# Real URDF limits.  Upstream clips joint6 at +/-2.8715, which is 0.78 rad past
# its mechanical stop: the servo would push at full force into a limit
# constraint, burning power and chattering.
ARM_TARGET_CLIP: dict[str, tuple[float, float]] = {
    "joint1": (-2.6180, 2.6180),
    "joint2": (0.0, 3.1416),
    "joint3": (-2.9671, 0.0),
    "joint[4-5]": (-1.5533, 1.5533),
    "joint6": (-2.0944, 2.0944),
}
