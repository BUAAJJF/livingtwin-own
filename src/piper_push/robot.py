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

import dataclasses
import functools
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


# ---------------------------------------------------------------------------
# Identified plant and deployment limits
#
# From the 2026-08-16 free-air sysid campaign in piperx-mjlab (branch
# yf/yoyo-cc-sim, docs/arm_sysid_20260816.md).  The numbers above this line are
# the URDF-conversion defaults the push task was trained on and are left alone
# so its checkpoints keep loading; everything below is what the pick task uses.
# ---------------------------------------------------------------------------

# Effective gains, not the announced MIT impedance table.  The drives run a
# 200 Hz law tau = kp(p_des - p) + kd(v_des - v) + t_ff, and their inner
# current/velocity loops make the impedance about 5x stiffer than the nominal
# kp 25: 2.51 N.m of measured payload bias over ~20 mrad of measured sag gives
# kp_eff ~ 125 on J2.  kd scales with kp, so the velocity lead is unchanged.
SYSID_GAINS: dict[str, tuple[float, float]] = {
    "joint[1-3]": (125.0, 6.5),
    "joint4": (60.0, 4.0),
    "joint5": (60.0, 3.0),
    "joint6": (40.0, 2.0),
}

# Per-joint Coulomb friction from constant-velocity strokes.  MuJoCo's
# frictionloss IS Coulomb-with-stiction, which the sysid prefers over
# Coulomb + viscous: the apparent viscous coefficient is Stribeck-negative at
# these speeds.  The push task's flat 0.3 is 3.8x too high on joint2 and 10x
# too high on joint5.
SYSID_COULOMB_NM: dict[str, float] = {
    "joint1": 0.25,
    "joint2": 1.13,
    "joint3": 0.54,
    "joint4": 0.08,
    "joint5": 0.03,
    "joint6": 0.05,
}

# Where the deployment safety shell trips, rad/s (PiPER-X manual).  Exceeding
# these ends a run on hardware, so a policy that plans through them is planning
# something the robot will not execute.
JOINT_TRIP_RAD_S: dict[str, float] = {
    "joint1": 3.1415927,  # 180 deg/s
    "joint2": 3.3510322,  # 192 deg/s
    "joint3": 3.1415927,  # 180 deg/s
    "joint4": 3.9269908,  # 225 deg/s
    "joint5": 3.9269908,  # 225 deg/s
    "joint6": 3.9269908,  # 225 deg/s
}

# The command path gets a derated fraction of the trip; the trip itself stays as
# a termination for the dynamic overspeed a command limiter cannot prevent.
#
# The classical stack uses 0.9, but 0.9 measured a trip on an ordinary transport
# move here: a position servo tracking a constant-velocity ramp lags, builds
# error, and overshoots the commanded rate by about 11%, and this simulation has
# none of the drive's inner velocity loop to absorb that.  0.75 puts the
# overshoot back under the shell.
COMMAND_DERATE = 0.75
COMMAND_RATE_LIMIT_RAD_S: dict[str, float] = {
    j: COMMAND_DERATE * v for j, v in JOINT_TRIP_RAD_S.items()
}

# Not measured.  The URDF's 3 m/s is the mechanism's, not the drive's, and a
# gripper that shuts in 17 ms is a policy exploit waiting to happen: erring
# slow costs throughput, erring fast costs sim-to-real.  MUST be replaced with
# a hardware measurement before S5.
GRIPPER_RATE_LIMIT_M_S = 0.10

GRIPPER_FORCE_N = 10.0
"""The drive's rated force, from the URDF effort limit."""
GRIPPER_STIFFNESS = 800.0
GRIPPER_DAMPING = 18.0
"""Also not measured, and NOT what the URDF conversion handed down.

The inherited kp of 40 makes the rated force unreachable: a position servo
squeezing a 35 mm object with its target at zero produces 40 x 0.0175 = 0.7 N,
so the gripper drops everything.  (S0 only ever measured 10 N because it drove
ctrl to -0.25, a quarter-metre of virtual overshoot, which the action space
cannot express.)  Real hardware takes a position and drives to it with up to
its rated force, so kp is set to reach that force over the SMALLEST object's
half width in the distribution: 800 x 0.0125 = 10 N, saturated.  (Sizing it on
the cube variant's 17.5 mm instead leaves the 25 mm objects at 7.5 N, which the
regression test catches.)  Replace with a measured force-vs-command curve
before S5.
"""

# Position limits: the intersection of the URDF and the manual.  The manual is
# tighter on joint5 (75 deg against the URDF's 89) and looser on joint4 and
# joint6, where the URDF's mechanical stop governs.
SAFE_TARGET_CLIP: dict[str, tuple[float, float]] = {
    "joint1": (-2.6179938, 2.6179938),
    "joint2": (0.0, 3.1415926),
    "joint3": (-2.9670597, 0.0),
    "joint4": (-1.5533430, 1.5533430),
    "joint5": (-1.3089969, 1.3089969),  # manual, tighter than the URDF
    "joint6": (-2.0943951, 2.0943951),
}

# ---------------------------------------------------------------------------
# Robot profiles
# ---------------------------------------------------------------------------

GRIPPER_JOINT_EXPR = ("gripper_joint1",)
GRIPPER_OPEN_M = 0.05
"""Full open, per finger.  The jaw gap is twice this: 100 mm, measured."""
PAD_PRIORITY = 3
"""Above the object's 2, so the pads own the object-pad contact.

Measured (S0): with pads and object both at priority 1, MuJoCo mixes friction
by elementwise max and neither side can lower it -- pad mu of 0.30, 0.15 and
0.08 gave identical slip.  Giving the object priority instead couples the grasp
to the table contact AND drops the pads' condim 6, losing torsional friction.
Pads 3 > object 2 > terrain 0 separates them: the pads own object-pad (condim
6, pad friction), the object owns object-table (condim 3, object friction).
"""


@dataclasses.dataclass(frozen=True)
class RobotProfile:
    """A hardware configuration of the arm.

    The wrist payload is its own body with its own gravity-compensation flag,
    because the hardware's feedforward model carries gravity for every mass it
    knows about and nothing else.  An unmodelled 4.1 N at the gripper IS the
    measured 20 mrad joint2 sag.
    """

    name: str
    wrist_payload_kg: float = 0.0
    wrist_payload_pos: tuple[float, float, float] = (0.0, 0.0, 0.08)
    payload_gravcomp: float = 0.0
    arm_gravcomp: float = 1.0


BARE_GRIPPER = RobotProfile(name="bare_gripper")
"""Nominal for S1: vision comes from a fixed third-person camera, so the wrist
carries no RealSense and no force/torque sensor."""

WRIST_SENSOR_PAYLOAD_4P1N = RobotProfile(
    name="wrist_sensor_payload_4p1N",
    # Kunwei F/T + RealSense + brackets, 4.1 N as weighed on 2026-08-15.
    wrist_payload_kg=4.1 / 9.81,
    wrist_payload_pos=(0.0, 0.0, 0.08),
    payload_gravcomp=0.0,
)
"""The rig the sysid gains were identified on.  Switch to this and re-validate
if the hardware gets the wrist sensors back."""

PROFILES: dict[str, RobotProfile] = {
    p.name: p for p in (BARE_GRIPPER, WRIST_SENSOR_PAYLOAD_4P1N)
}

# Gripper open at rest.  qpos and ctrl agree at t = 0 so there is no slam
# transient at reset, and an open hand is the posture an approach starts from.
# The midpoint of the joint range the task actually needs, solved by IK over
# the whole spawn sector at every grasp height plus the bin (scratchpad
# s0/jrange.py).  The pushing task's posture expresses 3% of those poses:
# joint2 alone covers 34% of them, so a policy started there could not reach a
# grasp however well it learned.
PICK_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "joint1": 0.06,
        "joint2": 1.72,
        "joint3": -1.25,
        "joint4": 1.13,
        "gripper_joint1": GRIPPER_OPEN_M,
        "gripper_joint2": -GRIPPER_OPEN_M,
    },
    joint_vel={".*": 0.0},
)

# As the push task's, except the pads take priority 3 (see PAD_PRIORITY).
PICK_COLLISION = CollisionCfg(
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
        "[lr]f_pad": PAD_PRIORITY,
        ".*_collision": 0,
    },
)


def get_pick_spec(profile: RobotProfile = BARE_GRIPPER) -> mujoco.MjSpec:
    """The identified plant, as a spec.

    Gravity compensation and the payload are set HERE rather than on the
    compiled model: writing ``body_gravcomp`` after compilation leaves
    ``qfrc_gravcomp`` at zero and changes nothing (measured in S0).
    """
    spec = get_spec()

    for joint in spec.joints:
        coulomb = SYSID_COULOMB_NM.get(joint.name)
        if coulomb is not None:
            joint.frictionloss = coulomb

    # The production command path streams t_ff = full inverse dynamics, so the
    # real arm is gravity compensated for every mass its model knows about.
    for body in spec.bodies:
        if body.name and body.name != "world":
            body.gravcomp = profile.arm_gravcomp

    if profile.wrist_payload_kg > 0.0:
        payload = spec.body("gripper_base").add_body(
            name="wrist_payload", pos=profile.wrist_payload_pos
        )
        payload.gravcomp = profile.payload_gravcomp
        payload.add_geom(
            name="wrist_payload_mass",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.02, 0.0, 0.0),
            contype=0,
            conaffinity=0,
            mass=profile.wrist_payload_kg,
            rgba=(0.30, 0.30, 0.35, 0.7),
            group=2,
        )
    return spec


def _sysid_actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
    arm = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(expr,), stiffness=kp, damping=kd, effort_limit=100
        )
        for expr, (kp, kd) in SYSID_GAINS.items()
    )
    gripper = BuiltinPositionActuatorCfg(
        target_names_expr=GRIPPER_JOINT_EXPR,
        stiffness=GRIPPER_STIFFNESS,
        damping=GRIPPER_DAMPING,
        effort_limit=GRIPPER_FORCE_N,
    )
    return arm + (gripper,)


def get_pick_robot_cfg(profile: RobotProfile | str = BARE_GRIPPER) -> EntityCfg:
    if isinstance(profile, str):
        profile = PROFILES[profile]
    return EntityCfg(
        init_state=PICK_HOME_KEYFRAME,
        collisions=(PICK_COLLISION,),
        spec_fn=functools.partial(get_pick_spec, profile),
        articulation=EntityArticulationInfoCfg(
            actuators=_sysid_actuators(), soft_joint_pos_limit_factor=0.9
        ),
    )


# Action scaling for the pick task: the half-span of the needed joint range
# with headroom, not the pushing task's values.  joint6 is capped below its
# measured 1.88 because a parallel jaw at roll theta and theta + pi is the same
# grasp, so +-1.6 already covers every orientation a box can present.
PICK_ARM_SCALE: dict[str, float] = {
    "joint1": 0.90,
    "joint2": 0.60,
    "joint3": 0.75,
    "joint4": 0.40,
    "joint5": 0.30,
    "joint6": 1.60,
}
GRIPPER_SCALE = GRIPPER_OPEN_M / 2.0
GRIPPER_OFFSET = GRIPPER_OPEN_M / 2.0
GRIPPER_CLIP: dict[str, tuple[float, float]] = {"gripper_joint1": (0.0, GRIPPER_OPEN_M)}
