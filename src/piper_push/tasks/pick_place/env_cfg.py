"""PiPER-X tidying a table: pick an object up, put it in the bin, repeat.

State-only for now.  The observation is split into ``proprio`` / ``object`` /
``privileged`` groups rather than a single vector so that the vision stage is
one line of runner config -- swap ``object`` for ``camera`` on the actor and
leave the critic alone.  Nothing else about the task has to change.

Every number that describes what the robot can do comes from the S0 audit
(artifact 8ea7b8a6) or the 2026-08-16 sysid, and says where it came from.
"""

from __future__ import annotations

import os

import copy
import dataclasses
import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import GeomCfg
from mjlab.viewer import ViewerConfig

from piper_push import camera, layout, objects, robot as piper, shapes
from piper_push.actions import RateLimitedJointPositionActionCfg
from piper_push.tasks.pick_place import mdp as pick_mdp

OBJECT = "object"
BIN = "bin"
TASK = "pick"
PAD_SENSOR = "pad_contact"
PALM_SENSOR = "palm_contact"
TABLE_GUARD_SENSOR = "table_clearance_guard"
TABLE_IMPACT_SENSOR = "robot_table_impact"
CONTROL_DECIMATION = 10

# S0: a straight-down grasp is usable at every azimuth within +-80 deg for
# r in [0.16, 0.52] m.  Objects live well inside that; the end-effector gets a
# wider envelope because it has to stand over objects at the edge and reach the
# bin, which sits outside the spawn sector.
SPAWN_RADIUS = (0.24, 0.46)
SPAWN_ANGLE = layout.rotate_angle_range((-0.14, 0.73))
# 82 to 132 deg after the installed rig's +90 deg layout rotation; the bin is
# at +0.94 rad (54 deg).
EE_ENVELOPE_RADIUS = (0.14, 0.54)
EE_ENVELOPE_ANGLE = layout.rotate_angle_range((-1.05, 1.05))
OBJECT_LOST_RADIUS = (0.10, 0.58)
OBJECT_LOST_ANGLE = layout.rotate_angle_range((-1.22, 1.22))
# Where the object is ALLOWED to be, which is not the same as where it starts.
# Before the common layout rotation the bin sits at -0.63 rad and the spawn
# sector stops at -0.14; their separation is unchanged by the +90 degree yaw.
# A guard policing only the spawn sector charges 10 x 0.372 x 0.49 = 1.83 per
# step for carrying the object to the bin, against a carry that pays 1.08. Measured:
# drop_error sat at 0.21 m for 3000 iterations, which is exactly the sector
# boundary, and the penalty read -0.023 because the policy was obeying it.
OBJECT_ALLOWED_RADIUS = (0.15, 0.55)
OBJECT_ALLOWED_ANGLE = layout.rotate_angle_range((-0.95, 0.90))
# S0: straight down runs out at 150 mm for r <= 0.35 and 130 mm at r = 0.42.
# Release happens at 115 mm, so 0.30 is clear of every useful pose and well
# under where the arm ends up if it flings itself.
EE_CEILING = 0.30

# One iteration advances common_step_counter by num_steps_per_env.
STEPS_PER_ITERATION = 32


def arm() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=piper.ARM_JOINT_EXPR)


def arm_actuators() -> SceneEntityCfg:
  return SceneEntityCfg("robot", actuator_names=piper.ARM_JOINT_EXPR)


def ee() -> SceneEntityCfg:
  return SceneEntityCfg("robot", site_names=piper.GRASP_SITE)


def ghost_links() -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=piper.GHOST_LINKS)


def sight_arm() -> SceneEntityCfg:
  """The links that can stand between the camera and the object.

  link1 is left out: it is the shoulder, it barely moves in the plane of the
  view, and charging it would put a constant term on a pose the policy cannot
  change without giving up the workspace.
  """
  return SceneEntityCfg("robot", body_names=("link[2-6]",))


def sight_hand() -> SceneEntityCfg:
  """The gripper, charged separately because it is the one that matters.

  It is the part that arrives at the object, so it is the part most often
  between the object and a camera bolted to the world -- and it is the part
  whose approach direction the policy has the most freedom to choose.
  """
  return SceneEntityCfg("robot",
                        body_names=("gripper_base", "gripper_link[12]"))


def fingers() -> SceneEntityCfg:
  """The two finger bodies, whose separation IS the jaw axis."""
  return SceneEntityCfg("robot", body_names=("gripper_link[12]",))


def _ramp(
  name: str, a: float, b: float, c: float, at_b: int = 200, at_c: int = 500
) -> CurriculumTermCfg:
  """Three-stage weight schedule, in iterations."""
  return CurriculumTermCfg(
    func=mdp.reward_curriculum,
    params={
      "reward_name": name,
      "stages": [
        {"step": 0, "weight": a},
        {"step": at_b * STEPS_PER_ITERATION, "weight": b},
        {"step": at_c * STEPS_PER_ITERATION, "weight": c},
      ],
    },
  )


def make_pick_place_env_cfg(
  play: bool = False,
  profile: str = "bare_gripper",
  shape_variety: float = 1.0,
  vision: bool = False,
  wrist: bool = False,
  num_objects: int = 1,
  bounded_actions: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Build the task.

  ``bounded_actions`` (the default since 2026-09-05) makes a = +-1 the safe
  target clip on every arm joint and pairs with the tanh head in
  ``rl_cfg``; ``False`` is the original convention (scale = PICK_ARM_SCALE
  about the home pose, nothing bounding a) that the ``-V1`` task ids keep for
  the checkpoints trained under it.

  ``shape_variety`` scales the object randomisation about its centre: 0 is a
  single fixed cube, 1 the full verified distribution.  The smoke test runs at
  0 on purpose -- a bug in the reward is far easier to see when every
  environment holds the same object, and a full distribution hides it.

  ``num_objects`` puts several objects on the table at once, to be cleared one
  at a time.  The command decides which is the target; every reward, metric and
  termination reads that target through the same accessors they used when there
  was only one, so clutter changes what the policy has to see and not how the
  task is scored.  Each object needs its own contact sensors -- a single sensor
  filtered to all of them would say that a pad is touching *an* object, and
  every judgement here is about a particular one.

  ``vision`` adds the third-person camera and the observation group built
  from it.  It does not remove the object state -- the critic keeps it, and
  which groups reach the actor is decided in the runner config.

  ``wrist`` adds a SECOND camera, on the hand, in its own observation group.
  The two views fail in opposite ways and that is the whole reason for the
  second one: measured on the rig, the arm blocked the third-person camera's
  line to the object 52% of the time on a fast run, and it blocks it precisely
  when the hand is over the object -- which is when a wrist camera is pointed
  straight at it.  A wrist camera's own failure is losing the object out of
  frame when the hand is elsewhere, which is when the third-person view is
  clear.  Separate groups, not extra channels on one: they have different
  intrinsics, different range, different noise, and the encoder should not be
  made to share filters across two sensors that agree about nothing.

  The vision variant also keeps the proprioception the *state* policy was
  trained on, under ``full_proprio``.  Two things need it and both are
  privileged: the distillation teacher, which is a state policy and would not
  fit its own first layer otherwise, and the vision critic, which has no reason
  to be handicapped by a constraint that exists because the actor has to run on
  a robot.
  """

  multi = num_objects > 1
  obj_names = (
    tuple(f"{OBJECT}_{i}" for i in range(num_objects)) if multi else (OBJECT,)
  )
  pad_names = (
    tuple(f"{PAD_SENSOR}_{i}" for i in range(num_objects)) if multi else (PAD_SENSOR,)
  )
  palm_names = (
    tuple(f"{PALM_SENSOR}_{i}" for i in range(num_objects)) if multi else (PALM_SENSOR,)
  )

  def blend(rng: tuple[float, float]) -> tuple[float, float]:
    mid = 0.5 * (rng[0] + rng[1])
    return (mid + (rng[0] - mid) * shape_variety, mid + (rng[1] - mid) * shape_variety)

  proprio = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "ee_pose": ObservationTermCfg(
      func=pick_mdp.ee_pose_b,
      params={"asset_cfg": ee()},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "gripper": ObservationTermCfg(
      func=pick_mdp.gripper_opening, noise=Unoise(n_min=-0.001, n_max=0.001)
    ),
    # The pads are the one proprioceptive channel that survives to hardware
    # unchanged: the drive reports its own current, and "am I squeezing
    # something" is the cheapest reliable bit of it.
    "pad_contact": ObservationTermCfg(
      func=pick_mdp.pad_contact, params={"command_name": TASK}
    ),
    # The reward switches behaviour on this flag; hiding it would make the MDP
    # non-Markov in exactly the dimension the task turns on.
    "grasped": ObservationTermCfg(func=pick_mdp.grasp_state, params={"command_name": TASK}),
    # Under the bounded convention the policy emits u and the arm receives
    # tanh(u); the policy is told what the arm received.
    "actions": ObservationTermCfg(
      func=pick_mdp.bounded_last_action if bounded_actions else mdp.last_action),
  }

  object_state = {
    "object_pose": ObservationTermCfg(
      func=pick_mdp.object_pose_b,
      params={"command_name": TASK},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "object_vel": ObservationTermCfg(
      func=pick_mdp.object_lin_vel_b,
      params={"command_name": TASK},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "ee_to_object": ObservationTermCfg(
      func=pick_mdp.ee_to_object,
      params={"command_name": TASK, "asset_cfg": ee()},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "object_to_drop": ObservationTermCfg(
      func=pick_mdp.object_to_drop,
      params={"command_name": TASK},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "object_shape": ObservationTermCfg(
      func=pick_mdp.object_shape, params={"command_name": TASK}
    ),
  }

  # Mass, table friction and centre-of-mass offset: things a camera could never
  # read off a single frame, so the critic gets them and the actor does not.
  privileged = {
    "object_physics": ObservationTermCfg(
      func=pick_mdp.object_physics, params={"command_name": TASK}
    ),
  }
  if multi:
    # The critic is told where the rest of the table is.  The actor is not: it
    # is given one target and the depth image, which already contains the
    # others.  That asymmetry is the point -- a value function has to know how
    # much work is left, and a fixed-width state vector is a poor way to carry
    # a variable number of objects, which is the argument for the camera in the
    # first place.
    privileged["clutter"] = ObservationTermCfg(
      func=pick_mdp.clutter_state, params={"command_name": TASK}
    )

  observations = {
    "proprio": ObservationGroupCfg(dict(proprio), enable_corruption=not play),
    "object": ObservationGroupCfg(dict(object_state), enable_corruption=not play),
    "privileged": ObservationGroupCfg(dict(privileged), enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "arm": RateLimitedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=piper.ARM_JOINT_EXPR,
      scale=piper.BOUNDED_ARM_SCALE if bounded_actions else piper.PICK_ARM_SCALE,
      offset=piper.BOUNDED_ARM_OFFSET if bounded_actions else 0.0,
      clip=piper.SAFE_TARGET_CLIP,
      use_default_offset=not bounded_actions,
      bounded=bounded_actions,
      velocity_limit=piper.COMMAND_RATE_LIMIT_RAD_S,
    ),
    # Its own term because the gripper wants the full [0, 0.05] travel from
    # a in [-1, 1], which offset + scale gives and use_default_offset does not.
    "gripper": RateLimitedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=piper.GRIPPER_JOINT_EXPR,
      scale=piper.GRIPPER_SCALE,
      offset=piper.GRIPPER_OFFSET,
      clip=piper.GRIPPER_CLIP,
      use_default_offset=False,
      bounded=bounded_actions,
      velocity_limit={"gripper_joint1": piper.GRIPPER_RATE_LIMIT_M_S},
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    TASK: pick_mdp.PickCommandCfg(
      # The object is replaced on success, not on a timer, so this is only a
      # backstop; the episode timeout does the real work.
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
      # Every object is a new object, which is what a table is.  Off when there
      # is nothing to redraw: the fixed-cube variant would pay the constant
      # recompute for a shape that cannot change.
      reshape_on_place=shape_variety > 0.0,
      object_name=OBJECT,
      object_names=obj_names if multi else (),
      pad_sensor_names=pad_names if multi else (),
      pad_sensor_name=PAD_SENSOR,
      spawn_radius=SPAWN_RADIUS,
      spawn_angle=SPAWN_ANGLE,
      bin_center=objects.BIN_CENTER,
      bin_inner=objects.BIN_INNER,
      bin_rim_z=objects.BIN_WALL_HEIGHT,
    )
  }

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        # Centre on the measured D455 plane.  The range covers the 3.92 mm
        # hand-eye residual and a small remount/table shift; play/evaluation is
        # exact so reported clearance still refers to the physical setup.
        "pose_range": {} if play else {
          "z": (-0.004, 0.004),
          "roll": (-math.radians(0.25), math.radians(0.25)),
          "pitch": (-math.radians(0.25), math.radians(0.25)),
        },
        "velocity_range": {},
      },
    ),
    # The bin is a mocap body, and this is the only thing that moves it onto
    # each environment's own patch of table.
    "reset_bin": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg(BIN),
      },
    ),
    "reset_arm": EventTermCfg(
      func=pick_mdp.reset_arm_valid_posture,
      mode="reset",
      params={
        # Wide, with rejection sampling behind it.  A policy trained around one
        # posture cannot recover from any other, and a wide range without the
        # check puts a fifth of episodes underground.
        "position_range": (-0.7, 0.7),
        # RESET_FULL_RANGE=1 replaces that delta with the whole soft-limit
        # box.  Off by default: every result on record was measured on the
        # narrow one, and the wide one is a different task -- the arm has to
        # recover from postures the old policies never saw.
        "full_range": os.environ.get("RESET_FULL_RANGE", "0")
                      not in ("0", "false", "False"),
        "asset_cfg": arm(),
        "ee_cfg": ee(),
        "link_cfg": ghost_links(),
      },
    ),
    # Reset-mode, not startup: a per-environment object fixed for the whole run
    # makes an environment that drew a hard object permanently hard, and the
    # push task spent six rounds discovering what that does to a training
    # curve.  Re-rolling every episode costs a broadphase bound recompute on
    # the reset envs only.
    # One term per object.  The command looks them up by name when it redraws
    # an object at placement time, so the names are part of the contract:
    # "object_shape" with one object, "object_shape_<i>" with several.
    **{
      ("object_shape" if not multi else f"object_shape_{i}"): EventTermCfg(
        func=shapes.randomize_object_shape,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg(name),
          "mass_range": objects.OBJECT_MASS_RANGE,
          "friction_range": objects.OBJECT_FRICTION_RANGE,
          "variety": shape_variety,
        },
      )
      for i, name in enumerate(obj_names)
    },
    # The grasp knob.  S0 measured the failure boundary at mu ~ mass in kg
    # (0.30 holds 300 g, 0.40 holds 400 g), so this range clears the heaviest
    # object in the distribution by 38%.
    "pad_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=piper.FINGER_PADS),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": blend((0.55, 1.15)),
      },
    ),
  }

  rewards = {
    # -- the ladder ----------------------------------------------------------
    # Every rung pays more than the one below it, and the guidance rungs decay
    # so that by the end only the events are worth anything.  Computed rather
    # than guessed; the numbers are in the commit message.
    #
    #                                    early   late
    #   far from the object               0.20   0.05
    #   at the object, pads on it         1.33   0.31
    #   just grasped, 18 mm up            1.41   0.40
    #   carrying, halfway                 2.26   0.61
    #   held over the bin                 3.00   0.80
    #   released, object in the bin       3.00   1.00
    #
    # and hovering over the bin is worth 3.00/step early against 1.49 for
    # completing a cycle, but 0.80 against 1.49 once the guidance has decayed:
    # the decay is what makes letting go the best thing left to do.
    "reach": RewardTermCfg(
      func=pick_mdp.reach_object,
      weight=1.0,
      params={
        "command_name": TASK,
        "std": 0.12,
        "asset_cfg": ee(),
      },
    ),
    "pads_touching": RewardTermCfg(
      func=pick_mdp.pads_touching, weight=0.5, params={"command_name": TASK}
    ),
    # Without this, closing on the object is a pay cut: reach and pads_touching
    # are both gated on not-grasped and switch off together at exactly the
    # moment lift and transport are still near zero.  Four of five
    # configurations spent 3000 iterations learning to touch and never close.
    "holding": RewardTermCfg(
      func=pick_mdp.holding, weight=0.8, params={"command_name": TASK}
    ),
    "lift": RewardTermCfg(
      func=pick_mdp.lift_height, weight=0.8, params={"command_name": TASK, "target": 0.12}
    ),
    "transport": RewardTermCfg(
      # std 0.30, not 0.15: a 0.15 kernel is already saturated 20 cm out, which
      # is where the object starts, so the potential had no gradient along the
      # carry at all.  The policy grasped, lifted, and stood still holding it.
      func=pick_mdp.transport, weight=1.5, params={"command_name": TASK, "std": 0.30}
    ),
    "object_in_bin": RewardTermCfg(
      func=pick_mdp.object_in_bin, weight=3.0, params={"command_name": TASK}
    ),
    # -- the engine ----------------------------------------------------------
    # Progress, not a potential.  A potential saturates over the bin, so
    # hovering there collects it forever; progress pays for the trip and
    # nothing for standing still.
    "transport_progress": RewardTermCfg(
      func=pick_mdp.transport_progress,
      # 25 made the whole 0.30 m carry worth 7.5 points against 70 for standing
      # still the same 100 steps: the trip did not pay for itself.  120 makes
      # carrying 1.08 a step against 0.54 for holding position.
      weight=120.0,
      params={"command_name": TASK, "clip": 0.05},
    ),
    # -- the events ----------------------------------------------------------
    "grasp": RewardTermCfg(
      func=pick_mdp.grasp_bonus, weight=40.0, params={"command_name": TASK}
    ),
    "place": RewardTermCfg(
      func=pick_mdp.place_bonus, weight=250.0, params={"command_name": TASK}
    ),
    # -- do not cheat --------------------------------------------------------
    "thrown": RewardTermCfg(
      func=pick_mdp.object_thrown, weight=-20.0, params={"command_name": TASK}
    ),
    "object_astray": RewardTermCfg(
      func=pick_mdp.object_outside_spawn,
      weight=-10.0,
      params={
        "command_name": TASK,
        "radius_range": OBJECT_ALLOWED_RADIUS,
        "angle_range": OBJECT_ALLOWED_ANGLE,
      },
    ),
    # -- stay inside the machine's envelope ----------------------------------
    "ee_out_of_reach": RewardTermCfg(
      func=pick_mdp.ee_outside_envelope,
      weight=-20.0,
      params={
        "radius_range": EE_ENVELOPE_RADIUS,
        "angle_range": EE_ENVELOPE_ANGLE,
        "asset_cfg": ee(),
      },
    ),
    "ee_too_high": RewardTermCfg(
      func=pick_mdp.ee_above_height,
      weight=-30.0,
      params={"max_height": EE_CEILING, "asset_cfg": ee()},
    ),
    "arm_below_table": RewardTermCfg(
      func=pick_mdp.link_below_height,
      weight=-20.0,
      params={"min_height": 0.02, "asset_cfg": ghost_links()},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-20.0, params={"asset_cfg": arm()}
    ),
    "over_trip": RewardTermCfg(
      func=pick_mdp.joint_speed_over_trip,
      # Headroom, not the trip point.  The shell is a hardware fact and the
      # 11% ramp overshoot the servo adds on top of a command has to fit
      # underneath it, so the policy is charged from 85% and the remaining
      # 15% is what the overshoot spends.
      weight=-20.0,
      params={
        "limits": piper.JOINT_TRIP_RAD_S,
        "asset_cfg": arm(),
        "headroom": 0.85,
      },
    ),
    # The palm is not a tool.
    "palm_push": RewardTermCfg(
      func=pick_mdp.palm_pushing,
      weight=-4.0,
      params={"sensor_names": palm_names},
    ),
    # -- keep the camera's view of the object, and arrive without shoving it --
    #
    # Five terms, all of them state quantities, all of them therefore learnable
    # by the TEACHER.  That is the point: the student imitates the teacher's
    # actions, so a habit the teacher never formed is one the student cannot
    # copy.  Every one of these exists because the deployment failed on it.
    #
    # Measured on the rig 2026-09-01: the arm blocks the fixed camera's line to
    # the object, the loop holds when the mask empties, and holding cannot
    # uncover what it is covering -- one run froze in a single pose for 15.3 s
    # and spent 2429 steps held.  And objects were batted off the table by a
    # hand that arrived closed.
    "premature_touch": RewardTermCfg(
      func=pick_mdp.premature_touch,
      weight=-6.0,
      params={"command_name": TASK, "palm_sensors": palm_names,
              "clearance_m": 0.015},
    ),
    "jaws_ready": RewardTermCfg(
      func=pick_mdp.jaws_ready,
      weight=0.4,
      params={"command_name": TASK, "near_m": 0.10, "clearance_m": 0.015},
    ),
    # The tube between camera and object.  Two terms rather than one so the
    # hand can be priced above the arm without a second radius: the hand is
    # what arrives at the object and what most often ends up in front of it.
    "sight_arm": RewardTermCfg(
      func=pick_mdp.sight_cylinder,
      weight=-2.0,
      params={"command_name": TASK, "asset_cfg": sight_arm(), "radius": 0.07},
    ),
    "sight_hand": RewardTermCfg(
      func=pick_mdp.sight_cylinder,
      weight=-4.0,
      params={"command_name": TASK, "asset_cfg": sight_hand(), "radius": 0.07},
    ),
    "wrist_side_on": RewardTermCfg(
      func=pick_mdp.wrist_side_on,
      weight=0.8,
      params={"command_name": TASK, "asset_cfg": fingers(), "near_m": 0.20},
    ),
    # Contact, not proximity.  See the note on ``table_touch``: the shell this
    # task removed charged for being near the table and the teacher answered by
    # not reaching.  This charges for arriving hard, and leaves the approach
    # free.
    "table_touch": RewardTermCfg(
      func=pick_mdp.table_touch,
      weight=-2.0,
      params={"impact_sensor": TABLE_IMPACT_SENSOR, "force_threshold_n": 1.0},
    ),
    # Fingertip/table contact is deliberately not a reward or termination.
    # A binary MuJoCo contact is poorly transferable to the real setup (which
    # has no table force sensor), and a light brush is acceptable during a
    # grasp.  The sensors remain in the scene for offline contact auditing;
    # link geometry, speed and smoothness terms still discourage a dangerous
    # arm-level strike without making simulator contact part of the policy.
    # -- move smoothly and cheaply -------------------------------------------
    "action_rate": RewardTermCfg(
      func=pick_mdp.action_rate_l2_bounded if bounded_actions else mdp.action_rate_l2,
      weight=-0.15),
    "action_acc": RewardTermCfg(
      func=pick_mdp.action_acc_l2_bounded if bounded_actions else mdp.action_acc_l2,
      weight=-0.08),
    "joint_vel": RewardTermCfg(
      func=mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": arm()}
    ),
    "joint_acc": RewardTermCfg(
      func=mdp.joint_acc_l2, weight=-2.0e-7, params={"asset_cfg": arm()}
    ),
    "joint_torques": RewardTermCfg(
      func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": arm_actuators()}
    ),
    "mech_power": RewardTermCfg(
      func=mdp.electrical_power_cost, weight=-3.0e-3, params={"asset_cfg": arm()}
    ),
    "terminated": RewardTermCfg(func=mdp.is_terminated, weight=-300.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "object_lost": TerminationTermCfg(
      func=pick_mdp.object_lost,
      params={
        "command_name": TASK,
        "radius_range": OBJECT_LOST_RADIUS,
        "angle_range": OBJECT_LOST_ANGLE,
        "z_max": 0.45,
      },
    ),
    # The safety shell ends the run on hardware, so it ends the episode here.
    "over_speed": TerminationTermCfg(
      func=pick_mdp.joint_velocity_trip,
      params={"limits": piper.JOINT_TRIP_RAD_S, "asset_cfg": arm()},
    ),
    "nan": TerminationTermCfg(func=mdp.nan_detection),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        # Bit 2 is reserved for the inactive fingertip safety shells.  Objects
        # remain on bit 1 and therefore never pair with those shells.
        geoms=(GeomCfg(geom_names_expr=("terrain",), conaffinity=3),),
      ),
      num_envs=1,
      env_spacing=1.6,
      entities={
        "robot": piper.get_pick_robot_cfg(profile),
        **{n: EntityCfg(spec_fn=objects.get_object_spec) for n in obj_names},
        BIN: EntityCfg(
          spec_fn=objects.get_bin_spec,
          init_state=EntityCfg.InitialStateCfg(
            pos=(objects.BIN_CENTER[0], objects.BIN_CENTER[1], 0.0)
          ),
        ),
      },
      sensors=(
        ContactSensorCfg(
          name=TABLE_GUARD_SENSOR,
          primary=ContactMatch(
            mode="geom", pattern="[lr]f_table_guard", entity="robot"
          ),
          secondary=ContactMatch(mode="geom", pattern="terrain"),
          fields=("found", "dist"),
          reduce="mindist",
        ),
        ContactSensorCfg(
          name=TABLE_IMPACT_SENSOR,
          primary=ContactMatch(
            mode="geom",
            pattern=(".*_collision", "[lr]f_pad"),
            entity="robot",
            exclude=("base_link_collision",),
          ),
          secondary=ContactMatch(mode="geom", pattern="terrain"),
          fields=("force",),
          reduce="maxforce",
          history_length=CONTROL_DECIMATION,
        ),
        # Both pads, filtered to the object: "is either pad touching anything"
        # would count the table and the bin wall as a grasp.
        *(
          ContactSensorCfg(
            name=pad,
            primary=ContactMatch(
              mode="geom", pattern=piper.FINGER_PADS, entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="object", entity=obj),
            fields=("found", "force"),
            reduce="netforce",
          )
          for pad, obj in zip(pad_names, obj_names, strict=True)
        ),
        # The gripper body against the object.  Its own sensor because the
        # palm is not a pad: touching the object with it is a shove, and
        # without a term that can see it the behaviour is free.
        *(
          ContactSensorCfg(
            name=palm,
            primary=ContactMatch(
              mode="geom", pattern=piper.PALM_GEOMS, entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="object", entity=obj),
            fields=("found",),
          )
          for palm, obj in zip(palm_names, obj_names, strict=True)
        ),
      ),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={
      # Learn the task first, then tighten the style.  At full weight from step
      # zero the smoothness penalties are three times the reach reward
      # (measured: action_acc -0.49 and action_rate -0.32 against reach +0.22),
      # and the safest policy is to stop moving.
      "action_rate_weight": _ramp("action_rate", -0.02, -0.06, -0.15),
      "action_acc_weight": _ramp("action_acc", -0.01, -0.03, -0.08),
      # And let the guidance fade, so the events end up being the only thing
      # worth anything.  Every one of these is a state the policy can sit in;
      # at their starting weights, hovering over the bin pays twice what
      # finishing the job does.
      "reach_decay": _ramp("reach", 1.0, 0.5, 0.25, 600, 1400),
      "pads_decay": _ramp("pads_touching", 0.5, 0.25, 0.10, 600, 1400),
      "holding_decay": _ramp("holding", 0.8, 0.4, 0.20, 600, 1400),
      "lift_decay": _ramp("lift", 0.8, 0.4, 0.20, 600, 1400),
      "transport_decay": _ramp("transport", 1.5, 1.0, 0.60, 600, 1400),
      "in_bin_decay": _ramp("object_in_bin", 3.0, 2.0, 1.0, 600, 1400),
      # The visibility terms are ramped IN rather than applied at full weight,
      # and this is not caution for its own sake.  A field over a region the
      # arm has to work in is exactly the shape of the 5 mm proximity shell
      # that drove the robust teacher to inactivity, and the camera sits at a
      # corner of the workspace so the tube covers a real part of it.  Learn to
      # pick things up first, then learn to do it without standing in the way.
      "sight_arm_weight": _ramp("sight_arm", -0.3, -1.0, -2.0, 200, 600),
      "sight_hand_weight": _ramp("sight_hand", -0.6, -2.0, -4.0, 200, 600),
      "table_touch_weight": _ramp("table_touch", -0.3, -1.0, -2.0, 200, 600),
      # And the two hints fade, like every other hint here: opening the jaws
      # early and turning them across the view are things to do on the way to a
      # placement, not things worth doing instead of one.
      "jaws_ready_decay": _ramp("jaws_ready", 0.4, 0.2, 0.10, 600, 1400),
      "wrist_decay": _ramp("wrist_side_on", 0.8, 0.5, 0.30, 600, 1400),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=piper.VIEWER_BODY,
      distance=1.3,
      elevation=-30.0,
      azimuth=150.0,
    ),
    sim=SimulationCfg(
      # S0 measured 12 contacts for one grasped object with no bin.  Five
      # objects settling in a bin add object-object pairs this does not
      # contain, and contact overflow shows up as silent tunnelling rather
      # than an error, so the budget is generous from the start.
      # Measured on the rebuilt scene: 4.4 contacts per environment on average
      # and 4.4 at the peak, with one object and three parts to it.  The budget
      # is per world and it is captured into the CUDA graph, so an idle
      # allowance is not free -- 512 would not fit alongside another job on the
      # same card.  Overflow surfaces as silent tunnelling rather than an
      # error, so this still leads the measured need by 30x, and the
      # multi-object stage should re-measure rather than inherit it.
      # Measured, not guessed: 4.4 contacts per environment with one object and
      # 13 with three.  128 was set from the first of those with a wide margin;
      # the margin is kept rather than the number, because 512 was part of what
      # ran a 4096-environment run out of memory.
      nconmax=128 * (2 if num_objects > 1 else 1),
      njmax=800 * (2 if num_objects > 1 else 1),
      mujoco=MujocoCfg(
        # 2 ms, not 5.  A 1.4 m/s release covers 6.85 mm in a 5 ms step, which
        # is most of the way through a finger pad before the solver has seen
        # anything: the grasp the policy learned was not the grasp the
        # hardware will make, and the bounce that decides whether a thrown
        # object stays in the bin was resolved from a state that never
        # physically occurred.  Measured across the same policy, dropping to
        # 2 ms takes pad interpenetration from 2.71 mm to 0.84 mm at p95 and
        # from 12.8 mm to 5.2 mm at worst, for 20% of the throughput.
        timestep=0.002,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=CONTROL_DECIMATION,  # 500 Hz physics, 50 Hz control.
    episode_length_s=12.0,
  )

  if vision:
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (camera.camera_cfg(),)
    cfg.observations["camera"] = ObservationGroupCfg(
      terms={
        "scene": ObservationTermCfg(
          func=pick_mdp.CameraScene,
          params={
            "sensor_name": camera.CAMERA_NAME,
            "command_name": TASK,
            "cutoff_distance": camera.CUTOFF_M,
            # ``play`` gets the clean sensor.  Not because deployment is
            # clean -- it is the opposite -- but because a recorded rollout
            # has to show what the policy did, and a run whose depth was
            # corrupted differently from the last one cannot be compared to
            # it.  ``scripts/record_vision.py --sensor real`` turns it back on
            # when the question is what the camera does rather than what the
            # policy does, and ``piper_push.perturb`` turns individual axes of
            # it back on for an evaluation that names them.
            "noise_cfg": dataclasses.replace(
              camera.DEPTH_NOISE, strength=0.0 if play else 1.0
            ),
            "mask_jitter_px": 0 if play else camera.MASK_JITTER_PX,
          },
        )
      },
      enable_corruption=False,
      concatenate_terms=True,
    )
    # A camera that never moves is a camera the policy overfits to,
    # and a real one moves the first time somebody leans on the frame.
    cfg.events["camera_pose"] = EventTermCfg(
      func=camera.randomize_camera_pose,
      mode="startup" if play else "reset",
      params={
        "pos_jitter": 0.0 if play else camera.POS_JITTER_M,
        "rot_jitter": 0.0 if play else camera.ROT_JITTER_RAD,
      },
    )

    # The grasp flag is computed from the object's velocity and lift
    # height, which nothing on the real robot can measure.  A policy
    # that has to survive deployment gets the servo error instead --
    # the same quantity the drive reports as current -- and an RNN to
    # remember what it did with it.
    # Copied before the surgery below, and deep-copied because the observation
    # manager resolves scene entities into the term configs it is handed -- two
    # groups sharing one term object resolve it twice.
    cfg.observations["full_proprio"] = ObservationGroupCfg(
      copy.deepcopy(proprio), enable_corruption=False
    )
    # Uncorrupted, and so is the object state, because everything that reads
    # these two groups is privileged.  For the teacher: its actions are the
    # labels, and noise it can see and the student cannot is irreducible label
    # variance, which does not move where the regression converges, only how
    # fast it gets there.  For the critic: it is estimating a value, and noise
    # in its input is variance in the advantage the actor is updated from.
    cfg.observations["object"].enable_corruption = False

    del cfg.observations["proprio"].terms["grasped"]
    cfg.observations["proprio"].terms["squeeze"] = ObservationTermCfg(
      func=pick_mdp.gripper_squeeze,
      noise=None if play else Unoise(n_min=-0.0005, n_max=0.0005),
    )

  if wrist:
    assert vision, "the wrist camera supplements the scene camera, not replaces it"
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (camera.wrist_camera_cfg(),)
    cfg.observations["wrist"] = ObservationGroupCfg(
      terms={
        "scene": ObservationTermCfg(
          func=pick_mdp.CameraScene,
          params={
            "sensor_name": camera.WRIST_CAMERA_NAME,
            "command_name": TASK,
            "cutoff_distance": camera.WRIST_CUTOFF_M,
            "noise_cfg": dataclasses.replace(
              camera.WRIST_DEPTH_NOISE, strength=0.0 if play else 1.0
            ),
            "mask_jitter_px": 0 if play else camera.MASK_JITTER_PX,
          },
        )
      },
      enable_corruption=False,
      concatenate_terms=True,
    )
    # The bracket, not the tripod.  A wrist camera does not get knocked, but
    # it is bolted to a printed part and no two are seated the same, so the
    # jitter is a machining tolerance and it is an order of magnitude tighter
    # than the third-person camera's.
    cfg.events["wrist_camera_pose"] = EventTermCfg(
      func=camera.randomize_camera_pose,
      mode="startup" if play else "reset",
      params={
        "pos_jitter": 0.0 if play else camera.WRIST_POS_JITTER_M,
        "rot_jitter": 0.0 if play else camera.WRIST_ROT_JITTER_RAD,
        "sensor_name": camera.WRIST_CAMERA_NAME,
        "nominal_pos": camera.WRIST_CAMERA_POS,
        "nominal_quat": camera.WRIST_CAMERA_QUAT,
      },
    )

  if play:
    cfg.episode_length_s = 40.0
    cfg.curriculum = {}
    cfg.observations["proprio"].enable_corruption = False
    cfg.observations["object"].enable_corruption = False

  return cfg
