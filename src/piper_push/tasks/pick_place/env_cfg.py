"""PiPER-X tidying a table: pick an object up, put it in the bin, repeat.

State-only for now.  The observation is split into ``proprio`` / ``object`` /
``privileged`` groups rather than a single vector so that the vision stage is
one line of runner config -- swap ``object`` for ``camera`` on the actor and
leave the critic alone.  Nothing else about the task has to change.

Every number that describes what the robot can do comes from the S0 audit
(artifact 8ea7b8a6) or the 2026-08-16 sysid, and says where it came from.
"""

from __future__ import annotations

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
from mjlab.viewer import ViewerConfig

from piper_push import objects, robot as piper
from piper_push.actions import RateLimitedJointPositionActionCfg
from piper_push.tasks.pick_place import mdp as pick_mdp

OBJECT = "object"
BIN = "bin"
TASK = "pick"
PAD_SENSOR = "pad_contact"

# S0: a straight-down grasp is usable at every azimuth within +-80 deg for
# r in [0.16, 0.52] m.  Objects live well inside that; the end-effector gets a
# wider envelope because it has to stand over objects at the edge and reach the
# bin, which sits outside the spawn sector.
SPAWN_RADIUS = (0.24, 0.46)
SPAWN_ANGLE = (-0.14, 0.73)  # -8 to +42 deg; the bin is at -0.63 rad
EE_ENVELOPE_RADIUS = (0.14, 0.54)
EE_ENVELOPE_ANGLE = (-1.05, 1.05)  # +-60 deg
OBJECT_LOST_RADIUS = (0.10, 0.58)
OBJECT_LOST_ANGLE = (-1.22, 1.22)
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


def make_pick_place_env_cfg(
  play: bool = False,
  profile: str = "bare_gripper",
  shape_variety: float = 1.0,
) -> ManagerBasedRlEnvCfg:
  """Build the task.

  ``shape_variety`` scales the object randomisation about its centre: 0 is a
  single fixed cube, 1 the full verified distribution.  The smoke test runs at
  0 on purpose -- a bug in the reward is far easier to see when every
  environment holds the same object, and a full distribution hides it.
  """

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
      func=pick_mdp.pad_contact, params={"sensor_name": PAD_SENSOR}
    ),
    # The reward switches behaviour on this flag; hiding it would make the MDP
    # non-Markov in exactly the dimension the task turns on.
    "grasped": ObservationTermCfg(func=pick_mdp.grasp_state, params={"command_name": TASK}),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  object_state = {
    "object_pose": ObservationTermCfg(
      func=pick_mdp.object_pose_b,
      params={"object_name": OBJECT},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "object_vel": ObservationTermCfg(
      func=pick_mdp.object_lin_vel_b,
      params={"object_name": OBJECT},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "ee_to_object": ObservationTermCfg(
      func=pick_mdp.ee_to_object,
      params={"object_name": OBJECT, "asset_cfg": ee()},
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

  observations = {
    "proprio": ObservationGroupCfg(dict(proprio), enable_corruption=not play),
    "object": ObservationGroupCfg(dict(object_state), enable_corruption=not play),
    "privileged": ObservationGroupCfg(dict(privileged), enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "arm": RateLimitedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=piper.ARM_JOINT_EXPR,
      scale=piper.PICK_ARM_SCALE,
      clip=piper.SAFE_TARGET_CLIP,
      use_default_offset=True,
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
      velocity_limit={"gripper_joint1": piper.GRIPPER_RATE_LIMIT_M_S},
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    TASK: pick_mdp.PickCommandCfg(
      # The object is replaced on success, not on a timer, so this is only a
      # backstop; the episode timeout does the real work.
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
      object_name=OBJECT,
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
      params={"pose_range": {}, "velocity_range": {}},
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
    "object_size": EventTermCfg(
      func=dr.geom_size,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(OBJECT, geom_names=("object_geom",)),
        "operation": "abs",
        "distribution": "uniform",
        # A dict, not a tuple of three: dr treats a bare tuple as one range
        # shared by every axis, which then fails to unpack.
        "ranges": {i: blend(r) for i, r in enumerate(objects.OBJECT_HALF_EXTENT_RANGE)},
      },
    ),
    "object_mass": EventTermCfg(
      func=dr.body_mass,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(OBJECT, body_names=("object",)),
        "operation": "abs",
        "distribution": "uniform",
        "ranges": blend(objects.OBJECT_MASS_RANGE),
      },
    ),
    # How the object slides on the table and settles in the bin.  It does NOT
    # set how the object is held: the pads outrank it (robot.PAD_PRIORITY), so
    # the grasp reads pad friction instead.  Measured in S0 -- with equal
    # priorities MuJoCo mixes by elementwise max and both knobs go dead.
    "object_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(OBJECT, geom_names=("object_geom",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": blend(objects.OBJECT_FRICTION_RANGE),
      },
    ),
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
    # -- find it, hold it, move it ------------------------------------------
    "reach": RewardTermCfg(
      func=pick_mdp.reach_object,
      weight=2.0,
      params={
        "command_name": TASK,
        "object_name": OBJECT,
        "std": 0.12,
        "asset_cfg": ee(),
      },
    ),
    "pads_touching": RewardTermCfg(
      func=pick_mdp.pads_touching, weight=1.0, params={"command_name": TASK}
    ),
    "grasp": RewardTermCfg(
      func=pick_mdp.grasp_bonus, weight=60.0, params={"command_name": TASK}
    ),
    "lift": RewardTermCfg(
      func=pick_mdp.lift_height, weight=2.0, params={"command_name": TASK, "target": 0.12}
    ),
    "transport": RewardTermCfg(
      func=pick_mdp.transport, weight=5.0, params={"command_name": TASK, "std": 0.15}
    ),
    "place": RewardTermCfg(
      func=pick_mdp.place_bonus, weight=300.0, params={"command_name": TASK}
    ),
    # -- do not cheat --------------------------------------------------------
    # Batting the object into the air is the cheapest way to satisfy any height
    # reward, so it is charged whenever the object is up without being held.
    "thrown": RewardTermCfg(
      func=pick_mdp.object_thrown, weight=-20.0, params={"command_name": TASK}
    ),
    "object_astray": RewardTermCfg(
      func=pick_mdp.object_outside_spawn,
      weight=-10.0,
      params={
        "command_name": TASK,
        "radius_range": SPAWN_RADIUS,
        "angle_range": SPAWN_ANGLE,
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
    # The command path already caps the target's slew at 0.9x the safety
    # shell's trip; this charges the dynamic overspeed it cannot prevent.
    "over_trip": RewardTermCfg(
      func=pick_mdp.joint_speed_over_trip,
      weight=-5.0,
      params={"limits": piper.JOINT_TRIP_RAD_S, "asset_cfg": arm()},
    ),
    # -- move smoothly and cheaply -------------------------------------------
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.15),
    "action_acc": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.08),
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
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.6,
      entities={
        "robot": piper.get_pick_robot_cfg(profile),
        OBJECT: EntityCfg(spec_fn=objects.get_object_spec),
        BIN: EntityCfg(
          spec_fn=objects.get_bin_spec,
          init_state=EntityCfg.InitialStateCfg(
            pos=(objects.BIN_CENTER[0], objects.BIN_CENTER[1], 0.0)
          ),
        ),
      },
      sensors=(
        # Both pads, filtered to the object: "is either pad touching anything"
        # would count the table and the bin wall as a grasp.
        ContactSensorCfg(
          name=PAD_SENSOR,
          primary=ContactMatch(mode="geom", pattern=piper.FINGER_PADS, entity="robot"),
          secondary=ContactMatch(mode="body", pattern="object", entity=OBJECT),
          fields=("found", "force"),
          reduce="netforce",
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
      # (measured on the first smoke: action_acc -0.49 and action_rate -0.32
      # against reach +0.22), and the safest policy is to stop moving.  This is
      # the same cliff the pushing task fell off.
      "action_rate_weight": CurriculumTermCfg(
        func=mdp.reward_curriculum,
        params={
          "reward_name": "action_rate",
          "stages": [
            {"step": 0, "weight": -0.02},
            {"step": 200 * STEPS_PER_ITERATION, "weight": -0.06},
            {"step": 500 * STEPS_PER_ITERATION, "weight": -0.15},
          ],
        },
      ),
      "action_acc_weight": CurriculumTermCfg(
        func=mdp.reward_curriculum,
        params={
          "reward_name": "action_acc",
          "stages": [
            {"step": 0, "weight": -0.01},
            {"step": 200 * STEPS_PER_ITERATION, "weight": -0.03},
            {"step": 500 * STEPS_PER_ITERATION, "weight": -0.08},
          ],
        },
      ),
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
      nconmax=256,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,  # 200 Hz physics, 50 Hz control.
    episode_length_s=12.0,
  )

  if play:
    cfg.episode_length_s = 40.0
    cfg.curriculum = {}
    cfg.observations["proprio"].enable_corruption = False
    cfg.observations["object"].enable_corruption = False

  return cfg
