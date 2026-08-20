"""PiPER-X continuous multi-goal cube pushing.

Joint-space control, gripper permanently shut.  The policy sees only
proprioception, its own end-effector pose, the cube's pose, and the goal; it
outputs six joint position targets at 50 Hz.  Goals resample on a timer and
immediately on success, so an episode is a stream of pushes and the reward for
being fast is simply that more goals fit in the same wall clock.
"""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.envs import mdp
from mjlab.tasks.manipulation import mdp as manip_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from piper_push import robot as piper
from piper_push.cube import CUBE_HALF_SIZE, get_cube_spec
from piper_push.tasks.push_cube import mdp as push_mdp

# SceneEntityCfg.resolve() fills ids in place, so each term gets a fresh one
# rather than sharing a single mutated instance.
def arm() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=piper.ARM_JOINT_EXPR)


def arm_actuators() -> SceneEntityCfg:
  return SceneEntityCfg("robot", actuator_names=piper.ARM_JOINT_EXPR)


def ee() -> SceneEntityCfg:
  return SceneEntityCfg("robot", site_names=piper.GRASP_SITE)


def ghost_links() -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=piper.GHOST_LINKS)

CUBE = "cube"
GOAL = "push_goal"

# Where the cube may start and where goals may be placed.  Both sit well inside
# the PiPER-X's ~0.62 m reach so that a goal is never unreachable.
CUBE_SPAWN_X = (0.32, 0.48)
CUBE_SPAWN_Y = (-0.15, 0.15)
GOAL_BOUNDS_X = (0.28, 0.52)
GOAL_BOUNDS_Y = (-0.20, 0.20)

# Iterations are converted to env steps for the curricula: one iteration
# advances `common_step_counter` by `num_steps_per_env`.
STEPS_PER_ITERATION = 32


def make_push_cube_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, params={"asset_cfg": arm()}, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, params={"asset_cfg": arm()}, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "ee_pose": ObservationTermCfg(
      func=push_mdp.ee_pose_b, params={"asset_cfg": ee()}, noise=Unoise(n_min=-0.005, n_max=0.005)
    ),
    "cube_pose": ObservationTermCfg(
      func=push_mdp.object_pose_b,
      params={"object_name": CUBE},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "cube_vel": ObservationTermCfg(
      func=push_mdp.object_lin_vel_b,
      params={"object_name": CUBE},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "ee_to_cube": ObservationTermCfg(
      func=manip_mdp.ee_to_object_distance,
      params={"object_name": CUBE, "asset_cfg": ee()},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "cube_to_goal": ObservationTermCfg(
      func=manip_mdp.object_to_goal_distance,
      params={"object_name": CUBE, "command_name": GOAL},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    # The goal resamples on a timer, so without the remaining phase the value
    # function cannot tell a fresh goal from one about to expire.
    "goal_phase": ObservationTermCfg(func=push_mdp.goal_phase, params={"command_name": GOAL}),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(dict(actor_terms), enable_corruption=not play),
    "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    # The gripper actuator is deliberately absent: an unwritten position target
    # stays at 0, which for gripper_joint1 (URDF range [0, 0.05]) is fully shut.
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=piper.ARM_JOINT_EXPR,
      scale=piper.ARM_ACTION_SCALE,
      clip=piper.ARM_TARGET_CLIP,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    GOAL: push_mdp.PushCommandCfg(
      entity_name=CUBE,
      resampling_time_range=(2.5, 4.0),
      debug_vis=True,
      success_threshold=0.025,
      min_goal_separation=0.06,
      dwell_steps=3,
      resample_on_success=True,
      goal_z=CUBE_HALF_SIZE,
      cube_spawn_x=CUBE_SPAWN_X,
      cube_spawn_y=CUBE_SPAWN_Y,
      goal_radius_range=(0.07, 0.16),
      goal_bounds_x=GOAL_BOUNDS_X,
      goal_bounds_y=GOAL_BOUNDS_Y,
    )
  }

  events = {
    # Mandatory for a fixed-base robot: this is the only thing that applies
    # env_origins, without it every arm stacks at the world origin.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}},
    ),
    "reset_arm": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        # Arm joints only, so the gripper stays exactly shut.
        "asset_cfg": arm(),
      },
    ),
    # Startup-only randomisation: paid once, never in the hot loop.
    "cube_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(CUBE, geom_names=("cube_geom",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.30, 0.60),
      },
    ),
    "pad_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=piper.FINGER_PADS),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.60, 1.30),
      },
    ),
  }

  rewards = {
    # -- get the cube to the goal, fast ------------------------------------
    "goal_progress": RewardTermCfg(
      func=push_mdp.goal_progress,
      weight=20.0,
      params={"command_name": GOAL, "object_name": CUBE, "clip": 0.5},
    ),
    "goal_reached": RewardTermCfg(
      func=push_mdp.goal_reached_bonus, weight=100.0, params={"command_name": GOAL}
    ),
    "cube_at_goal": RewardTermCfg(
      func=manip_mdp.bring_object_reward,
      weight=2.0,
      params={"command_name": GOAL, "object_name": CUBE, "std": 0.05},
    ),
    # -- bootstrap: find the cube, then get behind it -----------------------
    "reach_cube": RewardTermCfg(
      func=manip_mdp.staged_position_reward,
      weight=1.0,
      params={
        "command_name": GOAL,
        "object_name": CUBE,
        "reaching_std": 0.15,
        "bringing_std": 0.15,
        "asset_cfg": ee(),
      },
    ),
    "push_alignment": RewardTermCfg(
      func=push_mdp.push_alignment,
      weight=1.5,
      params={
        "command_name": GOAL,
        "object_name": CUBE,
        "contact_std": 0.08,
        "asset_cfg": ee(),
      },
    ),
    # -- energy -------------------------------------------------------------
    # Sum of clamp(tau * qdot, 0) in watts.  There is no gravity compensation
    # anywhere in this model, so holding still is free (tau != 0 but qdot == 0)
    # and only actual mechanical work is charged.
    "mech_power": RewardTermCfg(
      func=mdp.electrical_power_cost, weight=-3.0e-3, params={"asset_cfg": arm()}
    ),
    "joint_torques": RewardTermCfg(
      func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": arm_actuators()}
    ),
    # -- move as little as possible, as smoothly as possible ----------------
    "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=-2.0e-3, params={"asset_cfg": arm()}),
    "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.0e-7, params={"asset_cfg": arm()}),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.15),
    "action_acc": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.08),
    # Free below max_vel, quadratic above.  The function returns a positive
    # sum, so the weight must be negative.
    "joint_vel_hinge": RewardTermCfg(
      func=manip_mdp.joint_velocity_hinge_penalty,
      weight=-0.5,
      params={"max_vel": 1.5, "asset_cfg": arm()},
    ),
    # -- stay physical ------------------------------------------------------
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-20.0, params={"asset_cfg": arm()}
    ),
    "cube_airborne": RewardTermCfg(
      func=push_mdp.object_airborne,
      weight=-20.0,
      params={"object_name": CUBE, "height": 0.055},
    ),
    "arm_below_table": RewardTermCfg(
      func=push_mdp.link_below_height,
      weight=-20.0,
      params={"min_height": 0.02, "asset_cfg": ghost_links()},
    ),
    "cube_past_goals": RewardTermCfg(
      func=push_mdp.object_outside_box,
      weight=-15.0,
      params={"object_name": CUBE, "x_range": GOAL_BOUNDS_X, "y_range": GOAL_BOUNDS_Y},
    ),
    # Losing the cube used to cost -2.0, exactly one goal, against the ~20 a
    # good episode banks -- so being reckless at the boundary was nearly free.
    "terminated": RewardTermCfg(func=mdp.is_terminated, weight=-500.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "cube_lost": TerminationTermCfg(
      func=push_mdp.object_out_of_bounds,
      # Only slightly wider than the goal bounds.  A cube shoved well past
      # them lands where the arm cannot get behind it any more, and with no
      # episode timeout in play mode it would sit there forever; ending the
      # episode instead makes losing the cube cost something.
      # Wider than the goal box by more than one push length, so a single
      # overshoot is recoverable; cube_past_goals supplies the pressure to
      # come back before it gets this far.
      params={
        "object_name": CUBE,
        "x_range": (0.22, 0.60),
        "y_range": (-0.28, 0.28),
        "z_max": 0.20,
      },
    ),
    "nan": TerminationTermCfg(func=mdp.nan_detection),
  }

  # Learn the task first, then tighten the style.  Ramping the penalties from
  # the start makes standing still the safest policy.
  curriculum = {
    "vel_hinge_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.5},
          {"step": 150 * STEPS_PER_ITERATION, "weight": -2.0},
          {"step": 300 * STEPS_PER_ITERATION, "weight": -5.0},
        ],
      },
    ),
    "reach_decay": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "reach_cube",
        "stages": [
          {"step": 0, "weight": 1.0},
          {"step": 250 * STEPS_PER_ITERATION, "weight": 0.2},
        ],
      },
    ),
    "align_decay": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "push_alignment",
        "stages": [
          {"step": 0, "weight": 1.5},
          {"step": 300 * STEPS_PER_ITERATION, "weight": 0.5},
        ],
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.5,
      entities={"robot": piper.get_push_robot_cfg(), CUBE: EntityCfg(spec_fn=get_cube_spec)},
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=piper.VIEWER_BODY,
      distance=1.2,
      elevation=-35.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(
      # A shut gripper against a 50 mm cube on a plane makes far more contacts
      # than the lifting tasks these numbers usually come from; overflow shows
      # up as the cube silently tunnelling through the paddle.
      nconmax=96,
      njmax=900,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,  # 200 Hz physics, 50 Hz control.
    episode_length_s=8.0,
  )

  if play:
    # Finite, unlike the usual play override: without a timeout a stalled env
    # sits motionless forever and misrepresents what the policy does.
    cfg.episode_length_s = 30.0
    cfg.curriculum = {}
    cfg.observations["actor"].enable_corruption = False

  return cfg
