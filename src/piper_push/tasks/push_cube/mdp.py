"""MDP terms specific to continuous multi-goal cube pushing.

Everything else the task needs already exists in ``mjlab.envs.mdp`` and
``mjlab.tasks.manipulation.mdp``; only what is genuinely new lives here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand, LiftingCommandCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_ROBOT = SceneEntityCfg("robot")


##
# Command.
##


def sector_sample(
  count: int, radius_range: tuple[float, float], half_angle: float, device: str
) -> torch.Tensor:
  """Uniform over the *area* of an annular sector centred on the base."""
  lo, hi = radius_range
  u = sample_uniform(0.0, 1.0, (count,), device=device)
  radius = torch.sqrt(lo * lo + u * (hi * hi - lo * lo))
  angle = sample_uniform(-half_angle, half_angle, (count,), device=device)
  return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)


def sector_clamp(
  xy: torch.Tensor, radius_range: tuple[float, float], half_angle: float
) -> torch.Tensor:
  """Nearest point of the sector, in polar coordinates."""
  radius = torch.norm(xy, dim=-1).clamp(*radius_range)
  angle = torch.atan2(xy[:, 1], xy[:, 0]).clamp(-half_angle, half_angle)
  return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)


def sector_violation(
  xy: torch.Tensor, radius_range: tuple[float, float], half_angle: float
) -> torch.Tensor:
  """Distance by which a point lies outside the sector, in metres."""
  radius = torch.norm(xy, dim=-1)
  radial = (radius_range[0] - radius).clamp_min(0.0) + (
    radius - radius_range[1]
  ).clamp_min(0.0)
  over = (torch.atan2(xy[:, 1], xy[:, 0]).abs() - half_angle).clamp_min(0.0)
  return radial + radius * over  # arc length, so both terms are metres


@dataclass(kw_only=True)
class PushCommandCfg(LiftingCommandCfg):
  """A planar push goal that follows the cube instead of teleporting it."""

  goal_z: float = 0.025
  """Height of the goal marker: the cube's centre when it rests on the table."""
  workspace_radius: tuple[float, float] = (0.39, 0.58)
  """Cube and goals live in an annular sector about the base, not a box.

  A revolute base makes the reachable set annular, and the near-centre corner
  of a box falls in the arm's inner blind spot: to push a cube outward the
  gripper has to stand *behind* it, closer in still, and below roughly 0.33 m
  it cannot.  Pushing inward stays easy from there, so successes ratchet the
  cube toward the base until it parks somewhere it can never be pushed out of.
  """
  workspace_half_angle: float = 0.5585
  """32 degrees either side of straight ahead."""
  cube_clearance_m: float = 0.12
  """Keep the cube this far from the gripper at reset, so a randomised arm
  posture does not start inside the cube and fling it."""
  goal_radius_range: tuple[float, float] = (0.06, 0.16)
  """How far ahead of the cube a new goal is placed."""
  min_goal_separation: float = 0.05
  """Must exceed ``success_threshold``: a goal that spawns on top of the cube
  would complete instantly and hand the policy a free bonus forever."""
  dwell_steps: int = 3
  """Consecutive in-radius steps required before a goal counts as reached."""
  resample_on_success: bool = True
  stall_timeout_s: float = 4.0
  """How long without completing a goal counts as stalled.  Healthy play
  completes one every 0.3 s, so this only fires on genuine failure."""

  def __post_init__(self) -> None:
    if self.min_goal_separation <= self.success_threshold:
      raise ValueError(
        f"min_goal_separation ({self.min_goal_separation}) must exceed "
        f"success_threshold ({self.success_threshold}); otherwise a freshly "
        "sampled goal can already be satisfied."
      )

  def build(self, env: "ManagerBasedRlEnv") -> "PushCommand":
    return PushCommand(self, env)


class PushCommand(LiftingCommand):
  """Planar goal for pushing, resampled on a timer *and* on success.

  Subclasses ``LiftingCommand`` because several stock manipulation terms
  (``object_to_goal_distance``, ``target_position``) ``isinstance``-check it.
  The cube is placed here rather than by a reset event.  That is not a style
  choice: ``_reset_idx`` runs events before command resampling but only calls
  ``sim.forward()`` afterwards, so a command that read the cube's pose on the
  reset path would see the *previous* episode's position and scatter the first
  goal of every episode.  Placing the cube here means the sampled position is
  known directly.  Mid-episode resamples leave the cube alone and read its live
  pose, which by then is current.
  """

  cfg: PushCommandCfg

  def __init__(self, cfg: PushCommandCfg, env: "ManagerBasedRlEnv"):
    # The base class teleports the object on every resample when this is set.
    cfg.object_pose_range = None
    super().__init__(cfg, env)
    self._robot: Entity = env.scene["robot"]
    self._ee_site = self._robot.find_sites(("grasp_site",))[0]
    zeros = torch.zeros(self.num_envs, device=self.device)
    self.just_succeeded = zeros.clone()
    self.goals_completed = zeros.clone()
    self._dwell = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._resetting = False
    # Counted off the global step counter rather than accumulated locally:
    # _update_metrics also runs on the reset path, so a local accumulator
    # would advance twice on any step where some env resets.
    self._last_goal_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.metrics["idle_time"] = zeros.clone()
    self.metrics["planar_error"] = zeros.clone()
    self.metrics["goals_completed"] = self.goals_completed

  @property
  def idle_time(self) -> torch.Tensor:
    """Seconds since this env last completed a goal (or was reset)."""
    steps = self._env.common_step_counter - self._last_goal_step
    return steps.clamp_min(0).float() * self._env.step_dt

  def stall_phase(self) -> torch.Tensor:
    return (self.idle_time / max(self.cfg.stall_timeout_s, 1e-6)).clamp(0.0, 1.0)

  def _cube_xy_local(self) -> torch.Tensor:
    return (self.object.data.root_link_pos_w - self._env.scene.env_origins)[:, :2]

  def _update_metrics(self) -> None:
    super()._update_metrics()
    # Cleared first: the reward at step t+1 consumes the flag set at step t,
    # and a flag that is never cleared pays the bonus on every single step.
    self.just_succeeded.zero_()

    # Planar error, so a cube that tips onto an edge is not scored as a miss.
    error = torch.norm(
      (self.target_pos - self.object.data.root_link_pos_w)[:, :2], dim=-1
    )
    inside = error < self.cfg.success_threshold
    self._dwell = torch.where(inside, self._dwell + 1, torch.zeros_like(self._dwell))
    hit = self._dwell >= self.cfg.dwell_steps
    self.just_succeeded[hit] = 1.0
    self.goals_completed += hit.float()
    self._dwell[hit] = 0

    if self.cfg.resample_on_success:
      # CommandTerm.compute() checks the timer right after this method, so
      # zeroing it here resamples within the very same step.
      self.time_left[hit] = 0.0

    self._last_goal_step[hit] = self._env.common_step_counter

    self.metrics["planar_error"] = error
    self.metrics["at_goal"] = inside.float()
    self.metrics["goals_completed"] = self.goals_completed
    self.metrics["idle_time"] = self.idle_time

  def _place_cube(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Drop the cube at a fresh pose and return its local xy."""
    count = len(env_ids)
    sector = (self.cfg.workspace_radius, self.cfg.workspace_half_angle)
    # The arm was randomised by a reset event moments ago, but derived
    # quantities only refresh on the next forward(); ask for one so the
    # clearance test below sees where the gripper actually is.
    self._env.sim.forward()
    ee_xy = (
      self._robot.data.site_pos_w[:, self._ee_site].squeeze(1)
      - self._env.scene.env_origins
    )[env_ids, :2]

    xy = sector_sample(count, *sector, self.device)
    for _ in range(4):
      too_close = torch.norm(xy - ee_xy, dim=-1) < self.cfg.cube_clearance_m
      if not bool(too_close.any()):
        break
      retry = sector_sample(count, *sector, self.device)
      xy = torch.where(too_close.unsqueeze(-1), retry, xy)

    # A hair above the half-extent so it settles instead of interpenetrating.
    z = torch.full((count, 1), self.cfg.goal_z + 1e-4, device=self.device)
    yaw = sample_uniform(-math.pi, math.pi, (count,), device=self.device)
    quat = quat_from_euler_xyz(
      torch.zeros(count, device=self.device), torch.zeros(count, device=self.device), yaw
    )
    pose = torch.cat([xy + self._env.scene.env_origins[env_ids, :2], z, quat], dim=-1)
    self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(count, 6, device=self.device), env_ids=env_ids
    )
    return xy

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    cube_xy = (
      self._place_cube(env_ids) if self._resetting else self._cube_xy_local()[env_ids]
    )
    sector = (self.cfg.workspace_radius, self.cfg.workspace_half_angle)

    goal_xy = torch.zeros(count, 2, device=self.device)
    pending = torch.ones(count, device=self.device, dtype=torch.bool)
    for _ in range(4):
      if not bool(pending.any()):
        break
      radius = sample_uniform(*self.cfg.goal_radius_range, (count,), device=self.device)
      angle = sample_uniform(-math.pi, math.pi, (count,), device=self.device)
      offset = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], -1)
      candidate = sector_clamp(cube_xy + offset, *sector)
      goal_xy = torch.where(pending.unsqueeze(-1), candidate, goal_xy)
      # Clamping can drag a good candidate back onto the cube, so the
      # separation test has to run after it, not before.
      pending = torch.norm(goal_xy - cube_xy, dim=-1) < self.cfg.min_goal_separation
    if bool(pending.any()):
      # Fallback: push straight toward the middle of the workspace.
      mid = 0.5 * (sector[0][0] + sector[0][1])
      centre = torch.tensor([mid, 0.0], device=self.device)
      away = centre - cube_xy
      away = away / away.norm(dim=-1, keepdim=True).clamp_min(1e-6)
      rescue = sector_clamp(cube_xy + away * self.cfg.goal_radius_range[0], *sector)
      goal_xy = torch.where(pending.unsqueeze(-1), rescue, goal_xy)

    goal_z = torch.full((count, 1), self.cfg.goal_z, device=self.device)
    self.target_pos[env_ids] = (
      torch.cat([goal_xy, goal_z], dim=-1) + self._env.scene.env_origins[env_ids]
    )
    self._dwell[env_ids] = 0
    self.episode_success[env_ids] = 0.0

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    self._resetting = True
    try:
      extras = super().reset(env_ids)
    finally:
      self._resetting = False
    self.just_succeeded[env_ids] = 0.0
    self.goals_completed[env_ids] = 0.0
    self._dwell[env_ids] = 0
    self._last_goal_step[env_ids] = self._env.common_step_counter
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    for batch in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self.target_pos[batch].cpu().numpy(),
        radius=float(self.cfg.success_threshold),
        color=self.cfg.viz.target_color,
        label=f"push_goal_{batch}",
      )


##
# Observations.
##


def _to_base_frame(robot: Entity, pos_w: torch.Tensor) -> torch.Tensor:
  return quat_apply(quat_inv(robot.data.root_link_quat_w), pos_w - robot.data.root_link_pos_w)


def ee_pose_b(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """End-effector position plus 6-D rotation, in the robot base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  pos_b = _to_base_frame(robot, robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1))
  quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  rotation = _rotation_6d(quat_inv(robot.data.root_link_quat_w), quat_w)
  return torch.cat([pos_b, rotation], dim=-1)


def object_pose_b(
  env: "ManagerBasedRlEnv", object_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Cube position plus 6-D rotation, in the robot base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  pos_b = _to_base_frame(robot, obj.data.root_link_pos_w)
  rotation = _rotation_6d(quat_inv(robot.data.root_link_quat_w), obj.data.root_link_quat_w)
  return torch.cat([pos_b, rotation], dim=-1)


def object_lin_vel_b(
  env: "ManagerBasedRlEnv", object_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  return quat_apply(quat_inv(robot.data.root_link_quat_w), obj.data.root_link_lin_vel_w)


def stall_phase(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """How close this env is to being declared stalled.

  Without it the stall termination would be invisible to the policy and the
  MDP would stop being Markov in exactly the dimension that ends the episode.
  """
  command = cast(PushCommand, env.command_manager.get_term(command_name))
  return command.stall_phase().unsqueeze(-1)


def goal_phase(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Fraction of the goal's resampling window that is left.

  The goal resamples on a timer, so without this the MDP is unobservable in
  exactly the dimension the value function needs.
  """
  command = cast(PushCommand, env.command_manager.get_term(command_name))
  horizon = max(command.cfg.resampling_time_range[1], 1e-6)
  return (command.time_left / horizon).clamp(0.0, 1.0).unsqueeze(-1)


def _rotation_6d(parent_quat_inv: torch.Tensor, child_quat_w: torch.Tensor) -> torch.Tensor:
  """First two columns of the relative rotation matrix.

  A continuous parameterisation: unlike yaw or a raw quaternion it has no
  wraparound or sign ambiguity for the network to fight.
  """
  matrix = matrix_from_quat(quat_mul(parent_quat_inv, child_quat_w))
  return matrix[..., :2].reshape(matrix.shape[0], 6)


##
# Rewards.
##


class goal_progress:
  """Closing speed toward the goal, in m/s.

  Integrated over a push this equals the distance closed, so the policy is paid
  for making progress *fast* rather than for loitering near the goal.  The
  stored distance is re-seeded whenever the goal is resampled, otherwise a
  resample would read as one enormous instantaneous gain.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self._previous = torch.zeros(env.num_envs, device=env.device)
    self._counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    object_name: str,
    clip: float = 0.5,
  ) -> torch.Tensor:
    command = cast(PushCommand, env.command_manager.get_term(command_name))
    obj: Entity = env.scene[object_name]
    distance = torch.norm((command.target_pos - obj.data.root_link_pos_w)[:, :2], dim=-1)
    stale = command.command_counter != self._counter
    progress = (self._previous - distance) / env.step_dt
    progress = torch.where(stale, torch.zeros_like(progress), progress)
    self._previous = distance
    self._counter = command.command_counter.clone()
    return progress.clamp(-clip, clip)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    index = slice(None) if env_ids is None else env_ids
    self._previous[index] = 0.0
    # Anything the live counter cannot equal, so the first step after a reset
    # is treated as stale and pays no progress.
    self._counter[index] = -1


def goal_reached_bonus(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  command = cast(PushCommand, env.command_manager.get_term(command_name))
  return command.just_succeeded


def push_alignment(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  contact_std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward standing *behind* the cube relative to the goal.

  Getting on the correct side is the entire difficulty of pushing, and no stock
  term expresses it: a pure distance reward is equally happy with the pusher
  parked between the cube and its goal.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(PushCommand, env.command_manager.get_term(command_name))
  ee_xy = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)[:, :2]
  cube_xy = obj.data.root_link_pos_w[:, :2]
  to_cube = cube_xy - ee_xy
  to_goal = command.target_pos[:, :2] - cube_xy
  distance = torch.norm(to_cube, dim=-1)
  alignment = torch.einsum(
    "bi,bi->b",
    to_cube / distance.clamp_min(1e-6).unsqueeze(-1),
    to_goal / torch.norm(to_goal, dim=-1).clamp_min(1e-6).unsqueeze(-1),
  )
  gate = torch.exp(-(distance**2) / contact_std**2)
  return alignment * gate


def object_airborne(
  env: "ManagerBasedRlEnv", object_name: str, height: float
) -> torch.Tensor:
  """Discourage scooping or launching the cube instead of pushing it."""
  obj: Entity = env.scene[object_name]
  z = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return (z - height).clamp_min(0.0)


def object_outside_workspace(
  env: "ManagerBasedRlEnv",
  object_name: str,
  radius_range: tuple[float, float],
  half_angle: float,
) -> torch.Tensor:
  """How far the cube has strayed past the region goals can be placed in.

  Termination alone is a cliff with no slope leading up to it: by the time it
  fires the cube is already gone.  This charges for the overshoot itself, so
  there is a gradient pushing the policy to keep the cube where it can still
  work with it.
  """
  obj: Entity = env.scene[object_name]
  pos = obj.data.root_link_pos_w - env.scene.env_origins
  return sector_violation(pos[:, :2], radius_range, half_angle)


def ee_above_height(
  env: "ManagerBasedRlEnv", max_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """How far the gripper has risen above the height pushing happens at.

  There was a floor penalty but no ceiling, and nothing else bounds the working
  volume. A fast push ends with the cube sliding away and the load vanishing,
  and the arm carries on along its commanded trajectory into a pose it has no
  business being in; from there it never comes back.
  """
  robot: Entity = env.scene[asset_cfg.name]
  z = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)[:, 2] - env.scene.env_origins[:, 2]
  return (z - max_height).clamp_min(0.0)


def link_below_height(
  env: "ManagerBasedRlEnv", min_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Keep the arm above the table.

  The forearm links carry no collision geometry (that keeps the contact set
  small and the solve fast), so nothing physically stops the policy from
  sweeping them through the floor -- poses that would be impossible on
  hardware.  This charges for it instead.
  """
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
  return torch.sum((min_height - z).clamp_min(0.0), dim=1)


##
# Terminations.
##


def stalled(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """No goal completed for stall_timeout_s.

  A stalled arm is motionless in a state the policy has no answer for, and
  nothing in the dynamics ever clears it -- so the episode has to.
  """
  command = cast(PushCommand, env.command_manager.get_term(command_name))
  return command.idle_time > command.cfg.stall_timeout_s


def object_out_of_bounds(
  env: "ManagerBasedRlEnv",
  object_name: str,
  radius_range: tuple[float, float],
  half_angle: float,
  z_max: float,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  pos = obj.data.root_link_pos_w - env.scene.env_origins
  outside = sector_violation(pos[:, :2], radius_range, half_angle) > 0.0
  return outside | (pos[:, 2] > z_max)
