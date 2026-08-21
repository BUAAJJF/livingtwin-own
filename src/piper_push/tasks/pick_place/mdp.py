"""Terms for PiPER-X tabletop tidying.

The interesting part of this file is what counts as a grasp and what counts as
a placement.  A dense reward can be generous -- it only has to point the way --
but the two success flags decide what the policy is actually optimising, and
every cheap way to satisfy them is a way to score without doing the task.  So
neither flag is a distance test:

* a grasp requires force on BOTH pads, the object off the table, the object
  moving WITH the hand, and all of that holding for several consecutive steps.
  Bumping the object satisfies none of it; resting it on a closed gripper
  fails the two-pad test; hooking it on a finger fails the relative-velocity
  test as soon as the hand turns.
* a placement requires the object inside the bin's footprint, its top below the
  rim, no pad touching it any more, and it at rest -- carried over the bin in a
  shut gripper is the exact state this excludes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.viewer import DebugVisualizer

FINGERTIP_DROP_M = 0.0123
"""How far the finger mesh reaches past ``grasp_site`` along the approach axis.

Measured in S0 from the compiled model: the pad reaches 11.5 mm, the finger
mesh 12.3 mm, and the mesh is what touches the table first.
"""


# ---------------------------------------------------------------------------
# Spawn geometry
# ---------------------------------------------------------------------------


def sector_sample(
  count: int,
  radius_range: tuple[float, float],
  angle_range: tuple[float, float],
  device: str | torch.device,
) -> torch.Tensor:
  """Uniform over the AREA of an annular sector, not over (r, theta).

  Sampling radius uniformly would crowd the inner edge, which is also the part
  of the workspace the arm finds hardest -- the sample would silently become a
  curriculum nobody chose.
  """
  lo, hi = radius_range
  u = sample_uniform(0.0, 1.0, (count,), device=device)
  radius = torch.sqrt(lo * lo + u * (hi * hi - lo * lo))
  angle = sample_uniform(angle_range[0], angle_range[1], (count,), device=device)
  return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)


def sector_violation(
  xy: torch.Tensor, radius_range: tuple[float, float], angle_range: tuple[float, float]
) -> torch.Tensor:
  """Distance outside an annular sector, in metres (arc length for the angle)."""
  radius = torch.norm(xy, dim=-1)
  radial = (radius_range[0] - radius).clamp_min(0.0) + (
    radius - radius_range[1]
  ).clamp_min(0.0)
  angle = torch.atan2(xy[:, 1], xy[:, 0])
  over = (angle_range[0] - angle).clamp_min(0.0) + (angle - angle_range[1]).clamp_min(
    0.0
  )
  return radial + radius * over


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class PickCommandCfg(CommandTermCfg):
  object_name: str = "object"
  robot_name: str = "robot"
  pad_sensor_name: str = "pad_contact"
  grasp_site: str = "grasp_site"

  spawn_radius: tuple[float, float] = (0.24, 0.46)
  spawn_angle: tuple[float, float] = (-0.14, 0.73)
  """Radians. Clear of the bin, which sits at -0.63 rad."""
  spawn_clearance_m: float = 0.09
  """How far the object spawns from wherever the arm was just reset to."""
  spawn_attempts: int = 6

  bin_center: tuple[float, float] = (0.30, -0.22)
  bin_inner: tuple[float, float] = (0.080, 0.070)
  bin_rim_z: float = 0.060
  release_clearance_m: float = 0.055
  """How far above the rim the drop target sits."""
  place_margin_m: float = 0.012
  """How far inside the footprint the object's centre must be to count."""

  grasp_force_n: float = 0.4
  """Per-pad normal force that counts as holding rather than brushing."""
  grasp_lift_m: float = 0.018
  """Clearance under the object before a grasp counts. Below this the table is
  still carrying it."""
  grasp_rel_vel: float = 0.12
  """m/s between object and grasp site. A hooked or balanced object breaks this
  the moment the hand accelerates."""
  grasp_reach_m: float = 0.09
  grasp_dwell: int = 3

  place_settle_vel: float = 0.06
  place_dwell: int = 5

  drop_penalty_height: float = 0.05
  """An object above this while NOT grasped has been thrown, not placed."""

  def build(self, env: "ManagerBasedRlEnv") -> "PickCommand":
    return PickCommand(self, env)


class PickCommand(CommandTerm):
  """Object placement, grasp/place bookkeeping, and the drop target.

  The object is placed here rather than by a reset event for the same reason
  the push task does it: ``_reset_idx`` runs events before command resampling
  but only calls ``sim.forward()`` afterwards, so a command that read the
  object's pose on the reset path would see the previous episode's position.
  """

  cfg: PickCommandCfg

  def __init__(self, cfg: PickCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self._robot: Entity = env.scene[cfg.robot_name]
    self._object: Entity = env.scene[cfg.object_name]
    self._pads = env.scene[cfg.pad_sensor_name]
    self._site = self._robot.find_sites((cfg.grasp_site,))[0][0]
    # find_geoms returns indices into the ENTITY's geom list; the per-world
    # model arrays are global, and the entity-local index there lands on the
    # terrain plane instead.
    local_geom = self._object.find_geoms(("object_geom",))[0][0]
    self._object_geom = int(self._object.indexing.geom_ids[local_geom])

    zeros = torch.zeros(self.num_envs, device=self.device)
    self.grasped = zeros.clone().bool()
    self.placed = zeros.clone().bool()
    self.just_grasped = zeros.clone()
    self.just_placed = zeros.clone()
    self.objects_placed = zeros.clone()
    self._grasp_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._place_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # Paid once per object.  A rising-edge bonus that pays every time is a
    # grab-drop-grab loop worth more than finishing the task, and the policy
    # will find it long before it finds the bin.
    self._grasp_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.grasp_attempts = torch.zeros(self.num_envs, device=self.device)
    self._resetting = False

    origin = env.scene.env_origins
    center = torch.tensor(cfg.bin_center, device=self.device)
    self._drop_local = torch.zeros(self.num_envs, 3, device=self.device)
    self._drop_local[:, :2] = center
    self._drop_local[:, 2] = cfg.bin_rim_z + cfg.release_clearance_m
    del origin

    self.metrics["objects_placed"] = self.objects_placed
    self.metrics["grasp_rate"] = zeros.clone()
    self.metrics["drop_error"] = zeros.clone()
    # Attempts per placement is the number that exposes a grab-drop loop: a
    # policy doing the task has one, a policy farming the bonus has many.
    self.metrics["grasp_attempts"] = self.grasp_attempts

  # -- geometry ------------------------------------------------------------

  @property
  def object_half_size(self) -> torch.Tensor:
    """Per-env half-extents. Read live because startup randomisation writes
    them into the per-world model after this term is built."""
    return torch.as_tensor(self._env.sim.model.geom_size[:])[:, self._object_geom]

  @property
  def command(self) -> torch.Tensor:
    """Drop target in the robot base frame."""
    return self._drop_local

  def _site_pos_w(self) -> torch.Tensor:
    return self._robot.data.site_pos_w[:, self._site]

  def _object_pos_local(self) -> torch.Tensor:
    return self._object.data.root_link_pos_w - self._env.scene.env_origins

  # -- state ---------------------------------------------------------------

  def _gripper_opening(self) -> torch.Tensor:
    idx = self._robot.find_joints(("gripper_joint1",))[0][0]
    return 2.0 * self._robot.data.joint_pos[:, idx]

  def _update_metrics(self) -> None:
    obj = self._object_pos_local()
    site = self._site_pos_w() - self._env.scene.env_origins
    half = self.object_half_size

    found = self._pads.data.found
    force = self._pads.data.force
    assert found is not None and force is not None
    per_pad = torch.linalg.norm(force, dim=-1)
    both = (found > 0).all(dim=1) & (per_pad > self.cfg.grasp_force_n).all(dim=1)

    rel_v = torch.linalg.norm(
      self._object.data.root_link_lin_vel_w - self._robot.data.site_lin_vel_w[:, self._site],
      dim=-1,
    )
    near = torch.linalg.norm(obj - site, dim=-1) < self.cfg.grasp_reach_m
    lifted = obj[:, 2] - half[:, 2] > self.cfg.grasp_lift_m
    secure = both & near & (rel_v < self.cfg.grasp_rel_vel) & lifted

    self._grasp_count = torch.where(
      secure, self._grasp_count + 1, torch.zeros_like(self._grasp_count)
    )
    grasped_now = self._grasp_count >= self.cfg.grasp_dwell
    fresh = grasped_now & ~self.grasped
    self.just_grasped = (fresh & ~self._grasp_paid).float()
    if not self._resetting:
      self.grasp_attempts += fresh.float()
    self._grasp_paid |= grasped_now
    self.grasped = grasped_now

    # -- placement
    delta = (obj[:, :2] - self._drop_local[:, :2]).abs()
    inner = torch.tensor(self.cfg.bin_inner, device=self.device)
    inside = (delta < (inner - self.cfg.place_margin_m)).all(dim=-1)
    # The object's BOTTOM below the rim, not its top: a 60 mm object in a
    # 60 mm bin has its top exactly at the rim when it is resting on the
    # floor, so the top rule is unsatisfiable for anything tall.  The bottom
    # rule still excludes the state it was written for -- carried over the bin
    # in a shut gripper leaves the bottom above the rim -- and ``released``
    # plus ``settled`` close the rest of that door.
    below_rim = obj[:, 2] - half[:, 2] < self.cfg.bin_rim_z - 0.005
    released = ~(found > 0).any(dim=1)
    settled = torch.linalg.norm(self._object.data.root_link_lin_vel_w, dim=-1) < (
      self.cfg.place_settle_vel
    )
    placed_now = inside & below_rim & released & settled
    self._place_count = torch.where(
      placed_now, self._place_count + 1, torch.zeros_like(self._place_count)
    )
    done = self._place_count >= self.cfg.place_dwell
    self.just_placed = (done & ~self.placed).float()
    self.placed = done

    if not self._resetting:
      self.objects_placed += self.just_placed
      self.metrics["grasp_rate"] = self.grasped.float()
      self.metrics["drop_error"] = torch.linalg.norm(
        obj - self._drop_local, dim=-1
      )
      respawn = self.just_placed.nonzero().flatten()
      if len(respawn) > 0:
        self._place_object(respawn)
        self._place_count[respawn] = 0
        self.placed[respawn] = False
        self._grasp_paid[respawn] = False

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._place_object(env_ids)
    self._grasp_count[env_ids] = 0
    self._place_count[env_ids] = 0
    self._grasp_paid[env_ids] = False
    self.grasp_attempts[env_ids] = 0.0
    self.grasped[env_ids] = False
    self.placed[env_ids] = False

  def reset(self, env_ids) -> dict[str, float]:
    self._resetting = True
    try:
      extras = super().reset(env_ids)
    finally:
      self._resetting = False
    return extras

  # -- placement -----------------------------------------------------------

  def _place_object(self, env_ids: torch.Tensor) -> None:
    """Drop the object somewhere reachable and clear of the hand.

    Clearance is beside-or-above, not planar: forbidding the whole column under
    the gripper would make "hand already over the object" an unreachable start
    state, and that is exactly the state a fresh grasp begins from.
    """
    count = len(env_ids)
    half = self.object_half_size[env_ids]
    site = (self._site_pos_w()[env_ids] - self._env.scene.env_origins[env_ids])
    tip_z = site[:, 2] - FINGERTIP_DROP_M

    xy = sector_sample(count, self.cfg.spawn_radius, self.cfg.spawn_angle, self.device)
    for _ in range(self.cfg.spawn_attempts):
      planar = torch.linalg.norm(xy - site[:, :2], dim=-1)
      clear = (planar > self.cfg.spawn_clearance_m) | (
        tip_z > 2.0 * half[:, 2] + self.cfg.spawn_clearance_m
      )
      if bool(clear.all()):
        break
      fresh = sector_sample(
        count, self.cfg.spawn_radius, self.cfg.spawn_angle, self.device
      )
      xy = torch.where(clear.unsqueeze(-1), xy, fresh)

    yaw = sample_uniform(-3.14159, 3.14159, (count,), device=self.device)
    pose = torch.zeros(count, 7, device=self.device)
    pose[:, :2] = xy
    pose[:, 2] = half[:, 2]
    pose[:, 3] = torch.cos(yaw / 2)
    pose[:, 6] = torch.sin(yaw / 2)
    pose[:, :3] += self._env.scene.env_origins[env_ids]
    self._object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self._object.write_root_link_velocity_to_sim(
      torch.zeros(count, 6, device=self.device), env_ids=env_ids
    )

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    target = self._drop_local + self._env.scene.env_origins
    for batch in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=target[batch].cpu().numpy(),
        radius=0.02,
        color=(0.2, 0.9, 0.4),
        label=f"drop_target_{batch}",
      )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def _rotation_6d(parent_quat_inv: torch.Tensor, child_quat_w: torch.Tensor):
  rel = quat_mul(parent_quat_inv, child_quat_w)
  mat = matrix_from_quat(rel)
  return torch.cat([mat[:, :, 0], mat[:, :, 1]], dim=-1)


def ee_pose_b(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  site = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  inv = quat_conjugate(robot.data.root_link_quat_w)
  pos = quat_apply_inverse(robot.data.root_link_quat_w, site - robot.data.root_link_pos_w)
  rot = _rotation_6d(inv, robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1))
  return torch.cat([pos, rot], dim=-1)


def object_pose_b(env: "ManagerBasedRlEnv", object_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  obj: Entity = env.scene[object_name]
  inv = quat_conjugate(robot.data.root_link_quat_w)
  pos = quat_apply_inverse(
    robot.data.root_link_quat_w, obj.data.root_link_pos_w - robot.data.root_link_pos_w
  )
  rot = _rotation_6d(inv, obj.data.root_link_quat_w)
  return torch.cat([pos, rot], dim=-1)


def object_lin_vel_b(env: "ManagerBasedRlEnv", object_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  obj: Entity = env.scene[object_name]
  return quat_apply_inverse(robot.data.root_link_quat_w, obj.data.root_link_lin_vel_w)


def ee_to_object(
  env: "ManagerBasedRlEnv", object_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  site = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return quat_apply_inverse(
    robot.data.root_link_quat_w, obj.data.root_link_pos_w - site
  )


def object_to_drop(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  cmd: PickCommand = env.command_manager.get_term(command_name)
  return cmd.command - cmd._object_pos_local()


def gripper_opening(env: "ManagerBasedRlEnv") -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  idx = robot.find_joints(("gripper_joint1",))[0][0]
  return 2.0 * robot.data.joint_pos[:, idx].unsqueeze(-1)


def grasp_state(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Whether the policy is currently holding the object.

  Visible to the actor on purpose: the reward switches behaviour on this flag,
  so hiding it would make the MDP non-Markov in exactly the dimension the task
  turns on.
  """
  cmd: PickCommand = env.command_manager.get_term(command_name)
  return cmd.grasped.float().unsqueeze(-1)


def pad_contact(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  found = env.scene[sensor_name].data.found
  assert found is not None
  return (found > 0).float()


def object_shape(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """The object's half-extents.

  Privileged in spirit -- a camera would have to infer it -- but the actor sees
  it in the state-only stage so that swapping in the image later is the only
  change that matters.
  """
  cmd: PickCommand = env.command_manager.get_term(command_name)
  return cmd.object_half_size


def object_physics(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Mass, table friction and centre-of-mass offset. Critic only."""
  cmd: PickCommand = env.command_manager.get_term(command_name)
  body = cmd._object.indexing.body_ids[0]
  mass = torch.as_tensor(env.sim.model.body_mass[:])[:, body].unsqueeze(-1)
  fric = torch.as_tensor(env.sim.model.geom_friction[:])[:, cmd._object_geom, 0:1]
  ipos = torch.as_tensor(env.sim.model.body_ipos[:])[:, body]
  return torch.cat([mass, fric, ipos], dim=-1)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def reach_object(
  env: "ManagerBasedRlEnv", command_name: str, object_name: str, std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Dense until the object is held, then off: a policy still being paid to
  hover near the object has a reason not to commit to lifting it."""
  cmd: PickCommand = env.command_manager.get_term(command_name)
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  site = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  d = torch.linalg.norm(obj.data.root_link_pos_w - site, dim=-1)
  return (1.0 - torch.tanh(d / std)) * (~cmd.grasped).float()


def pads_touching(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Both pads on the object. Both, not either: one pad is a shove."""
  cmd: PickCommand = env.command_manager.get_term(command_name)
  found = cmd._pads.data.found
  assert found is not None
  return ((found > 0).all(dim=1) & ~cmd.grasped).float()


def holding(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Paid every step the object is held.

  Without it, taking hold of the object is a pay CUT.  ``reach`` and
  ``pads_touching`` are both gated on not-grasped and switch off together at
  the instant of the grasp, while ``lift`` and ``transport`` are still near
  zero because the object has only just left the table: measured, 2.66 per step
  before against 0.75 after.  A one-shot acquisition bonus pays for that once;
  every step afterwards the policy is worse off for holding on, and across five
  configurations and 3000 iterations four of them learned to touch the object
  and never close on it.
  """
  cmd: PickCommand = env.command_manager.get_term(command_name)
  return cmd.grasped.float()


class transport_progress:
  """Distance the held object closed on the drop point since the last step.

  A progress term rather than a potential, and that is the point: a potential
  saturates when the object arrives over the bin, so hovering there collects it
  forever and letting go costs a fortune.  Progress pays for the trip and
  nothing for standing still, which leaves the placement bonus as the only
  thing left to earn.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    del cfg
    self._env = env
    self._previous = torch.zeros(env.num_envs, device=env.device)
    self._valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str, clip: float = 0.05
  ) -> torch.Tensor:
    cmd: PickCommand = env.command_manager.get_term(command_name)
    dist = torch.linalg.norm(cmd.command - cmd._object_pos_local(), dim=-1)
    delta = (self._previous - dist).clamp(-clip, clip)
    # No credit for the step the object was picked up or put down on: the
    # distance jumps then for reasons that are not travel.
    out = delta * (self._valid & cmd.grasped).float()
    self._previous = dist
    self._valid = cmd.grasped.clone()
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._valid[env_ids] = False


def grasp_bonus(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  cmd: PickCommand = env.command_manager.get_term(command_name)
  return cmd.just_grasped


def place_bonus(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  cmd: PickCommand = env.command_manager.get_term(command_name)
  return cmd.just_placed


def transport(
  env: "ManagerBasedRlEnv", command_name: str, std: float
) -> torch.Tensor:
  """Only while the object is actually held."""
  cmd: PickCommand = env.command_manager.get_term(command_name)
  d = torch.linalg.norm(cmd.command - cmd._object_pos_local(), dim=-1)
  return (1.0 - torch.tanh(d / std)) * cmd.grasped.float()


def lift_height(
  env: "ManagerBasedRlEnv", command_name: str, target: float
) -> torch.Tensor:
  cmd: PickCommand = env.command_manager.get_term(command_name)
  z = cmd._object_pos_local()[:, 2] - cmd.object_half_size[:, 2]
  return (z.clamp(0.0, target) / target) * cmd.grasped.float()


def object_in_bin(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Dense credit for the object being down inside the bin, held or not.

  Off by default (weight 0).  It exists because the last step of the task is a
  cliff: while the object is held over the bin the transport reward is already
  saturated, so nothing points toward opening the hand, and the 300-point
  placement bonus has to be found by chance.  This is the ramp up to it.

  It cannot be collected without the object genuinely being in the bin, and the
  placement bonus still requires the release, the settle and the dwell.
  """
  cmd: PickCommand = env.command_manager.get_term(command_name)
  pos = cmd._object_pos_local()
  inside = (
    (pos[:, :2] - torch.tensor(cmd.cfg.bin_center, device=pos.device)).abs()
    < torch.tensor(cmd.cfg.bin_inner, device=pos.device)
  ).all(dim=-1)
  below_rim = pos[:, 2] - cmd.object_half_size[:, 2] < cmd.cfg.bin_rim_z
  return (inside & below_rim).float()


def object_thrown(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Airborne without being held -- except over the bin, where that is the task.

  Knocking the object into the air is the cheapest way to satisfy any height
  reward, so it is charged directly.  But an unconditional version charges the
  release: the instant the gripper opens over the bin the object is unheld and
  falling, and the policy learns never to let go.  The term then reads as
  harmless (measured -0.0002) precisely because it is being obeyed.
  """
  cmd: PickCommand = env.command_manager.get_term(command_name)
  pos = cmd._object_pos_local()
  z = pos[:, 2] - cmd.object_half_size[:, 2]
  over_bin = (
    (pos[:, :2] - torch.tensor(cmd.cfg.bin_center, device=pos.device)).abs()
    < torch.tensor(cmd.cfg.bin_inner, device=pos.device)
  ).all(dim=-1)
  loose = (~cmd.grasped) & (~over_bin)
  return (z - cmd.cfg.drop_penalty_height).clamp_min(0.0) * loose.float()


def object_outside_spawn(
  env: "ManagerBasedRlEnv",
  command_name: str,
  radius_range: tuple[float, float],
  angle_range: tuple[float, float],
) -> torch.Tensor:
  """Charged only when the object is loose. Inside the bin it is meant to be
  outside the spawn sector."""
  cmd: PickCommand = env.command_manager.get_term(command_name)
  xy = cmd._object_pos_local()[:, :2]
  in_bin = (
    (xy - torch.tensor(cmd.cfg.bin_center, device=xy.device)).abs()
    < torch.tensor(cmd.cfg.bin_inner, device=xy.device)
  ).all(dim=-1)
  return sector_violation(xy, radius_range, angle_range) * (~in_bin).float()


def ee_outside_envelope(
  env: "ManagerBasedRlEnv",
  radius_range: tuple[float, float],
  angle_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  xy = (
    robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)[:, :2]
    - env.scene.env_origins[:, :2]
  )
  return sector_violation(xy, radius_range, angle_range)


def ee_above_height(
  env: "ManagerBasedRlEnv", max_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  z = (
    robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)[:, 2]
    - env.scene.env_origins[:, 2]
  )
  return (z - max_height).clamp_min(0.0)


def link_below_height(
  env: "ManagerBasedRlEnv", min_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """The ghost links carry no collision geometry, so nothing stops them going
  through the table; this is what stops them instead."""
  robot: Entity = env.scene[asset_cfg.name]
  z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[
    :, 2
  ].unsqueeze(-1)
  return (min_height - z).clamp_min(0.0).sum(dim=-1)


def joint_speed_over_trip(
  env: "ManagerBasedRlEnv", limits: dict[str, float], asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """How far past the deployment safety shell's trip points the arm is going."""
  robot: Entity = env.scene[asset_cfg.name]
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids].abs()
  names = [robot.joint_names[i] for i in asset_cfg.joint_ids]
  cap = torch.tensor(
    [limits.get(n, float("inf")) for n in names], device=vel.device
  )
  return (vel - cap).clamp_min(0.0).sum(dim=-1)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def object_lost(
  env: "ManagerBasedRlEnv",
  command_name: str,
  radius_range: tuple[float, float],
  angle_range: tuple[float, float],
  z_max: float,
) -> torch.Tensor:
  cmd: PickCommand = env.command_manager.get_term(command_name)
  pos = cmd._object_pos_local()
  xy = pos[:, :2]
  in_bin = (
    (xy - torch.tensor(cmd.cfg.bin_center, device=xy.device)).abs()
    < torch.tensor(cmd.cfg.bin_inner, device=xy.device)
  ).all(dim=-1)
  out = sector_violation(xy, radius_range, angle_range) > 0.0
  return (out & ~in_bin) | (pos[:, 2] > z_max)


def joint_velocity_trip(
  env: "ManagerBasedRlEnv", limits: dict[str, float], asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """The safety shell ends the run on hardware; it ends the episode here."""
  robot: Entity = env.scene[asset_cfg.name]
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids].abs()
  names = [robot.joint_names[i] for i in asset_cfg.joint_ids]
  cap = torch.tensor(
    [limits.get(n, float("inf")) for n in names], device=vel.device
  )
  return (vel > cap).any(dim=-1)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def reset_arm_valid_posture(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  position_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  ee_cfg: SceneEntityCfg,
  link_cfg: SceneEntityCfg,
  min_ee_height: float = 0.05,
  min_link_height: float = 0.03,
  attempts: int = 6,
) -> None:
  """Start from a varied posture, but never from inside the table.

  Rejection sampling rather than a narrow range: the push task learned that a
  policy trained around one posture cannot recover from any other, and that a
  wide range without this check puts a fifth of episodes underground.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  default = robot.data.default_joint_pos[env_ids][:, joint_ids]
  limits = robot.data.soft_joint_pos_limits[env_ids][:, joint_ids]
  zero = torch.zeros_like(default)
  accepted = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
  chosen = default.clone()
  origins = env.scene.env_origins[env_ids]

  for _ in range(attempts):
    candidate = default + sample_uniform(
      *position_range, default.shape, device=env.device
    )
    candidate = torch.max(torch.min(candidate, limits[..., 1]), limits[..., 0])
    chosen = torch.where(accepted.unsqueeze(-1), chosen, candidate)
    robot.write_joint_state_to_sim(chosen, zero, joint_ids=joint_ids, env_ids=env_ids)
    env.sim.forward()
    ee_z = (
      robot.data.site_pos_w[env_ids][:, ee_cfg.site_ids].squeeze(1)[:, 2]
      - origins[:, 2]
    )
    link_z = (
      robot.data.body_link_pos_w[env_ids][:, link_cfg.body_ids, 2] - origins[:, 2:3]
    )
    accepted |= (ee_z > min_ee_height) & (link_z.min(dim=1).values > min_link_height)
    if bool(accepted.all()):
      break

  final = torch.where(accepted.unsqueeze(-1), chosen, default)
  robot.write_joint_state_to_sim(final, zero, joint_ids=joint_ids, env_ids=env_ids)
