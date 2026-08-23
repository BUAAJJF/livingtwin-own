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

import mujoco
import torch
import torch.nn.functional as F
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

from piper_push import depth_noise, objects, shapes

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
  """The object, when there is one.  Ignored if ``object_names`` is set."""
  object_names: tuple[str, ...] = ()
  """Several objects on the table at once, cleared one at a time.

  The command owns which one is the target; every reward, metric and
  termination reads the target through the same accessors they used when there
  was only ever one, so clutter changes what the policy sees and not how the
  task is scored.  Empty means the single-object task."""
  pad_sensor_names: tuple[str, ...] = ()
  """One pad sensor per object, in the same order as ``object_names``.

  A single sensor filtered to all of them would report that a pad is touching
  *an* object, and every judgement here -- is it grasped, was it carried,
  did it settle in the bin -- is about a particular one."""
  robot_name: str = "robot"
  pad_sensor_name: str = "pad_contact"
  grasp_site: str = "grasp_site"

  spawn_radius: tuple[float, float] = (0.24, 0.46)
  spawn_angle: tuple[float, float] = (-0.14, 0.73)
  """Radians. Clear of the bin, which sits at -0.63 rad."""
  spawn_clearance_m: float = 0.09
  spawn_object_gap: float = 0.012
  """Clear space between two spawned objects, on top of their half-widths.

  Only enough to keep them from spawning inside one another.  Objects landing
  next to each other is the point of a cleanup task, not something to design
  out."""
  stray_radius: tuple[float, float] = (0.10, 0.62)
  """An object outside this planar band has been batted out of the workspace.

  It is put back rather than written off: with several objects the table can
  only be cleared if every object is reachable, so one knocked into the corner
  would stall the episode for as long as it lasted.  The cost shows up as
  ``objects_strayed`` instead of as a deadlock."""
  redraw_on_place: tuple[str, ...] = ()
  """Which object parameters are redrawn when an object is put back.

  This is the *cadence* knob, and it is deliberately separate from the ranges
  the parameters are drawn from, which live in the reset event and do not
  change.  Every setting below draws from identical marginals; the only thing
  that differs is how long a value is held.

    ``("shape", "mass", "friction")``  every object is a new object (OBJ-All)
    ``()``                             one object per episode, re-posed (EP-All)
    any subset                         that quantity turns over per object and
                                       the rest are held for the episode

  The reset event always draws all three, so an episode always *starts* with a
  fresh object; the question this controls is what happens at the ninth
  placement of the same episode.

  Redrawing costs about 14% of training throughput at 1024 environments,
  because the per-world model fields it writes are only visible after the
  derived constants are recomputed, and at scale a placement happens on nearly
  every step."""

  reshape_on_place: bool = False
  """Backwards-compatible alias: true means redraw everything on placement.

  Kept because it is what the trained checkpoints' recorded configs say.  When
  ``redraw_on_place`` is left empty this is what decides the cadence, so the
  default configuration means exactly what it did before the split.

  The shape randomiser is a reset event, so without either of these the whole
  episode is one geometry re-posed, and a recurrent policy can identify it once
  and coast on that for the remaining dozen placements.  Measured on the vision
  policy: 58.2 objects a minute with the shape held for the episode, 54.0 when
  every object is a new one.  The state teacher, which is fed the shape and has
  no memory to carry, loses 2.0% over the same change."""

  spawn_attempts: int = 6
  """How many times to resample a spawn pose before giving up on clearance."""

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
    self._names = tuple(cfg.object_names) or (cfg.object_name,)
    self._objects: list[Entity] = [env.scene[n] for n in self._names]
    pad_names = tuple(cfg.pad_sensor_names) or (cfg.pad_sensor_name,)
    assert len(pad_names) == len(self._names), (
      f"{len(self._names)} objects but {len(pad_names)} pad sensors"
    )
    self._pads_all = [env.scene[n] for n in pad_names]
    self._site = self._robot.find_sites((cfg.grasp_site,))[0][0]
    # All three parts of each object, for the segmentation mask: an object
    # that is currently a cylinder has its core collapsed to a millimetre, so
    # a mask built from the core alone would be empty exactly when the shape
    # is not a box.
    #
    # find_geoms returns indices into the ENTITY's geom list; the per-world
    # model arrays are global, and the entity-local index there lands on the
    # terrain plane instead.
    self._geom_table = torch.tensor(
      [
        [int(o.indexing.geom_ids[i])
         for i in o.find_geoms(objects.OBJECT_GEOMS, preserve_order=True)[0]]
        for o in self._objects
      ],
      dtype=torch.long,
      device=self.device,
    )

    zeros = torch.zeros(self.num_envs, device=self.device)
    self.grasped = zeros.clone().bool()
    self.placed = zeros.clone().bool()
    self.just_grasped = zeros.clone()
    self.just_placed = zeros.clone()
    self._shape_terms: dict[int, object | None] = {}
    self.objects_placed = zeros.clone()
    self._grasp_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._place_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._knock_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._knocked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # Paid once per object.  A rising-edge bonus that pays every time is a
    # grab-drop-grab loop worth more than finishing the task, and the policy
    # will find it long before it finds the bin.
    self._grasp_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # Which object is being cleared, and which are already in the bin.  With
    # one object the target is always zero and ``_cleared`` is the placed flag
    # under another name, so the single-object path is unchanged.
    self.target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._cleared = torch.zeros(
      self.num_envs, len(self._names), dtype=torch.bool, device=self.device
    )
    self._rows = torch.arange(self.num_envs, device=self.device)
    # Whole tables emptied, and objects that had to be fetched back from
    # outside the workspace.  Both only mean anything with clutter.
    self.table_clears = torch.zeros(self.num_envs, device=self.device)
    self.objects_strayed = torch.zeros(self.num_envs, device=self.device)
    self._retarget_pending = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.grasp_attempts = torch.zeros(self.num_envs, device=self.device)
    # Objects that reached the bin without ever having been picked up.  Logged
    # rather than silently discarded: a suppressed exploit is one you stop
    # being able to see.
    self.knocked_in = torch.zeros(self.num_envs, device=self.device)
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
    self.metrics["knocked_in"] = self.knocked_in
    if self.num_objects > 1:
      self.metrics["table_clears"] = self.table_clears
      self.metrics["objects_strayed"] = self.objects_strayed

  # -- geometry ------------------------------------------------------------

  @property
  def num_objects(self) -> int:
    return len(self._names)

  @property
  def redraw_on_place(self) -> tuple[str, ...]:
    """Which object parameters are redrawn on placement.

    Resolved from the config on every read rather than cached in ``__init__``,
    so a cadence experiment can set it on the built environment without
    rebuilding the scene -- which for the vision task means not re-rendering
    and re-allocating a camera to change one tuple.

    The explicit ``redraw_on_place`` wins; ``reshape_on_place`` is the older
    boolean spelling of "all of them", and is what the trained checkpoints'
    recorded configs contain.
    """
    if self.cfg.redraw_on_place:
      return tuple(self.cfg.redraw_on_place)
    return shapes.ALL_QUANTITIES if self.cfg.reshape_on_place else ()

  @property
  def target_geom_ids(self) -> torch.Tensor:
    """Global geom ids the target mask should light up, per environment.

    Per environment rather than a constant, because with several objects on
    the table the mask has to say *which one*, and that changes as each is
    cleared.
    """
    return self._geom_table[self.target]

  def _gather(self, per_object: torch.Tensor) -> torch.Tensor:
    """Pick out the target's row from a (B, N, ...) stack."""
    return per_object[self._rows, self.target]

  def _stack(self, read) -> torch.Tensor:
    """Read the same quantity off every object as (B, N, ...)."""
    return torch.stack([read(o) for o in self._objects], dim=1)

  @property
  def all_half_sizes(self) -> torch.Tensor:
    """(B, N, 3) bounding half-extents, one row per object."""
    return torch.stack(
      [shapes.object_half_size(self._env, n) for n in self._names], dim=1
    )

  @property
  def object_half_size(self) -> torch.Tensor:
    """Bounding half-extents of the target, per environment.

    Not one geom's size: an object is up to three parts, and the spawn height,
    the lift test and the bin-rim test all want the extent of the whole body
    rather than of whichever part comes first.
    """
    return self._gather(self.all_half_sizes)

  @property
  def all_pos_local(self) -> torch.Tensor:
    """(B, N, 3) object positions in the environment's own frame."""
    return self._stack(lambda o: o.data.root_link_pos_w) - (
      self._env.scene.env_origins.unsqueeze(1)
    )

  @property
  def command(self) -> torch.Tensor:
    """Drop target in the robot base frame."""
    return self._drop_local

  def _site_pos_w(self) -> torch.Tensor:
    return self._robot.data.site_pos_w[:, self._site]

  def _object_pos_local(self) -> torch.Tensor:
    return self._gather(self.all_pos_local)

  # -- state ---------------------------------------------------------------

  @property
  def pad_found(self) -> torch.Tensor:
    """(B, 2) contact flags for the pads, against the target only."""
    return self._gather(torch.stack([p.data.found for p in self._pads_all], dim=1))

  @property
  def target_core_geom(self) -> torch.Tensor:
    """(B,) global geom id of the target's core, for per-world model lookups."""
    return self._geom_table[self.target, 0]

  @property
  def cleared(self) -> torch.Tensor:
    """(B, N) which objects are already in the bin."""
    return self._cleared

  def target_pos_w(self) -> torch.Tensor:
    return self._gather(self._stack(lambda o: o.data.root_link_pos_w))

  def target_quat_w(self) -> torch.Tensor:
    return self._gather(self._stack(lambda o: o.data.root_link_quat_w))

  def target_lin_vel_w(self) -> torch.Tensor:
    return self._target_lin_vel_w()

  def _target_lin_vel_w(self) -> torch.Tensor:
    return self._gather(self._stack(lambda o: o.data.root_link_lin_vel_w))

  def _gripper_opening(self) -> torch.Tensor:
    idx = self._robot.find_joints(("gripper_joint1",))[0][0]
    return 2.0 * self._robot.data.joint_pos[:, idx]

  def _update_metrics(self) -> None:
    if bool(self._retarget_pending.any()):
      self._retarget(self._retarget_pending.nonzero().flatten())
      self._retarget_pending[:] = False
    obj = self._object_pos_local()
    site = self._site_pos_w() - self._env.scene.env_origins
    half = self.object_half_size

    # The pad sensor belonging to the target, not any pad sensor: touching a
    # different object is not a grasp of this one.
    found = self._gather(torch.stack([p.data.found for p in self._pads_all], dim=1))
    force = self._gather(torch.stack([p.data.force for p in self._pads_all], dim=1))
    assert found is not None and force is not None
    per_pad = torch.linalg.norm(force, dim=-1)
    both = (found > 0).all(dim=1) & (per_pad > self.cfg.grasp_force_n).all(dim=1)

    rel_v = torch.linalg.norm(
      self._target_lin_vel_w() - self._robot.data.site_lin_vel_w[:, self._site],
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
    settled = torch.linalg.norm(self._target_lin_vel_w(), dim=-1) < (
      self.cfg.place_settle_vel
    )
    # "It ended up in the bin" is not "it was put in the bin".  Every clause
    # below was satisfied by a policy that batted the object in and never
    # closed its fingers once: 14.37 placements an episode at a grasp rate of
    # 0.0000.  The object has to have been carried there, so a placement only
    # counts if a genuine grasp of THIS object happened first.
    in_bin_now = inside & below_rim & released & settled
    placed_now = in_bin_now & self._grasp_paid
    self._place_count = torch.where(
      placed_now, self._place_count + 1, torch.zeros_like(self._place_count)
    )
    done = self._place_count >= self.cfg.place_dwell
    self.just_placed = (done & ~self.placed).float()
    self.placed = done
    # The same dwell test on the objects that arrived without a grasp, so the
    # exploit stays measurable.
    self._knock_count = torch.where(
      in_bin_now & ~self._grasp_paid,
      self._knock_count + 1,
      torch.zeros_like(self._knock_count),
    )
    knocked_now = self._knock_count >= self.cfg.place_dwell
    just_knocked = (knocked_now & ~self._knocked).float()
    self._knocked = knocked_now

    if not self._resetting:
      self.objects_placed += self.just_placed
      self.knocked_in += just_knocked
      self.metrics["grasp_rate"] = self.grasped.float()
      self.metrics["drop_error"] = torch.linalg.norm(
        obj - self._drop_local, dim=-1
      )
      # An object that arrived by being knocked in is respawned too -- leaving
      # it there would let the policy bat one object in and then farm the
      # reach reward on an empty table.
      respawn = (self.just_placed + just_knocked).nonzero().flatten()
      if len(respawn) > 0:
        self._place_count[respawn] = 0
        self._knock_count[respawn] = 0
        self.placed[respawn] = False
        self._knocked[respawn] = False
        self._grasp_paid[respawn] = False
        if self.num_objects == 1:
          self._place_object(respawn)
        else:
          # It stays in the bin.  The table refills only once it is empty,
          # which is what makes this a cleanup task rather than the same
          # pick-and-place with spectators.
          self._cleared[respawn, self.target[respawn]] = True
          empty = self._cleared[respawn].all(dim=-1)
          if (~empty).any():
            self._retarget(respawn[~empty])
          if empty.any():
            self.table_clears[respawn[empty]] += 1.0
            self._place_all(respawn[empty])

      self._recover_strays()

  def _recover_strays(self) -> None:
    """Put back any object batted out of reach.

    Without this the table can stop being clearable: one object nudged into a
    corner is never picked up, the set never empties, and the environment
    spends the rest of the episode on a task it cannot finish.  Putting it back
    keeps the episode productive and leaves the cost visible in the metric
    rather than hidden in a stalled reward.
    """
    if self.num_objects == 1:
      return
    r = torch.linalg.norm(self.all_pos_local[:, :, :2], dim=-1)
    lo, hi = self.cfg.stray_radius
    astray = ((r < lo) | (r > hi)) & ~self._cleared
    if not bool(astray.any()):
      return
    self.objects_strayed += astray.sum(dim=-1).float()
    for idx in range(self.num_objects):
      rows = astray[:, idx].nonzero().flatten()
      if rows.numel():
        self._place_one(idx, rows)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.num_objects == 1:
      self._place_object(env_ids)
    else:
      self._place_all(env_ids)
      self.table_clears[env_ids] = 0.0
      self.objects_strayed[env_ids] = 0.0
    self._grasp_count[env_ids] = 0
    self._place_count[env_ids] = 0
    self._knock_count[env_ids] = 0
    self._grasp_paid[env_ids] = False
    self.grasp_attempts[env_ids] = 0.0
    self.knocked_in[env_ids] = 0.0
    self.grasped[env_ids] = False
    self.placed[env_ids] = False
    self._knocked[env_ids] = False

  def reset(self, env_ids) -> dict[str, float]:
    self._resetting = True
    try:
      extras = super().reset(env_ids)
    finally:
      self._resetting = False
    return extras

  # -- placement -----------------------------------------------------------

  def _reshape(self, idx: int, env_ids: torch.Tensor) -> None:
    """Redraw one object's geometry before it is put back on the table.

    Before, not after: the placement height is computed from the object's
    half-extent, so a taller object dropped into the pose chosen for a shorter
    one starts inside the table.

    The randomiser and its parameters are read off the reset event rather than
    duplicated here, so there is one description of what an object can be.  The
    lookup is lazy because the command manager is built before the event
    manager, and skipped entirely if the task has no shape event -- the fixed
    cube variant does not.
    """
    from mjlab.managers.event_manager import RecomputeLevel

    if idx not in self._shape_terms:
      name = "object_shape" if self.num_objects == 1 else f"object_shape_{idx}"
      try:
        self._shape_terms[idx] = self._env.event_manager.get_term_cfg(name)
      except (KeyError, ValueError):
        self._shape_terms[idx] = None
    term = self._shape_terms[idx]
    if term is None:
      return
    # The event's own params carry the ranges; only the cadence is overridden
    # here, so there is still one description of what an object can be.
    params = dict(term.params)
    params["redraw"] = self.redraw_on_place
    term.func(self._env, env_ids, **params)
    self._env.sim.recompute_constants(RecomputeLevel.set_const)

  def _place_object(self, env_ids: torch.Tensor) -> None:
    """Put each environment's current target back on the table.

    Dispatched per object index rather than vectorised across them, because
    which object is the target differs by environment and the write goes to a
    different entity for each.  With three objects that is three small writes.
    """
    if self.num_objects == 1:
      self._place_one(0, env_ids)
      return
    target = self.target[env_ids]
    for idx in range(self.num_objects):
      rows = env_ids[target == idx]
      if rows.numel():
        self._place_one(idx, rows)

  def _place_all(self, env_ids: torch.Tensor) -> None:
    """Refill the table.  Used on reset and once the last object is cleared."""
    for idx in range(self.num_objects):
      self._place_one(idx, env_ids)
    self._cleared[env_ids] = False
    # Choosing the nearest object needs the poses that were just written, and
    # on the reset path they are not readable yet: _reset_idx runs the events
    # and resamples the command before sim.forward(), so a read here returns
    # the previous episode's positions.  Aim at the first object, which is
    # always a valid uncleared one, and choose properly on the next step.
    self.target[env_ids] = 0
    self._retarget_pending[env_ids] = True

  def _place_one(self, idx: int, env_ids: torch.Tensor) -> None:
    """Drop one object somewhere reachable, clear of the hand and of the rest.

    Clearance from the hand is beside-or-above, not planar: forbidding the
    whole column under the gripper would make "hand already over the object"
    an unreachable start state, and that is exactly the state a fresh grasp
    begins from.

    Clearance from the other objects is only enough to keep them from spawning
    inside one another.  Objects landing next to each other is the point of
    this task, not something to design out.
    """
    if self.redraw_on_place and not self._resetting:
      self._reshape(idx, env_ids)
    count = len(env_ids)
    half = self.all_half_sizes[env_ids, idx]
    site = (self._site_pos_w()[env_ids] - self._env.scene.env_origins[env_ids])
    tip_z = site[:, 2] - FINGERTIP_DROP_M

    others = None
    if self.num_objects > 1:
      pos = self.all_pos_local[env_ids]                       # (count, N, 3)
      keep = [j for j in range(self.num_objects) if j != idx]
      others = pos[:, keep, :2]                               # (count, N-1, 2)
      # Half-widths of this object and each other one, so the test scales with
      # what was actually drawn rather than with the widest thing possible.
      mine = half[:, :2].amax(dim=-1, keepdim=True)           # (count, 1)
      theirs = self.all_half_sizes[env_ids][:, keep, :2].amax(dim=-1)
      self._min_sep = mine + theirs + self.cfg.spawn_object_gap

    xy = sector_sample(count, self.cfg.spawn_radius, self.cfg.spawn_angle, self.device)
    for _ in range(self.cfg.spawn_attempts):
      planar = torch.linalg.norm(xy - site[:, :2], dim=-1)
      clear = (planar > self.cfg.spawn_clearance_m) | (
        tip_z > 2.0 * half[:, 2] + self.cfg.spawn_clearance_m
      )
      if others is not None:
        gap = torch.linalg.norm(xy.unsqueeze(1) - others, dim=-1)
        clear = clear & (gap > self._min_sep).all(dim=-1)
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
    obj = self._objects[idx]
    obj.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    obj.write_root_link_velocity_to_sim(
      torch.zeros(count, 6, device=self.device), env_ids=env_ids
    )

  def _retarget(self, env_ids: torch.Tensor) -> None:
    """Aim at the nearest object still on the table.

    Nearest to the hand, and only re-evaluated when an object is cleared: a
    target that tracked the gripper continuously would let the policy change
    its mind by moving, and the reward would follow it around instead of
    driving it anywhere.
    """
    if self.num_objects == 1:
      return
    pos = self.all_pos_local[env_ids]
    site = (self._site_pos_w()[env_ids] - self._env.scene.env_origins[env_ids])
    dist = torch.linalg.norm(pos - site.unsqueeze(1), dim=-1)
    dist = dist.masked_fill(self._cleared[env_ids], float("inf"))
    self.target[env_ids] = dist.argmin(dim=-1)

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    target = self._drop_local + self._env.scene.env_origins
    for batch in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=target[batch].cpu().numpy(),
        radius=0.02,
        # RGBA, not RGB: the offscreen renderer hands this straight to
        # mjv_initGeom, which rejects a three-element colour.
        color=(0.2, 0.9, 0.4, 1.0),
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


# These read the target through the command rather than an entity by name.
# With one object the two are the same thing; with several, "the object" is a
# question only the command can answer, and having two answers to it is how a
# reward ends up shaping towards one object while the metric scores another.


def object_pose_b(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  cmd: PickCommand = env.command_manager.get_term(command_name)
  inv = quat_conjugate(robot.data.root_link_quat_w)
  pos = quat_apply_inverse(
    robot.data.root_link_quat_w, cmd.target_pos_w() - robot.data.root_link_pos_w
  )
  rot = _rotation_6d(inv, cmd.target_quat_w())
  return torch.cat([pos, rot], dim=-1)


def object_lin_vel_b(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  cmd: PickCommand = env.command_manager.get_term(command_name)
  return quat_apply_inverse(robot.data.root_link_quat_w, cmd.target_lin_vel_w())


def ee_to_object(
  env: "ManagerBasedRlEnv", command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd: PickCommand = env.command_manager.get_term(command_name)
  site = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return quat_apply_inverse(robot.data.root_link_quat_w, cmd.target_pos_w() - site)


def clutter_state(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Where the objects that are NOT the target are, and whether they are gone.

  Privileged, and the whole reason the critic can still do its job in clutter:
  the actor is told which object to fetch and can see the rest in the depth
  image, but the value of a state depends on how much is left and where, and
  that is exactly what a fixed-width state vector struggles to carry.  Four
  numbers per object -- position relative to the hand, and cleared or not --
  in a fixed order, so the width is constant.
  """
  cmd: PickCommand = env.command_manager.get_term(command_name)
  robot: Entity = env.scene["robot"]
  site = robot.data.site_pos_w[:, cmd._site]
  rel = cmd.all_pos_local + env.scene.env_origins.unsqueeze(1) - site.unsqueeze(1)
  rel = quat_apply_inverse(
    robot.data.root_link_quat_w.unsqueeze(1).expand(-1, cmd.num_objects, -1), rel
  )
  return torch.cat([rel, cmd.cleared.float().unsqueeze(-1)], dim=-1).flatten(1)


def object_to_drop(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  cmd: PickCommand = env.command_manager.get_term(command_name)
  return cmd.command - cmd._object_pos_local()


def gripper_opening(env: "ManagerBasedRlEnv") -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  idx = robot.find_joints(("gripper_joint1",))[0][0]
  return 2.0 * robot.data.joint_pos[:, idx].unsqueeze(-1)


def gripper_squeeze(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """How far the gripper is being asked to close past where it actually is.

  This is the deployable half of ``grasp_state``.  A position servo produces
  force proportional to exactly this error, so on hardware it is what the drive
  reports as current -- whereas ``grasp_state`` is computed from the object\'s
  velocity and lift height, which no sensor on this robot can see.  A vision
  policy that has to work on the real arm gets this and the pad contacts; it
  does not get to know it is holding something.
  """
  robot: Entity = env.scene["robot"]
  idx = robot.find_joints(("gripper_joint1",))[0][0]
  target = robot.data.joint_pos_target[:, idx]
  actual = robot.data.joint_pos[:, idx]
  return (actual - target).unsqueeze(-1)


class CameraScene:
  """Three channels: the whole scene in depth, the target, and the two crossed.

  Kept as separate channels rather than handed over as ``depth * mask``.  The
  masked depth alone says where the target is and nothing about what is around
  it, so the policy could not see the bin it is carrying to, the arm that is
  about to occlude the object, or the other objects it will have to come back
  for.  The masked channel is still there because it is the cheapest possible
  encoding of "this one", and the network should not have to learn a product it
  can be given.

  Depth is normalised against a fixed far plane, not per frame: per-frame
  normalisation is immune to sensor bias and destroys absolute scale, which is
  the cue that says how tall the object is.

  A class rather than a function because the sensor has state.  A third of a
  D405's error is a fixed pattern that does not change between frames
  (``piper_push.depth_noise``), and so is the surface quality of whatever is on
  the table -- neither can be redrawn every step without turning a systematic
  error into something the policy can average away in three frames.  Both are
  drawn at reset, which is what the ``reset`` hook here is for.

  The mask is corrupted too, and for the same reason the depth is.  On the real
  robot it comes out of ``hardware/deploy/mask.py``, which segments the depth
  image: where the sensor returned nothing, the segmenter has nothing to label,
  and its boundary is a pixel or so off wherever it did.  A perfect mask in
  simulation is a channel the policy learns to trust completely and then does
  not get.
  """

  def __init__(self, cfg, env) -> None:
    del cfg  # the parameters arrive through __call__, as for a plain term
    self._env = env
    self._corr: depth_noise.DepthCorruption | None = None
    self._noise_cfg: depth_noise.DepthNoiseCfg | None = None
    self._mask_jitter = 0

  def _build(self, sensor_name: str, shape, device, noise_cfg, mask_jitter):
    from piper_push import camera as camera_mod

    self._noise_cfg = noise_cfg
    self._mask_jitter = int(mask_jitter)
    height, width = int(shape[-2]), int(shape[-1])
    self._corr = depth_noise.DepthCorruption(
      num_envs=self._env.num_envs,
      height=height,
      width=width,
      f_px_per_rad=camera_mod.f_px_per_rad(height=height),
      device=device,
      cfg=noise_cfg,
    )
    del sensor_name

  def reset(self, env_ids=None) -> None:
    if self._corr is not None:
      self._corr.reset(env_ids)

  def _jitter_mask(self, mask: torch.Tensor) -> torch.Tensor:
    """Move the mask boundary by a pixel, in a direction drawn per environment.

    Three outcomes rather than a symmetric blur: a real segmenter's boundary is
    biased one way for a whole scene -- a threshold that includes the shadow at
    the base of every object, or one that clips every silhouette -- and a
    per-pixel coin flip would average that bias to zero and teach the policy
    that the mask edge is unbiased, which it is not.
    """
    k = 2 * self._mask_jitter + 1
    grown = F.max_pool2d(mask, k, 1, self._mask_jitter)
    shrunk = -F.max_pool2d(-mask, k, 1, self._mask_jitter)
    pick = torch.randint(0, 3, (mask.shape[0], 1, 1, 1), device=mask.device)
    return torch.where(pick == 0, shrunk, torch.where(pick == 1, mask, grown))

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    command_name: str,
    cutoff_distance: float = 1.5,
    min_depth: float = 0.05,
    noise_cfg: "depth_noise.DepthNoiseCfg | None" = None,
    mask_jitter_px: int = 1,
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    depth = sensor.data.depth
    seg = sensor.data.segmentation
    assert depth is not None and seg is not None

    depth = depth.permute(0, 3, 1, 2)          # (B, 1, H, W)
    cmd: PickCommand = env.command_manager.get_term(command_name)

    ids = seg[..., 0]
    types = seg[..., 1]
    # (B, K), not (K,): with several objects on the table the mask has to say
    # which one is the target, and that changes as each is cleared.
    target = cmd.target_geom_ids.to(ids.device)
    is_geom = types == int(mujoco.mjtObj.mjOBJ_GEOM)
    mask = ((ids.unsqueeze(-1) == target[:, None, None, :]).any(-1) & is_geom)
    mask = mask.float().unsqueeze(1)

    cfg = noise_cfg if noise_cfg is not None else depth_noise.DepthNoiseCfg()
    if cfg.strength > 0.0:
      if self._corr is None:
        self._build(sensor_name, depth.shape, depth.device, cfg, mask_jitter_px)
      # Clamp before corrupting, not after: the far plane is the sky, and the
      # relative gradient at the horizon of an unclamped depth buffer is
      # enormous and entirely fictional.
      clean = depth.clamp(min=min_depth, max=cutoff_distance)
      depth, valid = self._corr(clean)
      # A hole reads as the far plane.  It has to read as *something*, and this
      # is the convention hardware/deploy/obs.py maps the driver's zero onto,
      # so the two pipelines agree about what "no data" looks like.
      depth = torch.where(valid, depth, torch.full_like(depth, cutoff_distance))
      if self._mask_jitter > 0:
        mask = self._jitter_mask(mask)
      mask = mask * valid.to(mask.dtype)

    norm = torch.clamp(
      torch.clamp(depth, min=min_depth, max=cutoff_distance) / cutoff_distance,
      0.0, 1.0,
    )
    return torch.cat([norm, mask, norm * mask], dim=1)


def grasp_state(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Whether the policy is currently holding the object.

  Visible to the actor on purpose: the reward switches behaviour on this flag,
  so hiding it would make the MDP non-Markov in exactly the dimension the task
  turns on.
  """
  cmd: PickCommand = env.command_manager.get_term(command_name)
  return cmd.grasped.float().unsqueeze(-1)


def pad_contact(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Are the pads touching the target.

  The target, not any object: this is the one proprioceptive channel that
  survives to hardware, and on hardware "the drive is loaded" means the thing
  in the fingers, which is the thing being fetched.
  """
  cmd: PickCommand = env.command_manager.get_term(command_name)
  return (cmd.pad_found > 0).float()


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
  bodies = torch.tensor(
    [o.indexing.body_ids[0] for o in cmd._objects], device=cmd.device
  )
  body = bodies[cmd.target]
  rows0 = torch.arange(cmd.num_envs, device=cmd.device)
  mass = torch.as_tensor(env.sim.model.body_mass[:])[rows0, body].unsqueeze(-1)
  fric = torch.as_tensor(env.sim.model.geom_friction[:])[rows0, cmd.target_core_geom, 0:1]
  ipos = torch.as_tensor(env.sim.model.body_ipos[:])[rows0, body]
  return torch.cat([mass, fric, ipos], dim=-1)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def reach_object(
  env: "ManagerBasedRlEnv", command_name: str, std: float, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Dense until the object is held, then off: a policy still being paid to
  hover near the object has a reason not to commit to lifting it."""
  cmd: PickCommand = env.command_manager.get_term(command_name)
  robot: Entity = env.scene[asset_cfg.name]
  site = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  d = torch.linalg.norm(cmd.target_pos_w() - site, dim=-1)
  return (1.0 - torch.tanh(d / std)) * (~cmd.grasped).float()


def pads_touching(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Both pads on the object. Both, not either: one pad is a shove."""
  cmd: PickCommand = env.command_manager.get_term(command_name)
  found = cmd.pad_found
  assert found is not None
  return ((found > 0).all(dim=1) & ~cmd.grasped).float()


def palm_pushing(env: "ManagerBasedRlEnv", sensor_names: tuple[str, ...]) -> torch.Tensor:
  """The gripper's body touching the object.

  Nothing forbade this, so the policy used the palm as a bat: it drove the
  gripper body into the object 7.9 mm deep (p95) to shove it around, which is a
  contact the hardware would answer by knocking the object away rather than by
  moving it.  Shrinking the contact\'s softness does not remove the behaviour,
  only the depth it shows up at, so the behaviour is priced instead.

  Charged against every object, not just the target: batting a bystander out of
  the way with the palm is the same contact and the same problem.
  """
  hit = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  for name in sensor_names:
    found = env.scene[name].data.found
    assert found is not None
    hit |= (found > 0).any(dim=1)
  return hit.float()


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
  """Dense credit for the object being down inside the bin, and NOT held.

  The "and not held" is the whole design.  Paid regardless of the grip it is
  one more state the policy can sit in, and the arithmetic says sitting in it
  beats finishing: 2.00 a step against 1.48 for completing a cycle.  Gated on
  the release it is a bridge to the placement bonus instead of a substitute
  for it.

  It exists because the last step of the task is a cliff: while the object is held over the bin the transport reward is already
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
  # And only for an object that was carried there; otherwise this is a reward
  # for batting things into the bin.
  return (inside & below_rim & ~cmd.grasped & cmd._grasp_paid).float()


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
  env: "ManagerBasedRlEnv",
  limits: dict[str, float],
  asset_cfg: SceneEntityCfg,
  headroom: float = 1.0,
) -> torch.Tensor:
  """How far past ``headroom`` of the safety shell's trip points the arm goes.

  Charging only above the trip point itself leaves everything below it free,
  and free is where a throughput objective will sit: measured on the first
  policy that solved the task, all six joints peaked between 0.984 and 1.000
  of their trip speed, with the per-step worst joint at 0.750 in the median.
  That is not a policy that occasionally brushes the limit, it is one that
  rides it, and on hardware the servo overshoots a commanded ramp by about
  11%, so there is nothing left to absorb it.  Pricing the approach gives the
  margin somewhere to come from.
  """
  robot: Entity = env.scene[asset_cfg.name]
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids].abs()
  names = [robot.joint_names[i] for i in asset_cfg.joint_ids]
  cap = torch.tensor(
    [limits.get(n, float("inf")) for n in names], device=vel.device
  ) * headroom
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
