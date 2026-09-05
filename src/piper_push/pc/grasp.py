"""Top-K 6D grasp candidates from the workspace point cloud, and the candidate lock (P2).

What this is, honestly: no pretrained grasp detector with a checkpoint and a
licence was available on this machine within the timebox, so the proposals are
ANALYTIC top-down candidates from the cloud's geometry, as the brief allows.
The pipeline is the one the brief names --

    cloud -> proposals -> 6D NMS -> collision check -> workspace/reach filter -> top-K -> lock

-- with these concrete stages, all batched in torch and identical in the
simulator and on the robot (both hand this module base-frame points):

* **arm removal** for the proposal step only (the encoder never sees a mask):
  points within a sphere of each arm body are dropped, from body positions the
  robot reports in both worlds;
* **a 1 cm top-down height map** over the workspace; cells inside the bin's
  footprint are excluded (the rim is 60 mm tall and would otherwise propose);
* **connected components** of cells 15-130 mm above the table, by label
  propagation on the grid -- this IS the NMS: one component, one place;
* **two candidates per component**, closing across each principal extent of
  the component's box, so K = 32 covers 16 components; width is the extent
  across the closing axis plus 10 mm, clipped to the 50 mm jaw;
* **collision margin**: distance to the nearest other component minus the two
  half-widths; **reachability**: the base-frame position inside the
  end-effector envelope (``EE_ENVELOPE_*``, z below the 150 mm straight-down
  limit).  There is no IK solver in this repository; "IK/reachability" here is
  that envelope test and is reported as such.

Candidate features (18): position xyz (0-2), rotation as the first two
columns of R (3-8), width (9), score (10), collision margin (11), reachable
(12), pose error to the grasp site dx, dy, dz (13-15) and distance (16),
feasible (17).  The 6D rotation is a top-down frame whose
x axis is the closing direction at the candidate's yaw.

**The lock.**  A candidate is chosen when none is held: the nearest reachable,
feasible one to the grasp site (the same rule the task's command uses to pick
its target, so the executor is aimed where the teacher was).  It is kept, and
its features tracked to the matching candidate within 30 mm on every fresh
frame, until it has been missing for 15 frames (picked up, or gone), or has
been held for 10 s without success.  Candidates only change on fresh frames.
Switches are counted (``switches``), as are frames with no candidate at all.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F

from piper_push import objects
from piper_push.tasks.pick_place import env_cfg as task

K = 32
FEAT = 18
I_SCORE, I_REACH, I_DIST, I_FEASIBLE = 10, 12, 16, 17
CELL = 0.01
X_RANGE = (-0.65, 0.65)
Y_RANGE = (-0.05, 0.65)
GX = int(round((X_RANGE[1] - X_RANGE[0]) / CELL))
GY = int(round((Y_RANGE[1] - Y_RANGE[0]) / CELL))
H_MIN, H_MAX = 0.015, 0.130
ARM_BODIES = ("link2", "link3", "link4", "link5", "link6", "gripper_base", "gripper_link1", "gripper_link2")
ARM_RADII = (0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05)
MATCH_M = 0.03
LOST_FRAMES = 15
LOCK_TIMEOUT_STEPS = 500
MAX_COMPONENTS = K // 2
JAW_M = 0.05


def rot6d_topdown(yaw: torch.Tensor) -> torch.Tensor:
  """Columns x = (cos, sin, 0) (closing axis) and y = (-sin, cos, 0); z is down."""
  c, s = torch.cos(yaw), torch.sin(yaw)
  z = torch.zeros_like(yaw)
  return torch.stack([c, s, z, -s, c, z], dim=-1)


def height_map(points: torch.Tensor, inside: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-cell max height and count from (B, ..., 3) points and their (B, ...) mask."""
  b = points.shape[0]
  p = points.reshape(b, -1, 3)
  m = inside.reshape(b, -1)
  ix = ((p[..., 0] - X_RANGE[0]) / CELL).floor().long()
  iy = ((p[..., 1] - Y_RANGE[0]) / CELL).floor().long()
  ok = m & (ix >= 0) & (ix < GX) & (iy >= 0) & (iy < GY)
  idx = torch.where(ok, iy * GX + ix, torch.zeros_like(ix))
  h = torch.full((b, GX * GY), -1.0, device=points.device)
  h.scatter_reduce_(1, idx, torch.where(ok, p[..., 2], torch.full_like(p[..., 2], -1.0)), "amax")
  n = torch.zeros((b, GX * GY), device=points.device)
  n.scatter_add_(1, idx, ok.float())
  return h, n


def components(occ: torch.Tensor, iters: int = 24) -> torch.Tensor:
  """Connected-component labels (B, GY*GX), 0 where empty, by max-label propagation."""
  b = occ.shape[0]
  base = (torch.arange(GX * GY, device=occ.device, dtype=torch.float32) + 1.0).expand(b, -1)
  lab = torch.where(occ, base, torch.zeros_like(base)).view(b, 1, GY, GX)
  o = occ.view(b, 1, GY, GX)
  for _ in range(iters):
    lab = torch.where(o, F.max_pool2d(lab, 3, 1, 1), torch.zeros_like(lab))
  return lab.view(b, -1).long()


@dataclasses.dataclass
class Proposals:
  feats: torch.Tensor      # (B, K, FEAT)
  valid: torch.Tensor      # (B, K)
  n_components: torch.Tensor


def propose(points: torch.Tensor, inside: torch.Tensor, arm_pos: torch.Tensor, ee_pos: torch.Tensor,
            table_z: float = 0.0) -> Proposals:
  """Candidates from base-frame points.  ``arm_pos`` (B, nb, 3), ``ee_pos`` (B, 3)."""
  b = points.shape[0]
  dev = points.device
  h, n = height_map(points, inside)
  cx = X_RANGE[0] + (torch.arange(GX, device=dev).float() + 0.5) * CELL
  cy = Y_RANGE[0] + (torch.arange(GY, device=dev).float() + 0.5) * CELL
  cyy, cxx = torch.meshgrid(cy, cx, indexing="ij")
  cxx, cyy = cxx.reshape(-1), cyy.reshape(-1)
  rel = h - table_z
  occ = (n > 0) & (rel > H_MIN) & (rel < H_MAX)
  # The arm, by sphere cover of its bodies.
  cells = torch.stack([cxx.expand(b, -1), cyy.expand(b, -1), h], dim=-1)          # (B, NC, 3)
  radii = torch.tensor(ARM_RADII, device=dev).view(1, 1, -1)
  d_arm = torch.cdist(cells, arm_pos)                                              # (B, NC, nb)
  occ &= ~((d_arm < radii).any(dim=-1))
  # The bin's footprint, walls included.
  bx, by = objects.BIN_CENTER
  hx, hy = (objects.BIN_INNER[0] + objects.BIN_WALL_THICKNESS + 0.015,
            objects.BIN_INNER[1] + objects.BIN_WALL_THICKNESS + 0.015)
  occ &= ~((cxx.abs() * 0 + (cxx - bx).abs() < hx) & ((cyy - by).abs() < hy)).expand(b, -1)
  lab = components(occ)
  nc = GX * GY + 1
  ones = occ.float()
  count = torch.zeros((b, nc), device=dev).scatter_add_(1, lab, ones)
  sx = torch.zeros((b, nc), device=dev).scatter_add_(1, lab, cxx.expand(b, -1) * ones)
  sy = torch.zeros((b, nc), device=dev).scatter_add_(1, lab, cyy.expand(b, -1) * ones)
  hmax = torch.full((b, nc), -1.0, device=dev).scatter_reduce_(1, lab, torch.where(occ, rel, torch.full_like(rel, -1.0)), "amax")
  big = torch.full((b, nc), 9.0, device=dev)
  xmin = big.clone().scatter_reduce_(1, lab, torch.where(occ, cxx.expand(b, -1), torch.full_like(rel, 9.0)), "amin")
  ymin = big.clone().scatter_reduce_(1, lab, torch.where(occ, cyy.expand(b, -1), torch.full_like(rel, 9.0)), "amin")
  xmax = (-big).clone().scatter_reduce_(1, lab, torch.where(occ, cxx.expand(b, -1), torch.full_like(rel, -9.0)), "amax")
  ymax = (-big).clone().scatter_reduce_(1, lab, torch.where(occ, cyy.expand(b, -1), torch.full_like(rel, -9.0)), "amax")
  count[:, 0] = 0.0
  top = count.topk(MAX_COMPONENTS, dim=1)                                          # (B, C)
  ci = top.indices
  ccount = top.values
  present = ccount >= 2.0
  g = lambda t: torch.gather(t, 1, ci)
  mx = g(sx) / ccount.clamp_min(1.0)
  my = g(sy) / ccount.clamp_min(1.0)
  mh = g(hmax)
  wx = (g(xmax) - g(xmin) + CELL).clamp_min(CELL)
  wy = (g(ymax) - g(ymin) + CELL).clamp_min(CELL)
  # Two candidates per component: close across x (yaw 0) and across y (yaw pi/2).
  yaw = torch.stack([torch.zeros_like(mx), torch.full_like(mx, math.pi / 2)], dim=-1)   # (B, C, 2)
  width = torch.stack([wx, wy], dim=-1) + 0.010
  pos = torch.stack([mx, my, table_z + (0.5 * mh).clamp(0.012, 0.060)], dim=-1)         # (B, C, 3)
  pos = pos.unsqueeze(2).expand(-1, -1, 2, -1)
  feasible = present.unsqueeze(-1) & (width <= JAW_M) & (mh.unsqueeze(-1) >= H_MIN)
  score = feasible.float() * (1.0 - width / 0.06).clamp(0.0, 1.0) * (mh.unsqueeze(-1) / 0.05).clamp(0.3, 1.0)
  # Collision margin: nearest other component, minus the half-widths.
  centres = torch.stack([mx, my], dim=-1)                                            # (B, C, 2)
  dc = torch.cdist(centres, centres)
  dc = dc + torch.eye(MAX_COMPONENTS, device=dev) * 9.0
  dc = torch.where(present.unsqueeze(1), dc, torch.full_like(dc, 9.0))
  nearest = dc.amin(dim=-1)                                                          # (B, C)
  margin = (nearest.unsqueeze(-1) - 0.5 * width - 0.5 * torch.maximum(wx, wy).unsqueeze(-1)).clamp(-0.05, 0.20)
  r = torch.hypot(pos[..., 0], pos[..., 1])
  ang = torch.atan2(pos[..., 1], pos[..., 0])
  reach = ((r > task.EE_ENVELOPE_RADIUS[0]) & (r < task.EE_ENVELOPE_RADIUS[1])
           & (ang > task.EE_ENVELOPE_ANGLE[0]) & (ang < task.EE_ENVELOPE_ANGLE[1]) & (pos[..., 2] < 0.15))
  err = pos - ee_pos.view(b, 1, 1, 3)
  dist = err.norm(dim=-1, keepdim=True)
  feats = torch.cat([pos, rot6d_topdown(yaw), width.unsqueeze(-1), score.unsqueeze(-1), margin.unsqueeze(-1),
                     reach.float().unsqueeze(-1), err, dist, feasible.float().unsqueeze(-1)], dim=-1)
  feats = feats.reshape(b, K, FEAT)
  valid = (present.unsqueeze(-1).expand(-1, -1, 2)).reshape(b, K)
  feats = torch.where(valid.unsqueeze(-1), feats, torch.zeros_like(feats))
  return Proposals(feats=feats, valid=valid, n_components=present.sum(dim=1))


class GraspCandidates:
  """The observation term.  Produces ``topk`` (B, K, FEAT) and ``locked`` (B, FEAT + 2)."""

  def __init__(self, cfg, env) -> None:
    self._env = env
    self._body_ids = None
    self._stamp = None
    self._epoch = 0
    self.topk = None
    self.locked = None
    self._lock_pos = None
    self._lock_yaw = None
    self._lock_on = None
    self._lock_age = None
    self._lost = None
    self.switches = None
    self.no_candidate_steps = None
    self.frames = 0
    env._pc_grasp_owner = self

  def reset(self, env_ids=None) -> None:
    self._epoch += 1
    if self._lock_on is None:
      return
    dev = self._lock_on.device
    ids = (torch.arange(self._lock_on.shape[0], device=dev) if env_ids is None
           else torch.as_tensor(env_ids, device=dev).reshape(-1))
    if ids.numel() == 0:
      return
    self._lock_on[ids] = False
    self._lock_age[ids] = 0
    self._lost[ids] = 0
    self.switches[ids] = 0
    self.no_candidate_steps[ids] = 0
    self.topk[ids] = 0.0
    self.locked[ids] = 0.0

  def _arm_positions(self, env) -> torch.Tensor:
    robot = env.scene["robot"]
    if self._body_ids is None:
      ids, _ = robot.find_bodies(list(ARM_BODIES), preserve_order=True)
      self._body_ids = torch.as_tensor(ids, device=env.device)
    pos = robot.data.body_link_pos_w[:, self._body_ids] - env.scene.env_origins.unsqueeze(1)
    return pos

  def __call__(self, env, command_name: str = "pick") -> torch.Tensor:
    stamp = (int(env.common_step_counter), self._epoch)
    if stamp == self._stamp and self.topk is not None:
      return self.topk
    owner = getattr(env, "_pc_cloud_owner", None)
    b, dev = env.num_envs, env.device
    if self.topk is None:
      self.topk = torch.zeros(b, K, FEAT, device=dev)
      self.locked = torch.zeros(b, FEAT + 2, device=dev)
      self._lock_pos = torch.zeros(b, 3, device=dev)
      self._lock_yaw = torch.zeros(b, device=dev)
      self._lock_on = torch.zeros(b, dtype=torch.bool, device=dev)
      self._lock_age = torch.zeros(b, dtype=torch.long, device=dev)
      self._lost = torch.zeros(b, dtype=torch.long, device=dev)
      self.switches = torch.zeros(b, dtype=torch.long, device=dev)
      self.no_candidate_steps = torch.zeros(b, dtype=torch.long, device=dev)
    self._stamp = stamp
    if owner is None or owner.full_points is None or owner.fresh is None:
      return self.topk
    fresh = owner.fresh
    cmd = env.command_manager.get_term(command_name)
    ee = cmd._site_pos_w() - env.scene.env_origins
    props = propose(owner.full_points, owner.full_inside, self._arm_positions(env), ee)
    feats, valid = props.feats, props.valid
    # Sort by score, invalid last.
    order = (feats[..., I_SCORE] - (~valid).float() * 10.0).argsort(dim=1, descending=True)
    feats = torch.gather(feats, 1, order.unsqueeze(-1).expand(-1, -1, FEAT))
    valid = torch.gather(valid, 1, order)
    # Only fresh frames update what the policy sees.
    self.topk = torch.where(fresh.view(-1, 1, 1), feats, self.topk)
    self.frames += 1
    # -- the lock, on fresh frames -------------------------------------------
    pos = feats[..., :3]
    usable = valid & (feats[..., I_REACH] > 0.5) & (feats[..., I_FEASIBLE] > 0.5)
    d_lock = (pos - self._lock_pos.unsqueeze(1)).norm(dim=-1)
    d_lock = torch.where(valid, d_lock, torch.full_like(d_lock, 9.0))
    match_d, match_i = d_lock.min(dim=1)
    matched = self._lock_on & (match_d < MATCH_M) & fresh
    self._lost = torch.where(fresh & self._lock_on & ~matched, self._lost + 1, self._lost)
    self._lost = torch.where(matched, torch.zeros_like(self._lost), self._lost)
    self._lock_age = torch.where(self._lock_on, self._lock_age + 1, self._lock_age)
    release = self._lock_on & ((self._lost > LOST_FRAMES) | (self._lock_age > LOCK_TIMEOUT_STEPS))
    self._lock_on &= ~release
    d_ee = feats[..., I_DIST]
    d_ee = torch.where(usable, d_ee, torch.full_like(d_ee, 9.0))
    best_d, best_i = d_ee.min(dim=1)
    acquire = fresh & ~self._lock_on & (best_d < 9.0)
    self.switches += (acquire & (self._lock_age > 0)).long()
    idx = torch.where(acquire, best_i, match_i)
    chosen = feats[torch.arange(b, device=dev), idx]
    update = acquire | matched
    self._lock_pos = torch.where(update.unsqueeze(-1), chosen[:, :3], self._lock_pos)
    self._lock_yaw = torch.where(update, torch.atan2(chosen[:, 4], chosen[:, 3]), self._lock_yaw)
    self._lock_on |= acquire
    self._lock_age = torch.where(acquire, torch.zeros_like(self._lock_age), self._lock_age)
    self._lost = torch.where(acquire, torch.zeros_like(self._lost), self._lost)
    locked = torch.cat([chosen, (self._lock_age.float() / 100.0).clamp(0, 1).unsqueeze(-1),
                        self._lock_on.float().unsqueeze(-1)], dim=-1)
    locked = torch.where(self._lock_on.unsqueeze(-1), locked, torch.zeros_like(locked))
    self.locked = torch.where(update.unsqueeze(-1) | release.unsqueeze(-1), locked, self.locked)
    self.no_candidate_steps += (~usable.any(dim=1)).long()
    return self.topk


def locked_candidate(env) -> torch.Tensor:
  owner = getattr(env, "_pc_grasp_owner", None)
  if owner is None or owner.locked is None:
    return torch.zeros(env.num_envs, FEAT + 2, device=env.device)
  return owner.locked
