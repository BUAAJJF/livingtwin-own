"""Workspace point clouds and metric depth from the scene camera, at the D455's cadence.

What the policy is shown, and why it is built this way:

**The measured sensor, then geometry.**  The depth image comes out of
``pick_place.mdp.CameraScene`` -- the fitted D455 model (z^2 noise, frozen
pattern, spatial correlation, edge dropout, disparity quantisation, the
randomised scenery beyond the table) -- with every mask feature switched off.
Only then is it unprojected, so a hole is a missing point and a noisy pixel is
a displaced point, exactly as the robot's own pipeline produces them.

**Base frame, workspace only, table removed.**  Every pixel is put back in
space with the camera's intrinsics and the per-environment (randomised) camera
pose, so the policy never sees pixels: it sees where things are relative to
the robot.  Points outside the annular working sector, below 10 mm above the
table or above 0.45 m are dropped.  The arm's own visible points are kept --
they are real geometry and the deployment cannot remove them without a model.
No instance labels and no segmentation reach the policy.

**30 Hz through a 50 Hz loop.**  The camera delivers a new frame on three of
every five control steps (``fresh_at``), with a per-episode phase; between
frames the last cloud is held.  On top of that sits the measured processing
latency of the robust profile (a per-episode lag of 0..4 steps, mean 43 ms),
implemented as a ring buffer here rather than through mjlab's delay buffer so
that the cloud and its ``vision_meta`` -- age, fresh, valid -- are delayed
together.  ``vision_age`` is the time from capture to use, in control steps;
``vision_fresh`` says the cloud changed this step; ``vision_valid`` says the
frame had enough points to mean anything.

**Sampling.**  ``num_points`` are drawn with replacement from the surviving
pixels, so the set is fixed-size and permutation-free.  Fewer than
``MIN_POINTS`` survivors marks the frame invalid.  Training adds 2 mm per-point
jitter and a 5 mm per-frame offset (the calibration residual); play adds
nothing.  P1a and P1b read the identical tensor.

The depth variant (P0) is the same pipeline stopped before sampling: metric
depth in metres with everything outside the workspace zeroed, plus a validity
channel.  No per-frame normalisation anywhere.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from piper_push import camera as camera_mod
from piper_push.tasks.pick_place import env_cfg as task
from piper_push.tasks.pick_place import mdp as pick_mdp

CAMERA_HZ = 30.0
CONTROL_HZ = 50.0
CADENCE = (3, 5)
"""New frames per control steps: 30 Hz into 50 Hz."""
MAX_LAG = 4
DEFAULT_LATENCY_PROBS = (0.05, 0.15, 0.45, 0.30, 0.05)
"""The robust profile's processing latency over 0..4 control steps (mean 43 ms)."""
META_DIM = 3
AGE_NORM = 10.0
"""``vision_age`` is reported as ``min(age, 10) / 10``."""
MIN_POINTS = 16
POINT_DIM = 4
"""x, y, z in the base frame, and a 1/0 flag that the frame was valid."""


@dataclasses.dataclass(frozen=True)
class WorkspaceCfg:
  """The region a point has to be in to be shown to the policy.

  The end-effector envelope widened by 0.25 rad and 80 mm, so the arm over the
  bin and objects at the sector edge survive; 10 mm above the plane removes
  the table under the robust profile's +-7 mm height and 0.5 deg tilt.
  """

  r_min: float = 0.10
  r_max: float = 0.62
  angle_lo: float = task.EE_ENVELOPE_ANGLE[0] - 0.25
  angle_hi: float = task.EE_ENVELOPE_ANGLE[1] + 0.25
  z_min: float = 0.010
  z_max: float = 0.45


WORKSPACE = WorkspaceCfg()


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate ``v`` (B, ..., 3) by the wxyz quaternions ``q`` (B, 4)."""
  shape = [q.shape[0]] + [1] * (v.dim() - 2) + [3]
  xyz = q[:, 1:].reshape(shape).expand_as(v)
  w = q[:, :1].reshape(shape[:-1] + [1])
  t = 2.0 * torch.cross(xyz, v, dim=-1)
  return v + w * t + torch.cross(xyz, t, dim=-1)


def camera_rays(height: int, width: int, fovy_deg: float, device) -> torch.Tensor:
  """(H, W, 3) directions in the MuJoCo camera frame (looks down -z, +y up), z = -1."""
  f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
  vv, uu = torch.meshgrid(
    torch.arange(height, device=device, dtype=torch.float32),
    torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
  return torch.stack([(uu - (width - 1) / 2.0) / f,
                      -(vv - (height - 1) / 2.0) / f,
                      -torch.ones_like(uu)], dim=-1)


def unproject(depth: torch.Tensor, rays: torch.Tensor, cam_pos: torch.Tensor,
              cam_quat: torch.Tensor) -> torch.Tensor:
  """(B, H, W) z-depth in metres -> (B, H, W, 3) points in the camera's parent frame."""
  local = rays.unsqueeze(0) * depth.unsqueeze(-1)
  return cam_pos.view(-1, 1, 1, 3) + quat_rotate(cam_quat, local)


def in_workspace(pts: torch.Tensor, ws: WorkspaceCfg = WORKSPACE) -> torch.Tensor:
  x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
  r = torch.hypot(x, y)
  a = torch.atan2(y, x)
  return ((r > ws.r_min) & (r < ws.r_max) & (a > ws.angle_lo) & (a < ws.angle_hi)
          & (z > ws.z_min) & (z < ws.z_max))


def fresh_at(t: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
  """Whether a new camera frame arrives at control step ``t`` (0 at reset).

  Three of every five steps, starting from a per-episode phase; the first
  step after a reset always has a frame.
  """
  n, d = CADENCE
  k = torch.div((t + phase) * n, d, rounding_mode="floor")
  kp = torch.div((t - 1 + phase) * n, d, rounding_mode="floor")
  return (k != kp) | (t == 0)


def sample_points(pts: torch.Tensor, inside: torch.Tensor, num_points: int,
                  augment: bool) -> tuple[torch.Tensor, torch.Tensor]:
  """Draw ``num_points`` from the surviving pixels, with replacement.

  Returns ``(B, N, 4)`` and the per-environment count of survivors.
  """
  b = pts.shape[0]
  flat = pts.reshape(b, -1, 3)
  m = inside.reshape(b, -1)
  count = m.sum(dim=1)
  # Survivors only.  A row with no survivor draws uniformly and is then zeroed
  # and flagged invalid below; a tiny floor weight on every pixel would let a
  # far-plane point through once in a hundred thousand draws, which is enough
  # to put a point half a metre under the table in every batch.
  weights = torch.where(count.view(-1, 1) > 0, m.float(), torch.ones_like(m, dtype=torch.float32))
  idx = torch.multinomial(weights, num_points, replacement=True)
  sel = torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
  ok = (count >= MIN_POINTS).view(b, 1, 1)
  if augment:
    sel = sel + 0.002 * torch.randn_like(sel) + (0.010 * torch.rand(b, 1, 3, device=sel.device) - 0.005)
  sel = torch.where(ok, sel, torch.zeros_like(sel))
  flag = ok.float().expand(b, num_points, 1)
  return torch.cat([sel, flag], dim=-1), count


def draw_lags(probs, n: int, device) -> torch.Tensor:
  p = torch.as_tensor(probs, dtype=torch.float32, device=device)
  p = p / p.sum()
  return torch.multinomial(p.expand(n, -1), 1).squeeze(-1)


def ring_step(history: torch.Tensor, meta_hist: torch.Tensor, write: int, out_new: torch.Tensor,
              meta_new: torch.Tensor, fresh: torch.Tensor, lag: torch.Tensor, first: bool
              ) -> tuple[int, torch.Tensor, torch.Tensor]:
  """Advance the capture ring by one control step and read the delayed slot.

  ``write`` is the slot the LAST step wrote.  A step without a fresh frame
  keeps that slot's cloud (the newest frame), a fresh step stores ``out_new``;
  the policy reads ``lag`` slots back.  Returns the new write index and the
  delayed ``(cloud, meta)`` for every environment.

  Kept as a pure function so a test can pin the one property that matters
  and that the first version got wrong: on a held step the cloud the policy
  receives is identical to the previous step's, not the frame from two steps
  back (which is what reading ``write - 1`` before incrementing produced).
  """
  length = history.shape[0]
  prev = torch.zeros_like(out_new) if first else history[write]
  write = (write + 1) % length
  keep = fresh.view(-1, *([1] * (out_new.dim() - 1)))
  history[write] = torch.where(keep, out_new, prev)
  meta_hist[write] = meta_new
  read = (write - lag) % length
  ar = torch.arange(out_new.shape[0], device=out_new.device)
  return write, history[read, ar], meta_hist[read, ar]


class WorkspaceCloud:
  """The observation term.  Owns the cadence, the latency ring and the sensor.

  ``mode`` is ``"cloud"`` (B, N, 4) or ``"depth"`` (B, 2, H, W).  The
  ``vision_meta`` term reads this object's delayed meta; the grasp term (P2)
  reads its full-resolution, capture-side point set on fresh frames.
  """

  def __init__(self, cfg, env) -> None:
    self._env = env
    self._scene = pick_mdp.CameraScene(cfg, env)
    self._rays = None
    self._phase = None
    self._lag = None
    self._t_last = None
    self._history = None
    self._meta_hist = None
    self._write = 0
    self._stamp = None
    self._epoch = 0
    self._out = None
    self.meta = None
    self.params: dict = {}
    # Capture-side products for the grasp term: full point set, survivors, fresh.
    self.full_points = None
    self.full_inside = None
    self.fresh = None
    self.lags = None
    env._pc_cloud_owner = self

  # -- state --------------------------------------------------------------

  def _init_state(self, n: int, dev, shape, latency_probs) -> None:
    length = MAX_LAG + 1
    self._phase = torch.randint(0, CADENCE[1], (n,), device=dev)
    self._lag = draw_lags(latency_probs, n, dev)
    self._t_last = torch.zeros(n, dtype=torch.long, device=dev)
    self._history = torch.zeros((length, n, *shape), device=dev)
    self._meta_hist = torch.zeros((length, n, META_DIM), device=dev)
    self._write = 0
    self.lags = self._lag

  def reset(self, env_ids=None) -> None:
    self._scene.reset(env_ids)
    self._epoch += 1
    if self._phase is None:
      return
    dev = self._phase.device
    ids = (torch.arange(self._phase.shape[0], device=dev) if env_ids is None
           else torch.as_tensor(env_ids, device=dev).reshape(-1))
    if ids.numel() == 0:
      return
    self._phase[ids] = torch.randint(0, CADENCE[1], (ids.numel(),), device=dev)
    self._lag[ids] = draw_lags(self.params.get("latency_probs", DEFAULT_LATENCY_PROBS), ids.numel(), dev)
    self._t_last[ids] = 0
    self._history[:, ids] = 0.0
    self._meta_hist[:, ids] = 0.0

  # -- one capture --------------------------------------------------------

  def _capture(self, env, sensor_name, command_name, num_points, cutoff_distance, min_depth,
               noise_cfg, mask_jitter_px, scenery_dr, augment, mode, workspace):
    img = self._scene(env, sensor_name, command_name, cutoff_distance, min_depth, noise_cfg,
                      mask_jitter_px, True, scenery_dr, None, None, None)
    depth = img[:, 0] * float(cutoff_distance)            # (B, H, W) metres, holes at the far plane
    valid = depth < float(cutoff_distance) - 1e-3
    b, h, w = depth.shape
    if self._rays is None or self._rays.shape[:2] != (h, w):
      self._rays = camera_rays(h, w, camera_mod.FOVY_DEG, depth.device)
    sensor = env.scene[sensor_name]
    ci = sensor.camera_idx
    cam_pos = env.sim.model.cam_pos[:, ci].to(torch.float32)
    cam_quat = env.sim.model.cam_quat[:, ci].to(torch.float32)
    pts = unproject(depth, self._rays, cam_pos, cam_quat)
    inside = valid & in_workspace(pts, workspace)
    self.full_points, self.full_inside = pts, inside
    if mode == "depth":
      out = torch.stack([torch.where(inside, depth, torch.zeros_like(depth)), inside.float()], dim=1)
      count = inside.reshape(b, -1).sum(dim=1)
    else:
      out, count = sample_points(pts, inside, int(num_points), bool(augment))
    return out, count

  # -- the term -----------------------------------------------------------

  def __call__(self, env, sensor_name: str, command_name: str, num_points: int = 512,
               cutoff_distance: float = 1.5, min_depth: float = 0.05, noise_cfg=None,
               mask_jitter_px: int = 0, scenery_dr: bool = False,
               latency_probs=DEFAULT_LATENCY_PROBS, augment: bool = False,
               mode: str = "cloud", workspace: WorkspaceCfg = WORKSPACE) -> torch.Tensor:
    stamp = (int(env.common_step_counter), self._epoch)
    if stamp == self._stamp and self._out is not None:
      return self._out
    self.params = {"latency_probs": tuple(float(x) for x in latency_probs), "mode": mode,
                   "num_points": int(num_points)}
    out_new, count = self._capture(env, sensor_name, command_name, num_points, cutoff_distance,
                                   min_depth, noise_cfg, mask_jitter_px, scenery_dr, augment,
                                   mode, workspace)
    b = out_new.shape[0]
    dev = out_new.device
    if self._phase is None or self._history.shape[1] != b or self._history.shape[2:] != out_new.shape[1:]:
      self._init_state(b, dev, out_new.shape[1:], latency_probs)
    t = env.episode_length_buf.to(dev).long()
    fresh = fresh_at(t, self._phase)
    self.fresh = fresh
    self._t_last = torch.where(fresh, t, self._t_last)
    age = (t - self._t_last) + self._lag
    valid_frame = (count >= MIN_POINTS)
    meta_new = torch.stack([(age.float() / AGE_NORM).clamp(0.0, 1.0), fresh.float(),
                            valid_frame.float()], dim=-1)
    self._write, self._out, self.meta = ring_step(
      self._history, self._meta_hist, self._write, out_new, meta_new, fresh, self._lag,
      first=self._stamp is None)
    self._stamp = stamp
    return self._out


def vision_meta(env) -> torch.Tensor:
  """``[vision_age / 10, vision_fresh, vision_valid]`` as the policy receives them.

  Read from the cloud term, which the observation manager evaluates first (the
  ``camera`` group is registered before ``vision_meta``).
  """
  owner = getattr(env, "_pc_cloud_owner", None)
  if owner is None or owner.meta is None:
    return torch.zeros(env.num_envs, META_DIM, device=env.device)
  return owner.meta
