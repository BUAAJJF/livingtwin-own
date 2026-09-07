"""The point-cloud policy's observation, built from a real D455 frame.

Same construction as ``piper_push.pc.cloud`` builds in the simulator, with the
real camera's numbers in place of the simulator's:

* the D455's own intrinsics (``rig.K``, 848x480) unproject every measured
  pixel in the OpenCV camera frame, and the calibrated ``rig.T_base_cam`` puts
  the points in the robot base frame -- no re-rendering for the cloud routes;
* the workspace is the same annular sector; "above the table" is measured
  against the CALIBRATED plane (``rig.table_z``, ``rig.table_normal_base``),
  which the simulator, whose table is z = 0 with a randomised tilt, does not
  need;
* ``sample_points`` is the simulator's function, called with the augmentation
  off, so the robot's cloud is drawn the way an evaluation cloud is drawn;
* for P0 the frame is first re-rendered to the policy camera by
  ``rectify.Reprojector`` (the same operation the depth-image line used), then
  cropped to the workspace by unprojecting the policy grid with the simulator
  camera's pose -- so P0 sees metres on the 168x224 grid it trained on, with
  0 where nothing is in the workspace and a validity channel.

The vision_meta channel is not built here: age, fresh and valid depend on the
control loop's clock and on when the frame was captured, and ``pc_run.py``
owns those.  What this returns alongside the observation is the survivor count
the ``valid`` flag is decided from (``MIN_POINTS``, as in training).

For P2 the base-frame points, the survivor mask, the arm's body positions and
the grasp-site position are handed to ``piper_push.pc.grasp.propose`` -- the
same batched code the simulator runs, batch of one.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from . import config, rectify
from piper_push import camera as sim_camera
from piper_push.pc import cloud, grasp

ARM_BODIES = grasp.ARM_BODIES


@dataclasses.dataclass
class CloudFrame:
  obs: np.ndarray
  """(N, 4) float32 for the cloud routes, (2, H, W) for P0."""
  count: int
  """Survivors inside the workspace above the plane -- the ``valid`` flag is ``count >= MIN_POINTS``."""
  valid: bool
  points_base: torch.Tensor | None = None
  """(1, H*W, 3) base-frame points (cloud routes), for the P2 proposals."""
  inside: torch.Tensor | None = None


class CloudObs:
  def __init__(self, rig: config.Rig, mode: str = "cloud", num_points: int = cloud.POINT_DIM * 128,
               device: str = "cuda:0", workspace: cloud.WorkspaceCfg = cloud.WORKSPACE,
               height_min_m: float = 0.010) -> None:
    if mode not in ("cloud", "depth"):
      raise ValueError(mode)
    self.mode = mode
    self.num_points = int(num_points)
    self.device = torch.device(device)
    self.workspace = workspace
    self.height_min = float(height_min_m)
    K = np.asarray(rig.K if rig.K is not None else rectify._default_d405_K(), dtype=np.float64)
    self.K = K
    self.width, self.height = config.D405_WIDTH, config.D405_HEIGHT
    rays = rectify.ray_grid(K, self.width, self.height)                       # (H, W, 3), z = 1
    self._rays = torch.as_tensor(rays.reshape(-1, 3), dtype=torch.float32, device=self.device)
    T = np.asarray(rig.T_base_cam, dtype=np.float64)
    self._R = torch.as_tensor(T[:3, :3], dtype=torch.float32, device=self.device)
    self._t = torch.as_tensor(T[:3, 3], dtype=torch.float32, device=self.device)
    n = np.asarray(rig.table_normal_base if rig.table_normal_base is not None else (0.0, 0.0, 1.0), dtype=np.float64)
    n = n / np.linalg.norm(n)
    self._n = torch.as_tensor(n, dtype=torch.float32, device=self.device)
    self._p0 = torch.tensor([0.0, 0.0, float(rig.table_z)], device=self.device)
    self.table_z = float(rig.table_z)
    if mode == "depth":
      self.reproj = rectify.Reprojector(rig, device=str(self.device))
      h, w = config.HEIGHT, config.WIDTH
      self._policy_rays = cloud.camera_rays(h, w, config.FOVY_DEG, self.device)
      self._cam_pos = torch.tensor(sim_camera.CAMERA_POS, dtype=torch.float32, device=self.device).view(1, 3)
      self._cam_quat = torch.tensor(sim_camera.CAMERA_QUAT, dtype=torch.float32, device=self.device).view(1, 4)

  def _to_base(self, depth_m: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.as_tensor(np.asarray(depth_m, dtype=np.float32), device=self.device).reshape(-1)
    valid = (d > config.MIN_DEPTH_M) & (d < config.CUTOFF_M)
    pts_cam = self._rays * d.unsqueeze(-1)
    pts = pts_cam @ self._R.T + self._t
    return pts, valid

  def _inside(self, pts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    ws = self.workspace
    x, y = pts[..., 0], pts[..., 1]
    r = torch.hypot(x, y)
    a = torch.atan2(y, x)
    height = ((pts - self._p0) * self._n).sum(-1)
    return (valid & (r > ws.r_min) & (r < ws.r_max) & (a > ws.angle_lo) & (a < ws.angle_hi)
            & (height > self.height_min) & (height < ws.z_max))

  def __call__(self, depth_m: np.ndarray) -> CloudFrame:
    if self.mode == "depth":
      dpol, vpol, _ = self.reproj(np.asarray(depth_m, dtype=np.float32))
      dt = torch.as_tensor(np.asarray(dpol, dtype=np.float32), device=self.device).unsqueeze(0)   # (1, H, W)
      vt = torch.as_tensor(np.asarray(vpol, dtype=bool), device=self.device).unsqueeze(0)
      pts = cloud.unproject(dt, self._policy_rays, self._cam_pos, self._cam_quat)                  # (1, H, W, 3)
      inside = vt & (dt > config.MIN_DEPTH_M) & cloud.in_workspace(pts, self.workspace)
      out = torch.stack([torch.where(inside, dt, torch.zeros_like(dt)), inside.float()], dim=1)[0]
      count = int(inside.sum())
      return CloudFrame(obs=out.cpu().numpy(), count=count, valid=count >= cloud.MIN_POINTS)
    pts, valid = self._to_base(depth_m)
    inside = self._inside(pts, valid)
    sampled, count = cloud.sample_points(pts.unsqueeze(0), inside.unsqueeze(0), self.num_points, augment=False)
    c = int(count[0])
    return CloudFrame(obs=sampled[0].cpu().numpy(), count=c, valid=c >= cloud.MIN_POINTS,
                      points_base=pts.unsqueeze(0), inside=inside.unsqueeze(0))


def object_points(frame: CloudFrame, arm_pos: torch.Tensor, plane_p0: torch.Tensor, plane_n: torch.Tensor,
                  h_min: float = grasp.H_MIN, h_max: float = grasp.H_MAX) -> int:
  """How many of the frame's workspace points could belong to an object.

  Points more than ``h_min`` above the calibrated plane (a real object, not
  table noise), outside the sphere cover of the arm's bodies (``ARM_BODIES``,
  ``ARM_RADII`` as the P2 proposer uses them) and outside the bin's footprint
  with its walls.  Zero for a few frames means the table is empty and the
  policy has nothing to do -- a state it was never trained in (the simulator
  refills the table the instant a placement registers) and in which it
  wanders.  ``run.py`` holds on it.
  """
  if frame.points_base is None or frame.inside is None:
    return 0
  pts = frame.points_base[0][frame.inside[0]]
  if pts.shape[0] == 0:
    return 0
  h = ((pts - plane_p0) * plane_n).sum(-1)
  keep = (h > float(h_min)) & (h < float(h_max))
  pts = pts[keep]
  if pts.shape[0] == 0:
    return 0
  radii = torch.tensor(grasp.ARM_RADII, device=pts.device).view(1, -1)
  d_arm = torch.cdist(pts, arm_pos.reshape(-1, 3))
  keep = ~((d_arm < radii).any(dim=-1))
  from piper_push import objects
  bx, by = objects.BIN_CENTER
  hx = objects.BIN_INNER[0] + objects.BIN_WALL_THICKNESS + 0.015
  hy = objects.BIN_INNER[1] + objects.BIN_WALL_THICKNESS + 0.015
  keep &= ~(((pts[:, 0] - bx).abs() < hx) & ((pts[:, 1] - by).abs() < hy))
  return int(keep.sum())


class GraspObs:
  """P2 on the robot: proposals from the frame's base-frame points, and the lock.

  Uses ``piper_push.pc.grasp.GraspCandidates``' lock logic on a batch of one by
  driving ``propose`` directly; the arm's body positions come from the
  deployment kinematics (``proprio.Kinematics``), the grasp site likewise.
  """

  def __init__(self, kin, device: str = "cuda:0") -> None:
    import mujoco
    self.kin = kin
    self.device = torch.device(device)
    self.body_ids = [mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ARM_BODIES]
    missing = [n for n, i in zip(ARM_BODIES, self.body_ids) if i < 0]
    if missing:
      raise RuntimeError(f"deployment model has no bodies {missing}")
    self.topk = np.zeros((grasp.K, grasp.FEAT), dtype=np.float32)
    self.locked = np.zeros(grasp.FEAT + 2, dtype=np.float32)
    self._lock_pos = None
    self._lock_age = 0
    self._lost = 0
    self.switches = 0
    self.no_candidate_frames = 0
    self.frames = 0

  def reset(self) -> None:
    self._lock_pos = None
    self._lock_age = 0
    self._lost = 0
    self.locked[:] = 0.0

  def update(self, frame: CloudFrame) -> None:
    """On a fresh frame: recompute the candidates and advance the lock."""
    if frame.points_base is None:
      return
    arm = torch.as_tensor(np.asarray(self.kin.data.xpos[self.body_ids], dtype=np.float32),
                          device=self.device).unsqueeze(0)
    site = self.kin.site_pos
    site = site() if callable(site) else site
    ee = torch.as_tensor(np.asarray(site, dtype=np.float32), device=self.device).unsqueeze(0)
    props = grasp.propose(frame.points_base, frame.inside, arm, ee, table_z=0.0)
    feats, valid = props.feats[0], props.valid[0]
    order = (feats[:, grasp.I_SCORE] - (~valid).float() * 10.0).argsort(descending=True)
    feats, valid = feats[order], valid[order]
    self.topk = feats.cpu().numpy()
    self.frames += 1
    f = self.topk
    v = valid.cpu().numpy()
    usable = v & (f[:, grasp.I_REACH] > 0.5) & (f[:, grasp.I_FEASIBLE] > 0.5)
    if not usable.any():
      self.no_candidate_frames += 1
    matched = None
    if self._lock_pos is not None:
      d = np.linalg.norm(f[:, :3] - self._lock_pos, axis=-1)
      d[~v] = 9.0
      j = int(d.argmin())
      if d[j] < grasp.MATCH_M:
        matched = j
        self._lost = 0
      else:
        self._lost += 1
      self._lock_age += 1
      if self._lost > grasp.LOST_FRAMES or self._lock_age > grasp.LOCK_TIMEOUT_STEPS:
        self._lock_pos = None
        matched = None
    if self._lock_pos is None and usable.any():
      dist = np.where(usable, f[:, grasp.I_DIST], 9.0)
      j = int(dist.argmin())
      if self._lock_age > 0:
        self.switches += 1
      self._lock_pos = f[j, :3].copy()
      self._lock_age = 0
      self._lost = 0
      matched = j
    if self._lock_pos is None:
      self.locked[:] = 0.0
    elif matched is not None:
      self._lock_pos = f[matched, :3].copy()
      self.locked[:grasp.FEAT] = f[matched]
      self.locked[grasp.FEAT] = min(self._lock_age / 100.0, 1.0)
      self.locked[grasp.FEAT + 1] = 1.0
