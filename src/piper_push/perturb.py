"""Session-persistent simulator mismatch, as an opt-in overlay on any task.

Everything here is inert at its defaults and nothing in the training or
evaluation path sets it.  ``apply_session_mismatch`` mutates an already-built
environment *config*; a task registered without it is byte-for-byte the task
that was trained.

The axes are chosen to be things that are **fixed for a deployment session and
unknown before it starts** -- how the camera actually ended up mounted, what
the depth sensor's scale error is, how long the command path really takes,
how worn the finger pads are.  That is deliberately *not* the same set as the
object-level randomisation the policy was trained across: mass, per-object
friction and shape turn over every object and the policy has already seen
their whole range.  A session parameter is one the policy cannot average over
within a run, which is what makes it worth calibrating.

Where each axis sits relative to what the policy was trained on is recorded in
``AXES`` and written into every result file, because "outside the training
range" and "at the tail of it" are different claims about the same number.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch

from piper_push import camera as cam


@dataclass(frozen=True)
class Axis:
  """One mismatch parameter, and where it stands relative to training."""

  name: str
  unit: str
  nominal: float
  group: str
  trained_range: tuple[float, float] | None
  """What the policy saw during training, in this parameter's own units.
  ``None`` means the quantity was not modelled at all -- the simulator was
  simply correct about it, which is the strongest form of "outside the
  training distribution"."""
  hardware: str
  """What a real deviation of this size would physically be."""


AXES: dict[str, Axis] = {
  # -- A. camera and depth --------------------------------------------------
  "cam_pitch_deg": Axis(
    "cam_pitch_deg", "deg", 0.0, "camera", (-2.0, 2.0),
    "mount tilted in elevation; eye-to-hand calibration residual is ~0.4 deg, "
    "a knocked frame is a few degrees"),
  "cam_yaw_deg": Axis(
    "cam_yaw_deg", "deg", 0.0, "camera", (-2.0, 2.0),
    "mount rotated in azimuth; same source as pitch"),
  "cam_pos_x_m": Axis(
    "cam_pos_x_m", "m", 0.0, "camera", (-0.02, 0.02),
    "camera translated along the table axis; a re-clamped mount moves 1-3 cm"),
  "cam_pos_z_m": Axis(
    "cam_pos_z_m", "m", 0.0, "camera", (-0.02, 0.02),
    "camera raised or lowered on its post"),
  "depth_scale": Axis(
    "depth_scale", "x", 1.0, "camera", None,
    "stereo baseline or focal-length error: every depth reading multiplied "
    "by a constant.  A 2% error at 0.8 m is 16 mm"),
  "depth_bias_m": Axis(
    "depth_bias_m", "m", 0.0, "camera", None,
    "constant range offset from the sensor's own calibration"),
  "depth_dropout": Axis(
    "depth_dropout", "fraction", cam.DEPTH_DROPOUT, "camera", (0.0, 0.02),
    "share of pixels returning no range.  The trained model is i.i.d. "
    "Bernoulli; a real sensor drops whole regions"),
  "depth_dropout_blob": Axis(
    "depth_dropout_blob", "fraction", 0.0, "camera", None,
    "STRUCTURED no-return: contiguous patches rather than salt-and-pepper, "
    "which is what dark, thin and specular surfaces actually produce"),
  "obs_latency_steps": Axis(
    "obs_latency_steps", "control steps", 0.0, "camera", None,
    "camera exposure, transport and inference delay.  One step is 20 ms; a "
    "USB depth camera at 30 fps with a copy and a forward pass is 2-4"),
  # -- B. robot dynamics ----------------------------------------------------
  "action_latency_steps": Axis(
    "action_latency_steps", "control steps", 0.0, "robot", None,
    "command transport: USB-CAN hop, driver queue, 200 Hz inner loop"),
  "joint_response_scale": Axis(
    "joint_response_scale", "x", 1.0, "robot", None,
    "fraction of each commanded displacement the servo actually completes "
    "within one control period, under load"),
  "servo_damping_scale": Axis(
    "servo_damping_scale", "x", 1.0, "robot", None,
    "kd multiplier.  The identified plant is zeta ~ 0.35 on joints 1-3, so "
    "this moves how much the arm overshoots a velocity ramp"),
  "action_deadband_rad": Axis(
    "action_deadband_rad", "rad", 0.0, "robot", None,
    "stiction and encoder quantisation: commanded steps below this produce "
    "no motion at all"),
  # -- C. gripper and contact ----------------------------------------------
  "gripper_rate_scale": Axis(
    "gripper_rate_scale", "x", 1.0, "gripper", None,
    "closure speed against the 0.10 m/s the simulator assumes -- a number "
    "robot.py flags as NOT measured and required before deployment"),
  "gripper_latency_steps": Axis(
    "gripper_latency_steps", "control steps", 0.0, "gripper", None,
    "the gripper's own command path, which is slower than the arm's"),
  "pad_friction_scale": Axis(
    "pad_friction_scale", "x", 1.0, "gripper", (0.55 / 0.85, 1.15 / 0.85),
    "finger-pad wear, dust or a rubber change.  Scales the trained "
    "[0.55, 1.15] band about its midpoint"),
  "table_friction_scale": Axis(
    "table_friction_scale", "x", 1.0, "gripper", (0.4 / 0.7, 1.0 / 0.7),
    "this table is more or less slippery than the training tables.  Scales "
    "the object-table coefficient, which is what governs that contact"),
}


@dataclass
class SessionMismatchCfg:
  """One deployment session's worth of simulator error. All defaults inert."""

  cam_pitch_deg: float = 0.0
  cam_yaw_deg: float = 0.0
  cam_pos_x_m: float = 0.0
  cam_pos_z_m: float = 0.0
  depth_scale: float = 1.0
  depth_bias_m: float = 0.0
  depth_dropout: float | None = None
  depth_dropout_blob: float = 0.0
  obs_latency_steps: int = 0

  action_latency_steps: int = 0
  joint_response_scale: float = 1.0
  servo_damping_scale: float = 1.0
  action_deadband_rad: float = 0.0

  gripper_rate_scale: float = 1.0
  gripper_latency_steps: int = 0
  pad_friction_scale: float = 1.0
  table_friction_scale: float = 1.0

  def active(self) -> dict[str, float]:
    """Only the axes that are actually doing something."""
    out = {}
    for f in fields(self):
      v = getattr(self, f.name)
      if v is None:
        continue
      nom = AXES[f.name].nominal if f.name in AXES else f.default
      if f.name == "depth_dropout":
        if abs(v - cam.DEPTH_DROPOUT) > 1e-12:
          out[f.name] = v
      elif abs(float(v) - float(nom)) > 1e-12:
        out[f.name] = v
    return out

  def is_inert(self) -> bool:
    return not self.active()

  def to_json(self) -> dict:
    d = asdict(self)
    d["_active"] = self.active()
    d["_axes"] = {
      k: {"unit": a.unit, "nominal": a.nominal, "group": a.group,
          "trained_range": list(a.trained_range) if a.trained_range else None,
          "hardware": a.hardware}
      for k, a in AXES.items() if k in d["_active"]
    }
    return d


# ---------------------------------------------------------------------------
# Camera: a class-based observation term, so the latency buffer has a reset
# ---------------------------------------------------------------------------


class PerturbedCameraScene:
  """``pick_mdp.camera_scene`` with a session's depth and timing error on top.

  Class-based because observation latency needs a per-environment ring buffer
  and that buffer has to be flushed on an episode boundary -- mjlab calls
  ``reset`` on any term whose ``func`` provides one.

  The ordering is the sensor's: geometry first (already baked into the render
  by the camera pose), then the sensor's own scale and bias, then its
  dropouts, and only then the normalisation the policy's input expects.  Doing
  the scale after the clamp to the far plane would quietly turn a scale error
  into a clipping error.
  """

  def __init__(self, cfg, env):
    from piper_push.tasks.pick_place import mdp as pick_mdp

    self._inner = pick_mdp.camera_scene
    p = cfg.params
    self._scale = float(p.get("depth_scale", 1.0))
    self._bias = float(p.get("depth_bias_m", 0.0))
    self._blob = float(p.get("depth_dropout_blob", 0.0))
    self._latency = max(int(p.get("obs_latency_steps", 0)), 0)
    self._buf: list[torch.Tensor] = []
    self._env = env

  def __call__(self, env, sensor_name: str, command_name: str,
               cutoff_distance: float = 1.5, min_depth: float = 0.05,
               noise_m: float = 0.0, dropout: float = 0.0,
               depth_scale: float = 1.0, depth_bias_m: float = 0.0,
               depth_dropout_blob: float = 0.0,
               obs_latency_steps: int = 0) -> torch.Tensor:
    del depth_scale, depth_bias_m, depth_dropout_blob, obs_latency_steps

    sensor = env.scene[sensor_name]
    raw = sensor.data.depth
    assert raw is not None

    # Sensor-frame error, applied to the metric range before anything else
    # looks at it.  Written back onto the sensor so the inner term -- which
    # owns the masking, the clamp and the normalisation -- needs no changes
    # and cannot drift out of step with the unperturbed path.
    restore = None
    if self._scale != 1.0 or self._bias != 0.0 or self._blob > 0.0:
      restore = raw.clone()
      d = raw * self._scale + self._bias
      if self._blob > 0.0:
        d = _blob_dropout(d, self._blob, cutoff_distance)
      sensor.data.depth = d

    try:
      obs = self._inner(env, sensor_name, command_name, cutoff_distance,
                        min_depth, noise_m, dropout)
    finally:
      if restore is not None:
        sensor.data.depth = restore

    if not self._latency:
      return obs
    # Seed the pipeline with the first real frame rather than zeros: an empty
    # buffer should mean "the camera has not moved yet", not "the scene is
    # black", which is a state the policy has never seen and would react to.
    while len(self._buf) < self._latency:
      self._buf.append(obs.clone())
    self._buf.append(obs.clone())
    return self._buf.pop(0)

  def reset(self, env_ids=None) -> None:
    if not self._latency or not self._buf:
      return
    if env_ids is None:
      self._buf.clear()
      return
    # Flush only the environments that reset, by making their delayed frames
    # equal to the newest one available.
    newest = self._buf[-1]
    for slot in self._buf[:-1]:
      slot[env_ids] = newest[env_ids]


def _blob_dropout(depth: torch.Tensor, frac: float, far: float) -> torch.Tensor:
  """Contiguous no-return patches, not salt-and-pepper.

  A real depth sensor loses whole surfaces -- a dark object face, a specular
  highlight, a thin edge -- and the i.i.d. Bernoulli model already in the
  observation term is the one kind of dropout that averages out under a
  convolution.  Structured holes are generated by thresholding a smoothed
  noise field, which costs one blur and produces patches whose size is set by
  the kernel rather than by the pixel grid.
  """
  b, h, w = depth.shape[0], depth.shape[1], depth.shape[2]
  field = torch.rand(b, 1, h, w, device=depth.device)
  k = 9
  pad = k // 2
  blur = torch.nn.functional.avg_pool2d(
    torch.nn.functional.pad(field, (pad, pad, pad, pad), mode="reflect"),
    kernel_size=k, stride=1)
  # Threshold so that the hole fraction is the one asked for.  Not
  # torch.quantile: at 512 environments the flattened field is 19M elements
  # and quantile refuses above 16M, which would fail only at full evaluation
  # scale and not in any smoke test.  A box blur of 81 uniforms is very close
  # to normal, so the mean and standard deviation of each image give the
  # threshold directly.
  mu = blur.mean(dim=(2, 3), keepdim=True)
  sd = blur.std(dim=(2, 3), keepdim=True).clamp(min=1e-9)
  z = torch.special.ndtri(torch.tensor(frac, device=depth.device,
                                       dtype=blur.dtype))
  holes = (blur < mu + z * sd).permute(0, 2, 3, 1).expand_as(depth)
  return torch.where(holes, torch.full_like(depth, far), depth)


# ---------------------------------------------------------------------------
# Applying a session to a task config
# ---------------------------------------------------------------------------


def apply_session_mismatch(env_cfg, mm: SessionMismatchCfg) -> dict:
  """Mutate ``env_cfg`` in place. Returns what was actually applied."""
  from mjlab.managers.observation_manager import ObservationTermCfg

  applied = mm.active()
  if not applied:
    return {}

  # -- camera extrinsics: fold a fixed offset into the per-episode jitter ----
  if any(k in applied for k in
         ("cam_pitch_deg", "cam_yaw_deg", "cam_pos_x_m", "cam_pos_z_m")):
    ev = env_cfg.events.get("camera_pose")
    if ev is None:
      raise ValueError("camera perturbation requested on a task with no camera")
    ev.func = randomize_camera_pose_offset
    ev.params = dict(ev.params)
    ev.params.update(
      pitch_offset=math.radians(mm.cam_pitch_deg),
      yaw_offset=math.radians(mm.cam_yaw_deg),
      pos_offset=(mm.cam_pos_x_m, 0.0, mm.cam_pos_z_m),
    )

  # -- depth sensor and observation timing ----------------------------------
  if any(k in applied for k in ("depth_scale", "depth_bias_m",
                                "depth_dropout", "depth_dropout_blob",
                                "obs_latency_steps")):
    grp = env_cfg.observations.get("camera")
    if grp is None:
      raise ValueError("depth perturbation requested on a task with no camera")
    term = grp.terms["scene"]
    params = dict(term.params)
    if mm.depth_dropout is not None:
      params["dropout"] = mm.depth_dropout
    params.update(depth_scale=mm.depth_scale, depth_bias_m=mm.depth_bias_m,
                  depth_dropout_blob=mm.depth_dropout_blob,
                  obs_latency_steps=mm.obs_latency_steps)
    grp.terms["scene"] = ObservationTermCfg(
      func=PerturbedCameraScene, params=params, noise=term.noise)

  # -- the arm's command path ------------------------------------------------
  arm = env_cfg.actions["arm"]
  arm.latency_steps = mm.action_latency_steps
  arm.response_scale = mm.joint_response_scale
  arm.deadband = mm.action_deadband_rad

  # -- the gripper's, which is a different path and a slower one -------------
  grip = env_cfg.actions["gripper"]
  grip.latency_steps = mm.gripper_latency_steps
  grip.slew_scale = mm.gripper_rate_scale

  # -- servo damping ---------------------------------------------------------
  if "servo_damping_scale" in applied:
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    env_cfg.events["servo_damping"] = EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=("joint[1-6]",)),
        "operation": "scale",
        "distribution": "uniform",
        # A degenerate range, because this is a fixed session error and not a
        # randomisation: every environment gets the same damping multiplier.
        "kp_range": (1.0, 1.0),
        "kd_range": (mm.servo_damping_scale, mm.servo_damping_scale),
      },
    )

  # -- contact ---------------------------------------------------------------
  if "pad_friction_scale" in applied:
    ev = env_cfg.events["pad_friction"]
    lo, hi = ev.params["ranges"]
    ev.params = dict(ev.params)
    ev.params["ranges"] = (lo * mm.pad_friction_scale, hi * mm.pad_friction_scale)

  if "table_friction_scale" in applied:
    # The object owns the object-table contact (pads 3 > object 2 > terrain 0),
    # so a slipperier table is a scale on the object's friction draw, not on
    # the terrain's -- setting the terrain's would change nothing at all.
    for name, ev in env_cfg.events.items():
      if name.startswith("object_shape"):
        ev.params = dict(ev.params)
        flo, fhi = ev.params["friction_range"]
        ev.params["friction_range"] = (flo * mm.table_friction_scale,
                                       fhi * mm.table_friction_scale)

  return applied


def aim_offset(fwd: torch.Tensor, right: torch.Tensor, up: torch.Tensor,
               pitch: float, yaw: float) -> torch.Tensor:
  """Turn the optical axis by a fixed pitch and yaw about the camera's own axes.

  Rodrigues, applied to the forward vector only; the frame is rebuilt from the
  result by the caller.  This is a *session* error, so it is a constant and
  not a sample: a mount that is 3 degrees low is 3 degrees low all afternoon.
  """
  for angle, axis in ((pitch, right), (yaw, up)):
    if angle == 0.0:
      continue
    c, s = math.cos(angle), math.sin(angle)
    fwd = torch.nn.functional.normalize(
      fwd * c + torch.cross(axis, fwd, dim=-1) * s, dim=-1)
  return fwd


def randomize_camera_pose_offset(env, env_ids, pos_jitter=0.0, rot_jitter=0.0,
                                 pitch_offset=0.0, yaw_offset=0.0,
                                 pos_offset=(0.0, 0.0, 0.0)) -> None:
  """The stock camera event with a fixed, session-persistent pose error.

  The offset is applied to the nominal mount and the aim point is *not*
  re-derived from it, which is the whole point: a miscalibrated camera looks
  somewhere other than where the policy was told it looks.  The per-episode
  jitter still happens on top, so a session error does not silently remove the
  robustness the policy was trained with.
  """
  from piper_push.camera import CAMERA_AIM, CAMERA_NAME, CAMERA_POS

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  cam_idx = env.scene.sensors[CAMERA_NAME].camera_idx
  base = torch.tensor(CAMERA_POS, device=env.device) + torch.tensor(
    pos_offset, device=env.device)
  pos = base + (2 * torch.rand(n, 3, device=env.device) - 1) * pos_jitter

  aim = torch.tensor(CAMERA_AIM, device=env.device)
  fwd = torch.nn.functional.normalize(aim - pos, dim=-1)
  world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3)
  right = torch.nn.functional.normalize(torch.cross(fwd, world_up, dim=-1), dim=-1)
  up = torch.cross(right, fwd, dim=-1)

  fwd = aim_offset(fwd, right, up, pitch_offset, yaw_offset)
  right = torch.nn.functional.normalize(torch.cross(fwd, world_up, dim=-1), dim=-1)
  up = torch.cross(right, fwd, dim=-1)

  axis = torch.nn.functional.normalize(torch.randn(n, 3, device=env.device), dim=-1)
  ang = (2 * torch.rand(n, 1, device=env.device) - 1) * rot_jitter
  k = torch.sin(ang / 2) * axis
  dq = torch.cat([torch.cos(ang / 2), k], dim=-1)

  R = torch.stack([right, up, -fwd], dim=-1)
  w = torch.sqrt(torch.clamp(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], min=1e-9)) / 2
  q = torch.stack([
    w,
    (R[:, 2, 1] - R[:, 1, 2]) / (4 * w),
    (R[:, 0, 2] - R[:, 2, 0]) / (4 * w),
    (R[:, 1, 0] - R[:, 0, 1]) / (4 * w),
  ], dim=-1)

  w0, v0 = dq[:, :1], dq[:, 1:]
  w1, v1 = q[:, :1], q[:, 1:]
  quat = torch.cat([
    w0 * w1 - (v0 * v1).sum(-1, keepdim=True),
    w0 * v1 + w1 * v0 + torch.cross(v0, v1, dim=-1),
  ], dim=-1)
  quat = torch.nn.functional.normalize(quat, dim=-1)

  env.sim.model.cam_pos[env_ids, cam_idx] = pos
  env.sim.model.cam_quat[env_ids, cam_idx] = quat


# mjlab expands per-world model fields only when an event declares it needs
# them; without this the writes land in world 0 and every environment silently
# shares one camera.
from mjlab.managers.event_manager import requires_model_fields  # noqa: E402

randomize_camera_pose_offset = requires_model_fields(
  "cam_pos", "cam_quat")(randomize_camera_pose_offset)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_mismatch_args(parser) -> None:
  g = parser.add_argument_group(
    "session mismatch",
    "Simulator error held fixed for a whole run. All inert by default.")
  for f in fields(SessionMismatchCfg):
    a = AXES.get(f.name)
    kind = int if "steps" in f.name else float
    g.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=None,
                   help=(f"{a.unit}, nominal {a.nominal}: {a.hardware}"
                         if a else None))


def mismatch_from_args(args) -> SessionMismatchCfg:
  kw = {}
  for f in fields(SessionMismatchCfg):
    v = getattr(args, f.name, None)
    if v is not None:
      kw[f.name] = v
  return SessionMismatchCfg(**kw)
