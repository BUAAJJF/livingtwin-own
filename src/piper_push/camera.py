"""The fixed third-person camera the vision policy sees the table through.

Everything here was chosen by measurement, and the measurements are recorded
next to the numbers because the alternatives all look reasonable:

  * The camera hangs off ``base_link``, not the world.  Parallel environments
    are laid out on a grid with real offsets, so a camera fixed in the world
    frame frames environment zero and misses every other one.  Base frame is
    also what eye-to-hand calibration produces on the real rig, so the number
    written here is the number that comes back from the calibration.

  * Of eight candidate viewpoints scored over 256 environments, the one placed
    near the bin was the only failure: the target object was visible in 58% of
    frames against 100% everywhere else, because the bin walls occlude the
    object area and the view runs nearly along the table.  It is the position
    that feels right and it is the one that does not work.

  * The field of view is wider than the workspace needs.  45 degrees fits it
    with a 12% margin and puts the bin's far corner at 97% of the half-width,
    which a camera mounted two centimetres off would clip.  52 degrees spends
    some resolution to buy back that margin, which is the same trade as
    randomising the pose: calibration only has to land inside the envelope.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.sensor import CameraSensorCfg

CAMERA_NAME = "scene_cam"
PARENT_BODY = "robot/base_link"

CAMERA_POS = (0.589, 0.535, 0.480)
"""Metres in the robot base frame: 796 mm out along the table at +42.2 deg to
the robot's left, 480 mm up."""

CAMERA_AIM = (0.321, 0.071, 0.030)
"""What it points at -- the centroid of the object area and the bin together.
Deliberately biased towards the bin, so aiming by eye at the object area puts
the bin out of frame."""

FOVY_DEG = 52.0
WIDTH, HEIGHT = 224, 168
CUTOFF_M = 1.5
"""Far plane for depth normalisation.  Fixed rather than per-frame: a per-frame
min-max normalisation is immune to depth bias and throws away absolute scale,
which is exactly the cue that says how tall the object is."""

# Randomisation envelope around the nominal.  Sized at roughly five times the
# residual of a decent eye-to-hand calibration, so the real camera only has to
# land inside it rather than hit the number.
POS_JITTER_M = 0.020
ROT_JITTER_RAD = math.radians(2.0)

# Depth sensor realism.  These are PLACEHOLDERS: a real depth camera's noise
# grows with distance and it drops out entirely on dark, specular and thin
# surfaces, and neither is a Gaussian.  Measure the actual sensor against a
# known plane and put the fitted model here before S5 -- guessing this is the
# single largest sim2real risk in the vision stage.
DEPTH_NOISE_M = 0.004
DEPTH_DROPOUT = 0.02


def look_at_quat(pos, target=CAMERA_AIM) -> tuple[float, float, float, float]:
  """MuJoCo camera orientation looking from ``pos`` to ``target``.

  A MuJoCo camera looks down its own -z with +y up in the image.
  """
  pos = np.asarray(pos, dtype=float)
  fwd = np.asarray(target, dtype=float) - pos
  fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
  right /= np.linalg.norm(right)
  up = np.cross(right, fwd)
  R = np.stack([right, up, -fwd], axis=1)
  w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
  q = np.array([w,
                (R[2, 1] - R[1, 2]) / (4 * w),
                (R[0, 2] - R[2, 0]) / (4 * w),
                (R[1, 0] - R[0, 1]) / (4 * w)])
  q /= np.linalg.norm(q)
  return tuple(float(x) for x in q)


def fovx_deg(fovy: float = FOVY_DEG, width: int = WIDTH, height: int = HEIGHT) -> float:
  return 2 * math.degrees(math.atan(math.tan(math.radians(fovy / 2)) * width / height))


def camera_cfg(width: int = WIDTH, height: int = HEIGHT) -> CameraSensorCfg:
  return CameraSensorCfg(
    name=CAMERA_NAME,
    parent_body=PARENT_BODY,
    pos=CAMERA_POS,
    quat=look_at_quat(CAMERA_POS),
    fovy=FOVY_DEG,
    width=width,
    height=height,
    data_types=("depth", "segmentation"),
    use_textures=False,
    use_shadows=False,
    # Group 0 is the scene, group 2 the robot's visual meshes.  Leaving the
    # robot out renders a table the arm is not standing on, and on the real rig
    # the arm is the largest thing in frame and the main occluder.
    enabled_geom_groups=(0, 2),
  )


@requires_model_fields("cam_pos", "cam_quat")
def randomize_camera_pose(
  env,
  env_ids: torch.Tensor | None,
  pos_jitter: float = POS_JITTER_M,
  rot_jitter: float = ROT_JITTER_RAD,
) -> None:
  """Jitter the camera about its nominal pose, per environment.

  This is what makes the calibration requirement reachable.  A policy trained
  at one exact viewpoint fails when the camera is a centimetre off, and a
  camera is a centimetre off as soon as someone brushes the frame.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  cam_idx = env.scene.sensors[CAMERA_NAME].camera_idx
  base = torch.tensor(CAMERA_POS, device=env.device)
  pos = base + (2 * torch.rand(n, 3, device=env.device) - 1) * pos_jitter

  # Re-aim at the same point rather than perturbing the quaternion: a camera
  # nudged on its mount still points roughly where it was aimed, and this keeps
  # the jitter from quietly walking the target out of frame.
  aim = torch.tensor(CAMERA_AIM, device=env.device)
  fwd = torch.nn.functional.normalize(aim - pos, dim=-1)
  world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3)
  right = torch.nn.functional.normalize(torch.cross(fwd, world_up, dim=-1), dim=-1)
  up = torch.cross(right, fwd, dim=-1)

  # Then a small free rotation on top, which is the part a mount actually gets
  # wrong.
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
