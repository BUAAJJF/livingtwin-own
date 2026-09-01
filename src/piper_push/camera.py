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

from piper_push import d455_noise, depth_noise

CAMERA_NAME = "scene_cam"
PARENT_BODY = "robot/base_link"

CAMERA_POS = (-0.45412298521619643, 0.7674826404014604, 0.4887319349211286)
"""Metres in the robot base frame, measured by the 2026-08-26 D455 hand-eye
calibration (3.92 mm residual)."""

CAMERA_AIM = (0.01973900232618861, 0.4720208499413744, -0.01103385281335817)
"""Intersection of the measured D455 optical axis with its fitted table plane."""

CAMERA_QUAT = (
  0.45621114674051794,
  0.1920645871413448,
  -0.36009682378417324,
  -0.7907672612447207,
)
"""MuJoCo ``wxyz`` camera orientation in the robot base frame.

Unlike a look-at point, this retains the measured 1.91 degree camera roll.
The sign is arbitrary; this is the positive-w representative of the D455
OpenCV extrinsic after the ``diag(1, -1, -1)`` optical-to-MuJoCo conversion.
"""

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

# Depth sensor realism.  These used to be two placeholders -- 4 mm of Gaussian
# noise and 2% of uniformly scattered dropout -- with a note saying they were
# guesses and that guessing was the largest sim2real risk in the vision stage.
# The camera has since been measured (``hardware/depth_bench``) and the fitted
# model lives in ``piper_push.depth_noise``, which carries every number and the
# measurement it came from.  The short version of what the guess got wrong:
# the error grows as z^2; its static and temporal components differ; it is
# spatially correlated rather than independent; and invalid depth concentrates
# at stereo discontinuities instead of being uniformly scattered.  The D455
# model also quantises disparity on the hardware's 1/32-pixel grid.
DEPTH_NOISE = d455_noise.DEPTH_NOISE
MASK_JITTER_PX = 1
"""How far the target mask's boundary can be wrong.  On hardware the mask comes
from ``hardware/deploy/mask.py``, not from a segmentation buffer."""


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


def quat_matrix(quat=CAMERA_QUAT) -> np.ndarray:
  """Rotation matrix for a MuJoCo ``wxyz`` quaternion."""
  w, x, y, z = (float(v) for v in quat)
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
     2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
     2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w),
     1 - 2 * (x * x + y * y)],
  ], dtype=np.float64)


def fovx_deg(fovy: float = FOVY_DEG, width: int = WIDTH, height: int = HEIGHT) -> float:
  return 2 * math.degrees(math.atan(math.tan(math.radians(fovy / 2)) * width / height))


def f_px_per_rad(fovy: float = FOVY_DEG, height: int = HEIGHT) -> float:
  """Pixels per radian at the image centre.

  The measured sensor model states its correlation length and its edge
  threshold in angle, not in pixels, so that changing the policy's resolution
  or field of view does not silently change what the sensor does.  This is the
  conversion, and it is the only one.
  """
  return 0.5 * height / math.tan(math.radians(fovy) / 2)


def camera_cfg(width: int = WIDTH, height: int = HEIGHT) -> CameraSensorCfg:
  return CameraSensorCfg(
    name=CAMERA_NAME,
    parent_body=PARENT_BODY,
    pos=CAMERA_POS,
    quat=CAMERA_QUAT,
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


WRIST_CAMERA_NAME = "wrist_cam"
WRIST_PARENT_BODY = "robot/gripper_base"
# Behind and above the pads, tilted down the approach axis: far enough back
# that the fingers frame the view rather than fill it.  Nothing is mounted yet
# -- this is the pose to BUILD to if the measurement below says it is worth it.
# Derived, not guessed: the grasp site sits at (0, 0, 0.1265) in gripper_base,
# so the approach axis is +z and the camera looks along it from 120 mm back.
# This is the ideal on-axis mount -- an upper bound on what a wrist view can
# see, before any bracket has to make room for the fingers.
WRIST_CAMERA_POS = (0.0, 0.0, 0.0065)
WRIST_CAMERA_QUAT = (0.0, 0.7071067811865476, 0.7071067811865476, 0.0)
WRIST_FOVY_DEG = 70.0

WRIST_CUTOFF_M = 0.8
"""Metres.  The third-person camera cuts off at 1.5 m because it stands back
from the table; the wrist camera is 120 mm from the grasp site and a D405's
useful range starts at 70 mm, so a far clip that reaches the far wall would
spend most of the image's dynamic range on geometry the hand cannot act on."""

# A bracket is not a calibration.  The third-person camera moves when somebody
# leans on the frame, which is why its jitter is 20 mm; a wrist mount moves by
# its own machining tolerance and by however well the bolt pattern was seated,
# which is smaller and does not drift between runs.
WRIST_POS_JITTER_M = 0.004
WRIST_ROT_JITTER_RAD = math.radians(1.5)

WRIST_DEPTH_NOISE = depth_noise.DepthNoiseCfg()
"""The D405 model, which is what ``depth_noise`` fits by default.

The third-person camera is a D455 and carries the separately fitted
``d455_noise`` config; the wrist is a D405 and must not inherit it.  The two
sensors differ in exactly the way that matters here -- the D455's 95 mm stereo
baseline quantises disparity coarsely at range, and the D405's 18 mm baseline
at 120 mm does not -- so sharing one config would model the wrist as blurrier
than it is and the policy would learn to distrust the better sensor.
"""


def wrist_camera_cfg(width: int = WIDTH, height: int = HEIGHT) -> CameraSensorCfg:
  """A camera on the hand, which the arm cannot occlude.

  The third-person camera loses the target exactly when the hand reaches for
  it: measured on the rig, 15% of frames had a detectable object while it was
  present, and 71% of the pixels where the object should have been belonged to
  the arm or the table instead.  A camera that travels with the hand cannot be
  blocked by the arm -- but it has the opposite failure, losing the object out
  of frame whenever the hand is not pointed at it, so the two are complements
  and the number that matters is the union.
  """
  return CameraSensorCfg(
    name=WRIST_CAMERA_NAME,
    parent_body=WRIST_PARENT_BODY,
    pos=WRIST_CAMERA_POS,
    quat=WRIST_CAMERA_QUAT,
    fovy=WRIST_FOVY_DEG,
    width=width,
    height=height,
    data_types=("depth", "segmentation"),
    use_textures=False,
    use_shadows=False,
    enabled_geom_groups=(0, 2),
  )


@requires_model_fields("cam_pos", "cam_quat")
def randomize_camera_pose(
  env,
  env_ids: torch.Tensor | None,
  pos_jitter: float = POS_JITTER_M,
  rot_jitter: float = ROT_JITTER_RAD,
  sensor_name: str = CAMERA_NAME,
  nominal_pos: tuple[float, float, float] = CAMERA_POS,
  nominal_quat: tuple[float, float, float, float] = CAMERA_QUAT,
) -> None:
  """Jitter the camera about its nominal pose, per environment.

  This is what makes the calibration requirement reachable.  A policy trained
  at one exact viewpoint fails when the camera is a centimetre off, and a
  camera is a centimetre off as soon as someone brushes the frame.

  The defaults are the third-person D455.  A wrist camera passes its own
  name and nominal pose: same mechanism, but the jitter it models is a
  bracket's machining tolerance rather than a knocked tripod.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  cam_idx = env.scene.sensors[sensor_name].camera_idx
  base = torch.tensor(nominal_pos, device=env.device)
  pos = base + (2 * torch.rand(n, 3, device=env.device) - 1) * pos_jitter

  # A small free rotation on top of the *full* measured orientation.  Building
  # it again from a look-at point would silently discard the D455's measured
  # roll, which was one of the out-of-distribution axes in the old scene.
  axis = torch.nn.functional.normalize(torch.randn(n, 3, device=env.device), dim=-1)
  ang = (2 * torch.rand(n, 1, device=env.device) - 1) * rot_jitter
  k = torch.sin(ang / 2) * axis
  dq = torch.cat([torch.cos(ang / 2), k], dim=-1)

  q = torch.tensor(nominal_quat, device=env.device).expand(n, 4)

  w0, v0 = dq[:, :1], dq[:, 1:]
  w1, v1 = q[:, :1], q[:, 1:]
  quat = torch.cat([
    w0 * w1 - (v0 * v1).sum(-1, keepdim=True),
    w0 * v1 + w1 * v0 + torch.cross(v0, v1, dim=-1),
  ], dim=-1)
  quat = torch.nn.functional.normalize(quat, dim=-1)

  env.sim.model.cam_pos[env_ids, cam_idx] = pos
  env.sim.model.cam_quat[env_ids, cam_idx] = quat
