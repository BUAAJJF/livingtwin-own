"""The three channels, built the way ``pick_place.mdp.CameraScene`` builds them.

There is no cleverness here on purpose.  Every line has a counterpart in the
simulator and the whole value of the file is that the two agree; anything this
does differently is a distribution shift the policy was never trained through,
and it will show up as a policy that works in simulation and reaches past the
object on the robot.

The one thing that needs saying out loud is what a hole reads as.  The driver
returns 0 for a pixel it could not measure.  Zero is a perfectly good depth as
far as the normalisation is concerned -- it would come out as 0.0, the nearest
possible reading, so every hole would look like something pressed against the
lens.  The simulator's convention is the far plane, and this maps the driver's
zero onto it.  Both files say so, next to the line that does it.
"""

from __future__ import annotations

import numpy as np

from . import config


def normalise(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
  """Metres to the [0, 1] the policy was trained on.

  ``clamp(clamp(d, min, cutoff) / cutoff, 0, 1)`` -- the inner clamp is the
  sensor's range and the outer one is defensive, exactly as in the simulator.
  """
  d = np.asarray(depth, dtype=np.float32)
  if valid is None:
    valid = d > 0
  # A hole reads as the far plane.  ``CameraScene`` does the same, and this is
  # the only convention the two share by agreement rather than by derivation,
  # so it is written down in both places.
  d = np.where(valid, d, np.float32(config.CUTOFF_M))
  d = np.clip(d, config.MIN_DEPTH_M, config.CUTOFF_M) / config.CUTOFF_M
  return np.clip(d, 0.0, 1.0).astype(np.float32)


def flatten_scene(depth: np.ndarray, valid: np.ndarray, plane: np.ndarray,
                 points_base: np.ndarray | None = None,
                 arm=None, above_m: float = 0.40,
                 below_m: float = 0.02) -> tuple:
  """Replace the room with the table plane, which is all the simulator has.

  The scene the policy was trained against contains exactly one piece of
  scenery: an infinite MuJoCo ``PLANE`` at z = 0, with the bin, the object and
  the robot standing on it.  No walls, no floor at another height, no lab.

  Deployment's channel 0 is the lab.  Injecting one recorded frame's channel 0
  into the simulator -- mask untouched, everything else the simulator's own --
  took the trained policy from 170 objects placed to zero.  Every other
  difference that had been checked was exact: the calibration, the policy image
  to a pixel, all thirty-six proprioception values, forward kinematics, servo
  tracking.  This was not.

  Three things are kept and everything else is replaced:

  * **the tabletop and what stands on it**, from ``below_m`` under the plane to
    ``above_m`` over it;
  * **the arm**, by its own sphere cover rather than by height.  The first
    version used a 300 mm ceiling for this and the arm reaches 426-459 mm, so
    the top third of the robot was erased in every single frame -- while the
    comment beside the constant argued that cropping the arm out would be a
    second difference introduced to fix the first.  The model knows where the
    arm is; a threshold only guesses.
  * **holes**, which read as the far plane in both pipelines already.

  Pass ``points_base`` (the policy grid unprojected into the base frame) and
  ``arm`` (``proprio.Kinematics.link_spheres()``) to get the arm test.  Without
  them the height band alone is used and the arm above ``above_m`` is lost,
  which is a worse observation than not flattening at all.
  """
  from . import mask as _mask

  d = np.asarray(depth, dtype=np.float32)
  v = np.asarray(valid, dtype=bool)
  pl = np.asarray(plane, dtype=np.float32)
  finite = np.isfinite(pl)
  delta = np.where(finite, pl - d, 0.0)
  keep = v & finite & (delta > -below_m) & (delta < above_m)

  if points_base is not None and arm is not None:
    is_arm = _mask.arm_mask(
      np.asarray(points_base, dtype=np.float64).reshape(-1, 3), arm,
      clearance_m=0.0).reshape(d.shape)
    keep |= v & finite & is_arm

  out = np.where(keep, d, pl)
  out = np.where(finite, out, d)
  return out.astype(np.float32), (v | (finite & ~keep))


def camera_obs(
  depth: np.ndarray, valid: np.ndarray, target: np.ndarray
) -> np.ndarray:
  """``(3, H, W)`` float32: scene depth, the target mask, and their product.

  Args:
    depth: ``(H, W)`` metres on the policy's grid.
    valid: ``(H, W)`` bool, from the resampling.
    target: ``(H, W)`` bool -- the target instance only, not every object.
  """
  norm = normalise(depth, valid)
  # The mask cannot claim a pixel the sensor did not return.  The simulator
  # multiplies its mask by the same validity for exactly this reason: a mask
  # that survives where the depth does not is a channel the policy would learn
  # to trust more than the depth, and on the robot it is the other way round.
  mask = (np.asarray(target, dtype=bool) & np.asarray(valid, dtype=bool))
  mask = mask.astype(np.float32)
  return np.stack([norm, mask, norm * mask], axis=0)


def flatten(camera: np.ndarray) -> np.ndarray:
  """The layout the actor's first layer expects: channels, then rows, then
  columns, flattened.  ``torch.cat([...], dim=1).flatten(1)`` in the simulator.
  """
  return np.ascontiguousarray(camera, dtype=np.float32).reshape(-1)
