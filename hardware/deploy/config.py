"""Every number the deployed pipeline needs, and where each one comes from.

Three sources, and the difference matters:

* **imported from the simulator** -- resolution, field of view, far plane,
  nominal camera pose.  These are not deployment choices.  The policy was
  trained against them and if the robot uses a different value the policy is
  being shown an image it has never seen.  They are imported rather than
  copied so that changing ``piper_push.camera`` cannot leave this file
  quietly stale.

* **measured on the rig** -- the camera extrinsic and the table plane.  Both
  come out of ``calibrate.py`` and are written to ``rig.json``; the values
  here are only what to assume before that has been run, and running without
  it is a diagnostic, not a deployment.

* **properties of the hardware** -- the D405's own resolution and the CAN
  interface the arm is on.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

# mjlab first, and not for anything this module uses.  Importing mjlab runs its
# entry-point scan, which imports ``piper_push.tasks``, which builds the task
# configs, which reads ``piper_push.camera``.  Reaching for the camera module
# first therefore starts that chain from inside itself and the task package
# fails to register -- with a warning, not an error, so the first symptom is a
# task id that does not exist.  The tests in ``tests/`` carry the same import
# and the same note.
import mjlab.tasks  # noqa: F401

from piper_push import camera as sim_camera
from piper_push import layout as sim_layout
from piper_push import objects as sim_objects

HERE = pathlib.Path(__file__).resolve().parent
RIG_FILE = HERE / "rig.json"

# --- imported from the simulator; do not edit here --------------------------

WIDTH = sim_camera.WIDTH
HEIGHT = sim_camera.HEIGHT
FOVY_DEG = sim_camera.FOVY_DEG
CUTOFF_M = sim_camera.CUTOFF_M
MIN_DEPTH_M = 0.05
"""Matches the ``min_depth`` of ``pick_place.mdp.CameraScene``."""

CAMERA_POS = np.asarray(sim_camera.CAMERA_POS, dtype=np.float64)
CAMERA_AIM = np.asarray(sim_camera.CAMERA_AIM, dtype=np.float64)

CONTROL_HZ = 50.0
"""The rate the policy was trained at: ``decimation=10`` over a 2 ms step."""

# --- hardware ---------------------------------------------------------------

D405_WIDTH, D405_HEIGHT = 848, 480
D405_FPS = 30
"""The camera runs at 30 Hz and the policy at 50.  The loop does not wait for
a new frame; it reuses the last one.  That is not a compromise, it is what the
simulator does too -- the camera sensor is read at the control rate and the
renderer does not produce a new image every physics step either."""

CAN_INTERFACE = "can0"

# --- measured on the rig; written by calibrate.py ---------------------------

TABLE_Z_M = 0.0
"""Height of the table surface in the robot base frame.  Zero is where the
simulator puts it; the real number comes from the calibration, which fits the
plane rather than trusting anyone's tape measure."""

BIN_CENTER = tuple(sim_objects.BIN_CENTER)
BIN_INNER = tuple(sim_objects.BIN_INNER)
BIN_OUTER = tuple(h + sim_objects.BIN_WALL_THICKNESS for h in sim_objects.BIN_INNER)
"""Outer half-extents.  ``BIN_INNER`` is the opening; the walls stand outside
it, and a segmenter that masks only the opening reports the rim -- 60 mm of
vertical wall, squarely inside the 24-90 mm the objects occupy -- as the
nearest object to the hand."""
"""Where the bin is and how big it is, in the base frame.  Imported from the
simulator for the same reason the camera pose is -- on the rig the bin goes
where this says, and the segmenter needs to know so it does not report the rim
as the tallest object on the table."""

_WORKSPACE_XY_UNROTATED = ((-0.10, 0.75), (-0.45, 0.45))
WORKSPACE = (*sim_layout.rotate_aabb_xy(_WORKSPACE_XY_UNROTATED), (-0.02, 0.40))
"""Base-frame box that anything interesting is inside, as (x, y, z) ranges.
The coarse bound: it throws away the far wall, the floor and whatever else is
in frame.  It is *not* where an object may be -- see ``WORKSPACE_SECTOR``,
which is the shape training actually uses and which this box circumscribes."""

# The sector, imported rather than restated.  ``piper_push`` is already fully
# imported by this point (see the mjlab note at the top of this file), so the
# task module is safe to read here and cannot drift from what was trained.
from piper_push.tasks.pick_place import env_cfg as _task  # noqa: E402

WORKSPACE_SECTOR = (
  _task.OBJECT_ALLOWED_RADIUS,
  _task.OBJECT_ALLOWED_ANGLE,
  (-0.02, 0.40),
)
"""Where an object is allowed to be, as ``((r_lo, r_hi), (a_lo, a_hi), (z_lo,
z_hi))`` in base-frame polar coordinates -- the same annular sector
``OBJECT_ALLOWED_RADIUS`` / ``OBJECT_ALLOWED_ANGLE`` define for training.

The deployment used the circumscribing *box* instead, and the box contains two
regions the sector does not.  Both produced instances on the rig:

* **inside the inner radius**, r < 150 mm, which is where the robot's own base
  column stands.  It came back as a 102 mm tall, 283 px component -- squarely
  inside the 24-90 mm the objects occupy once smoothing is allowed for -- and
  ``link_spheres`` does not remove it, because the sphere cover follows the
  collision geometry and the base's cover does not reach the table.
* **outside the outer radius**, r > 550 mm, where the camera's own mount sits.
  That came back as a 1218 px component, seven times an object's area, and the
  nearest-to-the-hand rule chose it in 15% of a recorded session.

Neither is a segmentation failure: both are real things standing above the
table.  They are simply not places the task ever puts an object, and training
has always known that."""


@dataclasses.dataclass
class Rig:
  """What calibration measured about this particular rig.

  ``T_base_cam`` is the transform that takes a point in the *camera optical*
  frame -- OpenCV convention, x right, y down, z forward -- into the robot base
  frame.  That is what ``cv2.calibrateHandEye`` returns and what the camera
  driver's own frame is, so nothing has to be flipped on the way in.  The
  simulator's camera is a MuJoCo one and looks down its own -z with +y up; the
  conversion between the two lives in ``rectify.py`` and nowhere else.
  """

  T_base_cam: np.ndarray
  table_z: float = TABLE_Z_M
  table_normal_base: np.ndarray | None = None
  """Unit normal of the measured table plane in the robot base frame.  The
  plane passes through the point (0, 0, table_z).  Old rig files omit it and
  are treated as level for backward compatibility."""
  K: np.ndarray | None = None
  """D405 depth intrinsics at ``(D405_WIDTH, D405_HEIGHT)``.  Read from the
  camera at run time; stored so a recorded session can be replayed without
  one."""
  serial: str | None = None
  residual_mm: float | None = None
  """Hand-eye reprojection residual.  Recorded because a calibration that has
  not been checked is a calibration that will be blamed for something else."""
  table_tilt_deg: float | None = None
  table_flatness_mm: float | None = None
  """How far off level the table is in the base frame, and how far its surface
  departs from the plane that was fitted to it.  Stored rather than printed
  and forgotten because the simulator's table is exactly flat and exactly
  level, and neither of those is randomised -- so these two numbers are a
  measured sim2real gap, and ``scripts/rig_to_sim.py`` is what reads them."""

  @classmethod
  def nominal(cls) -> "Rig":
    """Where the camera is *supposed* to be, from the simulator's own pose.

    Useful for a dry run and for the round-trip test, and wrong by however far
    the mount actually is.  The policy was trained with 20 mm and 2 degrees of
    camera jitter, so this is inside the envelope if the mount is good -- but
    "inside the envelope" is an argument for calibrating, not against it.
    """
    return cls(T_base_cam=sim_camera_extrinsic(), residual_mm=None)

  @classmethod
  def load(cls, path: pathlib.Path | str = RIG_FILE) -> "Rig":
    d = json.loads(pathlib.Path(path).read_text())
    return cls(
      T_base_cam=np.asarray(d["T_base_cam"], dtype=np.float64),
      table_z=float(d.get("table_z", TABLE_Z_M)),
      table_normal_base=(
        np.asarray(d["table_normal_base"], dtype=np.float64)
        if d.get("table_normal_base") is not None else None),
      K=np.asarray(d["K"], dtype=np.float64) if d.get("K") else None,
      serial=d.get("serial"),
      residual_mm=d.get("residual_mm"),
      table_tilt_deg=d.get("table_tilt_deg"),
      table_flatness_mm=d.get("table_flatness_mm"),
    )

  def save(self, path: pathlib.Path | str = RIG_FILE) -> None:
    pathlib.Path(path).write_text(json.dumps({
      "T_base_cam": self.T_base_cam.tolist(),
      "table_z": self.table_z,
      "table_normal_base": (
        np.asarray(self.table_normal_base).tolist()
        if self.table_normal_base is not None else None),
      "K": self.K.tolist() if self.K is not None else None,
      "serial": self.serial,
      "residual_mm": self.residual_mm,
      "table_tilt_deg": self.table_tilt_deg,
      "table_flatness_mm": self.table_flatness_mm,
    }, indent=2))


def sim_camera_extrinsic() -> np.ndarray:
  """The simulator camera's pose, as an OpenCV-convention camera-to-base.

  MuJoCo's camera looks down -z with +y up in the image; OpenCV's looks down
  +z with +y down.  The rotation between them is ``diag(1, -1, -1)``, and this
  is the one place in the deployed code that knows it.
  """
  pos = CAMERA_POS
  # MuJoCo camera axes are (right, up, backward); OpenCV's are (right,
  # down, forward).  Reading the configured quaternion, rather than rebuilding
  # it from CAMERA_AIM, preserves the calibrated D455 roll.
  R = sim_camera.quat_matrix() @ np.diag([1.0, -1.0, -1.0])
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = pos
  return T
