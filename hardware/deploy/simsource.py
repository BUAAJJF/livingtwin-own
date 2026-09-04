"""An RGB-D source backed by the simulator, shaped like the real camera.

The deployment's perception stack -- ``mask.DepthSegmenter``,
``mask.TargetTracker``, ``lifecycle.TargetLifecycle``, ``rectify.Reprojector``,
``obs.camera_obs`` -- has never been run against anything but a real camera or
a recording of one.  Everything simulation knows about the target it gets from
the renderer's segmentation buffer, by a completely separate path.  So the two
have no shared implementation and no way to disagree in public, and the errors
this repository has actually shipped were exactly of that kind: a segmenter
constant that discarded the target, an occlusion metric that measured the wrong
geometry, a mask process fitted to the wrong session.

This closes the loop.  It renders the simulated scene **at the real sensor's
resolution and intrinsics**, hands it to the same ``Frame`` the RealSense
reader produces, and lets the deployment's own code segment it.  What the
simulator additionally knows -- which pixels are really the target -- becomes
ground truth that the rig can never supply, so the perception stack can be
scored rather than eyeballed.

Two conventions have to be right or every number downstream is quietly wrong,
and both are checked rather than asserted; see ``self_check``.

* **Intrinsics.**  MuJoCo's camera has square pixels and a principal point
  forced to the exact centre.  The D455's measured ``K`` is
  ``fx = fy = 427.35``, ``cx, cy = 426.84, 242.74`` at 848x480.  Choosing
  ``fovy = 2 atan(H / 2 fy) = 58.637 deg`` reproduces the focal length exactly;
  the principal point still lands 2.8 px away and nothing can be done about
  that inside MuJoCo.  Derived from the rig file rather than typed in, so
  recalibrating the camera moves the simulator with it.

* **Frame.**  MuJoCo cameras look down their own -z with +y up; OpenCV, which
  is what ``config.Rig.T_base_cam`` and ``cv2.calibrateHandEye`` speak, looks
  down +z with +y down.  The conversion is one flip, ``diag(1, -1, -1)``, and
  applying it twice or not at all both produce a plausible-looking point cloud
  in the wrong place.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib

import numpy as np

from . import config, rectify, sensor

# OpenCV's camera frame from MuJoCo's: x stays, y and z flip.
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


def fovy_for(K: np.ndarray, height: int) -> float:
  """The MuJoCo vertical FOV that reproduces a measured focal length."""
  return 2.0 * math.degrees(math.atan(height / (2.0 * float(K[1, 1]))))


def quat_to_mat(q) -> np.ndarray:
  w, x, y, z = (float(v) for v in q)
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ], dtype=np.float64)


@dataclasses.dataclass
class SimFrameBundle:
  """One rendered step: what the camera sees, and what is actually there."""

  frame: sensor.Frame
  truth_mask: np.ndarray
  """``(H, W)`` bool at sensor resolution -- the target's own pixels, from the
  renderer.  This is the thing no real camera can hand you."""
  joints: np.ndarray
  """``(8,)`` arm joints plus the two finger positions, as ``proprio`` wants."""
  grasped: bool


class SimSource:
  """Render an mjlab scene as if it were the deployment's D455.

  Owns the environment, so the caller drives it with ``step(action)`` exactly
  as it drives an arm, and reads frames exactly as it reads a camera.  That
  symmetry is the point: a validation run and a deployment run should differ
  in where the pixels come from and what the actions reach, and in nothing
  else.
  """

  def __init__(self, task: str = "Mjlab-Pick-Place-PiperX-Robust",
               device: str = "cuda:0",
               rig_path: str | pathlib.Path = "hardware/deploy/rig_d455.json",
               num_envs: int = 1, env_id: int = 0, seed: int = 0,
               width: int = config.D405_WIDTH,
               height: int = config.D405_HEIGHT) -> None:
    import torch
    import mjlab.tasks  # noqa: F401  -- registers task ids
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import CameraSensorCfg
    from mjlab.tasks.registry import load_env_cfg
    from piper_push import camera as sim_camera

    self._torch = torch
    self.env_id = int(env_id)
    self.width, self.height = int(width), int(height)

    real = json.loads(pathlib.Path(rig_path).read_text())
    K_real = np.asarray(real["K"], dtype=np.float64)
    self.fovy = fovy_for(K_real, self.height)

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = int(num_envs)
    # The STATE task by default, which carries no camera of its own.  mjlab
    # requires every camera in a scene to agree on ``use_textures`` and the
    # policy camera sets it False, so sharing a scene with it would mean
    # flat-shaded RGB -- useless for anything that looks at colour, which is
    # the entire reason this renders RGB.  Nothing is lost: the policy's
    # 224x168 view is produced from this camera by the deployment's own
    # Reprojector, which is the path being validated.
    existing = tuple(cfg.scene.sensors or ())
    clash = [s for s in existing if getattr(s, "use_textures", True) is False]
    if clash:
      raise ValueError(
        f"{task} already has camera(s) {[s.name for s in clash]} with "
        "use_textures=False; every camera in an mjlab scene must agree, so "
        "textured RGB needs a task without the policy camera (the default "
        "state task) -- the policy view comes from the Reprojector here")
    cfg.scene.sensors = existing + (CameraSensorCfg(
      name="rig_cam", parent_body=sim_camera.PARENT_BODY,
      pos=sim_camera.CAMERA_POS, quat=sim_camera.CAMERA_QUAT,
      fovy=self.fovy, width=self.width, height=self.height,
      data_types=("rgb", "depth", "segmentation"),
      use_textures=True, use_shadows=False,
      enabled_geom_groups=(0, 2)),)

    torch.manual_seed(int(seed))
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    self.env.reset()
    self.cmd = self.env.command_manager.get_term("pick")
    self.robot = self.env.scene["robot"]
    self.sensor = self.env.scene["rig_cam"]
    self._index = -1

    R_mj = quat_to_mat(sim_camera.CAMERA_QUAT)
    T = np.eye(4)
    T[:3, :3] = R_mj @ _MJ_TO_CV
    T[:3, 3] = np.asarray(sim_camera.CAMERA_POS, dtype=np.float64)
    self.rig = config.Rig(
      T_base_cam=T,
      table_z=0.0,
      table_normal_base=np.array([0.0, 0.0, 1.0]),
      K=rectify.mujoco_K(self.width, self.height, self.fovy),
      serial="simulated",
      residual_mm=0.0,
    )
    self.meta = {"width": self.width, "height": self.height,
                 "K": self.rig.K.tolist(), "source": "simulation",
                 "fovy_deg": self.fovy, "task": task}

  # -- driving it ---------------------------------------------------------

  @property
  def action_dim(self) -> int:
    return int(self.env.action_manager.total_action_dim)

  def step(self, action=None) -> None:
    torch = self._torch
    if action is None:
      action = torch.zeros(self.env.num_envs, self.action_dim,
                           device=self.env.device)
    self.env.step(action)
    self._index += 1

  # -- reading it ---------------------------------------------------------

  def latest(self) -> SimFrameBundle:
    import mujoco
    b = self.env_id
    depth = self.sensor.data.depth[b, ..., 0].cpu().numpy().astype(np.float32)
    rgb = self.sensor.data.rgb[b].cpu().numpy()
    seg = self.sensor.data.segmentation[b]
    ids, types = seg[..., 0], seg[..., 1]
    tgt = self.cmd.target_geom_ids[b].to(ids.device)
    truth = ((ids.unsqueeze(-1) == tgt.view(1, 1, -1)).any(-1)
             & (types == int(mujoco.mjtObj.mjOBJ_GEOM))).cpu().numpy()

    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]).astype(np.uint8)
    q = self.robot.data.joint_pos[b].cpu().numpy().astype(np.float64)
    frame = sensor.Frame(depth=depth, gray=gray, stamp=float(self._index) * 0.02,
                         index=self._index)
    frame.rgb = rgb                                        # sim-only extra
    return SimFrameBundle(frame=frame, truth_mask=truth, joints=q,
                          grasped=bool(self.cmd.grasped[b]))

  def close(self) -> None:
    self.env.close()

  # -- the two conventions, checked ---------------------------------------

  def self_check(self, tol_mm: float = 8.0) -> dict:
    """Unproject the target's own pixels and see where they land.

    If the frame convention or the intrinsics are wrong the cloud is still a
    cloud -- plausible, dense, and somewhere else.  So this takes the pixels
    the renderer says are the target, unprojects them with the ``Rig`` this
    class just built, and compares against the object the simulator reports.
    A flip in ``_MJ_TO_CV`` moves it by the better part of a metre; a
    factor-of-two focal error moves it by the stand-off distance.

    Compared against the object's SURFACE, not its centre.  The camera sees
    the top of a box and the centre is half its height below; checking the
    centre reports a ~18 mm error on a correct calibration, and the only way
    to make that pass is a tolerance loose enough to hide a real fault.  So
    the residual is the distance from each unprojected point to the object's
    bounding box, which is zero for a surface point and grows for a wrong
    pose either way.
    """
    b = self.env_id
    bundle = self.latest()
    ys, xs = np.nonzero(bundle.truth_mask & (bundle.frame.depth > 0))
    if xs.size < 20:
      return {"ok": False, "reason": "target not visible enough to check",
              "pixels": int(xs.size)}
    z = bundle.frame.depth[ys, xs].astype(np.float64)
    K = self.rig.K
    cam = np.stack([(xs - K[0, 2]) / K[0, 0] * z,
                    (ys - K[1, 2]) / K[1, 1] * z, z], axis=1)
    base = cam @ self.rig.T_base_cam[:3, :3].T + self.rig.T_base_cam[:3, 3]

    centre = self.cmd._object_pos_local()[b].cpu().numpy().astype(np.float64)
    half = self.cmd.object_half_size[b].cpu().numpy().astype(np.float64)
    # Distance to the axis-aligned box.  The object can be rotated, so this is
    # a lower bound on the true surface distance -- fine for catching a frame
    # flip, and honest about not being a pose check.
    outside = np.maximum(np.abs(base - centre) - half, 0.0)
    resid = np.linalg.norm(outside, axis=1)
    med = float(np.median(resid)) * 1000.0
    return {"ok": med < tol_mm, "pixels": int(xs.size),
            "median_distance_to_object_box_mm": round(med, 1),
            "p90_mm": round(float(np.percentile(resid, 90)) * 1000.0, 1),
            "centroid_offset_mm": round(
              float(np.linalg.norm(np.median(base, axis=0) - centre)) * 1000.0, 1),
            "tolerance_mm": tol_mm}
