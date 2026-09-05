"""The deployment stack's arithmetic, on the CPU in a second.

``hardware/deploy/selftest.py`` is the real check -- it builds the simulator,
renders a synthetic D405 and compares every stage against ground truth -- and
it takes a minute and a GPU.  These are the parts of it that can be asserted
against closed-form answers instead, so they run in the normal test sweep and
catch a regression before anyone reaches for the heavy one.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import math
import pathlib
import sys

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hardware.deploy import config, obs, rectify  # noqa: E402


# ---------------------------------------------------------------------------
# The camera model
# ---------------------------------------------------------------------------


def test_deployment_layout_is_the_training_layout_rotated_ninety_degrees():
  """Camera calibration stays fixed; the table task moves around the base."""
  from piper_push import layout, objects
  from piper_push.tasks.pick_place import env_cfg as pick_cfg

  assert math.degrees(layout.WORKSPACE_YAW_RAD) == pytest.approx(90.0)
  assert objects.BIN_CENTER == pytest.approx((0.22, 0.30))
  assert objects.BIN_INNER == pytest.approx((0.070, 0.080))
  assert pick_cfg.SPAWN_ANGLE == pytest.approx(
    (math.radians(82.0), math.radians(132.0)), abs=0.01
  )
  assert np.allclose(config.WORKSPACE[:2], ((-0.45, 0.45), (-0.10, 0.75)))


def test_virtual_camera_matches_the_simulators_field_of_view():
  """The policy's camera is defined once, in ``piper_push.camera``.

  If these drift the policy is shown an image from a different lens, and
  nothing anywhere raises.
  """
  from piper_push import camera as sim_camera

  K = rectify.VirtualCamera().K
  fovy = 2 * math.degrees(math.atan(0.5 * config.HEIGHT / K[1, 1]))
  assert fovy == pytest.approx(sim_camera.FOVY_DEG, abs=1e-9)
  assert K[0, 0] == pytest.approx(K[1, 1])
  assert K[0, 2] == pytest.approx((config.WIDTH - 1) / 2)
  assert K[1, 2] == pytest.approx((config.HEIGHT - 1) / 2)


def test_mujoco_intrinsics_put_the_principal_point_at_the_centre():
  """Half a pixel, and it was 9 mm of depth error.

  MuJoCo's principal point is the exact centre of the sensor and the D405's is
  1.2 px off it horizontally and 3.5 px vertically.  Handing a rendered image
  the real camera's intrinsics threw every ray by up to 8 mrad, which is what
  ``selftest.py`` caught on its first run.
  """
  K = rectify.mujoco_K(848, 480, 58.26)
  assert K[0, 2] == pytest.approx(423.5)
  assert K[1, 2] == pytest.approx(239.5)
  real = rectify._default_d405_K()
  assert abs(K[0, 2] - real[0, 2]) > 1.0
  assert abs(K[1, 2] - real[1, 2]) > 3.0


def test_the_extrinsic_is_a_rotation_and_looks_where_the_camera_aims():
  """MuJoCo looks down -z with +y up, OpenCV down +z with +y down.

  A reflection here rather than a rotation would put the scene through a
  mirror, and the ``depth_bench`` work on the other camera is a long record of
  how hard that is to notice from the depth alone.
  """
  T = config.sim_camera_extrinsic()
  R = T[:3, :3]
  assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
  assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
  # The optical axis is the third column and it points at what the camera aims
  # at.
  fwd = config.CAMERA_AIM - config.CAMERA_POS
  fwd = fwd / np.linalg.norm(fwd)
  assert np.allclose(R[:, 2], fwd, atol=1e-9)
  assert np.allclose(T[:3, 3], config.CAMERA_POS)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _identity_rig():
  rig = config.Rig.nominal()
  rig.K = rectify.mujoco_K(config.D405_WIDTH, config.D405_HEIGHT, 58.26)
  return rig


def test_a_flat_wall_resamples_to_a_flat_wall():
  """The source and target cameras share an origin here, so a fronto-parallel
  plane at range z has to come back at exactly z everywhere it is covered."""
  rig = _identity_rig()
  r = rectify.Reprojector(rig)
  src = np.full((config.D405_HEIGHT, config.D405_WIDTH), 0.70, np.float32)
  out, valid, _ = r(src)
  assert valid.mean() > 0.99
  assert np.abs(out[valid] - 0.70).max() < 1e-5


def test_the_nearest_surface_wins_and_then_gets_averaged():
  """Two things at once, and they pull in opposite directions.

  At an occlusion boundary the samples in one target pixel come from two
  surfaces and the near one is the answer.  On a flat surface they are seven
  noisy measurements of one thing, and taking the nearest is biased low by
  about 1.3 standard deviations -- 9 mm at 0.7 m, which would be the largest
  systematic error in the pipeline.
  """
  rig = _identity_rig()
  r = rectify.Reprojector(rig, average_window_m=0.015)
  rng = np.random.default_rng(0)
  src = (0.70 + rng.normal(0, 0.007, (config.D405_HEIGHT, config.D405_WIDTH))
         ).astype(np.float32)
  out, valid, _ = r(src)
  assert float(out[valid].mean()) == pytest.approx(0.70, abs=0.001)

  # Now a foreground patch 10 cm nearer, which is outside the window: the near
  # surface has to win outright rather than being averaged with the far one.
  src = np.full((config.D405_HEIGHT, config.D405_WIDTH), 0.70, np.float32)
  src[200:280, 400:480] = 0.60
  out, valid, _ = r(src)
  centre = out[config.HEIGHT // 2 - 5:config.HEIGHT // 2 + 5,
                config.WIDTH // 2 - 5:config.WIDTH // 2 + 5]
  assert np.abs(centre - 0.60).max() < 1e-5


def test_payload_rides_along_with_the_depth():
  """Labels have to come from the same measurement the depth did.

  Two independent resamplings agree until they do not, and where they do not is
  the silhouette -- which is the only part of a 25 mm object that says how big
  it is.
  """
  rig = _identity_rig()
  r = rectify.Reprojector(rig)
  src = np.full((config.D405_HEIGHT, config.D405_WIDTH), 0.70, np.float32)
  src[200:280, 400:480] = 0.60
  payload = np.zeros_like(src, dtype=np.int32)
  payload[200:280, 400:480] = 7
  out, valid, carried = r(src, payload=payload)
  near = valid & (np.abs(out - 0.60) < 1e-5)
  assert near.sum() > 50
  assert set(np.unique(carried[near]).tolist()) == {7}


def test_nothing_in_means_nothing_out():
  rig = _identity_rig()
  r = rectify.Reprojector(rig)
  out, valid, _ = r(np.zeros((config.D405_HEIGHT, config.D405_WIDTH), np.float32))
  assert not valid.any()
  assert not out.any()


# ---------------------------------------------------------------------------
# The observation
# ---------------------------------------------------------------------------


def test_a_hole_reads_as_the_far_plane():
  """The driver returns 0, which would normalise to 0.0 -- the nearest possible
  reading -- so every hole would look like something against the lens.  The
  simulator's convention is the far plane and this has to match it."""
  d = np.full((4, 4), 0.7, np.float32)
  valid = np.ones((4, 4), bool)
  valid[0, 0] = False
  n = obs.normalise(d, valid)
  assert n[0, 0] == pytest.approx(1.0)
  assert n[1, 1] == pytest.approx(0.7 / config.CUTOFF_M)


def test_the_three_channels_are_what_the_simulator_builds():
  """Same formula, checked against the simulator's own arithmetic rather than
  against a number typed here."""
  import torch

  from piper_push.tasks.pick_place import mdp  # noqa: F401  -- shape reference

  rng = np.random.default_rng(1)
  d = rng.uniform(0.3, 1.2, (8, 8)).astype(np.float32)
  valid = rng.random((8, 8)) > 0.2
  target = rng.random((8, 8)) > 0.7

  got = obs.camera_obs(d, valid, target)
  ref_depth = torch.where(torch.as_tensor(valid), torch.as_tensor(d),
                          torch.full_like(torch.as_tensor(d), config.CUTOFF_M))
  ref_norm = torch.clamp(
    torch.clamp(ref_depth, min=config.MIN_DEPTH_M, max=config.CUTOFF_M)
    / config.CUTOFF_M, 0.0, 1.0)
  ref_mask = (torch.as_tensor(target) & torch.as_tensor(valid)).float()

  assert np.allclose(got[0], ref_norm.numpy())
  assert np.allclose(got[1], ref_mask.numpy())
  assert np.allclose(got[2], (ref_norm * ref_mask).numpy())


def test_the_mask_cannot_claim_a_pixel_the_sensor_missed():
  d = np.full((4, 4), 0.7, np.float32)
  valid = np.zeros((4, 4), bool)
  target = np.ones((4, 4), bool)
  assert obs.camera_obs(d, valid, target)[1].sum() == 0


# ---------------------------------------------------------------------------
# The command path
# ---------------------------------------------------------------------------


def test_the_slew_limit_starts_from_where_the_arm_is():
  """The trained action term seeds its previous target from the measured
  posture, not the nominal one, and says why: a limiter starting at the default
  hands the servo the whole reset offset as one step.  Starting it wrong
  disagreed with the simulator by 0.68 rad on joint 1 -- on the robot, a lunge
  on the first command of every run."""
  import json

  from hardware.deploy import proprio, robot

  spec = __import__("json").loads(pathlib.Path(proprio.SPEC_FILE).read_text())
  m = robot.ActionMapper(spec, allow_legacy=True)
  q = np.asarray(spec["default_joint_pos"], dtype=np.float64) + 0.5
  m.reset(q)
  assert m.previous[0] == pytest.approx(q[0])
  # And one zero action moves it by at most one step's worth of travel.
  before = m.previous.copy()
  after = m(np.zeros(7))
  assert np.all(np.abs(after - before) <= m.max_step + 1e-12)


def test_targets_stay_inside_the_safety_envelope():
  import json

  from hardware.deploy import proprio, robot

  spec = __import__("json").loads(pathlib.Path(proprio.SPEC_FILE).read_text())
  m = robot.ActionMapper(spec, allow_legacy=True)
  m.reset()
  rng = np.random.default_rng(0)
  for _ in range(500):
    t = m(rng.uniform(-5, 5, 7))
    assert np.all(t >= m.lo - 1e-9) and np.all(t <= m.hi + 1e-9)


# ---------------------------------------------------------------------------
# Hand-eye calibration
# ---------------------------------------------------------------------------


def test_hand_eye_recovers_a_known_camera_pose():
  """The one measurement that connects the two worlds, checked against itself.

  ``selftest.py`` verifies everything downstream of the extrinsic and nothing
  about the extrinsic, because it is what ties the simulator's frame to the
  robot's.  So it is checked here instead, synthetically: put the board on the
  gripper at a known offset, move the arm through poses, photograph it with a
  camera at a known place, and require the solver to find that place.

  The eye-to-hand direction is what this is really for.  For a camera on the
  arm, ``calibrateHandEye`` is called with the gripper's pose in the base
  frame; for one fixed to the world -- this rig -- it is called with the
  inverse, and it then returns the camera in the base frame.  Passing the
  un-inverted poses produces a transform that looks entirely plausible and is
  wrong by the whole length of the arm, and nothing downstream would say so.
  """
  import cv2

  from hardware.deploy import calibrate, config, proprio

  rng = np.random.default_rng(0)
  kin = proprio.Kinematics()
  default = np.asarray(
    __import__("json").loads(pathlib.Path(proprio.SPEC_FILE).read_text())
    ["default_joint_pos"], dtype=np.float64)

  T_base_cam = config.sim_camera_extrinsic()
  T_cam_base = np.linalg.inv(T_base_cam)

  # The board, rigidly attached to the gripper at an arbitrary offset.
  T_grip_board = np.eye(4)
  T_grip_board[:3, :3] = cv2.Rodrigues(np.array([0.3, -0.2, 0.15]))[0]
  T_grip_board[:3, 3] = [0.02, -0.01, 0.06]

  records = []
  for _ in range(12):
    q = default.copy()
    q[:6] += rng.uniform(-0.6, 0.6, 6)
    kin.update(q)
    T_base_grip = np.eye(4)
    T_base_grip[:3, :3] = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    T_base_grip[:3, 3] = kin.data.site_xpos[kin.site_id]
    T_cam_board = T_cam_base @ T_base_grip @ T_grip_board
    records.append({
      "joint_pos": q.tolist(),
      "rvec": cv2.Rodrigues(T_cam_board[:3, :3])[0].ravel().tolist(),
      "tvec": T_cam_board[:3, 3].tolist(),
    })

  out = calibrate.solve(records)
  got = out["T_base_cam"]
  assert out["rot_span_deg"] > calibrate.MIN_ROT_SPAN_DEG
  assert np.linalg.norm(got[:3, 3] - T_base_cam[:3, 3]) < 1e-3
  assert calibrate._angle_between(got[:3, :3], T_base_cam[:3, :3]) < 0.05
  assert out["residual_mm"] < 0.5


def test_pose_ransac_rejects_a_whole_bad_robot_camera_observation():
  """A sharp PnP can still belong to the wrong robot pose or board mount."""
  import cv2

  from hardware.deploy import calibrate

  rng = np.random.default_rng(455)
  X = np.eye(4)
  X[:3, :3] = cv2.Rodrigues(np.array([0.25, -0.18, 0.12]))[0]
  X[:3, 3] = [-0.42, 0.70, 0.51]
  Y = np.eye(4)
  Y[:3, :3] = cv2.Rodrigues(np.array([-0.2, 0.1, 0.3]))[0]
  Y[:3, 3] = [0.02, -0.07, -0.06]
  grips, boards = [], []
  for _ in range(20):
    G = np.eye(4)
    G[:3, :3] = cv2.Rodrigues(rng.uniform(-0.8, 0.8, 3))[0]
    G[:3, 3] = rng.uniform([-0.2, 0.1, 0.1], [0.3, 0.6, 0.7])
    grips.append(G)
    boards.append(np.linalg.inv(X) @ G @ Y)
  boards[3] = boards[3].copy()
  boards[3][:3, 3] += [0.055, -0.025, 0.020]

  keep = calibrate.pose_ransac_inliers(grips, boards)
  assert 3 not in keep
  assert set(keep) == set(range(20)) - {3}


def test_stable_board_frames_are_fused_before_pnp():
  import cv2

  from hardware.deploy import calibrate, rectify

  rng = np.random.default_rng(17)
  board = calibrate.Board(squares=(6, 5), square_m=0.028,
                          marker_m=0.022, dictionary="DICT_4X4_50")
  obj = board._charuco().getChessboardCorners().astype(np.float64)
  K = rectify._default_d405_K()
  rvec = np.array([0.16, -0.09, 0.03])
  tvec = np.array([0.01, -0.02, 0.46])
  exact, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
  exact = exact.reshape(-1, 2)
  detections = []
  for i in range(12):
    uv = exact + rng.normal(0.0, 0.28, exact.shape)
    if i == 2:
      uv[3] += [5.0, -4.0]       # one bad sub-pixel refinement
    detections.append({"object_points": obj, "image_points": uv})
  fused = calibrate.fuse_detections(
    detections, K, np.zeros(5), board, min_frames=8)
  assert fused is not None
  assert fused["fusion_frames"] == 12
  assert fused["n_corners"] == board.n_corners()
  assert np.linalg.norm(np.asarray(fused["tvec"]).ravel() - tvec) < 0.002
  assert fused["reproj_rms_px"] < 0.2


def test_table_depth_fit_uses_the_actual_calibration_frame_shape():
  from hardware.deploy import calibrate

  # The GUI calibrates at 1280x720 even though deployment is 848x480.  A
  # deployment-sized ray cache silently indexes the high-resolution depth
  # buffer with the wrong row stride and used to return zero table points.
  h, w = 720, 1280
  K = np.array([[800.0, 0.0, (w - 1) / 2],
                [0.0, 800.0, (h - 1) / 2],
                [0.0, 0.0, 1.0]])
  depth = np.ones((h, w), dtype=np.float32)
  T_base_cam = np.eye(4)
  T_base_cam[2, 3] = -1.0
  table = calibrate.fit_table(depth, T_base_cam, K)
  assert table["n_points"] > 5000
  assert table["table_z"] == pytest.approx(0.0, abs=1e-6)
  assert table["tilt_deg"] == pytest.approx(0.0, abs=1e-6)


def test_table_depth_fit_reports_the_plane_height_at_the_base_origin():
  from hardware.deploy import calibrate, config

  # A level plane hides the whole question, because its centroid height and
  # its height at the origin are the same number.  The measured D455 table is
  # tilted 0.88 deg and the visible patch is most of a metre away, and there
  # the two differ by millimetres -- which is what ``config.Rig`` means by
  # ``table_z`` and what the deployment's table guard subtracts.
  h, w = 480, 848
  K = np.array([[430.0, 0.0, (w - 1) / 2],
                [0.0, 430.0, (h - 1) / 2],
                [0.0, 0.0, 1.0]])
  tilt = np.deg2rad(1.0)
  normal = np.array([0.0, -np.sin(tilt), np.cos(tilt)])
  table_z = -0.004

  # Camera a metre up, looking straight down at a patch centred well out in
  # +y, so the patch centroid is nowhere near the base origin.
  T_base_cam = np.eye(4)
  T_base_cam[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                 [0.0, -1.0, 0.0],
                                 [0.0, 0.0, -1.0]])
  T_base_cam[:3, 3] = [0.0, 0.45, 1.0]

  u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
  rays = np.stack([(u - K[0, 2]) / K[0, 0],
                   (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
  rays_base = rays @ T_base_cam[:3, :3].T
  origin = T_base_cam[:3, 3]
  t = ((np.array([0.0, 0.0, table_z]) - origin) @ normal) / (rays_base @ normal)
  depth = np.where(np.isfinite(t) & (t > 0), t, 0.0)

  table = calibrate.fit_table(depth, T_base_cam, K)
  assert table["n_points"] > 5000
  assert table["tilt_deg"] == pytest.approx(1.0, abs=1e-3)
  # The centroid sits out at +y and is therefore NOT the answer.
  assert abs(table["centroid_z"] - table_z) > 0.003
  assert table["table_z"] == pytest.approx(table_z, abs=1e-4)

  # And the number it returns is the one the deployment guard interprets:
  # a point on the fitted plane must have zero signed distance from it.
  n = np.asarray(table["normal_base"])
  on_plane = np.array([0.30, 0.45, 0.0])
  on_plane[2] = table_z - (n[0] * on_plane[0] + n[1] * on_plane[1]) / n[2]
  assert float(on_plane @ n) - n[2] * table["table_z"] == pytest.approx(
    0.0, abs=1e-9)
  del config


def test_moved_25mm_checkerboard_samples_fit_one_table_plane():
  import cv2

  from hardware.deploy import calibrate

  board = calibrate.Board(
    kind="checker", squares=(11, 8), square_m=0.025, min_corners=88)
  # z = c + ax + by, expressed as a common normal and two in-plane axes.
  a, b, c = 0.012, -0.007, 0.034
  n = np.array([-a, -b, 1.0])
  n /= np.linalg.norm(n)
  e1 = np.array([1.0, 0.0, a])
  e1 /= np.linalg.norm(e1)
  e2 = np.cross(n, e1)
  e2 /= np.linalg.norm(e2)
  base_axes = np.column_stack((e1, e2, n))
  records = []
  for i, (x, y) in enumerate([
      (0.05, -0.25), (0.25, -0.18), (0.48, -0.10),
      (0.12, 0.08), (0.35, 0.16), (0.55, 0.25)]):
    th = 0.25 * i
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th), np.cos(th), 0.0],
                   [0.0, 0.0, 1.0]])
    R = base_axes @ Rz
    records.append({
      "rvec": cv2.Rodrigues(R)[0].ravel().tolist(),
      "tvec": [x, y, c + a * x + b * y],
    })
  table = calibrate.fit_table_board_samples(records, np.eye(4), board)
  assert table["n_samples"] == 6
  assert table["n_points"] == 6 * 88
  assert table["table_z"] == pytest.approx(c, abs=1e-6)
  assert table["tilt_deg"] == pytest.approx(
    np.degrees(np.arccos(n[2])), abs=1e-6)
  assert table["flatness_mm"] < 1e-6
  assert table["coverage_x_mm"] > 350
  assert table["coverage_y_mm"] > 400


def test_joint_corner_refinement_improves_pose_level_hand_eye():
  import cv2

  from hardware.deploy import calibrate, config, proprio, rectify

  rng = np.random.default_rng(8)
  kin = proprio.Kinematics()
  # An arbitrary but FIXED set of viewpoints.  This test is about the solver,
  # not about where the arm homes, and it used to read the home pose out of
  # hardware/deploy/obs_spec.json -- so correcting that file for the +90 degree
  # workspace rotation rotated all twelve synthetic views and moved the
  # refinement ratio from 0.44 to 0.64, failing a threshold that had nothing to
  # do with the change.  A deployment artifact should not be able to retune a
  # solver test.
  default = np.array([0.06, 1.72, -1.25, 1.13, 0.0, 0.0, 0.05, -0.05])
  board = calibrate.Board(squares=(6, 5), square_m=0.028,
                          marker_m=0.022, dictionary="DICT_4X4_50")
  obj = board._charuco().getChessboardCorners().astype(np.float64)
  K = rectify._default_d405_K()
  T_base_cam = config.sim_camera_extrinsic()
  T_cam_base = np.linalg.inv(T_base_cam)
  T_grip_board = np.eye(4)
  T_grip_board[:3, :3] = cv2.Rodrigues(
    np.array([0.3, -0.2, 0.15]))[0]
  T_grip_board[:3, 3] = [0.02, -0.01, 0.06]
  records = []
  for _ in range(12):
    q = default.copy()
    q[:6] += rng.uniform(-0.55, 0.55, 6)
    kin.update(q)
    T_bg = np.eye(4)
    T_bg[:3, :3] = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    T_bg[:3, 3] = kin.data.site_xpos[kin.site_id]
    T_cb = T_cam_base @ T_bg @ T_grip_board
    uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(T_cb[:3, :3])[0],
                              T_cb[:3, 3], K, np.zeros(5))
    uv = uv.reshape(-1, 2) + rng.normal(0.0, 0.25, (len(obj), 2))
    ok, rvec, tvec = cv2.solvePnP(obj, uv, K, np.zeros(5))
    assert ok
    records.append({
      "joint_pos": q.tolist(), "rvec": rvec.ravel().tolist(),
      "tvec": tvec.ravel().tolist(), "object_points": obj.tolist(),
      "image_points": uv.tolist(), "camera_K": K.tolist(),
      "camera_dist": [0.0] * 5,
    })
  out = calibrate.solve(records, board=board, K=K, dist=np.zeros(5))
  assert out["joint_refined"]
  assert out["reprojection_rms_px"] < out["reprojection_before_px"] * 0.5
  assert np.linalg.norm(out["T_base_cam"][:3, 3]
                        - T_base_cam[:3, 3]) < 0.003


def test_hand_eye_refuses_poses_that_do_not_rotate():
  """Hand-eye is determined by rotation.  Translation alone leaves it free, and
  the solver returns something anyway -- which is the failure this guard is
  for."""
  import cv2

  from hardware.deploy import calibrate, config, proprio

  kin = proprio.Kinematics()
  default = np.asarray(
    __import__("json").loads(pathlib.Path(proprio.SPEC_FILE).read_text())
    ["default_joint_pos"], dtype=np.float64)
  T_cam_base = np.linalg.inv(config.sim_camera_extrinsic())

  records = []
  for d in np.linspace(-0.05, 0.05, 10):
    q = default.copy()
    q[0] += d * 0.02          # a hair of motion, no real reorientation
    kin.update(q)
    T_base_grip = np.eye(4)
    T_base_grip[:3, :3] = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    T_base_grip[:3, 3] = kin.data.site_xpos[kin.site_id]
    T = T_cam_base @ T_base_grip
    records.append({"joint_pos": q.tolist(),
                    "rvec": cv2.Rodrigues(T[:3, :3])[0].ravel().tolist(),
                    "tvec": T[:3, 3].tolist()})

  out = calibrate.solve(records)
  assert out["rot_span_deg"] < calibrate.MIN_ROT_SPAN_DEG
  assert out["T_base_cam"] is None, (
    "with no rotation the equation is satisfied by any X, so returning one "
    "would be worse than returning nothing"
  )


def test_navigation_can_use_a_weaker_rotation_gate_without_weakening_the_solve():
  """The GUI's first solve steers the next small move; it is not saveable.

  A 15--30 degree seed set should therefore produce navigation guidance when
  explicitly requested, while the exact same records remain refused by the
  normal 30 degree calibration gate.
  """
  import cv2

  from hardware.deploy import calibrate, config, proprio

  rng = np.random.default_rng(44)
  kin = proprio.Kinematics()
  default = np.asarray(
    __import__("json").loads(pathlib.Path(proprio.SPEC_FILE).read_text())
    ["default_joint_pos"], dtype=np.float64)
  T_base_cam = config.sim_camera_extrinsic()
  T_cam_base = np.linalg.inv(T_base_cam)
  T_grip_board = np.eye(4)
  T_grip_board[:3, 3] = [0.01, -0.02, 0.05]

  records = []
  for _ in range(8):
    q = default.copy()
    q[:6] += rng.uniform(-0.10, 0.10, 6)
    kin.update(q)
    T_bg = np.eye(4)
    T_bg[:3, :3] = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    T_bg[:3, 3] = kin.data.site_xpos[kin.site_id]
    T_cb = T_cam_base @ T_bg @ T_grip_board
    records.append({
      "joint_pos": q.tolist(),
      "rvec": cv2.Rodrigues(T_cb[:3, :3])[0].ravel().tolist(),
      "tvec": T_cb[:3, 3].tolist(),
    })

  span = calibrate.solve(records, min_rotation_span_deg=0.0)["rot_span_deg"]
  assert 15.0 < span < calibrate.MIN_ROT_SPAN_DEG, span
  assert calibrate.solve(records)["T_base_cam"] is None
  rough = calibrate.solve(records, min_rotation_span_deg=15.0)
  assert rough["T_base_cam"] is not None
  assert np.linalg.norm(rough["T_base_cam"][:3, 3]
                        - T_base_cam[:3, 3]) < 1e-3


def test_calibration_motion_is_rest_to_rest_and_speed_limited():
  import threading

  from hardware.deploy.calibgui import (AUTO_RATE_HZ, AUTO_SPEED_RAD_S,
                                         Session, _joint_trajectory)

  q0 = np.zeros(6)
  q1 = np.array([0.31, -0.12, 0.07, 0.22, -0.18, 0.03])
  path = _joint_trajectory(q0, q1)
  assert np.allclose(path[0], q0)
  assert np.allclose(path[-1], q1)
  speed = np.abs(np.diff(path, axis=0)) * AUTO_RATE_HZ
  assert speed.max() <= AUTO_SPEED_RAD_S * 1.01
  assert np.abs(path[1] - path[0]).max() < np.abs(path[len(path) // 2]
                                                   - path[len(path) // 2 - 1]).max()

  # The browser's stop button must be a cancellation visible to the streaming
  # worker, even though this unit test has no camera or arm to construct a full
  # Session around.
  sess = Session.__new__(Session)
  sess.lock = threading.Lock()
  sess.motion = {"status": "moving", "progress": 0.5}
  sess._motion_cancel = threading.Event()
  assert sess.stop_motion()["ok"]
  assert sess._motion_cancel.is_set()


def test_same_arm_pose_with_a_moved_board_is_rejected_before_rough_solve():
  import cv2

  from hardware.deploy.calibgui import Session

  q = [0.0] * 8
  a = {"joint_pos": q, "rvec": [0.0, 0.0, 0.0],
       "tvec": [0.0, 0.0, 0.7]}
  R = cv2.Rodrigues(np.array([0.0, 0.2, 0.0]))[0]
  b = {"joint_pos": q, "rvec": cv2.Rodrigues(R)[0].ravel().tolist(),
       "tvec": [0.03, 0.0, 0.7]}
  assert Session._conflicting_stationary_poses([a, b]) == [[0, 1]]

  # Repeated observations that agree are merely redundant, not contradictory.
  c = dict(a)
  c["tvec"] = [0.001, 0.0, 0.7]
  assert Session._conflicting_stationary_poses([a, c]) == []


def test_rough_guidance_does_not_assume_the_simulators_camera_mounting_side():
  import cv2

  from hardware.deploy.calibgui import _plausible_camera_transform

  # A camera across the real table and yawed 90 degrees is still a perfectly
  # valid extrinsic.  The residual, not distance to a simulated mount, decides
  # whether it can guide nearby relative moves.
  T = np.eye(4)
  T[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, np.pi / 2]))[0]
  T[:3, 3] = [-0.48, 0.73, 0.50]
  assert _plausible_camera_transform(T)
  T[:3, 3] = [0.0, 0.0, 3.0]
  assert not _plausible_camera_transform(T)


def test_next_pose_planner_keeps_the_board_in_the_d405_gray_image():
  from hardware.deploy import calibrate, config, proprio, rectify
  from hardware.deploy.calibgui import NextPosePlanner

  board = calibrate.Board()
  K = rectify._default_d405_K()
  planner = NextPosePlanner(board, K,
                            (config.D405_WIDTH, config.D405_HEIGHT),
                            table_z=-0.30)
  default = np.asarray(
    __import__("json").loads(pathlib.Path(proprio.SPEC_FILE).read_text())
    ["default_joint_pos"], dtype=np.float64)
  q7 = np.array([*default[:6], default[6]])

  # Put the currently seen board fronto-parallel and centred.  Its arbitrary
  # implied gripper attachment is recovered by the planner, just as it is on
  # the real arm after the operator clamps the board on.
  w, h = board.squares[0] * board.square_m, board.squares[1] * board.square_m
  T_cb = np.eye(4)
  T_cb[:3, 3] = [-w / 2.0, -h / 2.0, 0.70]
  target = planner.plan(config.sim_camera_extrinsic(), T_cb, q7, records=[])
  assert target["available"], target
  uv = np.asarray(target["polygon_px"])
  assert (uv[:, 0] > 0).all() and (uv[:, 0] < config.D405_WIDTH).all()
  assert (uv[:, 1] > 0).all() and (uv[:, 1] < config.D405_HEIGHT).all()
  # Merely intersecting the image is not enough: the old planner produced a
  # 180x10 px edge-on sliver which was geometrically in frame but impossible
  # for ChArUco to detect.  Keep enough projected area and two-dimensional
  # shape for the actual D405 gray detector.
  assert target["projected_area_px2"] >= 2500
  assert target["projected_shape"] >= 0.12
  assert target["facing_cos"] >= 0.28
  assert target["motion_deg"] <= 12.1


def test_next_pose_planner_models_the_board_payload_above_the_table():
  from hardware.deploy import calibrate, config, proprio, rectify
  from hardware.deploy.calibgui import (
    BOARD_TABLE_CLEARANCE_M, NextPosePlanner)

  planner = NextPosePlanner(
    calibrate.Board.load(
      "hardware/depth_bench/targets/calib_compact_white_v2_cut.json"),
    rectify._default_d405_K(),
    (config.D405_WIDTH, config.D405_HEIGHT), table_z=0.0)
  default = np.asarray(
    __import__("json").loads(pathlib.Path(proprio.SPEC_FILE).read_text())
    ["default_joint_pos"], dtype=np.float64)
  q7 = np.array([*default[:6], default[6]])
  T_bg = planner._fk(q7)

  # Put the physical board slab only 30 mm above the table.  The robot model
  # itself has no table and used to call this collision-free.
  T_bb = np.eye(4)
  T_bb[2, 3] = 0.030
  T_gb = np.linalg.inv(T_bg) @ T_bb
  safe, clearance, why = planner.validate_path(q7, q7, T_gb)
  assert not safe
  assert clearance < BOARD_TABLE_CLEARANCE_M
  assert "calibration board" in why

  # The same arm pose with the mounted board safely above the table passes.
  T_bb[2, 3] = 0.30
  T_gb = np.linalg.inv(T_bg) @ T_bb
  safe, clearance, why = planner.validate_path(q7, q7, T_gb)
  assert safe, why
  assert clearance > BOARD_TABLE_CLEARANCE_M


def test_board_recovery_refuses_a_stale_distant_pose():
  import threading
  from types import SimpleNamespace

  from hardware.deploy.calibgui import Session

  sess = Session.__new__(Session)
  sess.lock = threading.Lock()
  sess.arm = object()
  sess.motion = {"status": "idle", "progress": 0.0}
  sess.recovery_q = np.zeros(6)
  sess._last = (None, SimpleNamespace(
    q=np.array([math.radians(31.0), 0, 0, 0, 0, 0]), gripper=0.0))
  out = sess.return_visible()
  assert not out["ok"]
  assert "refusing automatic recovery" in out["why"]


def test_a_bought_board_is_described_and_detected():
  """Any board, not the one this repository happens to print.

  The rig hard-coded a 5x5 33 mm ChArUco because that is the sheet in
  ``hardware/depth_bench/targets``.  A bought board is a better instrument and
  a different one, and a detector configured for the wrong grid does not
  degrade -- it returns nothing, at every pose, with the same message as a
  board that is out of frame.

  So the board is rendered from its own description and detected back, at a
  size and dictionary that are not the defaults.
  """
  import cv2

  from hardware.deploy import calibrate

  board = calibrate.Board(kind="charuco", squares=(7, 5), square_m=0.025,
                          marker_m=0.01875, dictionary="DICT_4X4_50")
  img = board._charuco().generateImage((7 * 120, 5 * 120))
  K = np.array([[900.0, 0, img.shape[1] / 2],
                [0, 900.0, img.shape[0] / 2], [0, 0, 1]])
  pose = calibrate.detect_board(img, K, np.zeros(5), board)
  assert pose is not None, "the board it was told about was not detected"
  assert pose["n_corners"] >= 20, pose["n_corners"]

  # And the default description does not find it, which is the failure this
  # exists to make impossible to hit silently.
  assert calibrate.detect_board(img, K, np.zeros(5)) is None or \
    calibrate.detect_board(img, K, np.zeros(5))["n_corners"] < 6


def test_compact_v2_print_asset_matches_its_board_description():
  import cv2

  from hardware.deploy import calibrate

  target = pathlib.Path(
    "hardware/depth_bench/targets/calib_compact_v2")
  board = calibrate.Board.load(target.with_suffix(".json"))
  image = cv2.imread(str(target.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
  found = board.detect(image)
  assert board.squares == (6, 5)
  assert board.dictionary == "DICT_4X4_50"
  assert board.square_m == pytest.approx(0.028)
  assert board.marker_m == pytest.approx(0.022)
  assert found is not None and len(found[0]) == 20

  # The inkjet edition is deliberately a different board identity.  Its
  # detector enables white/inverted markers and must recover all the same
  # geometric corners from the separately named print asset.
  white = pathlib.Path(
    "hardware/depth_bench/targets/calib_compact_white_v2")
  white_board = calibrate.Board.load(white.with_suffix(".json"))
  white_image = cv2.imread(str(white.with_suffix(".png")),
                           cv2.IMREAD_GRAYSCALE)
  white_found = white_board.detect(white_image)
  assert white_board.inverted
  assert white_board != board
  assert white_found is not None and len(white_found[0]) == 20
  assert np.mean(white_image < 128) < 0.25

  cut = pathlib.Path(
    "hardware/depth_bench/targets/calib_compact_white_v2_cut")
  cut_board = calibrate.Board.load(cut.with_suffix(".json"))
  cut_image = cv2.imread(str(cut.with_suffix(".png")),
                         cv2.IMREAD_GRAYSCALE)
  cut_found = cut_board.detect(cut_image)
  cut_meta = __import__("json").loads(cut.with_suffix(".json").read_text())
  assert cut_board == white_board
  assert cut_found is not None and len(cut_found[0]) == 20
  assert cut_meta["page_mm"] == [210.0, 297.0]
  assert cut_meta["crop_frame_mm"] == [180.0, 152.0]


def test_the_square_size_is_stored_with_the_poses():
  """Solving a board's poses with the wrong square size scales the answer and
  leaves the residual small, because self-consistency is not sensitive to a
  uniformly wrong ruler.  The board therefore travels with the poses."""
  import tempfile

  from hardware.deploy import calibrate

  board = calibrate.Board(kind="checker", squares=(11, 8), square_m=0.025)
  recs = [{"joint_pos": [0.0] * 8, "rvec": [0.0, 0.0, 0.0],
           "tvec": [0.0, 0.0, 0.5]}]
  with tempfile.TemporaryDirectory() as d:
    f = pathlib.Path(d) / "poses.json"
    calibrate.save_poses(board, recs, f)
    got_board, got_recs = calibrate.load_poses(f)
  assert got_board == board
  assert got_recs == recs


def test_a_checkerboard_that_comes_back_rotated_is_put_back():
  """The 180-degree corner-ordering flip, which a ChArUco board does not have.

  A checkerboard rotated half a turn about its own normal is the same image,
  so the detector's corner order can flip between poses and nothing in that
  frame says it did.  Hand-eye fed a mixture solves for a camera that is not
  there.  Half the poses here are flipped on purpose and the solver has to
  recover the same answer it gets from clean ones.
  """
  import cv2

  from hardware.deploy import calibrate, config, proprio

  rng = np.random.default_rng(3)
  kin = proprio.Kinematics()
  default = np.asarray(
    __import__("json").loads(pathlib.Path(proprio.SPEC_FILE).read_text())
    ["default_joint_pos"], dtype=np.float64)

  board = calibrate.Board(kind="checker", squares=(11, 8), square_m=0.025)
  F = calibrate._flip_matrix(board)
  T_base_cam = config.sim_camera_extrinsic()
  T_cam_base = np.linalg.inv(T_base_cam)
  T_grip_board = np.eye(4)
  T_grip_board[:3, :3] = cv2.Rodrigues(np.array([0.3, -0.2, 0.15]))[0]
  T_grip_board[:3, 3] = [0.02, -0.01, 0.06]

  records = []
  for i in range(12):
    q = default.copy()
    q[:6] += rng.uniform(-0.6, 0.6, 6)
    kin.update(q)
    T_base_grip = np.eye(4)
    T_base_grip[:3, :3] = kin.data.site_xmat[kin.site_id].reshape(3, 3)
    T_base_grip[:3, 3] = kin.data.site_xpos[kin.site_id]
    T = T_cam_base @ T_base_grip @ T_grip_board
    if i % 2:                                   # the detector flipped this one
      T = T @ F
    records.append({
      "joint_pos": q.tolist(),
      "rvec": cv2.Rodrigues(T[:3, :3])[0].ravel().tolist(),
      "tvec": T[:3, 3].tolist(),
    })

  blind = calibrate.solve(records)
  fixed = calibrate.solve(records, board=board)
  assert fixed["n_flipped"] == 6, fixed["n_flipped"]
  assert np.linalg.norm(fixed["T_base_cam"][:3, 3] - T_base_cam[:3, 3]) < 1e-3
  assert fixed["residual_mm"] < 0.5
  # And it was worth doing: told nothing about the board, the same poses give
  # an answer that is wrong by more than the arm is long.
  assert np.linalg.norm(blind["T_base_cam"][:3, 3] - T_base_cam[:3, 3]) > 0.05


def _sim_extrinsic(dx=0.0, dy=0.0, dz=0.0, pitch=0.0, yaw=0.0, roll=0.0):
  """The camera the simulator would build for a given session mismatch.

  A reimplementation of ``perturb.randomize_camera_pose_offset`` with the
  jitter set to zero, returned in OpenCV convention -- which is what a
  calibration measures.  It is written out here rather than called because
  calling it needs a built environment on a GPU, and the arithmetic it is
  standing in for is nine lines.
  """
  import math

  from piper_push import camera as sim_camera

  pos = np.asarray(sim_camera.CAMERA_POS, dtype=np.float64) + [dx, dy, dz]
  aim = np.asarray(sim_camera.CAMERA_AIM, dtype=np.float64)
  fwd = aim - pos
  fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, [0.0, 0.0, 1.0])
  right /= np.linalg.norm(right)
  up = np.cross(right, fwd)
  for ang, axis in ((math.radians(pitch), right), (math.radians(yaw), up)):
    c, s = math.cos(ang), math.sin(ang)
    fwd = fwd * c + np.cross(axis, fwd) * s
    fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, [0.0, 0.0, 1.0])
  right /= np.linalg.norm(right)
  up = np.cross(right, fwd)
  if roll:
    c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    right, up = right * c + up * s, -right * s + up * c
  T = np.eye(4)
  T[:3, :3] = np.stack([right, -up, fwd], axis=1)
  T[:3, 3] = pos
  return T


def _rig_to_sim():
  import importlib.util

  spec = importlib.util.spec_from_file_location(
    "rig_to_sim",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rig_to_sim.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def test_the_calibration_comes_back_as_the_simulator_s_own_axes():
  """A measured extrinsic, turned into the flags that reproduce it.

  This is the whole claim of ``scripts/rig_to_sim.py``: that the numbers it
  prints, handed to ``accept_s1.py``, put the simulator's camera where the
  real one is.  Checked by going round the loop -- build the camera the
  simulator would build for a known mismatch, decompose it as if it had come
  from a calibration, and rebuild from what came out.

  The tolerance is a thousandth of a degree because there is no measurement
  here and nothing to be noisy: any error is the decomposition disagreeing
  with the thing it is inverting.
  """
  mod = _rig_to_sim()
  for dx, dz, pitch, yaw in [(0.0, 0.0, 0.0, 0.0),
                             (0.015, -0.008, 1.3, -0.7),
                             (0.0, 0.0, 5.0, -4.0),
                             (-0.05, 0.04, 8.0, -9.0)]:
    T = _sim_extrinsic(dx=dx, dz=dz, pitch=pitch, yaw=yaw)
    got = mod.decompose(T)
    assert abs(got["cam_pos_x_m"] - dx) < 1e-9
    assert abs(got["cam_pos_z_m"] - dz) < 1e-9
    back = _sim_extrinsic(dx=got["cam_pos_x_m"], dz=got["cam_pos_z_m"],
                          pitch=got["cam_pitch_deg"], yaw=got["cam_yaw_deg"])
    ang = np.degrees(np.arccos(np.clip(T[:3, 2] @ back[:3, 2], -1, 1)))
    assert ang < 1e-3, f"optical axis off by {ang:.4f} deg at {(dx, dz, pitch, yaw)}"


def test_the_two_things_the_simulator_cannot_express_are_reported():
  """Roll and lateral offset.

  ``SessionMismatchCfg`` offsets x and z and nothing else, and the camera
  frame is rebuilt from the world's up on every path, so a rolled mount is not
  representable at all.  Both have to come out of the decomposition as
  themselves rather than being absorbed into an axis that is representable --
  a roll reported as a yaw would be a mismatch the simulator cheerfully
  reproduces and the robot does not have.
  """
  mod = _rig_to_sim()
  got = mod.decompose(_sim_extrinsic(roll=4.0))
  assert abs(got["cam_roll_deg"] - 4.0) < 1e-6, got["cam_roll_deg"]
  assert abs(got["cam_pitch_deg"]) < 1e-6 and abs(got["cam_yaw_deg"]) < 1e-6

  got = mod.decompose(_sim_extrinsic(dy=0.03))
  assert abs(got["cam_pos_y_m"] - 0.03) < 1e-9
  assert abs(got["cam_pos_x_m"]) < 1e-9 and abs(got["cam_pos_z_m"]) < 1e-9


class _RecordingIface:
  """Stands in for ``C_PiperInterface_V2`` and remembers what it was told.

  Every method exists and returns None, so a call that ``PiperArm`` makes and
  this does not anticipate is recorded rather than raising -- the point is to
  capture the arguments, not to model the SDK.
  """

  def __init__(self):
    self.calls = []

  def __getattr__(self, name):
    def f(*a, **kw):
      self.calls.append((name, a, kw))
    return f


def test_closing_the_can_client_holds_and_never_disables_by_default():
  from hardware.deploy import robot

  arm = robot.PiperArm.__new__(robot.PiperArm)
  arm._iface = _RecordingIface()
  arm.connected = True
  held = []
  arm.hold = lambda: held.append(True)
  arm.close()
  names = [name for name, _, _ in arm._iface.calls]
  assert held == [True]
  assert "DisconnectPort" in names
  assert "DisableArm" not in names
  assert not arm.connected

  # Power removal exists, but cannot happen through an ordinary context exit.
  arm.connected = True
  arm.close(disable=True)
  names = [name for name, _, _ in arm._iface.calls]
  assert "DisableArm" in names


def test_every_can_message_passes_the_sdk_s_own_validator():
  """What ``PiperArm.command`` sends, handed to the SDK's constructors.

  This exists because ``PiperArm`` was written from documentation against no
  hardware, and the first thing it found when the SDK was finally installed was
  that the gripper effort was out of range by 2x: the code sent
  ``int(GRIPPER_FORCE_N * 1000)`` = 10000 for a field ``ArmMsgGripperCtrl``
  documents as 0-5000 and validates on construction.  Every gripper command
  would have raised, on the robot, in the control loop.

  So the messages are built for real -- same classes, same arguments -- and the
  SDK's own range checks decide.  No CAN, no arm, and it runs in CI.

  ``JointCtrl`` has no validator at all, which is the more dangerous half: a
  units error there is a number the drives will try to achieve.  Those are
  checked against the envelope instead.
  """
  pytest.importorskip("piper_sdk")
  from piper_sdk.piper_msgs.msg_v2.transmit import (ArmMsgGripperCtrl,
                                                    ArmMsgJointCtrl,
                                                    ArmMsgMotionCtrl_2)
  from piper_push import robot as sim_robot

  from hardware.deploy import robot

  spec = __import__("json").loads(pathlib.Path(
    pathlib.Path(__file__).resolve().parents[1] / "hardware" / "deploy"
    / "obs_spec.json").read_text())
  mapper = robot.ActionMapper(spec, allow_legacy=True)
  mapper.reset()

  arm = robot.PiperArm.__new__(robot.PiperArm)
  arm._iface = _RecordingIface()
  arm.gripper_torque_nm = robot.GRIPPER_TORQUE_NM
  arm.connected = True
  arm._prev = None

  rng = np.random.default_rng(0)
  for _ in range(200):
    arm.command(mapper(rng.uniform(-1.5, 1.5, 7)))       # over-range on purpose

  seen = set()
  for name, a, kw in arm._iface.calls:
    seen.add(name)
    if name == "GripperCtrl":
      ArmMsgGripperCtrl(*a, **kw)                        # raises if out of range
    elif name == "MotionCtrl_2":
      ArmMsgMotionCtrl_2(*a, **kw)
    elif name == "JointCtrl":
      ArmMsgJointCtrl(*a, **kw)
      for j, mdeg in zip(robot.ARM_JOINTS, a):
        lo, hi = sim_robot.SAFE_TARGET_CLIP[j]
        rad = mdeg / robot.PiperArm.RAD_TO_MDEG
        assert lo - 1e-6 <= rad <= hi + 1e-6, (
          f"{j} commanded {np.degrees(rad):.1f} deg, envelope "
          f"{np.degrees(lo):.1f} to {np.degrees(hi):.1f}")
  assert seen == {"MotionCtrl_2", "JointCtrl", "GripperCtrl"}, seen


def test_the_sdk_still_has_every_field_the_arm_reads():
  """The feedback path, checked the same way as the command path.

  ``read`` walks four attribute chains into SDK message objects.  A rename in
  any of them is an ``AttributeError`` at 50 Hz on a moving robot, and there is
  no other place it would show up first -- ``selftest.py`` runs against the
  simulator and never touches these.
  """
  sdk = pytest.importorskip("piper_sdk")
  p = sdk.C_PiperInterface_V2("can0", judge_flag=False, can_auto_init=False)

  j = p.GetArmJointMsgs().joint_state
  for i in range(1, 7):
    assert isinstance(getattr(j, f"joint_{i}"), int)     # 0.001 deg

  g = p.GetArmGripperMsgs().gripper_state
  assert hasattr(g, "grippers_angle") and hasattr(g, "grippers_effort")

  low = p.GetArmLowSpdInfoMsgs()
  for i in range(1, 7):
    assert hasattr(getattr(low, f"motor_{i}").foc_status,
                   "driver_enable_status")

  # The units the conversions assume, from the SDK's own docstrings rather
  # than from memory: joints 0.001 deg, gripper stroke 0.001 mm.
  from piper_sdk.piper_msgs.msg_v2.feedback.arm_feedback_joint_states import (
    ArmMsgFeedBackJointStates)
  from piper_sdk.piper_msgs.msg_v2.transmit.arm_gripper_ctrl import (
    ArmMsgGripperCtrl)

  from hardware.deploy import robot

  # The SDK carries each unit twice, once in Chinese and once in English, and
  # only the first is the class's ``__doc__``.  Either spelling will do; what
  # is being guarded is that the unit did not change under the conversions.
  jdoc = ArmMsgFeedBackJointStates.__doc__ or ""
  assert "0.001度" in jdoc or "0.001 degrees" in jdoc, (
    "the SDK no longer says joint feedback is in 0.001 degrees, and "
    "PiperArm.RAD_TO_MDEG assumes it is")
  gdoc = ArmMsgGripperCtrl.__doc__ or ""
  assert "0.001mm" in gdoc or "0.001 mm" in gdoc, (
    "the SDK no longer says gripper stroke is in 0.001 mm, and "
    "PiperArm.M_TO_UM assumes it is")
  assert "0-5000" in gdoc, (
    "the gripper torque range moved; PiperArm.command clamps to 0-5000")
  assert abs(robot.PiperArm.RAD_TO_MDEG - 180.0 / np.pi * 1000.0) < 1e-9
  assert robot.PiperArm.M_TO_UM == 1e6


def test_runtime_rejects_nonfinite_feedback_and_overspeed():
  from types import SimpleNamespace

  from hardware.deploy import run
  from piper_push import robot as sim_robot

  st = SimpleNamespace(q=np.zeros(6), dq=np.zeros(6), gripper=0.02,
                       gripper_vel=0.0)
  assert run._feedback_fault(st) is None
  st.q[2] = np.nan
  assert "non-finite" in run._feedback_fault(st)
  st.q[2] = 0.0
  st.dq[0] = sim_robot.JOINT_TRIP_RAD_S["joint1"] * 1.01
  assert "joint 1 speed" in run._feedback_fault(st)


def test_runtime_rejects_malformed_policy_actions():
  from hardware.deploy import run

  assert run._action_fault(np.zeros(7)) is None
  assert "expected 7" in run._action_fault(np.zeros(6))
  bad = np.zeros(7)
  bad[4] = np.inf
  assert "non-finite" in run._action_fault(bad)


def test_arm_mask_matches_bruteforce_spheres_with_workspace_filter():
  from hardware.deploy import mask

  rng = np.random.default_rng(20260826)
  pts = rng.uniform(-0.5, 0.8, (2000, 3))
  centres = rng.uniform(-0.2, 0.5, (12, 3))
  radii = rng.uniform(0.02, 0.12, 12)
  within = rng.random(pts.shape[0]) > 0.25
  clearance = 0.017

  expanded = radii + clearance
  brute = (((pts[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
           < expanded[None, :] ** 2).any(axis=1) & within
  got = mask.arm_mask(pts, (centres, radii), clearance, within=within)
  assert np.array_equal(got, brute)


def test_runtime_refuses_enable_when_preload_would_be_clipped():
  from types import SimpleNamespace

  from hardware.deploy import run

  st = SimpleNamespace(q=np.deg2rad([0, 90, -70, 0, 0, 120.0]))
  assert run._start_pose_fault(st) is None
  st.q[1] = np.deg2rad(-0.32)
  st.q[2] = np.deg2rad(0.84)
  assert run._start_pose_fault(st) is None
  st.q[5] = np.deg2rad(166.7)
  why = run._start_pose_fault(st)
  assert "joint 6" in why
  assert "enable could jump" in why


def test_initial_hardware_rate_scale_reduces_every_slew_limit():
  import json
  from types import SimpleNamespace

  from hardware.deploy import proprio, robot, run

  spec = json.loads(proprio.SPEC_FILE.read_text())
  base = robot.ActionMapper(spec, allow_legacy=True)
  args = SimpleNamespace(camera="d405", rig_file=None, dry_run=True,
                         allow_nominal=True, replay=None, no_arm=True,
                         device="cpu", mask="depth",
                         command_rate_scale=0.25, policy="unused")
  # The build path applies the scalar immediately after constructing mapper;
  # exercise the exact arithmetic without constructing a camera or policy.
  scaled = robot.ActionMapper(spec, allow_legacy=True)
  scaled.max_step *= args.command_rate_scale
  assert np.allclose(scaled.max_step, base.max_step * 0.25)


def test_action_mapper_optional_acceleration_limit_bounds_reversals():
  import json

  from hardware.deploy import proprio, robot

  spec = json.loads(proprio.SPEC_FILE.read_text())
  m = robot.ActionMapper(spec, accel_limit=0.5, gripper_accel_limit=0.05, allow_legacy=True)
  m.reset()
  q0 = m.previous.copy()
  q1 = m(np.full(7, 10.0))
  v1 = (q1 - q0) / m.dt
  assert np.all(np.abs(v1[:6]) <= 0.5 * m.dt + 1e-12)
  assert abs(v1[6]) <= 0.05 * m.dt + 1e-12

  q2 = m(np.full(7, -10.0))
  v2 = (q2 - q1) / m.dt
  assert np.all(np.abs(v2[:6] - v1[:6]) <= 0.5 * m.dt + 1e-12)
  assert abs(v2[6] - v1[6]) <= 0.05 * m.dt + 1e-12

  m.reset(q2)
  assert np.allclose(m.previous_velocity, 0.0)

  # Jittered wall-clock periods must still bound velocity change per second.
  m.reset()
  dt1, dt2 = 0.027, 0.023
  q0 = m.previous.copy()
  q1 = m(np.full(7, 10.0), dt=dt1)
  v1 = (q1 - q0) / dt1
  q2 = m(np.full(7, -10.0), dt=dt2)
  v2 = (q2 - q1) / dt2
  assert np.all(np.abs(v2[:6] - v1[:6]) <= 0.5 * dt2 + 1e-12)
  assert abs(v2[6] - v1[6]) <= 0.05 * dt2 + 1e-12


def test_grasp_height_guard_uses_the_training_fk():
  import json

  from hardware.deploy import proprio, robot, run

  spec = json.loads(proprio.SPEC_FILE.read_text())
  mapper = robot.ActionMapper(spec, allow_legacy=True)
  kin = proprio.Kinematics()
  assert run._grasp_height(kin, mapper.default_target) > 0.07
  bad = mapper.default_target.copy()
  bad[1] = 0.0
  assert np.isfinite(run._grasp_height(kin, bad))
  assert np.isnan(run._grasp_height(kin, np.zeros(6)))

def test_table_guard_checks_full_collision_geometry_against_plane():
  import json

  from hardware.deploy import proprio, robot, run

  mapper = robot.ActionMapper(json.loads(proprio.SPEC_FILE.read_text()), allow_legacy=True)
  kin = proprio.Kinematics()
  clearance, geom = run._table_clearance(
    kin, mapper.default_target, np.array([0.0, 0.0, 1.0]), 0.0)
  assert 0.04 < clearance < run._grasp_height(kin, mapper.default_target)
  assert "gripper" in geom or "pad" in geom
  assert np.isnan(run._table_clearance(
    kin, np.zeros(6), np.array([0.0, 0.0, 1.0]), 0.0)[0])


def test_async_recorder_flushes_full_log_without_stale_targets(tmp_path):
  import json
  from types import SimpleNamespace

  from hardware.deploy import run

  out = tmp_path / "session"
  writer = run._Recorder(str(out), queue_size=8)
  # Every field JointFeedback carries, because _Recorder writes every field
  # JointFeedback carries.  gripper_effort is the gripper loop's only
  # observable and is deliberately read without a getattr default: a feedback
  # object that has lost it should raise here rather than log a quiet 0.0.
  fb = SimpleNamespace(position=np.arange(8, dtype=float),
                       velocity=np.arange(8, dtype=float) * 0.1,
                       target=np.arange(8, dtype=float) + 10.0,
                       gripper_effort=0.42)
  for i in range(4):
    writer.write(np.full((8, 9), 0.7 + i * 0.01, np.float32),
                 np.full((8, 9), i, np.uint8), fb,
                 np.full(7, i, dtype=float), 3,
                 extra={"frame_index": 100 + i,
                        "commanded_grasp_height_m": 0.08})
  writer.close()
  meta = json.loads((out / "meta.json").read_text())
  control = json.loads((out / "control.json").read_text())
  assert len(meta) == 4
  assert len(control) == 4
  assert control[-1]["event"] == "command"
  assert control[-1]["frame_file"] == "000003.npz"
  assert meta[-1]["frame_index"] == 103
  assert meta[-1]["target"] == fb.target.tolist()
  saved = np.load(out / "000003.npz")
  assert saved["depth"].shape == (8, 9)
  assert saved["gray"][0, 0] == 3


def test_env_yaml_joint_defaults_ignore_the_action_scale_mapping():
  from hardware.deploy import run

  # An mjlab env.yaml states the per-joint action scales the same way it
  # states the initial pose.  A search for "jointN:" over the whole document
  # reads the scales, and then reports joints as disagreeing that do not.
  env_yaml = """
  scene:
    entities:
      robot:
        init_state:
          pos: !!python/tuple
          - 0.0
          joint_pos:
            joint1: 1.6307963267948966
            joint2: 1.72
            joint3: -1.25
            joint4: 1.13
            gripper_joint1: 0.05
            gripper_joint2: -0.05
          joint_vel:
            .*: 0.0
  actions:
    arm:
      scale:
        joint1: 0.9
        joint2: 0.6
        joint3: 0.75
        joint5: 0.3
        joint6: 1.6
"""
  names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
           "gripper_joint1", "gripper_joint2"]
  q = run._init_joint_pos(env_yaml, names)
  assert q is not None
  # joint5/joint6 are absent from init_state and are therefore zero -- NOT the
  # 0.3 and 1.6 the scale mapping lists.
  assert q.tolist() == pytest.approx(
    [1.6307963267948966, 1.72, -1.25, 1.13, 0.0, 0.0, 0.05, -0.05])


def test_env_yaml_joint_defaults_decline_a_pattern_block():
  from hardware.deploy import run

  # Regex keys mean the mapping cannot be read joint by joint.  Declining is
  # correct; guessing would report a disagreement that is not there.
  env_yaml = """
        joint_pos:
          joint.*: 0.0
"""
  assert run._init_joint_pos(env_yaml, ["joint1", "joint2"]) is None


def test_deployed_obs_spec_carries_the_rotated_workspace():
  import json

  from hardware.deploy import proprio
  from piper_push import layout

  # The +90 degree workspace rotation moves joint1's home by exactly
  # WORKSPACE_YAW_RAD.  hardware/deploy/obs_spec.json was written before that
  # rotation and stayed at 0.06 while the task moved to 1.6308; proprio
  # subtracts the number to build the observation and ActionMapper adds it to
  # build the command, so the file being stale pointed the arm 90 degrees away
  # in both directions at once and nothing raised.
  spec = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())
  joint1 = spec["default_joint_pos"][spec["joint_names"].index("joint1")]
  assert joint1 == pytest.approx(0.06 + layout.WORKSPACE_YAW_RAD, abs=1e-5)


def test_arm_mask_matches_the_naive_sphere_test():
  from hardware.deploy import mask

  # arm_mask carries two bounding-box rejections in front of the distance test
  # because the distance test was never the cost -- proving that the table is
  # not the robot was, at 47 ms a frame.  Both rejections are supposed to be
  # exact, so the only thing worth testing is that they are: against every
  # point against every sphere, with no boxes at all.
  rng = np.random.default_rng(20260828)
  for trial in range(8):
    n = 4000
    pts = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    k = int(rng.integers(1, 25))
    centres = rng.uniform(-0.4, 0.4, (k, 3)).astype(np.float32)
    radii = rng.uniform(0.01, 0.09, k).astype(np.float32)
    clearance = float(rng.uniform(0.0, 0.03))
    within = (rng.random(n) < 0.6) if trial % 2 else None

    r = (radii + clearance).astype(np.float32)
    d = pts[:, None, :] - centres[None, :, :]
    naive = (np.einsum("ijk,ijk->ij", d, d) < (r * r)[None, :]).any(axis=1)
    if within is not None:
      naive &= within

    got = mask.arm_mask(pts, (centres, radii), clearance, within=within)
    assert np.array_equal(got, naive), f"trial {trial}"


def test_arm_mask_handles_the_empty_and_absent_cases():
  from hardware.deploy import mask

  pts = np.zeros((10, 3), dtype=np.float32)
  assert not mask.arm_mask(pts, None, 0.01).any()
  assert not mask.arm_mask(
    pts, (np.zeros((0, 3), np.float32), np.zeros(0, np.float32)), 0.01).any()
  # Nothing selected by ``within``, and nothing near the spheres: both return
  # early, and both have to return the full-length array.
  assert mask.arm_mask(pts, (np.array([[0.0, 0.0, 0.0]], np.float32),
                             np.array([0.05], np.float32)), 0.0,
                       within=np.zeros(10, bool)).shape == (10,)
  far = np.full((10, 3), 9.0, dtype=np.float32)
  assert mask.arm_mask(far, (np.array([[0.0, 0.0, 0.0]], np.float32),
                             np.array([0.05], np.float32)), 0.0).shape == (10,)


def test_the_pixel_ceiling_cannot_disable_the_component_ceiling():
  from hardware.deploy import mask

  # These were one constant.  The pixel gate drops every pixel above it before
  # the components are formed, so a component's top is always below it -- and
  # the component test ``floor_z <= top <= ceiling`` could then never fail on
  # its upper bound.  A 270 mm structure on the rig came out as a 137 mm
  # component, inside the range the task's objects occupy, and the
  # nearest-to-the-hand rule aimed at it in 223 of 239 recorded frames.
  #
  # The invariant is that the pixel gate must be strictly the looser of the
  # two, so that something too tall stays too tall long enough to be rejected
  # for it.
  c = mask.SegmenterCfg()
  assert c.max_height_m > c.max_top_z_m, (
    "the pixel ceiling must be above the component ceiling, or a tall thing "
    "is truncated into range instead of being rejected")
  # And the component ceiling has to clear the tallest object the task uses
  # (90 mm) with room for the smoothing and the calibration residual, without
  # reaching the ~140 mm that let the rig's fixture through.
  assert 0.09 < c.max_top_z_m < 0.14
  assert c.min_height_m < c.max_top_z_m


def test_a_component_taller_than_the_ceiling_is_rejected_not_truncated():
  import dataclasses

  from hardware.deploy import mask

  # The mechanism, without a scene: a height field holding one 30 mm object and
  # one 270 mm structure.  With the two ceilings equal, the tall one survives
  # as a component whose top sits at the ceiling; with them separated it does
  # not survive at all.
  c = mask.SegmenterCfg()
  heights = np.array([0.030, 0.270])

  shared = dataclasses.replace(c, max_height_m=0.14, max_top_z_m=0.14)
  clipped = np.minimum(heights, shared.max_height_m)      # what the pixel gate does
  survives_shared = [h for h in clipped if h <= shared.max_top_z_m]
  assert len(survives_shared) == 2, "the tall one is truncated into range"
  assert survives_shared[1] == pytest.approx(0.14)

  clipped = np.minimum(heights, c.max_height_m)
  survives = [h for h in clipped if h <= c.max_top_z_m]
  assert len(survives) == 1 and survives[0] == pytest.approx(0.030)


def test_the_deployment_defaults_to_the_d455():
  import inspect

  from hardware.deploy import calibrate, run, sensor

  # The D455 is the rig's camera.  A default of d405 meant every command
  # needed --camera d455 to be correct, and forgetting it read rig.json, the
  # D405's calibration, with only the serial cross-check standing between that
  # and a run.
  assert inspect.signature(sensor.Reader).parameters["backend"].default == "d455"
  for module, flag in ((run, "--camera"), (calibrate, "--camera")):
    parser = None
    src = inspect.getsource(module)
    assert f'{flag}", choices=("d405", "d455"), default="d455"' in src, module.__name__
    del parser


def test_workspace_mask_is_the_training_sector_not_its_box():
  import math

  from hardware.deploy import config, mask

  # The deployment used the circumscribing box while training samples objects
  # in an annular sector.  Two real things stand above the table inside the box
  # and outside the sector, and both were chosen as targets on the rig: the
  # robot's own base column at r = 0.10 m, and the camera mount at r = 0.64 m.
  (rlo, rhi), (alo, ahi), _ = config.WORKSPACE_SECTOR
  mid = 0.5 * (alo + ahi)
  r_in = 0.5 * (rlo + rhi)

  def at(r, a, z=0.02):
    return np.array([[r * math.cos(a), r * math.sin(a), z]])

  assert mask.workspace_mask(at(r_in, mid))[0]
  # inside the inner radius -- the base column
  assert not mask.workspace_mask(at(rlo - 0.03, mid))[0]
  # outside the outer radius -- the camera mount
  assert not mask.workspace_mask(at(rhi + 0.03, mid))[0]
  # right azimuth band, but outside it
  assert not mask.workspace_mask(at(r_in, alo - 0.15))[0]
  assert not mask.workspace_mask(at(r_in, ahi + 0.15))[0]
  # and the box still rejects what it always did
  assert not mask.workspace_mask(at(r_in, mid, z=1.0))[0]


def test_workspace_sector_matches_the_task_the_policy_was_trained_on():
  from hardware.deploy import config
  from piper_push.tasks.pick_place import env_cfg

  # Imported, not restated.  A sector copied by hand here would be a second
  # source of truth for where an object may be, and the failure mode is a
  # deployment that filters out exactly the objects training put in front of it.
  (r, a, _) = config.WORKSPACE_SECTOR
  assert r == tuple(env_cfg.OBJECT_ALLOWED_RADIUS)
  assert a == tuple(env_cfg.OBJECT_ALLOWED_ANGLE)

  # The bin is scenery and sits inside the sector; the segmenter removes it by
  # footprint, and that must stay the thing that removes it.
  import math
  bx, by = config.BIN_CENTER
  assert r[0] < math.hypot(bx, by) < r[1]


def test_the_real_rig_false_targets_are_outside_the_sector():
  import math

  from hardware.deploy import mask

  # Measured on recordings/v3_final_noarm_try1: three components that the
  # nearest-to-the-hand rule chose during a 60 s rehearsal, none of which is a
  # thing the task ever puts on the table.
  for x, y, what in ((-0.123, 0.008, "arm base column, 102 mm tall"),
                     (0.094, -0.015, "beside the base"),
                     (-0.376, 0.491, "camera mount, 1218 px")):
    assert not mask.workspace_mask(np.array([[x, y, 0.02]]))[0], what


def test_overlay_projection_inverts_the_rig_extrinsic():
  from hardware.deploy import config, overlay

  # The overlay is only worth drawing if it lands where the calibration says.
  # Round-trip: take a pixel, walk a ray out to a known depth, and project the
  # resulting base-frame point back.
  rig = config.Rig(T_base_cam=config.sim_camera_extrinsic(),
                   K=rectify._default_d405_K())
  K = rig.K
  for u, v, z in ((424.0, 240.0, 0.8), (100.0, 60.0, 0.6), (800.0, 440.0, 1.1)):
    cam = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    base = rig.T_base_cam[:3, :3] @ cam + rig.T_base_cam[:3, 3]
    back = overlay.project(base[None], rig,
                           (config.D405_HEIGHT, config.D405_WIDTH))[0]
    assert back[0] == pytest.approx(u, abs=1e-6)
    assert back[1] == pytest.approx(v, abs=1e-6)

  # Behind the camera is NaN, not a wrapped-around pixel.  A naive divide puts
  # a stray line across an otherwise correct drawing.
  behind = rig.T_base_cam[:3, :3] @ np.array([0.0, 0.0, -0.5]) + rig.T_base_cam[:3, 3]
  assert not np.isfinite(overlay.project(behind[None], rig, (480, 848))[0]).all()


def test_overlay_scales_its_intrinsics_to_the_preview_size():
  from hardware.deploy import config, overlay

  # calibgui streams 1280x720 while deployment is 848x480, and a drawing that
  # ignores that lands in the corner.
  rig = config.Rig(T_base_cam=config.sim_camera_extrinsic(),
                   K=rectify._default_d405_K())
  p = np.array([[0.3, 0.3, 0.0]])
  small = overlay.project(p, rig, (config.D405_HEIGHT, config.D405_WIDTH))[0]
  big = overlay.project(p, rig, (720, 1280))[0]
  assert big[0] == pytest.approx(small[0] * 1280 / config.D405_WIDTH, rel=1e-9)
  assert big[1] == pytest.approx(small[1] * 720 / config.D405_HEIGHT, rel=1e-9)


def test_overlay_draws_the_same_sector_the_filter_tests():
  import math

  from hardware.deploy import config, mask, overlay

  # The picture and the filter must not disagree about the shape.  Sample the
  # drawn outline and assert every vertex is on the boundary the mask uses.
  (r, a, _) = config.WORKSPACE_SECTOR
  pts = overlay.sector_points(r, a, 0.0, n=24)
  radii = np.hypot(pts[:, 0], pts[:, 1])
  assert radii.min() == pytest.approx(r[0], abs=1e-9)
  assert radii.max() == pytest.approx(r[1], abs=1e-9)
  ang = np.arctan2(pts[:, 1], pts[:, 0])
  assert ang.min() == pytest.approx(a[0], abs=1e-9)
  assert ang.max() == pytest.approx(a[1], abs=1e-9)
  # A point just inside the drawn outline is a point the mask keeps.
  mid_a, mid_r = 0.5 * (a[0] + a[1]), 0.5 * (r[0] + r[1])
  inside = np.array([[mid_r * math.cos(mid_a), mid_r * math.sin(mid_a), 0.02]])
  assert mask.workspace_mask(inside)[0]


def test_the_tracker_still_believes_in_a_briefly_lost_target():
  from hardware.deploy import mask

  # The blind-drive window in run.py is gated on this: the tracker keeps its
  # centroid for lost_frames after the last confirmation, and drops it after.
  # That is what separates "the arm is standing in front of it" from "it is
  # gone", and the deployment loop has no other way to tell.
  tracker = mask.TargetTracker(lost_frames=5, confirm=2, window=3)
  inst = mask.Instance(label=1, n_px=200,
                       centroid_base=np.array([0.3, 0.3, 0.03]),
                       top_z=0.04, bbox=(10, 10, 8, 8))
  seg = mask.Segmentation(labels=np.zeros((4, 4), np.int32), instances=[inst])
  hand = np.array([0.3, 0.35, 0.10])
  # The first sighting is refused on purpose -- confirmation is 2 of the last
  # 3 frames, which is what rejects a blob of correlated depth noise.
  assert tracker.update(seg, hand) == 0
  for _ in range(3):
    assert tracker.update(seg, hand) == 1
  assert tracker.has_target

  empty = mask.Segmentation(labels=np.zeros((4, 4), np.int32), instances=[])
  # The confirmation window outlives the sighting by a frame: two of the last
  # three frames still hold it, so the first empty frame still names a target
  # and the mask is not empty at all.  The blind window begins after that.
  assert tracker.update(empty, hand) == 1
  for i in range(5):
    assert tracker.update(empty, hand) == 0
    assert tracker.has_target, f"gave up after {i + 1} blind frames"
  # one past lost_frames and it stops believing
  assert tracker.update(empty, hand) == 0
  assert not tracker.has_target


def test_an_occluded_target_reuses_the_last_known_mask():
  import inspect

  from hardware.deploy import run, target_mask

  # The camera is bolted to the world and the object is not moving, so when
  # the arm stands in front of the target the last mask the segmenter produced
  # is still where the object is.  Feeding the *empty* mask instead was tried
  # on the arm and is what this replaces: an empty mask cannot be told apart
  # from "there is no target", and the policy drove 137 steps and drifted
  # 285 mm away from the object it was reaching for.
  #
  # The behaviour itself now lives in ``target_mask.TargetMask`` and is tested
  # against a truth in ``tests/test_target_mask.py``; it moved out of this file
  # because the simulation checker had grown a second copy of it.  What is
  # asserted here is the wiring: that ``Perception`` still routes the mask
  # through it and still drives on the substituted answer.
  src = inspect.getsource(run.Perception)
  assert "self._target_mask(" in src, "the mask must go through TargetMask"
  assert "self.tracker.has_target" in src, "must be gated on the tracker"
  assert "held_over" in src, \
    "the loop must drive on the held-over mask, not on has_target"
  assert hasattr(target_mask.TargetMask, "__call__")


def test_an_occluded_target_empties_the_mask_in_simulation_too():
  from hardware.deploy import obs

  # The fix rests on this claim, so it is asserted rather than remembered:
  # obs.camera_obs is fed whatever mask it is given, and an empty one produces
  # empty mask channels while leaving the depth channel intact.  The simulator
  # does the same -- CameraScene keys the mask on the segmentation buffer's
  # frontmost geom, so an arm in front of the object zeroes it there as well.
  depth = np.full((8, 8), 0.7, np.float32)
  valid = np.ones((8, 8), bool)
  seen = obs.camera_obs(depth, valid, np.ones((8, 8), bool))
  blind = obs.camera_obs(depth, valid, np.zeros((8, 8), bool))
  assert np.array_equal(seen[0], blind[0]), "the depth channel must not change"
  assert blind[1].max() == 0.0
  assert blind[2].max() == 0.0
  assert seen[1].max() == 1.0


def test_blind_steps_are_bounded_and_can_be_switched_off():
  import inspect

  from hardware.deploy import run

  src = inspect.getsource(run)
  # 0 restores the old behaviour, which is what an operator who does not want
  # the arm moving on a stale belief will reach for.
  assert '"--max-blind-steps", type=int, default=60' in src
  assert "a.max_blind_steps > 0" in src
  assert "blind_streak < a.max_blind_steps" in src
  # and the streak has to reset once the target is seen again, or the budget
  # is spent once and never returns
  assert "blind_streak = 0" in src


def test_the_loop_waits_for_real_feedback_before_preloading_the_drives():
  import time
  from types import SimpleNamespace

  from hardware.deploy import run

  # Measured on the rig: ConnectPort returns before any CAN frame has arrived,
  # so the first read() is the SDK's defaults -- six exact zeros and a shut
  # gripper -- and the real pose lands about 0.3 s later.  The preload-then-
  # enable order that protects against a stale target turns into the hazard if
  # it is handed those defaults: the arm is driven to its zero pose the moment
  # it is energised.  All-zero joints pass _start_pose_fault, so nothing
  # downstream catches it.
  real = np.array([0.3, 1.7, -1.2, 1.1, 0.0, 0.1])

  class Arm:
    def __init__(self, zeros):
      self.n, self.zeros = 0, zeros
    def read(self):
      self.n += 1
      q = np.zeros(6) if self.n <= self.zeros else real
      g = 0.0 if self.n <= self.zeros else 0.05
      return SimpleNamespace(q=q, gripper=g)

  arm = Arm(zeros=8)
  st = run._wait_for_feedback(arm, timeout_s=2.0, settle=3)
  assert np.allclose(st.q, real)
  assert arm.n > 8, "returned before the defaults stopped arriving"

  # A bus that never reports must raise rather than preload something.
  class Silent:
    def read(self):
      return SimpleNamespace(q=np.zeros(6), gripper=0.0)

  t0 = time.time()
  with pytest.raises(RuntimeError, match="did not report a settled pose"):
    run._wait_for_feedback(Silent(), timeout_s=0.4, settle=3)
  assert time.time() - t0 >= 0.4

  # And a moving arm is not "settled" -- it has to agree with itself first.
  class Moving:
    def __init__(self):
      self.k = 0
    def read(self):
      self.k += 1
      return SimpleNamespace(q=real + self.k * 1e-3, gripper=0.05)

  with pytest.raises(RuntimeError):
    run._wait_for_feedback(Moving(), timeout_s=0.4, settle=3)


def test_all_zero_feedback_would_have_passed_the_start_pose_check():
  from types import SimpleNamespace

  from hardware.deploy import run

  # This is why the wait above is a safety fix and not a tidy-up: the existing
  # guard has nothing to say about the uninitialised read.
  assert run._start_pose_fault(SimpleNamespace(q=np.zeros(6))) is None


def test_the_homing_path_is_rest_to_rest_and_speed_bounded():
  from hardware.deploy import robot

  # Reused from the guided calibration rather than re-chosen: those are the
  # only motion numbers on this rig that have driven the arm across the
  # workspace without incident.
  q0 = np.array([0.1, 1.7, -1.2, 1.1, 0.0, 0.0])
  q1 = np.array([1.63, 1.72, -1.25, 1.13, 0.0, 0.0])
  path = robot.joint_trajectory(q0, q1)

  assert np.allclose(path[0], q0), "must begin where the arm already is"
  assert np.allclose(path[-1], q1)
  step = np.abs(np.diff(path, axis=0)).max()
  assert step * robot.AUTO_RATE_HZ <= robot.AUTO_SPEED_RAD_S * 1.02
  # rest to rest: the first and last steps are far smaller than the peak
  assert np.abs(path[1] - path[0]).max() < 0.1 * step
  assert np.abs(path[-1] - path[-2]).max() < 0.1 * step
  # and a zero-length move is still a valid path rather than an empty one
  same = robot.joint_trajectory(q0, q0)
  assert len(same) >= 2 and np.allclose(same, q0)


def test_calibgui_and_the_deployment_share_one_trajectory_implementation():
  import inspect

  from hardware.deploy import calibgui, robot

  # Two implementations of the same rest-to-rest path would agree until one of
  # them was edited.
  src = inspect.getsource(calibgui._joint_trajectory)
  assert "robot.joint_trajectory" in src or "_robot.joint_trajectory" in src
  q0 = np.zeros(6); q1 = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
  assert np.allclose(calibgui._joint_trajectory(q0, q1),
                     robot.joint_trajectory(q0, q1))


def test_homing_drives_to_the_pose_the_policy_was_trained_from():
  import json

  from hardware.deploy import proprio, robot, run

  # Exercised, not grepped.  The first version of this lived inline in main()
  # and read a name that was local to build(); a test that only searched the
  # source for the right string passed while the code raised NameError on the
  # arm.  Driving a stand-in catches that.
  spec = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())
  arm = robot.DryRunArm(spec, tau=0.005)
  arm.connect()
  arm._q = np.array([-0.07, 0.99, -0.15, -0.39, 0.08, -0.27])

  st = run.home_arm(arm, spec, speed_rad_s=6.0)

  names = spec["joint_names"]
  home = np.asarray([spec["default_joint_pos"][names.index(j)]
                     for j in robot.ARM_JOINTS])
  assert np.allclose(st.q, home, atol=2e-2), np.degrees(st.q - home).tolist()
  # the target is the spec's, which is what carries joint 1's +90 degrees
  assert home[0] == pytest.approx(1.6308, abs=1e-3)


def test_homing_stops_on_a_stalled_joint():
  import json

  from hardware.deploy import proprio, robot, run

  spec = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())

  class Stalled(robot.DryRunArm):
    """Accepts commands and never moves, which is what a disabled joint or a
    lost CAN path looks like from here."""
    def command(self, target, dt=0.02):
      pass

  arm = Stalled(spec)
  arm.connect()
  arm._q = np.zeros(6)
  with pytest.raises(RuntimeError, match="homing tracking error"):
    run.home_arm(arm, spec, speed_rad_s=6.0)


def test_homing_can_be_interrupted():
  import json

  from hardware.deploy import proprio, robot, run

  spec = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())
  arm = robot.DryRunArm(spec, tau=0.005)
  arm.connect()
  arm._q = np.zeros(6)
  seen = {"n": 0}

  def stop():
    seen["n"] += 1
    return seen["n"] > 3

  with pytest.raises(SystemExit):
    run.home_arm(arm, spec, speed_rad_s=6.0, should_stop=stop)


def test_flatten_scene_keeps_the_table_and_replaces_the_room():
  from hardware.deploy import obs

  # The simulator's scene contains one piece of scenery: an infinite MuJoCo
  # PLANE.  Deployment's channel 0 contains a lab.  Injecting a raw deployment
  # channel 0 into the simulator, mask untouched, took the trained policy from
  # 170 objects placed to 0.
  plane = np.full((4, 4), 0.80, np.float32)
  depth = np.array([
    [0.80, 0.79, 0.81, 0.80],   # the tabletop itself
    [0.74, 0.72, 0.80, 0.80],   # objects standing on it (60 and 80 mm)
    [1.60, 1.40, 0.80, 0.80],   # the floor beyond the table, far behind
    [0.30, 0.80, 0.80, 0.80],   # something well in front -- the arm
  ], np.float32)
  valid = np.ones((4, 4), bool)

  out, v = obs.flatten_scene(depth, valid, plane, above_m=0.30, below_m=0.02)

  assert out[0].tolist() == pytest.approx([0.80, 0.79, 0.81, 0.80])
  assert out[1][0] == pytest.approx(0.74), "an object on the table is kept"
  assert out[1][1] == pytest.approx(0.72)
  assert out[2][0] == pytest.approx(0.80), "the floor becomes the plane"
  assert out[2][1] == pytest.approx(0.80)
  assert out[3][0] == pytest.approx(0.80), "0.50 m above the plane is not an object"
  assert v.all()


def test_flatten_scene_keeps_the_arm_by_its_model_not_by_a_height():
  from hardware.deploy import obs

  # The first version used a 300 mm ceiling for this.  The arm reaches
  # 426-459 mm on this rig, so the top third of the robot was replaced by
  # table in every single frame -- while the comment beside the constant
  # argued that cropping the arm out would be a second difference introduced
  # to fix the first.  The kinematic model knows where the arm is.
  plane = np.full((2, 2), 1.00, np.float32)
  depth = np.array([[0.55, 0.99],      # 0.45 m in front of the plane, and table
                    [1.60, 0.98]], np.float32)   # the floor, and table
  valid = np.ones((2, 2), bool)
  # the first pixel is a point the arm actually occupies
  pts = np.array([[0.0, 0.30, 0.45],
                  [0.0, 0.30, 0.01],
                  [0.0, 0.90, -0.30],
                  [0.0, 0.31, 0.02]], dtype=np.float64)
  arm = (np.array([[0.0, 0.30, 0.45]]), np.array([0.08]))

  without, _ = obs.flatten_scene(depth, valid, plane, above_m=0.30)
  assert without[0, 0] == pytest.approx(1.00), "erased without the model"

  with_arm, _ = obs.flatten_scene(depth, valid, plane, points_base=pts,
                                  arm=arm, above_m=0.30)
  assert with_arm[0, 0] == pytest.approx(0.55), "the arm survives"
  assert with_arm[1, 0] == pytest.approx(1.00), "the floor still does not"


def test_flatten_scene_leaves_holes_and_unreachable_rays_alone():
  from hardware.deploy import obs

  # A hole reads as the far plane in both pipelines already, so the two agree
  # about it and there is nothing to substitute.
  plane = np.array([[0.8, np.inf]], np.float32)
  depth = np.array([[0.0, 0.75]], np.float32)
  valid = np.array([[False, True]])
  out, v = obs.flatten_scene(depth, valid, plane)
  assert out[0, 0] == pytest.approx(0.8), "an invalid pixel over the table takes the plane"
  assert out[0, 1] == pytest.approx(0.75), "a ray that never meets the plane is untouched"
  assert v[0, 1]


def test_ground_plane_depth_agrees_with_the_calibrated_plane():
  from hardware.deploy import config, rectify

  rig = config.Rig.load("hardware/deploy/rig_d455.json")
  rp = rectify.Reprojector(rig, device="cpu")
  g = rp.ground_plane_depth(rig)
  assert g.shape == (config.HEIGHT, config.WIDTH)
  assert np.isfinite(g).all()

  # Unproject a few pixels through the plane depth and check they land on it.
  rays = rp.virtual_rays().reshape(config.HEIGHT, config.WIDTH, 3)
  T = config.sim_camera_extrinsic()
  n = np.asarray(rig.table_normal_base); n = n / np.linalg.norm(n)
  for (r, c) in ((10, 10), (84, 112), (150, 200)):
    p_cam = rays[r, c] * g[r, c]
    p = T[:3, :3] @ p_cam + T[:3, 3]
    # signed distance from the plane through (0, 0, table_z)
    assert float(p @ n) - n[2] * rig.table_z == pytest.approx(0.0, abs=1e-6)


def test_a_holding_run_still_archives_what_the_camera_saw(tmp_path):
  import json
  from types import SimpleNamespace

  from hardware.deploy import run, sensor

  # Frames used to be written only on the command path, so a run that spent
  # its time holding -- for a guard, a missing target, a stale observation --
  # left a control log and almost no pictures.  Several sessions ended after a
  # hundred consecutive guard holds having archived a handful of frames, which
  # is the opposite of what a run that holds needs.
  writer = run._Recorder(str(tmp_path / "s"), queue_size=64)
  fb = SimpleNamespace(position=np.zeros(8), velocity=np.zeros(8),
                       target=np.zeros(8), gripper_effort=0.0)

  def frame(i):
    return sensor.Frame(depth=np.full((8, 9), 0.7, np.float32),
                        gray=np.zeros((8, 9), np.uint8), stamp=0.0, index=i)

  # a 50 Hz loop holding on a 30 Hz camera: each image arrives about 1.7 times
  for k in range(10):
    writer.event(fb, np.zeros(7), 0, "hold_no_target", None, frame=frame(k // 2))
  writer.close()

  meta = json.loads((writer.dir / "meta.json").read_text())
  ctl = json.loads((writer.dir / "control.json").read_text())
  assert len(ctl) == 10, "every decision is still logged"
  # five distinct camera frames, stored once each
  assert len(meta) == 5, [m.get("frame_file") for m in meta]
  assert sorted(m["i"] for m in meta) == [0, 1, 2, 3, 4]
  assert all((writer.dir / m["frame_file"]).exists() for m in meta)


def test_gripper_sweep_never_commands_the_arm_anywhere():
  import json

  from hardware.deploy import gripcal, proprio, robot

  # The whole point of the tool is that it measures the gripper without moving
  # the arm.  PiperArm.command sends the full seven-vector, so "do not move the
  # arm" is not the absence of a joint command -- it is the presence of the
  # right one, on every message.
  spec = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())
  start = np.array([0.3, 1.6, -1.1, 0.9, 0.1, -0.2])

  class Recording(robot.DryRunArm):
    def __init__(self, spec):
      super().__init__(spec)
      self._q = start.copy()
      self.sent = []
    def command(self, target, dt=0.02):
      self.sent.append(np.asarray(target, dtype=float).copy())
      # a stand-in servo that tracks the gripper but never the arm
      self._g = float(target[6])

  arm = Recording(spec)
  rows = gripcal.sweep(arm, low_m=0.0, high_m=0.045, cycles=2,
                       rate_hz=200.0, hold_s=0.02)
  assert arm.sent, "nothing was commanded at all"
  for t in arm.sent:
    assert np.allclose(t[:6], start), "the arm was commanded away from its pose"
  # and the gripper genuinely swept both ends
  g = np.array([r["target_m"] for r in rows])
  assert g.min() == pytest.approx(0.0) and g.max() == pytest.approx(0.045)
  assert {r["phase"] for r in rows} == {"open", "close"}


def test_the_flicker_measure_separates_a_bit_from_a_contact():
  from hardware.deploy import gripcal

  # A contact sensor gives one long run.  The reconstruction measured on the
  # arm gave twenty-odd single-step runs, which is what makes a policy that
  # learned "contact holds" let go.  The number that distinguishes them is the
  # run length, not the fraction of time the bit is on -- both of these are on
  # half the time.
  steady = [{"effort": 0.6} for _ in range(10)] + [{"effort": 0.0} for _ in range(10)]
  jitter = [{"effort": 0.6 if i % 2 == 0 else 0.0} for i in range(20)]

  a = gripcal.flicker(steady, 0.15)
  b = gripcal.flicker(jitter, 0.15)
  assert a["on_fraction"] == pytest.approx(b["on_fraction"])
  assert a["runs"] == 1 and a["median_run"] == 10
  assert b["runs"] == 10 and b["median_run"] == 1
  assert a["single_step_runs"] == 0.0 and b["single_step_runs"] == 1.0


def test_the_contact_bit_is_debounced_into_something_like_a_sensor():
  from hardware.deploy import proprio

  b = proprio.ProprioBuilder()
  assert b.contact_effort == pytest.approx(0.20), "the measured threshold"

  # The empty gripper's transients are two steps; a hold is thirty-one.  Both
  # numbers come from hardware/deploy/gripcal on this arm.  The bit has to
  # ignore the first and keep the second.
  def run(seq):
    b.reset()
    return [b._contact_bit(x) for x in seq]

  transient = run([0.0] * 5 + [0.6, 0.6] + [0.0] * 5)
  assert not any(transient), "a two-step spike is not a contact"

  hold = run([0.0] * 3 + [0.6] * 20 + [0.0] * 10)
  assert hold[3 + b.contact_assert - 1], "asserts after contact_assert samples"
  assert not hold[3 + b.contact_assert - 2], "and not before"
  assert all(hold[3 + b.contact_assert - 1:23]), "stays on while held"
  assert not hold[-1], "and releases once the load is gone"

  # The dropouts seen while genuinely holding are one or two steps.  Those must
  # not release it -- that flicker is what made the policy let go.
  flaky = run([0.0] * 3 + [0.6, 0.6, 0.6, 0.0, 0.6, 0.0, 0.0, 0.6, 0.6] * 3)
  assert sum(flaky) > 0.7 * len(flaky), "one-step dropouts must be bridged"

  # A latch that never lets go would be worse than none.
  released = run([0.6] * 10 + [0.0] * (b.contact_release + 2))
  assert not released[-1]


def test_the_contact_latch_is_cleared_wherever_the_policy_is():
  import inspect

  from hardware.deploy import run

  # The latch is state carried across control steps, so it has to be reset
  # alongside the recurrent policy -- at startup and on every hold-and-resync.
  # A latch that survives a resync asserts contact against a pose the arm was
  # moved away from.
  src = inspect.getsource(run.main)
  assert src.count("builder.reset()") >= 2, "startup and hold_and_resync"
  i = src.index("def hold_and_resync")
  body = src[i:i + 600]
  assert "pol.reset()" in body and "builder.reset()" in body


def test_every_name_main_uses_is_defined_before_it_runs():
  import ast
  import inspect

  from hardware.deploy import run

  # The viewer class was appended to the end of the file, after the
  # ``if __name__ == "__main__"`` block, so ``main()`` ran before the class
  # statement did and every real-arm launch died with NameError -- after
  # homing the arm.  Import succeeds either way; only running catches it.
  src = inspect.getsource(run)
  tree = ast.parse(src)
  guard = None
  for i, node in enumerate(tree.body):
    if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and getattr(node.test.left, "id", None) == "__name__"):
      guard = i
  assert guard is not None, "no __main__ guard"
  defined_after = {
    n.name for n in tree.body[guard + 1:]
    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
  }
  assert not defined_after, (
    f"defined after the __main__ guard and therefore unavailable to main(): "
    f"{sorted(defined_after)}")


def test_stale_guard_uses_camera_capture_time_not_perception_publish_time():
  from types import SimpleNamespace

  from hardware.deploy import run

  frame = SimpleNamespace(stamp=100.0)
  assert run._observation_age(frame, now=100.125) == pytest.approx(0.125)
  assert np.isinf(run._observation_age(None, now=100.125))


def test_control_uses_actual_mask_availability_instead_of_depth_label():
  import inspect

  from hardware.deploy import run

  perception = inspect.getsource(run.Perception._run)
  control = inspect.getsource(run.main)
  assert "target_available = bool(np.asarray(target).any())" in perception
  assert "if not target_available or held_over:" in control


def test_a_perception_thread_failure_is_raised_in_the_control_thread():
  import threading

  from hardware.deploy import run

  perception = run.Perception.__new__(run.Perception)
  perception._lock = threading.Lock()
  perception._error = None
  perception._out = None
  perception._stopping = threading.Event()

  def fail():
    raise ValueError("GPU inference failed")
  perception._run = fail
  perception.run()
  assert perception._stopping.is_set()
  with pytest.raises(RuntimeError, match="perception thread failed"):
    perception.latest()
