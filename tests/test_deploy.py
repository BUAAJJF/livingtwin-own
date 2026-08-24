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

  spec = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())
  m = robot.ActionMapper(spec)
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

  spec = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())
  m = robot.ActionMapper(spec)
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
