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
