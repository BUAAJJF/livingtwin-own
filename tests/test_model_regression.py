"""Pin the facts S0 measured, so a later model edit cannot quietly undo them.

Every assertion here corresponds to something that was measured once and then
built on.  They are cheap -- spec compilation and arithmetic, no GPU, no
simulation -- because a check that is expensive to run is a check that stops
being run.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import mjlab.tasks  # noqa: F401  -- import mjlab before us; the entry point loads us back
import mujoco
import numpy as np
import pytest

from piper_push import objects
from piper_push import robot as piper


@pytest.fixture(scope="module")
def pick_model() -> mujoco.MjModel:
  return piper.get_pick_spec().compile()


def _geom(m: mujoco.MjModel, name: str) -> int:
  gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
  assert gid >= 0, f"no geom named {name}"
  return gid


# ---------------------------------------------------------------------------
# Gripper geometry
# ---------------------------------------------------------------------------


def test_pad_contact_face_is_the_large_face(pick_model):
  """29.6 x 23.0 mm, with the 3 mm half-thickness along the closing direction.

  The obvious reading of the pad's size tuple gives 29.6 x 6 mm, which is the
  edge, not the face.  Getting this wrong makes every grasp-envelope number
  wrong by a factor of four in contact area.
  """
  m = pick_model
  d = mujoco.MjData(m)
  pads = [_geom(m, n) for n in ("lf_pad", "rf_pad")]
  jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint1")
  jid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint2")
  d.qpos[:] = m.qpos0
  d.qpos[m.jnt_qposadr[jid]] = 0.02
  d.qpos[m.jnt_qposadr[jid2]] = -0.02
  mujoco.mj_kinematics(m, d)

  sep = d.geom_xpos[pads[0]] - d.geom_xpos[pads[1]]
  axis = sep / np.linalg.norm(sep)
  rot = d.geom_xmat[pads[0]].reshape(3, 3)
  local = int(np.argmax(np.abs(rot.T @ axis)))
  half = m.geom_size[pads[0]]
  face = sorted(2 * half[i] for i in range(3) if i != local)

  assert local == 2, "the closing direction should be the pad's local z"
  assert half[local] == pytest.approx(0.003), "3 mm half-thickness"
  assert face == pytest.approx([0.023, 0.0296], abs=1e-6)


def test_maximum_jaw_gap_is_100mm(pick_model):
  m = pick_model
  d = mujoco.MjData(m)
  pads = [_geom(m, n) for n in ("lf_pad", "rf_pad")]
  jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint1")
  jid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint2")
  d.qpos[:] = m.qpos0
  d.qpos[m.jnt_qposadr[jid]] = piper.GRIPPER_OPEN_M
  d.qpos[m.jnt_qposadr[jid2]] = -piper.GRIPPER_OPEN_M
  mujoco.mj_kinematics(m, d)
  centres = np.linalg.norm(d.geom_xpos[pads[0]] - d.geom_xpos[pads[1]])
  gap = centres - m.geom_size[pads[0]][2] - m.geom_size[pads[1]][2]
  assert gap == pytest.approx(0.100, abs=1e-4)


def test_finger_mesh_does_not_reach_past_the_pad(pick_model):
  """If it did, the finger would touch the object before the pad and every
  friction and force number below would describe the wrong contact."""
  m = pick_model
  d = mujoco.MjData(m)
  jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint1")
  jid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint2")
  d.qpos[:] = m.qpos0
  d.qpos[m.jnt_qposadr[jid]] = 0.02
  d.qpos[m.jnt_qposadr[jid2]] = -0.02
  mujoco.mj_kinematics(m, d)

  def world_bounds(gid: int) -> tuple[np.ndarray, np.ndarray]:
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
      i = m.geom_dataid[gid]
      v = m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]]
    else:
      h = m.geom_size[gid]
      v = np.array(
        [[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
      )
    w = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
    return w.min(0), w.max(0)

  # lf_pad sits on -y, so "inward" for its finger is +y.
  pad_lo, pad_hi = world_bounds(_geom(m, "lf_pad"))
  mesh_lo, mesh_hi = world_bounds(_geom(m, "gripper_link1_collision"))
  assert mesh_hi[1] <= pad_hi[1] + 1e-6


# ---------------------------------------------------------------------------
# Grip force
# ---------------------------------------------------------------------------


def test_gripper_reaches_rated_force_on_the_smallest_object():
  """A position servo produces kp x (target - actual).  With the target at zero
  and the jaws held apart by the object, that error is the object's half width,
  so kp has to be big enough that the smallest object in the distribution still
  saturates the drive's rated force.  The URDF-inherited kp of 40 gives 0.5 N
  and drops everything."""
  smallest_half_width = objects.OBJECT_HALF_EXTENT_RANGE[0][0]
  force = piper.GRIPPER_STIFFNESS * smallest_half_width
  assert force >= piper.GRIPPER_FORCE_N


def test_heaviest_object_is_holdable_at_the_lowest_pad_friction():
  """S0 measured the drop boundary at mu ~ mass in kg (0.30 held 300 g, 0.40
  held 400 g, 0.25 dropped 300 g)."""
  heaviest = objects.OBJECT_MASS_RANGE[1]
  lowest_pad_mu = 0.55  # env_cfg's pad_friction range, lower end
  assert lowest_pad_mu > heaviest * 1.2, "no margin over the measured boundary"


# ---------------------------------------------------------------------------
# Contact parameter ownership
# ---------------------------------------------------------------------------


def test_pads_outrank_the_object_which_outranks_the_ground():
  """MuJoCo mixes contact parameters by elementwise max when priorities are
  equal, which silently masks BOTH friction knobs at once -- measured in S0,
  where pad mu of 0.30, 0.15 and 0.08 all produced identical slip.  The
  ordering below is what separates them: the pads own object-pad (so pad
  friction is the grasp knob) and the object owns object-table (so its own
  friction is the sliding knob)."""
  assert piper.PAD_PRIORITY > objects.OBJECT_PRIORITY > 0


def test_pad_condim_survives_the_priority_ordering():
  """Torsional friction is what stops a grasped object spinning in the jaws;
  it comes from the pads' condim 6, and only if the pads win the mix.

  Read off the CollisionCfg rather than the compiled robot spec: condim and
  priority are applied when the entity is assembled into a scene, so the
  standalone spec still carries MuJoCo's defaults.
  """
  cfg = piper.PICK_COLLISION
  assert cfg.condim["[lr]f_pad"] == 6
  assert cfg.priority["[lr]f_pad"] == piper.PAD_PRIORITY
  assert cfg.priority[".*_collision"] < objects.OBJECT_PRIORITY


# ---------------------------------------------------------------------------
# Object distribution stays inside the measured envelope
# ---------------------------------------------------------------------------


def test_object_distribution_is_inside_the_grasp_envelope():
  """S0: width 20-50 mm across the jaws, height >= 24 mm, mass <= 600 g."""
  (wx, wy, hz) = objects.OBJECT_HALF_EXTENT_RANGE
  assert 2 * wx[0] >= 0.020 and 2 * wx[1] <= 0.050
  assert 2 * wy[0] >= 0.020 and 2 * wy[1] <= 0.050
  assert 2 * hz[0] >= 0.024
  assert objects.OBJECT_MASS_RANGE[1] <= 0.600


def test_widest_object_still_fits_the_jaws():
  widest = 2 * max(objects.OBJECT_HALF_EXTENT_RANGE[0][1], objects.OBJECT_HALF_EXTENT_RANGE[1][1])
  assert widest < 2 * piper.GRIPPER_OPEN_M


# ---------------------------------------------------------------------------
# Scene layout
# ---------------------------------------------------------------------------


def test_bin_is_reachable_and_outside_the_spawn_sector():
  from piper_push.tasks.pick_place import env_cfg as pick_cfg

  bx, by = objects.BIN_CENTER
  radius = float(np.hypot(bx, by))
  angle = float(np.arctan2(by, bx))
  lo, hi = pick_cfg.SPAWN_ANGLE
  # S0: a straight-down pose is usable for r in [0.16, 0.52] m.
  assert 0.16 <= radius <= 0.52
  assert angle < lo or angle > hi, "the bin sits inside the object spawn sector"


def test_release_height_is_inside_the_straight_down_envelope():
  """S0: straight down runs out at 150 mm for r <= 0.35 and 130 mm at 0.42."""
  from piper_push.tasks.pick_place import mdp as pick_mdp

  cfg = pick_mdp.PickCommandCfg(resampling_time_range=(1.0, 1.0))
  release_z = cfg.bin_rim_z + cfg.release_clearance_m
  radius = float(np.hypot(*cfg.bin_center))
  ceiling = 0.150 if radius <= 0.35 else 0.130
  assert release_z <= ceiling


def test_bin_walls_are_thick_enough_not_to_tunnel():
  """A 400 g object at transport speed crosses ~2 mm per physics substep at
  5 ms; walls thinner than that let objects leave without going over the rim."""
  assert objects.BIN_WALL_THICKNESS >= 0.006


# ---------------------------------------------------------------------------
# Command path
# ---------------------------------------------------------------------------


def test_command_rate_stays_under_the_safety_shell():
  """The position servo overshoots a constant-velocity ramp by about 11%, and
  nothing in this simulation plays the part of the drive's inner velocity loop.
  A derate of 0.9 measured an actual peak of 1.00x the trip on an ordinary
  transport move; 0.75 measured 0.897."""
  assert piper.COMMAND_DERATE * 1.15 < 1.0
  for joint, trip in piper.JOINT_TRIP_RAD_S.items():
    assert piper.COMMAND_RATE_LIMIT_RAD_S[joint] < trip


def test_action_space_covers_the_postures_the_task_needs():
  """Reference postures solved by IK over the spawn sector and the bin
  (scratchpad s0/jrange.py).  The pushing task's home and scale express 3% of
  them, which is a design error no amount of training can fix."""
  home = piper.PICK_HOME_KEYFRAME.joint_pos
  offsets = np.array([home.get(f"joint{i + 1}", 0.0) for i in range(6)])
  scale = np.array(
    [
      piper.PICK_ARM_SCALE[k]
      for k in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    ]
  )
  reference = np.array(
    [
      [0.000, 1.671, -1.380, 1.317, 0.000, 0.000],  # over an object, 125 mm up
      [0.000, 1.784, -1.142, 0.979, 0.000, 0.000],  # at the grasp
      [0.000, 1.658, -1.452, 1.396, 0.000, 0.000],  # lifted
      [-0.631, 1.789, -1.506, 1.369, 0.003, -0.718],  # over the bin
      [-0.648, 1.735, -1.521, 1.384, -0.034, -0.728],  # released
      [0.762, 1.251, -0.684, 0.856, 0.074, 0.000],  # far corner of the sector
      [-0.646, 2.185, -1.817, 1.398, -0.057, 0.000],  # near corner
    ]
  )
  inside = np.abs(reference - offsets) <= scale + 1e-9
  assert inside.all(), f"unreachable joints: {np.argwhere(~inside)}"


def test_target_clip_respects_both_the_urdf_and_the_manual(pick_model):
  """joint5's manual limit (75 deg) is tighter than the URDF's, joint4's and
  joint6's URDF stops are tighter than the manual's; the clip takes whichever
  binds."""
  m = pick_model
  for name, (lo, hi) in piper.SAFE_TARGET_CLIP.items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    urdf_lo, urdf_hi = m.jnt_range[jid]
    assert lo >= urdf_lo - 1e-6 and hi <= urdf_hi + 1e-6
  assert piper.SAFE_TARGET_CLIP["joint5"][1] == pytest.approx(1.3090, abs=1e-4)


# ---------------------------------------------------------------------------
# Robot profiles
# ---------------------------------------------------------------------------


def test_gravity_compensation_is_set_on_the_spec_not_the_model(pick_model):
  """Writing body_gravcomp on a compiled model does nothing: qfrc_gravcomp
  stays zero and the settled pose is unchanged (measured in S0)."""
  m = pick_model
  for name in ("link2", "link3", "gripper_base"):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    assert m.body_gravcomp[bid] == 1.0


def test_both_profiles_exist_and_differ_by_the_wrist_payload():
  bare = piper.get_pick_robot_cfg("bare_gripper").spec_fn().compile()
  loaded = piper.get_pick_robot_cfg("wrist_sensor_payload_4p1N").spec_fn().compile()
  delta_n = (loaded.body_mass.sum() - bare.body_mass.sum()) * 9.81
  assert delta_n == pytest.approx(4.1, abs=0.05)
  # The hardware's feedforward model does not know about the sensor stack, and
  # that unmodelled weight IS the measured 20 mrad joint2 sag.
  bid = mujoco.mj_name2id(loaded, mujoco.mjtObj.mjOBJ_BODY, "wrist_payload")
  assert loaded.body_gravcomp[bid] == 0.0


def test_identified_plant_replaced_the_urdf_defaults(pick_model):
  m = pick_model
  for name, expected in piper.SYSID_COULOMB_NM.items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert m.dof_frictionloss[m.jnt_dofadr[jid]] == pytest.approx(expected)
