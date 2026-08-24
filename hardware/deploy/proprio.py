"""The 36 numbers that go in front of the camera channels, in the right order.

None of this is difficult and all of it is easy to get subtly wrong, so every
term is built by name against ``obs_spec.json`` -- exported from a built
environment by ``scripts/export_obs_spec.py`` -- and the assembly refuses to
run if a term is missing, has moved, or has changed width.  A policy handed a
correct vector in the wrong order does not fail, it acts confidently and
wrongly, and there is nothing in a log that would tell you.

The order is worth looking at once, because reading it off ``env_cfg.py`` gives
the wrong answer.  The vision variant deletes ``grasped`` from the
proprioception group and adds ``squeeze``; Python dicts keep insertion order,
so ``squeeze`` lands at the end, *after* ``actions``, not where ``grasped``
was:

    joint_pos(8) joint_vel(8) ee_pose(9) gripper(1) pad_contact(2)
    actions(7) squeeze(1)

Two terms do not exist on the robot and have to be reconstructed.

``ee_pose`` is the grasp site in the base frame, and the honest way to get it
is to ask the same model the policy was trained against.  This loads
``piper_push.robot``'s own MuJoCo spec and runs forward kinematics on it, so
the site offset, the gripper geometry and the base frame are the simulator's by
construction rather than by a second implementation that agrees until someone
edits the URDF.

``pad_contact`` is two contact sensors in simulation and there are no contact
sensors on this gripper.  What there is, is a drive that reports its current,
and the comment in the environment that introduced the term says exactly this:
the deployable content of the channel is "the drive is loaded".  So both
elements get the same bit, from the gripper effort.  That is a real
approximation and it is the largest one in this file: in simulation the two
pads can differ, and a policy that learned something from the difference has
nothing to read here.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import mujoco
import numpy as np

_COLLISION_GROUP = 3

HERE = pathlib.Path(__file__).resolve().parent
SPEC_FILE = HERE / "obs_spec.json"


@dataclasses.dataclass
class JointFeedback:
  """What the arm reports, in the simulator's eight-joint order.

  Eight and not seven, because that is the width of ``joint_pos`` in the
  observation: six arm joints and both gripper fingers.  The second finger is
  a mechanical mimic of the first and no drive reports it, so ``from_arm``
  fills it in as the negative of the first -- which is what the equality
  constraint in the model does.

  ``gripper_effort`` is normalised, 1.0 being stall.  It is the only channel
  here that is not a joint quantity, and it is here because it is the one
  thing on the real gripper that can stand in for the pad contact sensors.
  """

  position: np.ndarray
  velocity: np.ndarray
  target: np.ndarray
  gripper_effort: float = 0.0

  @classmethod
  def from_arm(cls, q6, dq6, gripper_m, gripper_vel, target6,
               gripper_target_m, gripper_effort,
               joint_names: list[str] | None = None) -> "JointFeedback":
    names = list(joint_names or default_joint_names())
    n = len(names)
    pos = np.zeros(n)
    vel = np.zeros(n)
    tgt = np.zeros(n)
    for i, j in enumerate(("joint1", "joint2", "joint3",
                           "joint4", "joint5", "joint6")):
      k = names.index(j)
      pos[k], vel[k], tgt[k] = q6[i], dq6[i], target6[i]
    g1 = names.index("gripper_joint1")
    pos[g1], vel[g1], tgt[g1] = gripper_m, gripper_vel, gripper_target_m
    if "gripper_joint2" in names:
      g2 = names.index("gripper_joint2")
      pos[g2], vel[g2], tgt[g2] = -gripper_m, -gripper_vel, -gripper_target_m
    return cls(position=pos, velocity=vel, target=tgt,
               gripper_effort=float(gripper_effort))


#: Kept so that older call sites keep working; the two are the same record.
JointState = JointFeedback


def _spec(path: pathlib.Path | str = SPEC_FILE) -> dict:
  return json.loads(pathlib.Path(path).read_text())


def default_joint_names() -> list[str]:
  return list(_spec()["joint_names"])


class Kinematics:
  """Forward kinematics on the policy's own robot model.

  Built from ``piper_push.robot.get_pick_spec``, which is the spec the training
  environment builds its robot from.  No GL and no simulator is involved --
  ``mj_kinematics`` is pure geometry -- so this runs anywhere, including inside
  the control loop at 50 Hz, where it costs a few microseconds.
  """

  def __init__(self, joint_names: list[str] | None = None,
               site_name: str = "grasp_site"):
    from piper_push import robot as robot_mod

    self.model = robot_mod.get_pick_spec().compile()
    self.data = mujoco.MjData(self.model)
    self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                     site_name)
    if self.site_id < 0:
      raise RuntimeError(f"no site {site_name!r} in the robot model")
    # Addressed by name, not by index.  The environment reports its joints in
    # the order mjlab resolved them and this model compiles them in the order
    # the spec declares them; the two happen to agree today, and a silent
    # permutation of six joint angles is not a failure anyone would spot.
    self.joint_names = list(joint_names or default_joint_names())
    self._qadr = np.array([
      self.model.jnt_qposadr[
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
      ] for n in self.joint_names
    ])
    self._spheres: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    missing = [n for n in self.joint_names
               if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) < 0]
    if missing:
      raise RuntimeError(f"joints missing from the robot model: {missing}")

  def update(self, q: np.ndarray) -> None:
    self.data.qpos[self._qadr] = q
    mujoco.mj_kinematics(self.model, self.data)

  @property
  def site_pos(self) -> np.ndarray:
    """Where the grasp site is, in the base frame.

    The segmenter's target rule needs only this, not the full pose, and asking
    for the pose to slice three numbers off it invites someone to slice the
    wrong three.
    """
    return np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64)

  def ee_pose_b(self) -> np.ndarray:
    """``(9,)``: position, then the first two columns of the rotation matrix.

    The 6-D rotation, not a quaternion -- matching ``mdp._rotation_6d`` -- and
    for the reason that representation exists: it is continuous, so a policy
    regressing on it never sees the sign flip a quaternion has at the antipode.
    """
    pos = self.data.site_xpos[self.site_id]
    mat = self.data.site_xmat[self.site_id].reshape(3, 3)
    return np.concatenate([pos, mat[:, 0], mat[:, 1]]).astype(np.float32)

  def link_positions(self) -> np.ndarray:
    """``(L, 3)`` base-frame positions of every body, for the arm mask.

    The segmenter uses these to throw away the points that are the robot.  A
    sphere per body is coarse compared to the real link shapes; it is also
    conservative in the direction that matters, because losing a few object
    pixels next to the gripper costs less than declaring the forearm an object
    and reaching for it.
    """
    return np.asarray(self.data.xpos[1:], dtype=np.float64)

  def link_spheres(self) -> tuple[np.ndarray, np.ndarray]:
    """A sphere cover of the arm: ``(centres, radii)`` in the base frame.

    The segmenter subtracts these from the scene.  Everything about them is
    read off the model's own geometry -- the axis-aligned bounding box MuJoCo
    computes for each collision geom -- so a change to the gripper or to the
    payload moves this with it.

    One sphere per geom is not good enough and it is worth saying why, because
    it is the obvious implementation and it fails quietly.  ``geom_rbound`` for
    the upper arm is 230 mm; a single sphere that size, centred on the link,
    erases a 460 mm disc of table around it, which at this camera's range is
    most of the object area.  So each geom is covered by a row of spheres along
    its longest axis, each only as wide as the link's cross section.  Thirty
    spheres against a hundred thousand points is a few milliseconds and it
    removes the arm without removing the table it is over.

    Collision geoms only.  The visual meshes are the same shapes in group 2 and
    covering both would double the cost for nothing.
    """
    if self._spheres is None:
      import mujoco as _mj

      local, radii, geoms = [], [], []
      for g in range(self.model.ngeom):
        if self.model.geom_group[g] != _COLLISION_GROUP:
          continue
        aabb = self.model.geom_aabb[g].reshape(2, 3)
        centre, half = aabb[0], aabb[1]
        axis = int(np.argmax(half))
        cross = float(np.linalg.norm(np.delete(half, axis)))
        span = float(half[axis])
        # How many spheres of radius ``cross`` it takes to cover ``2*span``.
        n = int(np.clip(np.ceil(span / max(cross, 1e-4)), 1, 8))
        reach = max(span - cross, 0.0)
        for t in (np.linspace(-reach, reach, n) if n > 1 else [0.0]):
          c = centre.copy()
          c[axis] += t
          local.append(c)
          radii.append(cross)
          geoms.append(g)
      self._spheres = (np.asarray(local), np.asarray(radii),
                       np.asarray(geoms, dtype=int))
      del _mj
    local, radii, geoms = self._spheres
    R = self.data.geom_xmat[geoms].reshape(-1, 3, 3)
    centres = self.data.geom_xpos[geoms] + np.einsum("nij,nj->ni", R, local)
    return np.asarray(centres, dtype=np.float64), radii


class ProprioBuilder:
  """Assemble the proprioception vector and check it against the exported spec."""

  # Widths the code below produces.  Compared against the spec at construction:
  # if the environment changes and this does not, the mismatch is an exception
  # at start-up rather than a policy quietly reading joint velocities as if
  # they were an end-effector pose.
  WIDTHS = {
    "joint_pos": 8, "joint_vel": 8, "ee_pose": 9, "gripper": 1,
    "pad_contact": 2, "actions": 7, "squeeze": 1,
  }

  def __init__(self, spec_path: pathlib.Path | str = SPEC_FILE,
               contact_effort: float = 0.15) -> None:
    spec = _spec(spec_path)
    self.spec = spec
    self.terms = spec["groups"]["proprio"]["terms"]
    self.total = spec["groups"]["proprio"]["total"]
    self.default_q = np.asarray(spec["default_joint_pos"], dtype=np.float32)
    self.joint_names = spec["joint_names"]
    self.contact_effort = contact_effort

    unknown = [t["name"] for t in self.terms if t["name"] not in self.WIDTHS]
    if unknown:
      raise RuntimeError(
        f"obs_spec.json has terms this file cannot build: {unknown}. "
        "Re-run scripts/export_obs_spec.py, then teach ProprioBuilder the "
        "new terms -- do not deploy with them zero-filled."
      )
    for t in self.terms:
      if t["width"] != self.WIDTHS[t["name"]]:
        raise RuntimeError(
          f"term {t['name']!r} is {t['width']} wide in the environment and "
          f"{self.WIDTHS[t['name']]} here"
        )
    self.kin = Kinematics(self.joint_names)

  def __call__(self, js: JointFeedback,
               last_action: np.ndarray) -> np.ndarray:
    self.kin.update(js.position)
    gi = self.joint_names.index("gripper_joint1")
    loaded = float(abs(js.gripper_effort) > self.contact_effort)
    parts = {
      # ``joint_pos_rel`` and ``joint_vel_rel``: relative to the default pose,
      # and the default velocity is zero, so the second one is just velocity.
      "joint_pos": (js.position - self.default_q).astype(np.float32),
      "joint_vel": js.velocity.astype(np.float32),
      "ee_pose": self.kin.ee_pose_b(),
      "gripper": np.array([2.0 * js.position[gi]], dtype=np.float32),
      # One drive, so one bit, reported on both channels.  See the module
      # docstring: this is the largest approximation in the file.
      "pad_contact": np.array([loaded, loaded], dtype=np.float32),
      "actions": np.asarray(last_action, dtype=np.float32),
      # ``gripper_squeeze``: actual minus commanded, which is what a position
      # servo turns into force and what the drive reports as current.
      "squeeze": np.array([js.position[gi] - js.target[gi]], dtype=np.float32),
    }
    out = np.concatenate([parts[t["name"]] for t in self.terms])
    assert out.size == self.total, (out.size, self.total)
    return out.astype(np.float32)
