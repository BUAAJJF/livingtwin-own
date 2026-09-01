"""The tube between the camera and the object, and which side of it is free.

``sight_cylinder`` is the term most likely to break a run and the one whose
form is easiest to get subtly wrong.  Distance to the *segment* -- the obvious
implementation -- ends at the object, so a hand that has arrived sits at
distance zero and is charged forever, which reads as an instruction never to
grasp anything.  A cylinder has a far cap, and everything beyond it is free at
any lateral distance, which is what turns the penalty into "approach from
behind" rather than "stay away".

A sign error here inverts that and nothing else would notice: training would
simply produce a policy that prefers to stand in front of the object, and the
first evidence would be a bad hardware run a day later.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

# mdp first: importing ``piper_push.camera`` on its own pulls the task
# registry in behind it and the two arrive at each other half-built.
from piper_push.tasks.pick_place import mdp as pick_mdp
from piper_push import camera as sim_camera

CAM = np.asarray(sim_camera.CAMERA_POS, dtype=np.float64)


def _env(body_points, object_pos, grasped=False, site=None, opening=0.10,
         half=(0.03, 0.03, 0.04), pad_found=(0, 0)):
  """The smallest thing these terms will accept, so the geometry is testable."""
  n = 1
  bodies = torch.tensor(np.asarray(body_points, np.float32)).view(n, -1, 3)
  obj = torch.tensor(np.asarray([object_pos], np.float32))
  site_t = torch.tensor(np.asarray([site if site is not None else object_pos],
                                   np.float32))
  cmd = types.SimpleNamespace(
    _object_pos_local=lambda: obj,
    _site_pos_w=lambda: site_t,
    _gripper_opening=lambda: torch.tensor([float(opening)]),
    object_half_size=torch.tensor(np.asarray([half], np.float32)),
    pad_found=torch.tensor(np.asarray([pad_found], np.float32)),
    grasped=torch.tensor([bool(grasped)]),
  )
  robot = types.SimpleNamespace(
    data=types.SimpleNamespace(body_link_pos_w=bodies))
  scene = types.SimpleNamespace(
    env_origins=torch.zeros(n, 3),
    __getitem__=lambda self, k: robot,
  )
  # SimpleNamespace cannot carry __getitem__; a tiny class can.
  class _Scene:
    env_origins = torch.zeros(n, 3)
    def __getitem__(self, k): return robot
  env = types.SimpleNamespace(
    num_envs=n, device="cpu", scene=_Scene(),
    command_manager=types.SimpleNamespace(get_term=lambda name: cmd),
  )
  return env


def _cfg(k=1):
  return types.SimpleNamespace(name="robot", body_ids=list(range(k)))


OBJ = [0.0, 0.40, 0.03]
U = (np.asarray(OBJ) - CAM) / np.linalg.norm(np.asarray(OBJ) - CAM)


def test_a_body_behind_the_object_is_free_however_close_it_is():
  """The far cap is the whole point.

  A hand that has reached the object from behind sits just past the end of the
  axis.  Its lateral distance is nearly zero, and it must still cost nothing --
  otherwise the term forbids the grasp it is supposed to shape.
  """
  for back in (0.01, 0.05, 0.20):
    p = np.asarray(OBJ) + back * U
    cost = pick_mdp.sight_cylinder(
      _env([p], OBJ), "pick", _cfg(), radius=0.07)
    assert float(cost) == pytest.approx(0.0), f"charged {back} m behind"


def test_a_body_on_the_axis_in_front_of_the_object_is_charged():
  p = np.asarray(OBJ) - 0.15 * U          # 150 mm toward the camera
  cost = pick_mdp.sight_cylinder(_env([p], OBJ), "pick", _cfg(), radius=0.07)
  assert float(cost) == pytest.approx(0.07, abs=1e-3), "on-axis costs radius"


def test_the_charge_falls_off_with_lateral_distance_and_stops_at_the_radius():
  mid = np.asarray(OBJ) - 0.15 * U
  side = np.cross(U, [0.0, 0.0, 1.0])
  side = side / np.linalg.norm(side)
  costs = [float(pick_mdp.sight_cylinder(
    _env([mid + r * side], OBJ), "pick", _cfg(), radius=0.07))
    for r in (0.0, 0.035, 0.069, 0.071, 0.20)]
  assert costs[0] > costs[1] > costs[2] > 0.0
  assert costs[3] == pytest.approx(0.0) and costs[4] == pytest.approx(0.0)


def test_a_body_behind_the_camera_is_free():
  """t < 0 is outside the near cap.  Nothing behind the lens can occlude."""
  p = CAM - 0.30 * U
  assert float(pick_mdp.sight_cylinder(
    _env([p], OBJ), "pick", _cfg(), radius=0.07)) == pytest.approx(0.0)


def test_every_body_contributes():
  a = np.asarray(OBJ) - 0.15 * U
  b = np.asarray(OBJ) - 0.30 * U
  one = float(pick_mdp.sight_cylinder(_env([a], OBJ), "pick", _cfg(1), radius=0.07))
  two = float(pick_mdp.sight_cylinder(_env([a, b], OBJ), "pick", _cfg(2), radius=0.07))
  assert two == pytest.approx(2 * one, rel=1e-4)


def test_jaws_across_the_view_score_higher_than_jaws_along_it():
  """The reward is 1 when the finger axis is across the camera ray, 0 along."""
  side = np.cross(U, [0.0, 0.0, 1.0]); side /= np.linalg.norm(side)
  across = [np.asarray(OBJ) + 0.03 * side, np.asarray(OBJ) - 0.03 * side]
  along = [np.asarray(OBJ) + 0.03 * U, np.asarray(OBJ) - 0.03 * U]
  ra = float(pick_mdp.wrist_side_on(
    _env(across, OBJ, site=OBJ), "pick", _cfg(2), near_m=0.20))
  rl = float(pick_mdp.wrist_side_on(
    _env(along, OBJ, site=OBJ), "pick", _cfg(2), near_m=0.20))
  assert ra == pytest.approx(1.0, abs=1e-3)
  assert rl == pytest.approx(0.0, abs=1e-3)


def test_the_wrist_term_pays_nothing_far_away_and_unheld():
  side = np.cross(U, [0.0, 0.0, 1.0]); side /= np.linalg.norm(side)
  across = [np.asarray(OBJ) + 0.03 * side, np.asarray(OBJ) - 0.03 * side]
  far = list(np.asarray(OBJ) + np.asarray([0.6, 0.0, 0.0]))
  r = float(pick_mdp.wrist_side_on(
    _env(across, OBJ, site=far), "pick", _cfg(2), near_m=0.20))
  assert r == pytest.approx(0.0)


def test_closing_on_the_object_is_not_priced_as_a_shove():
  """The separator is the jaw opening, not the contact.

  Charging any pre-grasp contact would price the one moment the task pays for.
  A hand wide enough to receive the object is approaching it; a narrower one
  that is already touching is pushing it.
  """
  obj = OBJ
  wide = _env([obj], obj, opening=0.10, half=(0.03, 0.03, 0.04),
              pad_found=(1, 1))
  narrow = _env([obj], obj, opening=0.02, half=(0.03, 0.03, 0.04),
                pad_found=(1, 1))
  assert float(pick_mdp.premature_touch(wide, "pick", ())) == 0.0
  assert float(pick_mdp.premature_touch(narrow, "pick", ())) == 1.0


def test_a_grasped_object_is_never_a_premature_touch():
  obj = OBJ
  held = _env([obj], obj, opening=0.02, grasped=True, pad_found=(1, 1))
  assert float(pick_mdp.premature_touch(held, "pick", ())) == 0.0
