"""Session mismatch must be inert until asked for, and must bite when asked.

Two failure modes are worth more than the rest and both are silent:

* a perturbation that does nothing, which reads as "this axis does not
  matter" and would send Phase WM0 to a Red verdict for the wrong reason;
* a perturbation that leaks into the default path, which would quietly
  invalidate every number the repository already has.

These run on the CPU against config objects and small tensors.  The in-sim
counterpart, which checks that the values actually reach the simulator, is
scripts/check_perturb.py.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import math
from dataclasses import fields

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest
import torch

from piper_push import camera as cam
from piper_push import perturb
from piper_push.actions import RateLimitedJointPositionActionCfg


# ---------------------------------------------------------------------------
# The config itself
# ---------------------------------------------------------------------------


def test_default_is_inert():
  mm = perturb.SessionMismatchCfg()
  assert mm.is_inert()
  assert mm.active() == {}


def test_every_field_has_an_axis_description():
  """A knob nobody can interpret is a knob nobody should turn."""
  for f in fields(perturb.SessionMismatchCfg):
    assert f.name in perturb.AXES, f"{f.name} has no AXES entry"
    a = perturb.AXES[f.name]
    assert a.hardware, f"{f.name} has no hardware interpretation"
    assert a.group in ("camera", "robot", "gripper")


def test_nominal_matches_the_dataclass_default():
  """AXES.nominal is what the report prints as 'unperturbed'."""
  mm = perturb.SessionMismatchCfg()
  for f in fields(perturb.SessionMismatchCfg):
    got = getattr(mm, f.name)
    if got is None:  # depth_dropout means "leave the task's value"
      assert perturb.AXES[f.name].nominal == cam.DEPTH_DROPOUT
      continue
    assert float(got) == pytest.approx(perturb.AXES[f.name].nominal)


@pytest.mark.parametrize("name,value", [
  ("cam_pitch_deg", 3.0), ("depth_scale", 1.03), ("depth_bias_m", 0.01),
  ("obs_latency_steps", 2), ("action_latency_steps", 2),
  ("joint_response_scale", 0.85), ("action_deadband_rad", 0.002),
  ("gripper_rate_scale", 0.6), ("pad_friction_scale", 0.8),
  ("table_friction_scale", 0.8), ("servo_damping_scale", 0.5),
])
def test_one_axis_registers_as_active(name, value):
  mm = perturb.SessionMismatchCfg(**{name: value})
  assert not mm.is_inert()
  assert set(mm.active()) == {name}


def test_json_records_what_was_active_and_what_it_means():
  mm = perturb.SessionMismatchCfg(depth_scale=1.05)
  d = mm.to_json()
  assert d["_active"] == {"depth_scale": 1.05}
  assert "hardware" in d["_axes"]["depth_scale"]
  # Every axis is recorded, active or not, so a result file says what the
  # run did NOT perturb as well as what it did.
  assert d["depth_bias_m"] == 0.0


# ---------------------------------------------------------------------------
# The plant, which is the part with state
# ---------------------------------------------------------------------------


def _run(seq, latency=0, response=1.0, deadband=0.0, n=4, j=2):
  """Feed a sequence of commanded targets through the plant."""
  from piper_push.actions import apply_plant

  prev = torch.zeros(n, j)
  buf = [torch.zeros(n, j) for _ in range(latency)]
  out, held_total = [], 0.0
  for v in seq:
    eff, held = apply_plant(torch.full((n, j), float(v)), prev, buf,
                            response, deadband)
    held_total += float(held)
    prev = eff
    out.append(float(eff[0, 0]))
  return out, held_total


def test_plant_is_a_passthrough_at_defaults():
  seq = [0.3, -0.2, 0.9]
  out, held = _run(seq)
  assert out == pytest.approx(seq)
  assert held == 0.0


def test_latency_delays_the_command_by_exactly_n_steps():
  seq = [0.1, 0.2, 0.3, 0.4, 0.5]
  out, _ = _run(seq, latency=2)
  # The pipeline is primed with the reset posture (zero), so the first two
  # steps emit that and the rest emit the command from two steps ago.
  assert out[:2] == pytest.approx([0.0, 0.0])
  assert out[2:] == pytest.approx(seq[:3])


def test_response_scale_undershoots_every_step():
  (a, b), _ = _run([1.0, 1.0], response=0.5)
  assert a == pytest.approx(0.5)
  # It closes half the REMAINING gap each step, which is what a first-order
  # servo lag does -- not a permanent 50% error.
  assert b == pytest.approx(0.75)


def test_deadband_holds_small_steps_and_passes_large_ones():
  (small, big), held = _run([0.01, 0.5], deadband=0.05)
  assert small == pytest.approx(0.0)
  assert held == 8  # 4 envs x 2 joints, held on the first step only
  assert big == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Structured dropout
# ---------------------------------------------------------------------------


def test_blob_dropout_hits_the_requested_fraction():
  torch.manual_seed(0)
  d = torch.full((8, 60, 80, 1), 0.7)
  for frac in (0.05, 0.15, 0.30):
    out = perturb._blob_dropout(d, frac, far=1.5)
    got = (out > 1.4).float().mean().item()
    assert abs(got - frac) < 0.03, f"{frac} -> {got}"


def test_blob_dropout_is_contiguous_not_salt_and_pepper():
  """The point of it: holes must be spatially correlated.

  A Bernoulli mask at 15% has ~15% of each hole pixel's neighbours also holes;
  a blurred field has far more.  Without this the axis would be a slower way
  of writing the dropout the policy was already trained against.
  """
  torch.manual_seed(0)
  d = torch.full((4, 60, 80, 1), 0.7)
  holes = (perturb._blob_dropout(d, 0.15, 1.5) > 1.4)[..., 0].float()
  right = holes[:, :, 1:] * holes[:, :, :-1]
  neighbour_rate = right.sum() / holes[:, :, :-1].sum().clamp(min=1)
  iid = (torch.rand(4, 60, 80) < 0.15).float()
  iid_rate = (iid[:, :, 1:] * iid[:, :, :-1]).sum() / iid[:, :, :-1].sum().clamp(min=1)
  assert neighbour_rate > 3 * iid_rate, (
    f"blob {neighbour_rate:.2f} vs iid {iid_rate:.2f}")


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------


def _frame(n=16):
  pos = torch.tensor(cam.CAMERA_POS).expand(n, 3)
  aim = torch.tensor(cam.CAMERA_AIM)
  fwd = torch.nn.functional.normalize(aim - pos, dim=-1)
  up_w = torch.tensor([0.0, 0.0, 1.0]).expand(n, 3)
  right = torch.nn.functional.normalize(torch.cross(fwd, up_w, dim=-1), dim=-1)
  return fwd, right, torch.cross(right, fwd, dim=-1)


def test_zero_offset_leaves_the_optical_axis_exactly_alone():
  """The perturbed camera event must reduce to the stock one when unperturbed."""
  fwd, right, up = _frame()
  assert torch.equal(perturb.aim_offset(fwd, right, up, 0.0, 0.0), fwd)


@pytest.mark.parametrize("deg", [1.0, 3.0, 8.0])
def test_pitch_offset_turns_the_axis_by_exactly_that_angle(deg):
  fwd, right, up = _frame()
  turned = perturb.aim_offset(fwd, right, up, math.radians(deg), 0.0)
  cos = float((turned[0] * fwd[0]).sum().clamp(-1, 1))
  assert math.degrees(math.acos(cos)) == pytest.approx(deg, abs=1e-3)


def test_pitch_and_yaw_turn_about_different_axes():
  """Otherwise the two camera axes would be one axis under two names."""
  fwd, right, up = _frame()
  p = perturb.aim_offset(fwd, right, up, math.radians(4.0), 0.0)
  y = perturb.aim_offset(fwd, right, up, 0.0, math.radians(4.0))
  assert not torch.allclose(p, y, atol=1e-3)
  # Both are the same size step, in orthogonal directions.
  dp, dy = p[0] - fwd[0], y[0] - fwd[0]
  assert float(dp.norm()) == pytest.approx(float(dy.norm()), rel=1e-3)
  assert abs(float((dp / dp.norm() * (dy / dy.norm())).sum())) < 0.05
