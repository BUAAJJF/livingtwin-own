"""The frozen structural target: does it do what its docstring says?

These are arithmetic tests on the hook in isolation.  Whether it also changes
a real MJWarp rollout is Stage 0's audit, which runs a simulator; that cannot
run here, so the two halves are separate on purpose.
"""

from __future__ import annotations

import pytest
import torch

from piper_push.hidden_plant import HiddenPlantCfg, apply_hidden_plant


class FakeTerm:
  """The four attributes a command hook is allowed to reach for."""

  def __init__(self, n_env: int = 3, n_joint: int = 6, rest: float = 0.0):
    self.device = "cpu"
    self._num_targets = n_joint
    self._default = torch.full((n_env, n_joint), rest)
    self._previous_target = self._default.clone()

  @property
  def joint_pos(self):
    return self._default

  @property
  def joint_vel(self):
    return torch.zeros_like(self._default)


def build(**kw):
  term = FakeTerm(**{k: v for k, v in kw.items() if k in ("n_env", "rest")})
  cfg = HiddenPlantCfg(**{k: v for k, v in kw.items()
                          if k not in ("n_env", "rest")})
  hook = cfg.build(term)
  hook.reset()
  return term, hook


def test_a_command_that_never_moves_produces_no_motion():
  term, hook = build()
  u = torch.zeros(3, 6)
  for _ in range(20):
    out = hook(u, term)
  assert torch.allclose(out, torch.zeros(3, 6), atol=1e-12)


def test_the_backlash_band_swallows_a_small_command_step():
  # No lag: beta0 = 1 with kappa = 0 makes the servo instantaneous, which
  # isolates the gear play from the current limit.
  term, hook = build(beta0=1.0, kappa=(0.0,) * 6)
  small = torch.full((3, 6), 0.004)  # inside every joint's up-flank but j5
  out = hook(small, term)
  # joint5's up flank is 0.006, so 0.004 is inside it and it must not move.
  assert out[0, 4] == pytest.approx(0.0, abs=1e-12)
  # joint1's up flank is 0.012, likewise.
  assert out[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_a_large_command_arrives_one_flank_width_short():
  term, hook = build(beta0=1.0, kappa=(0.0,) * 6)
  out = hook(torch.full((3, 6), 0.5), term)
  up = torch.tensor(HiddenPlantCfg().backlash_up)
  assert torch.allclose(out[0], 0.5 - up, atol=1e-9)


def test_the_two_flanks_are_not_the_same_width():
  term, hook = build(beta0=1.0, kappa=(0.0,) * 6)
  hook(torch.full((3, 6), 0.5), term)
  out = hook(torch.full((3, 6), -0.5), term)
  dn = torch.tensor(HiddenPlantCfg().backlash_down)
  assert torch.allclose(out[0], -0.5 + dn, atol=1e-9)
  # And the asymmetry is real: joint1 lags more coming down than going up,
  # joint2 the other way round.  A single global backlash cannot be both.
  cfg = HiddenPlantCfg()
  assert cfg.backlash_down[0] > cfg.backlash_up[0]
  assert cfg.backlash_down[1] < cfg.backlash_up[1]


def test_a_reversal_inside_the_band_moves_nothing():
  term, hook = build(beta0=1.0, kappa=(0.0,) * 6)
  hook(torch.full((3, 6), 0.5), term)
  held = hook(torch.full((3, 6), 0.5), term).clone()
  # Back off by less than the sum of the flanks: the teeth stay put.
  out = hook(torch.full((3, 6), 0.5 - 0.004), term)
  assert torch.allclose(out, held, atol=1e-12)


def test_a_bigger_step_completes_a_smaller_fraction_of_itself():
  # Backlash off, lag on: the part `joint_response_scale` cannot say.
  term, hook = build(backlash_up=(0.0,) * 6, backlash_down=(0.0,) * 6)
  small = hook(torch.full((3, 6), 0.005), term).clone()
  term2, hook2 = build(backlash_up=(0.0,) * 6, backlash_down=(0.0,) * 6)
  big = hook2(torch.full((3, 6), 0.5), term2).clone()
  frac_small = float(small[0, 0]) / 0.005
  frac_big = float(big[0, 0]) / 0.5
  assert frac_small > frac_big
  # And the ratio is not a rounding artefact: at S_REF the fraction is
  # beta0 / (1 + kappa), which for joint1 is 0.85 / 2.6.
  term3, hook3 = build(backlash_up=(0.0,) * 6, backlash_down=(0.0,) * 6)
  at_ref = hook3(torch.full((3, 6), HiddenPlantCfg().s_ref), term3)
  assert float(at_ref[0, 0]) / HiddenPlantCfg().s_ref == pytest.approx(
    0.85 / 2.6, rel=1e-6)


def test_a_constant_response_scale_cannot_reproduce_the_lag():
  """The claim the whole phase rests on, checked rather than asserted.

  ``perturb``'s ``joint_response_scale`` is one number.  Whatever it is, it
  cannot match the target at two step sizes at once.
  """
  term, hook = build(backlash_up=(0.0,) * 6, backlash_down=(0.0,) * 6)
  fracs = []
  for step in (0.005, 0.5):
    t, h = build(backlash_up=(0.0,) * 6, backlash_down=(0.0,) * 6)
    fracs.append(float(h(torch.full((3, 6), step), t)[0, 0]) / step)
  assert abs(fracs[0] - fracs[1]) > 0.3


def test_resetting_one_environment_leaves_the_others_alone():
  term, hook = build(n_env=4)
  u = torch.full((4, 6), 0.3)
  for _ in range(5):
    hook(u, term)
  lag, flank = hook.state
  keep = flank[2:].clone()
  hook.reset(torch.tensor([0, 1]))
  assert torch.allclose(hook.state[1][2:], keep, atol=1e-12)
  assert torch.allclose(hook.state[1][:2], term._previous_target[:2], atol=1e-12)
  assert torch.allclose(hook.state[0][:2], term._previous_target[:2], atol=1e-12)


def test_the_state_survives_being_called_under_inference_mode():
  """The bug actions.py warns about, in the one place it would bite here.

  A hook that rebinds its state inside a rollout leaves a tensor the next
  reset cannot write to, and the failure appears only at evaluation scale.
  """
  term, hook = build()
  with torch.inference_mode():
    for _ in range(3):
      hook(torch.full((3, 6), 0.2), term)
  hook.reset()  # would raise if `_lag` had been rebound inside the block
  assert float(hook.state[0].abs().max()) == 0.0


@pytest.mark.parametrize("bad", [
  {"backlash_up": (0.01,) * 5},
  {"backlash_down": (-0.01,) * 6},
  {"beta0": 0.0},
  {"beta0": 1.5},
  {"s_ref": 0.0},
  {"kappa": (1.0,) * 3},
])
def test_a_malformed_target_is_refused_at_build(bad):
  term = FakeTerm()
  with pytest.raises(ValueError):
    HiddenPlantCfg(**bad).build(term)


def test_installing_appends_rather_than_replaces():
  class Cfg:
    actions = {"arm": type("A", (), {"command_hooks": ()})()}

  cfg = Cfg()
  sentinel = object()
  cfg.actions["arm"].command_hooks = (sentinel,)
  applied = apply_hidden_plant(cfg, HiddenPlantCfg())
  assert cfg.actions["arm"].command_hooks[0] is sentinel
  assert applied["name"] == "backlash+rate_lag"
  assert apply_hidden_plant(cfg, None) == {}
  assert len(cfg.actions["arm"].command_hooks) == 2


def test_the_frozen_constants_are_the_ones_the_plan_names():
  """A guard against the phase's own worst failure mode.

  Tuning the target until the result is pretty is the thing the
  specification forbids most explicitly.  These numbers were written into
  docs/ra_sim0_experiment_plan.md before any result run; if a later commit
  moves them, this test is what says so.
  """
  cfg = HiddenPlantCfg()
  assert cfg.backlash_up == (0.012, 0.018, 0.015, 0.008, 0.006, 0.010)
  assert cfg.backlash_down == (0.020, 0.009, 0.022, 0.005, 0.011, 0.006)
  assert cfg.beta0 == 0.85
  assert cfg.kappa == (1.6, 1.6, 1.6, 0.9, 0.9, 0.9)
  assert cfg.s_ref == 0.04
