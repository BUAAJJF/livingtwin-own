"""The stable actuator: is it the identity when untrained, and can it misbehave?

Phase RA-Sim-0's residual failed because its bound was the binding constraint.
This model's bounds are structural rather than a clip on the output, so the
things worth testing are that they are structural: that the effective command
cannot leave the joint range whatever the network says, that it cannot jump
faster than the pre-registered rate, and that a large accumulated lag is still
reachable -- because forbidding that is what made the old parameterisation
unable to express the target at all.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from piper_push import actuator as A


class FakeTerm:
  def __init__(self, n_env=4, n_joint=6, rest=0.0):
    self.device = "cpu"
    self._num_targets = n_joint
    self._default = torch.full((n_env, n_joint), float(rest))
    self._previous_target = self._default.clone()
    self._q = torch.zeros(n_env, n_joint)

  @property
  def joint_pos(self):
    return self._q

  @property
  def joint_vel(self):
    return torch.zeros_like(self._q)


def mk(**kw):
  return A.StableActuator(command_lo=(-1.0,) * 6, command_hi=(1.0,) * 6, **kw)


def test_an_untrained_model_passes_the_command_through():
  """Within the floor two identical MJWarp builds disagree by, 3.6e-7 rad."""
  m = mk()
  w = torch.zeros(4, 6)
  u = torch.full((4, 6), 0.0487)      # the largest step this task commands
  with torch.no_grad():
    ue, _, _, _ = m.step(torch.zeros(4, 6), torch.zeros(4, 6), u, u, w,
                         m.zero_hidden(4, "cpu"))
  assert float((ue - u).abs().max()) < 3.6e-7


def test_the_coefficients_start_at_their_identity_values():
  m = mk()
  with torch.no_grad():
    a, rp, rn, b, _ = m.coefficients(torch.zeros(2, A.FEATURE_DIM),
                                     m.zero_hidden(2, "cpu"))
  assert float(a.min()) > 1.0 - 1e-5
  assert float(rp.min()) > A.RATE_RANGE_RAD_S[1] - 1e-4
  assert float(rn.min()) > A.RATE_RANGE_RAD_S[1] - 1e-4
  assert float(b.abs().max()) == 0.0


def test_the_effective_command_cannot_leave_the_joint_range():
  """However hard the head is pushed.  This is the structural claim."""
  m = mk()
  with torch.no_grad():
    m.head.weight.normal_(0.0, 50.0)
    m.head.bias.normal_(0.0, 50.0)
    w = torch.full((64, 6), 0.99)
    h = m.zero_hidden(64, "cpu")
    for _ in range(200):
      u = torch.randn(64, 6) * 50.0     # absurd commands
      w_in = w
      ue, w, h, _ = m.step(torch.randn(64, 6), torch.randn(64, 6), u, u, w_in, h)
      assert float(ue.max()) <= 1.0 + 1e-6
      assert float(ue.min()) >= -1.0 - 1e-6
      assert torch.isfinite(ue).all()


def test_a_single_step_cannot_move_further_than_the_rate_limit():
  m = mk()
  with torch.no_grad():
    m.head.weight.normal_(0.0, 50.0)
    m.head.bias.normal_(0.0, 50.0)
    w = torch.zeros(64, 6)
    h = m.zero_hidden(64, "cpu")
    ceiling = A.RATE_RANGE_RAD_S[1] * m.dt
    for _ in range(50):
      u = torch.randn(64, 6) * 10.0
      prev = w
      ue, w, h, d = m.step(torch.randn(64, 6), torch.randn(64, 6), u, u, prev, h)
      assert float((ue - prev).abs().max()) <= ceiling + 1e-6


def test_a_large_accumulated_lag_is_still_reachable():
  """The property RA-Sim-0's bound forbade, and the reason it could not work.

  A constant command far from the state must be trackable, and the state must
  be allowed to sit far behind it on the way -- what is forbidden is getting
  there in one step.
  """
  m = mk()
  with torch.no_grad():
    # a slow drive: alpha small, rate small
    m.head.bias[0:6] = -2.0
    m.head.bias[6:18] = -3.0
    w = torch.zeros(1, 6)
    u = torch.full((1, 6), 0.9)
    h = m.zero_hidden(1, "cpu")
    lags = []
    for _ in range(100):
      ue, w, h, _ = m.step(torch.zeros(1, 6), torch.zeros(1, 6), u, u, w, h)
      lags.append(float((u - ue).abs().max()))
  assert max(lags) > 0.5           # a lag many times one action increment
  assert lags[-1] < lags[0]        # and it is closing, not diverging


def test_the_recursion_contracts_towards_a_held_command():
  m = mk()
  with torch.no_grad():
    m.head.weight.normal_(0.0, 5.0)
    w = torch.zeros(8, 6)
    u = torch.full((8, 6), 0.3)
    h = m.zero_hidden(8, "cpu")
    for _ in range(500):
      ue, w, h, _ = m.step(torch.zeros(8, 6), torch.zeros(8, 6), u, u, w, h)
    assert torch.isfinite(w).all()
    assert float(w.abs().max()) <= 1.0 + 1e-6


def test_the_feature_builder_cannot_be_handed_a_hidden_quantity():
  """Five tensors, and no sixth argument for the target's own state."""
  assert list(inspect.signature(A.build_features).parameters) == [
    "q", "qd", "u", "u_prev", "w"]


def test_the_module_never_mentions_the_hidden_target():
  import pathlib
  src = pathlib.Path(A.__file__).read_text()
  # `_lag` is deliberately absent from this list: the module has its own
  # `max_abs_command_lag` statistic, which is the model's distance from the
  # command it is chasing and has nothing to do with the target's state.
  for name in ("_flank", "hidden_plant", "HiddenPlant", "BACKLASH", "KAPPA",
               "S_REF", "over_speed", "reward", "trip"):
    assert name not in src, f"actuator.py mentions {name}"


def test_hidden_state_and_actuator_state_reset_per_environment(tmp_path):
  m = mk()
  with torch.no_grad():
    m.head.weight.normal_(0.0, 2.0)
  p = tmp_path / "act.pt"
  torch.save({"state_dict": m.state_dict(), "hidden": 64, "dt": 0.02,
              "cmd_lo": [-1.0] * 6, "cmd_hi": [1.0] * 6,
              "alpha_range": list(A.ALPHA_RANGE),
              "rate_range": list(A.RATE_RANGE_RAD_S),
              "bias_range": list(A.BIAS_RANGE_RAD)}, p)
  term = FakeTerm(n_env=4)
  hook = A.ActuatorHookCfg(checkpoint=str(p)).build(term)
  t = torch.full((4, 6), 0.4)
  for _ in range(10):
    hook(t, term)
  keep_h = hook._h[2:].clone()
  keep_w = hook._w[2:].clone()
  hook.reset(torch.tensor([0, 1]))
  assert torch.allclose(hook._h[2:], keep_h, atol=0.0)
  assert torch.allclose(hook._w[2:], keep_w, atol=0.0)
  assert float(hook._h[:2].abs().max()) == 0.0
  assert torch.allclose(hook._w[:2], term._previous_target[:2])


def test_a_disabled_hook_returns_the_command_untouched(tmp_path):
  m = mk()
  with torch.no_grad():
    m.head.bias.normal_(0.0, 3.0)
  p = tmp_path / "act.pt"
  torch.save({"state_dict": m.state_dict(), "hidden": 64, "dt": 0.02,
              "cmd_lo": [-1.0] * 6, "cmd_hi": [1.0] * 6,
              "alpha_range": list(A.ALPHA_RANGE),
              "rate_range": list(A.RATE_RANGE_RAD_S),
              "bias_range": list(A.BIAS_RANGE_RAD)}, p)
  term = FakeTerm()
  off = A.ActuatorHookCfg(checkpoint=str(p), enabled=False).build(term)
  t = torch.full((4, 6), 0.25)
  assert torch.allclose(off(t, term), t, atol=0.0)


def test_the_model_refuses_an_action_term_of_the_wrong_width(tmp_path):
  m = mk()
  p = tmp_path / "act.pt"
  torch.save({"state_dict": m.state_dict(), "hidden": 64, "dt": 0.02,
              "cmd_lo": [-1.0] * 6, "cmd_hi": [1.0] * 6,
              "alpha_range": list(A.ALPHA_RANGE),
              "rate_range": list(A.RATE_RANGE_RAD_S),
              "bias_range": list(A.BIAS_RANGE_RAD)}, p)
  with pytest.raises(ValueError):
    A.ActuatorHookCfg(checkpoint=str(p)).build(FakeTerm(n_joint=1))


def test_the_frozen_ranges_are_the_ones_the_plan_names():
  assert A.ALPHA_RANGE == (0.02, 1.0)
  assert A.RATE_RANGE_RAD_S == (0.05, 4.0)
  assert A.BIAS_RANGE_RAD == (-0.05, 0.05)
  assert A.FEATURE_DIM == 36
  assert sum(p.numel() for p in mk().parameters()) == 21144


def test_installing_appends_rather_than_replaces():
  class Cfg:
    actions = {"arm": type("A", (), {"command_hooks": ()})()}
  cfg = Cfg()
  sentinel = object()
  cfg.actions["arm"].command_hooks = (sentinel,)
  applied = A.apply_actuator(cfg, A.ActuatorHookCfg(checkpoint="x"))
  assert cfg.actions["arm"].command_hooks[0] is sentinel
  assert applied["kind"] == "stable_stateful_actuator"
  assert A.apply_actuator(cfg, None) == {}


def test_the_cli_default_installs_nothing():
  class Args:
    pass
  assert A.actuator_from_args(Args()) is None
  a = Args()
  a.actuator = ""
  assert A.actuator_from_args(a) is None
