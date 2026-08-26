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
  """Exactly, not approximately: the gates start at zero."""
  m = mk()
  w = torch.zeros(4, 6)
  u = torch.full((4, 6), 0.0487)      # the largest step this task commands
  with torch.no_grad():
    ue, _, _, _ = m.step(torch.zeros(4, 6), torch.zeros(4, 6), u, u, w,
                         m.zero_hidden(4, "cpu"))
  assert float((ue - u).abs().max()) == 0.0


def test_the_coefficients_start_at_their_identity_values():
  m = mk()
  with torch.no_grad():
    a, rp, rn, b, _ = m.coefficients(torch.zeros(2, A.FEATURE_DIM),
                                     m.zero_hidden(2, "cpu"))
  assert float(a.min()) == 1.0
  assert float(rp.min()) == A.RATE_RANGE_RAD_S[1]
  assert float(rn.min()) == A.RATE_RANGE_RAD_S[1]
  assert float(b.abs().max()) == 0.0


def test_the_effective_command_cannot_leave_the_joint_range():
  """However hard the head is pushed.  This is the structural claim."""
  m = mk()
  with torch.no_grad():
    m.head.weight.normal_(0.0, 50.0)
    m.head.bias.normal_(0.0, 50.0)
    m.gate.normal_(0.0, 5.0)          # the gate must not be a way out either
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
    m.gate.normal_(0.0, 5.0)
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
    # a slow drive: gate the alpha and rate deviations fully open, and put
    # the head where sigmoid is near 1 so both land near their lower bounds
    m.head.weight.zero_()
    m.head.bias.zero_()
    m.gate[0].fill_(1.0)     # alpha -> alpha_min side
    m.gate[1].fill_(1.0)     # rate+ -> rate_min side
    m.gate[2].fill_(1.0)
    m.head.bias[0:18] = 3.0
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
    m.gate.normal_(0.0, 2.0)
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


def test_the_module_never_reads_the_hidden_target():
  """A check on the code, not on the prose.

  The docstring names the forbidden quantities precisely because it is
  documenting that they are forbidden, so a grep over the file would flag its
  own warning.  This walks the syntax tree instead and looks at identifiers,
  attributes and non-docstring literals -- the things that could actually
  reach one.
  """
  import ast
  import pathlib

  tree = ast.parse(pathlib.Path(A.__file__).read_text())
  docstrings = set()
  for node in ast.walk(tree):
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
      d = ast.get_docstring(node, clean=False)
      if d:
        docstrings.add(d)
    # a bare string statement is an attribute docstring in this codebase
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
       and isinstance(node.value.value, str):
      docstrings.add(node.value.value)

  seen = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Name):
      seen.add(node.id)
    elif isinstance(node, ast.Attribute):
      seen.add(node.attr)
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
      if node.value not in docstrings:
        seen.add(node.value)

  for name in ("_flank", "hidden_plant", "HiddenPlant", "BACKLASH_UP",
               "BACKLASH_DOWN", "KAPPA", "S_REF", "over_speed", "reward",
               "trip", "success"):
    assert name not in seen, f"actuator.py's code reaches {name}"
  # and the one import list it is allowed to have
  imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
  assert "piper_push.hidden_plant" not in imports


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


def test_the_gate_is_the_only_thing_that_leaves_the_identity():
  """A live gradient at zero, which the saturated sigmoid did not have."""
  m = mk()
  u = torch.full((4, 6), 0.0487)
  z = torch.zeros(4, 6)
  ue, _, _, _ = m.step(z, z, u, u, torch.zeros(4, 6), m.zero_hidden(4, "cpu"))
  ue.sum().backward()
  assert float(m.gate.grad.abs().max()) > 1e-3
  # and the head is frozen until a gate moves, which is what a zero gate means
  assert float(m.head.weight.grad.abs().max()) == 0.0


def test_a_runaway_gate_cannot_take_a_coefficient_out_of_its_range():
  m = mk()
  with torch.no_grad():
    m.gate.fill_(1e6)
    a, rp, rn, b, _ = m.coefficients(torch.randn(16, A.FEATURE_DIM),
                                     m.zero_hidden(16, "cpu"))
  assert float(a.min()) >= A.ALPHA_RANGE[0] - 1e-9
  assert float(a.max()) <= A.ALPHA_RANGE[1] + 1e-9
  assert float(rp.min()) >= A.RATE_RANGE_RAD_S[0] - 1e-9
  assert float(rn.max()) <= A.RATE_RANGE_RAD_S[1] + 1e-9
  assert float(b.abs().max()) <= A.BIAS_RANGE_RAD[1] + 1e-9


def test_the_frozen_ranges_are_the_ones_the_plan_names():
  assert A.ALPHA_RANGE == (0.02, 1.0)
  assert A.RATE_RANGE_RAD_S == (0.05, 4.0)
  assert A.BIAS_RANGE_RAD == (-0.05, 0.05)
  assert A.FEATURE_DIM == 36
  assert sum(p.numel() for p in mk().parameters()) == 21168


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


def test_the_evaluator_actually_installs_every_hook_it_advertises():
  """A plumbing test, written after the plumbing silently did not happen.

  `scripts/ra_sim0_eval.py` grew a `--actuator` flag whose value reached the
  result file's provenance block but never reached `apply_actuator`, so the
  first run reported the nominal simulator's numbers under the model's name.
  The provenance said the model was installed and the simulator disagreed.
  """
  import ast
  import pathlib

  src = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "ra_sim0_eval.py"
  tree = ast.parse(src.read_text())
  build = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "build")
  called = {n.func.id for n in ast.walk(build)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
  args = {a.arg for a in build.args.kwonlyargs} | {a.arg for a in build.args.args}
  # every hook the signature accepts must have its installer called
  for arg, installer in (("actuator", "apply_actuator"),
                         ("residual", "apply_residual"),
                         ("hidden", "apply_hidden_plant")):
    assert arg in args
    assert installer in called, f"build() takes {arg} but never calls {installer}"


def test_every_pre_registered_stress_has_a_command_stream():
  """S5 had none, and the queue found out four runs in.

  The plan names five stresses; four of them are scripted command streams and
  the fifth reuses one.  A missing case is a stress that silently did not run,
  which is the same thing as a stress that passed.
  """
  import importlib.util
  import pathlib
  import torch

  path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "ra_sim1_stress.py"
  spec = importlib.util.spec_from_file_location("ra1_stress", path)
  m = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(m)
  for kind in ("S2", "S3", "S4", "S5"):
    a = m.scripted(kind, 4, 7, 32, "cpu", 1)
    assert a.shape == (32, 4, 7), kind
    assert torch.isfinite(a).all(), kind
    assert float(a.abs().max()) <= 1.0 + 1e-6, kind
  # S3 really does reverse every two steps
  a = m.scripted("S3", 2, 7, 12, "cpu", 1)
  flips = sum(1 for t in range(1, 12) if float(a[t, 0, 0]) != float(a[t - 1, 0, 0]))
  assert flips >= 5
