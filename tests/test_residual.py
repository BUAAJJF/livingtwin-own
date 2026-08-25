"""The learned correction, its bounds, and the rule about what it may see."""

from __future__ import annotations

import inspect

import pytest
import torch

from piper_push import residual as R
from piper_push.surrogate import TransitionSurrogate, fit_surrogate


class FakeTerm:
  def __init__(self, n_env: int = 4, n_joint: int = 6):
    self.device = "cpu"
    self._num_targets = n_joint
    self._default = torch.zeros(n_env, n_joint)
    self._previous_target = self._default.clone()
    self._q = torch.zeros(n_env, n_joint)
    self._qd = torch.zeros(n_env, n_joint)

  @property
  def joint_pos(self):
    return self._q

  @property
  def joint_vel(self):
    return self._qd


def test_an_untrained_residual_is_exactly_the_identity():
  net = R.ResidualNet()
  feat = torch.randn(7, R.FEATURE_DIM)
  h = net.zero_hidden(7, "cpu")
  d, _ = net(feat, h)
  assert float(d.abs().max()) == 0.0


def test_the_ensemble_is_identity_too_and_reports_zero_spread():
  ens = R.ResidualEnsemble(4)
  feat = torch.randn(7, R.FEATURE_DIM)
  hs = [m.zero_hidden(7, "cpu") for m in ens.members]
  mean, spread, hs2 = ens(feat, hs)
  assert float(mean.abs().max()) == 0.0
  assert float(spread.abs().max()) == 0.0
  assert len(hs2) == 4


def test_the_output_is_bounded_however_hard_the_head_is_pushed():
  net = R.ResidualNet()
  with torch.no_grad():
    net.head.weight.normal_(0.0, 50.0)
    net.head.bias.normal_(0.0, 50.0)
  d, _ = net(torch.randn(64, R.FEATURE_DIM) * 20.0, net.zero_hidden(64, "cpu"))
  assert float(d.abs().max()) <= R.DELTA_MAX + 1e-6
  # And the bound is reached, so it is a bound and not a decoration.
  assert float(d.abs().max()) > 0.9 * R.DELTA_MAX


def test_the_feature_vector_is_the_documented_concatenation():
  q = torch.arange(6.0).view(1, 6)
  qd = q + 10
  u = q + 20
  up = q + 30
  f = R.build_features(q, qd, u, up)
  assert f.shape == (1, R.FEATURE_DIM)
  assert torch.allclose(f[0, :6], q[0])
  assert torch.allclose(f[0, 6:12], qd[0])
  assert torch.allclose(f[0, 12:18], u[0])
  assert torch.allclose(f[0, 18:24], up[0])
  assert torch.allclose(f[0, 24:], q[0] - u[0])


def test_the_feature_builder_cannot_be_handed_a_hidden_quantity():
  """The phase's central prohibition, enforced by the signature.

  ``build_features`` takes four tensors and there is no fifth argument to
  smuggle the plant's flank state, the target's effective command, the trip
  label or the reward through.  Every consumer -- the offline trainer and the
  in-simulator hook -- goes through this one function.
  """
  params = list(inspect.signature(R.build_features).parameters)
  assert params == ["q", "qd", "u", "u_prev"]


def test_hidden_state_and_command_history_reset_per_environment():
  term = FakeTerm(n_env=4)
  ens = R.ResidualEnsemble(2)
  with torch.no_grad():
    for m in ens.members:
      m.head.weight.normal_(0.0, 1.0)
      m.head.bias.normal_(0.0, 1.0)
  blob = {"n_members": 2, "hidden": 64, "delta_max": R.DELTA_MAX,
          "state_dict": ens.state_dict()}
  import tempfile, pathlib
  with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "res.pt"
    torch.save(blob, p)
    hook = R.ResidualHookCfg(checkpoint=str(p)).build(term)
  target = torch.full((4, 6), 0.3)
  for _ in range(4):
    hook(target, term)
  keep = [h[2:].clone() for h in hook._h]
  hook.reset(torch.tensor([0, 1]))
  for h, k in zip(hook._h, keep):
    assert torch.allclose(h[2:], k, atol=0.0)
    assert float(h[:2].abs().max()) == 0.0
  assert torch.allclose(hook._u_prev[:2], term._previous_target[:2])


def test_a_zero_scale_hook_returns_the_command_untouched():
  term = FakeTerm()
  ens = R.ResidualEnsemble(2)
  with torch.no_grad():
    for m in ens.members:
      m.head.bias.fill_(3.0)
  import tempfile, pathlib
  with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "res.pt"
    torch.save({"n_members": 2, "hidden": 64, "delta_max": R.DELTA_MAX,
                "state_dict": ens.state_dict()}, p)
    on = R.ResidualHookCfg(checkpoint=str(p)).build(FakeTerm())
    off = R.ResidualHookCfg(checkpoint=str(p), scale=0.0).build(term)
  t = torch.full((4, 6), 0.25)
  assert torch.allclose(off(t, term), t, atol=0.0)
  assert not torch.allclose(on(t, FakeTerm()), t, atol=1e-6)


def test_a_residual_refuses_an_action_term_of_the_wrong_width():
  import tempfile, pathlib
  ens = R.ResidualEnsemble(1)
  with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "res.pt"
    torch.save({"n_members": 1, "hidden": 64, "delta_max": R.DELTA_MAX,
                "state_dict": ens.state_dict()}, p)
    with pytest.raises(ValueError):
      R.ResidualHookCfg(checkpoint=str(p)).build(FakeTerm(n_joint=1))


def test_one_member_stays_under_the_size_the_plan_committed_to():
  net = R.ResidualNet()
  assert sum(p.numel() for p in net.parameters()) < 100_000
  assert sum(p.numel() for p in R.ResidualEnsemble(4).parameters()) < 100_000


def test_the_surrogate_predicts_a_change_and_starts_near_identity():
  m = TransitionSurrogate()
  q = torch.randn(5, 6) * 0.3
  qd = torch.randn(5, 6) * 0.1
  nq, nqd = m(q, qd, q, q)
  assert nq.shape == q.shape and nqd.shape == qd.shape
  assert torch.isfinite(nq).all()


def test_the_surrogate_learns_a_plant_it_is_given():
  """A linear servo the surrogate has to recover, so a fit that returns the
  mean is distinguishable from a fit that works."""
  torch.manual_seed(0)
  n = 20000
  q = torch.randn(n, 6) * 0.4
  qd = torch.randn(n, 6) * 0.5
  yp = q + torch.randn(n, 6) * 0.05
  y = q + torch.randn(n, 6) * 0.05
  dq = 0.02 * qd + 0.3 * (y - q)
  dqd = -0.4 * qd + 8.0 * (y - q)
  feats = torch.cat([q, qd, yp, y], dim=-1)
  labels = torch.cat([dq, dqd], dim=-1)
  model, info = fit_surrogate(feats, labels, epochs=60, batch=512, log=None)
  assert info["val_nmse"] < 0.02
  for p in model.parameters():
    assert not p.requires_grad
