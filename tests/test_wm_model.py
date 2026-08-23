"""The head rebuilt from a checkpoint has to be the policy, and the dynamics
model has to start from the identity rather than the origin.

``ActorHead`` exists so that posterior inference needs no simulator: it is
reconstructed from ``actor_state_dict`` alone.  That is worth doing only if it
is provably the same function as the policy's own forward pass, so it is
checked against :func:`piper_push.wm_data.actor_head` on the real model class,
through a real checkpoint round-trip.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest
import torch
from tensordict import TensorDict

from piper_push import wm_data, wm_model
from piper_push.models import SpatialSoftmaxRecurrentModel
from piper_push.tasks.pick_place.rl_cfg import _CNN_CFG

N, PROPRIO_DIM, ACT_DIM = 4, 9, 7
IMG = (3, 24, 32)


@pytest.fixture(scope="module")
def policy():
  obs = TensorDict({"proprio": torch.randn(N, PROPRIO_DIM),
                    "camera": torch.rand(N, *IMG)}, batch_size=[N])
  torch.manual_seed(1)
  return SpatialSoftmaxRecurrentModel(
    obs=obs, obs_groups={"actor": ["proprio", "camera"]}, obs_set="actor",
    output_dim=ACT_DIM, cnn_cfg=_CNN_CFG, hidden_dims=(32, 32),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution",
                      "init_std": 0.6, "std_type": "scalar"},
    rnn_type="gru", rnn_hidden_dim=16, rnn_num_layers=1).eval()


@pytest.fixture(scope="module")
def checkpoint(policy, tmp_path_factory):
  p = tmp_path_factory.mktemp("ckpt") / "model.pt"
  torch.save({"actor_state_dict": policy.state_dict(), "iter": 0}, p)
  return p


# ---------------------------------------------------------------------------
# The actor head
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_head_from_checkpoint_is_the_policy(policy, checkpoint):
  head = wm_model.ActorHead.from_checkpoint(checkpoint)
  enc_dim = policy.rnn.rnn.input_size
  h = torch.zeros(1, N, policy.rnn.rnn.hidden_size)
  for t in range(5):
    enc = torch.randn(N, enc_dim, generator=torch.Generator().manual_seed(t))
    want, h_want = wm_data.actor_head(policy, enc, h)
    got, h_got = head(enc, h)
    assert torch.allclose(got, want, atol=1e-6), f"step {t}"
    assert torch.allclose(h_got, h_want, atol=1e-6)
    h = h_got


def test_head_rejects_a_non_recurrent_checkpoint(tmp_path):
  p = tmp_path / "state.pt"
  torch.save({"actor_state_dict": {"mlp.0.weight": torch.zeros(4, 4),
                                   "mlp.0.bias": torch.zeros(4)}}, p)
  with pytest.raises(ValueError, match="no recurrent layer"):
    wm_model.ActorHead.from_checkpoint(p)


def test_head_has_the_right_number_of_activations(policy, checkpoint):
  """One ELU between consecutive linear layers, none after the last.

  An activation on the output would squash the action; a missing one in the
  middle would make the whole head linear.  Both still run.
  """
  head = wm_model.ActorHead.from_checkpoint(checkpoint)
  kinds = [type(m).__name__ for m in head.mlp]
  assert kinds[-1] == "Linear"
  assert kinds == ["Linear", "ELU"] * (len(kinds) // 2) + ["Linear"]


# ---------------------------------------------------------------------------
# The dynamics model
# ---------------------------------------------------------------------------


def _model(z=6, p=4, a=3, n_theta=5):
  torch.manual_seed(0)
  return wm_model.LatentDynamics(z, p, a, n_theta, hidden=16, theta_dim=4)


def test_an_untrained_model_predicts_no_change():
  """Residual heads: the prior is 'nothing moved', not 'everything is zero'.

  With absolute heads the first thousand updates are spent learning to copy
  the input, and the multi-step loss is dominated by that instead of by the
  domain.
  """
  m = _model()
  m.z_head.weight.data.zero_(); m.z_head.bias.data.zero_()
  m.p_head.weight.data.zero_(); m.p_head.bias.data.zero_()
  z = torch.randn(2, 6)
  p = torch.randn(2, 4)
  (z_mu, _, dp, _, _, _), _ = m.step(z, p, torch.randn(2, 3),
                                     torch.zeros(2, dtype=torch.long), None)
  assert torch.allclose(z_mu, z)
  assert torch.allclose(dp, torch.zeros_like(p))


def test_the_domain_index_changes_the_prediction():
  """If theta did not enter, every candidate would score identically and the
  posterior would be the prior no matter what the data said."""
  m = _model()
  args = (torch.randn(2, 6), torch.randn(2, 4), torch.randn(2, 3))
  a, _ = m.step(*args, torch.zeros(2, dtype=torch.long), None)
  b, _ = m.step(*args, torch.full((2,), 3, dtype=torch.long), None)
  assert not torch.allclose(a[0], b[0])


def test_log_variance_is_clamped():
  m = _model()
  m.z_head.bias.data[6:] = 1e6
  (_, lv, _, _, _, _), _ = m.step(torch.randn(2, 6), torch.randn(2, 4),
                                  torch.randn(2, 3),
                                  torch.zeros(2, dtype=torch.long), None)
  assert float(lv.max()) == pytest.approx(wm_model.LOGVAR_MAX)


def test_open_loop_does_not_read_the_states_it_is_predicting():
  """The check that the multi-step loss is a multi-step loss.

  Perturbing the true states after step 0 must not change an open-loop
  rollout; if it does, the state is being fed back in and the model is being
  scored one step at a time under another name.
  """
  m = _model()
  z0, p0 = torch.randn(2, 6), torch.randn(2, 4)
  a = torch.randn(5, 2, 3)
  th = torch.full((2,), 2, dtype=torch.long)
  out1, _ = m.open_loop(z0, p0, a, th, None)
  out2, _ = m.open_loop(z0, p0, a, th, None)
  assert torch.allclose(out1[0], out2[0])
  # Teacher forcing with the same first state agrees on step 0 and, in
  # general, not afterwards.
  z = torch.cat([z0.unsqueeze(0), torch.randn(4, 2, 6)])
  p = torch.cat([p0.unsqueeze(0), torch.randn(4, 2, 4)])
  tf, _ = m.teacher_forced(z, p, a, th, None)
  assert torch.allclose(tf[0][0], out1[0][0], atol=1e-6)
  assert not torch.allclose(tf[0][-1], out1[0][-1], atol=1e-4)


def test_gaussian_nll_is_minimised_at_the_truth():
  x = torch.zeros(1)
  worse = wm_model.gaussian_nll(x, torch.ones(1), torch.zeros(1))
  best = wm_model.gaussian_nll(x, torch.zeros(1), torch.zeros(1))
  assert float(best) < float(worse)
  # A confident wrong prediction must cost more than an unconfident one.
  confident = wm_model.gaussian_nll(x, torch.ones(1), torch.full((1,), -4.0))
  assert float(confident) > float(worse)


# ---------------------------------------------------------------------------
# Normalisation and round-trip
# ---------------------------------------------------------------------------


def test_norm_standardises():
  x = torch.randn(100, 7) * 3.0 + 5.0
  n = wm_model.Norm.fit(x)
  y = n(x)
  assert float(y.mean().abs()) < 0.1
  assert float(y.std()) == pytest.approx(1.0, abs=0.1)


def test_norm_does_not_divide_by_a_constant_channel():
  x = torch.cat([torch.randn(100, 1), torch.full((100, 1), 2.0)], dim=1)
  y = wm_model.Norm.fit(x)(x)
  assert torch.isfinite(y).all()


def test_ensemble_round_trips(tmp_path):
  ms = [_model() for _ in range(3)]
  norms = {"z": wm_model.Norm.fit(torch.randn(50, 6)),
           "p": wm_model.Norm.fit(torch.randn(50, 4)),
           "a": wm_model.Norm.fit(torch.randn(50, 3))}
  e = wm_model.Ensemble(ms, norms)
  e.save(tmp_path / "e.pt")
  back = wm_model.Ensemble.load(tmp_path / "e.pt")
  assert len(back.members) == 3
  args = (torch.randn(2, 6), torch.randn(2, 4), torch.randn(2, 3),
          torch.zeros(2, dtype=torch.long), None)
  for m, b in zip(e.members, back.members):
    assert torch.allclose(m.step(*args)[0][0], b.step(*args)[0][0])
  assert torch.allclose(back.norms["z"].mean, norms["z"].mean)


def test_head_knows_where_the_image_encoding_starts(policy, checkpoint):
  """B1b regresses joint positions on the *image* half of the latent.

  Using the whole latent would fit perfectly at every candidate lag -- the
  first columns are the normalised proprioception the policy was handed -- and
  the comparison between candidates would measure nothing at all.
  """
  head = wm_model.ActorHead.from_checkpoint(checkpoint)
  assert head.obs_dim_1d == PROPRIO_DIM
  assert 0 < head.obs_dim_1d < policy.rnn.rnn.input_size
