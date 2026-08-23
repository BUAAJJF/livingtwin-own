"""The recorded log has to be replayable, and it has to stay deployable.

Two claims the world-model stage rests on:

* ``S_action`` compares ``pi(z_real)`` with ``pi(z_pred)``.  That is only the
  policy's own decision if :func:`wm_data.actor_head` is the policy's own
  forward pass with the convolutional stage lifted out.  "It is the same code"
  is true right up until somebody edits one of the two, so it is checked
  against the real model class, built the way the task builds it.
* nothing simulator-only can reach an estimator by accident.  The channel
  whitelist is enforced by a function the loaders call, not by a comment.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest
import torch
from tensordict import TensorDict

from piper_push import wm_data
from piper_push.models import SpatialSoftmaxRecurrentModel
from piper_push.tasks.pick_place.rl_cfg import _CNN_CFG

N, PROPRIO_DIM, ACT_DIM = 4, 9, 7
IMG = (3, 24, 32)


@pytest.fixture(scope="module")
def policy():
  obs = TensorDict({
    "proprio": torch.randn(N, PROPRIO_DIM),
    "camera": torch.rand(N, *IMG),
  }, batch_size=[N])
  torch.manual_seed(0)
  return SpatialSoftmaxRecurrentModel(
    obs=obs,
    obs_groups={"actor": ["proprio", "camera"]},
    obs_set="actor",
    output_dim=ACT_DIM,
    cnn_cfg=_CNN_CFG,
    hidden_dims=(32, 32),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution",
                      "init_std": 0.6, "std_type": "scalar"},
    rnn_type="gru",
    rnn_hidden_dim=16,
    rnn_num_layers=1,
  ).eval()


def _obs(seed: int) -> TensorDict:
  g = torch.Generator().manual_seed(seed)
  return TensorDict({
    "proprio": torch.randn(N, PROPRIO_DIM, generator=g),
    "camera": torch.rand(N, *IMG, generator=g),
  }, batch_size=[N])


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_actor_head_reproduces_the_policy(policy):
  """Encode, then head, must equal calling the model."""
  policy.reset()
  want = []
  for t in range(6):
    want.append(policy(_obs(t)).clone())

  policy.reset()
  h = wm_data.zero_hidden(policy, N, torch.device("cpu"))
  for t in range(6):
    enc = policy._encode(_obs(t))
    got, h = wm_data.actor_head(policy, enc, h)
    assert torch.allclose(got, want[t], atol=1e-6), f"step {t}"


@torch.no_grad()
def test_hidden_state_recorded_is_the_one_the_action_came_from(policy):
  """The pair (enc_t, hidden_t) has to be sufficient to reproduce a_t.

  Recording the hidden state *after* the step instead is the easy mistake, and
  it is invisible: the log still looks right, but every replayed action is one
  step stale and S_action measures the shift rather than the mismatch.
  """
  policy.reset()
  logged = []
  for t in range(5):
    h_prev = policy.rnn.hidden_state
    if h_prev is None:
      h_prev = wm_data.zero_hidden(policy, N, torch.device("cpu"))
    enc = policy._encode(_obs(t))
    act, h_new = wm_data.actor_head(policy, enc, h_prev)
    policy.rnn.hidden_state = h_new
    logged.append((enc.clone(), h_prev.clone(), act.clone()))

  for t, (enc, h_prev, act) in enumerate(logged):
    replay, _ = wm_data.actor_head(policy, enc, h_prev)
    assert torch.allclose(replay, act, atol=1e-6), f"step {t}"


@torch.no_grad()
def test_zero_hidden_matches_a_fresh_policy(policy):
  policy.reset()
  a = policy(_obs(0))
  policy.reset()
  b, _ = wm_data.actor_head(policy, policy._encode(_obs(0)),
                            wm_data.zero_hidden(policy, N, torch.device("cpu")))
  assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_the_label_is_not_a_channel():
  wm_data.assert_deployable(["enc", "proprio", "action"])
  with pytest.raises(ValueError, match="simulator label"):
    wm_data.assert_deployable(["enc", "lag"])
  with pytest.raises(ValueError, match="simulator label"):
    wm_data.assert_deployable(["shape"])


def test_unknown_channels_are_rejected():
  """A typo must not silently become an empty feature."""
  with pytest.raises(ValueError, match="unknown"):
    wm_data.assert_deployable(["enc", "reward"])
  with pytest.raises(ValueError, match="unknown"):
    wm_data.assert_deployable(["success"])


def test_no_privileged_channel_is_even_nameable():
  for name in ("reward", "success", "object_pose", "mass", "friction",
               "contact", "trips", "value"):
    assert name not in wm_data.DEPLOYABLE
    with pytest.raises(ValueError):
      wm_data.assert_deployable([name])


def test_every_channel_is_documented():
  for name in wm_data.DEPLOYABLE + wm_data.SIM_ONLY:
    assert wm_data.CHANNEL_DOC.get(name), name


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def _session(steps=40, n=2, done_at=None) -> wm_data.SessionSet:
  done = torch.zeros(steps, n, dtype=torch.bool)
  if done_at is not None:
    done[done_at, :] = True
  return wm_data.SessionSet(
    enc=torch.zeros(steps, n, 5, dtype=torch.half),
    hidden=torch.zeros(steps, n, 4, dtype=torch.half),
    proprio=torch.zeros(steps, n, 13),
    action=torch.zeros(steps, n, 7),
    servo=torch.zeros(steps, n, 1),
    done=done, shape=torch.zeros(steps, n, dtype=torch.uint8), lag=3)


def test_windows_cover_the_session():
  w = wm_data.windows(_session(), length=16, burn_in=8, stride=8)
  assert len(w) == 2 * len(range(0, 40 - 24 + 1, 8))
  assert all(0 <= t0 <= 16 for t0, _ in w)


def test_windows_do_not_straddle_an_episode_boundary():
  """A reset zeroes the policy's memory; a window across one is fitting that.

  Both the world model and every trajectory-matching score would otherwise be
  scored partly on how sharply the hidden state drops to zero, which happens
  identically in every domain.
  """
  s = _session(done_at=20)
  w = wm_data.windows(s, length=16, burn_in=8, stride=1)
  assert all(not bool(s.done[t0:t0 + 24 - 1, b].any()) for t0, b in w)
  assert wm_data.windows(s, length=16, burn_in=8, stride=1,
                         cross_episode=True) != w


def test_windows_can_be_restricted_to_named_environments():
  """Sessions are the split unit; a window must not cross into a held-out one."""
  w = wm_data.windows(_session(n=4), length=16, burn_in=8, stride=8,
                      envs=torch.tensor([1, 3]))
  assert {b for _, b in w} == {1, 3}


def test_describe_reports_arm_seconds_not_wall_clock():
  """The data budget in the report is arm-seconds: envs x steps / 50 Hz."""
  d = _session(steps=500, n=6).describe()
  assert d["arm_seconds"] == pytest.approx(60.0)


def test_zero_hidden_follows_the_policys_weights(policy):
  """Not the observation's device.

  mjlab's TensorDict reports device=None with its entries on the GPU, so
  deriving the hidden state's device from the observation silently puts it on
  the CPU and the first forward pass dies on a device mismatch -- four minutes
  into a rollout, not in any config check.
  """
  h = wm_data.zero_hidden(policy, N)
  assert h.device == next(policy.rnn.rnn.parameters()).device
  assert h.shape == (policy.rnn.rnn.num_layers, N, policy.rnn.rnn.hidden_size)


def test_session_view_puts_every_channel_on_one_device():
  """The done flag included.

  It is the only channel that is not a float, so it is the one a helper that
  converts floats can silently leave behind -- and the done-only control
  classifier then gets a CPU tensor and a GPU model.
  """
  from piper_push import wm_data as wd
  from piper_push import wm_infer

  s = wd.SessionSet(
    enc=torch.zeros(20, 3, 5, dtype=torch.half),
    hidden=torch.zeros(20, 3, 4, dtype=torch.half),
    proprio=torch.zeros(20, 3, 13), action=torch.zeros(20, 3, 7),
    servo=torch.zeros(20, 3, 1), done=torch.zeros(20, 3, dtype=torch.bool),
    shape=torch.zeros(20, 3, dtype=torch.uint8), lag=3)
  v = wm_infer.SessionView.from_session(s, 1, 0.2, device="cpu")
  devices = {v.enc.device, v.hidden.device, v.proprio.device,
             v.action.device, v.servo.device, v.done.device}
  assert len(devices) == 1
  assert v.enc.dtype == torch.float32       # promoted out of fp16 storage
