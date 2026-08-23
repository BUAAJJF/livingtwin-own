"""Observation latency: one implementation, and a distribution over it.

Two things have to hold before any WM1 number means anything.

* mjlab's ``DelayBuffer`` must serve the same sequence the Phase WM0 ring
  buffer served, or the gate thresholds inherited from WM0 -- 42.19 zero-shot,
  49.72 oracle, 47.46 to pass G2 -- are thresholds on a different experiment.
  ``test_matches_the_wm0_ring_buffer`` pins that, and
  ``results/wm1_latency/equivalence/`` carries the in-simulator re-measurement,
  because agreeing on a toy sequence is necessary and not sufficient.
* the per-environment lag must come from the prior that was asked for and
  must not change inside an episode.  A lag that silently resamples every step
  is observation *jitter*, which is a different mismatch with a different
  answer.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import math

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest
import torch
from mjlab.utils.buffers import DelayBuffer

from piper_push import latency


# ---------------------------------------------------------------------------
# One implementation
# ---------------------------------------------------------------------------


def _wm0_ring_buffer(frames: list[torch.Tensor], lag: int) -> list[torch.Tensor]:
  """The Phase WM0 implementation, verbatim, as the reference to match."""
  buf: list[torch.Tensor] = []
  out = []
  for obs in frames:
    if not lag:
      out.append(obs)
      continue
    while len(buf) < lag:
      buf.append(obs.clone())
    buf.append(obs.clone())
    out.append(buf.pop(0))
  return out


@pytest.mark.parametrize("lag", [1, 2, 3, 4])
def test_matches_the_wm0_ring_buffer(lag):
  frames = [torch.full((4, 3), float(t)) for t in range(20)]
  ref = _wm0_ring_buffer(frames, lag)

  buf = DelayBuffer(min_lag=lag, max_lag=lag, batch_size=4)
  got = []
  for obs in frames:
    buf.append(obs)
    got.append(buf.compute().clone())

  for t, (a, b) in enumerate(zip(ref, got)):
    assert torch.equal(a, b), f"step {t}: {a[0, 0]} vs {b[0, 0]}"


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_delayed_frame_is_the_one_from_lag_steps_ago(lag):
  """Stated independently of either implementation."""
  buf = DelayBuffer(min_lag=lag, max_lag=lag, batch_size=2)
  for t in range(12):
    buf.append(torch.full((2, 1), float(t)))
    assert buf.compute()[0, 0].item() == float(max(0, t - lag))


def test_reset_serves_fresh_frames_not_black_ones():
  """The one behaviour a hand-rolled buffer gets wrong.

  Zeros after a reset are a scene the policy has never seen, and it reacts to
  them; both implementations backfill instead.  They differ in the three-step
  transient afterwards -- mjlab ramps the lag up as history refills, the ring
  buffer held the reset frame -- which is why the empirical re-measurement in
  results/wm1_latency/equivalence/ exists rather than only this test.
  """
  buf = DelayBuffer(min_lag=3, max_lag=3, batch_size=2)
  for t in range(10):
    buf.append(torch.full((2, 1), float(t)))
    buf.compute()
  buf.reset(batch_ids=[1])
  buf.backfill(torch.full((2, 1), 99.0), torch.tensor([1]))
  served = buf.peek()
  assert served[1, 0].item() == 99.0     # reset env: the frame it just got
  assert served[0, 0].item() == 6.0      # the other env's timeline is untouched


def test_hold_prob_one_never_resamples():
  """What lets LatencyScene own the lags."""
  buf = DelayBuffer(min_lag=0, max_lag=4, batch_size=8, hold_prob=1.0)
  buf.set_lags(torch.tensor([3] * 8))
  for t in range(50):
    buf.append(torch.full((8, 1), float(t)))
    buf.compute()
    assert torch.equal(buf.current_lags, torch.tensor([3] * 8))


# ---------------------------------------------------------------------------
# The prior
# ---------------------------------------------------------------------------


def test_point_and_uniform():
  p = latency.LatencyPrior.point(3)
  assert p.argmax == 3
  assert p.mass(3) == 1.0
  assert p.entropy_bits == 0.0
  assert p.is_point_at(3)
  u = latency.LatencyPrior.uniform()
  assert u.entropy_bits == pytest.approx(math.log2(len(latency.LAGS)))


def test_probabilities_must_be_a_distribution():
  with pytest.raises(ValueError):
    latency.LatencyPrior((0.5, 0.2, 0.0, 0.0, 0.0))
  with pytest.raises(ValueError):
    latency.LatencyPrior((1.5, -0.5, 0.0, 0.0, 0.0))
  with pytest.raises(ValueError):
    latency.LatencyPrior((1.0, 0.0, 0.0))


def test_source_prior_is_a_point_mass_at_zero():
  """Named in the report as delta(0), so it had better be one.

  If the deployed policy had been trained under a latency range, retention
  would be a much weaker requirement and G4 would not mean what it says.
  """
  assert latency.P_SOURCE.is_point_at(0)


def test_mix_is_the_adaptation_distribution():
  q = latency.LatencyPrior.point(3)
  m = q.mix(latency.P_SOURCE, 0.75)
  assert m.mass(3) == pytest.approx(0.75)
  assert m.mass(0) == pytest.approx(0.25)
  assert q.mix(latency.P_SOURCE, 1.0).probs == q.probs
  with pytest.raises(ValueError):
    q.mix(latency.P_SOURCE, 1.5)


def test_from_scores_prefers_the_cheapest_candidate():
  """Scores are costs; the posterior has to fall the other way from logits."""
  q = latency.LatencyPrior.from_scores([5.0, 4.0, 3.0, 0.0, 6.0], temperature=1.0)
  assert q.argmax == 3
  hot = latency.LatencyPrior.from_scores([5.0, 4.0, 3.0, 0.0, 6.0], temperature=10.0)
  assert hot.entropy_bits > q.entropy_bits


def test_temperature_only_sharpens_it_does_not_move_the_mode():
  s = [2.0, 1.0, 0.5, 0.0, 3.0]
  modes = {latency.LatencyPrior.from_scores(s, temperature=t).argmax
           for t in (0.1, 1.0, 5.0)}
  assert modes == {3}


def test_fingerprint_deduplicates_identical_adaptation_runs():
  a = latency.LatencyPrior.from_scores([9, 9, 9, 0, 9], temperature=0.5)
  b = latency.LatencyPrior.point(3)
  assert a.fingerprint() == b.fingerprint()
  assert a.total_variation(b) < 1e-3


def test_sampling_follows_the_prior():
  q = latency.LatencyPrior((0.0, 0.0, 0.25, 0.75, 0.0))
  g = torch.Generator().manual_seed(0)
  s = q.sample(20000, g)
  assert set(s.tolist()) == {2, 3}
  assert (s == 3).float().mean().item() == pytest.approx(0.75, abs=0.02)


def test_max_lag_sizes_the_buffer():
  assert latency.LatencyPrior((0.5, 0.5, 0.0, 0.0, 0.0)).max_lag == 1
  assert latency.LatencyPrior.point(4).max_lag == 4


def test_json_reports_milliseconds():
  d = latency.LatencyPrior.point(3).to_json()
  assert d["mean_lag_ms"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Installing it
# ---------------------------------------------------------------------------


def _vision_cfg():
  from mjlab.tasks.registry import load_env_cfg

  return load_env_cfg("Mjlab-Pick-Place-PiperX-Vision", play=True)


def test_point_at_zero_leaves_the_config_alone():
  cfg = _vision_cfg()
  before = cfg.observations["camera"].terms["scene"]
  assert latency.apply_latency_prior(cfg, latency.P_SOURCE) == {}
  assert cfg.observations["camera"].terms["scene"] is before


def test_installing_a_prior_sets_the_native_delay_fields():
  cfg = _vision_cfg()
  q = latency.LatencyPrior((0.0, 0.0, 0.25, 0.75, 0.0))
  prov = latency.apply_latency_prior(cfg, q, seed=7)
  term = cfg.observations["camera"].terms["scene"]
  assert term.func is latency.LatencyScene
  assert (term.delay_min_lag, term.delay_max_lag) == (0, 3)
  assert term.delay_hold_prob == 1.0    # the term owns the lags
  assert term.delay_per_env is True
  assert term.params["latency_probs"] == q.probs
  assert prov["latency_prior"]["argmax"] == 3
  assert prov["latency_seed"] == 7
  # The camera term's own parameters survive: this is the same observation,
  # served late, not a different one.
  assert term.params["sensor_name"] == cfg.observations["camera"].terms[
    "scene"].params["sensor_name"]


def test_a_prior_with_no_delay_support_is_rejected_by_the_buffer_size():
  """max_lag == 0 would build no buffer at all, so it must not get that far."""
  cfg = _vision_cfg()
  assert latency.apply_latency_prior(cfg, latency.LatencyPrior.point(0)) == {}


def test_perturb_routes_obs_latency_to_the_native_fields():
  """One implementation in the tree, and this is the proof of it."""
  from piper_push import perturb

  cfg = _vision_cfg()
  perturb.apply_session_mismatch(cfg,
                                 perturb.SessionMismatchCfg(obs_latency_steps=3))
  term = cfg.observations["camera"].terms["scene"]
  assert (term.delay_min_lag, term.delay_max_lag) == (3, 3)
  assert not hasattr(perturb.PerturbedCameraScene, "_buf")


def test_args_round_trip():
  import argparse

  p = argparse.ArgumentParser()
  latency.add_latency_args(p)
  assert latency.prior_from_args(p.parse_args([])).is_point_at(0)
  assert latency.prior_from_args(p.parse_args(["--latency-lag", "3"])).argmax == 3
  q = latency.prior_from_args(p.parse_args(["--latency-probs", "0,0,1,3,0"]))
  assert q.mass(3) == pytest.approx(0.75)
