"""Servo damping as a per-environment domain parameter.

Phase WM1-A's axis reached the simulator through an observation term; this one
reaches it through a per-world field of the compiled model, which is a
different failure surface. The claims that matter:

* the shared categorical algebra behaves the same for three values as it did
  for five, including the mixing that the adaptation distribution is built
  from;
* a point mass at nominal leaves the task config untouched, so every earlier
  result stays reproducible;
* the write scales the derivative gain and not the proportional one, and is
  relative to the default field rather than the current one, so repeated resets
  do not compound.

The in-simulator counterpart -- that the per-environment event and Phase WM0's
config-level scaling produce the same model field -- is
`scripts/check_damping.py`, because no unit test can see the compiled model.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import math

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest
import torch

from piper_push import damping, prior


# ---------------------------------------------------------------------------
# The candidate set
# ---------------------------------------------------------------------------


def test_the_candidate_set_is_target_nominal_and_counter_direction():
  """Three values, and the third is not decoration.

  Without a candidate on the *other* side of nominal, a method could score by
  always answering "less damped than nominal" and nothing in the result would
  show it.
  """
  assert damping.VALUES == (0.75, 1.0, 1.5)
  assert damping.TARGET < damping.NOMINAL < damping.COUNTER


def test_the_source_prior_is_a_point_mass_at_nominal():
  assert damping.P_SOURCE.is_point_at(damping.NOMINAL)
  assert damping.P_SOURCE.mass(damping.TARGET) == 0.0


# ---------------------------------------------------------------------------
# The shared algebra, on three values instead of five
# ---------------------------------------------------------------------------


def test_point_and_uniform():
  p = damping.DampingPrior.point(0.75)
  assert p.argmax == 0.75
  assert p.entropy_bits == 0.0
  u = damping.DampingPrior.uniform()
  assert u.entropy_bits == pytest.approx(math.log2(3))
  assert u.mean == pytest.approx((0.75 + 1.0 + 1.5) / 3)


def test_probabilities_must_be_a_distribution():
  with pytest.raises(ValueError):
    damping.DampingPrior((0.5, 0.2, 0.0))
  with pytest.raises(ValueError):
    damping.DampingPrior((1.0, 0.0))


def test_mix_is_the_adaptation_distribution():
  q = damping.DampingPrior.point(0.75)
  m = q.mix(damping.P_SOURCE, 0.75)
  assert m.mass(0.75) == pytest.approx(0.75)
  assert m.mass(1.0) == pytest.approx(0.25)
  assert m.mass(1.5) == 0.0


def test_from_scores_prefers_the_cheapest_candidate():
  q = damping.DampingPrior.from_scores([0.0, 5.0, 9.0], temperature=1.0)
  assert q.argmax == 0.75
  assert damping.DampingPrior.from_scores([9.0, 0.0, 5.0]).argmax == 1.0


def test_fingerprint_deduplicates_identical_adaptation_runs():
  a = damping.DampingPrior.from_scores([0.0, 40.0, 40.0], temperature=0.5)
  assert a.fingerprint() == damping.DampingPrior.point(0.75).fingerprint()


def test_sampling_follows_the_prior():
  q = damping.DampingPrior((0.2, 0.3, 0.5))
  g = torch.Generator().manual_seed(0)
  s = q.sample(30000, g)
  for v, want in zip(damping.VALUES, q.probs):
    assert (s == v).float().mean().item() == pytest.approx(want, abs=0.02)


def test_json_names_the_axis_units():
  d = damping.DampingPrior.point(0.75).to_json()
  assert d["values"] == [0.75, 1.0, 1.5]
  assert "kd" in d["unit"]
  assert d["argmax"] == 0.75


# ---------------------------------------------------------------------------
# Installing it
# ---------------------------------------------------------------------------


def _vision_cfg():
  from mjlab.tasks.registry import load_env_cfg

  return load_env_cfg("Mjlab-Pick-Place-PiperX-Vision", play=True)


def test_a_point_mass_at_nominal_leaves_the_config_alone():
  cfg = _vision_cfg()
  before = set(cfg.events)
  assert damping.apply_damping_prior(cfg, damping.P_SOURCE) == {}
  assert set(cfg.events) == before


def test_installing_a_prior_adds_one_reset_event():
  cfg = _vision_cfg()
  q = damping.DampingPrior((0.5, 0.5, 0.0))
  prov = damping.apply_damping_prior(cfg, q, seed=3)
  ev = cfg.events[damping.EVENT_NAME]
  assert ev.mode == "reset"
  assert ev.func is damping.randomize_servo_damping
  assert ev.params["probs"] == q.probs
  assert ev.params["skip_gripper"] is True
  assert prov["damping_prior"]["argmax"] in (0.75, 1.0)
  assert prov["damping_seed"] == 3


# ---------------------------------------------------------------------------
# The write itself
# ---------------------------------------------------------------------------


class _Actuator:
  def __init__(self, names, ctrl_ids):
    self.cfg = type("C", (), {"target_names_expr": names})()
    self.global_ctrl_ids = torch.tensor(ctrl_ids)


class _Model:
  def __init__(self, n_env, n_ctrl):
    # biasprm is (0, -kp, -kd) per actuator, per world.
    self.actuator_biasprm = torch.zeros(n_env, n_ctrl, 3)
    self.actuator_biasprm[..., 1] = -80.0
    self.actuator_biasprm[..., 2] = -5.0


class _Env:
  def __init__(self, n_env=8, n_ctrl=4):
    self.num_envs = n_env
    self.device = "cpu"
    self.actuators = [_Actuator(("joint[1-3]",), [0, 1, 2]),
                      _Actuator(("gripper_joint1",), [3])]
    robot = type("R", (), {})()
    robot.actuators = self.actuators
    self.scene = {"robot": robot}
    self.sim = type("S", (), {})()
    self.sim.model = _Model(n_env, n_ctrl)
    self._default = self.sim.model.actuator_biasprm[0].clone()
    self.sim.get_default_field = lambda name: self._default


def _run(env, probs, env_ids=None):
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  damping.randomize_servo_damping(env, env_ids, probs, damping.VALUES,
                                  SceneEntityCfg("robot"), True)


def test_the_write_scales_kd_and_leaves_kp_alone():
  env = _Env()
  _run(env, (1.0, 0.0, 0.0))
  kd = -env.sim.model.actuator_biasprm[:, :3, 2]
  kp = -env.sim.model.actuator_biasprm[:, :3, 1]
  assert torch.allclose(kd, torch.full_like(kd, 5.0 * 0.75))
  assert torch.allclose(kp, torch.full_like(kp, 80.0))


def test_the_gripper_is_left_alone():
  """Phase WM0's axis excluded it, so this one must too or the two are not the
  same domain."""
  env = _Env()
  _run(env, (1.0, 0.0, 0.0))
  assert torch.allclose(env.sim.model.actuator_biasprm[:, 3, 2],
                        torch.full((8,), -5.0))


def test_repeated_resets_do_not_compound():
  """The write is relative to the default field, not the current one.  Scaling
  the current value every episode would drive kd to zero over a rollout and
  the domain would not be the one that was asked for."""
  env = _Env()
  for _ in range(5):
    _run(env, (1.0, 0.0, 0.0))
  kd = -env.sim.model.actuator_biasprm[:, :3, 2]
  assert torch.allclose(kd, torch.full_like(kd, 5.0 * 0.75))


def test_a_mixture_gives_different_environments_different_damping():
  env = _Env(n_env=2048)
  _run(env, (0.5, 0.5, 0.0))
  kd = -env.sim.model.actuator_biasprm[:, 0, 2]
  frac = (kd < 5.0 * 0.9).float().mean().item()
  assert frac == pytest.approx(0.5, abs=0.05)
  assert set(torch.unique(kd).tolist()) == {5.0 * 0.75, 5.0}


def test_a_partial_reset_leaves_the_other_environments_untouched():
  env = _Env(n_env=16)
  _run(env, (0.0, 0.0, 1.0))                      # everything to 1.5x
  _run(env, (1.0, 0.0, 0.0), torch.arange(0, 8))  # the first half to 0.75x
  kd = -env.sim.model.actuator_biasprm[:, 0, 2]
  assert torch.allclose(kd[:8], torch.full((8,), 5.0 * 0.75))
  assert torch.allclose(kd[8:], torch.full((8,), 5.0 * 1.5))


# ---------------------------------------------------------------------------
# The risk-averse tilt
# ---------------------------------------------------------------------------


def test_the_tilt_is_the_identity_at_zero_lambda():
  q = damping.DampingPrior((0.2, 0.5, 0.3))
  out = q.tilt(damping.trip_costs(), 0.0)
  assert out.probs == pytest.approx(q.probs, abs=1e-12)


def test_the_tilt_moves_mass_towards_the_dangerous_candidate():
  """0.75 trips 74 times as often as nominal, so any positive lambda has to
  favour it -- that is the whole mechanism."""
  q = damping.DampingPrior((1 / 3, 1 / 3, 1 / 3))
  out = q.tilt(damping.trip_costs(), 0.01)
  assert out.mass(damping.TARGET) > q.mass(damping.TARGET)
  assert out.mass(damping.COUNTER) < q.mass(damping.COUNTER)
  assert out.probs[0] > out.probs[1] > out.probs[2]


def test_a_point_mass_cannot_be_tilted():
  """The limitation the report has to state.  With no surviving candidate to
  move mass towards, risk-awareness has nothing to do -- so it can only help a
  posterior that is genuinely uncertain."""
  q = damping.DampingPrior.point(damping.NOMINAL)
  for lam in (0.0, 0.01, 1.0):
    assert q.tilt(damping.trip_costs(), lam).probs == pytest.approx(q.probs)


def test_a_large_lambda_concentrates_on_the_most_dangerous_survivor():
  q = damping.DampingPrior((0.001, 0.5, 0.499))
  out = q.tilt(damping.trip_costs(), 1.0)
  assert out.mass(damping.TARGET) > 0.999


def test_a_candidate_the_posterior_excluded_stays_excluded():
  """The tilt reweights; it does not resurrect.  A candidate ruled out by the
  data must not come back because it is dangerous."""
  q = damping.DampingPrior((0.0, 0.5, 0.5))
  out = q.tilt(damping.trip_costs(), 0.05)
  assert out.mass(damping.TARGET) == 0.0


def test_the_tilt_never_underflows_silently():
  q = damping.DampingPrior((0.0, 0.5, 0.5))
  with pytest.raises(ValueError, match="underflowed"):
    q.tilt(damping.trip_costs(), 1e4)


def test_a_negative_lambda_is_rejected():
  q = damping.DampingPrior.uniform()
  with pytest.raises(ValueError, match="non-negative"):
    q.tilt(damping.trip_costs(), -1.0)


def test_a_mismatched_cost_vector_is_rejected():
  q = damping.DampingPrior.uniform()
  with pytest.raises(ValueError, match="expected"):
    q.tilt((1.0, 2.0), 0.1)


def test_the_costs_match_what_the_dataset_measured():
  c = damping.trip_costs()
  assert len(c) == len(damping.VALUES)
  assert c[damping.VALUES.index(damping.TARGET)] > 100.0
  assert c[damping.VALUES.index(damping.COUNTER)] < c[
    damping.VALUES.index(damping.NOMINAL)]


def test_the_lambda_rule_gives_the_odds_it_promises():
  """Nine to one between the extremes of the cost range, from an undecided
  starting point -- that is the whole definition, so it is what gets pinned."""
  c = damping.trip_costs()
  lam = prior.risk_lambda(c)
  hi, lo = max(c), min(c)
  assert math.exp(lam * (hi - lo)) == pytest.approx(9.0)

  two = damping.DampingPrior((0.5, 0.0, 0.5))     # target against counter
  out = two.tilt(c, lam)
  assert out.probs[0] / out.probs[2] == pytest.approx(9.0)


def test_the_lambda_rule_is_scale_free():
  """Costs in trips per hour and trips per minute must give the same tilt."""
  c = damping.trip_costs()
  per_min = tuple(x / 60.0 for x in c)
  q = damping.DampingPrior.uniform()
  a = q.tilt(c, prior.risk_lambda(c))
  b = q.tilt(per_min, prior.risk_lambda(per_min))
  assert a.probs == pytest.approx(b.probs, abs=1e-12)


def test_the_lambda_rule_is_zero_when_every_candidate_costs_the_same():
  assert prior.risk_lambda((3.0, 3.0, 3.0)) == 0.0


def test_the_tilt_at_the_registered_lambda_does_what_the_report_will_claim():
  """Concrete numbers, so a change to the rule cannot slip past the report."""
  c = damping.trip_costs()
  lam = prior.risk_lambda(c)
  u = damping.DampingPrior.uniform().tilt(c, lam)
  assert u.mass(damping.TARGET) == pytest.approx(0.816, abs=0.002)
  assert u.mass(damping.NOMINAL) == pytest.approx(0.093, abs=0.002)
