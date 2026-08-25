"""The bookkeeping the accuracy gate rests on.

The replay itself needs a GPU and a simulator; these are the parts that decide
which of its numbers count, which is where a silent error would be worst -- a
mask that quietly keeps a segment straddling an episode boundary would make
every candidate look equally bad and the phase would report a null result it
never measured.
"""

from __future__ import annotations

import math

import torch

from piper_push import replay as rp


def make(T=12, E=2, done=None, shape=None, seed=0):
  g = torch.Generator().manual_seed(seed)
  z = lambda *s: torch.randn(*s, generator=g)
  return rp.Recording(
    q=z(T, E, 6), qd=z(T, E, 6), a=z(T, E, 7), gq=z(T, E, 1),
    obj=z(T, E, 13),
    done=torch.zeros(T, E, dtype=torch.bool) if done is None else done,
    shape=torch.zeros(T, E, dtype=torch.uint8) if shape is None else shape,
    u=z(T, E, 6), mode=torch.zeros(T, E, dtype=torch.uint8))


def test_a_clean_recording_keeps_every_step():
  rec = make()
  ok, hor = rp.segment_mask(rec, torch.zeros(12, 2, dtype=torch.bool), 4)
  assert bool(ok.all())
  assert list(hor[:, 0][:5]) == [1, 2, 3, 4, 1]


def test_a_reset_in_the_recording_kills_the_rest_of_its_segment():
  done = torch.zeros(12, 2, dtype=torch.bool)
  done[5, 0] = True
  rec = make(done=done)
  ok, _ = rp.segment_mask(rec, torch.zeros(12, 2, dtype=torch.bool), 4)
  # segment [4,5,6,7]: step 5 and everything after it in the segment go
  assert bool(ok[4, 0]) and not bool(ok[5, 0])
  assert not bool(ok[6, 0]) and not bool(ok[7, 0])
  # the next segment starts clean again
  assert bool(ok[8, 0])
  # the other environment is untouched
  assert bool(ok[:, 1].all())


def test_a_termination_in_the_candidate_kills_it_too():
  rec = make()
  cd = torch.zeros(12, 2, dtype=torch.bool)
  cd[2, 1] = True
  ok, _ = rp.segment_mask(rec, cd, 4)
  assert bool(ok[1, 1]) and not bool(ok[2, 1]) and not bool(ok[3, 1])
  assert bool(ok[:, 0].all())


def test_an_object_replacement_kills_its_segment():
  shape = torch.zeros(12, 2, dtype=torch.uint8)
  shape[7:, 0] = 3
  rec = make(shape=shape)
  ok, _ = rp.segment_mask(rec, torch.zeros(12, 2, dtype=torch.bool), 4)
  # the change is between step 6 and step 7, so the prediction at 6 is spoilt
  assert not bool(ok[6, 0]) and not bool(ok[7, 0])
  assert bool(ok[5, 0])


def test_nrms_is_one_when_the_candidate_predicts_no_motion():
  rec = make(T=6, E=3)
  ok, hor = rp.segment_mask(rec, torch.zeros(6, 3, dtype=torch.bool), 1)
  frozen = rec.q[:-1]  # predicting "the state did not change"
  out = rp.nrms(frozen, rec, ok, hor, 1, 1, "q")
  assert out["nrms"] == 0 or abs(out["nrms"] - 1.0) < 1e-6
  assert out["n"] == 5 * 3


def test_nrms_is_zero_for_a_perfect_candidate():
  rec = make(T=6, E=3)
  ok, hor = rp.segment_mask(rec, torch.zeros(6, 3, dtype=torch.bool), 1)
  out = rp.nrms(rec.q[1:], rec, ok, hor, 1, 1, "q")
  assert out["rms"] < 1e-6
  assert out["nrms"] < 1e-6


def test_nrms_returns_nan_rather_than_a_number_when_nothing_is_usable():
  rec = make(T=6, E=2, done=torch.ones(6, 2, dtype=torch.bool))
  ok, hor = rp.segment_mask(rec, torch.zeros(6, 2, dtype=torch.bool), 1)
  out = rp.nrms(rec.q[1:], rec, ok, hor, 1, 1, "q")
  assert out["n"] == 0 and math.isnan(out["nrms"])


def test_the_horizon_of_a_step_is_its_distance_from_the_anchor():
  rec = make(T=30, E=1)
  ok, hor = rp.segment_mask(rec, torch.zeros(30, 1, dtype=torch.bool), 25)
  assert int(hor[0, 0]) == 1 and int(hor[24, 0]) == 25 and int(hor[25, 0]) == 1


def test_a_multi_step_nrms_divides_by_the_motion_over_that_horizon():
  """The normaliser is the motion since the anchor, not since the last step.

  Getting this wrong would make the 25-step number look 25 times better than
  it is, which is exactly the direction that would flatter the phase.
  """
  T, E = 26, 1
  rec = make(T=T, E=E)
  # a ramp, so the motion over h steps is exactly h times the step
  rec.q[:] = torch.arange(T, dtype=torch.float32).view(T, 1, 1)
  ok, hor = rp.segment_mask(rec, torch.zeros(T, E, dtype=torch.bool), 25)
  # "no motion since the anchor" -- the baseline the normaliser is defined
  # against.  Note this is NOT rec.q[:-1], which is a one-step-frozen
  # predictor and gets *better* as the horizon lengthens.
  anchor = (torch.arange(T - 1) // 25) * 25
  frozen = rec.q[anchor]
  h1 = rp.nrms(frozen, rec, ok, hor, 25, 1, "q")
  h25 = rp.nrms(frozen, rec, ok, hor, 25, 25, "q")
  assert abs(h1["rms_motion"] - 1.0) < 1e-6
  assert abs(h25["rms_motion"] - 25.0) < 1e-6
  assert abs(h1["nrms"] - 1.0) < 1e-6
  assert abs(h25["nrms"] - 1.0) < 1e-6


def test_a_reversal_is_a_sign_change_in_the_command():
  rec = make(T=10, E=1)
  rec.u[:] = 0.0
  rec.u[:5, 0, 0] = torch.arange(5, dtype=torch.float32) * 0.1
  rec.u[5:, 0, 0] = 0.3 - torch.arange(5, dtype=torch.float32) * 0.1
  flip = rp.reversal_mask(rec, lookback=1)
  # du[3] is the last step up and du[4] the first step down, so the flip is
  # recorded at 4.  A plateau between them would not be a reversal, which is
  # the case the sign product gets right and a difference of differences
  # would not.
  assert bool(flip[4, 0, 0])
  assert not bool(flip[2, 0, 0])


def test_the_deadband_mask_is_the_small_commanded_steps():
  rec = make(T=5, E=1)
  rec.u[:] = 0.0
  rec.u[1, 0, 0] = 0.001
  rec.u[2, 0, 0] = 0.5
  m = rp.deadband_mask(rec, width=0.004)
  assert bool(m[0, 0, 0])
  assert not bool(m[1, 0, 0])
