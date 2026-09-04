"""The SAM2.1 adapter's logic, without a GPU.

TwinSight measured causal SAM2.1 at 85.1% correct target against the depth
detector's 35.1% on the same reviewed frames -- and at 13.0% *wrong* target,
where the depth detector is 0.0%.  A confident mask of the wrong object is the
one failure a policy cannot defend against, because nothing in its input
distinguishes it from a correct one.

So every test here is about the watchdog refusing, not about SAM succeeding.
The predictor is faked: what needs testing is when this file declines to
publish, and that must be checkable on a laptop with no model loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hardware.deploy.sam_tracker import (SamTargetTracker, State,  # noqa: E402
                                         WatchdogCfg)
from hardware.deploy.sam2_predictor import Sam2StreamingPredictor  # noqa: E402

H = W = 128
"""Big enough that a 60 px jump fits inside it.

At 64 the farthest two points are 89 px apart and a blob near the corner is
clipped, so its centroid moves back toward the middle: the "jump" this file
is about could not be expressed and the test passed a mask it meant to
reject.  A frame the failure does not fit in is not a test of the failure.
"""


def blob(cx, cy, r=8, h=H, w=W):
  ys, xs = np.mgrid[0:h, 0:w]
  return ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r


class FakePredictor:
  """Returns whatever the test tells it to, in order."""

  def __init__(self, outputs):
    self.outputs = list(outputs)
    self.anchored = 0
    self.resets = 0

  def anchor(self, image, mask):
    self.anchored += 1

  def propagate(self, image):
    return self.outputs.pop(0) if self.outputs else None

  def reset(self):
    self.resets += 1

class ReanchorPredictor(FakePredictor):
  def __init__(self, outputs):
    super().__init__(outputs)
    self.reanchored = 0

  def reanchor(self, mask):
    self.reanchored += 1



def _tracker(outputs, cfg=None):
  return SamTargetTracker(FakePredictor(outputs), cfg)


IMG = np.zeros((H, W), np.uint8)
ALL_DEPTH = np.ones((H, W), bool)


def test_nothing_is_published_before_the_depth_pipeline_has_chosen():
  """SAM may not introduce a target.  It carries one it was handed."""
  t = _tracker([blob(32, 32)])
  r = t.step(IMG, ALL_DEPTH)
  assert r.mask is None and r.state is State.UNINITIALIZED


def test_a_stable_target_is_carried():
  t = _tracker([blob(32, 32), blob(33, 32), blob(34, 33)])
  t.anchor(IMG, blob(32, 32))
  for _ in range(3):
    r = t.step(IMG, ALL_DEPTH)
    assert r.state is State.TRACKING and r.mask is not None


def test_a_mask_that_jumps_across_the_frame_is_refused():
  """The 13% failure, in its most recognisable form.

  At 30 Hz an object the arm is reaching for does not move 60 px between
  frames.  A tracker that does has changed subject, and the mask it returns is
  as confident as a correct one.
  """
  t = _tracker([blob(40, 40), blob(112, 112)])
  t.anchor(IMG, blob(40, 40))
  assert t.step(IMG, ALL_DEPTH).state is State.TRACKING
  r = t.step(IMG, ALL_DEPTH)
  assert r.mask is None and r.state is State.UNCERTAIN
  assert "jumped" in r.reason


def test_a_mask_that_swallows_the_scene_is_refused():
  t = _tracker([np.ones((H, W), bool)])
  t.anchor(IMG, blob(32, 32, r=4))
  r = t.step(IMG, ALL_DEPTH)
  assert r.mask is None and r.reason.startswith("area")


def test_a_mask_with_no_depth_under_it_is_refused():
  """SAM works on intensity and will outline a shadow.

  The policy's channels are depth; a mask over nothing measurable is not a
  target, however crisp its outline.
  """
  t = _tracker([blob(32, 32)])
  t.anchor(IMG, blob(32, 32))
  r = t.step(IMG, np.zeros((H, W), bool))
  assert r.mask is None and r.reason.startswith("depth")


def test_a_mask_that_is_mostly_gripper_is_refused():
  """The drift this pipeline invites: it anchors near the hand.

  Nothing else in the stack would notice -- the arm is the largest thing in
  frame and it moves with the object.
  """
  t = _tracker([blob(32, 32)])
  t.anchor(IMG, blob(32, 32))
  r = t.step(IMG, ALL_DEPTH, arm_mask=blob(32, 32, r=8))
  assert r.mask is None and r.reason.startswith("arm")


def test_one_bad_frame_is_not_a_lost_target_but_five_are():
  """A single rejection is noise; a run of them is a subject change.

  Losing the lock instantly would thrash; never losing it means publishing
  whatever SAM drifted onto.
  """
  cfg = WatchdogCfg(uncertain_before_lost=5)
  t = _tracker([None] * 6, cfg)
  t.anchor(IMG, blob(32, 32))
  for i in range(4):
    assert t.step(IMG, ALL_DEPTH).state is State.UNCERTAIN, f"strike {i}"
  assert t.step(IMG, ALL_DEPTH).state is State.LOST


def test_an_uncertain_frame_publishes_nothing():
  """Withheld, not passed through with a flag.

  The policy has no channel for "this mask is doubtful" -- it sees a mask or
  it does not -- so a doubtful mask handed over is indistinguishable from a
  good one.
  """
  t = _tracker([blob(112, 112)])
  t.anchor(IMG, blob(40, 40))
  r = t.step(IMG, ALL_DEPTH)
  assert r.state is State.UNCERTAIN and r.mask is None


def test_reset_clears_the_state_the_lifecycle_owns():
  """Placement, episode reset and the start of a carry all land here."""
  t = _tracker([blob(32, 32)])
  t.anchor(IMG, blob(32, 32))
  t.reset("placed")
  assert t.state is State.UNINITIALIZED
  assert t.p.resets == 1
  assert t.step(IMG, ALL_DEPTH).mask is None


def test_an_empty_anchor_is_ignored_rather_than_locked_onto_nothing():
  t = _tracker([])
  t.anchor(IMG, np.zeros((H, W), bool))
  assert t.state is State.UNINITIALIZED and t.p.anchored == 0


def test_rejections_are_counted_so_thresholds_can_be_calibrated():
  """The handoff requires the thresholds be fitted to labels, not taste.

  That is only possible if the reasons are recorded in a live run.
  """
  t = _tracker([blob(112, 112), None])
  t.anchor(IMG, blob(40, 40))
  t.step(IMG, ALL_DEPTH)
  t.step(IMG, ALL_DEPTH)
  assert t.rejections.get("jumped") == 1
  assert t.rejections.get("empty") == 1


def test_the_jump_budget_grows_with_the_gap_it_spans():
  """One rejection must not lock the tracker out for the rest of the episode.

  ``last_centre`` only advances on an accepted mask, so after a rejection the
  reference is frozen while the object keeps moving.  With a fixed threshold
  every later frame looks like a bigger jump and the test latches -- measured
  against the renderer it was discarding masks at IoU 0.76-0.95 and reporting
  108 px "jumps" on an object sitting in the gripper.  The displacement is per
  frame, so the budget over k frames is k times as large.
  """
  cfg = WatchdogCfg(max_step_px=20.0, uncertain_before_lost=99)
  # frame 1 teleports (rejected); frames 2-4 walk back at 15 px/frame, which
  # is inside budget once the two-, three- and four-frame gaps are allowed for.
  t = _tracker([blob(100, 40), blob(70, 40), blob(55, 40), blob(45, 40)], cfg)
  t.anchor(IMG, blob(40, 40))
  assert t.step(IMG, ALL_DEPTH).state is State.UNCERTAIN   # 60 px in one frame
  # 30 px after two frames: inside 2 x 20, so the target is re-acquired.
  r = t.step(IMG, ALL_DEPTH)
  assert r.state is State.TRACKING and r.mask is not None
  assert t.step(IMG, ALL_DEPTH).state is State.TRACKING


def test_a_sustained_wrong_target_is_still_refused_as_the_budget_grows():
  """The growing budget must not become no budget at all.

  A tracker sitting on the wrong object does not drift back; it stays there,
  and the gap to the last accepted centre keeps growing faster than the
  allowance.  This is the case the relaxation above must not break.
  """
  cfg = WatchdogCfg(max_step_px=20.0, uncertain_before_lost=5)
  t = _tracker([blob(40 + 45 * k, 40) for k in range(1, 6)], cfg)
  t.anchor(IMG, blob(40, 40))
  for _ in range(4):
    assert t.step(IMG, ALL_DEPTH).mask is None
  assert t.step(IMG, ALL_DEPTH).state is State.LOST


def test_a_withheld_mask_is_still_reported_so_the_watchdog_can_be_scored():
  """The rejection count cannot say whether the rejection was right.

  ``raw`` is never for the policy -- ``mask`` stays None -- but without it
  there is no way to ask "were the masks we threw away correct?", which is the
  measurement that showed both thresholds above were wrong.
  """
  t = _tracker([blob(112, 112)])
  t.anchor(IMG, blob(40, 40))
  r = t.step(IMG, ALL_DEPTH)
  assert r.mask is None
  assert r.raw is not None and r.raw.sum() > 0


# -- the anchor policy, which is what makes this runnable on a robot --------


def test_carry_never_starts_a_target_the_depth_stack_has_not_confirmed():
  """The rule the whole design rests on.  SAM is very good at returning a
  mask; it is 13% likely to be the wrong object, and nothing downstream can
  tell.  So with no depth mask and no anchor there is no target."""
  t = _tracker([blob(32, 32)])
  r = t.carry(IMG, np.zeros((H, W), bool), ALL_DEPTH)
  assert r.mask is None and t.p.anchored == 0


def test_carry_anchors_from_the_depth_mask_and_publishes_it_that_frame():
  t = _tracker([blob(32, 32)])
  m = blob(32, 32)
  r = t.carry(IMG, m, ALL_DEPTH)
  assert t.p.anchored == 1 and r.mask is not None
  assert np.array_equal(r.mask, m), "on the anchor frame they are the same mask"


def test_carry_keeps_propagating_while_the_depth_stack_has_nothing():
  """The gap the tracker exists for: the segmenter loses the object as the
  hand arrives -- 7% of frames within 30 mm, measured -- and SAM carries it."""
  t = _tracker([blob(33, 32), blob(34, 33), blob(35, 34)])
  t.carry(IMG, blob(32, 32), ALL_DEPTH)
  for _ in range(3):
    r = t.carry(IMG, np.zeros((H, W), bool), ALL_DEPTH)
    assert r.mask is not None and r.state is State.TRACKING
  assert t.p.anchored == 1, "no depth mask, no new anchor"


def test_carry_does_not_re_anchor_before_the_frame_budget():
  """Re-anchoring every frame the depth stack agrees would make SAM a
  redundant copy of the segmenter, including its failures."""
  t = _tracker([blob(32, 32)] * 4)
  m = blob(32, 32)
  t.carry(IMG, m, ALL_DEPTH)
  for _ in range(3):
    t.carry(IMG, m, ALL_DEPTH, reanchor_frames=30)
  assert t.p.anchored == 1


def test_carry_re_anchors_once_the_two_agree_after_the_budget():
  t = _tracker([blob(32, 32)] * 4)
  m = blob(32, 32)
  t.carry(IMG, m, ALL_DEPTH)
  for _ in range(3):
    t.carry(IMG, m, ALL_DEPTH, reanchor_frames=2, reanchor_iou=0.5)
  assert t.p.anchored == 2


def test_carry_does_not_re_anchor_when_the_two_disagree():
  """Disagreement is the case a refresh must NOT resolve by taking the newer
  mask: it is exactly as likely to be the segmenter picking up a bystander."""
  t = _tracker([blob(32, 32)] * 4)
  t.carry(IMG, blob(32, 32), ALL_DEPTH)
  for _ in range(3):
    t.carry(IMG, blob(100, 100, r=6), ALL_DEPTH, reanchor_frames=1,
            reanchor_iou=0.5)
  assert t.p.anchored == 1


def test_carry_recovers_from_lost_when_the_depth_stack_offers_a_target():
  """LOST must not be terminal, or one bad run of frames ends the session."""
  cfg = WatchdogCfg(uncertain_before_lost=2)
  t = _tracker([None, None, blob(32, 32)], cfg)
  t.carry(IMG, blob(32, 32), ALL_DEPTH)
  t.carry(IMG, np.zeros((H, W), bool), ALL_DEPTH)
  t.carry(IMG, np.zeros((H, W), bool), ALL_DEPTH)
  assert t.state is State.LOST
  r = t.carry(IMG, blob(32, 32), ALL_DEPTH)
  assert t.p.anchored == 2 and r.mask is not None


def test_carry_counts_the_frames_it_rescued_from_an_empty_depth_mask():
  """"SAM ran" is not the question.  On a recording there is no truth to score
  against, so the only evidence the integration did anything is how often it
  published a mask on a frame where the depth stack had none."""
  t = _tracker([blob(33, 32), blob(34, 33)])
  t.carry(IMG, blob(32, 32), ALL_DEPTH)          # anchor, depth had a mask
  t.carry(IMG, np.zeros((H, W), bool), ALL_DEPTH)   # rescued
  t.carry(IMG, blob(34, 33), ALL_DEPTH)             # both had one
  assert t.frames == 3 and t.published == 3 and t.rescued == 1


def test_a_withheld_frame_is_not_counted_as_published():
  cfg = WatchdogCfg(uncertain_before_lost=9)
  t = _tracker([blob(112, 112)], cfg)
  t.carry(IMG, blob(40, 40), ALL_DEPTH)
  t.carry(IMG, np.zeros((H, W), bool), ALL_DEPTH)
  assert t.frames == 2 and t.published == 1 and t.rescued == 0


def test_summary_and_report_are_both_available_after_a_run():
  t = _tracker([])
  assert "state=uninitialized" in t.summary()
  assert t.report()["state"] == "uninitialized"


def test_refresh_corrects_the_current_frame_without_pushing_it_twice():
  p = ReanchorPredictor([blob(32, 32)])
  t = SamTargetTracker(p)
  m = blob(32, 32)
  t.carry(IMG, m, ALL_DEPTH)
  t.carry(IMG, m, ALL_DEPTH, reanchor_frames=1)
  assert p.anchored == 1
  assert p.reanchored == 1


def test_anchor_trimming_releases_full_resolution_prompt_tensors():
  pred = Sam2StreamingPredictor.__new__(Sam2StreamingPredictor)
  pred.max_anchors = 3
  cond = {i: object() for i in range(6)}
  pred.state = {
    "output_dict_per_obj": {
      0: {"cond_frame_outputs": dict(cond)}},
    "temp_output_dict_per_obj": {
      0: {"cond_frame_outputs": {}}},
    "mask_inputs_per_obj": {
      0: dict(cond)},
    "point_inputs_per_obj": {
      0: dict(cond)},
  }
  pred._trim_anchors()
  kept = {3, 4, 5}
  assert set(pred.state["output_dict_per_obj"][0]["cond_frame_outputs"]) == kept
  assert set(pred.state["mask_inputs_per_obj"][0]) == kept
  assert set(pred.state["point_inputs_per_obj"][0]) == kept
