"""Target identity on the deployment side.

The property that matters is one line long: while the jaws are closed on
something, no table instance may become the target.  Measured over 20 recorded
sessions that happened 326 times across 142 carries, and every one of them
handed the policy a confident mask of the wrong object.

The rest of these pin the edges where a lock is easy to get wrong: taken on
air, held past a release, or dropped on a single noisy frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hardware.deploy.lifecycle import LifecycleCfg, Phase, TargetLifecycle  # noqa: E402


def _grasp(lc: TargetLifecycle, label: int, n: int = 5) -> None:
  """Drive it into HELD the way a real grasp does."""
  lc.update(label, jaws_closed=False, loaded=False)
  for _ in range(n):
    lc.update(label, jaws_closed=True, loaded=True)


def test_a_bystander_cannot_take_over_while_holding():
  """The headline.  326 real occurrences say this needs a test."""
  lc = TargetLifecycle()
  _grasp(lc, 7)
  assert lc.holding
  for _ in range(200):
    out = lc.update(11, jaws_closed=True, loaded=True)  # a different instance
    assert out == 0, "a table label reached the policy during a carry"
  assert lc.locked == 7
  assert lc.swaps_while_held > 0, "the refusals must be counted, not hidden"


def test_the_approach_refuses_to_follow_a_different_instance():
  """``TargetTracker`` re-selects the nearest object once its own is stale.

  That is right for "what is on the table" and wrong for "which one am I
  dealing with", and the control loop is where the difference lives.
  """
  lc = TargetLifecycle()
  assert lc.update(3, jaws_closed=False, loaded=False) == 3
  assert lc.update(9, jaws_closed=False, loaded=False) == 0, "followed a swap"
  assert lc.update(3, jaws_closed=False, loaded=False) == 3, "lost its own"
  assert lc.refused == 1


def test_a_lock_is_not_taken_on_a_closed_but_unloaded_gripper():
  """Closing on nothing is not a grasp.

  The jaw gap alone cannot tell them apart -- ``proprio._contact_bit`` exists
  because the drive current can, with hysteresis, measured on this gripper.
  A lock taken on air would then refuse every real object that follows.
  """
  lc = TargetLifecycle()
  lc.update(5, jaws_closed=False, loaded=False)
  for _ in range(20):
    lc.update(0, jaws_closed=True, loaded=False)
  assert not lc.holding
  assert lc.update(5, jaws_closed=False, loaded=False) == 5


def test_a_carry_survives_a_single_dropped_contact_frame():
  """The effort latch dips mid-carry; a one-frame dip is not a release.

  Releasing on it would hand the very next frame back to the table tracker,
  which is the failure mode this class exists to remove.
  """
  lc = TargetLifecycle()
  _grasp(lc, 4)
  lc.update(0, jaws_closed=False, loaded=False)          # one bad frame
  assert lc.holding, "released on a single frame"
  for _ in range(10):
    lc.update(0, jaws_closed=True, loaded=True)
  assert lc.holding and lc.locked == 4


def test_opening_the_jaws_for_long_enough_ends_the_carry():
  lc = TargetLifecycle()
  _grasp(lc, 4)
  for _ in range(LifecycleCfg().release_frames + 1):
    lc.update(0, jaws_closed=False, loaded=False)
  assert lc.phase is Phase.SEARCH and lc.locked == 0
  assert lc.update(12, jaws_closed=False, loaded=False) == 12, \
    "a new instance must be selectable after a release"


def test_holding_never_shows_the_policy_a_table_label():
  """Even the locked instance's own label is withheld during a carry.

  The object is in the gripper, so a table detection of it is either the patch
  it was lifted from or a coincidence.  The caller rebuilds the mask at the
  grasp site instead, and it can only do that if this returns nothing.
  """
  lc = TargetLifecycle()
  _grasp(lc, 6)
  assert lc.update(6, jaws_closed=True, loaded=True) == 0


def test_release_resets_everything_a_placement_should_reset():
  """``run.py`` never called ``TargetTracker.clear()``; this is where it goes."""
  lc = TargetLifecycle()
  _grasp(lc, 2)
  lc.release()
  assert lc.phase is Phase.SEARCH and lc.locked == 0 and not lc.holding
  assert lc.update(8, jaws_closed=False, loaded=False) == 8
