"""Which instance the robot is working on, as state the robot owns.

The segmenter answers "what is on the table"; the policy's mask channel has to
answer "which one am I dealing with".  Those are different questions, and
until now nothing on the deployment side held the difference: ``TargetTracker``
re-selects the instance nearest the hand once its old one has been missing for
``lost_frames``, and the control loop never told it when a placement finished.

That is not a rare failure.  Measured over 20 recorded sessions
(``scripts/measure_target_gaps.py``): **326 identity swaps while the jaws were
closed, across 142 closed-jaw spans** -- about 2.3 per carry.  A swap while
holding something is always wrong.  The object in the gripper cannot have
become a different object, and the mask that says otherwise is confident,
plausible, and points at the table.

So this is the small amount of state that makes the question answerable:

    SEARCH   -- nothing locked.  Take whatever the tracker offers.
    APPROACH -- locked to one label.  A different label is REFUSED, not
                followed; the tracker may only re-confirm the one we chose.
    HELD     -- the jaws are closed and loaded.  The mask comes from the grasp
                site, unconditionally, and no table instance can take over.
    (release) -- jaws open again: forget the instance and go back to SEARCH.

``HELD`` is entered on the jaw gap AND the effort latch, not on the gap alone.
``proprio.ProprioBuilder._contact_bit`` already applies hysteresis to the
drive current, measured on this gripper: at the 0.20 threshold an empty
gripper produced runs of median 2 control steps and a loaded one runs of
median 31.  The gap alone cannot tell "closed on an object" from "closed on
nothing", and closing on nothing is how a lock gets taken on air.
"""

from __future__ import annotations

import dataclasses
import enum


class Phase(enum.Enum):
  SEARCH = "search"
  APPROACH = "approach"
  HELD = "held"


@dataclasses.dataclass
class LifecycleCfg:
  release_frames: int = 5
  """Consecutive open-and-unloaded frames before a carry is called over.

  Not one: the effort latch dips through a carry, and releasing the lock on a
  single dip would hand the next frame back to the table tracker, which is the
  failure this exists to stop.  Five control steps is 100 ms.
  """

  grasp_frames: int = 3
  """Consecutive closed-and-loaded frames before a carry is called started.

  Symmetric with ``release_frames`` and for the same reason: the drive spikes
  on contact before it settles, and a single loaded frame during the approach
  is a squeeze, not a grasp.
  """


class TargetLifecycle:
  """Per-run target identity.  One instance, no history beyond the counters."""

  def __init__(self, cfg: LifecycleCfg | None = None) -> None:
    self.cfg = cfg or LifecycleCfg()
    self.phase = Phase.SEARCH
    self.locked: int = 0
    self._closed_for = 0
    self._open_for = 0
    self.refused = 0
    """Labels rejected because something else was already locked.

    Reported rather than silently dropped: a run with a high count is a run
    where the tracker kept trying to change its mind, which is worth seeing
    even when the lock did its job."""
    self.swaps_while_held = 0
    """Should be zero.  It is the number this class exists to drive to zero,
    so it is counted rather than assumed."""

  def update(self, label: int, jaws_closed: bool, loaded: bool) -> int:
    """Advance one control step; return the label the policy should be shown.

    ``label`` is what ``TargetTracker`` chose this frame -- 0 for nothing.
    The returned label is 0 whenever the mask should not come from the table,
    which includes every frame of a carry: there the caller rebuilds the mask
    at the grasp site instead, and a table label would be a bystander.
    """
    held_now = jaws_closed and loaded
    self._closed_for = self._closed_for + 1 if held_now else 0
    self._open_for = 0 if (jaws_closed or loaded) else self._open_for + 1

    if self.phase is Phase.HELD:
      if label and self.locked and label != self.locked:
        # Not followed -- counted.  The tracker offering a different instance
        # mid-carry is exactly the event this class refuses.
        self.swaps_while_held += 1
        self.refused += 1
      if self._open_for >= self.cfg.release_frames:
        self.phase = Phase.SEARCH
        self.locked = 0
        return 0
      return 0            # the caller rebuilds the mask at the grasp site

    if self._closed_for >= self.cfg.grasp_frames and self.locked:
      self.phase = Phase.HELD
      return 0

    if not label:
      return 0

    if self.phase is Phase.SEARCH:
      self.locked = int(label)
      self.phase = Phase.APPROACH
      return self.locked

    # APPROACH: only the instance we already chose.
    if int(label) == self.locked:
      return self.locked
    self.refused += 1
    return 0

  def release(self) -> None:
    """Called when a placement completes, or on any deliberate reset.

    The tracker's own ``clear()`` belongs here too; ``run.py`` never called it,
    which is why a finished placement left the old centroid in place.
    """
    self.phase = Phase.SEARCH
    self.locked = 0
    self._closed_for = 0
    self._open_for = 0

  @property
  def holding(self) -> bool:
    return self.phase is Phase.HELD

  def summary(self) -> str:
    return (f"lifecycle: phase={self.phase.value} locked={self.locked} "
            f"refused={self.refused} swaps_while_held={self.swaps_while_held}")
