"""SAM2.1 as a causal tracker for the target the depth segmenter already chose.

Not as a segmenter.  The distinction is the whole design and it comes from
TwinSight's audit (``results/twin/audit/SAM21_INTEGRATION_HANDOFF.md``), which
measured both on the same reviewed frames:

    variant                    empty mask    wrong target    correct
    post-fix depth detector        64.9%            0.0%       35.1%
    causal SAM2.1-small             1.95%          13.0%       85.1%

SAM2.1 almost never returns nothing, and that is exactly what the depth
detector is worst at.  But 13% wrong-target is a failure the depth detector
does not have, and a confident mask of the wrong object is worse for a policy
than no mask -- it has no way to tell them apart.  So the two are combined in
the direction that plays to each: **the depth segmenter and ``TargetTracker``
choose the instance; SAM2.1 only carries that choice between frames.**

Three rules follow, and they are the reason this file exists rather than a
call to SAM inside ``run.py``:

* **SAM never picks.**  It is anchored from a mask the existing pipeline
  produced, and re-anchored when the two agree.  It cannot introduce a target
  the rest of the stack has not confirmed.
* **A watchdog, because 13% is not rare.**  Area ratio against the anchor,
  centre displacement per frame, valid-depth fraction, and overlap with the
  projected arm.  On sustained disagreement the tracker says ``UNCERTAIN`` and
  then ``LOST`` rather than continuing to publish a mask nobody checked.
* **It stands down for a carry.**  Once the lifecycle says the object is held,
  the geometric reconstruction at the grasp site is more trustworthy than a
  visible-only mask of an object inside a gripper -- the handoff says so and
  the measurement agrees: the fingers own most of what is visible there.  That
  is an explicit state transition, not a blend.

The predictor is injected rather than constructed here.  SAM2's video API is
stateful and awkward to drive frame-by-frame, and the logic above -- which is
where the failures will be -- must be testable without a GPU.  ``FakePredictor``
in the tests exercises every path in this file.
"""

from __future__ import annotations

import dataclasses
import enum
import pathlib
from typing import Protocol

import numpy as np

CHECKPOINT = pathlib.Path(__file__).parent / "sam2_assets" / "sam2.1_hiera_small.pt"
"""Pinned by hash in ``sam2_assets/PROVENANCE.json``.  The audit ran from
``/tmp``; the handoff says a real integration may not."""


class State(enum.Enum):
  UNINITIALIZED = "uninitialized"
  TRACKING = "tracking"
  UNCERTAIN = "uncertain"
  LOST = "lost"


class Predictor(Protocol):
  """The two calls this needs from a SAM2 video predictor."""

  def anchor(self, image: np.ndarray, mask: np.ndarray) -> None: ...
  def propagate(self, image: np.ndarray) -> np.ndarray | None: ...
  def reset(self) -> None: ...


@dataclasses.dataclass
class WatchdogCfg:
  """Thresholds for calling a SAM mask untrustworthy.

  Starting values, deliberately loose, and every rejection is counted so they
  can be tightened from paired labels rather than taste.  The handoff is
  explicit that the audit's 13% wrong-target rate makes a watchdog mandatory
  and that its thresholds must be calibrated, not assumed.
  """

  area_ratio: tuple[float, float] = (0.25, 4.0)
  """Mask area against the anchor's.  A target that quadruples has grown into
  the arm or the table; one that quarters is a fragment."""

  max_step_px: float = 60.0
  """Centre displacement **per frame** since the last accepted mask.  At 30 Hz
  an object the robot is reaching for does not cross 60 px in a frame; a
  tracker that jumps has changed subject.

  Per frame, not per comparison, and the difference is not pedantry -- it was
  a bug.  ``last_centre`` only advances on an accepted mask, so after one
  rejection the reference is frozen while the object keeps moving, every
  subsequent frame looks like a larger jump, and the test latches: it rejects
  for the rest of the episode because it rejected once.  Measured against the
  renderer, the masks it was throwing away had IoU 0.76-0.95 with the true
  target and the reported jumps were 108 and 115 px at 30 Hz on an object in
  the gripper.  The budget therefore grows with the gap it is spanning."""

  min_depth_fraction: float = 0.30
  """Fraction of the mask with valid depth.  SAM works on intensity and will
  happily outline a shadow, which the policy cannot use: its channels are
  depth."""

  max_arm_fraction: float = 0.50
  """How much of the mask may be the robot.  Above this it is tracking the
  arm, which is the specific drift this pipeline invites.

  What counts as "the robot" matters more than the threshold.  The caller must
  pass a mask of the *arm*, not of the hand: the sphere cover the segmenter
  uses puts a 50 mm sphere on ``gripper_base`` and 31 mm on each finger, so an
  object being grasped is inside it by construction.  Handed that mask, this
  check fires hardest exactly when the tracker is most useful -- 0.88 and 0.94
  arm fraction on masks measured at IoU 0.86 and 0.91 against the renderer.
  ``scripts/sim_perception_check.sensor_arm_mask`` drops spheres near the grasp
  site for this reason; it is the same mistake the occlusion metric made when
  it reported 97.3% blocked against a pixel truth of 6.2%."""

  uncertain_before_lost: int = 5
  """Consecutive rejected frames before the mask is withdrawn.  One bad frame
  is noise; five is a subject change."""


@dataclasses.dataclass
class Report:
  mask: np.ndarray | None
  """What may be published.  ``None`` whenever the watchdog is not satisfied."""
  state: State
  reason: str = ""
  frames_since_anchor: int = 0
  area_px: int = 0
  raw: np.ndarray | None = None
  """What SAM actually returned, published or not.

  Never for the policy -- it is the withheld mask, which is the whole point of
  withholding it.  It is here so an offline check can ask the one question the
  rejection count cannot answer: was the rejected mask correct?  A watchdog
  tuned without that measurement is tuned by taste, which the handoff
  explicitly forbids."""


class SamTargetTracker:
  """Carry one already-chosen target between frames, and know when it has lost it."""

  def __init__(self, predictor: Predictor, cfg: WatchdogCfg | None = None) -> None:
    self.p = predictor
    self.cfg = cfg or WatchdogCfg()
    self.state = State.UNINITIALIZED
    self.anchor_area = 0
    self.anchor_centre: np.ndarray | None = None
    self.last_centre: np.ndarray | None = None
    self.frames_since_anchor = 0
    self._bad = 0
    self.rejections: dict[str, int] = {}
    self.anchors = 0
    self.frames = 0
    self.published = 0
    self.rescued = 0
    """Frames where SAM published a mask and the depth stack had none.

    The number the whole integration is for.  Without it a replay can only say
    that SAM ran -- ``anchors=3, rejections=7`` says nothing about whether the
    policy saw its target more often, and on a recording the renderer's truth
    does not exist to compare against."""

  # -- lifecycle ----------------------------------------------------------

  def reset(self, reason: str = "") -> None:
    """Forget the target.  Called on place, on episode reset, on a carry."""
    self.p.reset()
    self.state = State.UNINITIALIZED
    self.anchor_area = 0
    self.anchor_centre = None
    self.last_centre = None
    self.frames_since_anchor = 0
    self._bad = 0

  def anchor(self, image: np.ndarray, mask: np.ndarray) -> None:
    """Take the depth pipeline's mask as the definition of the target."""
    if not mask.any():
      return
    self.p.anchor(image, mask)
    self._set_anchor(mask)

  def _set_anchor(self, mask: np.ndarray) -> None:
    self.anchor_area = int(mask.sum())
    self.anchor_centre = _centre(mask)
    self.last_centre = self.anchor_centre
    self.state = State.TRACKING
    self.frames_since_anchor = 0
    self._bad = 0
    self.anchors += 1

  # -- one frame ----------------------------------------------------------

  def step(self, image: np.ndarray, depth_valid: np.ndarray,
           arm_mask: np.ndarray | None = None) -> Report:
    if self.state in (State.UNINITIALIZED, State.LOST):
      # LOST is not a state to propagate out of.  Left to keep running, the
      # tracker recovers the moment SAM's mask happens to pass the watchdog
      # again -- on whatever SAM drifted onto during the run of rejections,
      # which is precisely the confident-wrong-object case the watchdog is
      # there to stop.  Recovery has to come from a fresh anchor the depth
      # stack confirmed, so it waits.
      return Report(None, self.state,
                    "no anchor" if self.state is State.UNINITIALIZED
                    else "lost, awaiting re-anchor")

    mask = self.p.propagate(image)
    self.frames_since_anchor += 1
    if mask is None or not mask.any():
      return self._strike("empty", mask)

    why = self._check(mask, depth_valid, arm_mask)
    if why:
      return self._strike(why, mask)

    self._bad = 0
    self.state = State.TRACKING
    self.last_centre = _centre(mask)
    return Report(mask, self.state, "", self.frames_since_anchor,
                  int(mask.sum()), raw=mask)

  # -- one frame, with the anchor policy ----------------------------------

  def _count(self, published, depth_mask) -> None:
    self.frames += 1
    if published is not None and published.any():
      self.published += 1
      if depth_mask is None or not depth_mask.any():
        self.rescued += 1

  def carry(self, image: np.ndarray, depth_mask: np.ndarray,
            depth_valid: np.ndarray, arm_mask: np.ndarray | None = None,
            reanchor_frames: int = 30, reanchor_iou: float = 0.5) -> Report:
    """Advance the target and re-anchor it, in the one order that is safe.

    ``step`` alone is not enough to run this on a robot: something has to
    decide when SAM is handed a fresh definition of the target, and that
    decision is where a tracker either stays on the object or quietly adopts
    the arm.  Two rules, and no third:

    * **SAM never picks.**  An anchor comes only from a mask the depth
      segmenter and ``TargetTracker`` produced.  With no depth mask there is
      no anchor, whatever SAM thinks.
    * **A refresh needs agreement.**  Once tracking, the anchor is replaced
      only when the depth stack independently produces a mask that overlaps
      SAM's by ``reanchor_iou`` and at least ``reanchor_frames`` have passed.
      That is the one moment a new anchor is known not to be drift.

    Returned ``mask`` is what may be published: SAM's, or -- on the frame an
    anchor is taken -- the depth mask that defined it, because on that frame
    they are the same thing by construction.

    This lives here rather than in the caller because it has two callers that
    must not differ: ``run.py`` on the arm and ``scripts/sim_perception_check``
    against the renderer, which is the only place either can be scored.
    """
    rep = self.step(image, depth_valid, arm_mask)
    published = rep.mask
    if depth_mask is not None and depth_mask.any():
      stale = (rep.frames_since_anchor >= reanchor_frames
               and published is not None
               and _iou(published, depth_mask) >= reanchor_iou)
      if self.state in (State.UNINITIALIZED, State.LOST) or stale:
        refresh = getattr(self.p, "reanchor", None)
        if stale and callable(refresh):
          # ``step`` already pushed and encoded this image.  Correct that
          # frame in place instead of inserting a duplicate temporal frame.
          refresh(depth_mask)
          self._set_anchor(depth_mask)
        else:
          self.anchor(image, depth_mask)
        self._count(depth_mask, depth_mask)
        return Report(depth_mask, self.state, "anchored", 0,
                      int(depth_mask.sum()), raw=rep.raw)
    self._count(rep.mask, depth_mask)
    return rep

  def _check(self, mask, depth_valid, arm_mask) -> str:
    c = self.cfg
    area = int(mask.sum())
    if self.anchor_area:
      r = area / self.anchor_area
      if not (c.area_ratio[0] <= r <= c.area_ratio[1]):
        return f"area x{r:.2f}"
    centre = _centre(mask)
    if self.last_centre is not None:
      step = float(np.linalg.norm(centre - self.last_centre))
      # ``last_centre`` is as many frames old as there have been consecutive
      # rejections, so the budget is per frame, not per comparison.
      budget = c.max_step_px * (1 + self._bad)
      if step > budget:
        return f"jumped {step:.0f}px"
    frac = float((mask & depth_valid).sum()) / max(area, 1)
    if frac < c.min_depth_fraction:
      return f"depth {frac:.2f}"
    if arm_mask is not None:
      arm = float((mask & arm_mask).sum()) / max(area, 1)
      if arm > c.max_arm_fraction:
        return f"arm {arm:.2f}"
    return ""

  def _strike(self, why: str, raw=None) -> Report:
    self.rejections[why.split()[0]] = self.rejections.get(why.split()[0], 0) + 1
    self._bad += 1
    if self._bad >= self.cfg.uncertain_before_lost:
      self.state = State.LOST
      return Report(None, self.state, why, self.frames_since_anchor, raw=raw)
    self.state = State.UNCERTAIN
    # Withhold the mask while uncertain.  Publishing it "just in case" is how
    # a confident wrong mask reaches a policy that cannot question it.
    return Report(None, self.state, why, self.frames_since_anchor, raw=raw)

  def report(self) -> dict:
    """Machine-readable tracker and predictor state for ``run.json``."""
    out = {
      "state": self.state.value,
      "anchors": int(self.anchors),
      "frames": int(self.frames),
      "published": int(self.published),
      "rescued": int(self.rescued),
      "rejections": dict(self.rejections),
    }
    predictor_report = getattr(self.p, "report", None)
    if callable(predictor_report):
      out["predictor"] = predictor_report()
    return out

  def summary(self) -> str:
    bad = "  ".join(f"{k}={v}" for k, v in sorted(self.rejections.items()))
    n = max(self.frames, 1)
    return (f"sam: state={self.state.value} anchors={self.anchors} "
            f"published {self.published}/{self.frames} "
            f"({100 * self.published / n:.0f}%), of which {self.rescued} "
            f"({100 * self.rescued / n:.0f}% of frames) had no depth mask at "
            f"all  rejections[{bad}]")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
  u = float((a | b).sum())
  return float((a & b).sum()) / u if u else 0.0


def _centre(mask: np.ndarray) -> np.ndarray:
  ys, xs = np.nonzero(mask)
  if xs.size == 0:
    return np.zeros(2)
  return np.array([xs.mean(), ys.mean()])
