"""The two substitutions the deployment makes when the segmenter has nothing.

These are not segmentation and they are not cosmetic: measured against the
renderer, the grasp-site reconstruction is the difference between IoU 0.00 and
0.57 during a carry, and the hold-over is what stopped a powered run driving
137 blind steps and drifting 285 mm.  They were inline in ``run.py`` and a
second copy had already appeared in the simulation checker within a day, which
is why they now live in one module -- and why that module needs tests that do
not need a robot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hardware.deploy.target_mask import TargetMask  # noqa: E402

H, W = 8, 10


class FakeReproj:
  """Unprojects the policy grid to a plane, so "within r of the site" is a
  disc this file can draw by hand."""

  def __init__(self, pts):
    self.pts = pts

  def virtual_points_base(self, depth, rig):
    return self.pts


def grid_points(z=0.0):
  ys, xs = np.mgrid[0:H, 0:W]
  return np.stack([xs.ravel() * 0.01, ys.ravel() * 0.01,
                   np.full(H * W, z)], axis=1)


def blob(x0, x1, y0, y1):
  m = np.zeros((H, W), bool)
  m[y0:y1, x0:x1] = True
  return m


def _tm(radius=0.0, pts=None):
  return TargetMask(radius, FakeReproj(pts if pts is not None else grid_points()),
                    rig=object())


DEPTH = np.ones((H, W), np.float32)
VALID = np.ones((H, W), bool)
SITE = np.array([0.03, 0.03, 0.0])


def test_a_live_mask_passes_through_untouched():
  t = _tm()
  m = blob(2, 5, 2, 5)
  out, held = t(m, label=3, has_target=True, carrying=False,
                depth=DEPTH, valid=VALID, site=SITE)
  assert np.array_equal(out, m) and not held


def test_the_last_mask_is_re_used_while_the_arm_stands_in_front_of_it():
  """An empty mask is not "hidden", it is "there is no target", and the policy
  cannot tell those apart -- it drove 137 blind steps on the difference."""
  t = _tm()
  m = blob(2, 5, 2, 5)
  t(m, 3, True, False, DEPTH, VALID, SITE)
  out, held = t(np.zeros((H, W), bool), 0, True, False, DEPTH, VALID, SITE)
  assert np.array_equal(out, m) and held and t.holdovers == 1


def test_the_hold_over_ends_when_the_tracker_gives_up():
  """``lost_frames`` bounds how long "it is still there" stays credible."""
  t = _tm()
  t(blob(2, 5, 2, 5), 3, True, False, DEPTH, VALID, SITE)
  out, held = t(np.zeros((H, W), bool), 0, False, False, DEPTH, VALID, SITE)
  assert not out.any() and not held and t.last is None


def test_a_carry_rebuilds_the_mask_at_the_grasp_site_not_where_it_was_lifted():
  """The hold-over's premise -- camera bolted down, object not moving -- stops
  being true the moment the jaws close.  The last mask then points at the
  patch of table the object was lifted from."""
  t = _tm(radius=0.015)
  lifted_from = blob(0, 2, 0, 2)
  t(lifted_from, 3, True, False, DEPTH, VALID, SITE)
  out, held = t(np.zeros((H, W), bool), 0, True, True, DEPTH, VALID, SITE)
  assert held and t.rebuilds == 1
  assert not (out & lifted_from).any(), "still pointing at the empty table"
  ys, xs = np.nonzero(out)
  assert abs(xs.mean() * 0.01 - SITE[0]) < 0.005


def test_the_rebuild_invents_no_pixel_without_depth():
  """"No pixel is invented" is the property that makes this a reconstruction
  rather than a guess; a pixel the sensor never measured stays out."""
  t = _tm(radius=0.05)
  valid = np.zeros((H, W), bool)
  valid[0, 0] = True
  out, _ = t(np.zeros((H, W), bool), 0, True, True, DEPTH, valid, SITE)
  assert out.sum() <= 1


def test_a_zero_radius_disables_the_rebuild_entirely():
  t = _tm(radius=0.0)
  out, held = t(np.zeros((H, W), bool), 0, True, True, DEPTH, VALID, SITE)
  assert not out.any() and not held and t.rebuilds == 0


def test_an_empty_rebuild_leaves_the_previous_answer_alone():
  """Nothing within the radius has depth: the carry is real but unmeasurable,
  and replacing a hold-over with an empty mask would be strictly worse."""
  t = _tm(radius=0.001)
  m = blob(2, 5, 2, 5)
  t(m, 3, True, False, DEPTH, VALID, SITE)
  far = grid_points(z=9.0)
  t.reproj = FakeReproj(far)
  out, held = t(np.zeros((H, W), bool), 0, True, True, DEPTH, VALID, SITE)
  assert np.array_equal(out, m) and held and t.rebuilds == 0


def test_a_tracker_that_carries_the_object_keeps_its_own_mask():
  """The grasp-site sphere must not overwrite a better measurement.

  TwinSight's handoff recommends standing down to the kinematic
  reconstruction during a carry.  That was written before either could be
  scored: against the renderer over seven seeds, SAM's held mask is IoU 0.804
  and this reconstruction is 0.614, so applying the sphere on top of it is a
  measurable downgrade.  It stays as the fallback, not the override.
  """
  t = _tm(radius=0.05)
  t.rebuild_only_if_empty = True
  stale = blob(0, 2, 0, 2)
  t(stale, label=3, has_target=True, carrying=False,
    depth=DEPTH, valid=VALID, site=SITE)
  carried = blob(2, 5, 2, 5)
  out, held = t(carried, label=0, has_target=True, carrying=True,
                depth=DEPTH, valid=VALID, site=SITE)
  assert np.array_equal(out, carried) and not held and t.rebuilds == 0


def test_the_rebuild_is_still_the_fallback_when_the_tracker_has_nothing():
  t = _tm(radius=0.05)
  t.rebuild_only_if_empty = True
  out, held = t(np.zeros((H, W), bool), 0, False, True, DEPTH, VALID, SITE)
  assert held and t.rebuilds == 1 and out.any()


def test_the_default_still_overrides_because_the_segmenter_has_nothing_there():
  """Depth-only is the shipped path and must not change: during a carry the
  segmenter produces no mask, so there is nothing to preserve."""
  t = _tm(radius=0.015)
  assert t.rebuild_only_if_empty is False
  stale = blob(0, 2, 0, 2)      # far enough from the site to be distinguishable
  t(stale, 3, True, False, DEPTH, VALID, SITE)
  out, held = t(np.zeros((H, W), bool), 0, True, True, DEPTH, VALID, SITE)
  assert held and t.rebuilds == 1 and not (out & stale).any()
