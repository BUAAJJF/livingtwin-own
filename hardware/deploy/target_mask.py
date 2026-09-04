"""What the policy is shown as "the target", once the segmenter has spoken.

The segmenter and ``TargetTracker`` answer one question -- which instance on
the table is the target -- and they answer it from the current frame only.
Two situations on the real arm are not failures of that answer but questions it
cannot be asked, and both were found the hard way:

* the arm is standing in front of the object it is reaching for, so there is
  nothing to segment;
* the jaws are closed on the object, so from a fixed viewpoint it is inside
  the arm and is not a thing on the table at all.

The substitutions below are what the deployment does about that.  Neither is a
segmentation and neither invents a pixel; both replace a missing measurement
with something the robot knows by other means.

This lives in its own module because it is measured in two places that must
not drift apart.  ``run.py`` runs it against a D455; ``scripts/sim_perception_
check.py`` runs it against the renderer, which is the only place its output can
be scored against a truth.  When it was written twice -- once in each -- the
copy in the checker was already subtly different within a day.
"""

from __future__ import annotations

import numpy as np


class TargetMask:
  """Hold-over and grasp-site reconstruction, in the order ``run.py`` applies
  them.  Stateful: it remembers the last mask it published."""

  def __init__(self, held_radius: float, reproj, rig,
               rebuild_only_if_empty: bool = False) -> None:
    self.held_radius = float(held_radius)
    self.rebuild_only_if_empty = bool(rebuild_only_if_empty)
    """Whether the grasp-site sphere may overwrite a mask that already exists.

    False is the shipped depth-only behaviour and is right there: during a
    carry the segmenter has nothing, so there is nothing to overwrite.

    True is for a stack that carries the object through the grasp.  Scored
    against the renderer over seven seeds, SAM2.1's held mask is IoU 0.804
    against this reconstruction's 0.614 -- so applying the sphere on top of it
    is a downgrade, and measurably so.  TwinSight's handoff recommends the
    opposite ("the kinematic grasp-site mask reconstruction may be more
    trustworthy than a visible-only SAM mask occluded by the fingers"); that
    was written before either could be scored against a truth, and the
    measurement disagrees with it.  The reconstruction stays as the fallback
    for the frames SAM has nothing on."""
    self.reproj = reproj
    self.rig = rig
    self.last = None
    self.holdovers = 0
    self.rebuilds = 0

  def __call__(self, target, label, has_target, carrying, depth, valid, site):
    """Args:
      target: the segmenter's mask on the policy grid, possibly empty.
      label: the tracker's label this frame, 0 for none.
      has_target: whether the tracker still believes in a target at all.
      carrying: the lifecycle's ``holding``, or the jaw switch without one.
      depth, valid: the policy-grid depth and validity for this frame.
      site: ``(3,)`` grasp site in the base frame.

    Returns:
      ``(target, held_over)`` -- the mask to show the policy, and whether it
      came from a substitution rather than from this frame's segmentation.
    """
    held_over = False

    # The arm occludes its own target.  Reaching over an object puts the
    # forearm between it and a camera bolted to the world, and the object
    # goes from 400 pixels to nothing in a couple of frames -- measured on
    # the first powered run, which then held for 1168 of 1376 steps because
    # a held pose cannot uncover what it is covering.
    #
    # Feeding the empty mask through instead is worse, and that was tried:
    # an empty mask is not "the target is hidden", it is "there is no
    # target", and the policy has no way to tell them apart.  It drove for
    # 137 blind steps and drifted 285 mm away from the object.
    #
    # What is true here and not in general: this camera is bolted down and
    # the object is not moving.  So the last mask the segmenter produced is
    # still where the object is, and re-using it says exactly the right
    # thing -- "it is still there, you are standing in front of it".  The
    # tracker's ``lost_frames`` window bounds how long that stays credible;
    # past it the object may really have moved and the loop holds instead.
    # ``target`` may come from a causal tracker rather than the current depth
    # instance.  In that case there deliberately is no table ``label`` during
    # a depth gap or a carry, but the mask is still a current measurement.  The
    # old ordering treated ``label == 0`` as proof that ``target`` was empty
    # and replaced a good SAM mask with the last static table mask.  Use the
    # pixels as the authority for whether a current mask exists.
    current = bool(np.asarray(target).any())
    if label or current:
      self.last = target.copy()
    elif self.last is not None and has_target:
      target = self.last
      held_over = True
      self.holdovers += 1
    elif not has_target:
      self.last = None

    # Once the jaws are closed on the object, the sentence above stops being
    # true.  "The camera is bolted down and the object is not moving" is what
    # makes re-using the last mask correct during the approach; after a grasp
    # the object travels with the hand, and the last mask points at the patch
    # of table it was lifted from.
    #
    # Measured on recordings/v4_fixedseg_try6, which grasped successfully and
    # then froze: with the jaws closed the target was reported in 155 of 1894
    # frames, and the arm held one pose from t=15.7 s to t=45 s because a held
    # pose cannot uncover what it is covering.  The pixels were not the
    # problem -- 809 valid points sat within 50 mm of the grasp site -- and
    # neither was ``arm_mask``: sparing that volume moved detection only from
    # 8.2% to 11.2%.  What the segmenter cannot do is call the thing in the
    # gripper an object, because from a fixed viewpoint it is inside the arm.
    #
    # But the robot is not guessing where it is.  It is holding it.  The
    # object is at the grasp site, to within the jaw gap, and that is a
    # better measurement than the camera has.  So the mask is rebuilt there
    # from the depth that actually arrived: policy pixels whose unprojected
    # base point falls within ``held_radius`` of the grasp site.  No pixel is
    # invented -- a pixel with no depth stays out -- and the result is what
    # the simulator shows the policy at this moment, which is the object
    # travelling with the hand rather than an empty frame.
    #
    # With a lifecycle, ``holding`` is the authority and the rebuild is
    # unconditional: a table label during a carry is a bystander by
    # definition, so waiting for an EMPTY label before rebuilding -- which
    # is what this did -- hands the wrong object to the policy exactly when
    # the tracker is most confident about it.
    #
    # Scored against the renderer over 1200 steps, this reconstruction is
    # IoU 0.577 with the object's real silhouette while the depth segmenter
    # alone is 0.000 -- so it is a large improvement on nothing, and a
    # sphere-shaped approximation of a box.  A tracker that carries the real
    # outline through the grasp measures 0.866 on the same frames, which is
    # the argument for one; see ``sam_tracker``.
    if (carrying and self.held_radius > 0 and self.rig is not None
        and not (self.rebuild_only_if_empty and target.any())):
      pts = self.reproj.virtual_points_base(depth, self.rig)
      near = (np.linalg.norm(pts - np.asarray(site, dtype=np.float64), axis=1)
              < self.held_radius)
      near = near.reshape(target.shape) & valid
      if near.any():
        target = near
        held_over = True
        self.rebuilds += 1
    return target, held_over
