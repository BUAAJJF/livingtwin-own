"""Which pixels are the object the policy is currently fetching.

In simulation this channel is exact -- it comes out of a segmentation buffer
that knows which geom is which.  On the robot nothing knows, and the channel
still has to be filled, so this is where the deployment actually differs from
the training environment.  ``pick_place.mdp.CameraScene`` corrupts the
simulated mask to match what this can deliver: it drops wherever the depth
dropped, and its boundary is a pixel out.

Two backends behind one interface.

``DepthSegmenter`` needs no training and no labels.  The objects are the only
things on the table, so "above the table plane, inside the workspace, not the
arm" already isolates them, and connected components separates them from each
other.  It runs on the sensor's 848x480 grid rather than the policy's 224x168
-- a 40 mm object is 25 pixels across there and 9 here, and 9 pixels is not
enough to split two objects that are touching.

``YoloSegmenter`` is the one that earns its place where the first one cannot:
the bench measured the D405's fill rate on a blank white surface at 88% on
average and 42% at worst, and a depth segmenter has nothing to segment where
there is no depth.  The colour image is unaffected by that failure.  It is not
a semantic problem -- there is nothing to recognise, everything on the table
goes in the bin -- so there is no pretrained model to use and no labels to
collect by hand.  ``autolabel.py`` runs the depth segmenter over recorded
sessions and writes its output as the training set, which is exactly the right
division of labour: the depth segmenter is reliable most of the time, and
distilling it into an appearance model is how the good frames pay for the bad
ones.

Selection is the sim's rule, restated: the nearest object to the hand, and
re-evaluated only when one is cleared.  A target that follows the gripper
around lets the policy change its mind by moving.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from . import config, rectify


@dataclasses.dataclass
class Instance:
  """One thing on the table."""

  label: int
  n_px: int
  centroid_base: np.ndarray
  """(3,) metres in the robot base frame -- the mean of its points."""
  top_z: float
  """Height of its highest point above the table."""
  bbox: tuple[int, int, int, int]
  """(x, y, w, h) on the sensor grid."""


@dataclasses.dataclass
class Segmentation:
  labels: np.ndarray
  """int32 on the segmenter's own grid, which is the sensor's decimated by
  ``DepthSegmenter.decimate``.  0 is background.  Use ``full_mask`` to get one
  instance back on the full sensor grid, which is what the depth resampler
  wants as a payload."""
  instances: list[Instance]


@dataclasses.dataclass
class SegmenterCfg:
  min_height_m: float = 0.008
  """How far above the table a *pixel* has to be to be part of something.
  Deliberately low, and paired with ``min_top_z_sigmas``, which is the filter
  decides whether the thing it is part of is real: keeping the pixel threshold
  low means an object's sloping sides are included in its outline rather than
  cropped to its cap."""
  max_height_m: float = 0.14
  """Objects are 24-90 mm tall.  Anything half again as tall as the tallest is
  the arm, the bin or a person, and there is no reason to consider it."""
  min_area_px: int = 150
  """A 25 mm object at 0.8 m covers about 180 sensor pixels.  Set below that so
  a badly holed one still survives, and above the ~60 px a blob of correlated
  sensor noise makes: the measured correlation length of 8.5 px is 14 mm at
  0.7 m, which is a 9-pixel blob."""
  max_area_px: int = 20000
  open_px: int = 2
  """Morphological opening before labelling.  Depth noise puts single pixels
  above the threshold all over the table; two pixels of opening removes them
  and costs nothing on a 25-pixel object."""
  arm_clearance_m: float = 0.020
  """Added to each robot geom's own bounding radius.  The radii come from the
  model, so this is only the margin: enough to cover the calibration residual
  and the depth noise, not enough to swallow an object lying next to the
  fingers.  An object actually *in* the hand is excluded, which is correct --
  it is no longer a thing on the table to be found."""

  smooth_px: float = 6.0
  """Gaussian blur applied to the height field before thresholding, in sensor
  pixels.  Sized against the measured noise correlation length of 8.5 px: the
  blur has to average several of those to do anything, and has to stay well
  under the 25 px an object covers."""

  min_top_z_floor_m: float = 0.008
  min_top_z_sigmas: float = 3.4
  """How tall a component has to be to be an object: whichever is larger of the
  floor and ``sigmas`` times the sensor's noise **at that component's range**.

  This is the filter that does the most work.  A third of the sensor's error
  does not change between frames (``piper_push.depth_noise``), so a static bump
  in the noise is a phantom that the tracker's confirmation cannot reject -- it
  is in the same place every frame, exactly like an object.  Only its height
  distinguishes it.

  Measured against the noise rather than fixed, because a fixed number is wrong
  in both directions.  Tuned on a synthetic session with the fitted sensor
  model, 20 mm took the segmenter from 8.6 instances per frame to about 2, one
  of which was the object -- and then rejected a real 50-pixel object outright
  in a scene with no noise at all, because 20 mm is most of the height of the
  shortest thing the task uses.

  And it has to scale with range, not just with the frame.  The sensor's error
  grows as the square of the distance, so one threshold for the whole image is
  simultaneously too high at the near edge of the table and too low at the far
  edge -- which is exactly where the phantoms appeared: five scenes, and the
  false instances sat at 0.75-0.85 m with heights of 20-26 mm, while the object
  at 0.6 m was 28 mm.  The scatter is therefore measured as a coefficient on
  z^2 and the threshold is evaluated at each component's own range."""

  width_range_m: tuple[float, float] = (0.015, 0.16)
  """Longest horizontal extent a component may have.  The task's objects are
  25-45 mm across the jaws but up to 90 mm tall, and a toppled one presents its
  height as its footprint -- which is what set this: a 90 mm object lying down,
  grown by the smoothing, measured 114 mm and was thrown away by a 100 mm
  ceiling.  This is a weak filter and it is meant to be; ``max_elongation`` is
  the one that does the work."""

  max_elongation: float = 5.0
  """Long extent over short.  The objects are blocky -- their aspect ratio is
  drawn from 0.55 to 2.2 -- and the false positives that survive everything
  else are thin strips along the arm's outline and the table's far edge, which
  are ten to one or worse."""

  plane_fit_points: int = 20000
  """How many points the table plane is fitted on.  See the note where it is
  used: all of them costs 51 ms and the answer does not move."""

  bin_margin_m: float = 0.05
  """The bin is scenery, not an object, and its rim is 60 mm of vertical wall
  standing above the table -- squarely inside the 24-90 mm the objects occupy,
  and by far the most object-like thing in frame that is not one.  Excluded by
  footprint in the base frame, because that is knowledge the rig has and the
  segmenter should not have to rediscover it.

  The margin is 50 mm and not 30 because 30 leaked: a strip of rim 4-12 mm
  outside the excluded footprint came back as a persistent 50 mm-tall instance
  in the tracker, nearer to the hand than the real object in one scene out of
  five.  The margin has to cover the wall thickness, the calibration residual
  and the lateral spread the depth noise puts on a vertical surface."""


def full_mask(seg: "Segmentation", label: int, decimate: int) -> np.ndarray:
  """One instance's mask, back on the full sensor grid.

  Nearest-neighbour, which for an integer decimation is exact in the sense that
  matters: every source pixel is given the label of the sample it was
  represented by.  The outline is a decimated pixel coarser than the depth's,
  which is the price of segmenting cheaply and is smaller than the boundary
  error of the segmentation itself.
  """
  m = (seg.labels == label).astype(np.uint8)
  if decimate == 1:
    return m.astype(np.int32)
  return np.repeat(np.repeat(m, decimate, axis=0), decimate,
                   axis=1).astype(np.int32)


class DepthSegmenter:
  """Table-plane removal and connected components, no training required."""

  def __init__(self, rig: "config.Rig", reproj, cfg: SegmenterCfg | None = None,
               bin_footprint=None, decimate: int = 2):
    self.rig = rig
    # Its own resampler, at its own resolution.  Sharing the one the depth
    # channel uses would tie the segmenter's cost to the measurement's
    # resolution, and they do not need the same thing: the depth wants every
    # sample there is, and the segmenter wants enough pixels to separate two
    # objects that are touching, which is far fewer.
    self.decimate = max(1, int(decimate))
    self.reproj = (reproj if self.decimate == 1
                   else rectify.Reprojector(
                     rig, virtual=reproj.virtual, device=reproj.device,
                     decimate=self.decimate))
    self.full_reproj = reproj
    self.cfg = cfg or SegmenterCfg()
    self.bin_footprint = bin_footprint if bin_footprint is not None \
      else (config.BIN_CENTER, config.BIN_OUTER)

  def _height_above_table(self, pts: np.ndarray, in_box: np.ndarray) -> np.ndarray:
    """Signed height above a plane fitted to the table, robustly.

    Three rounds of reweighting rather than RANSAC: the table is most of what
    is in the box by a wide margin, so the only job is to stop the objects and
    the arm from tilting the fit, and a Tukey-ish weight does that in three
    passes over a few hundred thousand points.  RANSAC would be the right tool
    if the inlier fraction were in doubt; it is not.
    """
    near = np.flatnonzero(in_box
                          & (np.abs(pts[:, 2] - self.rig.table_z) < 0.05))
    if near.size < 500:
      self.plane = (np.array([0.0, 0.0, 1.0]), self.rig.table_z)
      self.plane_sigma = 0.0
      return pts[:, 2] - self.rig.table_z
    # A plane has three parameters and this has a quarter of a million points.
    # Fitting it on all of them costs 51 ms of a 20 ms control period and buys
    # nothing: the standard error on the fit is already a hundredth of the
    # sensor's noise at twenty thousand.  Strided rather than randomly sampled
    # so the same frame gives the same plane twice.
    if near.size > self.cfg.plane_fit_points:
      near = near[::max(1, near.size // self.cfg.plane_fit_points)]
    q = pts[near]
    wgt = np.ones(q.shape[0])
    for _ in range(3):
      mu = (q * wgt[:, None]).sum(0) / wgt.sum()
      cov = ((q - mu) * wgt[:, None]).T @ (q - mu) / wgt.sum()
      n = np.linalg.eigh(cov)[1][:, 0]
      if n[2] < 0:
        n = -n
      r = (q - mu) @ n
      s = 1.4826 * np.median(np.abs(r)) + 1e-4
      wgt = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
    self.plane = (n, float(mu @ n))
    self.plane_sigma = float(1.4826 * np.median(np.abs((q - mu) @ n)))
    return pts @ n - self.plane[1]

  def __call__(self, depth: np.ndarray, rgb: np.ndarray | None = None,
               arm=None) -> Segmentation:
    """Args:
      depth: ``(480, 848)`` metres from the sensor, 0 where invalid.
      rgb: unused here, and in the signature so that the two backends are
        interchangeable -- the loop should not have to know which one it has.
      arm: ``(centres, radii)`` from ``proprio.Kinematics.link_spheres``.
        Without it the arm is an object, and it is the largest one in frame.
    """
    del rgb
    c = self.cfg
    depth = self.reproj.source(depth)
    h, w = depth.shape
    pts = self.reproj.points_base(depth, self.rig)          # (N, 3), valid only
    # The reprojector already dropped everything outside the policy's field of
    # view, so this is its index, not every valid pixel in the frame.
    flat_idx = self.reproj.last_src

    (xlo, xhi), (ylo, yhi), (zlo, zhi) = config.WORKSPACE
    in_box = ((pts[:, 0] > xlo) & (pts[:, 0] < xhi)
              & (pts[:, 1] > ylo) & (pts[:, 1] < yhi)
              & (pts[:, 2] > zlo) & (pts[:, 2] < zhi))

    # The table plane is measured every frame, not taken from the calibration.
    # The camera's range bias is a per-unit unknown -- the bench measured -14 mm
    # at 0.7 m on this one and randomises +-15 mm around it, because the next
    # one will be different -- and a bias that size against a fixed 8 mm
    # threshold either turns the whole table into objects or hides the short
    # ones.  Fitting the plane makes the segmenter insensitive to it, and to
    # the table not being quite level in the base frame.
    height = self._height_above_table(pts, in_box)

    # Everything that is not a candidate is removed *before* the smoothing, not
    # after.  Removing it afterwards leaves a fringe: the blur carries the arm's
    # 300 mm of height several pixels out into the table around it, those pixels
    # are not inside any exclusion sphere, and they come back as tall thin
    # slivers hugging the arm's outline -- which is exactly what this produced
    # on the first run, three of them, and the tracker reached for one.
    usable = in_box.copy()
    if arm is not None:
      centres = np.asarray(arm[0], dtype=pts.dtype)
      radii = np.asarray(arm[1], dtype=pts.dtype) + c.arm_clearance_m
      # Bounding box first.  Twenty-four sphere tests over a quarter of a
      # million points is 116 ms -- six control periods -- and the arm occupies
      # a few percent of the frame, so almost all of that work is spent proving
      # that the table is not the robot.  Three comparisons reject it instead,
      # and the spheres then run on what is left.
      lo = (centres - radii[:, None]).min(axis=0)
      hi = (centres + radii[:, None]).max(axis=0)
      near_arm = np.flatnonzero(
        ((pts > lo) & (pts < hi)).all(axis=1) & in_box)
      if near_arm.size:
        sub = pts[near_arm]
        drop = np.zeros(near_arm.size, dtype=bool)
        # One geom at a time.  The broadcast form is two lines shorter and
        # allocates an (N, G, 3), which is 300 MB at full frame.
        for centre, radius in zip(centres, radii):
          drop |= ((sub - centre) ** 2).sum(axis=1) < radius * radius
        usable[near_arm[drop]] = False

    if self.bin_footprint is not None:
      (bx, by), (hx, hy) = self.bin_footprint
      m = c.bin_margin_m
      usable &= ~((np.abs(pts[:, 0] - bx) < hx + m)
                  & (np.abs(pts[:, 1] - by) < hy + m))

    # Smoothed before thresholding, in the image.  At 0.7 m the sensor's noise
    # is 10 mm and correlated across 8 pixels, so it is not something a
    # threshold survives and not something a 2-pixel opening removes: the blobs
    # it makes are the size of the objects being looked for.  A 4-pixel blur
    # averages several correlation lengths and halves it, and a 25-pixel object
    # does not notice.
    smooth_px = c.smooth_px / self.decimate
    if smooth_px > 0:
      img = np.zeros(h * w, dtype=np.float32)
      wt = np.zeros(h * w, dtype=np.float32)
      sel = flat_idx[usable]
      img[sel] = height[usable].astype(np.float32)
      wt[sel] = 1.0
      img = cv2.GaussianBlur(img.reshape(h, w), (0, 0), smooth_px)
      wt = cv2.GaussianBlur(wt.reshape(h, w), (0, 0), smooth_px)
      # Normalised by the weight, so a hole or an excluded neighbour does not
      # drag a pixel's height down towards zero.
      smoothed = img.reshape(-1) / np.maximum(wt.reshape(-1), 1e-3)
      enough = wt.reshape(-1)[flat_idx] > 0.25
      height = np.where(enough, smoothed[flat_idx], height)
      usable &= enough

    # The table's scatter *after* smoothing, as a coefficient on z^2.  Measured
    # here rather than derived from the pre-smoothing sigma: how much a blur
    # removes depends on how far the noise is correlated, and that is a
    # property of the camera.  Divided by the range squared because that is how
    # the error grows, so one number describes the whole image.
    rng = np.asarray(depth, dtype=np.float64).reshape(-1)[flat_idx]
    table = usable & (np.abs(height) < 0.02)
    self.noise_per_m = float(
      1.4826 * np.median(np.abs(height[table]) / np.maximum(rng[table], 1e-3) ** 2)
    ) if table.sum() > 500 else 0.0
    self.noise_sigma = self.noise_per_m * float(np.median(rng)) ** 2
    """Scatter at the median range in the frame, for reporting."""

    keep = usable & (height > c.min_height_m) & (height < c.max_height_m)

    fg = np.zeros(h * w, dtype=np.uint8)
    fg[flat_idx[keep]] = 255
    fg = fg.reshape(h, w)
    open_px = max(1, round(c.open_px / self.decimate))
    if c.open_px > 0:
      k = np.ones((2 * open_px + 1,) * 2, np.uint8)
      fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    # Which component each surviving 3-D point fell in, so the base-frame
    # statistics are grouped in one pass rather than one full-image scan per
    # component.
    comp = labels.reshape(-1)[flat_idx[keep]]
    order = np.argsort(comp, kind="stable")
    comp_sorted = comp[order]
    starts = np.searchsorted(comp_sorted, np.arange(n), side="left")
    ends = np.searchsorted(comp_sorted, np.arange(n), side="right")
    pts_keep = pts[keep][order]
    height_keep = height[keep][order]
    rng_keep = rng[keep][order]

    remap = np.zeros(n, dtype=np.int32)
    instances: list[Instance] = []
    # Why each component was thrown away.  Not diagnostics for their own sake:
    # when this returns nothing on the robot the only question is which filter
    # said no, and answering it by adding prints to a running control loop is
    # not a good afternoon.
    self.rejected: list[tuple[str, float]] = []
    for i in range(1, n):
      area = int(stats[i, cv2.CC_STAT_AREA])
      # Areas are quoted for the full sensor grid, so a decimated segmenter
      # sees the same object as fewer pixels and the thresholds have to follow.
      if not (c.min_area_px / self.decimate ** 2 <= area
              <= c.max_area_px / self.decimate ** 2):
        self.rejected.append(("area", float(area)))
        continue
      p = pts_keep[starts[i]:ends[i]]
      if p.shape[0] == 0:
        self.rejected.append(("no depth", 0.0))
        continue
      # 97th percentile, not the maximum: one noisy sample decides the maximum
      # and this number is a filter.
      top = float(np.percentile(height_keep[starts[i]:ends[i]], 97))
      z = float(np.median(rng_keep[starts[i]:ends[i]]))
      floor_z = max(c.min_top_z_floor_m,
                    c.min_top_z_sigmas * self.noise_per_m * z * z)
      if not (floor_z <= top <= c.max_height_m):
        self.rejected.append(("height", top))
        continue
      lo, hi = np.percentile(p[:, :2], [2, 98], axis=0)
      span = np.sort(hi - lo)
      extent = float(span[-1])
      if not (c.width_range_m[0] <= extent <= c.width_range_m[1]):
        self.rejected.append(("footprint", extent))
        continue
      if extent / max(float(span[0]), 1e-4) > c.max_elongation:
        self.rejected.append(("elongation", extent / max(float(span[0]), 1e-4)))
        continue
      remap[i] = len(instances) + 1
      instances.append(Instance(
        label=remap[i],
        n_px=area,
        centroid_base=p.mean(axis=0),
        top_z=top,
        bbox=(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
              int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
      ))
    return Segmentation(labels=remap[labels], instances=instances)


class TargetTracker:
  """Which instance is "this one", held steady across frames.

  Two jobs, and the second is the one that makes the depth segmenter usable at
  all on this camera.

  The first is the simulator's rule, restated for a scene where object identity
  has to be re-established every frame: aim at the nearest object to the hand,
  and do not re-aim until the current one is gone.  A target that follows the
  gripper around lets the policy change its mind by moving, and the mask
  channel would stop meaning what it meant during training.

  The second is confirmation.  The measured sensor puts 10 mm of noise on the
  table at 0.7 m, correlated across 8 pixels, and a blob of that two standard
  deviations high looks exactly like a short object: in five draws over one
  scene, two produced more than a dozen phantom instances, and the
  nearest-to-the-hand rule picked one.  What the phantoms cannot do is come
  back -- the noise has a lag-1 autocorrelation of -0.007, so each frame draws
  a new set in new places, while a real object is in the same place 50 times a
  second.  So an instance has to be seen ``confirm`` times in the last
  ``window`` frames before it can be chosen.  At 50 Hz that costs 60 ms of
  latency once, when the scene changes, and it is the difference between a
  segmenter that works on this camera and one that does not.
  """

  def __init__(self, match_radius_m: float = 0.06, lost_frames: int = 15,
               confirm: int = 3, window: int = 5):
    self.match_radius_m = match_radius_m
    self.lost_frames = lost_frames
    self.confirm = confirm
    self.window = window
    self._tracks: list[dict] = []
    self._centroid: np.ndarray | None = None
    self._missing = 0
    self.confirmed_labels: set[int] = set()

  @property
  def has_target(self) -> bool:
    return self._centroid is not None

  def _advance(self, seg: "Segmentation") -> list[tuple[int, np.ndarray]]:
    """Associate this frame's instances with the running tracks.

    Nearest-centroid association with a gate.  Good enough because the objects
    are static between frames and are further apart than the gate; a proper
    assignment would be the right thing if they were not.
    """
    for t in self._tracks:
      t["hits"].append(0)
      del t["hits"][:-self.window]

    for inst in seg.instances:
      cent = inst.centroid_base
      if not np.isfinite(cent).all():
        continue
      best, best_d = None, self.match_radius_m
      for t in self._tracks:
        d = float(np.linalg.norm(t["centroid"] - cent))
        if d < best_d:
          best, best_d = t, d
      if best is None:
        self._tracks.append({"centroid": cent.copy(), "hits": [1],
                             "label": inst.label, "area": inst.n_px})
      else:
        best["centroid"] = 0.5 * (best["centroid"] + cent)
        best["hits"][-1] = 1
        best["label"] = inst.label
        best["area"] = max(best["area"], inst.n_px)

    self._tracks = [t for t in self._tracks if sum(t["hits"]) > 0]
    live = [(sum(t["hits"]), t) for t in self._tracks
            if sum(t["hits"]) >= self.confirm]
    if not live:
      return []
    # Only the most persistent tracks are candidates, and then the nearest to
    # the hand among those.  The simulator's rule is "nearest", and keeping it
    # matters -- the mask channel has to mean what it meant in training -- but
    # applied to the raw candidate set on this camera it picks phantoms: a blob
    # of static depth noise near the gripper beats a real object 15 cm away.
    # What separates them is that a real object is seen in every frame and a
    # phantom is not, so the persistence gate goes first and the distance rule
    # decides between things that are all equally real.
    # Nearest to the hand among the confirmed, and nothing else.  Two extra
    # rules were tried here and both were removed: preferring the most
    # persistent track changed nothing, and preferring the larger one made
    # things worse -- in one scene it promoted a 178-pixel phantom over the
    # 83-pixel object that was actually nearer, and the tracker then spent the
    # rest of the sequence chasing something that was not there.  The area
    # distributions of real instances and phantoms overlap too much to sort on
    # (52/141/192 against 42/56/117 at the 10th, 50th and 90th percentile), and
    # the simulator's rule is the one the policy was trained under.
    return [(t["label"], t["centroid"]) for _, t in live]

  def update(self, seg: "Segmentation", hand_base: np.ndarray) -> int:
    """Return the label of the target in *this* frame, or 0 if there is none."""
    confirmed = self._advance(seg)
    # Every instance that has survived confirmation, not only the chosen one.
    # ``autolabel.py`` needs this: it labels all the objects on the table, and
    # labelling the phantoms alongside them would teach the model to
    # hallucinate exactly what the confirmation exists to reject.
    self.confirmed_labels = {int(lab) for lab, _ in confirmed}
    if not confirmed:
      self._missing += 1
      if self._missing > self.lost_frames:
        self._centroid = None
      return 0

    cents = np.stack([c for _, c in confirmed])
    if self._centroid is not None:
      d = np.linalg.norm(cents - self._centroid, axis=1)
      j = int(d.argmin())
      if d[j] <= self.match_radius_m:
        self._missing = 0
        self._centroid = cents[j]
        return confirmed[j][0]
      self._missing += 1
      if self._missing <= self.lost_frames:
        return 0
    j = int(np.linalg.norm(cents - np.asarray(hand_base), axis=1).argmin())
    self._missing = 0
    self._centroid = cents[j]
    return confirmed[j][0]

  def clear(self) -> None:
    """Forget the current target.  Call this when it has been placed."""
    self._centroid = None
    self._missing = 0


class YoloSegmenter:
  """Instance masks from the colour image, for when the depth has holes.

  Trained by ``train_yolo.py`` on labels that ``DepthSegmenter`` produced --
  see the module docstring.  Falls back to raising rather than silently
  returning nothing if the weights are missing, because a mask channel that is
  quietly all zeros looks exactly like "there is nothing on the table".
  """

  def __init__(self, weights: str, rig: "config.Rig", reproj,
               cfg: SegmenterCfg | None = None, conf: float = 0.25,
               device: str = "cuda:0"):
    from ultralytics import YOLO       # imported here: an optional dependency

    self.model = YOLO(weights)
    self.rig = rig
    self.reproj = reproj
    self.cfg = cfg or SegmenterCfg()
    self.conf = conf
    self.device = device
    self.decimate = 1
    """Full resolution.  The depth backend halves it to afford the per-point
    arithmetic; this one's cost is a forward pass and does not care."""

  def __call__(self, depth: np.ndarray, rgb: np.ndarray | None = None,
               arm=None) -> Segmentation:
    if rgb is None:
      raise ValueError(
        "the YOLO backend reads the colour image and was given none.  That is "
        "the whole reason it exists -- it works where the depth does not."
      )
    del arm      # the model was trained on frames the arm was already in
    h, w = depth.shape
    res = self.model.predict(rgb, conf=self.conf, device=self.device,
                             verbose=False)[0]
    out = np.zeros((h, w), dtype=np.int32)
    instances: list[Instance] = []
    if res.masks is None:
      return Segmentation(labels=out, instances=instances)

    pts_full = np.full((h * w, 3), np.nan, dtype=np.float64)
    # Indexed by ``last_src``, not by "every valid pixel": the resampler drops
    # the third of the frame that lies outside the policy's field of view
    # before it transforms anything, so those two sets are not the same.
    pts = self.reproj.points_base(depth, self.rig)
    pts_full[self.reproj.last_src] = pts

    for i, m in enumerate(res.masks.data.cpu().numpy()):
      m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
      area = int(m.sum())
      if area < self.cfg.min_area_px:
        continue
      p = pts_full[m.reshape(-1)]
      p = p[np.isfinite(p).all(axis=1)]
      if p.size == 0:
        # A detection the depth cannot place.  Kept -- this is the case the
        # model exists for -- but with no base-frame position, so the tracker
        # cannot match it and it will only be chosen if nothing else is.
        centroid = np.full(3, np.nan)
        top = float("nan")
      else:
        centroid = p.mean(axis=0)
        top = float(p[:, 2].max() - self.rig.table_z)
      out[m] = i + 1
      ys, xs = np.nonzero(m)
      instances.append(Instance(
        label=i + 1, n_px=area, centroid_base=centroid, top_z=top,
        bbox=(int(xs.min()), int(ys.min()),
              int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
      ))
    return Segmentation(labels=out, instances=instances)
