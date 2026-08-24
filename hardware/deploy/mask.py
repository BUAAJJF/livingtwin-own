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
from .yolo_backend import load_detector


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


def workspace_mask(pts: np.ndarray) -> np.ndarray:
  """Which points are inside the box anything interesting is inside.

  Shared by both backends rather than written twice.  The depth backend applies
  it to every point before it looks for components; the YOLO backend applies it
  to a detection after the fact, because a network that was trained on this
  table will occasionally fire on something across the room and the cheapest
  way to know is to ask where it is.
  """
  (xlo, xhi), (ylo, yhi), (zlo, zhi) = config.WORKSPACE
  return ((pts[:, 0] > xlo) & (pts[:, 0] < xhi)
          & (pts[:, 1] > ylo) & (pts[:, 1] < yhi)
          & (pts[:, 2] > zlo) & (pts[:, 2] < zhi))


def arm_mask(pts: np.ndarray, arm, clearance_m: float,
             within: np.ndarray | None = None) -> np.ndarray:
  """Which points are the robot, from the sphere cover of its own geometry.

  Bounding box first.  Twenty-four sphere tests over a quarter of a million
  points is 116 ms -- six control periods -- and the arm occupies a few percent
  of the frame, so almost all of that work is spent proving that the table is
  not the robot.  Three comparisons reject it instead, and the spheres then run
  on what is left.
  """
  out = np.zeros(pts.shape[0], dtype=bool)
  if arm is None:
    return out
  centres = np.asarray(arm[0], dtype=pts.dtype)
  radii = np.asarray(arm[1], dtype=pts.dtype) + clearance_m
  if centres.size == 0:
    return out
  lo = (centres - radii[:, None]).min(axis=0)
  hi = (centres + radii[:, None]).max(axis=0)
  near = ((pts > lo) & (pts < hi)).all(axis=1)
  if within is not None:
    near &= within
  idx = np.flatnonzero(near)
  if idx.size == 0:
    return out
  sub = pts[idx]
  drop = np.zeros(idx.size, dtype=bool)
  # One geom at a time.  The broadcast form is two lines shorter and allocates
  # an (N, G, 3), which is 300 MB at full frame.
  for centre, radius in zip(centres, radii):
    drop |= ((sub - centre) ** 2).sum(axis=1) < radius * radius
  out[idx[drop]] = True
  return out


def bin_mask(pts: np.ndarray, footprint, margin_m: float) -> np.ndarray:
  """Which points are the bin.  It is scenery, and its rim is the most
  object-like thing in frame that is not an object."""
  if footprint is None:
    return np.zeros(pts.shape[0], dtype=bool)
  (bx, by), (hx, hy) = footprint
  return ((np.abs(pts[:, 0] - bx) < hx + margin_m)
          & (np.abs(pts[:, 1] - by) < hy + margin_m))


def fit_table_plane(pts: np.ndarray, near: np.ndarray, table_z: float,
                    max_points: int = 20000):
  """Signed height above a plane fitted to the table, robustly.

  Returns ``(height, (normal, offset), sigma)``.

  Three rounds of reweighting rather than RANSAC: the table is most of what is
  in the box by a wide margin, so the only job is to stop the objects and the
  arm from tilting the fit, and a Tukey-ish weight does that in three passes
  over a few hundred thousand points.  RANSAC would be the right tool if the
  inlier fraction were in doubt; it is not.

  Fitted every frame rather than taken from the calibration.  The camera's
  range bias is a per-unit unknown -- the bench measured -14 mm at 0.7 m on
  this one and randomises +-15 mm around it, because the next one will be
  different -- and a bias that size against a fixed 8 mm threshold either turns
  the whole table into objects or hides the short ones.
  """
  sel = np.flatnonzero(near & (np.abs(pts[:, 2] - table_z) < 0.05))
  if sel.size < 500:
    return pts[:, 2] - table_z, (np.array([0.0, 0.0, 1.0]), table_z), 0.0
  # A plane has three parameters and this has a quarter of a million points.
  # Fitting it on all of them costs 51 ms of a 20 ms control period and buys
  # nothing: the standard error on the fit is already a hundredth of the
  # sensor's noise at twenty thousand.  Strided rather than randomly sampled so
  # the same frame gives the same plane twice.
  if sel.size > max_points:
    sel = sel[::max(1, sel.size // max_points)]
  q = pts[sel]
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
  sigma = float(1.4826 * np.median(np.abs((q - mu) @ n)))
  return pts @ n - float(mu @ n), (n, float(mu @ n)), sigma


def place_on_plane(origin: np.ndarray, ray: np.ndarray, plane) -> np.ndarray:
  """Where a ray meets the table, or NaN if it runs parallel to it."""
  n, d = plane
  denom = float(np.dot(ray, n))
  if abs(denom) < 1e-6:
    return np.full(3, np.nan)
  t = (d - float(np.dot(origin, n))) / denom
  if t <= 0:
    return np.full(3, np.nan)
  return origin + t * np.asarray(ray, dtype=np.float64)


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
    """Signed height above the table.  See ``fit_table_plane``; this only keeps
    the fit where the rest of the class can read it."""
    height, self.plane, self.plane_sigma = fit_table_plane(
      pts, in_box, self.rig.table_z, self.cfg.plane_fit_points)
    return height

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

    in_box = workspace_mask(pts)

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
    usable = in_box & ~arm_mask(pts, arm, c.arm_clearance_m, within=in_box)
    usable &= ~bin_mask(pts, self.bin_footprint, c.bin_margin_m)

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


@dataclasses.dataclass
class YoloCfg:
  """Everything about the colour backend that is not about geometry.

  The geometric filters are ``SegmenterCfg``'s and are shared with the depth
  backend on purpose: whatever finds a candidate, the tests for "is that thing
  in the workspace, is it the arm, is it the bin" have one implementation.
  """

  conf: float = 0.25
  """Detection confidence.  Low, because a false positive here still has to
  survive the base-frame filters and then three frames of tracking, while a
  false negative is an object that never gets picked up."""

  iou: float = 0.50
  """NMS overlap.  Only the ONNX path uses it; ultralytics does its own."""

  mask_threshold: float = 0.5

  min_placed_px: int = 20
  """Valid depth samples a detection needs before its own points decide where
  it is.  Below that the ray through it is intersected with the table plane
  instead -- see ``__call__``."""

  max_arm_fraction: float = 0.5
  """How much of a detection may be the robot before it is the robot."""

  max_detections: int = 32


class YoloSegmenter:
  """Instance masks from the colour image, for when the depth has holes.

  Trained by ``train_yolo.py`` on labels ``DepthSegmenter`` produced -- see
  ``autolabel.py``.  What this class adds on top of the network is everything
  that stops a detection from becoming a target it should not be, and it is
  most of the file, because a segmentation network trained on one table will
  fire on the chair behind it and nothing in the network knows that the chair
  is not on the table.

  Three things the previous version of this class did not do, each of which is
  a way to reach for the wrong thing:

  * **The workspace box.**  ``DepthSegmenter`` throws away everything outside
    ``config.WORKSPACE`` before it looks for anything.  A detection is now
    placed in the base frame and tested against the same box, so the room
    behind the table cannot produce a target.
  * **The arm.**  The model was trained on frames the arm was in, and its
    labels excluded the arm -- but "was not labelled" is not "will never be
    detected", and the gripper is the most object-shaped thing in frame.
  * **The bin.**  Same argument, and its rim is 60 mm of vertical wall standing
    in the middle of the range the objects occupy.

  And one thing that was worse than not doing it: a detection whose pixels are
  all holes used to be kept with a NaN centroid, on the reasoning that it would
  only be chosen if nothing else was.  It could not be chosen at all --
  ``TargetTracker`` skips instances it cannot place -- so the backend that
  exists to find objects the depth cannot see was dropping exactly those.  They
  are now placed by intersecting the ray through the detection with the fitted
  table plane, which gives the point on the table under the object: half its
  height low, and well inside the tracker's 60 mm association gate.
  """

  def __init__(self, weights: str, rig: "config.Rig", reproj,
               cfg: SegmenterCfg | None = None, conf: float | None = None,
               device: str = "cuda:0", yolo_cfg: "YoloCfg | None" = None,
               bin_footprint=None):
    self.rig = rig
    self.reproj = reproj
    self.cfg = cfg or SegmenterCfg()
    self.yolo = yolo_cfg or YoloCfg()
    if conf is not None:
      self.yolo = dataclasses.replace(self.yolo, conf=float(conf))
    self.device = device
    self.bin_footprint = bin_footprint if bin_footprint is not None \
      else (config.BIN_CENTER, config.BIN_OUTER)
    self.decimate = 1
    """Full resolution.  The depth backend halves it to afford the per-point
    arithmetic; this one's cost is a forward pass and does not care."""
    self.detector = load_detector(weights, device=device, cfg=self.yolo)
    self.rejected: list[tuple[str, float]] = []
    self.plane = (np.array([0.0, 0.0, 1.0]), float(rig.table_z))
    self.plane_sigma = 0.0
    self.noise_per_m = 0.0
    self.n_unplaced = 0
    """How many of the last frame's instances were positioned from the plane
    rather than from their own depth.  This is the number that says whether the
    backend is earning its place: if it is zero, the depth segmenter would have
    found everything."""

  def __call__(self, depth: np.ndarray, rgb: np.ndarray | None = None,
               arm=None) -> Segmentation:
    """Args:
      depth: ``(480, 848)`` metres from the sensor, 0 where invalid.  Used to
        place the detections, not to find them.
      rgb: the colour frame at the same resolution.  Required -- working
        without depth is the whole reason this backend exists.
      arm: ``(centres, radii)`` from ``proprio.Kinematics.link_spheres``.
    """
    if rgb is None:
      raise ValueError(
        "the YOLO backend reads the colour image and was given none.  That is "
        "the whole reason it exists -- it works where the depth does not."
      )
    c = self.cfg
    h, w = depth.shape
    out = np.zeros((h, w), dtype=np.int32)
    instances: list[Instance] = []
    self.rejected = []
    self.n_unplaced = 0

    masks = self.detector(rgb, (h, w))
    if masks.shape[0] == 0:
      return Segmentation(labels=out, instances=instances)

    pts = self.reproj.points_base(depth, self.rig)
    src = self.reproj.last_src
    in_box = workspace_mask(pts)
    height, self.plane, self.plane_sigma = fit_table_plane(
      pts, in_box, self.rig.table_z, c.plane_fit_points)
    is_arm = arm_mask(pts, arm, c.arm_clearance_m, within=in_box)
    is_bin = bin_mask(pts, self.bin_footprint, c.bin_margin_m)
    usable = in_box & ~is_arm & ~is_bin

    # The table's scatter, as a coefficient on z^2, so the height test below
    # scales with range the way the sensor's error does.  Measured here rather
    # than assumed, for the same reason the depth backend measures it.
    rng = np.asarray(depth, dtype=np.float64).reshape(-1)[src]
    table = usable & (np.abs(height) < 0.02)
    self.noise_per_m = float(
      1.4826 * np.median(np.abs(height[table])
                         / np.maximum(rng[table], 1e-3) ** 2)
    ) if int(table.sum()) > 500 else 0.0

    # Source pixel -> row of ``pts``, so a detection's mask can be turned into
    # its points in one gather instead of a search.
    row = np.full(h * w, -1, dtype=np.int64)
    row[src] = np.arange(src.shape[0])
    origin = self.reproj.camera_origin_base(self.rig)

    for m in masks:
      flat = m.reshape(-1)
      area = int(flat.sum())
      if area < c.min_area_px or area > c.max_area_px:
        self.rejected.append(("area", float(area)))
        continue
      r = row[flat]
      r = r[r >= 0]
      if r.size and float(is_arm[r].mean()) > self.yolo.max_arm_fraction:
        self.rejected.append(("arm", float(is_arm[r].mean())))
        continue
      good = r[usable[r]] if r.size else r

      if good.size >= self.yolo.min_placed_px:
        centroid, top, why = self._from_points(pts[good], height[good],
                                               rng[good])
        if why:
          self.rejected.append(why)
          continue
      else:
        centroid = self._from_plane(flat, origin, w)
        top = float("nan")
        self.n_unplaced += 1

      if not np.isfinite(centroid).all():
        self.rejected.append(("unplaceable", float(good.size)))
        continue
      # The same three tests the points would have failed, applied to the one
      # position we have.  A detection placed on the plane has no height and no
      # footprint, so this is all there is to go on -- and it is the test that
      # matters, because it is the one that rejects the room.
      if not bool(workspace_mask(centroid[None])[0]):
        self.rejected.append(("outside the workspace",
                              float(np.linalg.norm(centroid[:2]))))
        continue
      if bool(bin_mask(centroid[None], self.bin_footprint, c.bin_margin_m)[0]):
        self.rejected.append(("the bin", 0.0))
        continue
      if bool(arm_mask(centroid[None], arm, c.arm_clearance_m)[0]):
        self.rejected.append(("the arm", 0.0))
        continue

      label = len(instances) + 1
      out[m] = label
      ys, xs = np.nonzero(m)
      instances.append(Instance(
        label=label, n_px=area, centroid_base=centroid, top_z=top,
        bbox=(int(xs.min()), int(ys.min()),
              int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
      ))
    return Segmentation(labels=out, instances=instances)

  def _from_points(self, p, height, rng):
    """Position and height from a detection's own cloud, with the depth
    backend's filters.  Returns ``(centroid, top, reason_or_None)``."""
    c = self.cfg
    # Median, not mean.  A detection that clips a few pixels of the table
    # behind the object drags a mean several centimetres; the median does not
    # notice, and this position feeds a 60 mm association gate.
    centroid = np.median(p, axis=0)
    top = float(np.percentile(height, 97))
    z = float(np.median(rng))
    floor_z = max(c.min_top_z_floor_m, c.min_top_z_sigmas * self.noise_per_m * z * z)
    if not (floor_z <= top <= c.max_height_m):
      return centroid, top, ("height", top)
    lo, hi = np.percentile(p[:, :2], [2, 98], axis=0)
    span = np.sort(hi - lo)
    extent = float(span[-1])
    if not (c.width_range_m[0] <= extent <= c.width_range_m[1]):
      return centroid, top, ("footprint", extent)
    if extent / max(float(span[0]), 1e-4) > c.max_elongation:
      return centroid, top, ("elongation", extent / max(float(span[0]), 1e-4))
    return centroid, top, None

  def _from_plane(self, flat: np.ndarray, origin: np.ndarray,
                  width: int) -> np.ndarray:
    """Where a detection with no usable depth is, assuming it is on the table.

    The ray is taken through the mask's centroid pixel rather than its bounding
    box centre, because the objects are not convex and a bracket's box centre
    is not on the bracket.
    """
    idx = np.flatnonzero(flat)
    u = float(np.mean(idx % width))
    v = float(np.mean(idx // width))
    pixel = int(round(v)) * width + int(round(u))
    ray = self.reproj.rays_base(np.array([pixel]), self.rig)[0]
    return place_on_plane(origin, ray, self.plane)
