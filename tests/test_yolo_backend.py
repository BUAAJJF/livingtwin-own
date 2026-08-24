"""The colour mask path, checked where it can be checked without a robot.

Two things are worth a test here and the rest is not.

The first is the **ONNX decode**.  It is the one piece of this deployment that
reimplements somebody else's arithmetic: ultralytics has a post-processing
stage, and running the exported graph means writing it again.  A decode that is
subtly wrong -- boxes off by the letterbox offset, masks not cropped to their
box, the class column read as a coefficient -- produces masks that look
plausible and are not, and the only way to know is to compare against the
implementation it replaced.

The second is the **geometric rejection**.  ``YoloSegmenter`` was previously a
network and a resize; everything that stops a detection from becoming a target
it should not be is new, and each rule exists because there is a specific thing
in frame it rejects: the room behind the table, the robot's own arm, the bin.
Those are tested directly, with a fake detector, because they have nothing to
do with the network and should not need one to check.

Run with:  micromamba run -n mjlab python -m pytest tests/test_yolo_backend.py -q
"""

from __future__ import annotations

import dataclasses
import glob
import pathlib

import numpy as np
import pytest

import mjlab.tasks  # noqa: F401  -- before piper_push; see deploy/__init__.py

from hardware.deploy import config, mask, rectify, yolo_backend

HERE = pathlib.Path(__file__).resolve().parents[1] / "hardware" / "deploy"
ONNX = HERE / "yolo" / "best.onnx"
PT = HERE / "yolo" / "best.pt"
VAL = sorted(glob.glob(str(HERE / "yolo" / "data" / "images" / "val" / "*.png")))


class _FakeDetector:
  """Returns masks that were handed to it.  The geometry tests need a
  detection at a chosen place, not a network's opinion of one."""

  def __init__(self, masks):
    self.masks = masks

  def __call__(self, img, shape=None):
    return self.masks


def _segmenter(masks, rig=None, reproj=None):
  rig = rig or config.Rig.nominal()
  reproj = reproj or rectify.Reprojector(rig)
  seg = mask.YoloSegmenter.__new__(mask.YoloSegmenter)
  seg.rig, seg.reproj = rig, reproj
  seg.cfg, seg.yolo = mask.SegmenterCfg(), mask.YoloCfg()
  seg.device, seg.decimate = "cpu", 1
  seg.bin_footprint = (config.BIN_CENTER, config.BIN_OUTER)
  seg.detector = _FakeDetector(masks)
  seg.rejected, seg.n_unplaced = [], 0
  seg.plane = (np.array([0.0, 0.0, 1.0]), float(rig.table_z))
  seg.plane_sigma = seg.noise_per_m = 0.0
  return seg


def _table_scene(rig, reproj, boxes=()):
  """A depth frame of the bare table plus boxes at given base-frame spots.

  Returned with the pixel index of each box, so a test can hand exactly those
  pixels to the fake detector.
  """
  H, W = config.D405_HEIGHT, config.D405_WIDTH
  rays = reproj._rays.cpu().numpy()
  R, t = rig.T_base_cam[:3, :3], rig.T_base_cam[:3, 3]
  rb = rays @ R.T
  down = rb[:, 2] < -1e-6
  rng = np.where(down, -t[2] / np.where(down, rb[:, 2], -1.0), 0.0)
  pts = t + rng[:, None] * rb
  depth = rng.astype(np.float32)
  masks = []
  for cx, cy, half, height in boxes:
    sel = ((np.abs(pts[:, 0] - cx) < half) & (np.abs(pts[:, 1] - cy) < half)
           & down)
    depth[sel] -= height * 0.9
    masks.append(sel.reshape(H, W))
  return np.clip(depth, 0, 1.5).reshape(H, W), masks


# --------------------------------------------------------------------------
# the ONNX decode


@pytest.mark.skipif(not (ONNX.exists() and PT.exists()),
                    reason="no exported weights to compare")
def test_the_onnx_decode_agrees_with_ultralytics():
  """Masks from the hand-written decode against the library's own.

  The threshold is 0.80 and the measured median is 0.905 on this model.  It is
  not 1.0 and cannot be: ultralytics upsamples the prototype mask and crops it
  in a different order from this, which is about a pixel of boundary on a
  25-pixel object.  A regression that broke the decode would not land at 0.85,
  it would land near zero.
  """
  cv2 = pytest.importorskip("cv2")
  pytest.importorskip("onnxruntime")
  pytest.importorskip("ultralytics")
  cfg = dataclasses.replace(mask.YoloCfg(), conf=0.02)
  a = yolo_backend.load_detector(str(ONNX), cfg=cfg)
  b = yolo_backend.load_detector(str(PT), cfg=cfg, device="cpu")

  ious, n = [], 0
  for f in VAL[:8]:
    g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    A, B = a(g), b(g)
    n += len(A)
    for m in A:
      if len(B) == 0:
        continue
      j = int(np.argmax([(m & o).sum() / max((m | o).sum(), 1) for o in B]))
      ious.append((m & B[j]).sum() / max((m | B[j]).sum(), 1))
  assert n > 0, "the ONNX model detected nothing at all on the validation set"
  assert float(np.median(ious)) > 0.80, f"median IoU {np.median(ious):.3f}"


@pytest.mark.skipif(not ONNX.exists(), reason="no exported weights")
def test_the_letterbox_is_invertible():
  """The scale and the two offsets have to put a box back where it came from,
  or every detection is displaced by the size of the grey bars."""
  cv2 = pytest.importorskip("cv2")
  img = np.zeros((config.D405_HEIGHT, config.D405_WIDTH), np.uint8)
  img[100:140, 300:360] = 255
  canvas, r, top, left = yolo_backend.letterbox(
    yolo_backend._as_three_channel(img))
  ys, xs = np.nonzero(canvas[:, :, 0] > 127)
  back = (np.array([xs.min(), ys.min(), xs.max(), ys.max()])
          - np.array([left, top, left, top])) / r
  assert np.allclose(back, [300, 100, 359, 139], atol=1.5), back
  del cv2


def test_a_grayscale_frame_is_accepted():
  """The deployment's image is ``(H, W)`` -- the sensor's mono frame -- and a
  detector that only takes three channels would fail on the robot and nowhere
  else."""
  out = yolo_backend._as_three_channel(np.zeros((4, 5), np.uint8))
  assert out.shape == (4, 5, 3)
  with pytest.raises(ValueError):
    yolo_backend._as_three_channel(np.zeros((4, 5, 4), np.uint8))


# --------------------------------------------------------------------------
# the geometric rejection


def test_a_detection_outside_the_workspace_is_rejected():
  """The failure this exists for: a network trained on one table firing on the
  room behind it.  Nothing in the network knows the room is not the table."""
  rig = config.Rig.nominal()
  reproj = rectify.Reprojector(rig)
  depth, masks = _table_scene(rig, reproj, [(0.35, 0.05, 0.022, 0.05)])
  H, W = depth.shape
  inside = masks[0]
  # Something at the top of the frame, which looks across the table and out
  # into the room -- the same pixels the background ablation showed are 44% of
  # what the policy sees.
  outside = np.zeros((H, W), bool)
  outside[:40, W // 2 - 30:W // 2 + 30] = True

  seg = _segmenter(np.stack([inside, outside]))
  out = seg(depth, rgb=np.zeros((H, W), np.uint8))
  assert len(out.instances) == 1, [r for r in seg.rejected]
  assert any(r[0] == "outside the workspace" for r in seg.rejected), seg.rejected


def test_a_detection_on_the_arm_is_rejected():
  """The gripper is the most object-shaped thing in frame that is not an
  object, and the model was trained on frames it was in."""
  rig = config.Rig.nominal()
  reproj = rectify.Reprojector(rig)
  depth, masks = _table_scene(rig, reproj, [(0.35, 0.05, 0.022, 0.05)])
  seg = _segmenter(np.stack(masks))
  # A sphere cover that swallows the object's own position.
  arm = (np.array([[0.35, 0.05, 0.02]]), np.array([0.08]))
  out = seg(depth, rgb=np.zeros(depth.shape, np.uint8), arm=arm)
  assert len(out.instances) == 0
  assert any(r[0] in ("arm", "the arm") for r in seg.rejected), seg.rejected


def test_a_detection_on_the_bin_is_rejected():
  """Its rim is 60 mm of vertical wall standing in the middle of the range the
  objects occupy."""
  rig = config.Rig.nominal()
  reproj = rectify.Reprojector(rig)
  bx, by = config.BIN_CENTER
  depth, masks = _table_scene(rig, reproj, [(bx, by, 0.022, 0.05)])
  seg = _segmenter(np.stack(masks))
  out = seg(depth, rgb=np.zeros(depth.shape, np.uint8))
  assert len(out.instances) == 0
  assert any(r[0] == "the bin" for r in seg.rejected), seg.rejected


def test_a_detection_with_no_depth_is_placed_on_the_table():
  """The case the whole backend exists for.

  An object whose pixels are all holes used to come back with a NaN centroid,
  which ``TargetTracker`` skips -- so the backend that exists to find objects
  the depth cannot see was dropping exactly those.  The ray through it is now
  intersected with the table plane instead.
  """
  rig = config.Rig.nominal()
  reproj = rectify.Reprojector(rig)
  depth, masks = _table_scene(rig, reproj, [(0.35, 0.05, 0.025, 0.05)])
  holed = depth.copy()
  holed[masks[0]] = 0.0                     # the white-object failure, exactly

  seg = _segmenter(np.stack(masks))
  out = seg(holed, rgb=np.zeros(depth.shape, np.uint8))
  assert len(out.instances) == 1, seg.rejected
  assert seg.n_unplaced == 1
  c = out.instances[0].centroid_base
  assert np.isfinite(c).all(), "an unplaceable instance can never be a target"
  assert np.linalg.norm(c[:2] - np.array([0.35, 0.05])) < 0.03, c
  # And the depth segmenter, on the same frame, finds nothing there -- which is
  # the entire argument for having a second backend.
  d = mask.DepthSegmenter(rig, reproj)(holed)
  assert all(np.linalg.norm(i.centroid_base[:2] - [0.35, 0.05]) > 0.03
             for i in d.instances)


def test_the_fused_backend_adds_what_the_depth_one_missed():
  """Two objects, one of them white.  The depth backend finds one, the fused
  backend finds both, and the one it kept from the depth backend keeps the
  depth backend's outline."""
  rig = config.Rig.nominal()
  reproj = rectify.Reprojector(rig)
  depth, masks = _table_scene(rig, reproj,
                              [(0.32, 0.10, 0.025, 0.05),
                               (0.45, -0.08, 0.025, 0.05)])
  holed = depth.copy()
  holed[masks[1]] = 0.0

  depth_seg = mask.DepthSegmenter(rig, reproj)
  fused = mask.FusedSegmenter(depth_seg, _segmenter(np.stack(masks)))
  alone = depth_seg(holed)
  both = fused(holed, rgb=np.zeros(depth.shape, np.uint8))
  assert len(alone.instances) == 1, [i.centroid_base for i in alone.instances]
  assert len(both.instances) == 2, [i.centroid_base for i in both.instances]
  assert fused.n_from_yolo == 1
  found = sorted(np.linalg.norm(i.centroid_base[:2] - [0.45, -0.08])
                 for i in both.instances)
  assert found[0] < 0.03, found


def test_the_shared_filters_are_the_same_filters():
  """Both backends test "is it in the workspace" with one implementation.

  Written as a test because the failure it guards against is silent: two copies
  of a box that drift apart give a depth backend and a colour backend that
  disagree about where the table ends, and the disagreement only shows up as
  the robot occasionally reaching somewhere it should not.
  """
  pts = np.array([[0.3, 0.0, 0.05], [3.0, 0.0, 0.05], [0.3, 0.0, 1.0]])
  keep = mask.workspace_mask(pts)
  assert list(keep) == [True, False, False]
  (xlo, xhi), _, _ = config.WORKSPACE
  assert mask.workspace_mask(np.array([[xhi + 0.01, 0.0, 0.05]]))[0] == False  # noqa: E712
