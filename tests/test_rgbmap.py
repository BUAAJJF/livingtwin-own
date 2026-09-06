from __future__ import annotations

import numpy as np

from hardware.deploy.rgbmap import MappedSamPredictor, RgbDepthMapper


def _meta(h=12, w=16):
  intr = {"width": w, "height": h, "fx": 20.0, "fy": 20.0,
          "ppx": (w - 1) / 2, "ppy": (h - 1) / 2,
          "model": "distortion.none", "coeffs": [0, 0, 0, 0, 0]}
  return {
    "depth_intrinsics": dict(intr),
    "color_intrinsics": dict(intr),
    "depth_to_color": {
      "rotation": np.eye(3).reshape(-1).tolist(),
      "translation": [0, 0, 0],
    },
  }


def test_identity_projection_samples_the_same_mask():
  mapper = RgbDepthMapper(_meta(), seed_dilate_px=0)
  depth = np.full((12, 16), 0.7, np.float32)
  mask = np.zeros((12, 16), bool)
  mask[3:8, 5:11] = True
  assert np.array_equal(mapper.rgb_masks_to_depth(mask, depth), mask)
  assert np.array_equal(mapper.depth_mask_to_rgb(mask, depth), mask)


def test_invalid_depth_never_creates_a_policy_mask_pixel():
  mapper = RgbDepthMapper(_meta(), seed_dilate_px=0)
  depth = np.full((12, 16), 0.7, np.float32)
  depth[:, :5] = 0
  rgb = np.ones((12, 16), bool)
  out = mapper.rgb_masks_to_depth(rgb, depth)
  assert not out[:, :5].any()
  assert out[:, 5:].all()


def test_rgb_distortion_matches_realsense_projection():
  rs = __import__("pyrealsense2")
  for model in (rs.distortion.brown_conrady,
                rs.distortion.modified_brown_conrady,
                rs.distortion.inverse_brown_conrady):
    meta = _meta()
    meta["color_intrinsics"]["model"] = str(model)
    meta["color_intrinsics"]["coeffs"] = [
      0.01, -0.005, 0.002, -0.001, 0.0002]
    mapper = RgbDepthMapper(meta, seed_dilate_px=0)
    intr = rs.intrinsics()
    c = meta["color_intrinsics"]
    for key in ("width", "height", "fx", "fy", "ppx", "ppy"):
      setattr(intr, key, c[key])
    intr.model, intr.coeffs = model, c["coeffs"]
    x, y = np.array([0.1]), np.array([0.08])
    xd, yd = mapper._distort(x, y)
    actual = np.array([xd[0] * c["fx"] + c["ppx"],
                       yd[0] * c["fy"] + c["ppy"]])
    expected = rs.rs2_project_point_to_pixel(intr, [0.1, 0.08, 1.0])
    assert np.allclose(actual, expected, atol=1e-5)


def test_extrinsic_layout_matches_realsense_transform():
  rs = __import__("pyrealsense2")
  meta = _meta()
  # The SDK exposes this flat array in column-major order.
  meta["depth_to_color"] = {
    "rotation": [0, -1, 0, 1, 0, 0, 0, 0, 1],
    "translation": [0.1, 0.2, 0.3],
  }
  mapper = RgbDepthMapper(meta, seed_dilate_px=0)
  extr = rs.extrinsics()
  extr.rotation = meta["depth_to_color"]["rotation"]
  extr.translation = meta["depth_to_color"]["translation"]
  point = np.array([0.1, 0.08, 1.0])
  expected = rs.rs2_transform_point_to_point(extr, point.tolist())
  actual = mapper.R @ point + mapper.t
  assert np.allclose(actual, expected, atol=1e-6)


class _Predictor:
  def __init__(self, output):
    self.output = output
    self.anchor_mask = None

  def anchor(self, _image, mask):
    self.anchor_mask = mask.copy()

  def reanchor(self, mask):
    self.anchor_mask = mask.copy()

  def propagate(self, _image):
    return self.output.copy()

  def reset(self):
    pass


def test_sam_adapter_keeps_watchdog_masks_on_depth_grid():
  mask = np.zeros((12, 16), bool)
  mask[3:8, 5:11] = True
  base = _Predictor(mask)
  mapped = MappedSamPredictor(
    base, RgbDepthMapper(_meta(), seed_dilate_px=0))
  mapped.set_depth(np.full((12, 16), 0.7, np.float32))
  mapped.anchor(np.zeros((12, 16, 3), np.uint8), mask)
  out = mapped.propagate(np.zeros((12, 16, 3), np.uint8))
  assert np.array_equal(base.anchor_mask, mask)
  assert np.array_equal(out, mask)
  assert np.array_equal(mapped.last_rgb_mask, mask)
