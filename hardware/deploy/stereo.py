"""Depth from the raw imagers, through Fast-FoundationStereo on TensorRT.

The camera already ships a depth map, so this needs a reason to exist.  The
bench measured both on this unit on 2026-09-01, same frames, emitter on:

    native fill   88.6%      FoundationStereo fill   73.2%
    where both are defined, the medians differ by 3.7 mm

So the learned model is neither more complete nor measurably more accurate
here, and it costs 14.4 ms of GPU per frame.  What it does offer is a depth
map whose failures are *different* ones: the ASIC drops out on low texture and
at edges, and a network trained on passive stereo does not fail in the same
places.  That is worth having as an option and it is not worth having as the
default, which is exactly how it is wired -- ``--depth-source sensor`` stays
the default and this is opt-in.

It is also, on the evidence, not the fix for anything currently broken.  The
target losses that drove the deployment failure were measured on 2026-09-01
and *none* of them were depth: every pixel of the object was present with
valid depth in the frames where the segmenter reported nothing.  Reach for
this when depth is the problem, and know that today it is not.

Geometry: the engine returns disparity in the LEFT imager, and RealSense
defines the depth frame as that same imager, so ``depth = fx * b / disparity``
lands on the grid the rig was calibrated against with no further transform.
``fx * b`` is read from the device rather than the rig file, because the rig's
intrinsics describe the depth stream and this is the one place where the two
could drift apart.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_BENCH = (pathlib.Path(__file__).resolve().parents[1]
          / "depth_bench" / "foundation_stereo_bench")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_ENGINE = (_BENCH / "results" / "trt_fast_20-30-48_i4_480x864"
                  / "fast_foundationstereo_fp16.engine")
"""The 4-iteration engine: 14.42 ms against the 8-iteration engine's 17.32 ms,
and the two agree to a median 0.034 px of disparity.  The extra iterations buy
nothing this deployment can use."""


class StereoDepth:
  """Left/right uint8 in, metres out, on the depth grid.

  Holds a TensorRT context and CUDA buffers, so it belongs to one thread --
  the perception thread -- and must not be shared with the control loop.
  """

  def __init__(self, engine: str | pathlib.Path | None = None,
               width: int = 848, height: int = 480,
               focal_baseline: float | None = None,
               min_depth: float = 0.20, max_depth: float = 3.0) -> None:
    # TensorRT is vendored under the bench rather than installed into the
    # deployment environment, and it must be found without the caller having
    # to set PYTHONPATH -- a control loop that only runs when someone
    # remembered an environment variable is a control loop that will one day
    # run with the wrong depth.
    for d in (_BENCH, _BENCH / "_deps_trt"):
      if d.is_dir() and str(d) not in sys.path:
        sys.path.insert(0, str(d))
    import torch                                    # noqa: E402
    try:
      from benchmark_trt import Runner              # noqa: E402
    except ModuleNotFoundError as e:                # pragma: no cover
      raise ModuleNotFoundError(
        f"{e}.  The TensorRT wheels live in {_BENCH / '_deps_trt'}; if that "
        f"directory is missing, rebuild it before using --depth-source stereo"
      ) from e

    self._torch = torch
    self.path = pathlib.Path(engine or DEFAULT_ENGINE).resolve()
    if not self.path.exists():
      raise FileNotFoundError(
        f"no TensorRT engine at {self.path}.  Build one with "
        f"{_BENCH / 'build_trt_engine.py'}")
    torch.autograd.set_grad_enabled(False)
    self.runner = Runner(self.path)
    if len(self.runner.inputs) != 2 or len(self.runner.outputs) != 1:
      raise RuntimeError(
        f"expected two inputs and one output, got {self.runner.inputs} / "
        f"{self.runner.outputs}")

    shape = tuple(self.runner.engine.get_tensor_shape(self.runner.inputs[0]))
    self.pad = shape[-1] - width
    if shape[-2] != height or self.pad < 0:
      raise ValueError(
        f"engine input {shape} does not cover {width}x{height}")
    self.left_pad = self.pad // 2
    self.width, self.height = int(width), int(height)
    self.focal_baseline = focal_baseline
    self.min_depth, self.max_depth = float(min_depth), float(max_depth)
    self._mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
    self._std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
    self._out = self.runner.allocate_outputs()

  def calibrate(self, meta: dict) -> "StereoDepth":
    """Take ``fx * baseline`` from the camera the reader actually opened."""
    fb = meta.get("stereo_focal_baseline")
    if fb is None:
      fx = meta.get("fx_px")
      b = meta.get("stereo_baseline_m")
      fb = None if (fx is None or b is None) else float(fx) * float(b)
    if fb is None or not np.isfinite(fb) or fb <= 0:
      raise ValueError(
        "the camera did not report a stereo baseline, so disparity cannot be "
        "turned into metres; pass focal_baseline= explicitly")
    self.focal_baseline = float(fb)
    return self

  def _prep(self, gray: np.ndarray):
    torch = self._torch
    x = torch.from_numpy(np.ascontiguousarray(gray)).to(
      device="cuda", dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = (x.expand(-1, 3, -1, -1) / 255.0 - self._mean) / self._std
    return torch.nn.functional.pad(
      x, (self.left_pad, self.pad - self.left_pad, 0, 0),
      mode="replicate").contiguous()

  def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Depth in metres on the ``(height, width)`` depth grid, 0 where invalid.

    Zero, not NaN, and not a far value: that is what the sensor path returns
    for a missing reading and every stage downstream -- reprojection, the
    table fit, the segmenter -- already means "no measurement" by it.
    """
    if self.focal_baseline is None:
      raise RuntimeError("call calibrate() with the reader's meta first")
    if left.shape[:2] != (self.height, self.width):
      raise ValueError(f"left image is {left.shape[:2]}, expected "
                       f"{(self.height, self.width)}")
    if right.shape[:2] != left.shape[:2]:
      raise ValueError("left and right images differ in shape")

    self.runner.bind({self.runner.inputs[0]: self._prep(left),
                      self.runner.inputs[1]: self._prep(right)})
    self.runner.bind(self._out)
    self.runner.execute()
    raw = self._out[self.runner.outputs[0]]
    disp = raw[..., self.left_pad:self.left_pad + self.width]
    disp = disp.float().squeeze().cpu().numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
      depth = self.focal_baseline / disp
    # A disparity at or below zero is the model saying it does not know, and
    # the reciprocal of a near-zero disparity is a point at infinity that
    # would otherwise become a very confident wrong reading on the table.
    bad = ~np.isfinite(depth) | (disp <= 0)
    bad |= (depth < self.min_depth) | (depth > self.max_depth)
    depth[bad] = 0.0
    return depth.astype(np.float32)
