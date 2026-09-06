"""Run a YOLO26-seg model without ultralytics, and export one so you can.

``mask.YoloSegmenter`` needs instance masks and nothing else, and there are two
ways to get them.  This file exists because the deployment should not have to
import torch to segment an image -- not because doing so is slow.  That was the
assumption and it is wrong, measured on this machine over 30 frames of
848x480:

    ultralytics (.pt) on CUDA      8.3 ms   p95 12.0    120 Hz
    onnxruntime (.onnx) on CUDA   13.4 ms   p95 15.9     75 Hz
    onnxruntime (.onnx) on CPU    60.8 ms   p95 65.8     16 Hz   (4 threads)

So torch is the fast path and ONNX is the portable one, and the reason to reach
for ONNX is that it drops ultralytics, torch and a CUDA context from the robot
-- which matters if the robot is not this machine -- and not that it is
quicker.  On CPU it is too slow for the camera's rate and the perception thread
would run at 16 Hz; that is a real number to plan around and not a footnote.

So there are two detectors behind one call:

``UltralyticsDetector`` is what you develop against.  It reads ``.pt``, it is
the reference, and it is what ``train_yolo.py`` produces.

``OnnxDetector`` is what runs on the robot.  It reads ``.onnx`` and imports
numpy, cv2 and onnxruntime -- all three of which are already in the deployment
because the depth path and the policy need them.  Nothing else.

The decode is written against the export this repository actually produces and
was checked against it, which matters because YOLO26's head is not the one
every decoder on the internet is written for.  ``yolo26n-seg`` exports
``end2end=True``:

    images    (1, 3, 832, 832)  float32, RGB, 0-1, letterboxed with 114
    output0   (1, 300, 38)      x1 y1 x2 y2 conf cls, then 32 mask coefficients
    output1   (1, 32, 208, 208) mask prototypes

Three consequences.  The boxes are already corners in letterboxed input pixels,
not centre-width-height and not normalised.  There is **no NMS to run** -- the
head is end-to-end, the 300 rows are the final answer, and adding NMS here
would only remove detections the model meant to keep.  And the class column is
carried but useless: there is one class, "a thing on the table", because
everything on the table goes in the bin.

Checked against ultralytics on the 24 validation frames: boxes agree to 0.57
pixels and masks to a median IoU of 0.912, worst 0.864.  The residual is the
resampling order -- ultralytics upsamples the prototype mask and then crops it
to the box, this crops in the same order but resizes twice -- and it is about a
pixel of boundary on a 25-pixel object.  It is not zero and the number is here
so that nobody has to rediscover that it is not zero.

    python -m hardware.deploy.yolo_backend --export      # .pt -> .onnx
    python -m hardware.deploy.yolo_backend --bench       # what it costs a frame
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
YOLO_DIR = HERE / "yolo_d455"
"""The D455 model.  The D405 set is in ``yolo/``; ``run.py`` picks the
directory from ``--camera`` and refuses a model belonging to the other one."""

IMGSZ = 832
"""Close to the sensor's 848 wide, so the network sees a 40 mm object at the 25
pixels it actually covers rather than at the 9 a 640 crop would leave.  It has
to match what ``train_yolo.py`` trained at; the export bakes it in."""

PAD_VALUE = 114
"""Ultralytics' letterbox grey.  Not a free choice -- the model was trained
with this exact value in the bars and a different one is a different input."""


def _as_three_channel(img: np.ndarray) -> np.ndarray:
  """Accept a gray compatibility image or the deployment's raw BGR image.

  ONNX preprocessing below converts OpenCV/RealSense BGR to model RGB. Older
  checkpoints were trained from single-channel depth-aligned images expanded
  to three channels; code compatibility does not make those weights adapted
  to the new raw-colour input domain.
  """
  a = np.asarray(img)
  if a.ndim == 2:
    return np.repeat(a[:, :, None], 3, axis=2)
  if a.ndim == 3 and a.shape[2] == 3:
    return a
  raise ValueError(f"expected (H, W) or (H, W, 3), got {a.shape}")


def letterbox(img: np.ndarray, size: int = IMGSZ):
  """Resize keeping the aspect ratio, pad to square.  Returns
  ``(canvas, scale, top, left)`` -- the three numbers needed to undo it."""
  h, w = img.shape[:2]
  r = min(size / h, size / w)
  nh, nw = round(h * r), round(w * r)
  canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
  top, left = (size - nh) // 2, (size - nw) // 2
  canvas[top:top + nh, left:left + nw] = cv2.resize(
    img, (nw, nh), interpolation=cv2.INTER_LINEAR)
  return canvas, r, top, left


class OnnxDetector:
  """YOLO26-seg through onnxruntime.  Returns boolean masks on the source grid.

  ``providers`` defaults to CPU, and that is a decision rather than a fallback.
  The policy also runs on CPU with a capped thread pool for a measured reason
  -- onnxruntime's own pool is not governed by ``OMP_NUM_THREADS`` and
  uncapped it starved the perception thread from 22 ms to 878 ms -- and the
  same trap is here.  The cap is on by default and can be raised by someone who
  has measured that it helps.
  """

  def __init__(self, weights, cfg=None, providers=None, threads: int = 4):
    import onnxruntime as ort

    from .mask import YoloCfg

    self.cfg = cfg or YoloCfg()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(threads)
    opts.inter_op_num_threads = 1
    # See ``policy.Policy``: a spinning pool burns a core between frames, and
    # this thread has a 20 ms neighbour.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    self.session = ort.InferenceSession(
      str(weights), sess_options=opts,
      providers=list(providers or ["CPUExecutionProvider"]))
    self.input_name = self.session.get_inputs()[0].name
    shape = self.session.get_inputs()[0].shape
    self.imgsz = int(shape[2]) if isinstance(shape[2], int) else IMGSZ
    outs = self.session.get_outputs()
    if len(outs) != 2:
      raise RuntimeError(
        f"{weights} has {len(outs)} outputs, not the (detections, prototypes) "
        "pair a segmentation export produces.  Exporting a detection model "
        "instead of a segmentation one is the usual cause."
      )
    self.n_proto = 32

  def __call__(self, img: np.ndarray, shape=None) -> np.ndarray:
    """``(K, H, W)`` boolean masks on the grid of ``shape`` (default: the
    image's own)."""
    img = _as_three_channel(img)
    h, w = shape if shape is not None else img.shape[:2]
    canvas, r, top, left = letterbox(img, self.imgsz)
    x = np.ascontiguousarray(
      canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    det, proto = self.session.run(None, {self.input_name: x})

    d = det[0]
    keep = d[:, 4] > self.cfg.conf
    d = d[keep][:self.cfg.max_detections]
    if d.shape[0] == 0:
      return np.zeros((0, h, w), dtype=bool)

    p = proto[0]
    nm, mh, mw = p.shape
    # Sigmoid of a linear combination of the prototypes -- the whole of YOLO's
    # mask head, once the coefficients are out.
    z = d[:, 6:6 + nm] @ p.reshape(nm, -1)
    inst = (1.0 / (1.0 + np.exp(-z))).reshape(-1, mh, mw)

    boxes = (d[:, :4] - np.array([left, top, left, top])) / r
    nh, nw = round(h * r), round(w * r)
    out = np.zeros((d.shape[0], h, w), dtype=bool)
    for i in range(d.shape[0]):
      m = cv2.resize(inst[i], (self.imgsz, self.imgsz),
                     interpolation=cv2.INTER_LINEAR)
      m = m[top:top + nh, left:left + nw]
      m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
      a = m > self.cfg.mask_threshold
      # Cropped to its own box.  The prototypes are global -- one linear
      # combination covers the whole image -- so a coefficient vector that
      # describes this object also lights up every other object that looks like
      # it, and without the crop two instances come back as one.
      x1, y1, x2, y2 = boxes[i]
      box = np.zeros_like(a)
      box[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1] = True
      out[i] = a & box
    return out


class UltralyticsDetector:
  """The reference path.  Reads ``.pt``, needs torch, and is what the ONNX
  decode above was checked against."""

  def __init__(self, weights, cfg=None, device: str = "cuda:0"):
    from ultralytics import YOLO

    from .mask import YoloCfg

    self.cfg = cfg or YoloCfg()
    self.model = YOLO(str(weights))
    self.device = device

  def __call__(self, img: np.ndarray, shape=None) -> np.ndarray:
    img = _as_three_channel(img)
    h, w = shape if shape is not None else img.shape[:2]
    res = self.model.predict(img, imgsz=IMGSZ, conf=self.cfg.conf,
                             device=self.device, verbose=False)[0]
    if res.masks is None:
      return np.zeros((0, h, w), dtype=bool)
    raw = res.masks.data.cpu().numpy()[:self.cfg.max_detections]
    out = np.zeros((raw.shape[0], h, w), dtype=bool)
    for i, m in enumerate(raw):
      out[i] = cv2.resize(m.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_NEAREST) > 0.5
    return out


def load_detector(weights, device: str = "cuda:0", cfg=None, **kw):
  """Pick a backend from the file name.

  ``.onnx`` goes to onnxruntime and anything else to ultralytics, so
  ``--yolo-weights best.onnx`` is the only change needed to deploy without
  torch.
  """
  path = pathlib.Path(weights)
  if not path.exists():
    raise FileNotFoundError(
      f"no YOLO weights at {path}.  Record a session with `run.py --record`, "
      "label it with `autolabel.py`, train with `train_yolo.py` -- or run "
      "with `--mask depth`, which needs no model at all."
    )
  if path.suffix == ".onnx":
    return OnnxDetector(path, cfg=cfg, **kw)
  return UltralyticsDetector(path, cfg=cfg, device=device)


def export(weights=None, imgsz: int = IMGSZ, opset: int = 17) -> pathlib.Path:
  """``.pt`` to ``.onnx``, at the size the model was trained at."""
  from ultralytics import YOLO

  weights = pathlib.Path(weights or (YOLO_DIR / "best.pt"))
  if not weights.exists():
    raise FileNotFoundError(f"no weights at {weights}")
  out = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=opset,
                                  simplify=False, dynamic=False)
  return pathlib.Path(out)


def _bench(weights, frames: int, device: str) -> None:
  from . import config

  rng = np.random.default_rng(0)
  img = rng.integers(0, 255, (config.D405_HEIGHT, config.D405_WIDTH),
                     dtype=np.uint8)
  det = load_detector(weights, device=device)
  det(img)                       # the first call builds the graph
  ts = []
  for _ in range(frames):
    t0 = time.perf_counter()
    det(img)
    ts.append((time.perf_counter() - t0) * 1e3)
  ts = np.sort(np.asarray(ts))
  print(f"{pathlib.Path(weights).name}: p50 {np.median(ts):.1f} ms   "
        f"p95 {ts[int(0.95 * len(ts))]:.1f} ms   "
        f"worst {ts[-1]:.1f} ms   ({1000 / np.median(ts):.1f} Hz)")
  print("  The perception thread runs at the camera's rate, not the control "
        "loop's, so\n  what this has to beat is the frame period, not the 20 "
        "ms control period.")


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--export", action="store_true")
  p.add_argument("--bench", action="store_true")
  p.add_argument("--weights", default=None)
  p.add_argument("--frames", type=int, default=50)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()

  if a.export:
    print(f"wrote {export(a.weights)}")
  if a.bench:
    w = a.weights or (YOLO_DIR / "best.pt")
    _bench(w, a.frames, a.device)
  if not (a.export or a.bench):
    p.print_help()
  return 0


if __name__ == "__main__":
  sys.exit(main())
