"""Run SAM3+DART over a recorded session and write its instance masks to disk.

This is the half of the segmentation benchmark that cannot live in the
deployment's Python environment.  SAM3 (through DART) pins ``numpy < 2`` and
this repository runs 2.4, so the model runs in its own environment, writes
masks, and ``segbench.py`` replays them through the deployment's own geometry.
That split is not a workaround: it is what makes the comparison honest.  The
masks written here get exactly the workspace box, arm exclusion, bin footprint,
table-plane fit, height/footprint/elongation gates and three-frames-in-five
confirmation that ``mask.YoloSegmenter`` applies to YOLO's, because they are
run through that class.

What this file decides is only what SAM3 is asked and what it is shown.

**What it is asked.**  ``--classes`` are DART's open-vocabulary prompts.  The
task has no semantics -- everything on the table goes in the bin -- so the
prompt is not naming a category to keep, it is naming a thing to *find*, and
the geometry decides afterwards whether it is on the table and object-sized.
A narrow prompt misses an object shape it was not told about; a broad one costs
a forward pass per class and hands the geometry more to reject.  Both are
measurable and that is why this is a flag.

**What it is shown.**  The recording holds ``gray``, which is the D455's colour
stream converted to grayscale and warped into the depth grid -- so it is black
wherever the depth dropped, which is 14% of the frame and concentrated on
exactly the edges an object is made of.  This is a real handicap for a model
trained on natural colour images and the report has to say so rather than
average it away.  ``--inpaint`` fills the holes so the cost of the warping can
be measured separately from the cost of the greyness.

    python hardware/deploy/sam3_infer.py \
        --session recordings/v4_stereo_repro_scene2 \
        --out results/segbench/v4_stereo_repro_scene2/masks_sam3 \
        --classes "small block" "small box"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import time

import cv2
import numpy as np


def build(args):
  """The predictor, and the one-time cost of building it."""
  import torch
  from sam3.model_builder import build_sam3_image_model

  ckpt = args.checkpoint
  model = build_sam3_image_model(
    device=args.device, checkpoint_path=(None if ckpt in (None, "auto") else ckpt),
    load_from_HF=(ckpt in (None, "auto")), enable_segmentation=True)
  if args.fast:
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
    p = Sam3MultiClassPredictorFast(model, device=args.device,
                                    resolution=args.imgsz, use_fp16=True)
  else:
    from sam3.model.sam3_multiclass import Sam3MultiClassPredictor
    p = Sam3MultiClassPredictor(model, device=args.device,
                                resolution=args.imgsz, detection_only=False)
  p.set_classes(list(args.classes))
  return p, torch


def frame_image(z, inpaint: bool):
  """The recorded frame as the image SAM3 expects, and its shape.

  Grey replicated to three channels rather than colourised: the channels are
  identical and any mapping that makes them differ is inventing colour the
  sensor did not record.

  Returned as a PIL image and not as an array, which is not cosmetic.
  ``Sam3MultiClassPredictor.set_image`` takes ``shape[-2:]`` for an ndarray,
  which is correct for CHW and silently wrong for the HWC frame this is: an
  848x480 frame comes back with 848x3 masks and no error.  PIL carries width
  and height unambiguously.
  """
  from PIL import Image

  g = np.asarray(z["gray"], dtype=np.uint8)
  if inpaint:
    holes = (g == 0).astype(np.uint8)
    if holes.any():
      g = cv2.inpaint(g, holes, 3, cv2.INPAINT_TELEA)
  a = np.repeat(g[:, :, None], 3, axis=2)
  return Image.fromarray(a), a.shape[:2]


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--session", type=pathlib.Path, required=True)
  p.add_argument("--out", type=pathlib.Path, required=True)
  p.add_argument("--classes", nargs="+", default=["small block"])
  p.add_argument("--checkpoint", default="auto",
                 help="'auto' downloads facebook/sam3 from HuggingFace")
  p.add_argument("--imgsz", type=int, default=1008,
                 help="must be divisible by 14")
  p.add_argument("--confidence", type=float, default=0.3)
  p.add_argument("--nms", type=float, default=0.7)
  p.add_argument("--device", default="cuda")
  p.add_argument("--fast", action="store_true",
                 help="DART's batched FP16 path with presence early-exit")
  p.add_argument("--inpaint", action="store_true",
                 help="fill the depth-warp holes in the grayscale first")
  p.add_argument("--limit", type=int, default=None)
  p.add_argument("--warmup", type=int, default=3)
  a = p.parse_args()

  files = sorted(glob.glob(str(a.session / "*.npz")))
  if not files:
    print(f"no frames in {a.session}", file=sys.stderr)
    return 1
  if a.limit:
    files = files[:a.limit]
  a.out.mkdir(parents=True, exist_ok=True)

  predictor, torch = build(a)

  warm, _ = frame_image(np.load(files[0]), a.inpaint)
  for _ in range(a.warmup):
    predictor.predict_image(warm, confidence_threshold=a.confidence,
                            nms_threshold=a.nms)
  if a.device.startswith("cuda"):
    torch.cuda.synchronize()

  ms, n_det = [], []
  t_start = time.perf_counter()
  for k, f in enumerate(files):
    key = os.path.basename(f).split(".")[0]
    img, hw = frame_image(np.load(f), a.inpaint)
    t0 = time.perf_counter()
    r = predictor.predict_image(img, confidence_threshold=a.confidence,
                                nms_threshold=a.nms)
    if a.device.startswith("cuda"):
      torch.cuda.synchronize()
    ms.append((time.perf_counter() - t0) * 1e3)

    masks = r.get("masks")
    masks = (np.zeros((0, *hw), dtype=bool) if masks is None
             else np.asarray(_to_numpy(masks)).astype(bool).reshape(-1, *hw))
    scores = np.asarray(_to_numpy(r.get("scores")), dtype=np.float32).reshape(-1)
    boxes = np.asarray(_to_numpy(r.get("boxes")), dtype=np.float32).reshape(-1, 4)
    cls = np.asarray(_to_numpy(r.get("class_ids")), dtype=np.int32).reshape(-1)
    n = masks.shape[0]
    n_det.append(n)
    np.savez_compressed(
      a.out / f"{key}.npz", n=np.int32(n), hw=np.asarray(hw, np.int32),
      masks=(np.packbits(masks.reshape(n, -1), axis=1) if n else
             np.zeros((0, 0), dtype=np.uint8)),
      scores=scores, boxes=boxes, class_ids=cls)
    if k % 200 == 0:
      done = k + 1
      rate = done / max(time.perf_counter() - t_start, 1e-6)
      print(f"{done}/{len(files)}  {np.median(ms):.0f} ms/frame  "
            f"{rate:.1f} fps  {np.mean(n_det):.2f} det/frame", flush=True)

  meta = {
    "session": str(a.session), "frames": len(files), "classes": list(a.classes),
    "imgsz": a.imgsz, "confidence": a.confidence, "nms": a.nms,
    "fast": bool(a.fast), "inpaint": bool(a.inpaint),
    "device": a.device, "checkpoint": a.checkpoint,
    "ms": {"median": round(float(np.median(ms)), 1),
           "p95": round(float(np.percentile(ms, 95)), 1)},
    "detections_per_frame": round(float(np.mean(n_det)), 2),
  }
  (a.out / "_meta.json").write_text(json.dumps(meta, indent=1))
  print(json.dumps(meta, indent=1))
  return 0


def _to_numpy(x):
  if x is None:
    return np.zeros(0)
  return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


if __name__ == "__main__":
  sys.exit(main())
