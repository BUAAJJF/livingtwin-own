"""Does the colour model earn its place -- measured where it is supposed to.

``train_yolo.py --check`` prints mAP and says, in the file, that mAP is not the
question: the validation set is the frames the depth segmenter *could* label,
so mAP measures how well the student copied the teacher on the teacher's good
days.  The question is whether the student works where the teacher does not,
and no set built that way can answer it.

This answers it, in two parts, and neither needs a hand label.

**Agreement, where the depth works.**  Both backends run over a recording and
their instances are matched by position.  This is the sanity check: a student
that disagrees with its teacher on the easy frames has not learned the task,
and the number to look at is the fraction of the teacher's instances the
student also found, not the IoU of the ones it did.

**Coverage, where the depth does not.**  The failure the colour model exists
for is a surface the stereo matcher cannot match, and its signature is not a
noisier depth -- it is *no* depth, on the object, with the table around it
intact.  So that is what gets simulated: the depth inside each object is
knocked out at the rate the bench measured on a blank white surface (fill 88%
average, 41.9% worst), the two backends are re-run, and the count is of objects
that survive.  A depth segmenter has nothing to segment there by construction;
if the colour model does not hold up, the whole path is decoration.

The ablation is honest in one direction only, and it is worth saying which.  It
removes depth without changing the image, and on a real white object the image
changes too -- less contrast, softer edges.  So this is the *optimistic* half
of the real failure, and a model that fails here fails on the robot for
certain, while a model that passes here has only cleared the easier half.

    python -m hardware.deploy.yolo_eval recordings/sim
    python -m hardware.deploy.yolo_eval recordings/sim --weights yolo/best.onnx
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

from . import config, mask, proprio, rectify

WHITE_FILL = (0.88, 0.419)
"""What the bench measured on a blank white surface: 88% of pixels filled on
average and 41.9% in the worst shot.  Both are used -- the average is the case
that happens most and the worst is the case that decides."""


def _knockout(depth: np.ndarray, regions: np.ndarray, fill: float,
              rng: np.random.Generator) -> np.ndarray:
  """Drop depth inside ``regions`` down to ``fill``, leaving the rest alone.

  Blob-wise rather than per-pixel.  The measured dropout is correlated across
  8.5 pixels (``piper_push.depth_noise``), and knocking out independent pixels
  would leave a peppered mask that both a connected-components pass and a
  network sail straight through -- which would make this test pass for the
  wrong reason.
  """
  out = depth.copy()
  sel = regions > 0
  if not sel.any() or fill >= 1.0:
    return out
  h, w = depth.shape
  # A smooth random field thresholded at the required quantile: correlated
  # holes, of the right total area, without a loop.
  import cv2

  field = rng.standard_normal((h, w)).astype(np.float32)
  field = cv2.GaussianBlur(field, (0, 0), 8.5)
  thresh = float(np.quantile(field[sel], fill))
  out[sel & (field > thresh)] = 0.0
  return out


def _match(a: list, b: list, radius: float) -> int:
  """How many of ``a`` have a partner in ``b`` within ``radius`` metres."""
  if not a or not b:
    return 0
  pb = np.stack([i.centroid_base for i in b])
  ok = np.isfinite(pb).all(axis=1)
  pb = pb[ok]
  if pb.shape[0] == 0:
    return 0
  n = 0
  for i in a:
    if not np.isfinite(i.centroid_base).all():
      continue
    if float(np.linalg.norm(pb - i.centroid_base, axis=1).min()) < radius:
      n += 1
  return n


def run(session: pathlib.Path, weights: str, device: str, conf: float,
        stride: int, limit: int | None, radius: float, seed: int) -> dict:
  rig_file = session / "rig.json"
  rig = config.Rig.load(rig_file) if rig_file.exists() else config.Rig.load()
  reproj = rectify.Reprojector(rig)
  depth_seg = mask.DepthSegmenter(rig, reproj)
  yolo_seg = mask.YoloSegmenter(
    weights, rig, reproj, device=device,
    yolo_cfg=dataclasses.replace(mask.YoloCfg(), conf=conf))
  kin = proprio.Kinematics()
  rng = np.random.default_rng(seed)

  meta = json.loads((session / "meta.json").read_text())
  rows = []
  for rec in meta[::stride][:limit]:
    f = np.load(session / f"{rec['i']:06d}.npz")
    depth = f["depth"].astype(np.float32) / 10000.0
    gray = f["gray"]
    kin.update(np.asarray(rec["joint_pos"], dtype=np.float64))
    arm = kin.link_spheres()

    seg_d = depth_seg(depth, arm=arm)
    seg_y = yolo_seg(depth, rgb=gray, arm=arm)

    # The regions to knock out: what the depth segmenter says is an object.
    regions = np.zeros_like(depth, dtype=np.uint8)
    for inst in seg_d.instances:
      m = mask.full_mask(seg_d, inst.label, depth_seg.decimate) > 0
      regions[:m.shape[0], :m.shape[1]] |= m[:regions.shape[0],
                                             :regions.shape[1]].astype(np.uint8)

    row = {
      "i": rec["i"],
      "fill": float((depth > 0).mean()),
      "depth_n": len(seg_d.instances),
      "yolo_n": len(seg_y.instances),
      "yolo_unplaced": int(yolo_seg.n_unplaced),
      "agree": _match(seg_d.instances, seg_y.instances, radius),
    }
    for name, fill in (("avg", WHITE_FILL[0]), ("worst", WHITE_FILL[1])):
      holed = _knockout(depth, regions, fill, rng)
      row[f"depth_n_{name}"] = len(depth_seg(holed, arm=arm).instances)
      row[f"yolo_n_{name}"] = len(yolo_seg(holed, rgb=gray, arm=arm).instances)
    rows.append(row)
  return {"rows": rows, "session": str(session), "weights": str(weights),
          "conf": conf}


def report(res: dict) -> int:
  rows = res["rows"]
  if not rows:
    print("no frames")
    return 1
  g = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731

  print(f"{len(rows)} frames from {res['session']}, "
        f"{pathlib.Path(res['weights']).name} at conf {res['conf']}\n")

  d, y = g("depth_n"), g("yolo_n")
  print("as recorded")
  print(f"  instances per frame     depth {d.mean():5.2f}   yolo {y.mean():5.2f}")
  print(f"  frames with none        depth {100 * (d == 0).mean():5.1f}%  "
        f"yolo {100 * (y == 0).mean():5.1f}%")
  found = g("agree").sum()
  print(f"  of the depth backend's {int(d.sum())} instances, the colour model "
        f"also found {int(found)} ({100 * found / max(d.sum(), 1):.1f}%)")
  unplaced = g("yolo_unplaced").sum()
  print(f"  colour detections placed on the table plane rather than their own "
        f"depth: {int(unplaced)}")

  print("\nwith the object's depth knocked out, at the rate the bench measured")
  print(f"{'':22s} {'depth backend':>15s} {'colour backend':>16s}")
  for name, label in (("avg", "88% fill (average)"),
                      ("worst", "42% fill (worst)")):
    dn, yn = g(f"depth_n_{name}"), g(f"yolo_n_{name}")
    print(f"  {label:20s} {dn.mean():8.2f} /frame {yn.mean():11.2f} /frame")
    print(f"  {'':20s} {100 * (dn == 0).mean():7.1f}% empty "
          f"{100 * (yn == 0).mean():10.1f}% empty")

  # The verdict, stated rather than left to be read off.
  keep = g("depth_n_worst").mean() / max(d.mean(), 1e-9)
  ykeep = g("yolo_n_worst").mean() / max(y.mean(), 1e-9)
  print(f"\nat the worst measured fill the depth backend keeps "
        f"{100 * keep:.0f}% of what it found and the colour model keeps "
        f"{100 * ykeep:.0f}%.")
  if y.mean() < 0.5 * d.mean():
    print("\nThe colour model is finding much less than the depth segmenter on "
          "frames where\nthe depth is fine, so the comparison above is not "
          "about the ablation -- it is a\nmodel that has not been trained "
          "enough.  Fix that before reading anything else\nhere: more "
          "recordings, more epochs, and `train_yolo.py --check` to confirm it "
          "learned\nthe teacher at all.")
  return 0


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("session", help="a directory written by run.py --record")
  p.add_argument("--weights", default=str(
    pathlib.Path(__file__).resolve().parent / "yolo" / "best.pt"))
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--conf", type=float, default=mask.YoloCfg().conf)
  p.add_argument("--stride", type=int, default=4,
                 help="frames at 50 Hz are near-duplicates")
  p.add_argument("--limit", type=int, default=None)
  p.add_argument("--radius", type=float, default=0.05,
                 help="metres within which two backends mean the same object")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--json", default=None)
  a = p.parse_args()

  res = run(pathlib.Path(a.session), a.weights, a.device, a.conf, a.stride,
            a.limit, a.radius, a.seed)
  if a.json:
    pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
  return report(res)


if __name__ == "__main__":
  sys.exit(main())
