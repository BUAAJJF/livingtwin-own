"""Convert sparse, visually confirmed GUI captures to a YOLO-seg dataset.

Unlike video autolabelling, consecutive files are different arrangements and
therefore cannot pass a temporal tracker. The capture GUI already shows the
depth teacher's outlines and enables Save only for the expected instance
count. This converter repeats that exact geometric check on the stored frame,
applies depth-quality gates, and records every rejection in a manifest.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from . import config, mask, rectify
from .autolabel import polygons, write_dataset


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("sessions", nargs="+")
  p.add_argument("--out", required=True)
  p.add_argument("--min-fill", type=float, default=None,
                 help="whole-frame valid depth; default is max(0.50, p10)")
  p.add_argument("--min-instance-fill", type=float, default=0.80)
  a = p.parse_args()

  sessions = [pathlib.Path(x) for x in a.sessions]
  out = pathlib.Path(a.out)
  if out.exists() and any(out.iterdir()):
    raise SystemExit(f"refusing to overwrite non-empty dataset {out}")

  all_items = []
  manifest = {"sessions": [], "accepted": 0, "rejected": {}}
  for session in sessions:
    meta = json.loads((session / "meta.json").read_text())
    capture = json.loads((session / "capture.json").read_text())
    if not capture.get("manual_capture"):
      raise SystemExit(f"{session} is not a manual GUI capture")
    expected = int(capture.get("expected_objects", 0))
    if expected <= 0:
      raise SystemExit(f"{session} has no positive expected object count")
    rig = config.Rig.load(session / "rig.json")
    segmenter = mask.DepthSegmenter(rig, rectify.Reprojector(rig))

    fills = []
    for rec in meta:
      f = np.load(session / f"{rec['i']:06d}.npz")
      fills.append(float((f["depth"] > 0).mean()))
    threshold = (float(a.min_fill) if a.min_fill is not None else
                 max(0.50, float(np.quantile(fills, 0.10))))
    summary = {"path": str(session.resolve()), "seen": len(meta),
               "expected_objects": expected,
               "min_fill": threshold, "accepted": 0, "frames": []}

    for rec, fill in zip(meta, fills):
      i = int(rec["i"])
      f = np.load(session / f"{i:06d}.npz")
      depth = f["depth"].astype(np.float32) / 10000.0
      gray = f["gray"]
      reason = None
      if gray.ndim != 2 or gray.shape != depth.shape:
        reason = "gray_shape"
      elif fill < threshold:
        reason = "frame_fill"
      else:
        seg = segmenter(depth, arm=None)
        if len(seg.instances) != expected:
          reason = f"instance_count_{len(seg.instances)}"
        else:
          polys = []
          for inst in seg.instances:
            binary = mask.full_mask(
              seg, inst.label, segmenter.decimate) > 0
            instance_fill = float((depth[binary] > 0).mean())
            if instance_fill < a.min_instance_fill:
              reason = "instance_fill"
              break
            polys.extend(polygons(binary))
          if reason is None and len(polys) != expected:
            reason = f"polygon_count_{len(polys)}"
          if reason is None:
            all_items.append((session, i, gray, polys))
            summary["accepted"] += 1
            summary["frames"].append({"i": i, "accepted": True,
                                      "fill": fill,
                                      "instances": len(seg.instances)})
            continue
      manifest["rejected"][reason] = manifest["rejected"].get(reason, 0) + 1
      summary["frames"].append({"i": i, "accepted": False,
                                "fill": fill, "reason": reason})
    manifest["sessions"].append(summary)
    manifest["accepted"] += summary["accepted"]

  if len(all_items) < 20:
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels_manifest.json").write_text(
      json.dumps(manifest, indent=2) + "\n")
    print(f"only {len(all_items)} frames passed; refusing to build dataset")
    return 1
  write_dataset(all_items, out)
  (out / "labels_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n")
  print(f"accepted {len(all_items)} frame(s); rejected {manifest['rejected']}")
  print(f"wrote {out / 'data.yaml'}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
