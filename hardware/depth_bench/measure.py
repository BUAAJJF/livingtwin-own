#!/usr/bin/env python3
"""Record one static scene and reduce it to the bench's numbers.

Usage runs in two steps, because the sheet has to be framed before it is worth
recording anything:

    python measure.py --backend d405 --preview
    python measure.py --backend d405 --label 40cm --frames 60

The distance is not an argument.  It is measured from the ChArUco pose and
written into the result, so the sheet can simply be moved between captures and
``fit_noise.py`` recovers the trend afterwards.  Asking a person to place a
sheet at exactly 400 mm and then trusting the number they typed is how a bias
measurement becomes a measurement of the tape measure.

Everything is written to ``results/<backend>/<label>/``: the raw frame stack,
the metrics, and a figure.  The raw stack is kept because the three cameras are
being compared over several weeks and the metric definitions will move; when
they do, the old captures are re-reduced instead of re-shot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M
from capture import Capture, open_backend

HERE = Path(__file__).resolve().parent
DEFAULT_TARGET = HERE / "targets" / "target_a4.json"


def annotate(cap: Capture, board, spec) -> np.ndarray:
  """Grayscale + depth side by side, with the regions drawn where they landed.

  This is the check that the sheet is framed and that the region rectangles sit
  on the printed patches rather than beside them -- a pose that is subtly wrong
  produces plausible numbers about the wrong pixels.
  """
  vis = cv2.cvtColor(cap.gray, cv2.COLOR_GRAY2BGR)
  d = cap.depth[0]
  finite = d[d > 0]
  lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
  dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
  dv = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
  dv[d <= 0] = (0, 0, 0)  # invalid pixels stay black, not "far away"

  try:
    pose = M.detect_pose(cap.gray, cap.K, cap.dist, board)
  except RuntimeError as e:
    cv2.putText(vis, str(e)[:70], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    return np.hstack([vis, dv])

  colours = {"charuco": (0, 255, 0), "white": (255, 128, 0), "black": (0, 200, 255)}
  for name, region in spec["regions"].items():
    quad = np.array([
      [region["x_min_m"], region["y_min_m"], 0.0],
      [region["x_max_m"], region["y_min_m"], 0.0],
      [region["x_max_m"], region["y_max_m"], 0.0],
      [region["x_min_m"], region["y_max_m"], 0.0],
    ])
    proj, _ = cv2.projectPoints(quad, pose["rvec"], pose["tvec"], cap.K, cap.dist)
    pts = np.round(proj.reshape(-1, 2)).astype(np.int32)
    c = colours.get(name, (255, 255, 255))
    for img in (vis, dv):
      cv2.polylines(img, [pts], True, c, 2)
      cv2.putText(img, name, tuple(pts[0] + np.array([4, -6])),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
  cv2.putText(vis, f"{np.linalg.norm(pose['tvec']) * 1000:.0f} mm  "
                   f"tilt {pose['tilt_deg']:.0f} deg  "
                   f"{pose['n_corners']} corners  "
                   f"reproj {pose['reproj_rms_px']:.2f} px",
              (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
  return np.hstack([vis, dv])


def provenance() -> dict:
  def git(*a):
    try:
      return subprocess.check_output(["git", *a], cwd=HERE,
                                     stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
      return "unknown"

  return {"commit": git("rev-parse", "HEAD"),
          "dirty": bool(git("status", "--porcelain")),
          "opencv": cv2.__version__, "numpy": np.__version__}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--backend", required=True, choices=["d405", "zedx", "odin1"])
  ap.add_argument("--label", default=None, help="name for this capture, e.g. 40cm")
  ap.add_argument("--frames", type=int, default=60)
  ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
  ap.add_argument("--outlier-m", type=float, default=0.05)
  ap.add_argument("--preview", action="store_true",
                  help="grab a few frames, write the annotated view, measure nothing")
  ap.add_argument("--out", type=Path, default=HERE / "results")
  args, _ = ap.parse_known_args()

  backend = open_backend(args.backend)
  backend.add_args(ap)
  args = ap.parse_args()

  if args.preview:
    args.label = args.label or "preview"
    args.frames = min(args.frames, 10)
  if not args.label:
    ap.error("--label is required unless --preview")

  board, spec = M.load_target(args.target)
  cap = backend.grab(args, args.frames)

  outdir = args.out / args.backend / args.label
  outdir.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(outdir / "view.png"), annotate(cap, board, spec))

  if args.preview:
    print(f"preview -> {outdir/'view.png'}")
    try:
      pose = M.detect_pose(cap.gray, cap.K, cap.dist, board)
      print(f"  sheet at {np.linalg.norm(pose['tvec']) * 1000:.0f} mm, "
            f"tilt {pose['tilt_deg']:.1f} deg, {pose['n_corners']} corners, "
            f"reproj {pose['reproj_rms_px']:.2f} px")
    except RuntimeError as e:
      print(f"  NOT DETECTED: {e}")
    return

  cap.save(outdir / "frames.npz")
  result = M.evaluate(cap, board, spec, args.outlier_m)
  result["capture"] = cap.meta
  result["target"] = str(args.target)
  result["label"] = args.label
  result["provenance"] = provenance()
  (outdir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

  p = result["pose"]
  print(f"\n{args.backend} / {args.label}: sheet at {p['distance_m'] * 1000:.0f} mm, "
        f"tilt {p['tilt_deg']:.1f} deg, reproj {p['reproj_rms_px']:.2f} px")
  print(f"{'region':9s} {'range':>7s} {'fill':>7s} {'stable':>7s} {'bias':>9s} "
        f"{'sp.rms':>8s} {'temp':>8s} {'p95':>8s}")
  for name, r in result["regions"].items():
    if not r.get("n_pixels"):
      print(f"{name:9s}   (no pixels in frame)")
      continue
    print(f"{name:9s} {r['range_m'] * 1000:6.0f}m {r['fill'] * 100:6.1f}% "
          f"{r['stable_fill'] * 100:6.1f}% {r['bias_m'] * 1000:+8.2f}mm "
          f"{r['spatial_rms_m'] * 1000:7.2f}mm {r['temporal_std_m'] * 1000:7.2f}mm "
          f"{r['p95_abs_m'] * 1000:7.2f}mm")
  print(f"\n-> {outdir}")


if __name__ == "__main__":
  main()
