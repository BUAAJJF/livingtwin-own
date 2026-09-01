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
import view
from capture import Capture, open_backend

HERE = Path(__file__).resolve().parent
DEFAULT_TARGET = HERE / "targets" / "target_a4.json"


def annotate(cap: Capture, board, spec) -> np.ndarray:
  """The framing check: are the region outlines on the printed patches?

  A pose that is subtly wrong produces plausible numbers about the wrong
  pixels, and this is the only place that shows it.
  """
  return view.compose(cap.gray, cap.depth[0], cap.K, cap.dist, cap.rays, cap.T_dg,
                      spec, board=board, width=640)


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
  ap.add_argument("--backend", required=True,
                  choices=["d405", "d455", "zedx", "odin1"])
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
      print(f"  sheet at {pose['plane_distance_m'] * 1000:.0f} mm, "
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
    print(f"{name:9s} {r['range_m'] * 1000:5.0f}mm {r['fill'] * 100:6.1f}% "
          f"{r['stable_fill'] * 100:6.1f}% {r['bias_m'] * 1000:+8.2f}mm "
          f"{r['spatial_rms_m'] * 1000:7.2f}mm {r['temporal_std_m'] * 1000:7.2f}mm "
          f"{r['p95_abs_m'] * 1000:7.2f}mm")
  print(f"\n-> {outdir}")


if __name__ == "__main__":
  main()
