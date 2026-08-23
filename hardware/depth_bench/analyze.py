#!/usr/bin/env python3
"""Turn the captures into the two numbers ``src/piper_push/camera.py`` is missing.

    python analyze.py                 # every backend that has captures
    python analyze.py --backend d405

``camera.py`` currently carries::

    DEPTH_NOISE_M = 0.004
    DEPTH_DROPOUT = 0.02

marked as placeholders, with a note that guessing them is the single largest
sim-to-real risk in the vision stage.  This script is what replaces them.

Noise is not fitted as a constant, because it is not one.  For a stereo camera
the depth error follows from the disparity error::

    sigma_z = z^2 / (f * b) * sigma_disparity

so the captures are fitted as ``sigma_z = a * z^2`` and reported both as the
coefficient and as the implied ``sigma_disparity``, which is the number that
should stay roughly constant across distances if the sensor is behaving and the
model is the right one.  The fit matters because the simulator camera sits
0.70 m from what it aims at, and on a D405 -- rated to about 0.5 m -- that is an
extrapolation someone has to look at rather than a interpolation.

Dropout is reported per region rather than as one number, because the white and
black patches exist precisely to show that a single uniform probability is the
wrong model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM_RANGE_M = 0.70
"""Distance from the simulator camera to its aim point, from camera.py:
CAMERA_POS (0.589, 0.535, 0.480) to CAMERA_AIM (0.321, 0.071, 0.030)."""

REGIONS = ["charuco", "white", "black"]
LEGEND = {
  "charuco": "textured (best case)",
  "white": "blank white (no texture)",
  "black": "solid black (dark surface)",
}


def load(root: Path, backend: str) -> list[dict]:
  out = []
  for f in sorted((root / backend).glob("*/metrics.json")):
    if f.parent.name == "preview":
      continue
    out.append(json.loads(f.read_text()))
  return out


def fit_quadratic(z: np.ndarray, s: np.ndarray) -> dict:
  """Least squares for ``s = a z^2`` through the origin.

  Through the origin on purpose: a free intercept fits a constant noise floor
  that a stereo camera does not have, and with a handful of captures it will
  happily absorb the whole signal and report a flat sensor.
  """
  ok = np.isfinite(z) & np.isfinite(s)
  z, s = z[ok], s[ok]
  if z.size < 2:
    return {"a": float("nan"), "n": int(z.size), "resid_rms_m": float("nan")}
  a = float((z**2 @ s) / (z**2 @ z**2))
  resid = s - a * z**2
  return {"a": a, "n": int(z.size),
          "resid_rms_m": float(np.sqrt(np.mean(resid**2)))}


def report(backend: str, runs: list[dict]) -> dict:
  runs = sorted(runs, key=lambda r: r["pose"]["distance_m"])
  cap = runs[0]["capture"]
  print(f"\n=== {backend} — {cap.get('model', backend)} "
        f"fw {cap.get('firmware', '?')} — {len(runs)} captures ===")
  print(f"    {cap.get('resolution')} @ {cap.get('fps')} Hz, preset "
        f"{cap.get('preset')}, filters={cap.get('filters')}, "
        f"emitter: {cap.get('emitter', 'unknown')}")

  print(f"\n{'label':>10s} {'dist':>7s} {'tilt':>5s} | " +
        " | ".join(f"{LEGEND[r].split(' (')[0]:>28s}" for r in REGIONS))
  print(f"{'':>10s} {'':>7s} {'':>5s} | " +
        " | ".join(f"{'fill':>6s} {'bias':>7s} {'rms':>6s} {'p95':>6s}"
                   for _ in REGIONS))
  for r in runs:
    row = f"{r['label']:>10s} {r['pose']['distance_m'] * 1000:6.0f}mm " \
          f"{r['pose']['tilt_deg']:4.0f}° |"
    for name in REGIONS:
      m = r["regions"].get(name, {})
      if not m.get("n_pixels"):
        row += f" {'--':>6s} {'--':>7s} {'--':>6s} {'--':>6s} |"
        continue
      row += (f" {m['fill'] * 100:5.1f}% {m['bias_m'] * 1000:+6.2f} "
              f"{m['spatial_rms_m'] * 1000:5.2f} {m['p95_abs_m'] * 1000:5.2f} |")
    print(row)
  print("      (bias / rms / p95 in mm, against the ChArUco plane)")

  z = np.array([r["pose"]["distance_m"] for r in runs])
  summary = {"backend": backend, "n_captures": len(runs), "capture": cap,
             "sim_range_m": SIM_RANGE_M, "regions": {}}

  print(f"\n  noise model  sigma_z = a * z^2   (extrapolated to the simulator's "
        f"{SIM_RANGE_M:.2f} m)")
  f_px, b_m = cap.get("fx_px"), cap.get("stereo_baseline_m")
  for name in REGIONS:
    s = np.array([r["regions"].get(name, {}).get("spatial_rms_m", np.nan)
                  for r in runs], dtype=float)
    t = np.array([r["regions"].get(name, {}).get("temporal_std_m", np.nan)
                  for r in runs], dtype=float)
    fill = np.array([r["regions"].get(name, {}).get("fill", np.nan)
                     for r in runs], dtype=float)
    fit = fit_quadratic(z, s)
    at_sim = fit["a"] * SIM_RANGE_M**2
    disp = (fit["a"] * f_px * b_m) if (f_px and b_m) else float("nan")
    summary["regions"][name] = {
      "fit_a_per_m": fit["a"], "fit_resid_rms_m": fit["resid_rms_m"],
      "sigma_at_sim_range_m": at_sim,
      "implied_sigma_disparity_px": disp,
      "temporal_std_median_m": float(np.nanmedian(t)),
      "fill_min": float(np.nanmin(fill)) if fill.size else float("nan"),
      "fill_max": float(np.nanmax(fill)) if fill.size else float("nan"),
      "dropout_at_worst": float(1.0 - np.nanmin(fill)) if fill.size else float("nan"),
    }
    print(f"    {LEGEND[name]:26s} a={fit['a']:8.5f} /m  "
          f"sigma(0.70m)={at_sim * 1000:6.2f} mm  "
          f"sigma_disp={disp:5.3f} px  "
          f"fill {np.nanmin(fill) * 100:5.1f}–{np.nanmax(fill) * 100:5.1f}%")

  best = summary["regions"]["charuco"]
  print(f"\n  -> camera.py, if the policy only ever sees textured surfaces:")
  print(f"       DEPTH_NOISE_M = {best['sigma_at_sim_range_m']:.4f}")
  print(f"       DEPTH_DROPOUT = {best['dropout_at_worst']:.4f}")
  w, bl = summary["regions"]["white"], summary["regions"]["black"]
  print(f"     but dropout is {w['dropout_at_worst'] * 100:.1f}% on blank white and "
        f"{bl['dropout_at_worst'] * 100:.1f}% on black, against "
        f"{best['dropout_at_worst'] * 100:.1f}% on texture.")
  print(f"     A single uniform DEPTH_DROPOUT cannot represent a "
        f"{max(w['dropout_at_worst'], bl['dropout_at_worst']) / max(best['dropout_at_worst'], 1e-6):.0f}x "
        f"spread that is a function of what the surface looks like.")
  return summary


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--results", type=Path, default=HERE / "results")
  ap.add_argument("--backend", default=None)
  ap.add_argument("--json", type=Path, default=None)
  args = ap.parse_args()

  backends = ([args.backend] if args.backend
              else sorted(d.name for d in args.results.iterdir() if d.is_dir()))
  out = {}
  for b in backends:
    runs = load(args.results, b)
    if not runs:
      print(f"(no captures for {b})")
      continue
    out[b] = report(b, runs)

  if len(out) > 1:
    print("\n=== camera comparison, textured surface ===")
    print(f"{'backend':10s} {'sigma(0.70m)':>13s} {'fill white':>11s} "
          f"{'fill black':>11s}")
    for b, s in out.items():
      r = s["regions"]
      print(f"{b:10s} {r['charuco']['sigma_at_sim_range_m'] * 1000:11.2f}mm "
            f"{r['white']['fill_min'] * 100:10.1f}% "
            f"{r['black']['fill_min'] * 100:10.1f}%")

  if args.json:
    args.json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {args.json}")


if __name__ == "__main__":
  main()
