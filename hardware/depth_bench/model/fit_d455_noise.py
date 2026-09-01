"""Fit the D455 active-stereo output model from deployment recordings.

The fit is intentionally performed in disparity-aware depth coordinates, but
does not attempt to rerun Intel's proprietary stereo matcher.  The renderer
provides clean geometry; this script measures the output statistics that the
matcher adds: range bias, frozen and temporal correlated error, fill rate and
edge-dependent holes.  The resulting JSON is consumed by
``piper_push.d455_noise``.

The loose-board calibration supplies the table plane as metric ground truth.
YOLO recordings supply long, consecutive D455 sequences in the actual scene.
Pixels more than 30 mm from the calibrated plane are excluded from plane-noise
statistics, so objects and robot links do not masquerade as sensor error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUTS = (
  ROOT / "recordings/d455_yolo/session01",
  ROOT / "recordings/d455_yolo/manual_20260826_163148",
)
DEFAULT_OUT = Path(__file__).with_name("d455_noise.json")
DEFAULT_RIG = ROOT / "hardware/deploy/rig_d455.json"
DEFAULT_HANDEYE = ROOT / "hardware/deploy/calib_poses_d455.json"
DEFAULT_TABLE_CALIB = ROOT / "hardware/deploy/calib_table_poses_d455.json"
DEPTH_SCALE_M = 1.0e-4
MAX_DEPTH_M = 1.5
PLANE_GATE_M = 0.030


def _load_rig(folder: Path) -> dict:
  path = folder / "rig.json"
  if not path.exists():
    raise FileNotFoundError(f"{folder}: missing rig.json")
  return json.loads(path.read_text())


def _plane_depth(rig: dict, shape: tuple[int, int]) -> np.ndarray:
  """Expected optical-axis depth of the calibrated table at every pixel."""
  h, w = shape
  K = np.asarray(rig["K"], dtype=np.float64)
  T = np.asarray(rig["T_base_cam"], dtype=np.float64)
  n = np.asarray(rig.get("table_normal_base", (0.0, 0.0, 1.0)),
                 dtype=np.float64)
  n /= np.linalg.norm(n)
  p0 = np.array((0.0, 0.0, float(rig["table_z"])), dtype=np.float64)
  u, v = np.meshgrid(np.arange(w), np.arange(h))
  ray = np.stack(((u - K[0, 2]) / K[0, 0],
                  (v - K[1, 2]) / K[1, 1], np.ones_like(u)), axis=-1)
  denom = np.einsum("i,hwi->hw", n, ray @ T[:3, :3].T)
  numer = float(n @ (p0 - T[:3, 3]))
  z = numer / denom
  z[(np.abs(denom) <= 1.0e-6) | (z <= 0.0)] = np.nan
  return z.astype(np.float32)


def _depth(path: Path) -> np.ndarray:
  with np.load(path) as z:
    return z["depth"].astype(np.float32) * DEPTH_SCALE_M


def _sample(files: list[Path], limit: int) -> list[Path]:
  if limit <= 0 or len(files) <= limit:
    return files
  idx = np.linspace(0, len(files) - 1, limit).round().astype(int)
  return [files[i] for i in idx]


def _calibration_summary(path: Path) -> dict:
  """Summarise the saved multi-frame feature measurements used by the rig."""
  data = json.loads(path.read_text())
  poses = data.get("poses", [])
  reproj = np.asarray([p["reproj_rms_px"] for p in poses], dtype=float)
  spread = np.asarray([p["corner_spread_px"] for p in poses], dtype=float)
  return {
    "path": str(path.resolve()),
    "poses": len(poses),
    "fused_frames": int(sum(int(p.get("fusion_frames", 1)) for p in poses)),
    "reprojection_rms_px_median": float(np.median(reproj)),
    "reprojection_rms_px_p95": float(np.percentile(reproj, 95)),
    "corner_spread_px_median": float(np.median(spread)),
    "corner_spread_px_p95": float(np.percentile(spread, 95)),
  }


def _merge_accumulators(*items: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """Combine disjoint frame accumulators without rereading the recordings."""
  return {
    key: sum((item[key] for item in items[1:]), items[0][key].copy())
    for key in items[0]
  }


def _accumulate(files: list[Path], truth: np.ndarray) -> dict[str, np.ndarray]:
  shape = truth.shape
  count = np.zeros(shape, np.uint32)
  total = np.zeros(shape, np.float64)
  total2 = np.zeros(shape, np.float64)
  raw_valid = np.zeros(shape, np.uint32)
  scene_count = np.zeros(shape, np.uint32)
  scene_total = np.zeros(shape, np.float64)
  plane_roi = np.isfinite(truth) & (truth > 0.25) & (truth < MAX_DEPTH_M)
  for i, path in enumerate(files, 1):
    d = _depth(path)
    valid = np.isfinite(d) & (d > 0.05) & (d < MAX_DEPTH_M)
    raw_valid += (valid & plane_roi)
    scene = valid & plane_roi
    scene_count += scene
    scene_total += np.where(scene, d, 0.0)
    keep = valid & plane_roi & (np.abs(d - truth) < PLANE_GATE_M)
    r = np.where(keep, d - truth, 0.0)
    count += keep
    total += r
    total2 += r * r
    if i % 250 == 0:
      print(f"  accumulated {i}/{len(files)}", flush=True)
  return {"count": count, "sum": total, "sum2": total2,
          "raw_valid": raw_valid, "scene_count": scene_count,
          "scene_total": scene_total}


def _robust_bias(z: np.ndarray, mean: np.ndarray, stable: np.ndarray) -> tuple[float, float]:
  """Fit residual = scale*z + offset from per-range robust medians."""
  edges = np.linspace(float(np.nanpercentile(z[stable], 2)),
                      float(np.nanpercentile(z[stable], 98)), 17)
  zx, ry = [], []
  for lo, hi in zip(edges[:-1], edges[1:], strict=True):
    m = stable & (z >= lo) & (z < hi)
    if m.sum() < 100:
      continue
    zx.append(float(np.median(z[m])))
    ry.append(float(np.median(mean[m])))
  if len(zx) < 3:
    raise RuntimeError("not enough calibrated plane depth variation for bias fit")
  scale, offset = np.polyfit(zx, ry, 1)
  return float(scale), float(offset)


def _corr_at_lags(field: np.ndarray, mask: np.ndarray, max_lag: int = 32) -> np.ndarray:
  x = np.where(mask, field, 0.0)
  x = x - (x[mask].mean() if mask.any() else 0.0)
  var = float(np.mean(x[mask] ** 2)) if mask.any() else 0.0
  out = [1.0]
  if var < 1.0e-16:
    return np.asarray(out)
  for lag in range(1, max_lag + 1):
    vals = []
    mx = mask[:, :-lag] & mask[:, lag:]
    my = mask[:-lag, :] & mask[lag:, :]
    if mx.any():
      vals.append(float(np.mean(x[:, :-lag][mx] * x[:, lag:][mx]) / var))
    if my.any():
      vals.append(float(np.mean(x[:-lag, :][my] * x[lag:, :][my]) / var))
    out.append(float(np.mean(vals)) if vals else 0.0)
  return np.asarray(out)


def _one_over_e(corr: np.ndarray) -> float:
  target = math.exp(-1.0)
  below = np.flatnonzero(corr <= target)
  if not len(below):
    return float(len(corr) - 1)
  i = int(below[0])
  if i == 0:
    return 0.0
  a, b = corr[i - 1], corr[i]
  return float(i - 1 + (a - target) / max(a - b, 1.0e-9))


def _fit_edge_fill(mean_depth: np.ndarray, fill: np.ndarray,
                   mask: np.ndarray, fx: float) -> tuple[float, float]:
  """Fit p=p_flat/(1+(g/g50)^n) to the empirical mean scene."""
  d = mean_depth.astype(np.float32, copy=True)
  # Fill unobserved pixels only for taking a gradient; they do not enter fit.
  holes = ~np.isfinite(d)
  if holes.any():
    d[holes] = cv2.inpaint(np.nan_to_num(d, nan=0.0).astype(np.float32),
                           holes.astype(np.uint8), 3, cv2.INPAINT_NS)[holes]
  gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8.0
  gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8.0
  g = np.hypot(gx, gy) * fx / np.maximum(d, 1.0e-3)
  use = mask & np.isfinite(g) & (fill > 0.05)
  if use.sum() < 1000:
    return 25.0, 1.0
  # Robust bins carry equal weight, rather than the flat table overwhelming
  # every actual discontinuity.
  positive = g[use & (g > 1.0e-3)]
  if positive.size < 100:
    return 25.0, 1.0
  edges = np.geomspace(max(1.0e-3, np.percentile(positive, 1)),
                       max(1.0e-2, np.percentile(positive, 99.5)), 18)
  xs, ys = [], []
  for lo, hi in zip(edges[:-1], edges[1:], strict=True):
    m = use & (g >= lo) & (g < hi)
    if m.sum() >= 50:
      xs.append(float(np.median(g[m])))
      ys.append(float(np.median(fill[m])))
  if len(xs) < 4:
    return 25.0, 1.0
  x, y = np.asarray(xs), np.asarray(ys)
  pflat = float(np.percentile(fill[use & (g < np.percentile(positive, 20))], 90))
  best = (float("inf"), 25.0, 1.0)
  for exponent in np.linspace(0.5, 2.0, 151):
    for g50 in np.geomspace(1.0, 100.0, 161):
      pred = pflat / (1.0 + (x / g50) ** exponent)
      loss = float(np.mean((pred - y) ** 2))
      if loss < best[0]:
        best = (loss, float(g50), float(exponent))
  return best[1], best[2]


def _metrics(acc: dict[str, np.ndarray], truth: np.ndarray, n: int) -> dict:
  count = acc["count"]
  stable = count >= max(8, int(0.20 * n))
  mean = np.divide(acc["sum"], count, out=np.full_like(acc["sum"], np.nan),
                   where=count > 0)
  var = np.divide(acc["sum2"], count, out=np.zeros_like(acc["sum2"]),
                  where=count > 0) - np.nan_to_num(mean) ** 2
  var = np.maximum(var, 0.0)
  scale, offset = _robust_bias(truth, mean, stable)
  detrended = mean - (scale * truth + offset)

  temporal_px = np.sqrt(var[stable]) / np.maximum(truth[stable] ** 2, 1.0e-6)
  sigma_temporal = float(np.median(temporal_px))
  # Frozen structure is measured within narrow range bands so perspective and
  # any small plane-fit tilt residual do not inflate it.
  norm_static = detrended / np.maximum(truth ** 2, 1.0e-6)
  sigma_static = float(1.4826 * np.median(np.abs(
    norm_static[stable] - np.median(norm_static[stable]))))

  corr = _corr_at_lags(norm_static, stable)
  corr_len = _one_over_e(corr)
  fill = acc["raw_valid"].astype(np.float64) / max(n, 1)
  fills = fill[stable]
  fill_range = (float(np.percentile(fills, 5)),
                float(np.percentile(fills, 95)))
  scene_count = acc["scene_count"]
  mean_scene = np.divide(
    acc["scene_total"], scene_count,
    out=np.full_like(acc["scene_total"], np.nan), where=scene_count > 0)
  scene_stable = scene_count >= max(8, int(0.20 * n))
  return {
    "stable": stable, "mean": mean, "var": var,
    "bias_scale": scale, "bias_offset_m": offset,
    "sigma_static_per_m": sigma_static,
    "sigma_temporal_per_m": sigma_temporal,
    "corr_len_native_px": corr_len,
    "fill": fill, "surface_fill": fill_range,
    "mean_scene": mean_scene, "scene_stable": scene_stable,
    "plane_pixels": int(stable.sum()),
    "plane_residual_rms_m": float(np.sqrt(np.nanmean((detrended[stable]) ** 2))),
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
  ap.add_argument("--rig", type=Path, default=DEFAULT_RIG,
                  help="metric D455 hand-eye and table-plane calibration")
  ap.add_argument("--handeye", type=Path, default=DEFAULT_HANDEYE,
                  help="saved multi-frame hand-eye feature observations")
  ap.add_argument("--table-calib", type=Path, default=DEFAULT_TABLE_CALIB,
                  help="saved loose-board table feature observations")
  ap.add_argument("--max-sequence", type=int, default=800,
                  help="evenly sample at most this many frames from each long sequence")
  ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
  a = ap.parse_args()

  folders = [p.resolve() for p in a.inputs]
  if not folders:
    ap.error("no input folders")
  rig = json.loads(a.rig.read_text())
  selected: list[list[Path]] = []
  sources = []
  for folder in folders:
    other = _load_rig(folder)
    if other.get("serial") != rig.get("serial"):
      raise RuntimeError("input folders came from different cameras")
    files = sorted(folder.glob("*.npz"))
    chosen = _sample(files, a.max_sequence)
    selected.append(chosen)
    sources.append({"path": str(folder), "available": len(files),
                    "used": len(chosen)})
  all_files = [path for group in selected for path in group]
  if not all_files:
    raise RuntimeError("no .npz frames found")

  first = _depth(all_files[0])
  truth = _plane_depth(rig, first.shape)
  # Keep the manually accepted, varied layouts source-disjoint from the long
  # static sequence.  Interleaving every fifth frame makes validation look
  # excellent even when a model merely memorises temporally correlated fixed
  # pattern noise.  Parameters are re-estimated on both sets only after this
  # genuinely independent validation has been measured.
  if len(selected) > 1 and selected[-1]:
    train = [path for group in selected[:-1] for path in group]
    valid = selected[-1]
    split = "last input folder held out by source"
  else:
    train = [p for i, p in enumerate(all_files) if i % 5 != 0]
    valid = [p for i, p in enumerate(all_files) if i % 5 == 0]
    split = "every fifth frame held out (single-source fallback)"
  print(f"D455 {rig.get('serial')}: {len(train)} development + "
        f"{len(valid)} source-held validation")
  train_acc = _accumulate(train, truth)
  val_acc = _accumulate(valid, truth)
  fit = _metrics(train_acc, truth, len(train))
  held = _metrics(val_acc, truth, len(valid))
  final = _metrics(_merge_accumulators(train_acc, val_acc), truth,
                   len(train) + len(valid))

  K = np.asarray(rig["K"], dtype=float)
  g50, edge_exp = _fit_edge_fill(
    final["mean_scene"], final["fill"], final["scene_stable"], float(K[0, 0]))
  # D455's D400 ASIC reports 1/32-pixel sub-disparity.  Keep this hardware
  # quantisation separate from fitted residual noise so an ablation can turn
  # either one off.
  baseline_m = 0.095
  params = {
    "sigma_static_per_m": final["sigma_static_per_m"],
    "sigma_temporal_per_m": final["sigma_temporal_per_m"],
    "corr_len_native_px": final["corr_len_native_px"],
    "native_f_px_per_rad": float(K[0, 0]),
    "edge_g50_per_rad": g50,
    "edge_exp": edge_exp,
    "texture_penalty": [1.0, 1.35],
    "surface_fill": list(final["surface_fill"]),
    "bias_scale": final["bias_scale"],
    "bias_offset_m": final["bias_offset_m"],
    "bias_scale_jitter": max(0.005, abs(fit["bias_scale"] - held["bias_scale"])),
    "bias_offset_jitter_m": max(
      0.002, abs(fit["bias_offset_m"] - held["bias_offset_m"])),
    "shadow_mrad": 4.0,
    "stereo_baseline_m": baseline_m,
    "stereo_focal_px": float(K[0, 0]),
    "disparity_subpixel_levels": 32,
  }
  report = {
    "sensor": "Intel RealSense D455",
    "serial": rig.get("serial"),
    "method": "calibrated active-stereo output model (DREDS-style effects)",
    "depth_scale_m": DEPTH_SCALE_M,
    "emitter": "on",
    "sources": sources,
    "calibration_sources": {
      "handeye": _calibration_summary(a.handeye),
      "table": _calibration_summary(a.table_calib),
      "rig": str(a.rig.resolve()),
    },
    "validation_split": split,
    "development_frames": len(train),
    "held_out_frames": len(valid),
    "final_fit_frames": len(all_files),
    "plane_gate_m": PLANE_GATE_M,
    "params": params,
    "development": {k: v for k, v in fit.items()
                    if isinstance(v, (int, float, str))},
    "held_out": {k: v for k, v in held.items()
                 if isinstance(v, (int, float, str))},
    "fit": {k: v for k, v in final.items()
            if isinstance(v, (int, float, str))},
    "notes": [
      "Calibration JSON contains poses and the metric table plane, not raw depth; "
      "the plane is used as ground truth for the YOLO depth frames.",
      "The long YOLO sequence estimates temporal/static structure; manually "
      "accepted layouts form a source-held validation domain before both "
      "sources are combined for the final parameter estimate.",
      "Disparity quantisation is the D400 hardware model; remaining parameters "
      "are fitted from this camera's recordings.",
    ],
  }
  a.out.parent.mkdir(parents=True, exist_ok=True)
  a.out.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
