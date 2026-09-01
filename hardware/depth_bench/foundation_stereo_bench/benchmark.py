#!/usr/bin/env python3
"""Benchmark FoundationStereo and render a metric D455 depth comparison.

The CUDA-event number measures the model forward pass only.  The wall-clock
number additionally includes camera-array style CPU preprocessing, host to
device transfer, padding, unpadding, and copying the disparity back to CPU.
Disk image decoding is intentionally outside both measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--repo", type=Path, required=True,
                  help="checkout of NVlabs/FoundationStereo")
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--left", type=Path, required=True)
  ap.add_argument("--right", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--width", type=int, default=848)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--iters", type=int, nargs="+", default=[8, 16, 32])
  ap.add_argument("--warmup", type=int, default=5)
  ap.add_argument("--runs", type=int, default=30)
  ap.add_argument("--fx", type=float, default=None,
                  help="rectified left focal length in pixels")
  ap.add_argument("--baseline-m", type=float, default=None)
  ap.add_argument("--native-depth", type=Path, default=None,
                  help="D455 Z16 PNG/NPY for the same stereo pair")
  ap.add_argument("--native-depth-scale", type=float, default=0.001,
                  help="meters per native-depth integer unit")
  ap.add_argument("--near-m", type=float, default=0.15)
  ap.add_argument("--far-m", type=float, default=2.0)
  return ap.parse_args()


def load_image(path: Path, width: int, height: int) -> np.ndarray:
  image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  if image is None:
    raise FileNotFoundError(path)
  if image.ndim == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  elif image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
  else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  if image.shape[:2] != (height, width):
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
  return np.ascontiguousarray(image, dtype=np.uint8)


def summary_ms(values: list[float]) -> dict[str, float]:
  a = np.asarray(values, dtype=np.float64)
  return {
      "mean": float(a.mean()),
      "median": float(np.median(a)),
      "p90": float(np.percentile(a, 90)),
      "p95": float(np.percentile(a, 95)),
      "min": float(a.min()),
      "max": float(a.max()),
      "std": float(a.std()),
  }


def gpu_description() -> dict[str, object]:
  desc: dict[str, object] = {
      "name": torch.cuda.get_device_name(0),
      "capability": list(torch.cuda.get_device_capability(0)),
      "torch": torch.__version__,
      "torch_cuda": torch.version.cuda,
  }
  try:
    line = subprocess.check_output([
        "nvidia-smi", "--query-gpu=driver_version,power.limit,memory.total",
        "--format=csv,noheader,nounits"], text=True).strip()
    driver, watts, memory_mib = [x.strip() for x in line.split(",")]
    desc.update(driver=driver, power_limit_w=float(watts),
                memory_total_mib=float(memory_mib))
  except Exception:
    pass
  return desc


def load_native_depth(path: Path, scale: float, shape: tuple[int, int]) -> np.ndarray:
  if path.suffix.lower() == ".npy":
    raw = np.load(path)
  else:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  if raw is None:
    raise FileNotFoundError(path)
  if raw.ndim != 2:
    raise ValueError(f"native depth must be HxW, got {raw.shape}")
  depth = raw.astype(np.float32) * scale
  if depth.shape != shape:
    depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
  depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
  return depth


def colorize_depth(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
  valid = np.isfinite(depth_m) & (depth_m > 0)
  normalized = np.clip((depth_m - near_m) / (far_m - near_m), 0, 1)
  # Near objects are warm and far objects cool, matching common depth viewers.
  u8 = np.nan_to_num((1.0 - normalized) * 255.0).astype(np.uint8)
  color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
  color[~valid] = 0
  return color


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
  out = image.copy()
  cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
  cv2.putText(out, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
              (255, 255, 255), 2, cv2.LINE_AA)
  return out


def compare_depth(native: np.ndarray, predicted: np.ndarray, near_m: float,
                  far_m: float, out_path: Path) -> dict[str, float]:
  native_vis = label_panel(colorize_depth(native, near_m, far_m),
                           "RealSense D455 native depth")
  pred_vis = label_panel(colorize_depth(predicted, near_m, far_m),
                         "FoundationStereo metric depth")
  joined = np.concatenate([native_vis, pred_vis], axis=1)
  cv2.imwrite(str(out_path), joined)

  both = np.isfinite(native) & np.isfinite(predicted) & (native > 0) & (predicted > 0)
  native_valid = np.isfinite(native) & (native > 0)
  pred_valid = np.isfinite(predicted) & (predicted > 0)
  metrics: dict[str, float] = {
      "native_fill_percent": float(native_valid.mean() * 100),
      "foundation_stereo_fill_percent": float(pred_valid.mean() * 100),
      "overlap_percent": float(both.mean() * 100),
  }
  if both.any():
    error = predicted[both] - native[both]
    metrics.update(
        median_signed_difference_mm=float(np.median(error) * 1000),
        median_absolute_difference_mm=float(np.median(np.abs(error)) * 1000),
        p95_absolute_difference_mm=float(np.percentile(np.abs(error), 95) * 1000),
    )
  return metrics


def main() -> None:
  args = parse_args()
  args.repo = args.repo.resolve()
  args.checkpoint = args.checkpoint.resolve()
  sys.path.insert(0, str(args.repo))
  os.environ.setdefault("XFORMERS_DISABLED", "1")

  from omegaconf import OmegaConf
  from core.foundation_stereo import FoundationStereo
  from core.utils.utils import InputPadder

  args.out.mkdir(parents=True, exist_ok=True)
  left = load_image(args.left, args.width, args.height)
  right = load_image(args.right, args.width, args.height)
  if left.shape != right.shape:
    raise ValueError(f"stereo shapes differ: {left.shape} vs {right.shape}")

  cfg = OmegaConf.load(args.checkpoint.parent / "cfg.yaml")
  if "vit_size" not in cfg:
    cfg.vit_size = "vitl"
  model = FoundationStereo(cfg)
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  model.load_state_dict(checkpoint["model"])
  model = model.cuda().eval()
  torch.set_grad_enabled(False)
  torch.backends.cudnn.benchmark = True

  def cpu_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

  # Resident inputs model a deployment that keeps reusable frame buffers on GPU.
  left_gpu = cpu_tensor(left).cuda()
  right_gpu = cpu_tensor(right).cuda()
  padder = InputPadder(left_gpu.shape, divis_by=32, force_square=False)
  left_gpu, right_gpu = padder.pad(left_gpu, right_gpu)

  results: dict[str, object] = {
      "schema_version": 1,
      "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
      "gpu": gpu_description(),
      "host": {"platform": platform.platform(), "python": platform.python_version()},
      "foundation_stereo": {
          "repo_commit": subprocess.check_output(
              ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
          "checkpoint": str(args.checkpoint),
          "checkpoint_bytes": args.checkpoint.stat().st_size,
          "vit_size": str(cfg.vit_size),
          "input_width": args.width,
          "input_height": args.height,
          "autocast": "float16",
          "xformers_disabled": True,
      },
      "methodology": {
          "warmup_runs": args.warmup,
          "measured_runs": args.runs,
          "gpu_forward": "CUDA events around model.forward; inputs resident on GPU",
          "pipeline_wall": "perf_counter around CPU array conversion, H2D, pad, forward, unpad, D2H",
          "disk_decode_included": False,
          "batch_size": 1,
      },
      "benchmarks": [],
  }

  last_disp: np.ndarray | None = None
  for valid_iters in args.iters:
    def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
      with torch.autocast(device_type="cuda", dtype=torch.float16):
        return model(a, b, iters=valid_iters, test_mode=True)

    for _ in range(args.warmup):
      forward(left_gpu, right_gpu)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    gpu_ms: list[float] = []
    for _ in range(args.runs):
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      disparity = forward(left_gpu, right_gpu)
      end.record()
      end.synchronize()
      gpu_ms.append(start.elapsed_time(end))

    pipeline_ms: list[float] = []
    for _ in range(args.runs):
      t0 = time.perf_counter()
      a = cpu_tensor(left).cuda()
      b = cpu_tensor(right).cuda()
      run_padder = InputPadder(a.shape, divis_by=32, force_square=False)
      a, b = run_padder.pad(a, b)
      disparity = run_padder.unpad(forward(a, b).float())
      last_disp = disparity.squeeze().cpu().numpy()
      pipeline_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    row = {
        "valid_iters": valid_iters,
        "gpu_forward_ms": summary_ms(gpu_ms),
        "pipeline_wall_ms": summary_ms(pipeline_ms),
        "throughput_fps_from_median_gpu": 1000.0 / statistics.median(gpu_ms),
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
    }
    results["benchmarks"].append(row)
    print(json.dumps(row, indent=2))

  assert last_disp is not None
  np.save(args.out / "disparity_px.npy", last_disp)
  disparity_vis = cv2.applyColorMap(
      cv2.normalize(last_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
      cv2.COLORMAP_TURBO)
  cv2.imwrite(str(args.out / "disparity.png"), disparity_vis)

  if args.fx is not None or args.baseline_m is not None:
    if args.fx is None or args.baseline_m is None:
      raise ValueError("--fx and --baseline-m must be supplied together")
    depth_fs = args.fx * args.baseline_m / last_disp
    depth_fs[(last_disp <= 0) | ~np.isfinite(depth_fs)] = np.nan
    np.save(args.out / "foundation_stereo_depth_m.npy", depth_fs)
    if args.native_depth is not None:
      native = load_native_depth(args.native_depth, args.native_depth_scale,
                                 depth_fs.shape)
      results["depth_comparison"] = compare_depth(
          native, depth_fs, args.near_m, args.far_m, args.out / "comparison.png")

  (args.out / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
  print(f"results -> {args.out}")


if __name__ == "__main__":
  main()

