#!/usr/bin/env python3
"""Benchmark Fast-FoundationStereo at a D455-like resolution."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--repo", type=Path, required=True)
  ap.add_argument("--model", type=Path, required=True)
  ap.add_argument("--left", type=Path, required=True)
  ap.add_argument("--right", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--width", type=int, default=848)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--iters", type=int, nargs="+", default=[4, 8])
  ap.add_argument("--warmup", type=int, default=10)
  ap.add_argument("--runs", type=int, default=50)
  ap.add_argument("--max-disp", type=int, default=192)
  ap.add_argument("--backend", choices=("triton", "pytorch1"), default="triton")
  return ap.parse_args()


def load_image(path: Path, width: int, height: int) -> np.ndarray:
  image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  if image is None:
    raise FileNotFoundError(path)
  if image.ndim == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  else:
    image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
  if image.shape[:2] != (height, width):
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
  return np.ascontiguousarray(image, dtype=np.uint8)


def stats(values: list[float]) -> dict[str, float]:
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
  result: dict[str, object] = {
      "name": torch.cuda.get_device_name(0),
      "capability": list(torch.cuda.get_device_capability(0)),
      "torch": torch.__version__,
      "torch_cuda": torch.version.cuda,
  }
  try:
    line = subprocess.check_output([
        "nvidia-smi", "--query-gpu=driver_version,power.limit,memory.total",
        "--format=csv,noheader,nounits"], text=True).strip()
    driver, watts, memory_mib = [part.strip() for part in line.split(",")]
    result.update(driver=driver, power_limit_w=float(watts),
                  memory_total_mib=float(memory_mib))
  except Exception:
    pass
  return result


def main() -> None:
  args = parse_args()
  repo = args.repo.resolve()
  model_path = args.model.resolve()
  sys.path.insert(0, str(repo))
  from Utils import AMP_DTYPE, set_seed
  from core.utils.utils import InputPadder

  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  torch.backends.cudnn.benchmark = True
  left = load_image(args.left, args.width, args.height)
  right = load_image(args.right, args.width, args.height)

  with (model_path.parent / "cfg.yaml").open() as stream:
    cfg = yaml.safe_load(stream)
  model = torch.load(model_path, map_location="cpu", weights_only=False)
  model.args.max_disp = args.max_disp
  model.cuda().eval()

  def cpu_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

  left_gpu = cpu_tensor(left).cuda()
  right_gpu = cpu_tensor(right).cuda()
  padder = InputPadder(left_gpu.shape, divis_by=32, force_square=False)
  left_gpu, right_gpu = padder.pad(left_gpu, right_gpu)

  results: dict[str, object] = {
      "schema_version": 1,
      "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
      "gpu": gpu_description(),
      "host": {"platform": platform.platform(), "python": platform.python_version()},
      "fast_foundation_stereo": {
          "repo_commit": subprocess.check_output(
              ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
          "checkpoint": str(model_path),
          "checkpoint_bytes": model_path.stat().st_size,
          "checkpoint_name": model_path.parent.name,
          "config": cfg,
          "input_width": args.width,
          "input_height": args.height,
          "padded_width": int(left_gpu.shape[-1]),
          "padded_height": int(left_gpu.shape[-2]),
          "autocast": str(AMP_DTYPE),
          "cost_volume_backend": args.backend,
          "max_disp": args.max_disp,
      },
      "methodology": {
          "warmup_runs": args.warmup,
          "measured_runs": args.runs,
          "gpu_forward": "CUDA events; inputs resident on GPU",
          "pipeline_wall": "CPU tensor conversion, H2D, pad, forward, unpad, D2H",
          "disk_decode_included": False,
          "batch_size": 1,
      },
      "benchmarks": [],
  }

  last_disp: np.ndarray | None = None
  for valid_iters in args.iters:
    model.args.valid_iters = valid_iters

    def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
      with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        return model.forward(a, b, iters=valid_iters, test_mode=True,
                             optimize_build_volume=args.backend)

    for _ in range(args.warmup):
      forward(left_gpu, right_gpu)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    gpu_ms: list[float] = []
    for _ in range(args.runs):
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      forward(left_gpu, right_gpu)
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

    row = {
        "valid_iters": valid_iters,
        "gpu_forward_ms": stats(gpu_ms),
        "pipeline_wall_ms": stats(pipeline_ms),
        "throughput_fps_from_median_gpu": 1000.0 / statistics.median(gpu_ms),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 ** 2),
    }
    results["benchmarks"].append(row)
    print(json.dumps(row, indent=2))

  assert last_disp is not None
  args.out.mkdir(parents=True, exist_ok=True)
  np.save(args.out / "disparity_px.npy", last_disp)
  vis = cv2.applyColorMap(
      cv2.normalize(last_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
      cv2.COLORMAP_TURBO)
  cv2.imwrite(str(args.out / "disparity.png"), vis)
  (args.out / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
  print(f"results -> {args.out}")


if __name__ == "__main__":
  main()
