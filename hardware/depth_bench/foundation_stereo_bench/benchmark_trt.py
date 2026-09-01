#!/usr/bin/env python3
"""Benchmark a static Fast-FoundationStereo TensorRT engine."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--engine", type=Path, required=True)
  ap.add_argument("--left", type=Path, required=True)
  ap.add_argument("--right", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--reference", type=Path)
  ap.add_argument("--width", type=int, default=848)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--warmup", type=int, default=10)
  ap.add_argument("--runs", type=int, default=50)
  return ap.parse_args()


def load_image(path: Path, width: int, height: int) -> np.ndarray:
  image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  if image is None:
    raise FileNotFoundError(path)
  image = cv2.cvtColor(image if image.ndim == 2 else image[..., :3],
                       cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB)
  if image.shape[:2] != (height, width):
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
  return np.ascontiguousarray(image, dtype=np.uint8)


def stats(values: list[float]) -> dict[str, float]:
  a = np.asarray(values, dtype=np.float64)
  return {"mean": float(a.mean()), "median": float(np.median(a)),
          "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
          "min": float(a.min()), "max": float(a.max()), "std": float(a.std())}


def trt_to_torch(dtype: trt.DataType) -> torch.dtype:
  return {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
          trt.DataType.BF16: torch.bfloat16, trt.DataType.INT32: torch.int32,
          trt.DataType.INT8: torch.int8, trt.DataType.BOOL: torch.bool}[dtype]


class Runner:
  def __init__(self, engine_path: Path) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    if self.engine is None:
      raise RuntimeError(f"Could not deserialize {engine_path}")
    self.context = self.engine.create_execution_context()
    self.inputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                   if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    self.outputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                    if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

  def bind(self, tensors: dict[str, torch.Tensor]) -> None:
    for name in self.inputs:
      if name in tensors:
        self.context.set_input_shape(name, tuple(tensors[name].shape))
    for name, tensor in tensors.items():
      if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")
      self.context.set_tensor_address(name, int(tensor.data_ptr()))

  def allocate_outputs(self) -> dict[str, torch.Tensor]:
    return {name: torch.empty(tuple(self.context.get_tensor_shape(name)), device="cuda",
                              dtype=trt_to_torch(self.engine.get_tensor_dtype(name)))
            for name in self.outputs}

  def execute(self) -> None:
    if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
      raise RuntimeError("TensorRT execute_async_v3 failed")


def cpu_tensor(image: np.ndarray) -> torch.Tensor:
  return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()


def normalize_pad(tensor: torch.Tensor, pad_width: int) -> torch.Tensor:
  mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
  std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
  tensor = (tensor / 255.0 - mean) / std
  left_pad = pad_width // 2
  return F.pad(tensor, (left_pad, pad_width - left_pad, 0, 0), mode="replicate")


def main() -> None:
  args = parse_args()
  torch.autograd.set_grad_enabled(False)
  left = load_image(args.left, args.width, args.height)
  right = load_image(args.right, args.width, args.height)
  runner = Runner(args.engine.resolve())
  if len(runner.inputs) != 2 or len(runner.outputs) != 1:
    raise RuntimeError(f"Unexpected I/O: {runner.inputs}/{runner.outputs}")
  input_shape = tuple(runner.engine.get_tensor_shape(runner.inputs[0]))
  pad_width = input_shape[-1] - args.width
  if input_shape[-2] != args.height or pad_width < 0:
    raise ValueError(f"Engine shape {input_shape} does not cover {args.width}x{args.height}")
  left_pad = pad_width // 2

  resident = {runner.inputs[0]: normalize_pad(cpu_tensor(left).cuda(), pad_width),
              runner.inputs[1]: normalize_pad(cpu_tensor(right).cuda(), pad_width)}
  runner.bind(resident)
  outputs = runner.allocate_outputs()
  runner.bind(outputs)
  for _ in range(args.warmup):
    runner.execute()
  torch.cuda.synchronize()
  torch.cuda.reset_peak_memory_stats()

  gpu_ms = []
  for _ in range(args.runs):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    runner.execute()
    end.record()
    end.synchronize()
    gpu_ms.append(start.elapsed_time(end))

  pipeline_ms = []
  disparity = None
  for _ in range(args.runs):
    t0 = time.perf_counter()
    inputs = {runner.inputs[0]: normalize_pad(cpu_tensor(left).cuda(), pad_width),
              runner.inputs[1]: normalize_pad(cpu_tensor(right).cuda(), pad_width)}
    runner.bind(inputs)
    run_outputs = runner.allocate_outputs()
    runner.bind(run_outputs)
    runner.execute()
    raw = run_outputs[runner.outputs[0]]
    disparity = raw[..., :, left_pad:left_pad + args.width].float().squeeze().cpu().numpy()
    pipeline_ms.append((time.perf_counter() - t0) * 1000.0)

  assert disparity is not None
  result = {
      "schema_version": 1,
      "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
      "gpu": {"name": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability(0)),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda},
      "host": {"platform": platform.platform(), "python": platform.python_version()},
      "tensorrt": {"version": trt.__version__, "engine": str(args.engine.resolve()),
                   "engine_bytes": args.engine.stat().st_size, "inputs": runner.inputs,
                   "outputs": runner.outputs, "input_shape": list(input_shape)},
      "methodology": {"warmup_runs": args.warmup, "measured_runs": args.runs,
                      "gpu_forward": "CUDA events; preprocessed inputs/output resident on GPU",
                      "pipeline_wall": "CPU tensor conversion, H2D, normalize, pad, execute, unpad, D2H",
                      "disk_decode_included": False, "batch_size": 1},
      "gpu_forward_ms": stats(gpu_ms), "pipeline_wall_ms": stats(pipeline_ms),
      "throughput_fps_from_median_gpu": 1000.0 / statistics.median(gpu_ms),
      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
  if args.reference:
    reference = np.load(args.reference)
    error = np.abs(disparity - reference)
    result["reference_comparison"] = {
        "path": str(args.reference.resolve()), "mae_px": float(error.mean()),
        "median_abs_px": float(np.median(error)),
        "p95_abs_px": float(np.percentile(error, 95)), "max_abs_px": float(error.max())}

  args.out.mkdir(parents=True, exist_ok=True)
  np.save(args.out / "disparity_px.npy", disparity)
  vis = cv2.applyColorMap(cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                          cv2.COLORMAP_TURBO)
  cv2.imwrite(str(args.out / "disparity.png"), vis)
  (args.out / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
