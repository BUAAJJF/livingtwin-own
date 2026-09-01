#!/usr/bin/env python3
"""Build a TensorRT 11 engine from a statically-shaped mixed-precision ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import tensorrt as trt


def parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--onnx", type=Path, required=True)
  ap.add_argument("--engine", type=Path, required=True)
  ap.add_argument("--workspace-gib", type=float, default=8.0)
  ap.add_argument("--timing-cache", type=Path)
  return ap.parse_args()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def main() -> None:
  args = parse_args()
  args.onnx = args.onnx.resolve()
  args.engine = args.engine.resolve()
  args.engine.parent.mkdir(parents=True, exist_ok=True)

  logger = trt.Logger(trt.Logger.INFO)
  builder = trt.Builder(logger)
  flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
  network = builder.create_network(flags)
  parser = trt.OnnxParser(network, logger)

  print(f"Parsing {args.onnx}", flush=True)
  if not parser.parse_from_file(str(args.onnx)):
    errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
    raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")

  config = builder.create_builder_config()
  workspace_bytes = int(args.workspace_gib * 1024**3)
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
  config.avg_timing_iterations = 8

  if args.timing_cache and args.timing_cache.exists():
    cache = config.create_timing_cache(args.timing_cache.read_bytes())
    if not config.set_timing_cache(cache, ignore_mismatch=False):
      raise RuntimeError(f"Timing cache is incompatible: {args.timing_cache}")

  input_desc = []
  for index in range(network.num_inputs):
    tensor = network.get_input(index)
    input_desc.append({"name": tensor.name, "shape": list(tensor.shape),
                       "dtype": str(tensor.dtype)})
  output_desc = []
  for index in range(network.num_outputs):
    tensor = network.get_output(index)
    output_desc.append({"name": tensor.name, "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype)})
  print(json.dumps({"inputs": input_desc, "outputs": output_desc}, indent=2), flush=True)
  print(f"Building with {args.workspace_gib:g} GiB workspace...", flush=True)

  started = time.perf_counter()
  serialized = builder.build_serialized_network(network, config)
  build_seconds = time.perf_counter() - started
  if serialized is None:
    raise RuntimeError("TensorRT engine build failed")
  args.engine.write_bytes(serialized)

  if args.timing_cache:
    args.timing_cache.parent.mkdir(parents=True, exist_ok=True)
    args.timing_cache.write_bytes(bytes(config.get_timing_cache().serialize()))

  metadata = {
      "tensorrt": trt.__version__,
      "onnx": str(args.onnx),
      "onnx_sha256": sha256(args.onnx),
      "engine": str(args.engine),
      "engine_bytes": args.engine.stat().st_size,
      "workspace_gib": args.workspace_gib,
      "build_seconds": build_seconds,
      "strongly_typed": True,
      "inputs": input_desc,
      "outputs": output_desc,
  }
  metadata_path = args.engine.with_suffix(args.engine.suffix + ".build.json")
  metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
  print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
  main()
