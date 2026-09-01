#!/usr/bin/env python3
"""Live D455 native-depth versus Fast-FoundationStereo TensorRT viewer."""

from __future__ import annotations

import argparse
import json
import signal
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import torch.nn.functional as F

from benchmark_trt import Runner


def parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--engine", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--serial", default="")
  ap.add_argument("--width", type=int, default=848)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--min-depth", type=float, default=0.25)
  ap.add_argument("--max-depth", type=float, default=2.0)
  ap.add_argument("--max-error", type=float, default=0.25)
  ap.add_argument("--emitter", choices=("on", "off"), default="on")
  ap.add_argument("--warmup", type=int, default=10)
  ap.add_argument("--snapshot-after", type=int, default=30)
  return ap.parse_args()


def colorize_depth(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
  valid = np.isfinite(depth) & (depth > 0)
  scaled = np.clip((depth - lo) / (hi - lo), 0, 1)
  image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
  image[~valid] = (28, 28, 28)
  return image


def panel(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
  out = image.copy()
  cv2.rectangle(out, (0, 0), (out.shape[1], 58), (18, 18, 18), -1)
  cv2.putText(out, title, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
              (245, 245, 245), 2, cv2.LINE_AA)
  if subtitle:
    cv2.putText(out, subtitle, (16, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (210, 210, 210), 1, cv2.LINE_AA)
  return out


def save_snapshot(out_dir: Path, frame_no: int, composite: np.ndarray,
                  left: np.ndarray, right: np.ndarray, native: np.ndarray,
                  fs_depth: np.ndarray, disparity: np.ndarray,
                  metadata: dict[str, object]) -> Path:
  out_dir.mkdir(parents=True, exist_ok=True)
  stem = f"frame_{frame_no:06d}"
  path = out_dir / f"{stem}_comparison.png"
  cv2.imwrite(str(path), composite)
  cv2.imwrite(str(out_dir / f"{stem}_left_ir.png"), left)
  cv2.imwrite(str(out_dir / f"{stem}_right_ir.png"), right)
  np.save(out_dir / f"{stem}_native_depth_m.npy", native)
  np.save(out_dir / f"{stem}_fs_depth_m.npy", fs_depth)
  np.save(out_dir / f"{stem}_fs_disparity_px.npy", disparity)
  (out_dir / f"{stem}_metadata.json").write_text(
      json.dumps(metadata, indent=2) + "\n")
  return path


def main() -> None:
  args = parse_args()
  torch.autograd.set_grad_enabled(False)
  runner = Runner(args.engine.resolve())
  input_shape = tuple(runner.engine.get_tensor_shape(runner.inputs[0]))
  if input_shape[-2] != args.height or input_shape[-1] < args.width:
    raise ValueError(f"Engine input {input_shape} is incompatible with "
                     f"{args.width}x{args.height}")
  pad_width = input_shape[-1] - args.width
  left_pad = pad_width // 2
  mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1)
  std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)

  pipeline = rs.pipeline()
  config = rs.config()
  if args.serial:
    config.enable_device(args.serial)
  config.enable_stream(rs.stream.depth, args.width, args.height,
                       rs.format.z16, args.fps)
  config.enable_stream(rs.stream.infrared, 1, args.width, args.height,
                       rs.format.y8, args.fps)
  config.enable_stream(rs.stream.infrared, 2, args.width, args.height,
                       rs.format.y8, args.fps)
  profile = pipeline.start(config)
  sensor = profile.get_device().first_depth_sensor()
  if sensor.supports(rs.option.emitter_enabled):
    sensor.set_option(rs.option.emitter_enabled,
                      1.0 if args.emitter == "on" else 0.0)
  depth_scale = float(sensor.get_depth_scale())

  frames = pipeline.wait_for_frames()
  left_profile = frames.get_infrared_frame(1).profile.as_video_stream_profile()
  right_profile = frames.get_infrared_frame(2).profile.as_video_stream_profile()
  intr = left_profile.intrinsics
  extr = left_profile.get_extrinsics_to(right_profile)
  baseline_m = float(np.linalg.norm(np.asarray(extr.translation)))
  focal_baseline = float(intr.fx) * baseline_m
  device = profile.get_device()
  device_meta = {
      "model": device.get_info(rs.camera_info.name),
      "serial": device.get_info(rs.camera_info.serial_number),
      "firmware": device.get_info(rs.camera_info.firmware_version),
      "resolution": [args.width, args.height], "fps": args.fps,
      "emitter": args.emitter, "depth_scale_m": depth_scale,
      "fx_px": float(intr.fx), "baseline_m": baseline_m,
      "engine": str(args.engine.resolve()),
  }

  stop = False
  def request_stop(_signum: int, _frame: object) -> None:
    nonlocal stop
    stop = True
  signal.signal(signal.SIGINT, request_stop)
  signal.signal(signal.SIGTERM, request_stop)

  window = "D455 native depth vs Fast-FoundationStereo TensorRT"
  cv2.namedWindow(window, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(window, 1696, 960)
  inference_ms: deque[float] = deque(maxlen=60)
  processing_ms: deque[float] = deque(maxlen=60)
  last_frame_time = time.perf_counter()
  live_fps: deque[float] = deque(maxlen=60)
  auto_saved = False
  warmup_left = args.warmup

  print("LIVE_VIEW_STARTING " + json.dumps(device_meta), flush=True)
  try:
    while not stop:
      frames = pipeline.wait_for_frames(5000)
      depth_frame = frames.get_depth_frame()
      left_frame = frames.get_infrared_frame(1)
      right_frame = frames.get_infrared_frame(2)
      if not depth_frame or not left_frame or not right_frame:
        continue
      t0 = time.perf_counter()
      left = np.asanyarray(left_frame.get_data())
      right = np.asanyarray(right_frame.get_data())
      native = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

      def model_input(gray: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(
            device="cuda", dtype=torch.float32)
        x = (x.expand(-1, 3, -1, -1) / 255.0 - mean) / std
        return F.pad(x, (left_pad, pad_width - left_pad, 0, 0),
                     mode="replicate").contiguous()

      inputs = {runner.inputs[0]: model_input(left),
                runner.inputs[1]: model_input(right)}
      runner.bind(inputs)
      outputs = runner.allocate_outputs()
      runner.bind(outputs)
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      runner.execute()
      end.record()
      raw = outputs[runner.outputs[0]][..., left_pad:left_pad + args.width]
      disparity = raw.float().squeeze().cpu().numpy()
      end.synchronize()
      inference_ms.append(float(start.elapsed_time(end)))
      fs_depth = np.divide(focal_baseline, disparity,
                           out=np.zeros_like(disparity, dtype=np.float32),
                           where=np.isfinite(disparity) & (disparity > 0))
      fs_depth[(fs_depth < args.min_depth) | (fs_depth > args.max_depth)] = 0

      overlap = (native > 0) & (fs_depth > 0)
      error = np.zeros_like(native)
      error[overlap] = np.abs(native[overlap] - fs_depth[overlap])
      median_error = float(np.median(error[overlap])) if overlap.any() else float("nan")
      native_fill = float((native > 0).mean())
      fs_fill = float((fs_depth > 0).mean())

      now = time.perf_counter()
      dt = now - last_frame_time
      last_frame_time = now
      if dt > 0:
        live_fps.append(1.0 / dt)
      processing_ms.append((now - t0) * 1000.0)
      med_infer = float(np.median(inference_ms))
      med_process = float(np.median(processing_ms))
      med_fps = float(np.median(live_fps)) if live_fps else 0.0

      ir = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
      native_vis = colorize_depth(native, args.min_depth, args.max_depth)
      fs_vis = colorize_depth(fs_depth, args.min_depth, args.max_depth)
      error_vis = colorize_depth(np.where(overlap, error, 0), 0, args.max_error)
      error_vis[~overlap] = (28, 28, 28)
      top = np.hstack((
          panel(ir, "D455 left infrared",
                f"frame {left_frame.get_frame_number()} | emitter {args.emitter}"),
          panel(native_vis, "D455 native depth",
                f"shared range {args.min_depth:.2f}-{args.max_depth:.2f} m | fill {native_fill:.1%}")))
      bottom = np.hstack((
          panel(fs_vis, "Fast-FoundationStereo TensorRT depth",
                f"infer {med_infer:.1f} ms | fill {fs_fill:.1%}"),
          panel(error_vis, "Absolute depth difference",
                f"median {median_error * 1000:.1f} mm | display 0-{args.max_error * 1000:.0f} mm")))
      composite = np.vstack((top, bottom))
      footer = f"live {med_fps:.1f} FPS | processing {med_process:.1f} ms | S save | Q/Esc quit"
      cv2.putText(composite, footer, (18, composite.shape[0] - 16),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
      cv2.imshow(window, composite)

      if warmup_left > 0:
        warmup_left -= 1
      elif not auto_saved and args.snapshot_after >= 0:
        args.snapshot_after -= 1
        if args.snapshot_after <= 0:
          meta = dict(device_meta, frame=int(left_frame.get_frame_number()),
                      median_inference_ms=med_infer,
                      median_processing_ms=med_process, live_fps=med_fps,
                      native_fill=native_fill, fs_fill=fs_fill,
                      overlap_median_abs_depth_m=median_error)
          saved = save_snapshot(args.out, left_frame.get_frame_number(), composite,
                                left, right, native, fs_depth, disparity, meta)
          print(f"SNAPSHOT_SAVED {saved}", flush=True)
          auto_saved = True

      key = cv2.waitKey(1) & 0xFF
      if key in (ord("q"), 27):
        break
      if key == ord("s"):
        meta = dict(device_meta, frame=int(left_frame.get_frame_number()),
                    median_inference_ms=med_infer,
                    median_processing_ms=med_process, live_fps=med_fps,
                    native_fill=native_fill, fs_fill=fs_fill,
                    overlap_median_abs_depth_m=median_error)
        saved = save_snapshot(args.out, left_frame.get_frame_number(), composite,
                              left, right, native, fs_depth, disparity, meta)
        print(f"SNAPSHOT_SAVED {saved}", flush=True)
  finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("LIVE_VIEW_STOPPED", flush=True)


if __name__ == "__main__":
  main()
