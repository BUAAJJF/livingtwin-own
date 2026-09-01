#!/usr/bin/env python3
"""Capture one synchronized D455 native-depth/IR-stereo sample.

The emitted IR pair is rectified by the D455 firmware and is ready for
FoundationStereo.  Native depth remains in RealSense Z16 units; metadata.json
records the scale, rectified intrinsics, and measured stereo baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--width", type=int, default=848)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--warmup", type=int, default=30)
  ap.add_argument("--emitter", choices=("on", "off"), default="on")
  args = ap.parse_args()

  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
  config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
  config.enable_stream(rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps)
  profile = pipeline.start(config)
  try:
    sensor = profile.get_device().first_depth_sensor()
    if sensor.supports(rs.option.emitter_enabled):
      sensor.set_option(rs.option.emitter_enabled, 1.0 if args.emitter == "on" else 0.0)
    for _ in range(args.warmup):
      pipeline.wait_for_frames()
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    left_frame = frames.get_infrared_frame(1)
    right_frame = frames.get_infrared_frame(2)
    if not depth_frame or not left_frame or not right_frame:
      raise RuntimeError("D455 did not return a complete synchronized frameset")

    depth = np.asanyarray(depth_frame.get_data())
    left = np.asanyarray(left_frame.get_data())
    right = np.asanyarray(right_frame.get_data())
    intr = left_frame.profile.as_video_stream_profile().intrinsics
    right_profile = right_frame.profile.as_video_stream_profile()
    extr = left_frame.profile.get_extrinsics_to(right_profile)
    baseline_m = float(np.linalg.norm(np.asarray(extr.translation)))
    device = profile.get_device()
    meta = {
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": device.get_info(rs.camera_info.name),
        "serial": device.get_info(rs.camera_info.serial_number),
        "firmware": device.get_info(rs.camera_info.firmware_version),
        "resolution": [args.width, args.height],
        "fps": args.fps,
        "emitter": args.emitter,
        "depth_scale_m": sensor.get_depth_scale(),
        "left_intrinsics": {
            "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
            "model": str(intr.model), "coeffs": list(intr.coeffs),
        },
        "left_to_right_translation_m": list(extr.translation),
        "left_to_right_rotation": list(extr.rotation),
        "baseline_m": baseline_m,
        "frame_numbers": {
            "depth": depth_frame.get_frame_number(),
            "left": left_frame.get_frame_number(),
            "right": right_frame.get_frame_number(),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "left_ir.png"), left)
    cv2.imwrite(str(args.out / "right_ir.png"), right)
    cv2.imwrite(str(args.out / "native_depth_z16.png"), depth)
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"capture -> {args.out}")
  finally:
    pipeline.stop()


if __name__ == "__main__":
  main()

