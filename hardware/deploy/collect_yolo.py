"""Record D405/D455 frames for YOLO labels without driving the robot.

This is deliberately separate from ``run.py``.  ``run.py`` is a control loop:
it enables the drives, evaluates a policy and sends commands.  Dataset capture
needs none of those things.  It connects CAN only to read the physical joint
pose used by offline arm subtraction, never calls ``enable`` or ``command``,
and disconnects without changing drive state or targets.

Example::

  python -m hardware.deploy.collect_yolo --camera d455 --seconds 120 \
    --out recordings/d455_yolo/session01
"""

from __future__ import annotations

import argparse
import json
import pathlib
import signal
import sys
import time

import numpy as np

from . import config, robot, sensor
from .run import _Recorder


def _wait_for_arm_feedback(arm, timeout_s: float = 3.0):
  """Discard the SDK's zero-initialised CAN message and require stability."""
  deadline = time.monotonic() + timeout_s
  previous = None
  stable = 0
  last = None
  while time.monotonic() < deadline:
    last = arm.read()
    q = np.asarray(last.q, dtype=np.float64)
    initialized = bool(np.max(np.abs(q)) > 1e-5)
    if (initialized and previous is not None
        and np.max(np.abs(q - previous)) < np.radians(0.5)):
      stable += 1
      if stable >= 3:
        return last
    else:
      stable = 0
    previous = q.copy()
    time.sleep(0.05)
  shown = [] if last is None else np.degrees(last.q).round(2).tolist()
  raise RuntimeError(
    f"CAN feedback did not initialise and settle within {timeout_s:.1f} s; "
    f"last q_deg={shown}")


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--camera", choices=("d405", "d455"), default="d455")
  p.add_argument("--serial", default=None)
  p.add_argument("--rig-file", default=None)
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--seconds", type=float, default=120.0)
  p.add_argument("--out", required=True)
  a = p.parse_args()

  suffix = "" if a.camera == "d405" else f"_{a.camera}"
  rig_path = (pathlib.Path(a.rig_file) if a.rig_file
              else pathlib.Path(config.RIG_FILE).with_name(f"rig{suffix}.json"))
  if not rig_path.exists():
    raise SystemExit(f"no calibration at {rig_path}; solve and save it first")
  rig = config.Rig.load(rig_path)

  out = pathlib.Path(a.out)
  if out.exists() and any(out.iterdir()):
    raise SystemExit(f"refusing to overwrite non-empty recording {out}")

  reader = sensor.Reader(serial=a.serial, backend=a.camera)
  arm = None
  writer = None
  stopping = False

  def stop(*_):
    nonlocal stopping
    stopping = True

  signal.signal(signal.SIGINT, stop)
  try:
    frame = reader.wait_for_first()
    if (rig.serial and reader.serial
        and str(rig.serial) != str(reader.serial)):
      raise RuntimeError(
        f"{rig_path} belongs to camera {rig.serial}, connected camera is "
        f"{reader.serial}")
    rig.K = reader.K
    rig.serial = reader.serial

    arm = robot.PiperArm(a.can)
    arm.connect()                    # feedback only: no enable, no command
    settled = _wait_for_arm_feedback(arm)
    writer = _Recorder(str(out))
    rig.save(out / "rig.json")
    (out / "capture.json").write_text(json.dumps({
      "camera": a.camera,
      "serial": reader.serial,
      "model": reader.meta.get("model"),
      "gray_source": reader.meta.get("gray_source"),
      "emitter": reader.meta.get("emitter"),
      "resolution": reader.meta.get("resolution"),
      "fps": reader.meta.get("fps"),
      "rig_source": str(rig_path.resolve()),
      "read_only_arm": True,
    }, indent=2) + "\n")

    print(f"record-only: {reader.meta.get('model')} {reader.serial}, "
          f"{reader.meta.get('gray_source')} gray, emitter "
          f"{reader.meta.get('emitter')}")
    print("arm: CAN feedback only; drives will not be enabled or commanded")
    print("arm feedback settled at q_deg="
          f"{np.degrees(settled.q).round(1).tolist()}")
    print(f"recording for {a.seconds:.0f} s to {out}; Ctrl-C stops safely")

    last_index = -1
    deadline = time.monotonic() + a.seconds
    zeros = np.zeros(7, dtype=np.float32)
    while time.monotonic() < deadline and not stopping:
      frame = reader.latest()
      if frame is None or frame.index == last_index:
        time.sleep(0.002)
        continue
      last_index = frame.index
      st = arm.read()
      target = np.r_[st.q, st.gripper]
      fb = robot.feedback(st, target)
      writer.write(frame.depth, frame.gray, fb, zeros, 0)
  finally:
    if writer is not None:
      writer.close()
    reader.close()
    if arm is not None:
      arm.disconnect()               # no same-pose command on read-only exit
  return 0


if __name__ == "__main__":
  sys.exit(main())
