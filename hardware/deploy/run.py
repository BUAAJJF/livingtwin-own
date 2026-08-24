"""The 50 Hz loop: camera and joints in, joint targets out.

The interesting content of this file is what it does when something is wrong,
because the happy path is eight lines.  A vision policy on a real arm fails in
three ways that all look like "it moved oddly" from outside, and all three are
cheap to catch here and expensive to diagnose later.

**The frame is stale.**  A camera that stops delivering leaves its last frame
in the buffer.  The policy keeps acting on it and the arm keeps moving through
a scene that has changed.  ``--max-frame-age`` stops the arm instead.

**The loop is late.**  At 50 Hz the budget is 20 ms and the policy alone takes
several.  If a period is missed the commands still go out, just late, and the
slew limiter -- which assumes a fixed period -- lets the arm travel further per
command than the joint can manage.  Overruns are counted and reported, and a
sustained overrun stops the run.

**There is nothing to fetch.**  The mask channel is all zeros, which during
training only happened when the object was occluded for a moment.  A policy
driven by an empty mask for several seconds is not doing the task.  The loop
holds position and says so rather than letting the arm wander.

Three modes, in the order to use them:

    --dry-run              nothing plugged in; exercises the whole path
    --no-arm               real camera, no motion; watch the mask find things
    (neither)              the robot moves

Nothing here is safe to run unattended.  The first real run should be with the
arm's own emergency stop within reach and the workspace clear.
"""

from __future__ import annotations

import os

# Before numpy, and it has to be before numpy: the thread pools are sized at
# import time.  On a 24-core machine the default is 24 threads per pool, three
# pools deep, and a control loop that does under a millisecond of arithmetic
# spends ten times that waiting for them to be scheduled.  Measured on this
# one, on a loop with nothing but the policy in it: 14.8 ms median and 44
# overruns in 6 s at the default, 1.5 ms median and none with this set.  The
# work has no parallelism worth having -- the images are a quarter of a
# megapixel -- so the pools are pure cost.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_var, "1")

import argparse                                            # noqa: E402
import dataclasses                                         # noqa: E402
import json                                                # noqa: E402
import pathlib                                             # noqa: E402
import signal                                              # noqa: E402
import threading                                           # noqa: E402
import sys                                                 # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

from . import config, mask, obs, proprio, rectify, robot  # noqa: E402


class Rates:
  """What the loop actually achieved, which is not what it was asked for."""

  def __init__(self):
    self.steps = 0
    self.overruns = 0
    self.worst_ms = 0.0
    self.stale = 0
    self.no_target = 0
    self._t = []

  def note(self, ms: float, budget_ms: float) -> None:
    self.steps += 1
    self.worst_ms = max(self.worst_ms, ms)
    self._t.append(ms)
    if ms > budget_ms:
      self.overruns += 1

  def summary(self) -> str:
    t = np.asarray(self._t) if self._t else np.zeros(1)
    return (f"{self.steps} steps, median {np.median(t):.1f} ms, "
            f"p95 {np.percentile(t, 95):.1f} ms, worst {self.worst_ms:.1f} ms, "
            f"{self.overruns} overrun(s), {self.stale} stale frame(s), "
            f"{self.no_target} step(s) with nothing to fetch")


def build(a):
  """Everything the loop needs, and a clear error for whatever is missing."""
  spec = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())

  if pathlib.Path(config.RIG_FILE).exists():
    rig = config.Rig.load()
    print(f"rig: calibrated, residual "
          f"{rig.residual_mm if rig.residual_mm is not None else float('nan'):.2f} mm, "
          f"table at {rig.table_z * 1000:+.1f} mm")
  elif a.dry_run or a.allow_nominal:
    rig = config.Rig.nominal()
    rig.K = rectify._default_d405_K()
    print("rig: NOT CALIBRATED -- using the simulator's nominal camera pose.  "
          "This is a diagnostic, not a deployment; run "
          "`python -m hardware.deploy.calibrate --collect`.")
  else:
    raise SystemExit(
      f"no {config.RIG_FILE}.  Run `python -m hardware.deploy.calibrate "
      "--collect` then `--solve`, or pass --allow-nominal to run against the "
      "simulator's assumed camera pose and accept that it is not measured."
    )

  reader = None
  if a.replay:
    reader = _Replay(a.replay)
    print(f"replaying {reader.n} frames from {a.replay}")
  elif not a.dry_run:
    from . import sensor
    reader = sensor.Reader(serial=a.serial)
    reader.wait_for_first()
    rig.K = reader.K              # the camera's own, not the stored one
    rig.serial = reader.serial

  reproj = rectify.Reprojector(rig, device=a.device)
  segmenter = _segmenter(a, rig, reproj)
  tracker = mask.TargetTracker()
  builder = proprio.ProprioBuilder()
  mapper = robot.ActionMapper(spec, dt=1.0 / config.CONTROL_HZ)

  arm = (robot.DryRunArm(spec) if (a.dry_run or a.no_arm or a.replay)
         else robot.PiperArm(a.can))
  arm.connect()
  arm.enable()

  from .policy import Policy
  pol = Policy(a.policy, threads=a.policy_threads,
               providers=(["CUDAExecutionProvider", "CPUExecutionProvider"]
                          if a.policy_device == "cuda"
                          else ["CPUExecutionProvider"]))
  print(f"policy: {pol.path}, {pol.flat_width} + "
        f"{'x'.join(map(str, pol.image_shapes[0]))} in, hidden "
        f"{pol.hidden_shape}")
  return rig, reader, reproj, segmenter, tracker, builder, mapper, arm, pol


def _segmenter(a, rig, reproj):
  """Build the mask backend the flags asked for.

  Separate from ``setup`` because there are now three of them and the choice
  has a consequence worth reading in one place: the depth backend is the
  measurement, the colour backend is the inference, and ``fused`` is the depth
  backend with the colour one allowed to add instances it did not find.  Only
  ``fused`` covers the failure that motivated training a model at all without
  giving up the accuracy of the one that needs no training.
  """
  depth_seg = mask.DepthSegmenter(rig, reproj)
  if a.mask == "depth":
    return depth_seg
  cfg = mask.YoloCfg()
  if getattr(a, "yolo_conf", None) is not None:
    cfg = dataclasses.replace(cfg, conf=float(a.yolo_conf))
  yolo_seg = mask.YoloSegmenter(
    a.yolo_weights, rig, reproj, yolo_cfg=cfg,
    device=getattr(a, "yolo_device", None) or "cuda:0")
  if a.mask == "yolo":
    return yolo_seg
  return mask.FusedSegmenter(depth_seg, yolo_seg)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--policy", required=True,
                 help="exported policy.onnx, or the directory holding it")
  p.add_argument("--seconds", type=float, default=60.0)
  p.add_argument("--dry-run", action="store_true",
                 help="no camera and no arm: renders nothing, moves nothing")
  p.add_argument("--no-arm", action="store_true",
                 help="real camera, simulated arm")
  p.add_argument("--allow-nominal", action="store_true")
  p.add_argument("--mask", choices=("depth", "yolo", "fused"), default="depth",
                 help="'depth' needs no model and is the accurate one wherever "
                      "there is depth to segment.  'yolo' reads the mono image "
                      "instead and works where there is not.  'fused' runs the "
                      "first and lets the second add what it missed, which is "
                      "what should be on the robot -- at one forward pass a "
                      "frame.")
  p.add_argument("--yolo-weights", default=str(
    pathlib.Path(__file__).resolve().parent / "yolo" / "best.pt"),
                 help="``.pt`` goes through ultralytics and ``.onnx`` through "
                      "onnxruntime, which drops the torch dependency; see "
                      "yolo_backend.py for what each costs")
  p.add_argument("--yolo-conf", type=float, default=None,
                 help="detection confidence, overriding mask.YoloCfg")
  p.add_argument("--yolo-device", default=None,
                 help="where the detector runs.  Defaults to cuda:0 -- it is "
                      "the only part of the perception path that wants a GPU, "
                      "and it is 8 ms there against 61 on the CPU")
  p.add_argument("--device", default="cpu",
                 help="where the resampling runs.  CPU is the measured "
                      "default and it is not a fallback: the cloud is 400k "
                      "points and moving it to the GPU and back costs more "
                      "than the arithmetic saves -- 56 ms a frame against 22")
  p.add_argument("--policy-device", default="cpu", choices=("cpu", "cuda"),
                 help="where the network runs.  CPU with a capped thread pool "
                      "is enough at 50 Hz and leaves the cores for vision")
  p.add_argument("--policy-threads", type=int, default=2)
  p.add_argument("--serial", default=None)
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--max-obs-age", type=float, default=0.20,
                 help="seconds since the vision thread last published.  The "
                      "camera runs at 30 Hz and the vision at 15-30, so "
                      "anything past 200 ms is a fault, not jitter")
  p.add_argument("--max-overrun-streak", type=int, default=25)
  p.add_argument("--replay", default=None,
                 help="a recorded session to feed instead of the camera.  The "
                      "arm still runs (dry, unless --no-arm is off), so this "
                      "is what a rollout looked like from the policy's side, "
                      "and the only way to measure the loop's real cost "
                      "without hardware")
  p.add_argument("--record", default=None,
                 help="directory to write the session to, for replay and for "
                      "labelling a YOLO training set")
  a = p.parse_args()

  (rig, reader, reproj, segmenter, tracker, builder, mapper, arm,
   pol) = build(a)

  stopping = False

  def _stop(*_):
    nonlocal stopping
    stopping = True
  signal.signal(signal.SIGINT, _stop)

  dt = 1.0 / config.CONTROL_HZ
  budget_ms = dt * 1000
  rates = Rates()
  last_action = np.zeros(7, dtype=np.float32)
  overrun_streak = 0

  st = arm.read()
  mapper.reset(np.concatenate([st.q, [st.gripper, -st.gripper]]))
  pol.reset()

  writer = _Recorder(a.record) if a.record else None
  if writer is not None:
    # Beside the frames, so the session can be labelled later against the
    # camera it was actually recorded with rather than whichever calibration
    # is on disk when someone gets round to it.
    rig.save(pathlib.Path(a.record) / "rig.json")
  vision = None
  if reader is not None:
    vision = Perception(reader, reproj, segmenter, tracker,
                        proprio.Kinematics())
    vision.set_joints(np.concatenate([st.q, [st.gripper, -st.gripper]]))
    vision.start()

  blank = np.zeros((3, config.HEIGHT, config.WIDTH), np.float32)
  print(f"running for {a.seconds:.0f} s at {config.CONTROL_HZ:.0f} Hz; "
        "ctrl-c to stop")
  t_end = time.time() + a.seconds
  next_tick = time.time()
  try:
    while time.time() < t_end and not stopping:
      t0 = time.time()

      st = arm.read()
      fb = robot.feedback(st, mapper.previous)
      if vision is not None:
        vision.set_joints(fb.position)

      out = vision.latest() if vision is not None else None
      if out is None:
        camera, label, frame = blank, 0, None
        if vision is not None:
          # Nothing to act on yet.  Not an error in the first fraction of a
          # second, and not something to drive an arm with either.
          rates.stale += 1
          arm.hold()
          time.sleep(dt)
          continue
      else:
        camera, stamp, frame, label = out
        age = time.time() - stamp
        if age > a.max_obs_age:
          rates.stale += 1
          arm.hold()
          time.sleep(dt)
          continue
      if not label:
        rates.no_target += 1

      flat = builder(fb, last_action)
      last_action = pol(flat, camera)
      arm.command(mapper(last_action), dt)

      if writer is not None and frame is not None:
        writer.write(frame.depth, frame.gray, fb, last_action, label)

      ms = (time.time() - t0) * 1000
      rates.note(ms, budget_ms)
      overrun_streak = overrun_streak + 1 if ms > budget_ms else 0
      if overrun_streak >= a.max_overrun_streak:
        print(f"\nSTOPPING: {overrun_streak} consecutive steps over the "
              f"{budget_ms:.0f} ms budget.  The slew limiter assumes a fixed "
              "period, so late commands let the arm travel further than the "
              "joint can.")
        break

      next_tick += dt
      slack = next_tick - time.time()
      if slack > 0:
        time.sleep(slack)
      else:
        next_tick = time.time()
  finally:
    arm.hold()
    arm.close()
    if vision is not None:
      vision.close()
      print(vision.summary())
    if reader is not None:
      reader.close()
    if writer is not None:
      writer.close()
    print("\n" + rates.summary())
  return 0


class Perception(threading.Thread):
  """Segmentation and resampling, on their own thread, at their own rate.

  This is the piece that decides whether the whole thing runs.  The control
  loop is cheap -- proprioception, one forward pass, a joint command, 1.1 ms
  median -- and the vision is not: segmenting a quarter of a million points and
  re-rendering them costs 35 ms on this machine's CPU, and 60 before the
  segmenter was moved to half resolution.  The first version of this loop did
  it inline and stopped itself after twenty-five consecutive overruns.  It was
  right to.

  Putting it on a thread is not a workaround, it is the correct structure.  The
  camera produces 30 frames a second; running the vision more often than that
  computes the same answer twice.  The simulator does the same thing -- the
  camera sensor is sampled at the control rate and the renderer does not
  produce a new image on every physics step either.

  What it costs is latency, and the cost is explicit: the observation the
  policy acts on is one vision period old, which at the measured 28 Hz is 1.8
  control steps.  That is inside the 2-4 steps ``piper_push.perturb`` calls the
  hardware range for "a USB depth camera at 30 fps with a copy and a forward
  pass", so it is a delay the policy has already been evaluated against rather
  than a new one.  ``--max-obs-age`` is the ceiling, and the loop holds the arm
  rather than acting on anything older.
  """

  def __init__(self, reader, reproj, segmenter, tracker, kin):
    super().__init__(daemon=True, name="perception")
    self.reader = reader
    self.reproj = reproj
    self.segmenter = segmenter
    self.tracker = tracker
    self.kin = kin
    self._lock = threading.Lock()
    self._joints = None
    self._out = None
    # Not ``_stop``: ``threading.Thread`` already has one and shadowing it
    # replaces a method the interpreter calls when the thread ends.
    self._stopping = threading.Event()
    self.periods: list[float] = []
    self.frames = 0

  def set_joints(self, q) -> None:
    """The control loop's most recent joint reading.

    The segmenter needs the arm's pose to subtract it, and it must not read the
    arm itself: two threads talking to one CAN interface is a way to lose
    frames on both.
    """
    with self._lock:
      self._joints = np.asarray(q, dtype=np.float64).copy()

  def latest(self):
    with self._lock:
      return self._out

  def run(self) -> None:
    last_index = -1
    while not self._stopping.is_set():
      frame = self.reader.latest()
      with self._lock:
        q = None if self._joints is None else self._joints.copy()
      if frame is None or q is None or frame.index == last_index:
        time.sleep(0.002)
        continue
      last_index = frame.index
      t0 = time.time()

      self.kin.update(q)
      seg = self.segmenter(frame.depth, rgb=frame.gray,
                           arm=self.kin.link_spheres())
      label = self.tracker.update(seg, self.kin.site_pos)
      payload = (mask.full_mask(seg, label, self.segmenter.decimate)
                 if label else None)
      depth, valid, target = self.reproj(frame.depth, payload=payload)
      if target is None:
        target = np.zeros_like(valid)
      camera = obs.camera_obs(depth, valid, target > 0)

      self.periods.append(time.time() - t0)
      self.frames += 1
      with self._lock:
        self._out = (camera, time.time(), frame, label)

  def close(self) -> None:
    self._stopping.set()
    self.join(timeout=2.0)

  def summary(self) -> str:
    if not self.periods:
      return "perception: no frames"
    t = np.asarray(self.periods) * 1000
    return (f"perception: {self.frames} frames, median {np.median(t):.1f} ms "
            f"({1000 / max(np.median(t), 1e-6):.1f} Hz), p95 "
            f"{np.percentile(t, 95):.1f} ms")


class _Replay:
  """A recorded session, served with the same interface the camera has.

  Deliberately not a simulator of the camera: it does not sleep, it does not
  drop frames, and it runs off the end.  What it is for is measuring what the
  loop costs with a real image in it -- the segmenter is most of the work and
  ``--dry-run`` skips it entirely -- and for looking at a session again after
  something went wrong on the robot.
  """

  def __init__(self, path: str, fps: float = config.D405_FPS,
               loop: bool = True):
    self.dir = pathlib.Path(path)
    self.meta = json.loads((self.dir / "meta.json").read_text())
    self.n = len(self.meta)
    self.fps = float(fps)
    self.loop = loop
    self.t0 = time.time()
    self._i = -1
    self._frame = None
    self.age = 0.0

  def latest(self):
    """The frame the camera would be showing now, decoded at most once.

    Served on a clock rather than one per call, and cached, for two reasons.
    The camera delivers 30 frames a second whether anyone asks or not, so a
    reader that hands out a new frame per call lets the vision thread run
    through a session faster than it could ever have been recorded.  And
    decoding half a megabyte of npz costs 200 ms -- fifteen times the whole
    perception stage -- so a reader that decodes per call measures the file
    system and reports it as the cost of the pipeline.  That is what the first
    version did, and it made perception look like 4 Hz work.
    """
    from . import sensor

    k = int((time.time() - self.t0) * self.fps)
    k = k % self.n if self.loop else min(k, self.n - 1)
    if k != self._i:
      rec = self.meta[k]
      f = np.load(self.dir / f"{rec['i']:06d}.npz")
      self._frame = sensor.Frame(depth=f["depth"].astype(np.float32) / 10000.0,
                                 gray=f["gray"], stamp=time.time(), index=k)
      self._i = k
    return self._frame

  def close(self) -> None:
    pass


class _Recorder:
  """Everything needed to replay the session offline, and to label it.

  Depth and greyscale at full sensor resolution, because that is what
  ``autolabel.py`` wants and because a recording downsampled to the policy's
  grid can only ever reproduce the run, never re-examine it.
  """

  def __init__(self, path: str):
    self.dir = pathlib.Path(path)
    self.dir.mkdir(parents=True, exist_ok=True)
    self.n = 0
    self.meta = []

  def write(self, depth, gray, fb, action, label) -> None:
    np.savez_compressed(
      self.dir / f"{self.n:06d}.npz",
      depth=(np.asarray(depth) * 10000).astype(np.uint16),   # 0.1 mm, as the
      gray=gray if gray is not None else np.zeros((1, 1), np.uint8),
    )
    self.meta.append({
      "i": self.n, "t": time.time(),
      "joint_pos": np.asarray(fb.position).tolist(),
      "joint_vel": np.asarray(fb.velocity).tolist(),
      "target": np.asarray(fb.target).tolist(),
      "action": np.asarray(action).tolist(),
      "label": int(label),
    })
    self.n += 1

  def close(self) -> None:
    (self.dir / "meta.json").write_text(json.dumps(self.meta))
    print(f"recorded {self.n} frames to {self.dir}")


if __name__ == "__main__":
  sys.exit(main())
