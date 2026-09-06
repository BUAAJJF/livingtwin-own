"""Shadow-mode runner for the point-cloud policies: camera and joints in, actions out, NO motor commands.

What this measures, and what it refuses to do.  It runs the exact deployment
structure the brief specifies --

  * a camera thread at the D455's 30 Hz (``sensor.Reader``, or a recorded
    session through ``run._Replay`` served on its own clock);
  * a perception thread that turns the newest frame into the policy's cloud or
    depth observation (``pc_obs.CloudObs``, plus the P2 candidates and lock);
    a one-slot mailbox, latest wins, nothing queues behind a slow frame;
  * a 50 Hz control loop that never waits for perception: it reads the slot,
    stamps ``vision_meta`` = (age in control steps / 10, fresh, valid) from the
    frame's capture time, builds proprioception, runs the ONNX policy with its
    explicit GRU state, maps ``tanh(u)`` to joint targets through the same
    ``ActionMapper`` the arm would get -- and writes them to the log instead of
    the CAN bus.  Missed periods are counted, never caught up.

-- and it never enables a drive.  The three sources of joint state are the
recorded session (``--replay``), a real arm read without enabling anything
(``--arm-read``), or the dry-run arm.  A first real motion is a separate,
human-confirmed step; see the runbook.

Safety states are computed and logged exactly as a live run would act on
them: ``hold_no_vision``, ``hold_stale`` (age above ``--max-vision-age``),
``hold_no_depth`` (too few workspace points), ``hold_no_candidate`` (P2, no
locked candidate), ``hold_nonfinite``.

    python -m hardware.deploy.pc_run --policy bundles/pc_P1B --replay recordings/v4_stereo_try3 \\
        --seconds 30 --record recordings/pc_shadow_replay_try1
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_var, "1")

import argparse                                            # noqa: E402
import json                                                # noqa: E402
import pathlib                                             # noqa: E402
import threading                                           # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

from . import config, proprio, robot                       # noqa: E402
from .pc_obs import CloudObs, GraspObs                     # noqa: E402
from piper_push.pc import cloud as pc_cloud                # noqa: E402

CONTROL_DT = 1.0 / config.CONTROL_HZ


class PcPolicy:
  """ONNX Runtime with the recurrent state carried by hand; inputs matched by name."""

  def __init__(self, path: pathlib.Path, spec: dict, threads: int = 2, cuda: bool = True) -> None:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, int(threads))
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    self.sess = ort.InferenceSession(str(path), opts, providers=providers)
    inputs = self.sess.get_inputs()
    self.names = [i.name for i in inputs]
    self.nd_names = self.names[1:-1]
    self.shapes = {i.name: tuple(1 if not isinstance(d, int) else d for d in i.shape) for i in inputs}
    self.groups_1d = [g for g in spec["actor_groups"] if len(spec["groups"][g]["shape"]) == 1]
    self.flat_width = sum(spec["groups"][g]["total"] for g in self.groups_1d)
    if self.flat_width != self.shapes[self.names[0]][-1]:
      raise RuntimeError(f"obs_spec says the flat input is {self.flat_width} wide; the graph says "
                         f"{self.shapes[self.names[0]][-1]}")
    for g in self.nd_names:
      if g not in spec["groups"]:
        raise RuntimeError(f"graph input {g!r} is not a group in obs_spec.json")
    self.hidden_shape = self.shapes[self.names[-1]]
    self.hidden = np.zeros(self.hidden_shape, dtype=np.float32)
    self.provider = self.sess.get_providers()[0]

  def reset(self) -> None:
    self.hidden[:] = 0.0

  def __call__(self, parts_1d: dict[str, np.ndarray], parts_nd: dict[str, np.ndarray]) -> np.ndarray:
    flat = np.concatenate([np.asarray(parts_1d[g], dtype=np.float32).reshape(-1) for g in self.groups_1d])
    feed = {self.names[0]: flat.reshape(1, -1)}
    for g in self.nd_names:
      feed[g] = np.asarray(parts_nd[g], dtype=np.float32)[None]
    feed[self.names[-1]] = self.hidden
    u, self.hidden = self.sess.run(None, feed)
    return np.asarray(u, dtype=np.float64).reshape(-1)


class _RecordedJoints:
  """Joint feedback from a recorded session's control.json, served by time."""

  def __init__(self, path: pathlib.Path, names: list[str]) -> None:
    recs = json.loads((path / "control.json").read_text())
    self.recs = [r for r in recs if "joint_pos" in r]
    self.t = np.array([r["t"] for r in self.recs])
    self.t0 = time.time()
    self.names = names

  def read(self) -> proprio.JointFeedback:
    k = int(np.searchsorted(self.t, self.t[0] + (time.time() - self.t0)))
    r = self.recs[min(k, len(self.recs) - 1)]
    q = np.asarray(r["joint_pos"], dtype=np.float64)
    dq = np.asarray(r.get("joint_vel", np.zeros_like(q)), dtype=np.float64)
    tg = np.asarray(r.get("target", q), dtype=np.float64)
    return proprio.JointFeedback.from_arm(q[:6], dq[:6], q[6], dq[6], tg[:6], tg[6],
                                         float(r.get("gripper_effort", 0.0)), self.names)


class _DryJoints:
  def __init__(self, spec: dict) -> None:
    self.q = np.asarray(spec["default_joint_pos"], dtype=np.float64)
    self.names = list(spec["joint_names"])

  def read(self) -> proprio.JointFeedback:
    z = np.zeros(6)
    return proprio.JointFeedback.from_arm(self.q[:6], z, self.q[6], 0.0, self.q[:6], self.q[6], 0.0, self.names)


class _ArmReadOnly:
  """A real arm, connected for feedback only.  Drives are never enabled here."""

  def __init__(self, spec: dict, can: str) -> None:
    # connect() opens CAN and reads; enable() is what powers the drives, and it
    # is never called here.  The measured pose doubles as the "target" the
    # proprioception's squeeze term subtracts, so the squeeze reads zero.
    self.arm = robot.PiperArm(can)
    self.arm.connect()
    self.names = list(spec["joint_names"])
    self.last = None

  def read(self) -> proprio.JointFeedback:
    st = self.arm.read()
    self.last = st
    target = np.concatenate([np.asarray(st.q, dtype=np.float64), [float(st.gripper)]])
    return robot.feedback(st, target)

  def close(self) -> None:
    try:
      self.arm.disconnect()
    except Exception:
      pass


class Perception(threading.Thread):
  def __init__(self, reader, obs: CloudObs, grasp: GraspObs | None, joints_ref: dict) -> None:
    super().__init__(daemon=True, name="pc-perception")
    self.reader, self.obs, self.grasp, self.joints_ref = reader, obs, grasp, joints_ref
    self.slot = None
    self.lock = threading.Lock()
    self.compute_ms: list[float] = []
    self.frames = 0
    self.dropped = 0
    self._halt = threading.Event()
    self._last_index = -1

  def latest(self):
    with self.lock:
      return self.slot

  def stop(self) -> None:
    self._halt.set()

  def run(self) -> None:
    while not self._halt.is_set():
      frame = self.reader.latest()
      if frame is None or frame.index == self._last_index:
        time.sleep(0.001)
        continue
      if self._last_index >= 0 and frame.index > self._last_index + 1:
        self.dropped += frame.index - self._last_index - 1
      self._last_index = frame.index
      t0 = time.perf_counter()
      cf = self.obs(frame.depth)
      extra = None
      if self.grasp is not None:
        js = self.joints_ref.get("js")
        if js is not None:
          self.grasp.kin.update(js.position)
        self.grasp.update(cf)
        extra = (self.grasp.topk.copy(), self.grasp.locked.copy())
      ms = (time.perf_counter() - t0) * 1000.0
      self.compute_ms.append(ms)
      self.frames += 1
      with self.lock:
        self.slot = {"obs": cf.obs, "count": cf.count, "valid": cf.valid, "capture": frame.stamp,
                     "index": frame.index, "done": time.time(), "compute_ms": ms, "extra": extra}


def _pct(xs, q):
  return float(np.percentile(xs, q)) if len(xs) else float("nan")


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--policy", required=True, help="bundle directory: policy.onnx + obs_spec.json + manifest.json")
  p.add_argument("--route", default=None, help="P0 / P1A / P1B / P2; read from manifest.json when omitted")
  p.add_argument("--replay", default=None, help="recorded session to serve instead of a camera")
  p.add_argument("--camera", default="d455")
  p.add_argument("--serial", default=None)
  p.add_argument("--arm-read", action="store_true", help="read a real arm's joints (drives stay off)")
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--rig-file", default=str(pathlib.Path(__file__).with_name("rig_d455.json")))
  p.add_argument("--seconds", type=float, default=20.0)
  p.add_argument("--max-vision-age", type=float, default=0.2, help="seconds; older vision is a hold")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--policy-threads", type=int, default=2)
  p.add_argument("--record", required=True, help="a directory that does not exist yet")
  a = p.parse_args()

  bundle = pathlib.Path(a.policy)
  spec = json.loads((bundle / "obs_spec.json").read_text())
  manifest = json.loads((bundle / "manifest.json").read_text()) if (bundle / "manifest.json").exists() else {}
  route = a.route or manifest.get("route")
  if route not in ("P0", "P1A", "P1B", "P2"):
    raise SystemExit(f"route {route!r}: pass --route or put it in manifest.json")
  out = pathlib.Path(a.record)
  if out.exists():
    raise SystemExit(f"{out} exists; shadow runs never overwrite a session")
  out.mkdir(parents=True)

  rig = config.Rig.load(a.rig_file)
  policy = PcPolicy(bundle / "policy.onnx", spec, threads=a.policy_threads, cuda=a.device.startswith("cuda"))
  mapper = robot.ActionMapper(spec)
  builder = proprio.ProprioBuilder(bundle / "obs_spec.json")
  obs_builder = CloudObs(rig, mode="depth" if route == "P0" else "cloud", num_points=pc_cloud.POINT_DIM * 128, device=a.device)
  if route != "P0":
    obs_builder.num_points = int(spec["groups"]["camera"]["shape"][0])
  grasp = GraspObs(builder.kin, device=a.device) if route == "P2" else None

  if a.replay:
    from .run import _Replay
    reader = _Replay(a.replay, fps=config.D405_FPS, loop=True)
    joints = _RecordedJoints(pathlib.Path(a.replay), list(spec["joint_names"]))
    source = f"replay:{a.replay}"
  else:
    from . import sensor
    reader = sensor.Reader(serial=a.serial, backend=a.camera)
    reader.wait_for_first()
    joints = _ArmReadOnly(spec, a.can) if a.arm_read else _DryJoints(spec)
    source = "camera:" + ("arm-read" if a.arm_read else "dry-joints")

  joints_ref: dict = {"js": None}
  perception = Perception(reader, obs_builder, grasp, joints_ref)
  perception.start()

  js = joints.read()
  joints_ref["js"] = js
  mapper.reset(js.position)
  policy.reset()
  builder.reset()
  last_a = np.zeros(7)
  last_index = -1
  steps = int(a.seconds * config.CONTROL_HZ)
  loop_ms, ages, holds = [], [], {}
  fresh_count, overruns = 0, 0
  log = open(out / "shadow.jsonl", "w")
  t0 = time.perf_counter()
  print(f"shadow {route} from {source}; {steps} control steps; NO motor commands")
  for i in range(steps):
    t_sched = t0 + i * CONTROL_DT
    now = time.perf_counter()
    if now < t_sched:
      time.sleep(t_sched - now)
    elif now > t_sched + CONTROL_DT:
      overruns += 1
    t_step = time.perf_counter()
    js = joints.read()
    joints_ref["js"] = js
    slot = perception.latest()
    hold = None
    if slot is None:
      hold = "hold_no_vision"
      age_steps, fresh, valid = 10.0, 0.0, 0.0
      nd = None
    else:
      age_s = time.time() - slot["capture"]
      age_steps = age_s / CONTROL_DT
      fresh = float(slot["index"] != last_index)
      fresh_count += int(fresh)
      last_index = slot["index"]
      valid = float(slot["valid"])
      ages.append(age_s)
      nd = slot
      if age_s > a.max_vision_age:
        hold = "hold_stale"
      elif not slot["valid"]:
        hold = "hold_no_depth"
    meta = np.array([min(age_steps, pc_cloud.AGE_NORM) / pc_cloud.AGE_NORM, fresh, valid], dtype=np.float32)
    prop = builder(js, last_a)
    parts_1d = {"proprio": prop, "vision_meta": meta}
    parts_nd = {}
    if route == "P2":
      topk, locked = (nd["extra"] if (nd is not None and nd["extra"] is not None)
                      else (np.zeros((32, 18), np.float32), np.zeros(20, np.float32)))
      parts_1d["grasp_locked"] = locked
      parts_nd["grasp_topk"] = topk
      if hold is None and locked[-1] < 0.5:
        hold = "hold_no_candidate"
    else:
      shape = tuple(spec["groups"]["camera"]["shape"])
      parts_nd["camera"] = nd["obs"] if nd is not None else np.zeros(shape, np.float32)
    u = policy(parts_1d, parts_nd)
    if not np.isfinite(u).all():
      hold = hold or "hold_nonfinite"
      u = np.zeros(7)
    a_t = np.tanh(u)
    target = mapper(u) if hold is None else mapper.previous.copy()
    last_a = a_t
    if hold:
      holds[hold] = holds.get(hold, 0) + 1
    ms = (time.perf_counter() - t_step) * 1000.0
    loop_ms.append(ms)
    log.write(json.dumps({"i": i, "t": time.time(), "age_steps": round(float(age_steps), 3), "fresh": fresh,
                          "valid": valid, "hold": hold, "u": [round(float(x), 4) for x in u],
                          "a": [round(float(x), 4) for x in a_t], "target": [round(float(x), 5) for x in target],
                          "loop_ms": round(ms, 3), "vision_index": last_index,
                          "perception_ms": (round(nd["compute_ms"], 2) if nd else None)}) + "\n")
  log.close()
  perception.stop()
  elapsed = time.perf_counter() - t0
  try:
    import torch
    gpu_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
  except Exception:
    gpu_mb = None
  summary = {
    "route": route, "policy": str(bundle), "source": source, "rig": a.rig_file, "seconds": elapsed,
    "control_steps": steps, "overruns": overruns,
    "control_ms": {"p50": _pct(loop_ms, 50), "p95": _pct(loop_ms, 95), "max": float(max(loop_ms))},
    "perception_ms": {"p50": _pct(perception.compute_ms, 50), "p95": _pct(perception.compute_ms, 95),
                      "frames": perception.frames, "dropped_by_mailbox": perception.dropped},
    "effective_vision_hz": fresh_count / elapsed,
    "frame_age_s": {"p50": _pct(ages, 50), "p95": _pct(ages, 95)},
    "holds": holds, "hold_fraction": sum(holds.values()) / max(steps, 1),
    "onnx_provider": policy.provider, "gpu_max_alloc_mb": gpu_mb,
    "action_api": mapper.action_api, "action_api_status": mapper.action_api_status,
    "manifest": manifest,
  }
  if grasp is not None:
    summary["p2"] = {"switches": grasp.switches, "no_candidate_frames": grasp.no_candidate_frames, "frames": grasp.frames}
  (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps({k: summary[k] for k in ("control_ms", "perception_ms", "effective_vision_hz", "frame_age_s", "overruns", "holds", "onnx_provider")}, indent=1))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
