"""The perception thread for the point-cloud policies, with ``run.Perception``'s interface.

``run.py``'s control loop, guards, recorder and arm path stay exactly as they
are for the mask policies; this thread is what ``--obs pc`` swaps in for the
segmenter.  It publishes the same six-tuple the loop unpacks --
``(camera, published_at, frame, label, held_over, target_available)`` -- with
``camera`` the policy's cloud (N, 4) or depth (2, H, W) observation from
``pc_obs.CloudObs``, ``label`` 0 (there is no instance), ``held_over`` False
(there is no mask to hold over), and ``target_available`` meaning "enough
workspace points to be a frame" (``MIN_POINTS``), so the loop's existing
``hold_no_target`` branch is exactly the ``hold_no_depth`` state of the shadow
runner.  The P2 candidates and lock are computed here on every frame from the
joints the loop hands in through ``set_joints``.

One frame in, one observation out, latest wins, nothing queued: the loop reads
``latest()`` and never waits.
"""

from __future__ import annotations

import collections
import json
import pathlib
import threading
import time

import numpy as np

from . import proprio
from .pc_obs import CloudObs, GraspObs
from piper_push.pc import cloud as pc_cloud
from piper_push.pc import routes as pc_routes


class PcPerception(threading.Thread):
  def __init__(self, reader, rig, route: str, num_points: int, kin: proprio.Kinematics,
               device: str = "cuda:0", stereo=None, height_min_m: float | None = None) -> None:
    super().__init__(daemon=True, name="pc-perception")
    pc_routes.check_deployable(route)
    base = pc_routes.base_route(route)
    self.reader = reader
    self.route = route
    self.stereo = stereo
    self.stereo_misses = 0
    # The cut above the calibrated plane: the route's by default.  The
    # simulator's stress sweep (results/pc/gen4/robustness) found a cut 4 mm
    # too HIGH costs 18 % and 4 mm too low costs nothing, so a deployment may
    # lower it a little as a margin against a plane fitted slightly low.
    self.height_min_m = float(height_min_m) if height_min_m is not None else pc_routes.crop_z_min(route)
    self.obs = CloudObs(rig, mode="depth" if base == "P0" else "cloud", num_points=num_points, device=device,
                        height_min_m=self.height_min_m)
    self.kin = kin
    self.grasp = GraspObs(kin, device=device) if base == "P2" else None
    self._lock = threading.Lock()
    self._stopping = threading.Event()
    self._joints = None
    self._out = None
    self._view = None
    self.frames = 0
    self.dropped = 0
    self.invalid_frames = 0
    self.periods = collections.deque(maxlen=2000)
    self.compute_ms = collections.deque(maxlen=2000)
    self.capture_to_publish_ms = collections.deque(maxlen=2000)
    self.counts = collections.deque(maxlen=2000)
    self.held_frames = 0
    self.flatten = False
    self.sam_state = None
    self._first_finished = None
    self._last_finished = None

  # -- what the loop hands in ---------------------------------------------

  def set_joints(self, q) -> None:
    with self._lock:
      self._joints = np.asarray(q, dtype=np.float64).copy()

  def set_contact(self, loaded: bool) -> None:
    del loaded  # the cloud policies carry no target, so a contact bit changes nothing here

  def latest(self):
    with self._lock:
      return self._out

  def view(self):
    with self._lock:
      return self._view

  # -- the thread ---------------------------------------------------------

  def run(self) -> None:
    try:
      self._run()
    except Exception as e:  # noqa: BLE001
      self.error = e
      raise

  def _run(self) -> None:
    last_index = -1
    while not self._stopping.is_set():
      frame = self.reader.latest()
      with self._lock:
        q = None if self._joints is None else self._joints.copy()
      if frame is None or frame.index == last_index:
        time.sleep(0.002)
        continue
      if last_index >= 0 and frame.index > last_index + 1:
        self.dropped += frame.index - last_index - 1
      last_index = frame.index
      t0 = time.perf_counter()
      depth = frame.depth
      if self.stereo is not None:
        if frame.ir is None or frame.ir_right is None:
          self.stereo_misses += 1
        else:
          depth = self.stereo(frame.ir, frame.ir_right)
      cf = self.obs(np.asarray(depth, dtype=np.float32))
      extra = None
      if self.grasp is not None:
        if q is not None:
          self.kin.update(q)
        self.grasp.update(cf)
        extra = (self.grasp.topk.copy(), self.grasp.locked.copy())
      finished = time.perf_counter()
      published_at = time.time()
      self.periods.append(finished - t0)
      self.compute_ms.append((finished - t0) * 1000.0)
      self.capture_to_publish_ms.append(max(0.0, (published_at - float(frame.stamp)) * 1000.0))
      self.counts.append(int(cf.count))
      self.frames += 1
      if not cf.valid:
        self.invalid_frames += 1
      if self._first_finished is None:
        self._first_finished = finished
      self._last_finished = finished
      frame.mask_state = f"pc:{self.route}"
      frame.detections = [{"workspace_points": int(cf.count)}]
      frame.pc_extra = extra
      with self._lock:
        self._out = (cf.obs, published_at, frame, 0, False, bool(cf.valid))
        self._view = (frame, None, 0, cf.obs, q)

  def close(self) -> None:
    self._stopping.set()
    self.join(timeout=2.0)

  # -- what the run report records ----------------------------------------

  def report(self) -> dict:
    def stats(xs):
      xs = list(xs)
      if not xs:
        return None
      a = np.asarray(xs, dtype=np.float64)
      return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "n": len(xs)}
    span = ((self._last_finished - self._first_finished) if (self._first_finished is not None and self._last_finished) else 0.0)
    rate = (self.frames - 1) / span if span > 0 and self.frames > 1 else None
    out = {
      "route": self.route, "height_min_m": self.height_min_m, "frames": self.frames, "dropped_by_mailbox": self.dropped,
      "invalid_frames": self.invalid_frames, "rate_hz": rate,
      "compute_ms": stats(self.compute_ms), "capture_to_publish_ms": stats(self.capture_to_publish_ms),
      "workspace_points": stats(self.counts), "stereo_misses": self.stereo_misses,
      "num_points": self.obs.num_points if self.obs.mode == "cloud" else None, "mode": self.obs.mode,
    }
    if self.grasp is not None:
      out["p2"] = {"switches": self.grasp.switches, "no_candidate_frames": self.grasp.no_candidate_frames,
                   "frames": self.grasp.frames}
    return out

  def summary(self) -> str:
    r = self.report()
    c = r["compute_ms"] or {}
    a = r["capture_to_publish_ms"] or {}
    return (f"perception ({self.route}): {r['frames']} frames"
            + (f" at {r['rate_hz']:.1f} Hz" if r["rate_hz"] else "")
            + f", compute {c.get('p50', float('nan')):.1f} ms median / {c.get('p95', float('nan')):.1f} ms p95, "
            f"capture-to-publish {a.get('p50', float('nan')):.1f} / {a.get('p95', float('nan')):.1f} ms, "
            f"{r['invalid_frames']} frame(s) with too few workspace points, {r['dropped_by_mailbox']} dropped"
            + (f", p2 switches {r['p2']['switches']} no-candidate {r['p2']['no_candidate_frames']}" if "p2" in r else ""))


class PcRunPolicy:
  """The bundle's ONNX graph behind the loop's ``pol(flat, camera, ...)`` call.

  Same explicit-hidden-state stepping as ``policy.Policy``; the inputs are
  matched by name, the flat input is the actor's 1D groups in order (proprio,
  [grasp_locked], vision_meta) and the nd input is the cloud, the depth pair
  or the top-K set.  ``reset()`` clears the memory exactly as the loop expects.
  """

  def __init__(self, bundle: pathlib.Path, route: str, threads: int = 2, cuda: bool = True) -> None:
    import onnxruntime as ort
    bundle = pathlib.Path(bundle)
    pc_routes.check_deployable(route)
    self.pc_route = route
    self.base_route = pc_routes.base_route(route)
    # "zero": the graph was trained with a fifth, always-zero column that the
    # robot's 4-column cloud is padded with here.  "oracle" never gets here.
    self.target_channel = pc_routes.target_channel(route)
    self.spec = json.loads((bundle / "obs_spec.json").read_text())
    self.path = str(bundle / "policy.onnx")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, int(threads))
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    self.sess = ort.InferenceSession(self.path, opts, providers=providers)
    inputs = self.sess.get_inputs()
    self.input_names = [i.name for i in inputs]
    self.nd_names = self.input_names[1:-1]
    fixed = lambda s: tuple(1 if not isinstance(d, int) else d for d in s)
    self.flat_width = int(fixed(inputs[0].shape)[-1])
    self.image_shapes = [fixed(i.shape)[1:] for i in inputs[1:-1]]
    self.hidden_shape = fixed(inputs[-1].shape)
    self.hidden = np.zeros(self.hidden_shape, dtype=np.float32)
    self.groups_1d = [g for g in self.spec["actor_groups"] if len(self.spec["groups"][g]["shape"]) == 1]
    want = sum(int(self.spec["groups"][g]["total"]) for g in self.groups_1d)
    if want != self.flat_width:
      raise RuntimeError(f"{self.path}: flat input is {self.flat_width} wide, obs_spec says {want}")
    self.nd_group = "grasp_topk" if self.base_route == "P2" else "camera"
    if self.nd_names != [self.nd_group]:
      raise RuntimeError(f"{self.path}: nd inputs {self.nd_names}, expected [{self.nd_group}] for route {route}")
    self.num_points = (int(self.spec["groups"]["camera"]["shape"][0]) if self.base_route not in ("P0", "P2")
                       else pc_cloud.POINT_DIM * 128)
    self.provider = self.sess.get_providers()[0]

  def blank(self) -> np.ndarray:
    return blank_observation(self.spec, self.pc_route)

  def reset(self) -> None:
    self.hidden = np.zeros(self.hidden_shape, dtype=np.float32)

  def __call__(self, proprio_vec: np.ndarray, camera: np.ndarray, age_s: float = 0.0, fresh: bool = False,
               valid: bool = False, extra=None) -> np.ndarray:
    meta = vision_meta(age_s, fresh, valid)
    parts = {"proprio": np.asarray(proprio_vec, dtype=np.float32).reshape(-1), "vision_meta": meta}
    if self.base_route == "P2":
      topk, locked = (extra if extra is not None
                      else (np.zeros(tuple(self.spec["groups"]["grasp_topk"]["shape"]), np.float32),
                            np.zeros(int(self.spec["groups"]["grasp_locked"]["total"]), np.float32)))
      parts["grasp_locked"] = np.asarray(locked, dtype=np.float32).reshape(-1)
      nd = np.asarray(topk, dtype=np.float32)
    else:
      nd = pad_target_channel(np.asarray(camera, dtype=np.float32), self.target_channel)
    if tuple(nd.shape) != tuple(self.image_shapes[0]):
      raise ValueError(f"the policy expects {self.image_shapes[0]} for {self.nd_group} and got {tuple(nd.shape)}")
    flat = np.concatenate([parts[g].reshape(-1) for g in self.groups_1d])
    if flat.size != self.flat_width:
      raise ValueError(f"flat observation is {flat.size} wide, the graph wants {self.flat_width}")
    feed = {self.input_names[0]: flat.reshape(1, -1), self.nd_group: nd[None], self.input_names[-1]: self.hidden}
    u, self.hidden = self.sess.run(None, feed)
    return np.asarray(u, dtype=np.float64).reshape(-1)


def blank_observation(spec: dict, route: str) -> np.ndarray:
  """The zero observation of the policy's nd group, from the bundle's spec."""
  group = "grasp_topk" if pc_routes.base_route(route) == "P2" else "camera"
  return np.zeros(tuple(int(x) for x in spec["groups"][group]["shape"]), np.float32)


def pad_target_channel(cloud_obs: np.ndarray, target_channel: str) -> np.ndarray:
  """A 4-column robot cloud padded with the always-zero fifth column of a ``zero`` route."""
  if target_channel == "zero" and cloud_obs.ndim == 2 and cloud_obs.shape[-1] == pc_cloud.POINT_DIM:
    return np.concatenate([cloud_obs, np.zeros((cloud_obs.shape[0], 1), np.float32)], axis=-1)
  if target_channel == "oracle":
    raise RuntimeError("an oracle-only route cannot be fed from the robot")
  return cloud_obs


def vision_meta(age_s: float, fresh: bool, valid: bool) -> np.ndarray:
  steps = age_s * pc_cloud.CONTROL_HZ
  return np.array([min(steps, pc_cloud.AGE_NORM) / pc_cloud.AGE_NORM, float(fresh), float(valid)], dtype=np.float32)
