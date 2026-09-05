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
import queue                                               # noqa: E402
import re                                                  # noqa: E402
import signal                                              # noqa: E402
import threading                                           # noqa: E402
import sys                                                 # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

from . import config, lifecycle, mask, obs, proprio, rectify, robot  # noqa: E402
from . import target_mask                                  # noqa: E402
from piper_push import robot as sim_robot                  # noqa: E402


def _feedback_fault(st, speed_fraction: float = 1.0) -> str | None:
  """Return why measured arm feedback is unsafe to act on, if anything."""
  q = np.asarray(st.q, dtype=np.float64).reshape(-1)
  dq = np.asarray(st.dq, dtype=np.float64).reshape(-1)
  values = np.concatenate([q, dq, [st.gripper, st.gripper_vel]])
  if not np.isfinite(values).all():
    return "non-finite joint or gripper feedback"
  limits = np.asarray([
    sim_robot.JOINT_TRIP_RAD_S[j] for j in robot.ARM_JOINTS
  ], dtype=np.float64) * float(speed_fraction)
  bad = np.flatnonzero(np.abs(dq) > limits)
  if bad.size:
    i = int(bad[0])
    return (f"joint {i + 1} speed {dq[i]:+.3f} rad/s exceeds "
            f"{limits[i]:.3f} rad/s")
  return None


def _action_fault(action) -> str | None:
  """Reject malformed policy output before it reaches the command mapper."""
  a = np.asarray(action, dtype=np.float64).reshape(-1)
  if a.size != 7:
    return f"policy returned {a.size} actions, expected 7"
  if not np.isfinite(a).all():
    return "policy returned a non-finite action"
  return None


def _grasp_height(kin: proprio.Kinematics, target) -> float:
  """Grasp-site height for a seven-target command, in the robot base frame."""
  q = np.asarray(target, dtype=np.float64).reshape(-1)
  if q.size != 7 or not np.isfinite(q).all():
    return float("nan")
  kin.update(np.concatenate([q[:6], [q[6], -q[6]]]))
  return float(kin.site_pos[2])


def _table_clearance(kin: proprio.Kinematics, target, normal,
                     table_z: float) -> tuple[float, str]:
  """Moving robot collision geometry clearance above the measured table."""
  q = np.asarray(target, dtype=np.float64).reshape(-1)
  if q.size != 7 or not np.isfinite(q).all():
    return float("nan"), ""
  kin.update(np.concatenate([q[:6], [q[6], -q[6]]]))
  return kin.collision_plane_clearance(
    np.asarray(normal, dtype=np.float64), float(table_z))


def home_arm(arm, spec: dict, speed_rad_s: float = None,
             should_stop=None):
  """Drive to the pose every training episode began at, and return the result.

  The policy is recurrent and was never shown any other opening: each episode
  started at this keyframe with a freshly placed object.  A run that begins
  wherever the previous one stopped hands the GRU a state it has no prior for,
  and the arm carries whatever pose the last failure left it in.

  The target comes from the observation spec rather than from a constant here.
  That spec is exported from the trained task, and it is already what carries
  joint 1's +90 degree layout rotation -- a second copy of the home pose would
  be a second place for that rotation to be missing from.

  The path, the speed and the tracking check are the guided calibration's,
  which is the only motion code on this rig that has moved the arm across the
  workspace without incident.
  """
  names = list(spec["joint_names"])
  default = spec["default_joint_pos"]
  home = np.asarray([default[names.index(j)] for j in robot.ARM_JOINTS],
                    dtype=np.float64)
  grip = float(default[names.index(robot.GRIPPER_JOINT)])
  speed = robot.AUTO_SPEED_RAD_S if speed_rad_s is None else float(speed_rad_s)

  st = arm.read()
  print("homing to the training start pose "
        + " ".join(f"{x:+.1f}" for x in np.degrees(home)) + " deg")
  path = robot.joint_trajectory(st.q, home, speed_rad_s=speed)
  period = 1.0 / robot.AUTO_RATE_HZ
  deadline = time.monotonic()
  for i, q in enumerate(path):
    if should_stop is not None and should_stop():
      arm.hold()
      raise SystemExit("interrupted while homing")
    here = arm.read()
    # The calibration's own check: after half a second the servo has had time
    # to pick the motion up, so a large error is a stalled joint or a lost CAN
    # path rather than a slow one.
    err = float(np.max(np.abs(np.asarray(here.q) - q)))
    if i > robot.AUTO_RATE_HZ * 0.5 and err > robot.AUTO_TRACKING_ERROR_RAD:
      arm.hold()
      raise RuntimeError(
        f"homing tracking error {np.degrees(err):.1f} deg exceeds "
        f"{np.degrees(robot.AUTO_TRACKING_ERROR_RAD):.1f} deg at step {i} of "
        f"{len(path)}")
    arm.command(np.array([*q, grip]), period)
    deadline += period
    time.sleep(max(0.0, deadline - time.monotonic()))
  st = arm.read()
  print("homed; joints now "
        + " ".join(f"{x:+.1f}" for x in np.degrees(st.q)) + " deg")
  return st


def _frame_index(frame) -> "int | None":
  """The frame's index, or None when the loop has not had one.

  ``frame`` is None on a dry run, and for the first steps of any run before the
  camera has delivered.  Reaching through it for the event log crashed the
  whole loop -- after the arm was already held, so the data was safe and the
  exit code was not.
  """
  return None if frame is None else int(frame.index)


def _frame_stamp(frame) -> "float | None":
  return None if frame is None else float(frame.stamp)

def _observation_age(frame, now: float | None = None) -> float:
  """Age since camera capture, including every perception stage.

  The old guard measured from the time perception *finished*.  A 500 ms GPU
  stall therefore produced a freshly published, 500 ms-old observation that
  was allowed to move the arm.  The frame timestamp is the only timestamp that
  bounds what the policy is actually seeing.
  """
  stamp = _frame_stamp(frame)
  if stamp is None:
    return float("inf")
  return float((time.time() if now is None else now) - stamp)

def _timing_stats(values) -> dict | None:
  """JSON-safe distribution used by both live-loop and perception reports."""
  a = np.asarray(values, dtype=np.float64)
  if not a.size:
    return None
  return {
    "mean": float(a.mean()),
    "p50": float(np.percentile(a, 50)),
    "p90": float(np.percentile(a, 90)),
    "p95": float(np.percentile(a, 95)),
    "p99": float(np.percentile(a, 99)),
    "max": float(a.max()),
    "n": int(a.size),
  }



def _wrap_control(arm, a):
  """Put the arm on the command path ``--control`` asks for.

  Separate from ``build`` so the choice can be tested without a CAN bus: it
  decides how every joint command for the rest of the session reaches the
  drives, and a source grep is not a test of it.

  ``mit`` is the drives' fast-response law on the ordinary position path, which
  is what the SDK's ``piper_set_mit.py`` calls MIT mode.  It is not
  ``impedance`` -- that is ``JointMitCtrl``, which this arm's firmware accepts
  and ignores; see ``mit.MitDriver``.
  """
  mode = getattr(a, "control", "movej")
  if mode == "movej":
    print("control: MOVE J, plain response (--control movej)")
    return arm
  if mode == "mit":
    arm.response_mode = robot.RESPONSE_MIT
    print("control: MOVE J, MIT fast response (0xAD)")
    return arm

  from . import mit as mit_mod
  gains_file = pathlib.Path(getattr(a, "mit_gains", None) or mit_mod.GAINS_FILE)
  if not gains_file.exists():
    try:
      arm.disconnect()
    except Exception:
      pass
    raise RuntimeError(
      f"{gains_file} does not exist, so there is no measured impedance to "
      "write.  Run `python -m hardware.deploy.sysid plan` and work through "
      "it.  NOTE: on firmware S-V1.8-9 this rig ignores JointMitCtrl "
      "entirely -- --control impedance has never moved this arm.  Use "
      "--control mit.")
  gravity = float(getattr(a, "mit_gravity", 1.0))
  substeps = int(getattr(a, "mit_substeps", 4))
  wrapped = mit_mod.MitArm(
    arm, mit_mod.Gains.load(gains_file),
    gravity=(None if gravity == 0.0 else mit_mod.GravityFeedforward()),
    substeps=substeps, gravity_scale=gravity,
    torque_limit=float(getattr(a, "mit_torque_limit", 3.0)))
  print(f"control: per-joint impedance, gains from {gains_file}, {substeps} "
        f"frames per control step, gravity x{gravity:g}")
  return wrapped


def _wait_for_feedback(arm, timeout_s: float = 3.0, settle: int = 3):
  """Block until the drives have actually reported, and prove that they did.

  ``ConnectPort`` starts the CAN receivers asynchronously and the SDK's message
  objects exist before any frame has arrived, so the first ``read()`` returns
  their defaults: every joint exactly 0.000000 and the gripper shut.  Measured
  on this rig, the real values land about 0.3 s later.

  That matters because of what happens next.  The measured pose is preloaded
  into the drives and then the arm is enabled -- which is the right order, and
  is what stops enable chasing a stale target from a previous session.  Preload
  the *defaults* instead and the same mechanism drives the arm to its zero pose
  the instant it is energised, from wherever it is standing.  Nothing
  downstream catches it: all-zero joints sit comfortably inside
  ``SAFE_TARGET_CLIP``, so ``_start_pose_fault`` passes them.

  There used to be a fixed ``sleep(0.3)`` here, sized at exactly the delay that
  was observed.  A fixed wait cannot tell a slow bus from a silent one, so this
  waits for evidence instead: several consecutive reads that agree, and that
  are not the all-zero signature.  A real encoder does not report six exact
  zeros; the SDK before its first frame reports nothing else.
  """
  deadline = time.time() + timeout_s
  previous, agreed = None, 0
  while time.time() < deadline:
    st = arm.read()
    q = np.asarray(st.q, dtype=np.float64)
    if not np.isfinite(q).all():
      previous, agreed = None, 0
    elif np.all(q == 0.0) and float(st.gripper) == 0.0:
      # Not yet: the defaults, exactly.
      previous, agreed = None, 0
    elif previous is not None and np.allclose(q, previous, atol=1e-9):
      agreed += 1
      if agreed >= settle:
        return st
    else:
      previous, agreed = q, 0
      continue
    time.sleep(0.02)
  raise RuntimeError(
    f"the drives did not report a settled pose within {timeout_s:.1f} s on "
    f"{getattr(arm, 'can', 'the CAN interface')}.  Check `ip -details link "
    "show can0` says UP and ERROR-ACTIVE, and that the arm is powered -- "
    "refusing to preload and enable against unknown feedback.")


def _start_pose_fault(st, tolerance_rad: float = np.deg2rad(1.0)) -> str | None:
  """Refuse enable if CAN clipping could move a joint by more than one degree."""
  q = np.asarray(st.q, dtype=np.float64).reshape(-1)
  lo = np.asarray([sim_robot.SAFE_TARGET_CLIP[j][0]
                   for j in robot.ARM_JOINTS], dtype=np.float64)
  hi = np.asarray([sim_robot.SAFE_TARGET_CLIP[j][1]
                   for j in robot.ARM_JOINTS], dtype=np.float64)
  bad = np.flatnonzero((q < lo - tolerance_rad) | (q > hi + tolerance_rad))
  if bad.size:
    i = int(bad[0])
    return (f"joint {i + 1} starts at {np.degrees(q[i]):+.2f} deg outside "
            f"the deployable [{np.degrees(lo[i]):+.2f}, "
            f"{np.degrees(hi[i]):+.2f}] deg target range; preloading it would "
            "be clipped and enable could jump")
  return None


class Rates:
  """What the loop actually achieved, which is not what it was asked for."""

  def __init__(self):
    self.steps = 0
    self.overruns = 0
    self.worst_ms = 0.0
    self.stale = 0
    self.no_target = 0
    self.blind = 0
    self.guard_holds = 0
    self._t = []
    self._observation_age_ms = []
    self._publish_age_ms = []

  def note(self, ms: float, budget_ms: float) -> None:
    self.steps += 1
    self.worst_ms = max(self.worst_ms, ms)
    self._t.append(ms)
    if ms > budget_ms:
      self.overruns += 1


  def note_observation(self, observation_age_s: float,
                       publish_age_s: float) -> None:
    self._observation_age_ms.append(max(0.0, observation_age_s * 1000.0))
    self._publish_age_ms.append(max(0.0, publish_age_s * 1000.0))

  def report(self) -> dict:
    compute = _timing_stats(self._t)
    return {
      "steps": self.steps,
      "overruns": self.overruns,
      "stale": self.stale,
      "no_target": self.no_target,
      "blind": self.blind,
      "guard_holds": self.guard_holds,
      "median_ms": None if compute is None else compute["p50"],
      "p95_ms": None if compute is None else compute["p95"],
      "worst_ms": self.worst_ms,
      "control_compute_ms": compute,
      "observation_age_ms": _timing_stats(self._observation_age_ms),
      "perception_publish_age_ms": _timing_stats(self._publish_age_ms),
    }

  def summary(self) -> str:
    t = np.asarray(self._t) if self._t else np.zeros(1)
    return (f"{self.steps} steps, median {np.median(t):.1f} ms, "
            f"p95 {np.percentile(t, 95):.1f} ms, worst {self.worst_ms:.1f} ms, "
            f"{self.overruns} overrun(s), {self.stale} stale frame(s), "
            f"{self.no_target} step(s) with nothing to fetch"
            + (f" ({self.blind} on a held-over mask)" if self.blind else "")
            + (f", {self.guard_holds} guard hold(s)" if self.guard_holds
               else ""))


def build(a):
  """Everything the loop needs, and a clear error for whatever is missing."""
  spec = json.loads(pathlib.Path(proprio.SPEC_FILE).read_text())
  # The action mapping comes from the POLICY's exported spec when it has one:
  # a bounded policy driven through the repository's legacy spec would be a
  # different robot.  proprio keeps reading the repository spec for the
  # observation layout, which _check_obs_spec asserts is the same file.
  policy_spec_path = (pathlib.Path(a.policy) if getattr(a, "policy", None) else None)
  if policy_spec_path is not None:
    policy_spec_path = (policy_spec_path if policy_spec_path.is_dir() else policy_spec_path.parent) / "obs_spec.json"
  action_spec_source = spec
  if policy_spec_path is not None and policy_spec_path.exists():
    action_spec_source = json.loads(policy_spec_path.read_text())
  camera = str(getattr(a, "camera", "d405"))
  suffix = "" if camera == "d405" else f"_{camera}"
  rig_path = (pathlib.Path(a.rig_file) if getattr(a, "rig_file", None)
              else pathlib.Path(config.RIG_FILE).with_name(f"rig{suffix}.json"))

  if rig_path.exists():
    rig = config.Rig.load(rig_path)
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
      f"no {rig_path}.  Run `python -m hardware.deploy.calibgui "
      "--collect` then `--solve`, or pass --allow-nominal to run against the "
      "simulator's assumed camera pose and accept that it is not measured."
    )

  if (getattr(a, "min_table_clearance", None) is not None
      and rig.table_normal_base is None):
    raise SystemExit(
      f"{rig_path} has no table_normal_base; refusing calibrated-plane safety")

  reader = None
  if a.replay:
    reader = _Replay(a.replay)
    print(f"replaying {reader.n} frames from {a.replay}")
  elif not a.dry_run:
    from . import sensor
    reader = sensor.Reader(serial=a.serial, backend=camera,
                           stereo=(a.depth_source == "stereo"))
    reader.wait_for_first()
    if (rig.serial and reader.serial
        and str(rig.serial) != str(reader.serial)):
      reader.close()
      raise SystemExit(
        f"{rig_path} belongs to camera {rig.serial}, connected {camera.upper()} "
        f"is {reader.serial}; refusing to mix calibration and images")
    rig.K = reader.K              # the camera's own, not the stored one
    rig.serial = reader.serial

  reproj = rectify.Reprojector(rig, device=a.device)
  segmenter = _segmenter(a, rig, reproj)
  tracker = mask.TargetTracker()
  builder = proprio.ProprioBuilder()
  mapper = robot.ActionMapper(
    action_spec_source, dt=1.0 / config.CONTROL_HZ,
    accel_limit=getattr(a, "command_accel_limit", None),
    gripper_accel_limit=getattr(a, "gripper_accel_limit", None),
    allow_legacy=bool(getattr(a, "allow_legacy_action_api", False)),
  )
  print(f"action convention: {mapper.convention} ({mapper.action_api_status}), "
        f"squashed={mapper.squashed}")
  mapper.max_step *= float(getattr(a, "command_rate_scale", 1.0))

  arm = (robot.DryRunArm(spec) if (a.dry_run or a.no_arm or a.replay)
         else robot.PiperArm(a.can))
  arm.connect()
  real_arm = isinstance(arm, robot.PiperArm)
  if isinstance(arm, robot.PiperArm):
    start = _wait_for_feedback(arm)
  else:
    start = arm.read()
  # PiPER retains its last position target across client restarts.  Preload the
  # measured pose before enable, exactly as calibgui's already hardware-tested
  # motion worker does, or enable can jump towards a stale target immediately.
  start_fault = _start_pose_fault(start)
  if start_fault is not None:
    arm.disconnect()
    raise RuntimeError(start_fault)
  preload = np.concatenate([start.q, [start.gripper]])
  arm.command(preload)
  arm.enable()
  arm.command(preload)

  # The impedance path goes on AFTER enable and after the preload, both of
  # which are position-mode operations that exist to stop the drives chasing a
  # stale target.  Entering MIT before them would hand the arm an impedance
  # about a pose it has not been told to hold yet.
  if real_arm:
    arm = _wrap_control(arm, a)

  _check_policy_matches_spec(a.policy, spec)

  from .policy import Policy
  pol = Policy(a.policy, threads=a.policy_threads,
               providers=(["CUDAExecutionProvider", "CPUExecutionProvider"]
                          if a.policy_device == "cuda"
                          else ["CPUExecutionProvider"]))
  print(f"policy: {pol.path}, {pol.flat_width} + "
        f"{'x'.join(map(str, pol.image_shapes[0]))} in, hidden "
        f"{pol.hidden_shape}")
  return rig, reader, reproj, segmenter, tracker, builder, mapper, arm, pol


def _init_joint_pos(env_yaml: str, names: list[str]) -> np.ndarray | None:
  """The initial joint angles out of an mjlab ``env.yaml``, or None.

  Not a YAML parse: the dump carries ``!!python/object/apply`` tags and would
  need ``unsafe_load``.  Not a bare search for ``jointN:`` either -- the same
  file lists the per-joint *action scales* the same way, and a regex over the
  whole document reads 0.3 and 1.6 out of those and reports two joints as
  disagreeing when only one does.  So: find the ``joint_pos`` mapping whose
  keys are all literal joint names, and read only inside it.  Joints the block
  omits are zero, which is what the scene config means by leaving them out.
  """
  for m in re.finditer(r"^(\s+)joint_pos:\s*$", env_yaml, re.MULTILINE):
    indent = len(m.group(1))
    block = {}
    literal = True
    for line in env_yaml[m.end():].split("\n")[1:]:
      if not line.strip():
        continue
      pad = len(line) - len(line.lstrip())
      if pad <= indent:
        break
      key, _, value = line.strip().partition(":")
      try:
        block[key] = float(value.strip())
      except ValueError:
        literal = False
        break
      if key not in names:
        literal = False
    if not literal or not block:
      continue
    return np.asarray([block.get(n, 0.0) for n in names], dtype=np.float64)
  return None


def _check_policy_matches_spec(policy_path, spec: dict) -> None:
  """Was this policy trained against the observation spec on disk?

  ``proprio.py`` refuses to run when a term is *missing* from obs_spec.json.
  Nothing refused when a value in it went stale, and one did: after the +90
  degree workspace rotation, ``joint1``'s default sat at 0.06 rad in the file
  and 1.6308 in the task.  That number is subtracted from the measured angle
  to make the ``joint_pos`` observation and added to the action to make the
  joint target, so the policy was told joint 1 was 90 degrees further round
  than it was and commanded 90 degrees short of where it meant -- in both
  directions at once, inside ``SAFE_TARGET_CLIP``, with nothing raising.

  The rig file already cross-checks its serial against the connected camera
  for the same reason.  This is that check for the policy: an exported
  directory carries what it was built from, and the two have to agree.
  """
  path = pathlib.Path(policy_path)
  directory = path if path.is_dir() else path.parent
  disk = np.asarray(spec["default_joint_pos"], dtype=np.float64)
  names = list(spec["joint_names"])

  beside = directory / "obs_spec.json"
  if beside.exists():
    other = json.loads(beside.read_text())
    theirs = np.asarray(other["default_joint_pos"], dtype=np.float64)
    if list(other["joint_names"]) != names or theirs.shape != disk.shape:
      raise RuntimeError(
        f"{beside} describes different joints from {proprio.SPEC_FILE}")
    bad = np.flatnonzero(np.abs(theirs - disk) > 1e-4)
    if bad.size:
      detail = ", ".join(
        f"{names[i]} {disk[i]:+.4f} on disk vs {theirs[i]:+.4f} exported"
        for i in bad)
      raise RuntimeError(
        f"{proprio.SPEC_FILE} does not match {beside}: {detail}.  Re-run "
        "scripts/export_obs_spec.py on the task this policy was trained for; "
        "do not deploy across this difference.")
    print(f"obs spec: matches {directory.name}/obs_spec.json")
    return

  env_yaml = directory / "env.yaml"
  if env_yaml.exists():
    trained = _init_joint_pos(env_yaml.read_text(), names)
    if trained is None:
      print(f"obs spec: {env_yaml.name} states its initial joints as patterns "
            "rather than names; not comparing rather than guessing.")
      return
    bad = np.flatnonzero(np.abs(trained - disk) > 1e-3)
    if bad.size:
      raise RuntimeError(
        f"{proprio.SPEC_FILE} disagrees with {env_yaml}: "
        + "; ".join(f"{names[i]} {disk[i]:+.4f} on disk vs {trained[i]:+.4f} "
                    "in the policy's env.yaml" for i in bad)
        + ".  That value is subtracted from the observation and added to the "
          "command, so deploying across it points the arm somewhere else.  "
          "Re-run scripts/export_obs_spec.py on the policy's own task.")
    print(f"obs spec: consistent with {directory.name}/env.yaml")
    return

  print(f"obs spec: {directory.name} carries neither obs_spec.json nor "
        "env.yaml, so nothing here can confirm the policy was trained "
        "against the spec being used.  scripts/bringup_d455.sh writes one.")


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
  p.add_argument("--mask", choices=("depth", "yolo", "fused"), default=None,
                 help="'depth' needs no model and is the accurate one wherever "
                      "there is depth to segment.  'yolo' reads the mono image "
                      "instead and works where there is not.  'fused' runs the "
                      "first and lets the second add what it missed, which is "
                      "what should be on the robot -- at one forward pass a "
                      "frame.  Defaults to 'depth' -- see below for why it "
                      "is not 'fused'.")
  p.add_argument("--yolo-weights", default=None,
                 help="``.pt`` goes through ultralytics and ``.onnx`` through "
                      "onnxruntime, which drops the torch dependency; see "
                      "yolo_backend.py for what each costs.  Defaults to the "
                      "model trained for --camera; passing one trained for the "
                      "other camera is refused rather than silently used")
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
  p.add_argument("--control", choices=("mit", "movej", "impedance"),
                 default="mit",
                 help="how joint targets reach the drives.  'mit' is the "
                      "drives' fast-response law on the position path "
                      "(MotionCtrl_2's 0xAD), measured on this rig at 3.1x "
                      "less p95 tracking lag than 'movej'.  'impedance' is "
                      "per-joint JointMitCtrl, which firmware S-V1.8-9 "
                      "accepts and ignores -- it has never moved this arm")
  p.add_argument("--mit-gains", help="mit_gains.json; default is the one "
                                     "beside this package")
  p.add_argument("--mit-substeps", type=int, default=4,
                 help="MIT frames per control step.  The simulator "
                      "interpolates its target across the physics substeps; "
                      "one frame per step is the 50 Hz staircase that "
                      "interpolation exists to avoid")
  p.add_argument("--mit-gravity", type=float, default=1.0,
                 help="scale on the model's qfrc_bias feedforward, 0 to send "
                      "none")
  p.add_argument("--mit-torque-limit", type=float, default=3.0,
                 help="N.m per joint on the feedforward, under the CAN "
                      "field's own 8")
  p.add_argument("--camera", choices=("d405", "d455"), default="d455",
                 help="camera backend; also selects rig_<camera>.json")
  p.add_argument("--rig-file", default=None,
                 help="explicit calibration file; defaults to rig.json for "
                      "D405 and rig_<camera>.json otherwise")
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--max-obs-age", type=float, default=0.20,
                 help="seconds since the vision thread last published.  The "
                      "camera runs at 30 Hz and the vision at 15-30, so "
                      "anything past 200 ms is a fault, not jitter")
  p.add_argument("--max-overrun-streak", type=int, default=25)
  p.add_argument("--max-joint-speed-fraction", type=float, default=1.0,
                 help="fraction of the trained hardware trip speed at which "
                      "feedback immediately stops the run (0, 1]")
  p.add_argument("--command-rate-scale", type=float, default=1.0,
                 help="multiply the trained per-step command slew limits; use "
                      "0.25 for initial hardware motion tests")
  p.add_argument("--command-accel-limit", type=float, default=None,
                 help="optional arm command acceleration ceiling in rad/s^2")
  p.add_argument("--gripper-accel-limit", type=float, default=None,
                 help="optional gripper command acceleration ceiling in m/s^2")
  p.add_argument("--min-grasp-height", type=float, default=None,
                 help="hard base-frame floor for the MEASURED grasp site, "
                      "metres.  It used to apply to the commanded setpoint as "
                      "well; that stopped three healthy runs with the arm 5 to "
                      "9 cm up, because the setpoint leads the arm and sits "
                      "below the table 44.7%% of the time in normal operation. "
                      "The deployed policy takes the measured site no lower "
                      "than 13 mm, so 0.005 leaves room while still catching a "
                      "real dive; 0.065 is the conservative first-test value")
  p.add_argument("--min-table-clearance", type=float, default=None,
                 help="optional calibrated-plane geometry stop in metres; "
                      "omitting it (the default) permits light fingertip/table "
                      "contact and avoids treating an unmeasured contact event "
                      "as a deployment failure")
  p.add_argument("--depth-bias", type=float, default=0.0,
                 help="metres subtracted from every depth reading before the "
                      "segmenter, the reprojection or the policy see it.  "
                      "Positive pulls the scene towards the camera.  Two "
                      "independent measurements on 2026-09-01 put the "
                      "deployment's object about 19 mm further along the "
                      "approach than it is -- graspcheck read z = -18.9 mm in "
                      "the gripper frame, and the same policy in simulation "
                      "never takes the grasp site below 13.2 mm where the arm "
                      "reaches -41 mm -- so 0.019 is the measured value and 0 "
                      "(the default) leaves the pipeline as it was")
  p.add_argument("--record-queue", type=int, default=512,
                 help="camera frames the log writer may fall behind by before "
                      "the run stops rather than continue unlogged")
  p.add_argument("--no-record-compress", action="store_true",
                 help="store frames uncompressed.  zlib on the writer thread "
                      "is what ended three of today's runs, and it ends the "
                      "good ones first: while the policy holds, the writer has "
                      "the CPU and keeps up; once the policy is working, "
                      "inference takes the cores and the queue fills.  Costs "
                      "roughly 4x the disk and almost no CPU")
  p.add_argument("--target-lifecycle", action="store_true",
                 help="own the target's identity instead of letting the "
                      "segmenter re-choose it.  Locks the instance on first "
                      "sight, refuses a different one during the approach, "
                      "and during a carry refuses the table entirely and "
                      "rebuilds the mask at the grasp site.  Replaying the 20 "
                      "recorded sessions through it turns 326 identity swaps "
                      "while holding into 0.  Requires --held-target-radius.")
  p.add_argument("--held-target-radius", type=float, default=0.0,
                 help="metres.  Once the jaws are closed on the object the "
                      "camera cannot call it an object any more -- from a "
                      "fixed viewpoint it is inside the arm -- and the last "
                      "mask points at the table it was lifted from, so the "
                      "loop runs out of target and holds.  Measured on "
                      "v4_fixedseg_try6: grasped at 13 s, then one pose from "
                      "15.7 s to 45 s.  With this set, an empty mask while "
                      "holding is rebuilt from the depth that arrived, at the "
                      "grasp site, which is where the robot knows the object "
                      "is because it is holding it.  0.045 covers the largest "
                      "object; 0 (the default) keeps the old behaviour")
  p.add_argument("--target-tracker", default="depth",
                 choices=("depth", "sam21"),
                 help="who carries the target between frames.  'depth' is the "
                      "segmenter and TargetTracker alone, which is what every "
                      "run so far used.  'sam21' adds SAM2.1 as a causal "
                      "tracker on top: the depth stack still CHOOSES the "
                      "instance and SAM only carries that choice, because SAM "
                      "returns a confident mask of the wrong object 13%% of "
                      "the time on TwinSight's reviewed frames and the policy "
                      "has no way to tell that from a correct one.  Scored "
                      "against the renderer over seven seeds, it takes the "
                      "target from 47%% of approach frames to 98%% and from 0%% "
                      "to 100%% while held -- but that was textured RGB in "
                      "simulation and the rig feeds it grayscale IR, so the "
                      "rig numbers are not those numbers.  Costs ~28 ms a "
                      "frame on a 5090; watch the reported perception rate.")
  p.add_argument("--sam-checkpoint", default=None,
                 help="override the pinned sam2.1_hiera_small.pt")
  p.add_argument("--sam-vos-optimized", action="store_true",
                 help="reserved for SAM VOS compilation; currently refused "
                      "because the installed torch 2.13/SAM2.1 combination "
                      "fails its first propagated frame")
  p.add_argument("--allow-legacy-action-api", action="store_true",
                 help="drive the arm from an obs_spec.json that predates the action "
                      "convention stamp (unbounded v1, every export before 2026-09-05). "
                      "Refused otherwise.")
  p.add_argument("--gripper-closed", type=float, default=0.045,
                 help="metres of single-finger travel below which the jaws "
                      "count as closed on something, for --held-target-radius")
  p.add_argument("--max-blind-steps", type=int, default=60,
                 help="control steps the policy may keep driving after the "
                      "mask empties, while the tracker still believes the "
                      "target is there.  The simulator's mask empties the same "
                      "way when the arm passes in front of the object and the "
                      "policy acts through it; holding instead deadlocks, "
                      "because a held pose cannot uncover what it is covering. "
                      "0 restores the old hold-immediately behaviour.")
  p.add_argument("--guard-mode", choices=("stop", "hold"), default="stop",
                 help="applies to the COMMAND guards only -- the predicted "
                      "table clearance.  The two guards that test the arm's "
                      "own measured pose always stop, and holding would be "
                      "wrong for them: refusing a command while the arm is "
                      "already past the floor keeps it there, which is the "
                      "deadlock, not the fix.  Set the floor where the data "
                      "says instead.  What this does: "
                      "would breach it.  'stop' ends the session, which is "
                      "right for a first motion and wrong for measuring "
                      "anything: a policy that dives at the table produces "
                      "fifteen steps and no statistics.  'hold' refuses that "
                      "one command, holds the measured pose, counts it and "
                      "carries on -- the same thing a stale frame already "
                      "does, so it introduces no new kind of command.  A "
                      "guard already breached by the MEASURED pose still "
                      "stops in both modes; holding there would only keep the "
                      "arm where it should not be.")
  p.add_argument("--max-guard-hold-streak", type=int, default=100,
                 help="consecutive refused commands before --guard-mode hold "
                      "gives up.  The default is two seconds: past that the "
                      "policy is leaning on the floor and nothing is being "
                      "learned by watching it")
  p.add_argument("--startup-hold", type=float, default=3.0,
                 help="seconds to hold the measured starting pose after "
                      "enable and before the first policy command")
  p.add_argument("--depth-source", choices=("sensor", "stereo"),
                 default="sensor",
                 help="where depth comes from.  'sensor' is the camera's own "
                      "map and is what every measurement in this repository "
                      "describes.  'stereo' runs Fast-FoundationStereo on the "
                      "raw imagers instead: 14 ms of GPU per frame, and on "
                      "this unit it gave 73.2%% fill against the camera's "
                      "88.6%% while agreeing to 3.7 mm where both are "
                      "defined.  It is an option because its failures are in "
                      "different places, not because it is better.")
  p.add_argument("--stereo-engine", default=None,
                 help="TensorRT engine for --depth-source stereo; defaults to "
                      "the 4-iteration Fast-FoundationStereo engine in the "
                      "depth bench")
  p.add_argument("--replay", default=None,
                 help="a recorded session to feed instead of the camera.  The "
                      "arm still runs (dry, unless --no-arm is off), so this "
                      "is what a rollout looked like from the policy's side, "
                      "and the only way to measure the loop's real cost "
                      "without hardware")
  p.add_argument("--view", action="store_true",
                 help="open a window showing the depth, the segmentation and "
                      "the three channels the policy is given.  Drawn on its "
                      "own thread from whatever perception last produced, so "
                      "it costs the control loop nothing and is allowed to "
                      "fail without stopping the run.")
  p.add_argument("--view-hz", type=float, default=10.0)
  p.add_argument("--view-scale", type=float, default=1.0)
  p.add_argument("--flatten-scene", action="store_true",
                 help="replace everything that is not the tabletop or standing "
                      "on it with the calibrated plane, which is all the "
                      "simulator's scene contains -- one infinite MuJoCo "
                      "PLANE.  Injecting a raw deployment channel 0 into the "
                      "simulator takes the trained policy to zero objects "
                      "placed; flattening it first recovers some of that "
                      "against a frozen-frame control, and the honest test is "
                      "the arm.")
  p.add_argument("--home-first", action="store_true",
                 help="drive to the training start pose before handing over to "
                      "the policy.  Every episode the policy was trained on "
                      "began at that pose with a freshly placed object; a run "
                      "that starts wherever the last one stopped hands its GRU "
                      "an opening it never saw.  Uses the calibration GUI's "
                      "own rest-to-rest path, speed and tracking check.")
  p.add_argument("--home-speed", type=float, default=robot.AUTO_SPEED_RAD_S,
                 help="peak joint speed for that move, rad/s")
  p.add_argument("--log-root", default="logs/deploy",
                 help="where a session goes when --record is not given.  Each "
                      "run gets its own timestamped directory, so evidence "
                      "from two runs can never land in one folder and be read "
                      "as one run a week later.")
  p.add_argument("--review-stride", type=int, default=12,
                 help="frames between pictures in the review page written on "
                      "exit.  12 is about one picture a second and 45 MB for a "
                      "minute; lower it to see more and watch the size.  For "
                      "detail, leave this alone and re-render a window "
                      "afterwards: python -m hardware.deploy.review SESSION "
                      "--stride 1 --window 20 30 --html detail.html")
  p.add_argument("--no-review", action="store_true",
                 help="skip the review page.  It is written on exit by "
                      "default: re-segmenting the frames afterwards takes a "
                      "few seconds and it is the difference between a session "
                      "somebody looks at and one nobody does.")
  p.add_argument("--record", default=None,
                 help="directory to write the session to, for replay and for "
                      "labelling a YOLO training set")
  a = p.parse_args()

  if not (0.0 < a.max_joint_speed_fraction <= 1.0):
    p.error("--max-joint-speed-fraction must be in (0, 1]")
  if not (0.0 < a.command_rate_scale <= 1.0):
    p.error("--command-rate-scale must be in (0, 1]")
  if a.command_accel_limit is not None and a.command_accel_limit <= 0.0:
    p.error("--command-accel-limit must be positive")
  if a.gripper_accel_limit is not None and a.gripper_accel_limit <= 0.0:
    p.error("--gripper-accel-limit must be positive")
  if a.min_grasp_height is not None and not np.isfinite(a.min_grasp_height):
    p.error("--min-grasp-height must be finite")
  if (a.min_table_clearance is not None
      and (not np.isfinite(a.min_table_clearance)
           or a.min_table_clearance < 0.0)):
    p.error("--min-table-clearance must be finite and non-negative")
  if a.startup_hold < 0.0:
    p.error("--startup-hold must be non-negative")
  if a.max_guard_hold_streak < 1:
    p.error("--max-guard-hold-streak must be at least 1")
  if a.max_blind_steps < 0:
    p.error("--max-blind-steps must be non-negative")
  if a.sam_vos_optimized:
    p.error("--sam-vos-optimized is disabled: the installed torch 2.13/SAM2.1 "
            "combination fails its first propagated frame.  Eager SAM is the "
            "verified deployment path.")


  # Perception follows the camera, because the alternative is a command that
  # runs happily with the wrong model.  ``--camera d455`` with the default
  # weights used to load ``yolo/best.pt`` -- trained on D405 frames -- and
  # nothing anywhere said so; the rig file has a serial cross-check and this
  # had nothing.
  _YOLO_DIR = {"d405": "yolo", "d455": "yolo_d455"}
  here = pathlib.Path(__file__).resolve().parent
  # 'depth', on both cameras, and deliberately not the 'fused' the D455
  # campaign notes recommend.  Measured through this loop on a replayed D455
  # session rather than as a bare forward pass: depth alone is 23.4 ms a frame
  # (42.7 Hz) with zero control overruns, and fused is 127.8 ms (7.8 Hz) with
  # 23 -- of which YOLO is 108 ms, whatever --yolo-device says.  The 4.5 ms in
  # the campaign notes is the network; this is the stage.  Pass --mask fused
  # explicitly when the segmenter has nothing to segment and a stale
  # observation is the better trade, but read the two summary lines after.
  if a.mask is None:
    a.mask = "depth"
  if a.yolo_weights is None:
    a.yolo_weights = str(here / _YOLO_DIR[a.camera] / "best.pt")
  else:
    other = [c for c in _YOLO_DIR if c != a.camera]
    resolved = pathlib.Path(a.yolo_weights).resolve()
    for cam in other:
      if (here / _YOLO_DIR[cam]) in resolved.parents:
        p.error(f"--yolo-weights {a.yolo_weights} is the {cam.upper()} model "
                f"and --camera is {a.camera}.  Pass the matching weights, or "
                f"--mask depth if no appearance model is wanted.")
  if a.mask in ("yolo", "fused") and not pathlib.Path(a.yolo_weights).exists():
    p.error(f"--mask {a.mask} needs {a.yolo_weights}, which does not exist")
  if a.depth_source == "stereo":
    if a.replay:
      # A recording stores one depth map, not the imagers, so there is nothing
      # for the model to run on.  Failing here beats silently reviewing the
      # camera's depth as though it were the model's.
      p.error("--depth-source stereo needs the raw imagers, which a recording "
              "does not store; replay the session with --depth-source sensor")
    if a.camera != "d455":
      p.error(f"--depth-source stereo is calibrated for the d455's imagers, "
              f"not the {a.camera}")

  print(f"perception: --camera {a.camera}, --mask {a.mask}"
        + (f", weights {pathlib.Path(a.yolo_weights).parent.name}/"
           f"{pathlib.Path(a.yolo_weights).name}"
           if a.mask in ("yolo", "fused") else ""))

  if not a.record:
    a.record = str(pathlib.Path(a.log_root)
                   / time.strftime("%Y%m%d_%H%M%S"))
    print(f"logging to {a.record}")
  real_motion = not (a.dry_run or a.no_arm or a.replay)
  if real_motion:
    print("REAL ARM MOTION requested.  The run keeps the arm enabled on exit, "
          "holds on a stale/empty observation, and never deliberately "
          "powers the drives off.")
    print("Clear the workspace and keep the physical emergency stop in hand.")
    if not sys.stdin.isatty():
      raise SystemExit("refusing real motion without an interactive terminal")
    if input("type 'move' to connect and enable the arm: ").strip() != "move":
      return 1

  # Claim the run directory before touching the arm.  This prevents accidental
  # overwrite and leaves an audit trail even if setup fails before control.
  writer = (_Recorder(a.record, queue_size=a.record_queue,
                      compress=not a.no_record_compress)
            if a.record else None)
  try:
    (rig, reader, reproj, segmenter, tracker, builder, mapper, arm,
     pol) = build(a)
  except BaseException as e:
    if writer is not None:
      writer.event(None, None, 0, "startup_failure", {"error": repr(e)})
      writer.close()
      (writer.dir / "run.json").write_text(json.dumps({
        "stop_reason": "startup_failure: " + repr(e),
        "args": vars(a),
      }, indent=2) + "\n")
    raise

  stopping = False
  stop_reason = "time_limit"

  def _stop(*_):
    nonlocal stopping, stop_reason
    stopping = True
    stop_reason = "operator_interrupt"
  signal.signal(signal.SIGINT, _stop)

  dt = 1.0 / config.CONTROL_HZ
  budget_ms = dt * 1000
  rates = Rates()
  last_action = np.zeros(7, dtype=np.float32)
  overrun_streak = 0
  guard_hold_streak = 0
  blind_streak = 0

  st = arm.read()
  mapper.reset(np.concatenate([st.q, [st.gripper, -st.gripper]]))
  pol.reset()
  builder.reset()
  guard_kin = proprio.Kinematics()
  table_normal = (np.asarray(rig.table_normal_base, dtype=np.float64)
                  if rig.table_normal_base is not None
                  else np.array([0.0, 0.0, 1.0]))
  if a.min_table_clearance is not None:
    print(f"calibrated-table geometry margin: "
          f"{a.min_table_clearance * 1000:.0f} mm")
  print("start joints (deg):", " ".join(
    f"{x:+.1f}" for x in np.degrees(st.q)),
    f" gripper {st.gripper * 2000:.1f} mm jaw gap")
  if writer is not None:
    # Save the calibration beside its images, then record the first measured
    # state immediately after connection and safe measured-pose preload.
    rig.save(writer.dir / "rig.json")
    writer.event(robot.feedback(st, mapper.previous), last_action, 0,
                 "connected_enabled")

  # Not gated on real motion: a rehearsal that starts from a different pose
  # than the powered run is not a rehearsal of it, and the stand-in arm
  # follows the same path for free.
  if a.home_first:
    st = home_arm(arm, builder.spec, speed_rad_s=a.home_speed,
                  should_stop=lambda: stopping)
    mapper.reset(np.concatenate([st.q, [st.gripper, -st.gripper]]))
    pol.reset()
    if writer is not None:
      writer.event(robot.feedback(st, mapper.previous), last_action, 0,
                   "homed", {"reached_deg": np.degrees(st.q).tolist()})

  if real_motion and a.startup_hold:
    arm.hold()
    if writer is not None:
      writer.event(robot.feedback(st, mapper.previous), last_action, 0,
                   "startup_hold_begin",
                   {"duration_s": float(a.startup_hold)})
    print(f"holding measured start pose for {a.startup_hold:.1f} s; "
          "ctrl-c to abort")
    deadline = time.time() + a.startup_hold
    while time.time() < deadline and not stopping:
      time.sleep(min(0.1, max(0.0, deadline - time.time())))
    if writer is not None:
      held = arm.read()
      writer.event(robot.feedback(held, mapper.previous), last_action, 0,
                   "startup_hold_end")
  vision = None
  viewer = None
  if reader is not None:
    stereo_backend = None
    if a.depth_source == "stereo":
      from . import stereo as stereo_mod
      stereo_backend = stereo_mod.StereoDepth(a.stereo_engine)
      # fx * baseline from the camera that is actually plugged in, not from a
      # constant: this is the one number that turns disparity into metres and
      # getting it from the wrong unit is a silent scale error in every depth.
      stereo_backend.calibrate(reader.meta)
      print(f"depth: Fast-FoundationStereo {stereo_backend.path.parent.name}, "
            f"fx*b = {stereo_backend.focal_baseline:.5f} m*px")

    lc = None
    if a.target_lifecycle:
      # Refusing the table's label during a carry only helps if something
      # else supplies the mask; with no rebuild radius it would blank the
      # target for the whole carry, which is worse than the bystander it
      # prevents.  Caught here rather than discovered on the arm.
      if a.held_target_radius <= 0:
        raise SystemExit("--target-lifecycle needs --held-target-radius: "
                         "during a carry the table label is refused and the "
                         "mask is rebuilt at the grasp site instead")
      lc = lifecycle.TargetLifecycle()
      print(f"target: lifecycle on, rebuild radius "
            f"{1000 * a.held_target_radius:.0f} mm")

    sam = None
    if a.target_tracker == "sam21":
      # Imported here and nowhere else: it pulls in torch and a 184 MB
      # checkpoint, and a depth-only run must not pay for either.
      from .sam2_predictor import Sam2StreamingPredictor
      from .sam_tracker import SamTargetTracker
      kw = {} if a.sam_checkpoint is None else {"checkpoint": a.sam_checkpoint}
      pred = Sam2StreamingPredictor(
        device=(f"cuda:{a.yolo_device}"
                if str(a.yolo_device or "").isdigit() else "cuda"),
        vos_optimized=bool(a.sam_vos_optimized),
        **kw)
      sam = SamTargetTracker(pred)
      print(f"target: SAM2.1 carrying the depth stack's choice "
            f"(loaded in {pred.load_s:.1f} s).  The depth segmenter still "
            f"chooses the instance; SAM only carries it, and the watchdog "
            f"withholds rather than publishes when it disagrees.")
      if lc is None:
        # The only signal that says "this object is finished" is the
        # lifecycle's HELD -> SEARCH edge.  Without it SAM keeps propagating
        # the object it was anchored on after that object has been dropped in
        # the bin, and the next anchor cannot arrive until the watchdog gives
        # up on the old one.
        print("WARNING: --target-tracker sam21 without --target-lifecycle "
              "never clears the target on a placement; SAM will keep carrying "
              "the object it was anchored on until the watchdog loses it")

    vision = Perception(reader, reproj, segmenter, tracker,
                        proprio.Kinematics(), rig=rig,
                        flatten=getattr(a, "flatten_scene", False),
                        stereo=stereo_backend,
                        held_radius=a.held_target_radius,
                        gripper_closed_m=a.gripper_closed,
                        depth_bias=a.depth_bias, lifecycle=lc, sam=sam)
    if vision.flatten:
      print("scene: flattened onto the calibrated table plane")
    vision.set_joints(np.concatenate([st.q, [st.gripper, -st.gripper]]))
    vision.start()
    if a.view:
      viewer = Viewer(vision, rig, hz=a.view_hz, scale=a.view_scale)
      viewer.start()
      print("view: window opened; it is a reader and cannot stall the loop")

  blank = np.zeros((3, config.HEIGHT, config.WIDTH), np.float32)
  print(f"running for {a.seconds:.0f} s at {config.CONTROL_HZ:.0f} Hz; "
        "ctrl-c to stop")
  t_end = time.time() + a.seconds
  last_command_time = None

  def hold_and_resync():
    """Hold measured pose and reset every state that could resume from stale."""
    nonlocal last_action, last_command_time
    arm.hold()
    held = arm.read()
    mapper.reset(np.concatenate([held.q, [held.gripper, -held.gripper]]))
    pol.reset()
    builder.reset()
    last_action = np.zeros(7, dtype=np.float32)
    last_command_time = None
    return held

  def guard_hold():
    """Refuse one command without ending the episode the policy is in.

    Deliberately not ``hold_and_resync``.  That one is for an observation that
    was lost, where the recurrent state genuinely no longer describes
    anything, so it clears the GRU and the previous action.  Here the frame is
    fine and only the *command* was rejected: clearing memory every time the
    policy leans at the floor would erase the thing being measured, and zeroing
    the previous action would feed the policy a claim about itself that is not
    true.  The slew limiter is still reseeded from the measured pose, which is
    what stops a refused target accumulating into the next one.
    """
    nonlocal last_command_time
    arm.hold()
    held = arm.read()
    mapper.reset(np.concatenate([held.q, [held.gripper, -held.gripper]]))
    last_command_time = None
    return held

  try:
    while time.time() < t_end and not stopping:
      # Read feedback halfway through the physical command period, leaving the
      # normal compute path time to finish before the exact send slot.
      if last_command_time is not None:
        time.sleep(max(0.0, last_command_time + 0.5 * dt - time.time()))
      t0 = time.time()

      st = arm.read()
      fault = _feedback_fault(st, a.max_joint_speed_fraction)
      if fault is not None:
        stop_reason = "feedback_fault: " + fault
        if writer is not None:
          writer.event(robot.feedback(st, mapper.previous), last_action, 0,
                       "blocked_feedback_fault", {"fault": fault})
        print(f"\nSTOPPING: {fault}")
        break
      actual_clearance, actual_geom = _table_clearance(
        guard_kin, np.concatenate([st.q, [st.gripper]]),
        table_normal, rig.table_z)
      if (a.min_table_clearance is not None
          and actual_clearance < a.min_table_clearance):
        stop_reason = (f"measured_table_clearance {actual_clearance:.4f} m "
                       f"at {actual_geom} below {a.min_table_clearance:.4f} m")
        if writer is not None:
          writer.event(robot.feedback(st, mapper.previous), last_action, 0,
                       "blocked_measured_table_clearance", {
                         "measured_table_clearance_m": float(actual_clearance),
                         "minimum_table_clearance_m": float(a.min_table_clearance),
                         "closest_geometry": actual_geom,
                       })
        print("\nSTOPPING: " + stop_reason)
        break
      actual_height = _grasp_height(
        guard_kin, np.concatenate([st.q, [st.gripper]]))
      if (a.min_grasp_height is not None
          and actual_height < a.min_grasp_height):
        stop_reason = (f"measured_grasp_height {actual_height:.4f} m below "
                       f"{a.min_grasp_height:.4f} m")
        if writer is not None:
          writer.event(robot.feedback(st, mapper.previous), last_action, 0,
                       "blocked_measured_height", {
                         "measured_grasp_height_m": float(actual_height),
                         "minimum_grasp_height_m": float(a.min_grasp_height),
                       })
        print("\nSTOPPING: " + stop_reason)
        break
      fb = robot.feedback(st, mapper.previous)
      if vision is not None:
        vision.set_joints(fb.position)
        # The same latched bit the observation builder uses, so perception and
        # proprioception cannot disagree about whether the gripper is loaded.
        vision.set_contact(builder.contact_latched)

      out = vision.latest() if vision is not None else None
      if out is None:
        camera, label, frame, held_over = blank, 0, None, False
        target_available = vision is None
        published_at = None
        age = publication_age = 0.0
        if vision is not None:
          # Nothing to act on yet.  Not an error in the first fraction of a
          # second, and not something to drive an arm with either.
          rates.stale += 1
          held = hold_and_resync()
          if writer is not None:
            writer.event(robot.feedback(held, mapper.previous), last_action, 0,
                         "hold_no_observation")
          time.sleep(dt)
          continue
      else:
        camera, published_at, frame, label, held_over, target_available = out
        now = time.time()
        age = _observation_age(frame, now)
        publication_age = now - published_at
        rates.note_observation(age, publication_age)
        if age > a.max_obs_age:
          rates.stale += 1
          held = hold_and_resync()
          if writer is not None:
            writer.event(robot.feedback(held, mapper.previous), last_action,
                         label, "hold_stale_observation", {
                           "frame_index": _frame_index(frame),
                           "frame_stamp": _frame_stamp(frame),
                           "observation_age_s": float(age),
                           "perception_publish_age_s": float(publication_age),
                         }, frame=frame)
          time.sleep(dt)
          continue
      if not target_available or held_over:
        if not target_available:
          rates.no_target += 1
        # An empty mask is not the same thing as a lost observation, and the
        # simulator is the authority on which.  ``CameraScene`` builds the mask
        # from the segmentation buffer's frontmost geom, so when the arm passes
        # in front of the target the *simulated* mask empties too -- measured at
        # 0 px in the robust domain -- and the policy keeps acting through it.
        # Deployment used to hold instead, and on the first powered run that
        # deadlocked: the arm descended over the object, occluded it from the
        # fixed camera, lost the target, held its pose, and holding could not
        # change the view.  It stayed frozen for 1168 of 1376 steps.
        #
        # So a target the tracker still believes in is driven through, with the
        # empty mask the segmenter actually produced.  The tracker's own
        # ``lost_frames`` window decides how long it believes; ``--max-blind-
        # steps`` bounds it again in control steps, because the two clocks
        # differ and a camera that stopped delivering must not read as a target
        # that is merely hidden.
        believed = (target_available and vision is not None
                    and a.max_blind_steps > 0 and held_over
                    and blind_streak < a.max_blind_steps)
        if believed:
          blind_streak += 1
          rates.blind += 1
          if writer is not None:
            writer.event(robot.feedback(st, mapper.previous), last_action, 0,
                         "blind_target", {
                           "frame_index": _frame_index(frame),
                           "frame_stamp": _frame_stamp(frame),
                           "observation_age_s": float(age),
                           "consecutive_blind_steps": blind_streak,
                           "mask": "last known, object occluded",
                           "perception_publish_age_s": float(publication_age),
                         }, frame=frame)
        else:
          held = hold_and_resync()
          if writer is not None:
            writer.event(robot.feedback(held, mapper.previous), last_action, 0,
                         "hold_no_target", {
                           "frame_index": _frame_index(frame),
                           "frame_stamp": _frame_stamp(frame),
                           "observation_age_s": float(age),
                           "consecutive_blind_steps": blind_streak,
                           "perception_publish_age_s": float(publication_age),
                         }, frame=frame)
          time.sleep(dt)
          continue
      else:
        blind_streak = 0

      flat = builder(fb, last_action)
      last_action = pol(flat, camera)
      fault = _action_fault(last_action)
      if fault is not None:
        stop_reason = "action_fault: " + fault
        print(f"\nSTOPPING: {fault}")
        if writer is not None:
          writer.event(fb, last_action, label, "blocked_action_fault", {
            "fault": fault,
            "frame_index": _frame_index(frame),
            "observation_age_s": float(age),
          })
        break
      map_dt = (dt if last_command_time is None
                else max(dt, time.time() - last_command_time))
      command = mapper(last_action, dt=map_dt)
      target_clearance, target_geom = _table_clearance(
        guard_kin, command, table_normal, rig.table_z)
      if (a.min_table_clearance is not None
          and target_clearance < a.min_table_clearance):
        breach = (f"predicted_table_clearance {target_clearance:.4f} m "
                  f"at {target_geom} below {a.min_table_clearance:.4f} m")
        detail = {
          "frame_index": _frame_index(frame),
          "frame_stamp": _frame_stamp(frame),
          "observation_age_s": float(age),
          "measured_table_clearance_m": float(actual_clearance),
          "commanded_table_clearance_m": float(target_clearance),
          "minimum_table_clearance_m": float(a.min_table_clearance),
          "closest_geometry": target_geom,
        }
        if a.guard_mode == "hold":
          guard_hold_streak += 1
          rates.guard_holds += 1
          held = guard_hold()
          if writer is not None:
            writer.event(robot.feedback(held, mapper.previous), last_action,
                         label, "held_predicted_table_clearance",
                         detail | {"consecutive_guard_holds": guard_hold_streak},
                         frame=frame)
          if guard_hold_streak >= a.max_guard_hold_streak:
            stop_reason = (f"{guard_hold_streak} consecutive guard holds: "
                           + breach)
            print("\nSTOPPING: " + stop_reason)
            break
          time.sleep(dt)
          continue
        stop_reason = breach
        if writer is not None:
          writer.event(robot.feedback(st, command), last_action, label,
                       "blocked_predicted_table_clearance", detail)
        print("\nSTOPPING: " + stop_reason)
        break
      # The commanded grasp height is RECORDED and it is not a stop condition.
      #
      # There used to be a second grasp-height guard here that tested
      # FK(command) against ``--min-grasp-height``.  That sounds like the more
      # protective of the two and it is not, because the command is a setpoint
      # that runs far ahead of the arm and the arm never arrives at it -- the
      # next one replaces it 20 ms later.  Measured 2026-09-01 over 4800
      # samples of the deployed v4 policy in the domain it was trained in,
      # with the task succeeding throughout:
      #
      #   FK(commanded target)   p0 -214.9 mm   median 13.9 mm   44.7% below
      #                                                          the table
      #   measured grasp site    p0   13.2 mm   median 63.9 mm    0.0% below
      #                                                          the table
      #
      # So a floor anywhere near the table fires on the setpoint while the
      # robot is a hand's width up.  It did exactly that three times: v3
      # stopped at -5.7 mm against a 5 mm floor, v4 at 2.5 mm and at 18.5 mm
      # against a 20 mm floor, each within seconds of starting.  The
      # recordings put the arm at 93.9 mm and 55.0 mm at those moments, and the
      # operator watching it agreed -- at least 5 cm of clearance, every time.
      # Guarding the setpoint is not a stricter version of guarding the robot;
      # it is guarding a different quantity, and this one has no floor.
      #
      # Nothing is lost by removing it.  The MEASURED grasp height is already
      # guarded against the same ``--min-grasp-height`` earlier in this loop,
      # on the robot's own joint feedback, and that is the check that means
      # what the flag says.  The setpoint stays in the log so a run can still
      # be reviewed for a policy that commands into the table.
      commanded_height = _grasp_height(guard_kin, command)
      guard_hold_streak = 0
      if last_command_time is not None:
        time.sleep(max(0.0, last_command_time + dt - time.time()))
      command_time = time.time()
      command_interval = (None if last_command_time is None
                          else command_time - last_command_time)
      arm.command(command, dt)
      last_command_time = command_time

      ms = (time.time() - t0) * 1000
      if writer is not None and frame is not None:
        logged_fb = robot.feedback(st, command)
        writer.write(_logged_depth(frame), frame.gray, logged_fb,
                     last_action, label,
                     extra={
                       "frame_index": _frame_index(frame),
                       "frame_stamp": _frame_stamp(frame),
                       "observation_age_s": float(age),
                       "loop_ms_before_log_copy": float(ms),
                       "measured_grasp_height_m": float(actual_height),
                       "perception_publish_age_s": float(publication_age),
                       "commanded_grasp_height_m": float(commanded_height),
                       "measured_table_clearance_m": float(actual_clearance),
                       "commanded_table_clearance_m": float(target_clearance),
                       "closest_geometry": target_geom,
                       "command_interval_s": (
                         float(command_interval)
                         if command_interval is not None else None),
                       "mapping_dt_s": float(map_dt),
                     })
      rates.note(ms, budget_ms)
      overrun_streak = overrun_streak + 1 if ms > budget_ms else 0
      if overrun_streak >= a.max_overrun_streak:
        stop_reason = f"{overrun_streak} consecutive control overruns"
        if writer is not None:
          writer.event(robot.feedback(st, command), last_action, label,
                       "blocked_control_overrun", {
                         "consecutive_overruns": int(overrun_streak),
                         "loop_ms": float(ms),
                       })
        print(f"\nSTOPPING: {overrun_streak} consecutive steps over the "
              f"{budget_ms:.0f} ms budget.  The slew limiter assumes a fixed "
              "period, so late commands let the arm travel further than the "
              "joint can.")
        break

  except BaseException as e:
    stop_reason = "exception: " + repr(e)
    if writer is not None:
      writer.event(None, None, 0, "control_exception",
                   {"error": repr(e)})
    raise
  finally:
    arm.hold()
    final = None
    try:
      final_st = arm.read()
      final = {
        "q_rad": np.asarray(final_st.q).tolist(),
        "dq_rad_s": np.asarray(final_st.dq).tolist(),
        "gripper_m": float(final_st.gripper),
      }
      if writer is not None:
        held_target = np.concatenate([final_st.q, [final_st.gripper]])
        writer.event(robot.feedback(final_st, held_target), last_action, 0,
                     "final_hold", {"stop_reason": stop_reason})
    except Exception:
      pass
    arm.close()
    if viewer is not None:
      viewer.close()
      if viewer.failed:
        print(f"view: the window could not be drawn ({viewer.failed}); the "
              "run was unaffected")
    perception_report = None
    if vision is not None:
      vision.close()
      perception_report = vision.report()
      print(vision.summary())
    if reader is not None:
      reader.close()
    if writer is not None:
      writer.close()
      (writer.dir / "run.json").write_text(json.dumps({
        "stop_reason": stop_reason,
        "args": vars(a),
        "rates": rates.report(),
        "perception": perception_report,
        "final_feedback": final,
      }, indent=2) + "\n")
    print("\n" + rates.summary())
    if not getattr(a, "no_review", False):
      _write_review(writer.dir, stride=a.review_stride)
  return 0


def _write_review(session: pathlib.Path, stride: int = 12) -> None:
  """Re-segment the session and leave the page beside the data.

  Written here rather than left to whoever remembers, because the numbers that
  decide whether a run meant anything -- which object the tracker chose, how
  tall it read, how far the hand stayed from it -- are not in the terminal
  summary and take a re-run of the segmenter to recover.  A session nobody
  looks at is a session that was not worth recording.

  Never raises into the caller: the arm is already held and the data is
  already on disk, and a plotting failure must not read as a control failure.
  """
  try:
    t0 = time.time()
    from . import review
    d = review.run(session, stride=max(1, int(stride)), quality=84)
    meta = {}
    rj = session / "run.json"
    if rj.exists():
      meta = json.loads(rj.read_text())
    (session / "review.html").write_text(review.build_page(d, meta))
    (session / "review.json").write_text(json.dumps(
      {k: v for k, v in d.items() if k not in ("frames", "rows", "shown")},
      indent=1) + "\n")
    print(f"review: {session / 'review.html'}  ({time.time() - t0:.1f} s)")
    print(review.summary(d))
  except BaseException as e:            # noqa: BLE001 - see below
    # BaseException, not Exception.  The docstring above promises this never
    # raises into the caller, and ``except Exception`` does not keep that
    # promise: SystemExit derives from BaseException and walked straight
    # through, which ended a --dry-run with a nonzero exit and failed the
    # bring-up gate over a session that legitimately has no frames.
    print(f"review: could not be written ({e!r}); the data is intact in "
          f"{session}")


def _logged_depth(frame):
  """The depth a recording should hold: the one the policy was given."""
  d = getattr(frame, "policy_depth", None)
  return frame.depth if d is None else d


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

  def __init__(self, reader, reproj, segmenter, tracker, kin, rig=None,
               flatten: bool = False, stereo=None,
               held_radius: float = 0.0, gripper_closed_m: float = 0.045,
               depth_bias: float = 0.0, lifecycle=None, sam=None):
    super().__init__(daemon=True, name="perception")
    self.reader = reader
    # Held by THIS thread and nobody else: it owns a TensorRT context and CUDA
    # buffers, and the control loop must never touch it.
    self.stereo = stereo
    self.stereo_misses = 0
    # Rebuilding the target at the grasp site once the jaws are closed on it;
    # see the note where it is applied.  0 disables it.
    self.held_radius = float(held_radius)
    self.gripper_closed_m = float(gripper_closed_m)
    self.held_frames = 0
    self.depth_bias = float(depth_bias)
    self.reproj = reproj
    self.rig = rig
    self.flatten = bool(flatten) and rig is not None
    self._plane = (reproj.ground_plane_depth(rig) if self.flatten else None)
    """The calibrated tabletop, per policy pixel.  Fixed for the session --
    the camera is bolted down and the plane was measured -- so it is computed
    once rather than per frame."""
    self.segmenter = segmenter
    self.tracker = tracker
    self.kin = kin
    # Which instance the robot is working on.  ``None`` keeps the historical
    # behaviour, where the tracker's choice went straight to the policy and a
    # stale target let a bystander take over -- 326 times across 20 recorded
    # sessions, every one of them during a carry.  See ``lifecycle`` and
    # ``scripts/measure_target_gaps.py --replay-lifecycle``.
    self.lifecycle = lifecycle
    self._loaded = False
    self._lock = threading.Lock()
    self._joints = None
    self._out = None
    self._view = None
    self._error: BaseException | None = None
    # ``SamTargetTracker`` or None.  When it is present the grasp-site sphere
    # becomes a fallback rather than an override: measured against the
    # renderer, SAM's held mask is IoU 0.804 and the sphere is 0.614, so
    # rebuilding on top of a mask that already exists is a downgrade.
    self.sam = sam
    self.sam_state = "off"
    self._was_holding = False
    self._target_mask = target_mask.TargetMask(
      self.held_radius, reproj, rig, rebuild_only_if_empty=sam is not None)
    """The most recent mask that came from a confirmed target, kept so an
    occluded object still reads as present.  Cleared when the tracker gives up
    on it, never carried across a reset."""
    # Not ``_stop``: ``threading.Thread`` already has one and shadowing it
    # replaces a method the interpreter calls when the thread ends.
    self._stopping = threading.Event()
    self.periods: list[float] = []
    self.capture_to_publish_ms: list[float] = []
    self.stage_ms = {
      "depth_source": [],
      "segment_and_select": [],
      "sam": [],
      "reproject_and_observation": [],
    }
    self._first_finished: float | None = None
    self._last_finished: float | None = None
    self.frames = 0

  def set_joints(self, q) -> None:
    """The control loop's most recent joint reading.

    The segmenter needs the arm's pose to subtract it, and it must not read the
    arm itself: two threads talking to one CAN interface is a way to lose
    frames on both.
    """
    with self._lock:
      self._joints = np.asarray(q, dtype=np.float64).copy()

  def set_contact(self, loaded: bool) -> None:
    """The gripper's contact latch, from the control loop.

    ``proprio.ProprioBuilder._contact_bit`` already applies hysteresis to the
    drive current and was measured on this gripper; the perception thread had
    no way to see it, so "am I holding something" was being decided on the jaw
    gap alone.  A gap says the jaws are closed, not that they closed on
    anything.
    """
    with self._lock:
      self._loaded = bool(loaded)

  def latest(self):
    with self._lock:
      error, out = self._error, self._out
    if error is not None:
      raise RuntimeError("perception thread failed") from error
    return out

  def view(self):
    """The pieces a live viewer wants, or None.  Cheap and lock-brief."""
    with self._lock:
      return self._view

  def run(self) -> None:
    try:
      self._run()
    except BaseException as e:
      with self._lock:
        self._error = e
      self._stopping.set()

  def _run(self) -> None:
    last_index = -1
    while not self._stopping.is_set():
      frame = self.reader.latest()
      with self._lock:
        q = None if self._joints is None else self._joints.copy()
        loaded = self._loaded
      if frame is None or q is None or frame.index == last_index:
        time.sleep(0.002)
        continue
      last_index = frame.index
      t0 = time.perf_counter()

      self.kin.update(q)
      stage_t = time.perf_counter()
      # One depth map per frame, and every stage below sees the same one.
      # Computed here rather than in the reader so the 14 ms of GPU lands on
      # the thread that already owns the perception budget, and the reader
      # stays a pure latest-frame-wins buffer.
      raw_depth = frame.depth
      if self.stereo is not None:
        if frame.ir is None or frame.ir_right is None:
          # The imagers did not arrive.  Fall back rather than stall: the
          # camera's own depth is what this deployment ran on for its entire
          # history, and it is already in hand.
          self.stereo_misses += 1
        else:
          raw_depth = self.stereo(frame.ir, frame.ir_right)
          frame.policy_depth = raw_depth

      # One constant, subtracted once, before anything reads the depth.
      #
      # Measured two ways on 2026-09-01 and they agree, which is the only
      # reason this exists rather than a tuning knob:
      #
      #   graspcheck, object placed where a grasp would hold it, arm read-only:
      #     site - object in the gripper frame was z = -18.9 mm along the
      #     APPROACH axis.
      #   simulation, the same policy in the domain it was trained in:
      #     the grasp site never goes below 13.2 mm and is below the table
      #     0.00% of the time, while on the arm it reaches -6 to -41 mm.
      #
      # Both say the same thing: perception places the object about 19 mm
      # further along the approach than it is, so the policy drives 19 mm too
      # deep and either digs into the table or bats the object away.  It does
      # not show up in the segmenter's masks because that re-fits the table
      # plane every frame and is insensitive to a depth offset by
      # construction -- but the policy reads channel 0 directly, and there it
      # is a 19 mm lie about where the table and the object are.
      #
      # Applied here, to the single depth map the frame is built from, so the
      # segmenter, the reprojection and the policy cannot disagree about it.
      # Positive values pull the scene TOWARDS the camera.
      if self.depth_bias:
        raw_depth = np.where(raw_depth > 0.0, raw_depth - self.depth_bias, 0.0)
      self.stage_ms["depth_source"].append(
        (time.perf_counter() - stage_t) * 1000.0)
      stage_t = time.perf_counter()

      arm_spheres = self.kin.link_spheres()
      # Read before the lifecycle is updated, so a HELD -> SEARCH transition
      # (the object was let go) is visible as an edge below.
      self._was_holding = (self.lifecycle.holding
                           if self.lifecycle is not None else False)
      seg = self.segmenter(raw_depth, rgb=frame.gray, arm=arm_spheres)
      label = self.tracker.update(seg, self.kin.site_pos)
      if self.lifecycle is not None:
        jaws_closed = (q.size > 6 and self.gripper_closed_m > 0
                       and float(q[6]) < self.gripper_closed_m)
        label = self.lifecycle.update(label, jaws_closed, loaded)
      payload = (mask.full_mask(seg, label, self.segmenter.decimate)
                 if label else None)
      self.stage_ms["segment_and_select"].append(
        (time.perf_counter() - stage_t) * 1000.0)

      # SAM2.1 carries the instance the depth stack chose, between frames and
      # across its gaps.  It never introduces one: with no depth mask and no
      # existing anchor, ``carry`` returns nothing.  The order matters -- the
      # lifecycle has already had its say about WHICH instance, so what SAM is
      # handed is a target the rest of the stack has confirmed.
      if self.sam is not None:
        # NOT reset on ``tracker.has_target``.  That was tried and measured:
        stage_t = time.perf_counter()
        # the depth tracker's window expires after 15 frames without an
        # instance, and during a carry the segmenter produces none at all, so
        # resetting on it clears SAM on every frame of exactly the phase it
        # exists for -- SAM's held detection went from 100% to 1.2%.  What
        # clears the target is a lifecycle event: the object was placed, or a
        # different instance was confirmed.
        carrying_now = (self.lifecycle.holding
                        if self.lifecycle is not None else False)
        if self.lifecycle is not None and self._was_holding and not carrying_now:
          self.sam.reset("placed")
        chosen = (payload.astype(bool) if payload is not None
                  else np.zeros(raw_depth.shape, bool))
        arm_img = (self.segmenter.arm_image_mask(
                     raw_depth, arm_spheres, self.kin.site_pos)
                   if hasattr(self.segmenter, "arm_image_mask") else None)
        rep = self.sam.carry(frame.gray, chosen, raw_depth > 0, arm_img)
        self.sam_state = rep.state.value + (f" ({rep.reason})"
                                            if rep.reason else "")
        payload = (rep.mask.astype(np.int32)
                   if rep.mask is not None and rep.mask.any() else None)

        self.stage_ms["sam"].append(
          (time.perf_counter() - stage_t) * 1000.0)
      stage_t = time.perf_counter()
      depth, valid, target = self.reproj(raw_depth, payload=payload)
      if self._plane is not None:
        depth, valid = obs.flatten_scene(
          depth, valid, self._plane,
          points_base=self.reproj.virtual_points_base(depth, self.rig),
          arm=self.kin.link_spheres())
      if target is None:
        target = np.zeros_like(valid)

      # Hold-over while the arm covers the object, and reconstruction at the
      # grasp site during a carry.  Both live in ``target_mask`` so that
      # ``scripts/sim_perception_check.py`` scores THIS code against the
      # renderer rather than a second copy of it.
      carrying = (self.lifecycle.holding if self.lifecycle is not None
                  else (not label
                        and q.size > 6 and self.gripper_closed_m > 0
                        and float(q[6]) < self.gripper_closed_m))
      target, held_over = self._target_mask(
        target, label, self.tracker.has_target, carrying, depth, valid,
        self.kin.site_pos)
      self.held_frames = self._target_mask.rebuilds
      camera = obs.camera_obs(depth, valid, target > 0)
      self.stage_ms["reproject_and_observation"].append(
        (time.perf_counter() - stage_t) * 1000.0)

      finished = time.perf_counter()
      published_at = time.time()
      self.periods.append(finished - t0)
      self.capture_to_publish_ms.append(
        max(0.0, (published_at - float(frame.stamp)) * 1000.0))
      self.frames += 1
      if self._first_finished is None:
        self._first_finished = finished
      self._last_finished = finished
      target_available = bool(np.asarray(target).any())
      with self._lock:
        # Safety uses capture time; publish time only measures queueing.
        self._out = (camera, published_at, frame, label, held_over,
                     target_available)
        # References, not copies: the viewer annotates on its own thread at its
        # own rate and the segmenter has already finished with these.  A copy
        # here would put the viewer's cost on the perception thread, which is
        # the one thing that must not happen.
        self._view = (frame, seg, label, camera, q)

  def close(self) -> None:
    self._stopping.set()
    self.join(timeout=2.0)

  def report(self) -> dict:
    elapsed = (None if self._first_finished is None
               or self._last_finished is None
               else max(0.0, self._last_finished - self._first_finished))
    actual_hz = (None if elapsed is None or elapsed <= 0.0 or self.frames < 2
                 else float((self.frames - 1) / elapsed))
    out = {
      "frames": int(self.frames),
      "elapsed_s": elapsed,
      "actual_hz": actual_hz,
      "compute_ms": _timing_stats(np.asarray(self.periods) * 1000.0),
      "capture_to_publish_ms": _timing_stats(self.capture_to_publish_ms),
      "stages_ms": {name: _timing_stats(values)
                    for name, values in self.stage_ms.items()},
      "held_rebuilds": int(self.held_frames),
      "stereo_misses": int(self.stereo_misses),
    }
    if self.sam is not None:
      out["sam"] = self.sam.report()
    return out

  def summary(self) -> str:
    report = self.report()
    compute = report["compute_ms"]
    if compute is None:
      return "perception: no frames"
    hz = report["actual_hz"]
    hz_text = "n/a" if hz is None else f"{hz:.1f} Hz actual"
    return (f"perception: {self.frames} frames, {hz_text}, compute median "
            f"{compute['p50']:.1f} ms, p95 {compute['p95']:.1f} ms, "
            f"capture-to-publish p95 "
            f"{report['capture_to_publish_ms']['p95']:.1f} ms"
            + (f", {self.held_frames} frame(s) with the target rebuilt at the "
               f"grasp site" if self.held_frames else "")
            + (f", {self.stereo_misses} frame(s) without an imager pair"
               if self.stereo_misses else "")
            + (f"\n{self.sam.summary()}" if self.sam is not None else ""))


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
  """Asynchronous full-resolution replay log plus 50 Hz control metadata."""

  def __init__(self, path: str, queue_size: int = 512,
               compress: bool = True):
    # Compression is where a successful run goes to die, which is worth
    # spelling out because the coupling is backwards from what it looks like.
    # ``savez_compressed`` runs zlib on 1.2 MB per camera frame on the writer
    # thread.  While the policy is holding, that thread has the CPU to itself
    # and keeps up; the moment the policy starts *working*, inference takes the
    # cores, the writer falls behind, and the queue fills.  Measured today:
    # 1985 frames and 300 commands finished cleanly, while 1144 frames with 996
    # commands and 1433 frames with 1086 commands both hit the ceiling.  The
    # better the run, the sooner it is cut off.
    #
    # Uncompressed costs disk -- about 1.2 MB per frame against roughly a
    # quarter of that -- and costs the writer almost nothing.  For a 60 s run
    # that is a couple of gigabytes, which is the cheaper of the two.
    self.compress = bool(compress)
    self.dir = pathlib.Path(path)
    if self.dir.exists() and any(self.dir.iterdir()):
      raise RuntimeError(f"refusing to overwrite non-empty recording {self.dir}")
    self.dir.mkdir(parents=True, exist_ok=True)
    self.n = 0
    self._last_frame = -1
    self.meta = []
    self.control = []
    self._error = None
    self._closed = False
    self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
    self._thread = threading.Thread(target=self._worker, daemon=True,
                                    name="deployment-log-writer")
    self._thread.start()

  def write(self, depth, gray, fb, action, label, extra=None) -> None:
    if self._closed:
      raise RuntimeError("recording is already closed")
    if self._error is not None:
      raise RuntimeError(f"recording worker failed: {self._error}")
    # One picture per CAMERA frame, not one per control step.
    #
    # This method set ``_last_frame`` and never read it, so a 50 Hz control
    # loop against a 30 Hz camera stored every frame between one and two extra
    # times.  Measured on v4_stereo_repro_scene2: 2893 files for 1488 distinct
    # camera frames, and 38% of consecutive pairs byte-identical in depth.
    # ``event`` -- the path taken while holding -- has always deduplicated this
    # way; only this one did not.
    #
    # It cost three things.  The review page repeats pictures, so scrubbing it
    # reads as a low frame rate when the camera was delivering its full 33.4 ms
    # cadence.  The writer thread compressed each image about twice, which is
    # what filled the queue and ended three of today's runs.  And working
    # around that with --no-record-compress freed enough CPU to halve the
    # observation latency, moving a control condition nobody meant to move.
    #
    # The control row is still appended every step: the feedback and the action
    # are 50 Hz facts and they were not what was duplicated.
    idx = None if not extra else extra.get("frame_index")
    rec = self._record(fb, action, label, "command", extra)
    if idx is not None and int(idx) == self._last_frame:
      self.control.append(rec)
      return
    if idx is not None:
      self._last_frame = int(idx)
    i = self.n
    self.n += 1
    rec["i"] = i
    rec["frame_file"] = f"{i:06d}.npz"
    item = (
      rec,
      np.asarray(depth, dtype=np.float32).copy(),
      (np.asarray(gray).copy() if gray is not None
       else np.zeros((1, 1), np.uint8)),
    )
    try:
      self._queue.put_nowait(item)
    except queue.Full as e:
      raise RuntimeError(
        f"recording queue full at {self.n}; refusing an unlogged run") from e
    self.control.append(dict(rec))

  def _record(self, fb, action, label, event, extra=None) -> dict:
    rec = {"control_i": len(self.control), "t": time.time(),
           "event": str(event), "label": int(label)}
    if fb is not None:
      rec.update({
        "joint_pos": np.asarray(fb.position).copy().tolist(),
        "joint_vel": np.asarray(fb.velocity).copy().tolist(),
        "target": np.asarray(fb.target).copy().tolist(),
        # The gripper drive's normalised load, 1.0 being stall.  It is on
        # JointFeedback and was not being written down, and it is the only
        # observable the real gripper loop has: `pad_contact` is reconstructed
        # from it against a threshold that is documented as a guess, and
        # GRIPPER_TORQUE_NM is a starting point rather than a measurement.  A
        # session that squeezes something and does not log this cannot answer
        # either question afterwards.
        "gripper_effort": float(fb.gripper_effort),
      })
    if action is not None:
      rec["action"] = np.asarray(action).copy().tolist()
    if extra:
      rec.update(dict(extra))
    return rec

  def event(self, fb, action, label, event, extra=None, frame=None) -> None:
    """Record a control decision, with its camera frame when there is one.

    ``frame`` used to be unavailable here, so a step that held -- for a guard,
    for a missing target, for a stale observation -- left a line in the control
    log and no picture.  That is exactly backwards: a run that holds is a run
    somebody needs to look at, and several sessions ended after a hundred
    consecutive holds having archived only the handful of frames where a
    command was actually sent.

    Frames are deduplicated by the camera's own index, so a 50 Hz loop holding
    on a 30 Hz camera stores each image once rather than one per control step.
    """
    if self._closed:
      raise RuntimeError("recording is already closed")
    rec = self._record(fb, action, label, event, extra)
    if frame is not None and int(frame.index) != self._last_frame:
      self._last_frame = int(frame.index)
      rec["i"] = self.n
      rec["frame_file"] = f"{self.n:06d}.npz"
      self.n += 1
      self._queue.put((rec, _logged_depth(frame).copy(), frame.gray.copy()))
      self.control.append(rec)
      return
    self.control.append(rec)

  def _worker(self) -> None:
    while True:
      item = self._queue.get()
      try:
        if item is None:
          return
        rec, depth, gray = item
        if self._error is None:
          try:
            save = np.savez_compressed if self.compress else np.savez
            save(
              self.dir / f"{rec['i']:06d}.npz",
              depth=(depth * 10000).astype(np.uint16),
              gray=gray,
            )
            self.meta.append(rec)
          except Exception as e:
            self._error = e
      finally:
        self._queue.task_done()

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    self._queue.put(None)
    self._queue.join()
    self._thread.join(timeout=5.0)
    (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2) + "\n")
    (self.dir / "control.json").write_text(
      json.dumps(self.control, indent=2) + "\n")
    print(f"recorded {len(self.meta)} camera frames and "
          f"{len(self.control)} control events to {self.dir}")
    if self._error is not None:
      raise RuntimeError(f"recording worker failed: {self._error}")


class Viewer(threading.Thread):
  """A window showing what the segmenter sees, on its own thread and clock.

  Deliberately a *reader*.  It takes whatever the perception thread last
  produced, annotates its own copy, and draws; it never asks for a frame, never
  holds the perception lock for longer than a tuple assignment, and skips a
  redraw rather than falling behind.  The control loop does 1 ms of arithmetic
  in a 20 ms period and the perception thread is the expensive one -- putting a
  drawing cost on either is how a diagnostic aid becomes a control fault.

  It is also allowed to fail.  A missing display, a headless session, a Qt that
  cannot find its fonts: the run continues and says so once.  Nothing here is
  load-bearing.
  """

  def __init__(self, vision, rig, hz: float = 10.0, scale: float = 1.0):
    super().__init__(daemon=True, name="viewer")
    self.vision = vision
    self.rig = rig
    self.period = 1.0 / max(1e-3, float(hz))
    self.scale = float(scale)
    self._stopping = threading.Event()
    self.failed: str | None = None

  def run(self) -> None:
    import cv2

    from . import overlay

    title = "deploy: depth and segmentation"
    try:
      cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    except Exception as e:
      self.failed = f"{type(e).__name__}: {e}"
      return

    last = -1
    while not self._stopping.is_set():
      t0 = time.time()
      v = self.vision.view()
      if v is None or v[0] is None or int(v[0].index) == last:
        time.sleep(0.01)
        continue
      frame, seg, label, camera, q = v
      last = int(frame.index)
      try:
        gray = cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
        overlay.annotate(gray, self.rig, seg, label,
                         self.vision.segmenter.decimate,
                         caption=f"frame {frame.index}   "
                                 f"{len(seg.instances)} instance(s)   "
                                 f"target {label or '-'}")
        depth = frame.depth
        v8 = np.clip((depth - 0.35) / 0.85, 0, 1)
        dep = cv2.applyColorMap((v8 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        dep[depth <= 0] = (32, 32, 32)
        overlay.draw(dep, self.rig, labels=False)

        # The three channels the policy is actually given, at the size it
        # gets them, so the window answers "what did the network see" and not
        # only "what was on the table".
        tiles = []
        for ch, tint in ((0, (1.0, 1.0, 1.0)), (1, (0.45, 1.0, 0.45)),
                         (2, (1.0, 0.80, 0.35))):
          a = (np.clip(camera[ch], 0, 1) * 255).astype(np.uint8)
          t = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR).astype(np.float32)
          t = (t * np.asarray(tint, np.float32)).clip(0, 255).astype(np.uint8)
          tiles.append(cv2.resize(t, (gray.shape[1] // 3, gray.shape[0] // 3),
                                  interpolation=cv2.INTER_NEAREST))
        strip = np.hstack(tiles)
        strip = np.pad(strip, ((0, 0),
                               (0, 2 * gray.shape[1] - strip.shape[1]), (0, 0)))
        comp = np.vstack([np.hstack([gray, dep]), strip])
        if self.scale != 1.0:
          comp = cv2.resize(comp, None, fx=self.scale, fy=self.scale)
        cv2.imshow(title, comp)
        cv2.waitKey(1)
      except Exception as e:
        self.failed = f"{type(e).__name__}: {e}"
        break
      time.sleep(max(0.0, self.period - (time.time() - t0)))

    try:
      cv2.destroyWindow(title)
      cv2.waitKey(1)
    except Exception:
      pass

  def close(self) -> None:
    self._stopping.set()
    self.join(timeout=1.0)


if __name__ == "__main__":
  sys.exit(main())
