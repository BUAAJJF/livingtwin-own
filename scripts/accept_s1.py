"""The S1 acceptance gate: does a checkpoint pick and place unseen shapes?

The training log reports placements per episode, which conflates throughput
with reliability -- a policy that places twice as often but drops half of what
it grabs reads the same as one that never drops.  This scores the two
separately, and splits both by shape so a class the policy cannot handle
cannot hide inside a healthy average.

Acceptance, as agreed for S1:

    overall success              >= 85 %
    every shape class            >= 70 %
    post-grasp drop rate         <=  5 %

An *instance* is one object from the moment it is spawned until it is placed.
It counts as a success if it is placed within ``--budget`` seconds of
appearing; instances that the episode ended on before that budget elapsed are
censored rather than failed, because they were never given the chance.

A *grasp* is a contiguous span with both pads on the object that lifts it
clear of the table.  It is dropped if the object is neither placed nor still
held ``--drop-grace`` seconds later.

Usage:

    python scripts/accept_s1.py Mjlab-Pick-Place-PiperX <checkpoint.pt>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push import evalcfg
from piper_push.robot import JOINT_TRIP_RAD_S

# The three aspect classes the shape curriculum spans.  Cut at the ratio of the
# vertical half-extent to the mean horizontal one, so the label describes the
# grasp the object demands rather than its absolute size.
ASPECT_CUTS = (0.8, 1.6)
ASPECT_NAMES = ("flat", "cubic", "tall")

PASS_OVERALL = 0.85
PASS_PER_CLASS = 0.70
PASS_DROP_RATE = 0.05

# Joint-speed reporting.  SPEED_HEADROOM is the fraction of the safety shell at
# which the ``over_trip`` reward term starts charging, so the share of time
# above it is the share of time that term is doing any work.
SPEED_BINS, SPEED_MAX = 256, 1.6
SPEED_HEADROOM = 0.85


def aspect_class(half: torch.Tensor) -> torch.Tensor:
    """Index into ASPECT_NAMES for each row of a (N, 3) half-extent tensor."""
    ratio = half[:, 2] / half[:, :2].mean(dim=1).clamp(min=1e-6)
    return torch.bucketize(ratio, torch.tensor(ASPECT_CUTS, device=half.device))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _provenance(checkpoint: str) -> dict:
    """Everything needed to say which code and which weights produced a number.

    Recorded per run rather than written down afterwards: a result whose commit
    is remembered rather than stamped is a result that cannot be re-run.
    """
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ("git", *args), cwd=Path(__file__).resolve().parent.parent,
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
        except Exception:
            return ""

    import importlib.metadata as md

    import mujoco

    def version(dist: str) -> str:
        # The installed distribution's version, not a __version__ attribute:
        # neither mjlab nor rsl_rl_lib defines one, and reading the attribute
        # silently records "unknown" for the two libraries whose version
        # matters most here.
        try:
            return md.version(dist)
        except Exception:
            return "unknown"

    prov = {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        # The diff itself, not just "dirty": a number measured against
        # uncommitted code is only interpretable next to that code.
        "git_dirty": git("status", "--porcelain"),
        "git_diff": git("diff"),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mjlab": version("mjlab"),
        "rsl_rl": version("rsl-rl-lib"),
        "mujoco": mujoco.__version__,
        "mujoco_warp": version("mujoco-warp"),
        "warp": version("warp-lang"),
    }
    return prov


FLOOR_DROP_M = 0.75
"""How far the floor is below the table top, for the ``table<R>`` ablation.

Nothing in the simulator knows this number -- the scene is an infinite plane at
z=0 and there is no floor -- which is the whole point of the ablation.
"""


def _table_edge(mode: str, device):
  """Per-pixel fields for a table that stops, or ``None`` if this is not that.

  Returns ``(off, plane, floor)``: which pixels of the nominal view fall past
  the table's edge, what the bare plane reads there, and what a floor
  ``FLOOR_DROP_M`` below it would read instead.  All three are normalised the
  way the observation is, so the substitution is a ``where``.

  Built from the nominal camera pose, and the pose is randomised by 20 mm and
  2 degrees per reset, so the edge lands within about six pixels of where each
  environment would actually see it.  That is fine for the question -- is the
  policy sensitive to what is past the table -- and would not be fine for a
  calibrated number.
  """
  if not mode.startswith("table"):
    return None
  radius = float(mode[len("table"):])
  import numpy as np

  from piper_push import camera as cam

  pos = np.asarray(cam.CAMERA_POS, dtype=np.float64)
  fwd = np.asarray(cam.CAMERA_AIM, dtype=np.float64) - pos
  fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, (0.0, 0.0, 1.0))
  right /= np.linalg.norm(right)
  up = np.cross(right, fwd)
  ty = math.tan(math.radians(cam.FOVY_DEG) / 2.0)
  tx = ty * cam.WIDTH / cam.HEIGHT
  sx = (2.0 * (np.arange(cam.WIDTH) + 0.5) / cam.WIDTH - 1.0) * tx
  sy = (1.0 - 2.0 * (np.arange(cam.HEIGHT) + 0.5) / cam.HEIGHT) * ty
  gx, gy = np.meshgrid(sx, sy)
  d = fwd + gx[..., None] * right + gy[..., None] * up
  d /= np.linalg.norm(d, axis=-1, keepdims=True)

  down = d[..., 2] < -1e-9
  safe = np.where(down, d[..., 2], -1.0)

  def _range(z_plane):
    return np.where(down, (z_plane - pos[2]) / safe, np.inf)

  t_plane = _range(0.0)
  hit = pos + t_plane[..., None] * d
  # A ray that never comes down is past the edge of any table there is.
  off = (~down) | (np.hypot(hit[..., 0], hit[..., 1]) > radius)

  def _norm(t):
    t = np.where(np.isfinite(t), t, cam.CUTOFF_M)
    return np.clip(np.clip(t, 0.05, cam.CUTOFF_M) / cam.CUTOFF_M, 0.0, 1.0)

  to_t = lambda x: torch.as_tensor(x, device=device)
  return (to_t(off), to_t(_norm(t_plane)).float(),
          to_t(_norm(_range(-FLOOR_DROP_M))).float())


def _end_the_table(camera: torch.Tensor, edge) -> torch.Tensor:
  """Replace the plane past the edge with the floor, in the flat observation.

  Only pixels that are reading the bare plane are touched.  Anything nearer is
  the arm swinging out over the edge, and rewriting that would be ablating the
  robot rather than the table.
  """
  off, plane, floor = edge
  # The observation carries the image as ``(N, 3, H, W)``; keep that shape
  # rather than assuming it, so this survives the group being flattened later.
  img = camera.reshape(*camera.shape[:-3], 3, *plane.shape).clone()
  depth = img[..., 0, :, :]
  # Only pixels reading the bare plane are touched.  Anything nearer is the arm
  # swinging out over the edge, and rewriting that would be ablating the robot
  # rather than the table.
  bare = depth >= (plane - 0.02)
  depth = torch.where(bare & off, floor.expand_as(depth), depth)
  img[..., 0, :, :] = depth
  # The third channel is the product of the first two by construction.
  img[..., 2, :, :] = depth * img[..., 1, :, :]
  return img.reshape(camera.shape)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("checkpoint")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--budget", type=float, default=4.0,
                   help="seconds an instance is given before it counts as failed")
    p.add_argument("--drop-grace", type=float, default=1.5,
                   help="seconds after a grasp ends before it counts as dropped")
    p.add_argument("--lift-clear", type=float, default=0.010,
                   help="metres of clearance that make a pad contact a grasp")
    p.add_argument("--camera", default="real",
                   help="ablate the camera channel before the policy sees it.  "
                        "'blank' zeroes it, which is out of distribution and "
                        "tells you little; 'shuffled' hands each environment "
                        "another environment's image -- a real, correctly "
                        "normalised picture of the wrong table -- and a policy "
                        "that scores the same on it is not using the camera.  "
                        "'table<R>', e.g. table0.8, ends the table at R metres "
                        "from the base and drops the floor 0.75 m below it, "
                        "which is the one thing about the real scene the "
                        "simulator's infinite plane cannot represent.")
    # 'clean' is the protocol every number in docs/history/nominal_results_2026-08.md was measured
    # with and stays the default.  Until 2026-09-04 'measured' replaced the
    # task's noise model with the NOMINAL one at strength 1.0, which on a
    # -Robust task is a downgrade, and 'clean' left a -Robust task's sensor
    # on; both are now what their names say (piper_push.evalcfg).
    evalcfg.add_sensor_arg(p)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=None,
                   help="seeds the whole rollout.  Note that this makes a run "
                        "repeatable for a FIXED policy; it does not pair "
                        "scenes across policies, because the single global "
                        "stream is consumed in an order that depends on when "
                        "placements happen, which depends on the policy.")
    p.add_argument("--json", default=None,
                   help="write metrics, per-environment counts and provenance "
                        "here.  Plots are generated from this, never from the "
                        "printed table.")
    p.add_argument("--label", default="",
                   help="name for this run inside the JSON")

    # -- evaluation-time command shaping (Phase 1).  All inert by default; the
    # trained command path is already slew-limited to 0.62 x trip and
    # interpolated across substeps, so "none" here means that path, not a raw
    # policy output.
    g = p.add_argument_group("command shaping")
    g.add_argument("--slew-scale", type=float, default=1.0,
                   help="multiply the trained slew ceiling (1.0 = unchanged)")
    g.add_argument("--accel-limit", type=float, default=None,
                   help="ceiling on commanded joint acceleration, rad/s^2")
    g.add_argument("--lowpass-hz", type=float, default=None,
                   help="first-order low-pass on the joint target, Hz")
    g.add_argument("--interp", default="linear", choices=("linear", "cubic"),
                   help="within-control-step command ramp shape")

    # -- randomisation cadence (Phase 2).  The ranges never change; only how
    # long a drawn value is held.
    c = p.add_argument_group("randomisation cadence")
    c.add_argument("--cadence", default=None,
                   help="what is redrawn when an object is replaced: "
                        "'object' (all of it, the trained default), 'episode' "
                        "(nothing -- one object per episode, re-posed), or a "
                        "comma-separated subset of shape,mass,friction. "
                        "Unset leaves the task's own setting alone.")
    c.add_argument("--reset-hidden-on-respawn", action="store_true",
                   help="zero the recurrent state every time an object is "
                        "replaced, not just at the episode boundary.  The "
                        "control for 'is the memory carrying anything across "
                        "objects'.")
    from piper_push import evalcfg as _evalcfg  # noqa: E402
    _evalcfg.add_action_api_arg(p)
    a = p.parse_args()

    _evalcfg.apply_action_api_arg(a)
    env_cfg = load_env_cfg(a.task, play=True)
    # ``play`` turns the sensor model off so that two recordings of the same
    # policy can be compared; an evaluation that is asking what the robot
    # will do has to turn it back on -- to the sensor the task trains with.
    sensor_prov = evalcfg.apply_sensor(env_cfg, a.task, a.sensor)
    agent_cfg = load_rl_cfg(a.task)
    env_cfg.scene.num_envs = a.num_envs
    if a.seed is not None:
        env_cfg.seed = a.seed
    # Only the arm.  The gripper's rate limit is a grasp parameter, not a
    # safety-shell one, and shaping it would change what the policy can do
    # rather than how smoothly it does it.
    arm_action = env_cfg.actions["arm"]
    arm_action.slew_scale = a.slew_scale
    arm_action.accel_limit = a.accel_limit
    arm_action.lowpass_hz = a.lowpass_hz
    arm_action.interp = a.interp

    if a.cadence is not None:
        from piper_push.shapes import ALL_QUANTITIES
        if a.cadence == "object":
            redraw = ALL_QUANTITIES
        elif a.cadence == "episode":
            redraw = ()
        else:
            redraw = tuple(s.strip() for s in a.cadence.split(",") if s.strip())
            unknown = set(redraw) - set(ALL_QUANTITIES)
            if unknown:
                p.error(f"--cadence: unknown {sorted(unknown)}, "
                        f"expected a subset of {ALL_QUANTITIES}")
        cmd_cfg = env_cfg.commands["pick"]
        cmd_cfg.redraw_on_place = redraw
        # The boolean alias would otherwise still say "all of them" and win
        # nothing -- but leaving it true while asking for () is a contradiction
        # the property cannot see, so it is settled here.
        cmd_cfg.reshape_on_place = bool(redraw)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(a.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=a.device)
    # Straight into the evaluated network, then read back: ``runner.load``
    # with ``load_cfg={"actor": True}`` loads nothing on a -Distill* task.
    loaded = evalcfg.load_weights(runner, a.checkpoint, a.device)
    policy = runner.get_inference_policy(device=a.device)

    u = env.unwrapped
    pick = u.command_manager.get_term("pick")
    robot = u.scene["robot"]
    dt = u.step_dt
    dev = torch.device(a.device)
    n = a.num_envs

    jids, jnames = robot.find_joints([f"joint{i}" for i in range(1, 7)],
                                     preserve_order=True)
    trip = torch.tensor([JOINT_TRIP_RAD_S[j] for j in jnames], device=dev)

    # A recurrent policy carries its hidden state across whatever you feed it,
    # and nothing in ``get_inference_policy`` knows about episode boundaries.
    # Left alone, every environment starts each episode remembering the last
    # one, which is a state it will never be in on the robot.
    recurrent = bool(getattr(policy, "is_recurrent", False))
    if recurrent:
        policy.reset()

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Per-instance state.
    age = torch.zeros(n, device=dev)
    cls_now = aspect_class(pick.object_half_size.to(dev))
    # Per-grasp state.
    holding = torch.zeros(n, dtype=torch.bool, device=dev)
    lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    pending = torch.zeros(n, dtype=torch.bool, device=dev)   # grasp awaiting verdict
    pending_age = torch.zeros(n, device=dev)
    pending_cls = torch.zeros(n, dtype=torch.long, device=dev)

    n_cls = len(ASPECT_NAMES)
    ok = torch.zeros(n_cls, device=dev)
    fail = torch.zeros(n_cls, device=dev)
    grasps = torch.zeros(n_cls, device=dev)
    drops = torch.zeros(n_cls, device=dev)
    placed_total = torch.zeros((), device=dev)
    cycle_times: list[torch.Tensor] = []
    cycle_envs: list[torch.Tensor] = []
    stuck_steps = torch.zeros((), device=dev)
    # The same quantities kept per environment, which the aggregates above
    # cannot be recovered from.  A confidence interval on any of these has to
    # resample whole environments -- control steps inside one environment are
    # about as independent as consecutive frames of a video -- so the
    # per-environment vector is the unit the bootstrap needs and the aggregate
    # is not.
    per_env = {
        k: torch.zeros(n, device=dev)
        for k in ("placed", "ok", "fail", "grasps", "drops", "stuck", "trips",
                  "clears", "strays")
    }

    # -- Phase 1 instrumentation ------------------------------------------
    # The commanded derivatives, kept as histograms so any quantile and any
    # CVaR falls out of one scatter-add per step.  Peaks alone cannot answer
    # "how bad is the tail", which is the only interesting question about a
    # safety limit.
    arm_term = u.action_manager.get_term("arm")
    CMD_BINS = 256
    cmd_scale = {  # per-quantity histogram ceiling, in the quantity's own units
        "cmd_vel": 1.6,      # normalised by the trip speed
        "cmd_acc": 400.0,    # rad/s^2
        "cmd_jerk": 40000.0,  # rad/s^3
    }
    cmd_hist = {k: torch.zeros(len(jids), CMD_BINS, device=dev) for k in cmd_scale}
    cmd_peak = {k: torch.zeros(len(jids), device=dev) for k in cmd_scale}
    # Which part of the cycle a trip happened in.  Read one step late on
    # purpose: env.step() has already reset the tripped environments, so the
    # phase at the moment of the trip is the phase recorded before the step.
    PHASES = ("reach", "close", "carry", "release", "idle")
    phase_steps = torch.zeros(len(PHASES), device=dev)
    phase_trips = torch.zeros(len(PHASES), device=dev)
    prev_phase = torch.full((n,), PHASES.index("idle"), dtype=torch.long, device=dev)

    def _hist_add(name: str, value: torch.Tensor) -> None:
        """value is (B, J) in the quantity's own units."""
        v = value.abs()
        cmd_peak[name] = torch.maximum(cmd_peak[name], v.max(dim=0).values)
        idx = (v * (CMD_BINS / cmd_scale[name])).long().clamp(
            0, CMD_BINS - 1).t().contiguous()
        cmd_hist[name].scatter_add_(
            1, idx, torch.ones_like(idx, dtype=cmd_hist[name].dtype))
    peak_ratio = torch.zeros(len(jids), device=dev)
    # A peak over a third of a million samples is an extreme-value statistic:
    # one environment touching the shell for one step reads exactly the same as
    # a policy that lives there.  What the safety shell actually asks is how
    # much of the time the arm is near it, so keep the whole distribution.  A
    # histogram answers that and any quantile for one scatter-add per step.
    speed_hist = torch.zeros(len(jids), SPEED_BINS, device=dev)
    # The safety shell stops the run on hardware, so the rate it fires at is a
    # deployment number, not a diagnostic.  The gate was absorbing it silently:
    # a shell trip ends the episode, and an instance cut short by the end of an
    # episode is censored rather than failed.
    trips = torch.zeros((), device=dev)
    # Accumulated per step, not read at the end: these are per-episode counters
    # that the reset zeroes, so a single read at the end returns whatever the
    # last partial episode happened to hold and then gets divided by the whole
    # rollout's duration.  It read 186.8 tables an arm-hour against a
    # throughput of 38.4 objects a minute, which cannot both be true.
    clears_total = torch.zeros((), device=dev)
    strays_total = torch.zeros((), device=dev)
    prev_clears = torch.zeros(n, device=dev)
    prev_strays = torch.zeros(n, device=dev)
    has_cleanup = "table_clears" in pick.metrics
    sim_seconds = 0.0

    # The camera ablation, applied to the observation the policy is about to
    # read.  Rolled by one so no environment can be handed back its own image.
    cam_perm = torch.randperm(n, device=dev).roll(1)
    edge = _table_edge(a.camera, dev)

    def seen(o):
        if a.camera == "real":
            return o
        o = o.clone()
        if edge is not None:
            o["camera"] = _end_the_table(o["camera"], edge)
        else:
            o["camera"] = (torch.zeros_like(o["camera"]) if a.camera == "blank"
                           else o["camera"][cam_perm])
        return o

    with torch.inference_mode():
        for _ in range(a.steps):
            out = env.step(policy(seen(obs)))
            obs, dones = out[0], out[2]
            if recurrent:
                policy.reset(dones)
            sim_seconds += dt * n

            half = pick.object_half_size.to(dev)
            # Through the command, not off an entity: which object is being
            # scored is the command's answer, and with a table of them there
            # is no single entity that means "the object".
            pos = pick.target_pos_w() - u.scene.env_origins
            clear = pos[:, 2] - half[:, 2]
            both = (pick.pad_found > 0).all(dim=1)
            just_placed = pick.just_placed.bool()

            # Zero the memory at the object boundary as well as the episode
            # boundary.  This is the control that separates "the recurrent
            # state is integrating evidence about the object in the hand" from
            # "the recurrent state is carrying facts about the PREVIOUS object
            # that happen to still be true".  Only the second survives being
            # cleared here, and only the second is an artefact.
            if recurrent and a.reset_hidden_on_respawn and bool(just_placed.any()):
                policy.reset(just_placed)

            # The command the servo was handed this step, which is what a
            # deterministic filter can change; the measured speed below is what
            # the plant did with it, which is what the shell reacts to.
            _hist_add("cmd_vel", arm_term.cmd_vel / trip)
            _hist_add("cmd_acc", arm_term.cmd_acc)
            _hist_add("cmd_jerk", arm_term.cmd_jerk)

            ratio = robot.data.joint_vel[:, jids].abs() / trip
            peak_ratio = torch.maximum(peak_ratio, ratio.max(dim=0).values)
            bin_idx = (ratio * (SPEED_BINS / SPEED_MAX)).long().clamp(
                0, SPEED_BINS - 1).t().contiguous()
            speed_hist.scatter_add_(
                1, bin_idx, torch.ones_like(bin_idx, dtype=speed_hist.dtype))

            # --- instance outcome -------------------------------------------
            # Exactly one verdict per spawned object.  Scoring every elapsed
            # budget instead would count a stubborn object once per budget and
            # an easy one once per cycle, so the ratio would track how fast the
            # successes arrive rather than how many objects were dealt with.
            age += dt
            placed_total += just_placed.sum()
            won = just_placed & (age <= a.budget)
            lost = just_placed & (age > a.budget)
            ok.index_add_(0, cls_now[won], won[won].float())
            fail.index_add_(0, cls_now[lost], lost[lost].float())
            per_env["placed"] += just_placed.float()
            per_env["ok"] += won.float()
            per_env["fail"] += lost.float()
            if just_placed.any():
                cycle_times.append(age[just_placed].clone())
                cycle_envs.append(just_placed.nonzero().flatten().clone())
            # Per-instance scoring under-weights a stuck object, which
            # blocks its env and so presents fewer instances than a
            # healthy one.  This counts the time instead, so a policy that
            # jams on one shape cannot hide behind the throughput of the
            # envs that did not.
            stuck_steps += (age > a.budget).sum()
            per_env["stuck"] += (age > a.budget).float()
            age[just_placed] = 0.0

            # --- grasp outcome ----------------------------------------------
            # The verdict has to be read before the flag is cleared: the step
            # the pads leave the object is the same step ``both`` goes false,
            # so folding the clearance update in first erases the very grasp
            # being closed out.
            was_lifted = lifted
            lifted = (lifted | (clear > a.lift_clear)) & both
            new_grasp = holding & ~both & was_lifted
            g = new_grasp.nonzero(as_tuple=False).flatten()
            if g.numel():
                grasps.index_add_(0, cls_now[g], torch.ones(g.numel(), device=dev))
                per_env["grasps"] += new_grasp.float()
                pending[g] = True
                pending_age[g] = 0.0
                pending_cls[g] = cls_now[g]

            pending_age = torch.where(pending, pending_age + dt, pending_age)
            # A placement or a re-grasp within the grace window clears the
            # verdict; anything else is the object lying somewhere it should
            # not be.
            cleared = pending & (just_placed | both)
            pending[cleared] = False
            dropped = pending & (pending_age >= a.drop_grace)
            d = dropped.nonzero(as_tuple=False).flatten()
            if d.numel():
                drops.index_add_(0, pending_cls[d], torch.ones(d.numel(), device=dev))
                per_env["drops"] += dropped.float()
            pending[dropped] = False

            # Episode reset re-rolls the shape and ends whatever instance was
            # in flight.  An instance that had its whole budget and was still
            # on the table is a failure; one cut short by the horizon never had
            # the chance, so it is censored rather than counted.
            over_speed = u.termination_manager.get_term("over_speed")
            trips += over_speed.sum()
            per_env["trips"] += over_speed.float()
            # Attribute the trip to the phase the arm was in one step earlier:
            # env.step() has already reset the tripped environments, so reading
            # the phase now describes the fresh episode, not the trip.
            phase_trips.index_add_(
                0, prev_phase[over_speed],
                torch.ones(int(over_speed.sum()), device=dev))
            phase_now = torch.full((n,), PHASES.index("idle"),
                                   dtype=torch.long, device=dev)
            off_table = clear > a.lift_clear
            phase_now = torch.where(~both & ~off_table,
                                    torch.tensor(PHASES.index("reach"), device=dev),
                                    phase_now)
            phase_now = torch.where(both & ~off_table,
                                    torch.tensor(PHASES.index("close"), device=dev),
                                    phase_now)
            phase_now = torch.where(both & off_table,
                                    torch.tensor(PHASES.index("carry"), device=dev),
                                    phase_now)
            phase_now = torch.where(~both & off_table,
                                    torch.tensor(PHASES.index("release"), device=dev),
                                    phase_now)
            phase_steps.index_add_(0, phase_now, torch.ones(n, device=dev))
            prev_phase = phase_now
            if has_cleanup:
                # clamp(min=0) because a reset drops the counter to zero, and a
                # negative delta is that reset rather than un-cleared tables.
                d_clear = (pick.table_clears - prev_clears).clamp(min=0)
                d_stray = (pick.objects_strayed - prev_strays).clamp(min=0)
                clears_total += d_clear.sum()
                strays_total += d_stray.sum()
                per_env["clears"] += d_clear
                per_env["strays"] += d_stray
                prev_clears = pick.table_clears.clone()
                prev_strays = pick.objects_strayed.clone()

            e = dones.nonzero(as_tuple=False).flatten()
            if e.numel():
                stale = e[age[e] > a.budget]
                if stale.numel():
                    fail.index_add_(0, cls_now[stale],
                                    torch.ones(stale.numel(), device=dev))
                    per_env["fail"].index_add_(
                        0, stale, torch.ones(stale.numel(), device=dev))
                age[e] = 0.0
                pending[e] = False
                lifted[e] = False
            holding = both

            # The shape class an instance is attributed to, refreshed at the
            # END of every step rather than only on an episode boundary.
            #
            # It used to be refreshed only on `dones`.  That was correct while
            # an episode held one object, and became wrong at 6d1d0a9 when the
            # command started redrawing the object at every placement: every
            # instance after the first in an episode was scored against the
            # shape class of whatever the episode STARTED with.  Overall
            # success, throughput, drop rate and the trip counts are unaffected
            # -- they sum across classes -- but the per-shape breakdown was
            # attributing outcomes to the wrong rows.
            #
            # End of the step, not the start: on a placement step the object
            # has already been replaced by the time `half` is read, so the
            # value used to score the instance that just finished has to be the
            # one captured on the previous step, which is what this is.
            cls_now = aspect_class(half)

    # ---------------------------------------------------------------- report
    total_ok, total_fail = ok.sum(), fail.sum()
    overall = (total_ok / (total_ok + total_fail).clamp(min=1)).item()
    drop_rate = (drops.sum() / grasps.sum().clamp(min=1)).item()
    per_min = (placed_total / max(sim_seconds, 1e-6) * 60.0).item()

    print()
    print(f"  task        {a.task}")
    print(f"  checkpoint  {a.checkpoint}")
    print(f"  rollout     {n} envs x {a.steps} steps = {sim_seconds / 60:.1f} sim-minutes")
    print()
    print(f"  {'shape':8s} {'instances':>10s} {'success':>9s} {'grasps':>8s} {'drops':>8s}")
    class_pass = True
    for i, name in enumerate(ASPECT_NAMES):
        tot = (ok[i] + fail[i]).item()
        sr = (ok[i] / max(tot, 1)).item()
        dr = (drops[i] / grasps[i].clamp(min=1)).item()
        flag = "" if (sr >= PASS_PER_CLASS or tot == 0) else "   <-- below 70%"
        class_pass &= (sr >= PASS_PER_CLASS) or tot == 0
        print(f"  {name:8s} {tot:10.0f} {100 * sr:8.1f}% {grasps[i]:8.0f}"
              f" {100 * dr:7.1f}%{flag}")
    print()
    print(f"  overall success        {100 * overall:5.1f}%   (gate {100 * PASS_OVERALL:.0f}%)")
    print(f"  post-grasp drop rate   {100 * drop_rate:5.1f}%   (gate {100 * PASS_DROP_RATE:.0f}%)")
    print(f"  throughput             {per_min:5.1f} objects/min per arm")
    if cycle_times:
        c = torch.cat(cycle_times).float()
        q = torch.quantile(c, torch.tensor([0.5, 0.95], device=c.device))
        print(f"  time to place          {q[0]:5.2f} s median, {q[1]:5.2f} s p95")
    stuck = (stuck_steps * dt / max(sim_seconds, 1e-6)).item()
    print(f"  time with a stuck object {100 * stuck:5.1f}%   (object unplaced past the budget)")
    # Per arm-hour, and per hundred objects.  Not as a share of episodes: a
    # trip *is* an episode ending, so unless the rollout is long enough to
    # contain several full episodes the denominator is only the episodes that
    # ended early, and the ratio says which early ending was most common rather
    # than how often the arm stops.
    per_hour = (trips / max(sim_seconds, 1e-6) * 3600.0).item()
    per_100 = (100 * trips / placed_total.clamp(min=1)).item()
    minutes = 60.0 / per_hour if per_hour > 0 else float("inf")
    # Cleanup only.  An instance is still one object from the moment it becomes
    # the target, so everything above measures the same thing it always did;
    # these two are the questions that only exist once there is a table rather
    # than an object.
    if has_cleanup:
        clears = float(clears_total)
        strayed = float(strays_total)
        hours = sim_seconds / 3600.0
        print(f"  tables cleared         {clears / max(hours, 1e-9):5.1f} per arm-hour"
              f"   ({pick.num_objects} objects each)")
        print(f"  objects batted astray  {strayed / max(hours, 1e-9):5.1f} per arm-hour"
              f"   ({100 * strayed / max(placed_total.item(), 1):.1f} per 100 placed)")
    print(f"  safety-shell trips     {per_hour:5.1f} per arm-hour"
          f"   (one every {minutes:.1f} min, {per_100:.1f} per 100 placed)")
    print()
    # The distribution, not just its maximum.  ``over`` is the share of samples
    # above the point the speed penalty starts charging, which is the number
    # that says whether the penalty is doing anything.
    cum = speed_hist.cumsum(dim=1)
    tot_samples = cum[:, -1:].clamp(min=1)
    width = SPEED_MAX / SPEED_BINS
    edges = (torch.arange(SPEED_BINS, device=dev) + 1) * width

    def quantile(p: float) -> torch.Tensor:
        return edges[(cum >= tot_samples * p).float().argmax(dim=1)]

    p99, p999 = quantile(0.99), quantile(0.999)
    first_over = int(SPEED_HEADROOM / width)
    over = speed_hist[:, first_over:].sum(dim=1) / tot_samples.squeeze(1)

    def cvar(hist: torch.Tensor, hi: float, p: float) -> torch.Tensor:
        """Mean of the worst (1-p) of the samples, per joint, from a histogram.

        The expected value in the tail rather than its boundary: p99 says where
        the tail starts and says nothing about how far past it the worst
        excursions go, which for a limit that stops the robot is the part that
        matters.
        """
        bins = hist.shape[1]
        w = hi / bins
        centres = (torch.arange(bins, device=hist.device) + 0.5) * w
        total = hist.sum(dim=1, keepdim=True).clamp(min=1)
        cum = hist.cumsum(dim=1)
        # Everything strictly above the p-quantile bin, plus the part of that
        # bin which lies in the tail, so the answer is continuous in p.
        start = (cum >= total * p).float().argmax(dim=1)
        rows = torch.arange(hist.shape[0], device=hist.device)
        below = torch.where(
            torch.arange(bins, device=hist.device)[None, :] > start[:, None],
            hist, torch.zeros_like(hist))
        partial = (cum[rows, start] - total.squeeze(1) * p).clamp(min=0)
        mass = below.sum(dim=1) + partial
        weighted = (below * centres).sum(dim=1) + partial * centres[start]
        return weighted / mass.clamp(min=1e-9)

    def _hist_quantile(hist: torch.Tensor, hi: float, p: float) -> torch.Tensor:
        bins = hist.shape[1]
        e = (torch.arange(bins, device=hist.device) + 1) * (hi / bins)
        c = hist.cumsum(dim=1)
        return e[(c >= c[:, -1:].clamp(min=1) * p).float().argmax(dim=1)]

    speed_cvar95 = cvar(speed_hist, SPEED_MAX, 0.95)
    speed_cvar99 = cvar(speed_hist, SPEED_MAX, 0.99)
    print("  |qd| / safety-shell trip")
    print(f"    {'joint':8s} {'p99':>6s} {'p99.9':>7s} {'peak':>6s}"
          f" {'time>' + f'{SPEED_HEADROOM:.2f}':>10s}")
    for i, name in enumerate(jnames):
        flag = "   <-- lives at the shell" if over[i] > 0.01 else ""
        print(f"    {name:8s} {p99[i]:6.3f} {p999[i]:7.3f} {peak_ratio[i]:6.3f}"
              f" {100 * over[i]:9.2f}%{flag}")

    shaped = (a.slew_scale != 1.0 or a.accel_limit is not None
              or a.lowpass_hz is not None or a.interp != "linear")
    elements = max(float(arm_term.stats["elements"]), 1.0)
    print()
    print(f"  command shaping        slew x{a.slew_scale:g}"
          f"  accel {a.accel_limit}  lowpass {a.lowpass_hz}  {a.interp}"
          f"{'' if shaped else '   (as trained)'}")
    print(f"    commands clipped by the slew ceiling  "
          f"{100 * float(arm_term.stats['clipped_slew']) / elements:5.2f}%")
    if a.accel_limit is not None:
        print(f"    commands clipped by the accel ceiling "
              f"{100 * float(arm_term.stats['clipped_accel']) / elements:5.2f}%")
    if a.lowpass_hz is not None:
        print(f"    commands moved by the low-pass        "
              f"{100 * float(arm_term.stats['moved_lowpass']) / elements:5.2f}%")
    print(f"    worst-joint CVaR95 |qd|/trip  {speed_cvar95.max():.3f}"
          f"    CVaR99 {speed_cvar99.max():.3f}")
    print()
    print(f"  {'phase':9s} {'% of time':>10s} {'trips':>7s} {'% of trips':>11s}")
    for i, name in enumerate(PHASES):
        share = phase_steps[i] / phase_steps.sum().clamp(min=1)
        t = float(phase_trips[i])
        pct = 100 * t / max(float(trips), 1.0)
        print(f"  {name:9s} {100 * share:9.1f}% {t:7.0f} {pct:10.1f}%")

    verdict = (overall >= PASS_OVERALL and class_pass and drop_rate <= PASS_DROP_RATE)
    print()
    print(f"  S1 {'PASS' if verdict else 'FAIL'}")
    print()

    if a.json:
        hours = sim_seconds / 3600.0
        c = torch.cat(cycle_times).float() if cycle_times else torch.zeros(0)
        ce = torch.cat(cycle_envs).long() if cycle_envs else torch.zeros(0).long()
        out = {
            "label": a.label or Path(a.checkpoint).parent.name,
            "task": a.task,
            "verdict": "PASS" if verdict else "FAIL",
            "config": {
                "num_envs": n, "steps": a.steps, "budget": a.budget,
                "drop_grace": a.drop_grace, "lift_clear": a.lift_clear,
                "seed_requested": a.seed,
                # What the environment ended up on, which is what mjlab
                # writes back into the field.  With no --seed this is null and
                # the run is NOT reproducible; recorded either way so a result
                # cannot be mistaken for a repeatable one.
                "seed_effective": getattr(env_cfg, "seed", None),
                "episode_length_s": env_cfg.episode_length_s,
                "control_dt": dt,
                "num_objects": pick.num_objects,
                # What the command actually did, read off the built term, not
                # what was asked for on the command line.
                "redraw_on_place": list(pick.redraw_on_place),
                "cadence_arg": a.cadence,
                "camera": a.camera,
                "reset_hidden_on_respawn": bool(a.reset_hidden_on_respawn),
                "recurrent": recurrent,
                "sim_seconds": sim_seconds,
                "arm_hours": hours,
            },
            "metrics": {
                "throughput_per_min": per_min,
                "success": overall,
                "drop_rate": drop_rate,
                "stuck_fraction": stuck,
                "trips_per_arm_hour": per_hour,
                "trips_per_100_placed": per_100,
                "trips_total": float(trips),
                "placed_total": float(placed_total),
                "p50_s": float(q[0]) if cycle_times else None,
                "p95_s": float(q[1]) if cycle_times else None,
                "tables_per_arm_hour": (
                    float(clears_total) / max(hours, 1e-9) if has_cleanup else None),
                "strays_per_100_placed": (
                    100 * float(strays_total) / max(float(placed_total), 1.0)
                    if has_cleanup else None),
            },
            "per_class": {
                name: {
                    "instances": float(ok[i] + fail[i]),
                    "success": float(ok[i] / (ok[i] + fail[i]).clamp(min=1)),
                    "grasps": float(grasps[i]),
                    "drop_rate": float(drops[i] / grasps[i].clamp(min=1)),
                }
                for i, name in enumerate(ASPECT_NAMES)
            },
            # The bootstrap unit.  Everything here is a total over one
            # environment for the whole rollout, so resampling these rows with
            # replacement resamples independent arms.
            "per_env": {k: v.cpu().tolist() for k, v in per_env.items()},
            "per_env_seconds": dt * a.steps,
            "cycle_times_s": c.cpu().tolist(),
            "cycle_env_ids": ce.cpu().tolist(),
            "joint_speed": {
                "joints": jnames,
                "p99": p99.cpu().tolist(),
                "p999": p999.cpu().tolist(),
                "peak": peak_ratio.cpu().tolist(),
                "frac_above_headroom": over.cpu().tolist(),
                "cvar95": speed_cvar95.cpu().tolist(),
                "cvar99": speed_cvar99.cpu().tolist(),
                "headroom": SPEED_HEADROOM,
                "hist_bins": SPEED_BINS,
                "hist_max": SPEED_MAX,
                "hist": speed_hist.cpu().tolist(),
            },
            "shaping": {
                "slew_scale": a.slew_scale,
                "accel_limit": a.accel_limit,
                "lowpass_hz": a.lowpass_hz,
                "interp": a.interp,
                "as_trained": not shaped,
                "elements": elements,
                "frac_clipped_slew":
                    float(arm_term.stats["clipped_slew"]) / elements,
                "frac_clipped_accel":
                    float(arm_term.stats["clipped_accel"]) / elements,
                "frac_moved_lowpass":
                    float(arm_term.stats["moved_lowpass"]) / elements,
            },
            "command": {
                k: {
                    "peak": cmd_peak[k].cpu().tolist(),
                    "p99": [float(x) for x in
                            _hist_quantile(cmd_hist[k], cmd_scale[k], 0.99)],
                    "cvar95": cvar(cmd_hist[k], cmd_scale[k], 0.95).cpu().tolist(),
                    "cvar99": cvar(cmd_hist[k], cmd_scale[k], 0.99).cpu().tolist(),
                    "hist_max": cmd_scale[k],
                    "hist_bins": CMD_BINS,
                    "hist": cmd_hist[k].cpu().tolist(),
                }
                for k in cmd_scale
            },
            "phase": {
                "names": list(PHASES),
                "steps": phase_steps.cpu().tolist(),
                "trips": phase_trips.cpu().tolist(),
            },
            "provenance": _provenance(a.checkpoint),
            # The domain the task id does not carry: the import-time knobs
            # and the sensor actually run, plus which state dict was loaded.
            "domain": evalcfg.provenance(sensor=sensor_prov, weights=loaded),
        }
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"  wrote {a.json}")
        print()

    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
