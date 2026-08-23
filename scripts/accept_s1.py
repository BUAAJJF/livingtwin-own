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
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

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
    p.add_argument("--reshape-on-place", action="store_true",
                   help="draw a new shape for every object rather than one per "
                        "episode; see below")
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
    a = p.parse_args()

    env_cfg = load_env_cfg(a.task, play=True)
    agent_cfg = load_rl_cfg(a.task)
    env_cfg.scene.num_envs = a.num_envs
    if a.seed is not None:
        env_cfg.seed = a.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(a.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=a.device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=a.device)
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

    # The shape event is a reset event, so within an episode every object the
    # policy is handed is the same geometry re-posed.  A recurrent policy can
    # therefore identify the shape once and coast on it for the rest of the
    # episode -- legal here, and worth nothing on a real table where the next
    # object is a different object.  This redraws the shape as each object is
    # replaced, which is what deployment looks like.
    if a.reshape_on_place:
        from mjlab.managers.event_manager import RecomputeLevel

        from piper_push import shapes as _shapes

        term = u.event_manager.get_term_cfg("object_shape")

        def _reshape(ids: torch.Tensor) -> None:
            _shapes.randomize_object_shape(u, ids, **term.params)
            u.sim.recompute_constants(RecomputeLevel.set_const)
            # Re-place after re-shaping, not before: the placement height is
            # computed from the object's half-extent, so a taller object
            # dropped into the old pose starts inside the table.
            pick._place_object(ids)

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

    with torch.inference_mode():
        for _ in range(a.steps):
            out = env.step(policy(obs))
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
            if a.reshape_on_place and just_placed.any():
                _reshape(just_placed.nonzero(as_tuple=False).flatten())

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
                cls_now = aspect_class(pick.object_half_size.to(dev))
            holding = both

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
    print("  |qd| / safety-shell trip")
    print(f"    {'joint':8s} {'p99':>6s} {'p99.9':>7s} {'peak':>6s}"
          f" {'time>' + f'{SPEED_HEADROOM:.2f}':>10s}")
    for i, name in enumerate(jnames):
        flag = "   <-- lives at the shell" if over[i] > 0.01 else ""
        print(f"    {name:8s} {p99[i]:6.3f} {p999[i]:7.3f} {peak_ratio[i]:6.3f}"
              f" {100 * over[i]:9.2f}%{flag}")

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
                "reshape_on_place_flag": a.reshape_on_place,
                "seed_requested": a.seed,
                # What the environment ended up on, which is what mjlab
                # writes back into the field.  With no --seed this is null and
                # the run is NOT reproducible; recorded either way so a result
                # cannot be mistaken for a repeatable one.
                "seed_effective": getattr(env_cfg, "seed", None),
                "episode_length_s": env_cfg.episode_length_s,
                "control_dt": dt,
                "num_objects": pick.num_objects,
                "reshape_on_place_env": bool(
                    getattr(env_cfg.commands["pick"], "reshape_on_place", False)),
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
                "headroom": SPEED_HEADROOM,
                "hist_bins": SPEED_BINS,
                "hist_max": SPEED_MAX,
                "hist": speed_hist.cpu().tolist(),
            },
            "provenance": _provenance(a.checkpoint),
        }
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"  wrote {a.json}")
        print()

    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
