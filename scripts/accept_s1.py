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
from dataclasses import asdict

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
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    env_cfg = load_env_cfg(a.task, play=True)
    agent_cfg = load_rl_cfg(a.task)
    env_cfg.scene.num_envs = a.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(a.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=a.device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)

    u = env.unwrapped
    pick = u.command_manager.get_term("pick")
    obj = pick._object
    robot = u.scene["robot"]
    sensor = u.scene.sensors["pad_contact"]
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
    stuck_steps = torch.zeros((), device=dev)
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
    episodes = torch.zeros((), device=dev)
    sim_seconds = 0.0

    with torch.inference_mode():
        for _ in range(a.steps):
            out = env.step(policy(obs))
            obs, dones = out[0], out[2]
            if recurrent:
                policy.reset(dones)
            sim_seconds += dt * n

            half = pick.object_half_size.to(dev)
            pos = obj.data.root_link_pos_w - u.scene.env_origins
            clear = pos[:, 2] - half[:, 2]
            both = (sensor.data.found > 0).all(dim=1)
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
            if just_placed.any():
                cycle_times.append(age[just_placed].clone())
            # Per-instance scoring under-weights a stuck object, which
            # blocks its env and so presents fewer instances than a
            # healthy one.  This counts the time instead, so a policy that
            # jams on one shape cannot hide behind the throughput of the
            # envs that did not.
            stuck_steps += (age > a.budget).sum()
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
            pending[dropped] = False

            # Episode reset re-rolls the shape and ends whatever instance was
            # in flight.  An instance that had its whole budget and was still
            # on the table is a failure; one cut short by the horizon never had
            # the chance, so it is censored rather than counted.
            trips += u.termination_manager.get_term("over_speed").sum()
            episodes += dones.sum()
            e = dones.nonzero(as_tuple=False).flatten()
            if e.numel():
                stale = e[age[e] > a.budget]
                if stale.numel():
                    fail.index_add_(0, cls_now[stale],
                                    torch.ones(stale.numel(), device=dev))
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
    shell = (trips / episodes.clamp(min=1)).item()
    per_hour = (trips / max(sim_seconds, 1e-6) * 3600.0).item()
    print(f"  safety-shell trips     {100 * shell:5.1f}% of episodes"
          f"  ({per_hour:.1f} per arm-hour)")
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
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
