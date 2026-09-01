"""Where in the pick-and-place cycle does the safety shell actually fire?

The gate says how often; this says when.  It matters because the two answers
have opposite fixes: a shell that trips while the arm is reaching is a speed
problem, and a shell that trips at the moment the gripper closes or the object
leaves the hand is an impact problem -- a contact impulse the arm cannot
command away, which no reward weight on joint velocity will remove.

Phases are read off the state rather than the command term, which does not
expose one:

    reach      nothing in the gripper, object on the table
    close      pads in contact, object not yet clear of the table
    carry      pads in contact, object clear
    release    within RELEASE_WINDOW of the pads letting go of a lifted object
    idle       everything else (between objects)

Everything is read one control step *before* the step that trips, and there is
no way around it: the termination fires inside env.step() and the environment
resets in the same call, so by the time step() returns both the velocity that
tripped and the state it tripped in are gone.  Recomputing the trip predicate
afterwards finds nothing at all -- 0 trips against the 16 the gate counts over
the same rollout.  So the trip flag comes from the termination manager, and
the phase and the offending joint come from 20 ms earlier.
"""

import argparse
from collections import Counter
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push.robot import JOINT_TRIP_RAD_S

PHASES = ("reach", "close", "carry", "release", "idle")
RELEASE_WINDOW = 0.30
LIFT_CLEAR = 0.010

p = argparse.ArgumentParser()
p.add_argument("task")
p.add_argument("checkpoint")
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--steps", type=int, default=1800)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()

env_cfg = load_env_cfg(a.task, play=True)
agent_cfg = load_rl_cfg(a.task)
env_cfg.scene.num_envs = a.num_envs
env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), None, a.device)
runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
policy = runner.get_inference_policy(device=a.device)
recurrent = bool(getattr(policy, "is_recurrent", False))
if recurrent:
    policy.reset()

u = env.unwrapped
pick = u.command_manager.get_term("pick")
if pick.num_objects != 1:
    raise RuntimeError("trip_phase currently requires the single-object task")
obj = pick._objects[0]
robot = u.scene["robot"]
pads = u.scene.sensors["pad_contact"]
dt = u.step_dt
dev = torch.device(a.device)
n = a.num_envs

jids, jnames = robot.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)
trip = torch.tensor([JOINT_TRIP_RAD_S[j] for j in jnames], device=dev)

u.reset()
obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]

lifted = torch.zeros(n, dtype=torch.bool, device=dev)
since_release = torch.full((n,), 1e9, device=dev)
phase_steps = torch.zeros(len(PHASES), device=dev)
phase_trips = torch.zeros(len(PHASES), device=dev)
joint_trips = torch.zeros(len(jids), device=dev)
joint_by_phase = torch.zeros(len(PHASES), len(jids), device=dev)
near = torch.zeros((), device=dev)
sim_seconds = 0.0

with torch.inference_mode():
    for _ in range(a.steps):
        # --- the state the arm is in going into this step --------------------
        half = pick.object_half_size.to(dev)
        pos = obj.data.root_link_pos_w - u.scene.env_origins
        clear = pos[:, 2] - half[:, 2] > LIFT_CLEAR
        both = (pads.data.found > 0).all(dim=1)

        released = lifted & ~both
        since_release = torch.where(released, torch.zeros_like(since_release),
                                    since_release + dt)
        lifted = (lifted | clear) & both

        idx = torch.full((n,), PHASES.index("idle"), device=dev, dtype=torch.long)
        idx = torch.where(~both & ~clear, torch.tensor(PHASES.index("reach"), device=dev), idx)
        idx = torch.where(both & ~clear, torch.tensor(PHASES.index("close"), device=dev), idx)
        idx = torch.where(both & clear, torch.tensor(PHASES.index("carry"), device=dev), idx)
        idx = torch.where(since_release < RELEASE_WINDOW,
                          torch.tensor(PHASES.index("release"), device=dev), idx)
        ratio = robot.data.joint_vel[:, jids].abs() / trip
        phase_steps.index_add_(0, idx, torch.ones(n, device=dev))

        # --- step, then ask the environment whether the shell fired ----------
        out = env.step(policy(obs))
        obs, dones = out[0], out[2]
        if recurrent:
            policy.reset(dones)
        sim_seconds += dt * n

        tripped = u.termination_manager.get_term("over_speed").bool()
        if tripped.any():
            phase_trips.index_add_(0, idx[tripped], torch.ones(int(tripped.sum()), device=dev))
            # Which joint: the one closest to its limit going in.  The velocity
            # that actually crossed is unrecoverable, so this is attribution by
            # proximity, and it is only meaningful because the ratios it picks
            # from are already near 1.
            blame = ratio[tripped].argmax(dim=-1)
            one = torch.ones(int(tripped.sum()), device=dev)
            joint_trips.index_add_(0, blame, one)
            flat = idx[tripped] * len(jids) + blame
            joint_by_phase.view(-1).index_add_(0, flat, one)
            near += (ratio[tripped].max(dim=-1).values >= 0.9).sum()

        since_release = torch.where(dones.bool(), torch.full_like(since_release, 1e9), since_release)
        lifted = lifted & ~dones.bool()

total = phase_trips.sum().item()
hours = sim_seconds / 3600.0
print()
print(f"  {a.checkpoint}")
print(f"  {sim_seconds/60:.0f} arm-minutes, {int(total)} shell trips "
      f"({total/max(hours,1e-9):.1f} per arm-hour)")
print()
print(f"  {'phase':9s} {'% of time':>10s} {'trips':>7s} {'% of trips':>11s} {'per arm-hour':>13s}")
for i, name in enumerate(PHASES):
    share = phase_steps[i] / phase_steps.sum().clamp(min=1)
    t = phase_trips[i].item()
    rate = t / max(hours, 1e-9)
    print(f"  {name:9s} {100*share:9.1f}% {t:7.0f} {100*t/max(total,1):10.1f}% {rate:12.1f}")
print()
print(f"  {100 * near / max(total, 1):.0f}% of trips were already above 0.9 of the "
      "limit one step earlier")
print()
print("  trips attributed to the joint nearest its limit going in")
for j, name in enumerate(jnames):
    parts = "  ".join(f"{PHASES[i]}={joint_by_phase[i, j]:.0f}" for i in range(len(PHASES))
                      if joint_by_phase[i, j] > 0)
    print(f"    {name:8s} {joint_trips[j]:7.0f}   {parts}")
