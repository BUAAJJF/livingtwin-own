"""Does the policy still work at the end of the episode it started?

Every throughput number this project has quoted is a mean over an evaluation
window, and on 2026-09-03 that turned out to hide the largest single defect in
the campaign: the v7 sight teachers stop working part way through a long
episode.  They do not slow down -- environments die one at a time, with the
gripper commanded shut, and never restart.  `strong_teacher` falls from 18
placements a minute to 6 while its mean reads 15.

The mean cannot see this and neither can the reward: total per-step reward
falls 2.52 -> 1.85 across the same window, so it is not an equilibrium the
optimiser chose, it is a hole it left.  Nothing in the training loop is
measured over a horizon long enough to contain it -- `episode_length_s` is 12
in training, 40 in play, and the real arm never resets at all.

So this reports the ratio, not the mean:

    late/early    placements in the second half over the first
    survivors     environments still placing in the last window
    jaw           median gripper opening in the ones that stopped

A healthy policy is late/early ~ 1.0.  Measured this way, `v5_baseline` is
1.08, `strong_teacher` is 0.49, and the ordering across five teachers is
monotone in the sight-reward weight.

    python scripts/eval_endurance.py \\
        --checkpoint checkpoints/v7_teachers/strong_teacher.pt \\
        --steps 1200 --num-envs 256

`--reset-every` is the control: at a period shorter than the training horizon
the decay disappears entirely, which is how "the policy degrades" was told
apart from "the environment gets harder".
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def summarise(placed_by_env, jaw_end, window, step_dt):
  """Turn the per-window, per-environment placement counts into the verdict.

  Kept free of torch and of the simulator so the arithmetic -- which is where
  an off-by-one would silently invert the verdict -- is testable on a laptop.

  Args:
    placed_by_env: ``(W, B)`` placements by window and environment.
    jaw_end: ``(B,)`` final gripper opening in metres.
    window: steps per window.
    step_dt: seconds per control step.
  """
  p = np.asarray(placed_by_env, dtype=np.float64)
  w, b = p.shape
  half = w // 2
  rate = p.sum(axis=1) / b / (step_dt * window) * 60.0
  early = float(rate[:half].mean()) if half else float("nan")
  late = float(rate[half:].mean())
  first, last = p[0], p[-1]
  started = first > 0
  jaw = np.asarray(jaw_end, dtype=np.float64)
  stopped = started & (last == 0)
  return {
    "placed_per_min": rate.tolist(),
    "early_per_min": early,
    "late_per_min": late,
    # A policy that never places anything early has no decay to report and
    # must not be scored 1.0 for it.
    "late_over_early": (late / early) if early > 1e-9 else float("nan"),
    "started_envs": int(started.sum()),
    "stopped_envs": int(stopped.sum()),
    "survivor_fraction": float(1.0 - stopped.sum() / max(started.sum(), 1)),
    "survivor_placements_last": float(last[started & (last > 0)].mean())
                                if (started & (last > 0)).any() else 0.0,
    "survivor_placements_first": float(first[started].mean())
                                 if started.any() else 0.0,
    "jaw_mm_stopped": float(np.median(jaw[stopped]) * 1000) if stopped.any()
                      else float("nan"),
    "jaw_mm_running": float(np.median(jaw[~stopped]) * 1000) if (~stopped).any()
                      else float("nan"),
  }


def main() -> int:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Robust")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--steps", type=int, default=1200)
  p.add_argument("--window", type=int, default=100)
  p.add_argument("--reset-every", type=int, default=0,
                 help="control: reset the environments this often.  0 never, "
                      "which is what the arm does.")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=101)
  p.add_argument("--gate", type=float, default=None,
                 help="exit 1 if late/early is below this")
  p.add_argument("--out", default=None)
  a = p.parse_args()

  import torch
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from eval_occlusion import load_policy, reset_recurrent

  torch.manual_seed(a.seed)
  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = a.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  agent = load_rl_cfg(a.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(
    wrapped, asdict(agent), None, a.device)
  policy = load_policy(runner, a.checkpoint, a.device)
  cmd = env.command_manager.get_term("pick")
  robot = env.scene["robot"]

  env.reset()
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]
  prev = cmd.objects_placed.clone()
  per_window, acc = [], torch.zeros(a.num_envs, device=a.device)
  for t in range(a.steps):
    if a.reset_every and t and t % a.reset_every == 0:
      env.reset()
      obs = wrapped.get_observations()
      if isinstance(obs, tuple):
        obs = obs[0]
      prev = cmd.objects_placed.clone()
    with torch.inference_mode():
      action = policy(obs)
    obs, _, dones, _ = wrapped.step(action)
    reset_recurrent(policy, dones)
    now = cmd.objects_placed
    # ``objects_placed`` is cumulative and zeroed on reset, so the increment is
    # clamped: a reset must read as "no placements this step", not as a large
    # negative one.
    acc += (now - prev).clamp_min(0).float()
    prev = now.clone()
    if (t + 1) % a.window == 0:
      per_window.append(acc.cpu().numpy().copy())
      acc.zero_()
  jaw_end = robot.data.joint_pos[:, 6].cpu().numpy().copy()
  env.close()

  out = summarise(np.stack(per_window), jaw_end, a.window, env.step_dt)
  out.update({"checkpoint": a.checkpoint, "task": a.task, "seed": a.seed,
              "num_envs": a.num_envs, "steps": a.steps,
              "reset_every": a.reset_every})
  print(f"{a.checkpoint}   {a.num_envs} envs x {a.steps} steps  "
        f"reset_every={a.reset_every}")
  print("placed/min " + " ".join(f"{x:6.1f}" for x in out["placed_per_min"]))
  print(f"early {out['early_per_min']:.2f}   late {out['late_per_min']:.2f}   "
        f"late/early {out['late_over_early']:.2f}")
  print(f"of {out['started_envs']} envs placing in window 0, "
        f"{out['stopped_envs']} place nothing in the last window "
        f"({100 * (1 - out['survivor_fraction']):.0f}%)")
  print(f"survivors place {out['survivor_placements_last']:.2f} in the last "
        f"window against {out['survivor_placements_first']:.2f} in the first")
  print(f"final jaw: stopped {out['jaw_mm_stopped']:.1f} mm   "
        f"running {out['jaw_mm_running']:.1f} mm")
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
  if a.gate is not None:
    ok = out["late_over_early"] >= a.gate
    print(f"\nGATE late/early >= {a.gate}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
