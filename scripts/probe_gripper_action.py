"""Where does the raw gripper action actually live, and what does the policy see?

The gripper term maps a in [-1, 1] onto the jaw's [0, 50 mm] travel and clips
the *target*; nothing clips ``a``.  Every value below -1 commands the same
closed jaw, so a policy that drifts to -14 pays nothing for it (action_rate
and action_acc are differences, and a constant has none), while the same
value is fed back verbatim as the ``actions`` observation term and is the
label the student regresses on.  This prints, for one rollout, the raw
gripper action's quantiles, the fraction of steps outside [-1, 1], and the
same for the ``actions`` observation the policy is given.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push import evalcfg

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", required=True)
p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Robust")
p.add_argument("--num-envs", type=int, default=128)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--seed", type=int, default=101)
p.add_argument("--device", default="cuda:0")
evalcfg.add_sensor_arg(p, default="measured")
p.add_argument("--out", default=None)
a = p.parse_args()

torch.manual_seed(a.seed)
cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = a.num_envs
cfg.seed = a.seed
evalcfg.apply_sensor(cfg, a.task, a.sensor)
env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
agent = load_rl_cfg(a.task)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(wrapped, asdict(agent), None, a.device)
policy = evalcfg.load_policy(runner, a.checkpoint, a.device)
cmd = env.command_manager.get_term("pick")
robot = env.scene["robot"]

env.reset()
obs = wrapped.get_observations()
if isinstance(obs, tuple):
  obs = obs[0]
raw, seen, held, jaw, rate_share = [], [], [], [], []
prev = None
for t in range(a.steps):
  with torch.inference_mode():
    act = policy(obs)
  raw.append(act[:, 6].cpu().numpy().copy())
  if prev is not None:
    d2 = (act - prev).square()
    rate_share.append((d2[:, 6] / d2.sum(dim=1).clamp_min(1e-12)).cpu().numpy().copy())
  prev = act.clone()
  obs, _, dones, _ = wrapped.step(act)
  if getattr(policy, "is_recurrent", False):
    with torch.inference_mode():
      policy.reset(dones)
  # what the policy will see next step: the ``actions`` term of the proprio group
  seen.append(env.observation_manager.compute_group("proprio")[:, -7:][:, 6].cpu().numpy().copy()
              if False else env.action_manager.action[:, 6].cpu().numpy().copy())
  held.append(cmd.grasped.cpu().numpy().copy())
  jaw.append(robot.data.joint_pos[:, 6].cpu().numpy().copy())
raw, seen, held, jaw = map(np.stack, (raw, seen, held, jaw))
q = lambda x: {k: float(np.percentile(x, v)) for k, v in
               (("min", 0), ("p1", 1), ("p10", 10), ("p50", 50), ("p90", 90), ("max", 100))}
out = {
  "checkpoint": a.checkpoint, "task": a.task, "steps": a.steps, "num_envs": a.num_envs,
  "raw_gripper_action": q(raw),
  "frac_below_-1": float((raw < -1).mean()), "frac_above_1": float((raw > 1).mean()),
  "frac_below_-2": float((raw < -2).mean()),
  "while_held": {"frac_below_-1": float((raw[held] < -1).mean()) if held.any() else None,
                 "p50": float(np.median(raw[held])) if held.any() else None},
  "while_free": {"frac_below_-1": float((raw[~held] < -1).mean()),
                 "p50": float(np.median(raw[~held]))},
  "last_window": {"p50": float(np.median(raw[-100:])), "frac_below_-1": float((raw[-100:] < -1).mean()),
                  "jaw_mm_p50": float(np.median(jaw[-100:]) * 1000)},
  "first_window": {"p50": float(np.median(raw[:100])), "frac_below_-1": float((raw[:100] < -1).mean())},
  "obs_actions_term_equals_raw_frac": float(np.isclose(seen, raw).mean()),
  "gripper_share_of_action_rate_l2": {"mean": float(np.mean(rate_share)), "p50": float(np.median(rate_share))},
  "provenance": evalcfg.provenance(sensor=evalcfg.sensor_provenance(cfg, a.sensor)),
}
print(json.dumps(out, indent=1))
if a.out:
  import pathlib
  pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
