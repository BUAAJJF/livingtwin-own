"""Stage 6 of Phase RA-Sim-1: can the augmented simulator be made to misbehave?

    python scripts/ra_sim1_stress.py --actuator <ckpt> --test S1 S2 S3 S4 S5

Five pre-registered stresses, each at 64 environments in the *augmented*
simulator with terminations **enabled** -- unlike a replay, where they are off
so no candidate is scored on its luckiest survivors.  Here the safety shell is
the measurement, not an obstacle.

| id | what |
|----|------|
| S1 | 3,000-step closed-loop rollout of the frozen vision policy |
| S2 | scripted actions at the action space's full +-1.0 amplitude |
| S3 | direction reversal every two control steps, 12.5 Hz |
| S4 | ramps that drive every joint onto its command limit and hold |
| S5 | half the batch reset asynchronously every 137 steps, the other half never |

Nothing here is deleted for looking bad.  A non-finite state is the result.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
TASK = "Mjlab-Pick-Place-PiperX-Vision"
CKPT = ("logs/rsl_rl/piperx_pick_place_vision/"
        "2026-08-22_17-15-09_f3/model_1500.pt")
SHAPES = {"train": (0.34, 0.22, 0.16, 0.0, 0.0),
          "holdout": (0.0, 0.0, 0.0, 0.16, 0.12)}


def build(num_envs, device, seed, shapes, *, hidden=False, actuator=None,
          residual=None, damping=1.0, plant=None):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  import sys
  sys.path.insert(0, str(ROOT / "src"))
  from piper_push import perturb
  from piper_push.actuator import ActuatorHookCfg, apply_actuator
  from piper_push.hidden_plant import HiddenPlantCfg, apply_hidden_plant
  from piper_push.residual import ResidualHookCfg, apply_residual

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  w = SHAPES[shapes]
  for name, ev in cfg.events.items():
    if name.startswith("object_shape"):
      ev.params = dict(ev.params)
      ev.params["shape_weights"] = tuple(w)
  if damping != 1.0:
    perturb.apply_session_mismatch(
      cfg, perturb.SessionMismatchCfg(servo_damping_scale=damping))
  if hidden:
    apply_hidden_plant(cfg, HiddenPlantCfg())
  if residual:
    apply_residual(cfg, ResidualHookCfg(checkpoint=str(residual)))
  if actuator:
    apply_actuator(cfg, ActuatorHookCfg(checkpoint=str(actuator)))
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  if plant:
    env.action_manager.get_term("arm").set_plant(**plant)
  return env


def scripted(kind, n_env, n_act, steps, device, seed):
  g = torch.Generator().manual_seed(seed)
  t = torch.arange(steps, dtype=torch.float32).view(-1, 1, 1)
  if kind == "S2":                       # full amplitude, slow so it saturates
    ph = torch.rand(1, n_env, n_act, generator=g) * 6.2831853
    return torch.sign(torch.sin(2 * 3.14159265 * 0.01 * t + ph)).to(device)
  if kind == "S3":                       # reverse every two steps
    sq = torch.where((t.view(-1) // 2) % 2 == 0, 1.0, -1.0)
    return (sq.view(-1, 1, 1) * torch.ones(1, n_env, n_act)).to(device)
  if kind == "S4":                       # ramp onto the command limit and hold
    ramp = torch.clamp(t / max(steps * 0.25, 1.0), max=1.0)
    sgn = torch.where(torch.rand(1, n_env, n_act, generator=g) > 0.5, 1.0, -1.0)
    return (ramp * sgn).to(device)
  raise ValueError(kind)


def diagnostics(env, arm, hook, qs, qds, trips, wall, steps, n_env):
  q = torch.stack(qs)
  qd = torch.stack(qds)
  qacc = (qd[1:] - qd[:-1]) / float(env.step_dt)
  out = {
    "steps": steps, "num_envs": n_env,
    "env_steps_per_s": steps * n_env / max(wall, 1e-9),
    "nonfinite_q": int((~torch.isfinite(q)).sum()),
    "nonfinite_qd": int((~torch.isfinite(qd)).sum()),
    "max_abs_q": float(q.abs().max()),
    "max_abs_qd": float(qd.abs().max()),
    "p999_abs_qd": float(qd.abs().flatten().quantile(0.999)),
    "max_abs_qacc": float(qacc.abs().max()),
    "p999_abs_qacc": float(qacc.abs().flatten().quantile(0.999)),
    "safety_shell_events": int(sum(trips)),
    "safety_shell_per_arm_hour": (
      sum(trips) / max(steps * n_env / 50.0 / 3600.0, 1e-9)),
  }
  if hook is not None:
    st = dict(hook.stats)
    st["mean_abs_delta"] = st.pop("sum_abs_delta") / max(st["steps"], 1)
    out["actuator"] = st
    lo, hi = hook.model.cmd_lo, hook.model.cmd_hi
    out["command_range"] = {"lo": lo.tolist(), "hi": hi.tolist()}
  if torch.cuda.is_available():
    out["gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
  return out


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--tests", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
  ap.add_argument("--actuator", default="")
  ap.add_argument("--residual", default="")
  ap.add_argument("--hidden-target", action="store_true")
  ap.add_argument("--damping", type=float, default=1.0)
  ap.add_argument("--latency", type=int, default=0)
  ap.add_argument("--response", type=float, default=1.0)
  ap.add_argument("--deadband", type=float, default=0.0)
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--steps", type=int, default=3000)
  ap.add_argument("--seed", type=int, default=7701)
  ap.add_argument("--shapes", default="holdout", choices=tuple(SHAPES))
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--tag", required=True)
  ap.add_argument("--out", default="results/ra_sim1/stress")
  a = ap.parse_args()

  import sys
  sys.path.insert(0, str(ROOT / "src"))
  plant = {"latency_steps": a.latency, "response_scale": a.response,
           "deadband": a.deadband, "lowpass_hz": None}
  results = {"tag": a.tag, "arm": {
    "actuator": a.actuator or None, "residual": a.residual or None,
    "hidden_target": a.hidden_target, "damping": a.damping, **plant}}

  for test in a.tests:
    env = build(a.num_envs, a.device, a.seed, a.shapes,
                hidden=a.hidden_target, actuator=a.actuator or None,
                residual=a.residual or None, damping=a.damping, plant=plant)
    arm = env.action_manager.get_term("arm")
    hook = next((h for h in arm._hooks
                 if hasattr(h, "model") and hasattr(h, "stats")), None)
    robot = env.scene["robot"]
    jids, _ = robot.find_joints([f"joint{i}" for i in range(1, 7)],
                                preserve_order=True)
    n_act = env.action_manager.total_action_dim
    steps = a.steps if test != "S1" else a.steps
    if test == "S1":
      from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
      from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
      agent = load_rl_cfg(TASK)
      wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
      runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
        wrapped, asdict(agent), device=a.device)
      runner.load(str(ROOT / CKPT), load_cfg={"actor": True}, strict=True,
                  map_location=a.device)
      policy = runner.get_inference_policy(device=a.device)
      policy.reset()
      obs = wrapped.get_observations()
      if isinstance(obs, tuple):
        obs = obs[0]
    else:
      acts = scripted(test, a.num_envs, n_act, steps, a.device, a.seed)
      env.reset()

    qs, qds, trips = [], [], []
    half = torch.arange(0, a.num_envs // 2, device=a.device)
    t0 = time.time()
    with torch.no_grad():
      for t in range(steps):
        if test == "S1":
          act = policy(obs)
          out = wrapped.step(act)
          obs, dones = out[0], out[2]
          policy.reset(dones)
        else:
          env.step(acts[t])
          if test == "S5" and (t + 1) % 137 == 0:
            env.reset(env_ids=half)
        qs.append(robot.data.joint_pos[:, jids].clone().cpu())
        qds.append(robot.data.joint_vel[:, jids].clone().cpu())
        trips.append(int(env.termination_manager.get_term("over_speed").sum()))
    wall = time.time() - t0
    results[test] = diagnostics(env, arm, hook, qs, qds, trips, wall, steps,
                                a.num_envs)
    print(f"  {test}: {json.dumps({k: v for k, v in results[test].items() if k != 'actuator'})}",
          flush=True)
    env.close()

  d = Path(a.out)
  d.mkdir(parents=True, exist_ok=True)
  (d / f"{a.tag}.json").write_text(json.dumps(results, indent=2))
  print(f"wrote {d / (a.tag + '.json')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
