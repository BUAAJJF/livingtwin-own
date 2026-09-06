"""Deterministic-policy action statistics: saturation per dimension, step size, jaw, terminations.

The u telemetry in ``piper_push.squashed`` counts rollout *samples*, so a
deterministic evaluation (which never samples) leaves it silent.  This rolls
the deterministic policy out and reports, per action dimension, what the arm
actually received:

    frac_sat99 / frac_sat999   fraction of steps with |tanh(u)| above 0.99 / 0.999
    mean_abs_u, max_abs_u      the pre-squash output
    mean_abs_da                mean |tanh(u_t) - tanh(u_{t-1})|, the realised step
    jaw                        commanded opening (from a = tanh(u_6)) against the measured joint
    terminations               per-cause counts over the rollout, and the fraction of
                               environments that saw neither over_speed nor object_lost

    python scripts/pc/eval_actions.py --checkpoint X --task Mjlab-Pick-Place-PiperX-Robust \
        --num-envs 256 --steps 600 --seed 101 --out actions_s101.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Robust")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=101)
  p.add_argument("--out", default=None)
  from piper_push import evalcfg
  evalcfg.add_sensor_arg(p, default="measured")
  evalcfg.add_action_api_arg(p)
  a = p.parse_args()
  evalcfg.apply_action_api_arg(a)

  import torch
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from eval_occlusion import reset_recurrent
  from piper_push import robot as piper

  torch.manual_seed(a.seed)
  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = a.num_envs
  cfg.seed = a.seed
  sensor_prov = evalcfg.apply_sensor(cfg, a.task, a.sensor)
  bounded = bool(cfg.actions["arm"].bounded)
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  agent = load_rl_cfg(a.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(wrapped, asdict(agent), None, a.device)
  loaded = evalcfg.load_weights(runner, a.checkpoint, a.device)
  policy = runner.get_inference_policy(device=a.device)
  robot = env.scene["robot"]
  tm = env.termination_manager
  causes = [n for n in ("over_speed", "object_lost", "nan", "time_out") if n in tm.active_terms]

  env.reset()
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]
  n, dev = a.num_envs, a.device
  # P2: proposal recall against the true target and the lock's behaviour.
  gowner = getattr(env, "_pc_grasp_owner", None)
  cmd = env.command_manager.get_term("pick")
  recall_hit = recall_n = 0
  no_cand = 0
  dim = None
  sums = None
  prev_a = None
  jaw_cmd, jaw_meas = [], []
  term_counts = {c: 0 for c in causes}
  unsafe = torch.zeros(n, dtype=torch.bool, device=dev)
  gripper_hist = torch.zeros(20, device=dev)
  nonfinite = 0
  for t in range(a.steps):
    with torch.inference_mode():
      u = policy(obs)
    if not torch.isfinite(u).all():
      nonfinite += 1
      u = torch.nan_to_num(u)
    act = torch.tanh(u) if bounded else u
    if sums is None:
      dim = u.shape[-1]
      z = lambda: torch.zeros(dim, dtype=torch.float64, device=dev)
      sums = {k: z() for k in ("sat99", "sat999", "abs_u", "abs_da", "n_da")}
      max_abs_u = z()
    au = u.abs()
    at = act.abs()
    sums["sat99"] += (at > 0.99).sum(0)
    sums["sat999"] += (at > 0.999).sum(0)
    sums["abs_u"] += au.sum(0)
    max_abs_u = torch.maximum(max_abs_u, au.amax(0).double())
    if prev_a is not None:
      sums["abs_da"] += (act - prev_a).abs().sum(0)
      sums["n_da"] += n
    prev_a = act.clone()
    if bounded:
      jaw_c = piper.GRIPPER_OFFSET + piper.GRIPPER_SCALE * act[:, -1]
    else:
      jaw_c = (piper.GRIPPER_OFFSET + piper.GRIPPER_SCALE * act[:, -1]).clamp(0.0, piper.GRIPPER_OPEN_M)
    jaw_cmd.append(jaw_c.mean().item())
    jaw_meas.append(robot.data.joint_pos[:, 6].mean().item())
    gripper_hist += torch.histc(act[:, -1].float(), bins=20, min=-1.0, max=1.0)
    obs, _, dones, _ = wrapped.step(u)
    reset_recurrent(policy, dones)
    cowner = getattr(env, "_pc_cloud_owner", None)
    if gowner is not None and gowner.topk is not None and cowner is not None and cowner.fresh is not None:
      # On fresh frames while nothing is held: is there a candidate within 3 cm
      # (in the plane) of the target object?  That is the proposal recall.
      fresh = cowner.fresh & ~cmd.grasped
      if bool(fresh.any()):
        from piper_push.pc import grasp as _grasp
        objp = cmd._object_pos_local() if hasattr(cmd, "_object_pos_local") else None
        if objp is not None:
          cand = gowner.topk[..., :2]
          valid = gowner.topk[..., _grasp.I_FEASIBLE] > 0.5
          d = (cand - objp[:, None, :2]).norm(dim=-1)
          d = torch.where(valid, d, torch.full_like(d, 9.0))
          hit = (d.amin(dim=1) < 0.03) & fresh
          recall_hit += int(hit.sum()); recall_n += int(fresh.sum())
          no_cand += int((~valid.any(dim=1) & fresh).sum())
    for c in causes:
      flag = tm.get_term(c)
      term_counts[c] += int(flag.sum())
      if c in ("over_speed", "object_lost"):
        unsafe |= flag
  jaw_end = robot.data.joint_pos[:, 6].cpu().numpy()
  env.close()

  steps_total = float(a.steps * n)
  arm_minutes = a.steps * env.step_dt * n / 60.0
  out = {
    "checkpoint": a.checkpoint, "task": a.task, "seed": a.seed, "num_envs": n, "steps": a.steps,
    "bounded": bounded, "action_dim": dim,
    "frac_sat99": (sums["sat99"] / steps_total).tolist(),
    "frac_sat999": (sums["sat999"] / steps_total).tolist(),
    "mean_abs_u": (sums["abs_u"] / steps_total).tolist(),
    "max_abs_u": max_abs_u.tolist(),
    "mean_abs_da": (sums["abs_da"] / sums["n_da"].clamp_min(1)).tolist(),
    "mean_abs_da_all": float((sums["abs_da"].sum() / sums["n_da"].clamp_min(1)[0] / dim)),
    "jaw_cmd_mm_mean": float(np.mean(jaw_cmd) * 1000), "jaw_meas_mm_mean": float(np.mean(jaw_meas) * 1000),
    "jaw_cmd_mm_last100": float(np.mean(jaw_cmd[-100:]) * 1000), "jaw_meas_mm_last100": float(np.mean(jaw_meas[-100:]) * 1000),
    "jaw_end_mm_median": float(np.median(jaw_end) * 1000),
    "jaw_end_mm_p10": float(np.percentile(jaw_end, 10) * 1000),
    "gripper_action_hist20": (gripper_hist / gripper_hist.sum().clamp_min(1)).tolist(),
    "terminations": term_counts,
    "terminations_per_arm_minute": {c: v / arm_minutes for c, v in term_counts.items()},
    "safe_env_fraction": float(1.0 - unsafe.float().mean().item()),
    "safe_env_fraction_def": "environments with no over_speed and no object_lost termination over the whole rollout",
    "nonfinite_action_steps": nonfinite,
    "provenance": evalcfg.provenance(argv=sys.argv, sensor=sensor_prov, weights=loaded),
  }
  if gowner is not None:
    out["p2"] = {
      "proposal_recall_3cm": (recall_hit / recall_n) if recall_n else None,
      "fresh_unheld_frames": recall_n,
      "no_candidate_fraction": (no_cand / recall_n) if recall_n else None,
      "switches_per_env": float(gowner.switches.float().mean()) if gowner.switches is not None else None,
      "no_candidate_steps_per_env": float(gowner.no_candidate_steps.float().mean()) if gowner.no_candidate_steps is not None else None,
      "locked_fraction_end": float(gowner._lock_on.float().mean()) if gowner._lock_on is not None else None,
    }
    print("p2:", out["p2"])
  f = lambda xs: " ".join(f"{x:.3f}" for x in xs)
  print(f"sat>0.99  {f(out['frac_sat99'])}")
  print(f"sat>0.999 {f(out['frac_sat999'])}")
  print(f"mean|u|   {f(out['mean_abs_u'])}   max|u| {f(out['max_abs_u'])}")
  print(f"mean|da|  {f(out['mean_abs_da'])}   all {out['mean_abs_da_all']:.3f}")
  print(f"jaw cmd/meas mm: mean {out['jaw_cmd_mm_mean']:.1f}/{out['jaw_meas_mm_mean']:.1f}  last100 {out['jaw_cmd_mm_last100']:.1f}/{out['jaw_meas_mm_last100']:.1f}  end median {out['jaw_end_mm_median']:.1f} p10 {out['jaw_end_mm_p10']:.1f}")
  print(f"terminations {term_counts}  per arm-min {{{', '.join(f'{c}: {v:.3f}' for c, v in out['terminations_per_arm_minute'].items())}}}  safe env frac {out['safe_env_fraction']:.3f}  nonfinite {nonfinite}")
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
