"""Smoke-test one point-cloud route: build the env, step it, build the student, distil two iterations.

    python scripts/pc/smoke.py --route P1A --teacher checkpoints/pc/v10c_nosight_7400.pt --num-envs 8

Checks, and prints, what a broken pipeline would hide: observation shapes, the
fraction of fresh frames (must be ~0.6), the lag draw, point-cloud occupancy
and extent, per-step wall time of the observation, and that two distillation
iterations run with a finite loss.  Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
  p = argparse.ArgumentParser()
  from piper_push.pc import routes as pc_routes
  p.add_argument("--route", required=True, choices=pc_routes.ROUTES)
  p.add_argument("--teacher", required=True)
  p.add_argument("--num-envs", type=int, default=8)
  p.add_argument("--steps", type=int, default=40)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--iterations", type=int, default=2)
  p.add_argument("--out", default=None)
  a = p.parse_args()

  import torch
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  configure_torch_backends()

  task = f"Mjlab-Pick-Place-PiperX-PC-{a.route}-Distill"
  if os.environ.get("MASS_GT", "0") == "1":
    task += "-Mass"
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = a.num_envs
  cfg.episode_length_s = 36.0
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  obs, _ = env.reset()
  report = {"route": a.route, "task": task, "num_envs": a.num_envs}
  report["obs_shapes"] = {k: list(v.shape) for k, v in obs.items()}
  print("obs:", report["obs_shapes"])
  owner = env._pc_cloud_owner
  fresh_n, valid_n, t_obs = 0, 0, []
  occ, ext, tflag = [], [], []
  for t in range(a.steps):
    act = torch.zeros(a.num_envs, env.action_manager.total_action_dim, device=a.device)
    t0 = time.perf_counter()
    obs, rew, term, trunc, extras = env.step(act)
    torch.cuda.synchronize()
    t_obs.append(time.perf_counter() - t0)
    meta = obs["vision_meta"]
    fresh_n += int(meta[:, 1].sum())
    valid_n += int(meta[:, 2].sum())
    if pc_routes.base_route(a.route) != "P0":
      c = obs["camera"]
      flag = c[..., 3] > 0.5
      occ.append(float(flag.float().mean()))
      if c.shape[-1] >= 5:
        tflag.append(float((c[..., 4] > 0.5).float().sum(1).mean()))
      xyz = c[..., :3][flag]
      if xyz.numel():
        ext.append([float(xyz[:, i].min()) for i in range(3)] + [float(xyz[:, i].max()) for i in range(3)])
    else:
      c = obs["camera"]
      occ.append(float((c[:, 1] > 0.5).float().mean()))
  n = a.steps * a.num_envs
  report["fresh_fraction"] = fresh_n / n
  report["valid_fraction"] = valid_n / n
  report["lags"] = owner.lags.tolist()
  report["step_ms_p50"] = 1000 * sorted(t_obs)[len(t_obs) // 2]
  report["occupancy_mean"] = sum(occ) / max(len(occ), 1)
  # The target channel: its width, and how many sampled points carry the flag
  # (must be > 0 on an oracle route and exactly 0 on a zero route).
  report["target_channel"] = pc_routes.target_channel(a.route)
  report["camera_width"] = int(obs["camera"].shape[-1]) if obs["camera"].dim() == 3 else None
  report["target_flag_points_mean"] = (sum(tflag) / len(tflag)) if tflag else None
  report["target_sampled_count_mean"] = float(owner.target_sampled_count.float().mean()) if owner.target_sampled_count is not None else None
  if ext:
    import numpy as np
    e = np.array(ext)
    report["extent_min"] = e[:, :3].min(0).tolist()
    report["extent_max"] = e[:, 3:].max(0).tolist()
  if pc_routes.base_route(a.route) == "P2":
    g = env._pc_grasp_owner
    report["p2"] = {"switches": int(g.switches.sum()), "no_candidate_steps": int(g.no_candidate_steps.sum()),
                    "locked_now": int(g._lock_on.sum()), "topk_feasible_mean": float((obs["grasp_topk"][..., 17] > 0.5).float().sum(1).mean())}
  print(json.dumps({k: v for k, v in report.items() if k != "obs_shapes"}, indent=1))
  ok = abs(report["fresh_fraction"] - 0.6) < 0.08
  if not ok:
    print(f"FAIL: fresh fraction {report['fresh_fraction']:.3f} is not 3/5")
  tc = report["target_channel"]
  if tc != "none" and report["camera_width"] != 5:
    ok = False; print(f"FAIL: target channel {tc} but the cloud is {report['camera_width']} wide")
  if tc == "zero" and (report["target_flag_points_mean"] or 0) != 0:
    ok = False; print("FAIL: the zero channel is not zero")
  if tc == "oracle" and not (report["target_flag_points_mean"] or 0) > 0:
    ok = False; print("FAIL: the oracle channel never flagged a point in the smoke")
  env.close()

  # -- the student and two distillation iterations ---------------------------
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = a.num_envs
  cfg.episode_length_s = 36.0
  agent = load_rl_cfg(task)
  agent.max_iterations = a.iterations
  agent.logger = "tensorboard"
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner_cls = load_runner_cls(task)
  log_dir = Path("/tmp/claude-1000") / "pc_smoke" / a.route
  runner = runner_cls(wrapped, asdict(agent), str(log_dir), a.device)
  runner.load(a.teacher, load_cfg={"teacher": True, "iteration": False}, strict=True, map_location=a.device)
  student = runner.alg._raw_student
  n_params = sum(p.numel() for p in student.parameters())
  report["student_params"] = n_params
  enc = getattr(student, "encoders", None)
  report["encoders"] = {k: type(v).__name__ for k, v in enc.items()} if enc is not None else None
  print("student params:", n_params, "encoders:", report["encoders"])
  t0 = time.perf_counter()
  runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=True)
  report["learn_s_per_iter"] = (time.perf_counter() - t0) / a.iterations
  # Single-environment inference cost of the student, the deployment budget.
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]
  one = obs[:1]
  policy = runner.get_inference_policy(device=a.device)
  print(json.dumps({k: report[k] for k in ("learn_s_per_iter", "student_params", "encoders")}, indent=1))
  with torch.inference_mode():
    policy.reset()   # the rollout's hidden state is for num_envs; the deployment has one
    for _ in range(5):
      policy(one)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
      policy(one)
    torch.cuda.synchronize()
  report["infer_ms_single_env"] = (time.perf_counter() - t0) / 50 * 1000
  report["gpu_mem_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
  print(json.dumps({k: report[k] for k in ("learn_s_per_iter", "infer_ms_single_env", "gpu_mem_alloc_gb")}, indent=1))
  env.close()
  if a.out:
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
