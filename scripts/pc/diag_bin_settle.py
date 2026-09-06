"""Which clause of the placement test fails while an object sits in the bin unregistered?

Found on 2026-09-06 with the first-generation P1B student: a quarter of the
environments spend seconds with the object inside the bin footprint, released
and not grasped, without the placement registering.  ``PickCommand`` scores a
placement only when the object is inside the footprint less a margin, below
the rim, released, SETTLED (|v| < place_settle_vel) and the grasp was paid;
this rollout counts, on every such step, which clause fails, and for the
unsettled steps records the object's speed, spin, shape class and the hand's
distance -- so that "the object is rolling around in the bin on its own" can
be told from "the hand is still poking it".

    python scripts/pc/diag_bin_settle.py --checkpoint checkpoints/pc/pc_final_P1B_model_799.pt \\
        --task Mjlab-Pick-Place-PiperX-PC-P1B-Vision --out results/pc/gen2/audit/bin_settle_gen1_P1B_s101.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-PC-P1B-Vision")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--steps", type=int, default=1800)
  p.add_argument("--seed", type=int, default=101)
  p.add_argument("--device", default="cuda:0")
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
  from piper_push import objects, shapes

  torch.manual_seed(a.seed)
  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = a.num_envs
  cfg.seed = a.seed
  sensor_prov = evalcfg.apply_sensor(cfg, a.task, a.sensor)
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  agent = load_rl_cfg(a.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(wrapped, asdict(agent), None, a.device)
  loaded = evalcfg.load_weights(runner, a.checkpoint, a.device)
  policy = runner.get_inference_policy(device=a.device)
  cmd = env.command_manager.get_term("pick")
  dev = a.device
  inner = torch.tensor(cmd.cfg.bin_inner, device=dev)
  centre = torch.tensor(cmd.cfg.bin_center, device=dev)
  env.reset()
  obs = wrapped.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  n = a.num_envs
  counts = {k: 0 for k in ("in_footprint_not_grasped", "fail_inside_margin", "fail_below_rim", "fail_released",
                           "fail_settled", "fail_grasp_paid", "knock_counting", "all_pass")}
  steps_in = torch.zeros(n, device=dev)
  sp, dd, av, cl, hh, allc, bot = [], [], [], [], [], [], []
  far = pad = 0
  with torch.inference_mode():
    for _ in range(a.steps):
      u = policy(obs)
      obj = cmd._object_pos_local()
      half = cmd.object_half_size
      delta = (obj[:, :2] - centre).abs()
      foot = (delta < inner).all(-1) & ~cmd.grasped
      allc.append(shapes.object_shape_class(env).cpu())
      if foot.any():
        inside = (delta < (inner - cmd.cfg.place_margin_m)).all(-1)
        below = obj[:, 2] - half[:, 2] < cmd.cfg.bin_rim_z - 0.005
        found = cmd.pad_found
        released = ~(found > 0).any(-1)
        speed = torch.linalg.norm(cmd._target_lin_vel_w(), dim=-1)
        settled = speed < cmd.cfg.place_settle_vel
        paid = cmd._grasp_paid
        counts["in_footprint_not_grasped"] += int(foot.sum())
        counts["fail_inside_margin"] += int((foot & ~inside).sum())
        counts["fail_below_rim"] += int((foot & ~below).sum())
        counts["fail_released"] += int((foot & ~released).sum())
        counts["fail_settled"] += int((foot & ~settled).sum())
        counts["fail_grasp_paid"] += int((foot & ~paid).sum())
        counts["knock_counting"] += int((foot & (cmd._knock_count > 0)).sum())
        counts["all_pass"] += int((foot & inside & below & released & settled & paid).sum())
        steps_in += foot.float()
        bot.append((obj[:, 2] - half[:, 2])[foot].cpu())
        uns = foot & ~settled & paid & inside
        if uns.any():
          site = cmd._site_pos_w() - env.scene.env_origins
          d = torch.linalg.norm(obj - site, dim=-1)
          ang = torch.linalg.norm(cmd._stack(lambda o: o.data.root_link_ang_vel_w)[cmd._rows, cmd.target], dim=-1)
          sp.append(speed[uns].cpu()); dd.append(d[uns].cpu()); av.append(ang[uns].cpu())
          cl.append(shapes.object_shape_class(env)[uns].cpu()); hh.append(half[uns].cpu())
          far += int((uns & (d > 0.15)).sum()); pad += int((uns & (found > 0).any(-1)).sum())
      obs, _, dones, _ = wrapped.step(u)
      reset_recurrent(policy, dones)
  env.close()
  b = torch.cat(bot) if bot else torch.zeros(0)
  q = lambda t, x: float(t.quantile(x)) if t.numel() else None
  out = {"checkpoint": a.checkpoint, "task": a.task, "seed": a.seed, "num_envs": n, "steps": a.steps,
         "counts": counts,
         "envs_gt_3s_in_footprint_unregistered": int((steps_in > 150).sum()),
         "envs_gt_20s_in_footprint_unregistered": int((steps_in > 1000).sum()),
         "bottom_height_mm": {"p10": q(b, 0.1) and q(b, 0.1) * 1000, "p50": q(b, 0.5) and q(b, 0.5) * 1000, "p90": q(b, 0.9) and q(b, 0.9) * 1000},
         "bin_inner": list(cmd.cfg.bin_inner), "place_margin_m": cmd.cfg.place_margin_m, "rim_z": cmd.cfg.bin_rim_z,
         "settle_vel_threshold": cmd.cfg.place_settle_vel,
         "provenance": evalcfg.provenance(argv=sys.argv, sensor=sensor_prov, weights=loaded)}
  if sp:
    s_, d_, a_, c_, h_ = (torch.cat(x) for x in (sp, dd, av, cl, hh))
    ac = torch.cat(allc)
    out["unsettled_in_bin"] = {
      "steps": int(s_.numel()),
      "speed_m_s": {"p10": q(s_, 0.1), "p50": q(s_, 0.5), "p90": q(s_, 0.9), "max": float(s_.max())},
      "ang_vel_rad_s_p50": q(a_, 0.5),
      "hand_dist_m": {"p10": q(d_, 0.1), "p50": q(d_, 0.5), "p90": q(d_, 0.9)},
      "steps_hand_farther_than_15cm": far, "steps_pad_in_contact": pad,
      "shape_class_share": {nm: float((c_ == i).float().mean()) for i, nm in enumerate(objects.SHAPE_CLASSES)},
      "shape_class_base_rate_steps": {nm: float((ac == i).float().mean()) for i, nm in enumerate(objects.SHAPE_CLASSES)},
      "half_size_mm_p50": [float(x) for x in (h_.median(0).values * 1000)],
      "height_over_width_p50": float((h_[:, 2] / h_[:, :2].amax(-1)).median()),
    }
  print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, indent=1))
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
