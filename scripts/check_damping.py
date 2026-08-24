"""In-simulator plumbing: does the per-environment draw give the same plant?

    python scripts/check_damping.py --device cuda:0

Phase WM0 measured `servo_damping_scale` by scaling the actuator *config*
before the entity was built, which gives every environment the same damping.
Phase WM1-B needs a mixture, so it writes the compiled model's per-world
`actuator_biasprm` instead. Those are two different mechanisms and the whole of
WM1-B rests on them being the same domain, so this builds one environment each
way and compares the field the physics actually reads.

Three questions, and the second is the one a unit test cannot ask:

1. does the per-environment event at a point mass produce the same
   `actuator_biasprm` as Phase WM0's config-level scaling?
2. is the *proportional* gain untouched by both, so this is a damping axis and
   not a stiffness one?
3. does a mixture put the requested fraction of environments at each value, and
   leave the gripper alone?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from piper_push import damping, perturb

TASK = "Mjlab-Pick-Place-PiperX-Vision"


def _build(task, n_envs, device, seed, mutate):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = n_envs
  cfg.seed = seed
  mutate(cfg)
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  env.reset()
  return env


def _gains(env):
  """(kd, kp) per environment per actuator, as positive numbers."""
  b = env.unwrapped.sim.model.actuator_biasprm
  return -b[..., 2].clone(), -b[..., 1].clone()


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default=TASK)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=11)
  p.add_argument("--json", default=None)
  a = p.parse_args()

  out = {"task": a.task, "num_envs": a.num_envs, "target": damping.TARGET}

  # -- 1. the two mechanisms, at the same point mass -------------------------
  env_cfgpath = _build(a.task, a.num_envs, a.device, a.seed, lambda c:
                       perturb.apply_session_mismatch(
                         c, perturb.SessionMismatchCfg(
                           servo_damping_scale=damping.TARGET)))
  kd_cfg, kp_cfg = _gains(env_cfgpath)
  env_cfgpath.close()

  env_event = _build(a.task, a.num_envs, a.device, a.seed, lambda c:
                     damping.apply_damping_prior(
                       c, damping.DampingPrior.point(damping.TARGET)))
  kd_evt, kp_evt = _gains(env_event)
  env_event.close()

  env_nom = _build(a.task, a.num_envs, a.device, a.seed, lambda c: None)
  kd_nom, kp_nom = _gains(env_nom)
  env_nom.close()

  # The gripper actuator is the one whose kd the config path leaves alone; find
  # it by comparing against nominal rather than by index, so a reordering of
  # the actuator list cannot make this check pass for the wrong reason.
  touched = (kd_cfg[0] - kd_nom[0]).abs() > 1e-9
  out["actuators_touched"] = [int(i) for i in touched.nonzero().flatten()]
  out["actuators_untouched"] = [int(i) for i in (~touched).nonzero().flatten()]

  same = torch.allclose(kd_cfg, kd_evt, rtol=1e-6, atol=1e-9)
  out["point_mass_matches_config_path"] = bool(same)
  out["max_abs_kd_difference"] = float((kd_cfg - kd_evt).abs().max())
  out["kp_unchanged_config"] = bool(torch.allclose(kp_cfg, kp_nom))
  out["kp_unchanged_event"] = bool(torch.allclose(kp_evt, kp_nom))
  out["kd_ratio_on_touched"] = float(
    (kd_evt[0][touched] / kd_nom[0][touched]).mean()) if touched.any() else float("nan")

  # -- 3. a mixture ----------------------------------------------------------
  probs = (0.5, 0.25, 0.25)
  env_mix = _build(a.task, max(a.num_envs, 256), a.device, a.seed, lambda c:
                   damping.apply_damping_prior(
                     c, damping.DampingPrior(probs)))
  kd_mix, kp_mix = _gains(env_mix)
  kd_nom_row = kd_nom[0]
  counts = {}
  if touched.any():
    j = int(touched.nonzero().flatten()[0])
    ratio = kd_mix[:, j] / kd_nom_row[j]
    for v in damping.VALUES:
      counts[str(v)] = int((ratio - v).abs().lt(1e-4).sum())
  n = sum(counts.values()) or 1
  out["mixture_requested"] = dict(zip((str(v) for v in damping.VALUES), probs))
  out["mixture_observed"] = {k: v / n for k, v in counts.items()}
  out["mixture_n"] = n
  out["gripper_untouched_under_mixture"] = bool(
    torch.allclose(kd_mix[:, ~touched], kd_nom[:, ~touched].expand_as(
      kd_mix[:, ~touched])))
  env_mix.close()

  ok = (out["point_mass_matches_config_path"] and out["kp_unchanged_event"]
        and out["gripper_untouched_under_mixture"]
        and all(abs(out["mixture_observed"].get(str(v), 0.0) - pr) < 0.08
                for v, pr in zip(damping.VALUES, probs)))
  out["pass"] = bool(ok)

  print()
  print(f"  actuators whose kd moved: {out['actuators_touched']}, "
        f"untouched: {out['actuators_untouched']}")
  print(f"  event == config path at a point mass: "
        f"{out['point_mass_matches_config_path']} "
        f"(max |dkd| = {out['max_abs_kd_difference']:.3e})")
  print(f"  kd ratio on touched actuators: {out['kd_ratio_on_touched']:.4f} "
        f"(asked {damping.TARGET})")
  print(f"  kp unchanged: config {out['kp_unchanged_config']}, "
        f"event {out['kp_unchanged_event']}")
  print(f"  mixture over {out['mixture_n']} environments: "
        f"{ {k: round(v, 3) for k, v in out['mixture_observed'].items()} } "
        f"asked {out['mixture_requested']}")
  print(f"  gripper untouched under mixture: "
        f"{out['gripper_untouched_under_mixture']}")
  print(f"  PASS: {out['pass']}")

  if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"  wrote {a.json}")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
