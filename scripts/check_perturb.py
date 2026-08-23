"""Prove in simulation that each mismatch axis reaches the simulator.

tests/test_perturb.py checks the arithmetic in isolation.  This checks the
part no unit test can: that the value survives config application, manager
construction, per-world model expansion and the render, and that it does
*not* leak into an unperturbed run.

Two failures this is built to catch, both silent:

* an axis that is applied to world 0 only, because the event never declared
  the model field it writes -- every environment then shares one camera and
  the sweep measures a 512-fold-averaged version of nothing;
* an axis that changes the observation but not the *behaviour*, which is
  worth knowing before a whole sweep is spent on it.

    python scripts/check_perturb.py --device cuda:0 \\
        --json results/sim2real_sweep/plumbing_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from piper_push import perturb

TASK = "Mjlab-Pick-Place-PiperX-Vision"

# One representative, clearly-visible setting per axis. Not the sweep values --
# the question here is "does it arrive", not "how much does it matter".
PROBES: dict[str, float] = {
  "cam_pitch_deg": 4.0,
  "cam_yaw_deg": 4.0,
  "cam_pos_x_m": 0.05,
  "cam_pos_z_m": 0.05,
  "depth_scale": 1.10,
  "depth_bias_m": 0.03,
  "depth_dropout": 0.20,
  "depth_dropout_blob": 0.20,
  "obs_latency_steps": 3,
  "action_latency_steps": 3,
  "joint_response_scale": 0.60,
  "servo_damping_scale": 0.40,
  "action_deadband_rad": 0.010,
  "gripper_rate_scale": 0.40,
  "gripper_latency_steps": 3,
  "pad_friction_scale": 0.50,
  "table_friction_scale": 0.50,
}


def _probe(task: str, n: int, steps: int, device: str,
           mm: perturb.SessionMismatchCfg) -> dict:
  """Roll a fixed action sequence and summarise what the simulator did."""
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = n
  cfg.seed = 8181
  applied = perturb.apply_session_mismatch(cfg, mm)
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)

  torch.manual_seed(99)
  dim = env.action_manager.total_action_dim
  # A fixed, non-trivial command sequence: identical across conditions, so any
  # difference in the summary is the perturbation and not the policy.
  plan = torch.tanh(torch.randn(steps, 1, dim, device=device) * 0.6)

  cam_pos = []
  depth_mean = depth_far = None
  obs_first = None
  q_final = None
  with torch.inference_mode():
    env.reset()
    cam_idx = env.scene.sensors["scene_cam"].camera_idx
    cam_field = torch.as_tensor(env.sim.model.cam_pos)
    cam_pos = cam_field[:, cam_idx].clone()
    cam_quat = torch.as_tensor(env.sim.model.cam_quat)[:, cam_idx].clone()
    for i in range(steps):
      obs, *_ = env.step(plan[i].expand(n, dim).contiguous())
      if i == 2:
        img = obs["camera"]
        obs_first = img.clone()
        depth_mean = float(img[:, 0].mean())
        depth_far = float((img[:, 0] > 0.99).float().mean())
    q_final = env.scene["robot"].data.joint_pos.clone()

  out = {
    "applied": applied,
    # Whether the camera pose field was expanded per world at all.  In play
    # mode the jitter is zero by design, so every environment legitimately
    # gets the SAME pose -- a spread of zero is correct here and cannot be
    # used as the test.  What would be wrong is the field never being
    # expanded, which shows up as a leading axis of 1 instead of num_envs.
    "cam_field_worlds": int(cam_field.shape[0]),
    "cam_pos_mean": cam_pos.mean(0).tolist(),
    "cam_pos_env_spread": float(cam_pos.std(0).mean()),
    "cam_quat_env_spread": float(cam_quat.std(0).mean()),
    "depth_mean": depth_mean,
    "depth_at_far_plane": depth_far,
    "obs_checksum": float(obs_first.double().sum()),
    "joint_pos_checksum": float(q_final.double().sum()),
    "joint_pos_final": q_final.mean(0).tolist(),
  }
  env.close()
  return out


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default=TASK)
  p.add_argument("--num-envs", type=int, default=32)
  p.add_argument("--steps", type=int, default=40)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--json", default=None)
  p.add_argument("--only", default=None, help="comma-separated axis subset")
  a = p.parse_args()

  axes = list(PROBES) if a.only is None else [s.strip() for s in a.only.split(",")]

  base = _probe(a.task, a.num_envs, a.steps, a.device,
                perturb.SessionMismatchCfg())
  print(f"  {'axis':24s} {'obs Δ':>12s} {'joint Δ':>12s} {'worlds':>7s}  verdict")
  print(f"  {'(unperturbed)':24s} {'-':>12s} {'-':>12s}"
        f" {base['cam_field_worlds']:7d}")

  report = {"baseline": base, "axes": {}}
  ok = True
  for name in axes:
    mm = perturb.SessionMismatchCfg(**{name: PROBES[name]})
    r = _probe(a.task, a.num_envs, a.steps, a.device, mm)
    d_obs = abs(r["obs_checksum"] - base["obs_checksum"])
    d_q = abs(r["joint_pos_checksum"] - base["joint_pos_checksum"])
    rel_obs = d_obs / max(abs(base["obs_checksum"]), 1e-9)
    rel_q = d_q / max(abs(base["joint_pos_checksum"]), 1e-9)
    group = perturb.AXES[name].group
    # A camera axis has to move the observation; a plant axis has to move the
    # arm.  Either may move the other as a knock-on, and that is fine.
    want_obs = group == "camera"
    reached = rel_obs > 1e-6 if want_obs else rel_q > 1e-6
    ok &= reached
    r.update(delta_obs_rel=rel_obs, delta_joint_rel=rel_q, reached=reached)
    report["axes"][name] = r
    print(f"  {name:24s} {rel_obs:12.2e} {rel_q:12.2e}"
          f" {r['cam_field_worlds']:7d}  {'ok' if reached else 'NOT APPLIED'}")

  # The camera pose field must carry one row per world.  If it does not, the
  # write landed in world 0 and every environment shares one lens -- which
  # averages the perturbation away and reads as "this axis does not matter".
  for name, r in report["axes"].items():
    if perturb.AXES[name].group != "camera":
      continue
    if r["cam_field_worlds"] != a.num_envs:
      print(f"  ! {name}: cam_pos has {r['cam_field_worlds']} worlds, "
            f"expected {a.num_envs} -- the per-world write did not happen")
      ok = False

  print()
  print(f"  perturbation plumbing {'OK' if ok else 'BROKEN'}")
  report["verdict"] = "OK" if ok else "BROKEN"
  if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(report, indent=1))
    print(f"  wrote {a.json}")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
