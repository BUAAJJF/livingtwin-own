"""Stage 2: reward-free target-domain trajectories, and nothing else.

    python scripts/ra_sim0_collect.py --split train --seed 7101 \
        --num-envs 64 --steps 15000 --shapes train --domain target

One file per split.  What goes in it is what a deployed controller could log:
measured joint positions and velocities, the position targets it issued, the
gripper's state and command, and the object's pose -- the last only so that a
teacher-forced replay can put a candidate simulator back on the recorded
state, never as a model input.

What deliberately does not go in it: the hidden plant's internal states, the
effective command it produces, reward, success, the safety-shell label.  The
recorder never reads them.

Composition, fixed in docs/ra_sim0_experiment_plan.md: 70% of the steps are
the frozen policy's own rollout, 20% the same policy under a small
pre-registered action perturbation, 10% a scripted sweep with the gripper
cycling.  The mode of every step is recorded so a later split can be made on
it rather than on a guess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
TASK = "Mjlab-Pick-Place-PiperX-Vision"
CKPT = ("logs/rsl_rl/piperx_pick_place_vision/"
        "2026-08-22_17-15-09_f3/model_1500.pt")
SHAPE_PRESETS = {
  "all": None,
  "train": (0.34, 0.22, 0.16, 0.0, 0.0),
  "holdout": (0.0, 0.0, 0.0, 0.16, 0.12),
}
MODE_NATURAL, MODE_PERTURBED, MODE_PROBE = 0, 1, 2


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def provenance() -> dict:
  import importlib.metadata as md

  def git(*a):
    try:
      return subprocess.run(("git", *a), cwd=ROOT, capture_output=True,
                            text=True, timeout=15).stdout.strip()
    except Exception:
      return ""

  vers = {}
  for d in ("mjlab", "rsl-rl-lib", "mujoco", "mujoco-warp", "warp-lang", "torch"):
    try:
      vers[d] = md.version(d)
    except Exception:
      vers[d] = "?"
  return {"commit": git("rev-parse", "HEAD"),
          "dirty": bool(git("status", "--porcelain")),
          "versions": vers}


def probe_actions(n_env: int, n_act: int, steps: int, device: str, seed: int,
                  amp: float = 0.6) -> torch.Tensor:
  """A scripted no-load sweep: multi-frequency, with reversals, per arm.

  The point of the 10% probe is coverage the policy will not give: slow ramps
  that sit inside a backlash band for many steps, and fast reversals that
  cross it twice in three.  A policy doing a pick has neither.
  """
  g = torch.Generator().manual_seed(seed)
  t = torch.arange(steps, dtype=torch.float32).view(-1, 1, 1)
  out = torch.zeros(steps, n_env, n_act)
  for k, f in enumerate((0.004, 0.017, 0.061)):
    ph = torch.rand(1, n_env, n_act, generator=g) * 6.2831853
    a = (torch.rand(1, n_env, n_act, generator=g) * 0.5 + 0.5) * amp / 3.0
    out += a * torch.sin(2 * 3.14159265 * f * t + ph)
  # The gripper is the last action; drive it as a square wave so the pads
  # actually open and close rather than dithering about the midpoint.
  out[:, :, -1] = torch.sign(torch.sin(2 * 3.14159265 * 0.01 * t[:, :, 0]))
  return out.to(device)


def collect(split: str, seed: int, n_envs: int, steps: int, shapes: str,
            domain: str, device: str, out_dir: Path, perturb_sigma: float,
            perturb_clip: float, chunk_log: int = 1000) -> dict:
  from dataclasses import asdict as _asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import sys
  sys.path.insert(0, str(ROOT / "src"))
  from piper_push import shapes as shp
  from piper_push.hidden_plant import HiddenPlantCfg, apply_hidden_plant

  cfg = load_env_cfg(TASK, play=True)
  agent = load_rl_cfg(TASK)
  cfg.scene.num_envs = n_envs
  cfg.seed = seed
  target_cfg = None
  if domain == "target":
    target_cfg = HiddenPlantCfg()
    applied = apply_hidden_plant(cfg, target_cfg)
  elif domain == "nominal":
    applied = {}
  else:
    raise ValueError(f"unknown domain {domain!r}")

  w = SHAPE_PRESETS[shapes]
  if w is not None:
    for name, ev in cfg.events.items():
      if name.startswith("object_shape"):
        ev.params = dict(ev.params)
        ev.params["shape_weights"] = tuple(w)

  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    wrapped, _asdict(agent), device=device)
  ck = str(ROOT / CKPT)
  runner.load(ck, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  policy.reset()

  u = env
  robot = u.scene["robot"]
  jids, _ = robot.find_joints([f"joint{i}" for i in range(1, 7)],
                              preserve_order=True)
  gid = robot.find_joints(("gripper_joint1",))[0][0]
  arm = u.action_manager.get_term("arm")
  grip = u.action_manager.get_term("gripper")
  obj = u.scene["object"]

  n_nat = int(round(steps * 0.70))
  n_per = int(round(steps * 0.20))
  n_prb = steps - n_nat - n_per
  probe = probe_actions(n_envs, u.action_manager.total_action_dim, n_prb,
                        device, seed + 91)
  gen = torch.Generator(device=device).manual_seed(seed + 17)

  cols: dict[str, list[torch.Tensor]] = {
    k: [] for k in ("q", "qd", "u", "gq", "gu", "a", "obj", "done", "mode",
                    "shape")}
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]
  t0 = time.time()
  for t in range(steps):
    if t < n_nat:
      mode = MODE_NATURAL
    elif t < n_nat + n_per:
      mode = MODE_PERTURBED
    else:
      mode = MODE_PROBE
    q = robot.data.joint_pos[:, jids].clone()
    qd = robot.data.joint_vel[:, jids].clone()
    gq = robot.data.joint_pos[:, gid].unsqueeze(-1).clone()
    # mjlab keeps the pose and the velocity apart; write_root_state_to_sim
    # wants them concatenated in this order, which is what makes this
    # the resynchronisable form.
    o = torch.cat([obj.data.root_link_pose_w, obj.data.root_com_vel_w],
                  dim=-1).clone()
    sc = shp.object_shape_class(u, "object").to(torch.uint8).clone()

    with torch.inference_mode():
      act = policy(obs)
    if mode == MODE_PERTURBED:
      noise = torch.randn(act.shape, generator=gen, device=device) * perturb_sigma
      act = act + noise.clamp(-perturb_clip, perturb_clip)
    elif mode == MODE_PROBE:
      act = probe[t - n_nat - n_per]

    step = wrapped.step(act)
    obs, dones = step[0], step[2]
    policy.reset(dones)

    cols["q"].append(q.cpu())
    cols["qd"].append(qd.cpu())
    cols["gq"].append(gq.cpu())
    cols["obj"].append(o.cpu())
    cols["shape"].append(sc.cpu())
    cols["a"].append(act.float().cpu())
    # The command stream, read after the step: `_previous_target` is what the
    # controller ISSUED this step, which a deployed stack knows.  The hook's
    # output -- the target's effective command -- is never read.
    cols["u"].append(arm._previous_target.float().cpu())
    cols["gu"].append(grip._previous_target.float().cpu())
    cols["done"].append(dones.bool().cpu())
    cols["mode"].append(torch.full((n_envs,), mode, dtype=torch.uint8))

    if chunk_log and (t + 1) % chunk_log == 0:
      r = (t + 1) * n_envs / max(time.time() - t0, 1e-9)
      print(f"    step {t + 1}/{steps}  {r:,.0f} env-steps/s", flush=True)

  env.close()
  data = {k: torch.stack(v) for k, v in cols.items()}
  meta = {
    "split": split, "seed": seed, "num_envs": n_envs, "steps": steps,
    "shapes": shapes, "domain": domain, "task": TASK,
    "checkpoint": CKPT, "checkpoint_sha256": sha256(Path(ck)),
    "hidden_target": applied, "budget_s": steps / 50.0,
    "composition": {"natural": n_nat, "perturbed": n_per, "probe": n_prb},
    "perturb_sigma": perturb_sigma, "perturb_clip": perturb_clip,
    "wall_clock_s": time.time() - t0,
    "channels": {k: list(v.shape) for k, v in data.items()},
    "provenance": provenance(),
  }
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / f"{split}_{domain}_{seed}.pt"
  torch.save({"data": data, "meta": meta}, path)
  meta["file"] = str(path)
  meta["file_sha256"] = sha256(path)
  meta["file_mib"] = round(path.stat().st_size / 2**20, 1)
  (out_dir / f"{split}_{domain}_{seed}.json").write_text(json.dumps(meta, indent=2))
  print(json.dumps({k: meta[k] for k in
                    ("file", "file_mib", "budget_s", "wall_clock_s")}, indent=2))
  return meta


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--split", required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--steps", type=int, default=15000)
  ap.add_argument("--shapes", choices=tuple(SHAPE_PRESETS), default="train")
  ap.add_argument("--domain", choices=("target", "nominal"), default="target")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", default="results/ra_sim0/data")
  ap.add_argument("--perturb-sigma", type=float, default=0.05)
  ap.add_argument("--perturb-clip", type=float, default=0.15)
  a = ap.parse_args()
  collect(a.split, a.seed, a.num_envs, a.steps, a.shapes, a.domain, a.device,
          Path(a.out), a.perturb_sigma, a.perturb_clip)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
