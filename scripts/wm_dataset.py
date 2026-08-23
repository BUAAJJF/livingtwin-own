"""Roll the frozen policy in one latency domain and log what a robot could see.

    python scripts/wm_dataset.py --lag 3 --seed 909 --num-envs 48 \
        --steps 15000 --split target --shapes holdout

One file per (split, lag, seed).  The channels and what is deliberately
excluded from them are documented in :mod:`piper_push.wm_data`; the short
version is that reward, success, safety events, object state and the latency
itself are not in the file that inference reads.

Nothing here chooses a split boundary at analysis time.  The generation seed,
the environment index and the object shape classes are fixed when the data is
made, so "train" and "target" are different rollouts of different objects
under different RNG, not two views of one tensor.

``--shapes holdout`` puts zero probability on the shape classes the training
splits use, which is stronger than filtering the log afterwards: the policy's
recurrent state never carries a training object either.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push import latency, shapes, wm_data

TASK = "Mjlab-Pick-Place-PiperX-Vision"
CKPT = ("logs/rsl_rl/piperx_pick_place_vision/"
        "2026-08-22_17-15-09_f3/model_1500.pt")

# Classes 0-2 (box, cylinder, stepped) train; 3-4 (l_shape, capped) are held
# out.  The same two classes Phase WM0's identifiability probe held out, so the
# two results are about the same objects.
SHAPE_PRESETS = {
  "all": None,
  "train": (0.34, 0.22, 0.16, 0.0, 0.0),
  "holdout": (0.0, 0.0, 0.0, 0.16, 0.12),
}


def _sha256(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _provenance(checkpoint: str) -> dict:
  def git(*args: str) -> str:
    try:
      return subprocess.run(("git", *args),
                            cwd=Path(__file__).resolve().parent.parent,
                            capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
      return ""

  import importlib.metadata as md

  def version(dist: str) -> str:
    try:
      return md.version(dist)
    except Exception:
      return "unknown"

  return {
    "git_commit": git("rev-parse", "HEAD"),
    "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "git_dirty": git("status", "--porcelain"),
    "checkpoint": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": _sha256(checkpoint),
    "argv": sys.argv,
    "torch": torch.__version__,
    "mjlab": version("mjlab"),
    "rsl_rl": version("rsl-rl-lib"),
  }


@torch.inference_mode()
def collect(task: str, ckpt: str, lag: int, n_envs: int, steps: int,
            device: str, seed: int, shape_weights, chunk_log: int = 500):
  cfg = load_env_cfg(task, play=True)
  agent = load_rl_cfg(task)
  cfg.scene.num_envs = n_envs
  cfg.seed = seed

  prior = latency.LatencyPrior.point(lag)
  applied = latency.apply_latency_prior(cfg, prior, seed=seed)

  if shape_weights is not None:
    for name, ev in cfg.events.items():
      if name.startswith("object_shape"):
        ev.params = dict(ev.params)
        ev.params["shape_weights"] = tuple(shape_weights)

  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent),
                                                          device=device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  if not getattr(policy, "is_recurrent", False):
    raise RuntimeError("this dataset records a recurrent policy's hidden state")
  policy.reset()

  u = env.unwrapped
  robot = u.scene["robot"]
  jids, _ = robot.find_joints([f"joint{i}" for i in range(1, 7)],
                              preserve_order=True)
  gid = robot.find_joints(("gripper_joint1",))[0][0]

  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  cols: dict[str, list[torch.Tensor]] = {
    k: [] for k in ("enc", "hidden", "proprio", "action", "servo", "done",
                    "shape")}
  t_start = time.time()
  for t in range(steps):
    # The hidden state BEFORE this step's latent is consumed: the pair
    # (enc_t, hidden_t) is what the frozen actor turns into a_t, so a
    # *predicted* enc can be pushed through the same head from the same state.
    h_prev = policy.rnn.hidden_state
    if h_prev is None:
      h_prev = wm_data.zero_hidden(policy, u.num_envs, obs.device)
    enc = policy._encode(obs)
    act, h_new = wm_data.actor_head(policy, enc, h_prev)
    policy.rnn.hidden_state = h_new

    g = robot.data.joint_pos[:, gid]
    cols["enc"].append(enc.half().cpu())
    cols["hidden"].append(h_prev[-1].half().cpu())
    cols["proprio"].append(torch.cat([
      robot.data.joint_pos[:, jids],
      robot.data.joint_vel[:, jids],
      g.unsqueeze(-1)], dim=-1).float().cpu())
    cols["action"].append(act.float().cpu())
    cols["servo"].append(
      (g - robot.data.joint_pos_target[:, gid]).unsqueeze(-1).float().cpu())
    cols["shape"].append(shapes.object_shape_class(u, "object").to(torch.uint8).cpu())

    out = env.step(act)
    obs, dones = out[0], out[2]
    cols["done"].append(dones.bool().cpu())
    policy.reset(dones)

    if chunk_log and (t + 1) % chunk_log == 0:
      rate = (t + 1) * n_envs / max(time.time() - t_start, 1e-9)
      print(f"    step {t + 1}/{steps}  {rate:,.0f} env-steps/s", flush=True)

  env.close()
  return wm_data.SessionSet(
    enc=torch.stack(cols["enc"]),
    hidden=torch.stack(cols["hidden"]),
    proprio=torch.stack(cols["proprio"]),
    action=torch.stack(cols["action"]),
    servo=torch.stack(cols["servo"]),
    done=torch.stack(cols["done"]),
    shape=torch.stack(cols["shape"]),
    lag=lag,
    meta={"latency": applied, "seed": seed, "wall_clock_s": time.time() - t_start},
  )


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default=TASK)
  p.add_argument("--checkpoint", default=CKPT)
  p.add_argument("--lag", type=int, required=True, choices=list(latency.LAGS))
  p.add_argument("--seed", type=int, required=True)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=1500)
  p.add_argument("--split", required=True,
                 choices=("train", "val", "valh", "target", "benign"))
  p.add_argument("--shapes", default="train", choices=tuple(SHAPE_PRESETS))
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default="results/wm1_latency/data")
  a = p.parse_args()

  out = Path(a.out)
  name = f"{a.split}__lag{a.lag}__seed{a.seed}"
  print(f"  collecting {name}: {a.num_envs} envs x {a.steps} steps "
        f"({a.num_envs * a.steps / 50.0 / 60.0:.1f} arm-minutes), "
        f"shapes={a.shapes}")
  s = collect(a.task, a.checkpoint, a.lag, a.num_envs, a.steps, a.device,
              a.seed, SHAPE_PRESETS[a.shapes])
  s.meta.update(split=a.split, shapes=a.shapes,
                shape_weights=SHAPE_PRESETS[a.shapes],
                provenance=_provenance(a.checkpoint))
  s.save(out / f"{name}.pt")

  desc = s.describe()
  desc.update(file=f"{name}.pt", split=a.split, shapes=a.shapes, seed=a.seed)
  (out / f"{name}.json").write_text(json.dumps(desc, indent=1))
  print(f"  wrote {out / name}.pt  "
        f"({desc['bytes'] / 1e6:.0f} MB, {desc['arm_seconds'] / 60:.1f} "
        f"arm-minutes, {desc['episode_boundaries']} episode boundaries)")
  print(f"  shape class counts: {desc['shape_class_counts']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
