"""Can a target domain be told apart from reward-free rollout data alone?

The calibration idea in the README only works if a session's simulator
mismatch leaves a signature in what the robot can actually record: joint
states, the commands it issued, the gripper's servo error, and the policy's
own latent.  Not reward, not success, not contact truth, not the parameter
itself.  This measures whether that signature exists and generalises.

What is deliberately excluded from every input set:

* reward, return, success or placement labels;
* contact flags, object pose, mass, friction -- anything privileged;
* the perturbation value itself, which is the label.

Phase and step index are recorded but never fed to the probe: knowing "this
window is a grasp" is not available before the fact on hardware, and a probe
that leans on it would be reading the task schedule rather than the plant.

Leakage control, in order of how badly each would flatter the result:

* **split by environment.**  Every window from one environment goes to one
  side.  Windows overlap in time and neighbouring windows are near-copies, so
  a random split would report autocorrelation as accuracy.
* **held-out shapes.**  With ``--holdout-shape`` the test set is restricted to
  object shape classes absent from training, so the probe cannot succeed by
  memorising what the training objects looked like.
* **permutation control.**  The same pipeline on shuffled labels, which is the
  floor a real signal has to clear.
* **majority-class chance**, not 1/K.

    python scripts/identifiability.py --axis depth_scale --levels 0.93,1.0,1.07
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push import perturb, shapes

TASK = "Mjlab-Pick-Place-PiperX-Vision"
CKPT = ("logs/rsl_rl/piperx_pick_place_vision/"
        "2026-08-22_17-15-09_f3/model_1500.pt")

FEATURE_SETS = {
  # What a bare proprioceptive log gives you: no camera, no network internals.
  "proprio_servo": ("joint_pos", "joint_vel", "gripper", "servo", "action"),
  # Adding what the encoder makes of the image, without the recurrent state.
  "encoded_proprio": ("encoded",),
  # Everything the deployed policy itself computes.
  "actor_latent": ("latent", "action", "joint_pos", "joint_vel", "servo"),
}


def _stats(win: torch.Tensor) -> torch.Tensor:
  """Summarise a (T, C) window as per-channel statistics.

  Mean and spread say where the plant sat; the first-difference magnitude says
  how hard it was working, which is where latency and response errors show up;
  the last sample keeps a little of the phase without naming it.
  """
  d = win[1:] - win[:-1]
  return torch.cat([
    win.mean(0), win.std(0), win.amin(0), win.amax(0), win[-1],
    d.abs().mean(0),
  ])


@torch.inference_mode()
def collect(task: str, ckpt: str, mm: perturb.SessionMismatchCfg, label: int,
            n_envs: int, steps: int, window: int, stride: int,
            device: str, seed: int) -> dict:
  """Roll the frozen policy in one domain and cut its log into windows."""
  cfg = load_env_cfg(task, play=True)
  agent = load_rl_cfg(task)
  cfg.scene.num_envs = n_envs
  cfg.seed = seed
  perturb.apply_session_mismatch(cfg, mm)

  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent),
                                                          device=device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  recurrent = bool(getattr(policy, "is_recurrent", False))
  if recurrent:
    policy.reset()

  u = env.unwrapped
  robot = u.scene["robot"]
  pick = u.command_manager.get_term("pick")
  jids, _ = robot.find_joints([f"joint{i}" for i in range(1, 7)],
                              preserve_order=True)
  gid = robot.find_joints(("gripper_joint1",))[0][0]

  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  chans: dict[str, list[torch.Tensor]] = {k: [] for k in
                                          ("joint_pos", "joint_vel", "gripper",
                                           "servo", "action", "latent",
                                           "encoded")}
  shape_cls = []
  for t in range(steps):
    act = policy(obs)
    # The policy's own internals, read before the step so they correspond to
    # the observation that produced this action.
    if recurrent:
      h = policy.get_hidden_state()
      chans["latent"].append((h if not isinstance(h, tuple) else h[0])[-1].float().cpu())
      chans["encoded"].append(policy._encode(obs).float().cpu())
    chans["action"].append(act.float().cpu())
    chans["joint_pos"].append(robot.data.joint_pos[:, jids].float().cpu())
    chans["joint_vel"].append(robot.data.joint_vel[:, jids].float().cpu())
    g = robot.data.joint_pos[:, gid]
    chans["gripper"].append(g.unsqueeze(-1).float().cpu())
    chans["servo"].append(
      (g - robot.data.joint_pos_target[:, gid]).unsqueeze(-1).float().cpu())
    shape_cls.append(shapes.object_shape_class(u, "object").cpu())

    out = env.step(act)
    obs, dones = out[0], out[2]
    if recurrent:
      policy.reset(dones)

  stacked = {k: torch.stack(v) for k, v in chans.items() if v}   # (T, B, C)
  cls = torch.stack(shape_cls)                                    # (T, B)

  # Cut into overlapping windows, tagged with the environment they came from
  # and the shape class that was in the hand at the end of the window.
  feats: dict[str, list[torch.Tensor]] = {k: [] for k in FEATURE_SETS}
  env_ids, shapes_out = [], []
  # Skip the first window's worth: the recurrent state and the plant pipeline
  # are both still filling and would be a signature of the reset, not the
  # domain.
  for t0 in range(window, steps - window, stride):
    sl = slice(t0, t0 + window)
    for name, keys in FEATURE_SETS.items():
      if any(k not in stacked for k in keys):
        continue
      per_env = []
      for b in range(n_envs):
        per_env.append(_stats(torch.cat([stacked[k][sl, b] for k in keys], -1)))
      feats[name].append(torch.stack(per_env))
    env_ids.append(torch.arange(n_envs))
    shapes_out.append(cls[t0 + window - 1])

  env.close()
  return {
    "label": label,
    "features": {k: torch.cat(v) for k, v in feats.items() if v},
    "env": torch.cat(env_ids),
    "shape": torch.cat(shapes_out),
  }


def _fit(x: torch.Tensor, y: torch.Tensor, tr: torch.Tensor, te: torch.Tensor,
         k: int, epochs: int = 400) -> dict:
  """Multinomial logistic regression, plus a nearest-centroid baseline."""
  xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
  if len(xte) == 0 or len(xtr) == 0:
    return {"error": "empty split"}
  mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp(min=1e-6)
  xtr, xte = (xtr - mu) / sd, (xte - mu) / sd

  w = torch.zeros(x.shape[1], k, requires_grad=True)
  b = torch.zeros(k, requires_grad=True)
  opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-2)
  for _ in range(epochs):
    opt.zero_grad()
    torch.nn.functional.cross_entropy(xtr @ w + b, ytr).backward()
    opt.step()
  with torch.no_grad():
    pred = (xte @ w + b).argmax(-1)
    acc = (pred == yte).float().mean().item()
    # Nearest centroid: no capacity to overfit, so a gap between it and the
    # regression says the signature is there but not linearly separable.
    cents = torch.stack([xtr[ytr == c].mean(0) if (ytr == c).any()
                         else torch.zeros(xtr.shape[1]) for c in range(k)])
    nc = (torch.cdist(xte, cents).argmin(-1) == yte).float().mean().item()
    recalls = [float((pred[yte == c] == c).float().mean())
               for c in range(k) if (yte == c).any()]
  major = torch.bincount(ytr, minlength=k).argmax()
  chance = (yte == major).float().mean().item()
  return {
    "accuracy": acc, "nearest_centroid": nc,
    "balanced_accuracy": sum(recalls) / max(len(recalls), 1),
    "majority_chance": chance, "uniform_chance": 1.0 / k,
    "lift_over_majority": acc - chance,
    "n_train": int(len(xtr)), "n_test": int(len(xte)),
  }


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default=TASK)
  p.add_argument("--checkpoint", default=CKPT)
  p.add_argument("--axis", required=True)
  p.add_argument("--levels", required=True,
                 help="comma-separated, including the nominal")
  p.add_argument("--num-envs", type=int, default=96)
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--window", type=int, default=50, help="control steps (1 s)")
  p.add_argument("--stride", type=int, default=25)
  p.add_argument("--holdout-shape", action="store_true",
                 help="test only on object shape classes absent from training")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=555000)
  p.add_argument("--json", default=None)
  a = p.parse_args()

  levels = [float(s) for s in a.levels.split(",")]
  data = []
  for i, v in enumerate(levels):
    val = int(v) if "steps" in a.axis else v
    mm = perturb.SessionMismatchCfg(**{a.axis: val})
    print(f"  collecting {a.axis}={val} ...")
    data.append(collect(a.task, a.checkpoint, mm, i, a.num_envs, a.steps,
                        a.window, a.stride, a.device, a.seed + i))

  y = torch.cat([torch.full((len(d["env"]),), d["label"]) for d in data])
  envs = torch.cat([d["env"] for d in data])
  shp = torch.cat([d["shape"] for d in data])

  # Environments are the sessions: whole ones go to one side of the split.
  n_env = int(envs.max()) + 1
  g = torch.Generator().manual_seed(0)
  perm = torch.randperm(n_env, generator=g)
  test_env = set(perm[: max(1, n_env // 4)].tolist())
  is_test = torch.tensor([int(e) in test_env for e in envs.tolist()])
  if a.holdout_shape:
    # Two of the five classes held out entirely.  The probe then has to work
    # on objects it never saw, which is the case that matters: a real session
    # is not a re-run of the calibration objects.
    held = {3, 4}
    is_test = is_test & torch.tensor([int(c) in held for c in shp.tolist()])
    train_mask = ~torch.tensor([int(c) in held for c in shp.tolist()])
  else:
    train_mask = ~is_test
  tr = (train_mask & ~is_test).nonzero().flatten()
  te = is_test.nonzero().flatten()

  report = {
    "axis": a.axis, "levels": levels, "n_classes": len(levels),
    "config": {k: getattr(a, k) for k in
               ("num_envs", "steps", "window", "stride", "holdout_shape",
                "seed", "checkpoint")},
    "n_windows": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
    "sets": {},
  }
  print()
  print(f"  {'feature set':18s} {'dim':>5s} {'acc':>7s} {'balanced':>9s} "
        f"{'centroid':>9s} {'chance':>7s} {'shuffled':>9s}")
  for name in FEATURE_SETS:
    if name not in data[0]["features"]:
      continue
    x = torch.cat([d["features"][name] for d in data])
    res = _fit(x, y, tr, te, len(levels))
    # The floor: identical pipeline, labels shuffled between environments so
    # the shuffle cannot be undone by the grouping.
    yperm = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    ctrl = _fit(x, yperm, tr, te, len(levels))
    res["shuffled_control"] = ctrl["accuracy"]
    res["dim"] = int(x.shape[1])
    report["sets"][name] = res
    print(f"  {name:18s} {x.shape[1]:5d} {100 * res['accuracy']:6.1f}% "
          f"{100 * res['balanced_accuracy']:8.1f}% "
          f"{100 * res['nearest_centroid']:8.1f}% "
          f"{100 * res['majority_chance']:6.1f}% "
          f"{100 * ctrl['accuracy']:8.1f}%")

  best = max((r["accuracy"] - r["majority_chance"]
              for r in report["sets"].values()), default=0.0)
  report["best_lift_over_chance"] = best
  print()
  print(f"  best lift over majority chance: {100 * best:+.1f} pp")

  if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(report, indent=1))
    print(f"  wrote {a.json}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
