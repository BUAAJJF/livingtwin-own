"""What is the recurrent state carrying, and does any of it belong to the past?

Three diagnostics, none of which changes the policy:

1. **Linear probe.**  Decode, from the GRU hidden state alone, the shape class
   and mass bin of (a) the object in the hand right now and (b) the object
   BEFORE it.  The first is the memory doing its job -- a camera policy has to
   integrate evidence about what it is holding.  The second is the artefact:
   on a real table the previous object tells you nothing about this one, so a
   state that still encodes it is a state that learned an invariant the world
   does not have.

   The split is by ENVIRONMENT.  A random split over timesteps would put
   frames 40 ms apart in train and test and report the autocorrelation of the
   hidden state as probe accuracy, which is how this measurement is usually
   got wrong.

2. **History swap.**  Freeze the current observation and permute the hidden
   states across environments, so each policy sees its own present and someone
   else's past.  If the action does not move, nothing in the memory is being
   used; if it moves and keeps moving, the past is steering the present.

3. **Chance baselines.**  Reported as the majority-class rate, not 1/K.  The
   shape prior is (0.34, 0.22, 0.16, 0.16, 0.12), so a probe that learns
   nothing still reads 34% on a uniform-chance framing.

    python scripts/probe_hidden.py Mjlab-Pick-Place-PiperX-Vision ckpt.pt \\
        --json results/novelty_validation/probe/distilled.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from piper_push import shapes

MASS_BINS = 4


def _fit_linear_probe(x: torch.Tensor, y: torch.Tensor, groups: torch.Tensor,
                      n_classes: int, epochs: int = 300,
                      holdout: float = 0.25) -> dict:
  """Multinomial logistic regression with a group-disjoint split.

  Linear on purpose.  The question is whether the information is *present and
  linearly available* in the hidden state, not whether some network can be
  built to extract it -- a deep enough probe recovers almost anything and
  answers a different question.
  """
  uniq = torch.unique(groups)
  n_test = max(1, int(len(uniq) * holdout))
  # Deterministic, and by group: whole environments go to one side or the
  # other, so no two frames 40 ms apart straddle the split.
  perm = torch.randperm(len(uniq), generator=torch.Generator().manual_seed(0))
  test_groups = set(uniq[perm[:n_test]].tolist())
  is_test = torch.tensor([int(g) in test_groups for g in groups.tolist()])

  xtr, ytr = x[~is_test], y[~is_test]
  xte, yte = x[is_test], y[is_test]
  if len(xte) == 0 or len(xtr) == 0:
    return {"error": "empty split"}

  mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp(min=1e-6)
  xtr, xte = (xtr - mu) / sd, (xte - mu) / sd

  w = torch.zeros(x.shape[1], n_classes, requires_grad=True)
  b = torch.zeros(n_classes, requires_grad=True)
  opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-3)
  for _ in range(epochs):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(xtr @ w + b, ytr)
    loss.backward()
    opt.step()

  with torch.no_grad():
    pred = (xte @ w + b).argmax(-1)
    acc = (pred == yte).float().mean().item()
    train_acc = ((xtr @ w + b).argmax(-1) == ytr).float().mean().item()
  # The honest floor: always predict whichever class is commonest in TRAIN,
  # scored on test.
  major = torch.bincount(ytr, minlength=n_classes).argmax()
  chance = (yte == major).float().mean().item()

  # Raw accuracy and lift are both contaminated when two conditions have
  # different label marginals, which EP-All and OBJ-All emphatically do: under
  # EP-All the label is constant for a whole episode, so the empirical class
  # frequencies are lumpy and the majority baseline moves.  Balanced accuracy
  # -- the mean per-class recall -- is invariant to the class prior and is the
  # number the two conditions can be compared on.
  recalls = []
  for k in range(n_classes):
    m = yte == k
    if bool(m.any()):
      recalls.append((pred[m] == k).float().mean().item())
  bal = sum(recalls) / max(len(recalls), 1)

  return {
    "test_accuracy": acc, "train_accuracy": train_acc,
    "majority_class_baseline": chance,
    "lift_over_chance": acc - chance,
    "balanced_accuracy": bal,
    "balanced_chance": 1.0 / max(len(recalls), 1),
    "balanced_lift": bal - 1.0 / max(len(recalls), 1),
    "classes_present_in_test": len(recalls),
    "n_train": int(len(xtr)), "n_test": int(len(xte)),
    "n_train_envs": int(len(uniq) - n_test), "n_test_envs": n_test,
    # How many INDEPENDENT labels the test set really contains.  Under EP-All
    # the label is constant for a whole episode, so tens of thousands of frames
    # carry only a few hundred distinct (environment, label) facts, and a
    # confidence read off the frame count would be wildly overconfident.  Under
    # OBJ-All the label turns over every second or so and the two counts are
    # much closer.  Reported so the two conditions are not compared as though
    # they had the same statistical weight.
    "n_effective_test": int(len(set(zip(
      groups[is_test].tolist(), yte.tolist())))),
  }


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("task")
  p.add_argument("checkpoint")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--steps", type=int, default=3000)
  p.add_argument("--sample-every", type=int, default=10)
  p.add_argument("--swap-at", type=int, default=1500,
                 help="step to run the history-swap counterfactual at")
  p.add_argument("--swap-horizon", type=int, default=25)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=770077)
  p.add_argument("--cadence", default=None)
  p.add_argument("--reset-hidden-on-respawn", action="store_true",
                 help="zero the recurrent state at every object boundary.  "
                      "Gate A asks whether doing this reduces history-swap "
                      "sensitivity, which is only answerable by measuring the "
                      "swap under both settings.")
  p.add_argument("--json", default=None)
  a = p.parse_args()

  env_cfg = load_env_cfg(a.task, play=True)
  agent_cfg = load_rl_cfg(a.task)
  env_cfg.scene.num_envs = a.num_envs
  env_cfg.seed = a.seed
  if a.cadence is not None:
    from piper_push.shapes import ALL_QUANTITIES
    redraw = (ALL_QUANTITIES if a.cadence == "object"
              else () if a.cadence == "episode"
              else tuple(s.strip() for s in a.cadence.split(",") if s.strip()))
    env_cfg.commands["pick"].redraw_on_place = redraw
    env_cfg.commands["pick"].reshape_on_place = bool(redraw)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(a.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=a.device)
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=a.device)
  policy = runner.get_inference_policy(device=a.device)
  if not getattr(policy, "is_recurrent", False):
    print("  policy is not recurrent; nothing to probe")
    return 0
  policy.reset()

  u = env.unwrapped
  pick = u.command_manager.get_term("pick")
  dev = torch.device(a.device)
  n = a.num_envs

  def cur_labels():
    cls = shapes.object_shape_class(u, "object").clone()
    mass = shapes._state(u, "object")["mass"].clone()
    return cls, mass

  lo, hi = 0.05, 0.40  # objects.OBJECT_MASS_RANGE, for the bin edges
  cur_cls, cur_mass = cur_labels()
  prev_cls = cur_cls.clone()
  prev_mass = cur_mass.clone()
  seen = torch.zeros(n, dtype=torch.long, device=dev)  # placements this episode

  feats: list[torch.Tensor] = []
  lab = {k: [] for k in ("cur_cls", "prev_cls", "cur_mass", "prev_mass")}
  groups: list[torch.Tensor] = []
  swap = None

  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  def _clone(h):
    return h.clone() if not isinstance(h, tuple) else tuple(t.clone() for t in h)

  def _roll(h, k):
    return (h.roll(k, dims=1) if not isinstance(h, tuple)
            else tuple(t.roll(k, dims=1) for t in h))

  def _zero(h, mask):
    if isinstance(h, tuple):
      for t in h:
        t[:, mask] = 0.0
    else:
      h[:, mask] = 0.0

  h_main = h_alt = None   # the counterfactual branch, live only after the swap

  with torch.inference_mode():
    for step in range(a.steps):
      # -- history swap ----------------------------------------------------
      # Every environment is given some OTHER environment's past and its own
      # present.  Both branches are then fed the identical observation stream,
      # so any difference in action is attributable to the recurrent state and
      # to nothing else -- no divergence in what the two policies can see.
      if step == a.swap_at:
        h_main = _clone(policy.get_hidden_state())
        h_alt = _roll(h_main, 97 % n)
        swap = {"horizon": []}

      if h_alt is None:
        act = policy(obs)
      else:
        policy.reset(hidden_state=h_main)
        act = policy(obs).clone()
        h_main = _clone(policy.get_hidden_state())

        policy.reset(hidden_state=h_alt)
        alt = policy(obs).clone()
        h_alt = _clone(policy.get_hidden_state())

        scale = act.std(dim=0).clamp(min=1e-6)
        if len(swap["horizon"]) <= a.swap_horizon:
          swap["horizon"].append({
            "k": step - a.swap_at,
            "abs": float((alt - act).abs().mean()),
            "normalised": float(((alt - act).abs() / scale).mean()),
          })
        if step - a.swap_at == a.swap_horizon:
          swap["immediate_abs"] = swap["horizon"][0]["abs"]
          swap["immediate_normalised"] = swap["horizon"][0]["normalised"]
          swap["final_normalised"] = swap["horizon"][-1]["normalised"]
          swap["action_std_across_envs"] = float(act.std(dim=0).mean())
          # Stop paying for the second branch once the window has closed.
          policy.reset(hidden_state=h_main)
          h_alt = None

      out = env.step(act)
      obs, dones = out[0], out[2]
      policy.reset(dones)
      if h_alt is not None and bool(dones.any()):
        # An episode boundary clears BOTH branches, and clears them by writing
        # into each buffer directly.
        #
        # Not by re-reading the policy's hidden state, which is what this did
        # first: at this point the policy is holding the ALT branch's state
        # (it was the last one evaluated), so `h_main = clone(policy.hidden)`
        # silently replaced the main branch with the alt branch.  The two
        # became identical and the measured divergence dropped to exactly
        # zero on the first step any environment finished an episode -- which
        # read as "the memory's influence decays to nothing in five steps"
        # rather than as the bug it was.
        _zero(h_main, dones.bool())
        _zero(h_alt, dones.bool())

      # -- track which object is which -------------------------------------
      placed = pick.just_placed.bool()
      if bool(placed.any()):
        idx = placed.nonzero().flatten()
        prev_cls[idx] = cur_cls[idx]
        prev_mass[idx] = cur_mass[idx]
        seen[idx] += 1
        if a.reset_hidden_on_respawn:
          policy.reset(placed)
          if h_alt is not None:
            # Both branches, written directly -- see the note on the episode
            # boundary below for why this must not re-read the policy.
            _zero(h_main, placed)
            _zero(h_alt, placed)
      cur_cls, cur_mass = cur_labels()
      if bool(dones.any()):
        d = dones.nonzero().flatten()
        seen[d] = 0
        prev_cls[d] = cur_cls[d]
        prev_mass[d] = cur_mass[d]

      # -- sample the hidden state -----------------------------------------
      if step % a.sample_every == 0 and step > 50:
        h = policy.get_hidden_state()
        vec = (h if not isinstance(h, tuple) else h[0])[-1]   # (B, H), last layer
        # Only environments that have already finished at least one object,
        # because "the previous object" is otherwise the same object.
        keep = seen > 0
        if bool(keep.any()):
          feats.append(vec[keep].float().cpu())
          lab["cur_cls"].append(cur_cls[keep].cpu())
          lab["prev_cls"].append(prev_cls[keep].cpu())
          lab["cur_mass"].append(
            ((cur_mass[keep] - lo) / (hi - lo) * MASS_BINS).long().clamp(0, MASS_BINS - 1).cpu())
          lab["prev_mass"].append(
            ((prev_mass[keep] - lo) / (hi - lo) * MASS_BINS).long().clamp(0, MASS_BINS - 1).cpu())
          groups.append(keep.nonzero().flatten().cpu())

  x = torch.cat(feats)
  g = torch.cat(groups)
  print(f"  collected {len(x)} hidden states of width {x.shape[1]} "
        f"from {len(torch.unique(g))} environments")

  report = {
    "task": a.task, "checkpoint": a.checkpoint, "seed": a.seed,
    "cadence": a.cadence,
    "reset_hidden_on_respawn": bool(a.reset_hidden_on_respawn),
    "redraw_on_place": list(pick.redraw_on_place),
    "n_samples": int(len(x)), "hidden_width": int(x.shape[1]),
    "probes": {}, "history_swap": swap,
  }
  for key, k in (("cur_cls", 5), ("prev_cls", 5),
                 ("cur_mass", MASS_BINS), ("prev_mass", MASS_BINS)):
    y = torch.cat(lab[key])
    res = _fit_linear_probe(x, y, g, k)
    report["probes"][key] = res
    print(f"  {key:10s} acc {100 * res['test_accuracy']:5.1f}%  "
          f"maj {100 * res['majority_class_baseline']:5.1f}%  "
          f"lift {100 * res['lift_over_chance']:+5.1f}pp  |  "
          f"balanced {100 * res['balanced_accuracy']:5.1f}% vs "
          f"{100 * res['balanced_chance']:4.1f}% "
          f"({100 * res['balanced_lift']:+5.1f}pp)  "
          f"n_eff={res['n_effective_test']}")

  if swap:
    print(f"  history swap: mean |da| = {swap['immediate_abs']:.4f} "
          f"({swap['immediate_normalised']:.3f} of the across-env action spread)")

  if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(report, indent=1))
    print(f"  wrote {a.json}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
