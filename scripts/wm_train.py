"""Fit the parameter-conditioned dynamics ensemble on simulation logs.

    python scripts/wm_train.py --data results/wm1_latency/data \
        --out results/wm1_latency/model --members 4 --epochs 6

Nothing in here reads reward, success or the target split.  The only thing the
latency label is used for is to *condition* the model -- it is an input,
``theta``, exactly as it will be at inference time when every candidate value
is tried in turn.

Model selection is held-out one-step and multi-step prediction on ``val``, and
the run also reports the same on ``valh`` (unseen object shapes) so that a
model which only works on the objects it was fitted to is visible before
anything downstream depends on it.

The last table is the one that matters most: for validation windows whose true
latency is known, the NLL under each candidate.  If the true candidate is not
the cheapest there, no posterior built on this model can work, and the right
response is to fix the model rather than to run PPO.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from piper_push import latency, wm_data, wm_model

BURN_IN = 8
HORIZON = 16
ROLLOUT = 8


load_split = wm_data.load_split
batch_from = wm_data.batch_from


def make_index(sessions, stride: int, length: int):
  return wm_data.make_index(sessions, stride, length, BURN_IN)


def losses(m, norms, b, theta, weights):
  """One-step NLL on every channel, plus an open-loop rollout."""
  z = norms["z"](b["enc"])
  p = norms["p"](b["proprio"])
  a = norms["a"](b["action"])
  e = norms["e"](b["servo"])

  _, h = m.teacher_forced(z[:BURN_IN], p[:BURN_IN], a[:BURN_IN], theta, None)
  lo, hi = BURN_IN, BURN_IN + HORIZON
  out, _ = m.teacher_forced(z[lo:hi], p[lo:hi], a[lo:hi], theta, h)
  z_mu, z_lv, dp, p_lv, e_mu, e_lv = out
  l_z = wm_model.gaussian_nll(z[lo + 1:hi + 1], z_mu, z_lv).sum(-1).mean()
  l_p = wm_model.gaussian_nll(p[lo + 1:hi + 1] - p[lo:hi], dp, p_lv).sum(-1).mean()
  l_e = wm_model.gaussian_nll(e[lo + 1:hi + 1], e_mu, e_lv).sum(-1).mean()

  ro, _ = m.open_loop(z[lo], p[lo], a[lo:lo + ROLLOUT], theta, h)
  l_m = wm_model.gaussian_nll(z[lo + 1:lo + 1 + ROLLOUT], ro[0], ro[1]).sum(-1).mean()

  total = (weights["z"] * l_z + weights["p"] * l_p + weights["e"] * l_e
           + weights["m"] * l_m)
  return total, {"z": float(l_z), "p": float(l_p), "e": float(l_e),
                 "multi": float(l_m)}


@torch.no_grad()
def evaluate(members, norms, sessions, idx, length, device, batch=512,
             theta_override=None, limit=None):
  """Mean per-channel NLL over an index, optionally under a forced theta."""
  weights = {"z": 1.0, "p": 1.0, "e": 1.0, "m": 1.0}
  order = list(range(len(idx)))
  if limit is not None and len(order) > limit:
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(order), generator=g)[:limit].tolist()
  acc, n = {}, 0
  for i in range(0, len(order), batch):
    sel = order[i:i + batch]
    b = batch_from(sessions, idx, sel, length, device)
    theta = (torch.full_like(b["_lag"], theta_override)
             if theta_override is not None else b["_lag"])
    for m in members:
      _, parts = losses(m, norms, b, theta, weights)
      for k, v in parts.items():
        acc[k] = acc.get(k, 0.0) + v * len(sel)
    n += len(sel) * len(members)
  return {k: v / max(n, 1) for k, v in acc.items()}


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--data", default="results/wm1_latency/data")
  p.add_argument("--out", default="results/wm1_latency/model")
  p.add_argument("--members", type=int, default=4)
  p.add_argument("--epochs", type=int, default=6)
  p.add_argument("--batch", type=int, default=256)
  p.add_argument("--stride", type=int, default=16)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--hidden", type=int, default=192)
  p.add_argument("--subsample", type=float, default=0.8,
                 help="fraction of windows each ensemble member sees")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--shuffle-theta", action="store_true",
                 help="permutation control: train with the conditioning shuffled "
                      "within each batch, so theta carries no information.  "
                      "Any posterior built on the result is the floor.")
  a = p.parse_args()

  root = Path(a.data)
  length = BURN_IN + HORIZON + 1
  dev = a.device

  train = load_split(root, "train")
  val = load_split(root, "val")
  valh = load_split(root, "valh")
  tr_idx, tr_lags = make_index(train, a.stride, length)
  va_idx, _ = make_index(val, a.stride * 2, length)
  vh_idx, _ = make_index(valh, a.stride * 2, length)
  print(f"  windows: train {len(tr_idx):,}  val {len(va_idx):,}  "
        f"valh {len(vh_idx):,}")
  print(f"  train lag counts: "
        f"{torch.bincount(tr_lags, minlength=len(latency.LAGS)).tolist()}")

  # Normalisation from the training split alone.  Fitted on a sample of the
  # windows rather than the raw tensors so that it matches what the model
  # actually sees.
  g = torch.Generator().manual_seed(a.seed)
  fit_sel = torch.randperm(len(tr_idx), generator=g)[:4096].tolist()
  fb = batch_from(train, tr_idx, fit_sel, length, dev)
  norms = {
    "z": wm_model.Norm.fit(fb["enc"]),
    "p": wm_model.Norm.fit(fb["proprio"]),
    "a": wm_model.Norm.fit(fb["action"]),
    "e": wm_model.Norm.fit(fb["servo"]),
  }
  dims = {"z": fb["enc"].shape[-1], "p": fb["proprio"].shape[-1],
          "a": fb["action"].shape[-1]}
  print(f"  dims: {dims}")

  weights = {"z": 1.0, "p": 1.0, "e": 1.0, "m": 0.5}
  members, histories = [], []
  t0 = time.time()
  for k in range(a.members):
    torch.manual_seed(a.seed * 100 + k)
    m = wm_model.LatentDynamics(dims["z"], dims["p"], dims["a"],
                                len(latency.LAGS), hidden=a.hidden).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    gk = torch.Generator().manual_seed(a.seed * 1000 + k)
    keep = torch.randperm(len(tr_idx), generator=gk)[
      : int(a.subsample * len(tr_idx))].tolist()
    hist = []
    for ep in range(a.epochs):
      perm = torch.randperm(len(keep), generator=gk).tolist()
      run, nb = {}, 0
      for i in range(0, len(perm), a.batch):
        sel = [keep[j] for j in perm[i:i + a.batch]]
        b = batch_from(train, tr_idx, sel, length, dev)
        theta = b["_lag"]
        if a.shuffle_theta:
          theta = theta[torch.randperm(len(theta), generator=gk).to(theta.device)]
        loss, parts = losses(m, norms, b, theta, weights)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        for kk, v in parts.items():
          run[kk] = run.get(kk, 0.0) + v
        nb += 1
      hist.append({kk: v / nb for kk, v in run.items()})
      print(f"  member {k} epoch {ep}: "
            + "  ".join(f"{kk} {v / nb:8.3f}" for kk, v in run.items()),
            flush=True)
    members.append(m.eval())
    histories.append(hist)
  train_s = time.time() - t0

  ens = wm_model.Ensemble(members, norms)
  outdir = Path(a.out)
  outdir.mkdir(parents=True, exist_ok=True)
  name = "ensemble_shuffled.pt" if a.shuffle_theta else "ensemble.pt"
  ens.save(outdir / name)

  # -- held-out prediction --------------------------------------------------
  report = {
    "config": vars(a), "dims": dims, "burn_in": BURN_IN, "horizon": HORIZON,
    "rollout": ROLLOUT, "weights": weights,
    "n_windows": {"train": len(tr_idx), "val": len(va_idx), "valh": len(vh_idx)},
    "train_wall_clock_s": train_s,
    "history": histories,
    "held_out": {
      "val": evaluate(members, norms, val, va_idx, length, dev, limit=6000),
      "valh": evaluate(members, norms, valh, vh_idx, length, dev, limit=6000),
    },
  }
  print()
  for k, v in report["held_out"].items():
    print(f"  {k:6s} " + "  ".join(f"{kk} {vv:8.3f}" for kk, vv in v.items()))

  # -- does conditioning do anything? ---------------------------------------
  # For validation windows of known latency, the NLL under every candidate.
  # The diagonal has to be the cheapest column or the posterior cannot work,
  # and finding that out here costs a minute rather than a PPO run.
  print()
  print("  NLL(z) by true lag (row) under each candidate (column):")
  grid = {}
  for true_lag in latency.LAGS:
    sub = [(s, n) for s, n in val if s.lag == true_lag]
    if not sub:
      continue
    si, _ = make_index(sub, a.stride * 4, length)
    row = []
    for cand in latency.LAGS:
      r = evaluate(members, norms, sub, si, length, dev,
                   theta_override=cand, limit=2000)
      row.append(r["z"] + r["p"] + r["e"])
    grid[true_lag] = row
    best = min(range(len(row)), key=lambda i: row[i])
    mark = "  <-- correct" if latency.LAGS[best] == true_lag else "  <-- WRONG"
    print(f"    lag {true_lag}: " + " ".join(f"{v:9.3f}" for v in row) + mark)
  report["candidate_grid"] = {str(k): v for k, v in grid.items()}
  report["candidate_grid_argmin_correct"] = {
    str(k): latency.LAGS[min(range(len(v)), key=lambda i: v[i])] == k
    for k, v in grid.items()}

  rname = ("train_report_shuffled.json" if a.shuffle_theta
           else "train_report.json")
  (outdir / rname).write_text(json.dumps(report, indent=1))
  print(f"\n  wrote {outdir / name} and {rname} ({train_s / 60:.1f} min)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
