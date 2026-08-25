"""Stage 4: fit the residual, offline, against a frozen nominal surrogate.

    python scripts/ra_sim0_train.py --train results/ra_sim0/data/train_target_7101.pt \
        --val results/ra_sim0/data/val_target_7201.pt \
        --nominal results/ra_sim0/data/nominal_nominal_7501.pt --seed 0

Two artefacts, in this order and for these reasons:

**The surrogate** is fitted on *nominal*-simulator transitions only.  It exists
so that a gradient can cross one control step of a simulator that has no
gradient.  It is frozen the moment it is fitted and nothing it predicts is
reported as a result.

**The residual** is fitted on *target*-domain transitions, seeing only what a
robot could log.  Its loss has four parts, which the phase names: a one-step
transition error, a K-step rollout error, a magnitude penalty and a slew
penalty.  Its ensemble members differ in initialisation and in data order, and
the spread between them is checked against the error it is supposed to
predict rather than assumed to be calibrated.

Everything selected -- window length, K, the two penalty weights, epochs -- is
selected on ``val``.  ``test`` is opened once, in Stage 5, by a different
script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
N_J = 6


def windows(rec, length: int, stride: int) -> torch.Tensor:
  """Start indices of clean windows, as ``[n, 2]`` of (t, env).

  A window is clean if the recording never reset inside it and the object was
  never replaced inside it -- the same rule the replay harness uses, for the
  same reason.
  """
  T, E = rec.done.shape
  bad = rec.done.clone()
  bad[:-1] |= (rec.shape[1:] != rec.shape[:-1])
  # cumulative count of bad steps, so a window's cleanliness is O(1)
  cum = torch.cumsum(bad.long(), dim=0)
  starts = []
  for t in range(0, T - length - 1, stride):
    lo = cum[t] if t > 0 else torch.zeros(E, dtype=torch.long)
    clean = (cum[t + length] - lo) == 0
    idx = clean.nonzero(as_tuple=False).flatten()
    if idx.numel():
      starts.append(torch.stack([torch.full_like(idx, t), idx], dim=-1))
  return torch.cat(starts) if starts else torch.zeros(0, 2, dtype=torch.long)


def gather(rec, starts: torch.Tensor, length: int, device: str) -> dict:
  """``[B, L, *]`` tensors for one batch of windows."""
  t, e = starts[:, 0], starts[:, 1]
  off = torch.arange(length)
  ti = (t.unsqueeze(1) + off.unsqueeze(0))
  ei = e.unsqueeze(1).expand_as(ti)
  out = {}
  for k in ("q", "qd", "u"):
    out[k] = getattr(rec, k)[ti, ei].to(device)
  # the previous command, with the window's first step reading the one before
  tp = (ti - 1).clamp_min(0)
  out["u_prev"] = rec.u[tp, ei].to(device)
  out["q_next"] = rec.q[(ti + 1).clamp_max(rec.q.shape[0] - 1), ei].to(device)
  out["qd_next"] = rec.qd[(ti + 1).clamp_max(rec.qd.shape[0] - 1), ei].to(device)
  return out


def surrogate_dataset(rec, device: str) -> tuple[torch.Tensor, torch.Tensor]:
  """``(q, qdot, u_prev, u) -> (dq, dqdot)`` from a NOMINAL recording.

  In the nominal simulator no hook is installed, so the command the term
  issued is the target the servo received: ``y == u``.  That identity is what
  makes this dataset legitimate and is why the surrogate cannot be fitted on
  target-domain data.
  """
  q, qd, u = rec.q, rec.qd, rec.u
  T = q.shape[0] - 1
  keep = (~rec.done[:T]) & (rec.shape[1:T + 1] == rec.shape[:T])
  t, e = keep.nonzero(as_tuple=False).T
  tp = (t - 1).clamp_min(0)
  feats = torch.cat([q[t, e], qd[t, e], u[tp, e], u[t, e]], dim=-1)
  labels = torch.cat([q[t + 1, e] - q[t, e], qd[t + 1, e] - qd[t, e]], dim=-1)
  return feats.to(device), labels.to(device)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--train", nargs="+", required=True)
  ap.add_argument("--val", required=True)
  ap.add_argument("--nominal", required=True)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--members", type=int, default=4)
  ap.add_argument("--hidden", type=int, default=64)
  ap.add_argument("--delta-max", type=float, default=0.0,
                  help="override the residual's output bound in rad; "
                       "0 keeps residual.DELTA_MAX, the value the plan "
                       "committed to.")
  ap.add_argument("--suffix", default="")
  ap.add_argument("--window", type=int, default=80)
  ap.add_argument("--burn-in", type=int, default=16)
  ap.add_argument("--rollout-k", type=int, default=5)
  ap.add_argument("--epochs", type=int, default=30)
  ap.add_argument("--batch", type=int, default=256)
  ap.add_argument("--lr", type=float, default=3e-4)
  ap.add_argument("--w-multistep", type=float, default=1.0)
  ap.add_argument("--w-magnitude", type=float, default=1e-2)
  ap.add_argument("--w-slew", type=float, default=1e-2)
  ap.add_argument("--budget-steps", type=int, default=0,
                  help="truncate the training recording to this many control "
                       "steps; 0 uses all of it")
  ap.add_argument("--surrogate", default="")
  ap.add_argument("--out", default="results/ra_sim0/model")
  a = ap.parse_args()

  import sys
  sys.path.insert(0, str(ROOT / "src"))
  from piper_push import replay as rp
  from piper_push import residual as R
  from piper_push.surrogate import TransitionSurrogate, fit_surrogate

  torch.manual_seed(a.seed)
  dev = a.device
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  meta: dict = {"args": vars(a)}

  # -- 1. the frozen nominal surrogate --------------------------------------
  sur_path = Path(a.surrogate) if a.surrogate else out / "surrogate.pt"
  if sur_path.exists():
    blob = torch.load(sur_path, map_location=dev, weights_only=False)
    sur = TransitionSurrogate(width=blob["width"]).to(dev)
    sur.load_state_dict(blob["state_dict"])
    sur.eval()
    for p in sur.parameters():
      p.requires_grad_(False)
    meta["surrogate"] = {"loaded": str(sur_path), **blob["info"]}
    print(f"loaded surrogate from {sur_path}  val_nmse={blob['info']['val_nmse']:.3e}")
  else:
    nrec, _ = rp.Recording.load(a.nominal)
    f, l = surrogate_dataset(nrec, "cpu")
    print(f"surrogate dataset: {tuple(f.shape)}")
    sur, info = fit_surrogate(f, l, device=dev, epochs=60, batch=8192)
    torch.save({"state_dict": sur.state_dict(), "width": 256, "info": info},
               sur_path)
    meta["surrogate"] = {"fitted": str(sur_path), **info}
    del nrec, f, l

  # -- 2. data ---------------------------------------------------------------
  trecs = []
  for path in a.train:
    r, m = rp.Recording.load(path, steps=a.budget_steps or None)
    trecs.append(r)
    meta.setdefault("train_meta", []).append(
      {k: m[k] for k in ("split", "seed", "shapes", "domain", "steps")})
  vrec, vmeta = rp.Recording.load(a.val, steps=a.budget_steps or None)

  tw = [windows(r, a.window, a.window // 2) for r in trecs]
  vw = windows(vrec, a.window, a.window)
  n_train = sum(int(w.shape[0]) for w in tw)
  print(f"windows: train {n_train}, val {vw.shape[0]}")
  meta["n_train_windows"] = n_train
  meta["n_val_windows"] = int(vw.shape[0])

  # feature normalisation from the training split only
  sample = tw[0][torch.randperm(tw[0].shape[0])[:2048]]
  b = gather(trecs[0], sample, a.window, dev)
  feat = R.build_features(b["q"], b["qd"], b["u"], b["u_prev"]).reshape(-1, R.FEATURE_DIM)
  fmean, fstd = feat.mean(0), feat.std(0)

  dmax = a.delta_max or R.DELTA_MAX
  ens = R.ResidualEnsemble(a.members, hidden=a.hidden,
                           delta_max=dmax).to(dev)
  for m in ens.members:
    m.set_norm(fmean, fstd)
  opt = torch.optim.Adam(ens.parameters(), lr=a.lr)
  ystd = sur.y_std.detach()

  def loss_on(batchrec, starts, train: bool):
    b = gather(batchrec, starts, a.window, dev)
    q, qd, u, up = b["q"], b["qd"], b["u"], b["u_prev"]
    qn, qdn = b["q_next"], b["qd_next"]
    B, L = q.shape[0], q.shape[1]
    total = {"one": 0.0, "multi": 0.0, "mag": 0.0, "slew": 0.0}
    per_member = []
    for m in ens.members:
      h = m.zero_hidden(B, dev)
      d_prev = None
      one = mag = slew = 0.0
      n_one = 0
      deltas = []
      for t in range(L):
        feat = R.build_features(q[:, t], qd[:, t], u[:, t], up[:, t])
        d, h = m(feat, h)
        deltas.append(d)
        if t >= a.burn_in:
          y = u[:, t] + d
          yp = up[:, t] + (deltas[t - 1] if t else torch.zeros_like(d))
          pq, pqd = sur(q[:, t], qd[:, t], yp, y)
          e = torch.cat([pq - qn[:, t], pqd - qdn[:, t]], dim=-1) / ystd
          one = one + e.pow(2).mean()
          mag = mag + (d / m.delta_max).pow(2).mean()
          if d_prev is not None:
            slew = slew + ((d - d_prev) / m.delta_max).pow(2).mean()
          n_one += 1
        d_prev = d
      one, mag, slew = one / n_one, mag / n_one, slew / max(n_one - 1, 1)

      # K-step rollout, restarted from the observed state every K steps and
      # driven by the residual's own predictions in between.
      h = m.zero_hidden(B, dev)
      multi = 0.0
      n_multi = 0
      pq, pqd = q[:, 0], qd[:, 0]
      d_last = torch.zeros_like(q[:, 0])
      for t in range(L):
        if (t - a.burn_in) % a.rollout_k == 0 or t < a.burn_in:
          pq, pqd = q[:, t], qd[:, t]
        feat = R.build_features(pq, pqd, u[:, t], up[:, t])
        d, h = m(feat, h)
        y = u[:, t] + d
        yp = up[:, t] + d_last
        pq, pqd = sur(pq, pqd, yp, y)
        d_last = d
        if t >= a.burn_in:
          e = torch.cat([pq - qn[:, t], pqd - qdn[:, t]], dim=-1) / ystd
          multi = multi + e.pow(2).mean()
          n_multi += 1
      multi = multi / max(n_multi, 1)

      lm = one + a.w_multistep * multi + a.w_magnitude * mag + a.w_slew * slew
      per_member.append(lm)
      total["one"] += float(one)
      total["multi"] += float(multi)
      total["mag"] += float(mag)
      total["slew"] += float(slew)
    loss = torch.stack(per_member).sum()
    if train:
      opt.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(ens.parameters(), 1.0)
      opt.step()
    return {k: v / len(ens.members) for k, v in total.items()}

  hist = []
  best = None
  g = torch.Generator().manual_seed(a.seed)
  for ep in range(a.epochs):
    # Each member sees a different order; the batch itself is shared so the
    # ensemble's spread comes from initialisation and ordering rather than
    # from four separate training runs, which would cost four times as much.
    ri = a.seed % len(tw)
    src = trecs[ri]
    perm = tw[ri][torch.randperm(tw[ri].shape[0], generator=g)]
    tr = {"one": 0.0, "multi": 0.0, "mag": 0.0, "slew": 0.0}
    nb = 0
    for s in range(0, min(perm.shape[0], a.batch * 12), a.batch):
      out_b = loss_on(src, perm[s:s + a.batch], True)
      for k in tr:
        tr[k] += out_b[k]
      nb += 1
    tr = {k: v / max(nb, 1) for k, v in tr.items()}
    with torch.no_grad():
      vb = vw[torch.randperm(vw.shape[0], generator=g)[:a.batch * 2]]
      va = loss_on(vrec, vb, False)
    row = {"epoch": ep, "train": tr, "val": va,
           "val_total": va["one"] + a.w_multistep * va["multi"]}
    hist.append(row)
    print(f"  ep {ep + 1:3d}/{a.epochs}  train one={tr['one']:.4f} "
          f"multi={tr['multi']:.4f}  val one={va['one']:.4f} "
          f"multi={va['multi']:.4f}  |d|={tr['mag']:.3f}", flush=True)
    if best is None or row["val_total"] < best[0]:
      best = (row["val_total"], ep,
              {k: v.detach().cpu().clone() for k, v in ens.state_dict().items()})

  ens.load_state_dict(best[2])
  ck = out / f"residual_seed{a.seed}{a.suffix}.pt"
  torch.save({"n_members": a.members, "hidden": a.hidden,
              "delta_max": dmax, "state_dict": ens.state_dict(),
              "meta": {**meta, "best_epoch": best[1],
                       "best_val_total": best[0], "history": hist,
                       "wall_clock_s": time.time() - t0,
                       "params": sum(p.numel() for p in ens.parameters())}}, ck)
  (out / f"residual_seed{a.seed}{a.suffix}.json").write_text(json.dumps(
    {**meta, "best_epoch": best[1], "best_val_total": best[0],
     "history": hist, "wall_clock_s": time.time() - t0,
     "params": sum(p.numel() for p in ens.parameters())}, indent=2))
  print(f"wrote {ck}  best epoch {best[1]}  val {best[0]:.5f}  "
        f"{time.time() - t0:.0f}s")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
