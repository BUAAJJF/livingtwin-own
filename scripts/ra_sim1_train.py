"""Fit the stable stateful actuator model, offline, against a frozen surrogate.

    python scripts/ra_sim1_train.py --seed 0 --device cuda:0

The data windowing is imported from `scripts/ra_sim0_train.py` rather than
copied, so the two phases cut their training windows identically and cannot
drift apart; the surrogate is loaded from RA-Sim-0 and never refitted.

Everything the loss weighs is in docs/ra_sim1_experiment_plan.md and is not
tuned against a result.  Supervision is observable transitions only: the
hidden target's effective command is not a label and is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
  "ra0_train", ROOT / "scripts" / "ra_sim0_train.py")
_ra0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ra0)
windows, gather = _ra0.windows, _ra0.gather


def sha256(p: Path) -> str:
  h = hashlib.sha256()
  with open(p, "rb") as fh:
    for c in iter(lambda: fh.read(1 << 20), b""):
      h.update(c)
  return h.hexdigest()


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--train", default="results/ra_sim0/data/train_target_7101.pt")
  ap.add_argument("--val", default="results/ra_sim0/data/val_target_7201.pt")
  ap.add_argument("--surrogate", default="results/ra_sim0/model/surrogate.pt")
  ap.add_argument("--budget-steps", type=int, default=3000,
                  help="60 s of arm time, the budget RA-Sim-0's ladder showed "
                       "saturating.  More data is not allowed to hide a model "
                       "problem.")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--hidden", type=int, default=64)
  ap.add_argument("--window", type=int, default=80)
  ap.add_argument("--burn-in", type=int, default=16)
  ap.add_argument("--epochs", type=int, default=40)
  ap.add_argument("--batch", type=int, default=256)
  ap.add_argument("--batches-per-epoch", type=int, default=12)
  ap.add_argument("--lr", type=float, default=3e-4)
  ap.add_argument("--w-rate", type=float, default=1e-3)
  ap.add_argument("--w-hidden", type=float, default=1e-4)
  ap.add_argument("--w-smooth", type=float, default=1e-3)
  ap.add_argument("--out", default="results/ra_sim1/model")
  a = ap.parse_args()

  import sys
  sys.path.insert(0, str(ROOT / "src"))
  from piper_push import replay as rp
  from piper_push import robot as piper
  from piper_push.actuator import (ALPHA_RANGE, BIAS_RANGE_RAD,
                                   RATE_RANGE_RAD_S, StableActuator,
                                   build_features)
  from piper_push.surrogate import TransitionSurrogate

  torch.manual_seed(a.seed)
  dev = a.device
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  t0 = time.time()

  blob = torch.load(a.surrogate, map_location=dev, weights_only=False)
  sur = TransitionSurrogate(width=blob["width"]).to(dev)
  sur.load_state_dict(blob["state_dict"])
  sur.eval()
  for p in sur.parameters():
    p.requires_grad_(False)
  ystd = sur.y_std.detach()

  trec, tmeta = rp.Recording.load(a.train, steps=a.budget_steps)
  vrec, _ = rp.Recording.load(a.val, steps=a.budget_steps)
  tw = windows(trec, a.window, a.window // 2)
  vw = windows(vrec, a.window, a.window)
  print(f"windows: train {tw.shape[0]}, val {vw.shape[0]}  "
        f"(budget {a.budget_steps} steps = {a.budget_steps / 50:.0f} s)")

  lo = torch.tensor([piper.SAFE_TARGET_CLIP[f"joint{i}"][0] for i in range(1, 7)])
  hi = torch.tensor([piper.SAFE_TARGET_CLIP[f"joint{i}"][1] for i in range(1, 7)])
  model = StableActuator(hidden=a.hidden, dt=0.02,
                         command_lo=tuple(lo.tolist()),
                         command_hi=tuple(hi.tolist())).to(dev)

  sample = tw[torch.randperm(tw.shape[0])[:2048]]
  b = gather(trec, sample, a.window, dev)
  feat = build_features(b["q"], b["qd"], b["u"], b["u_prev"], b["u"])
  model.set_norm(feat.reshape(-1, feat.shape[-1]).mean(0),
                 feat.reshape(-1, feat.shape[-1]).std(0))
  opt = torch.optim.Adam(model.parameters(), lr=a.lr)
  rate_max = RATE_RANGE_RAD_S[1] * model.dt

  def run_batch(rec, starts, train: bool) -> dict:
    b = gather(rec, starts, a.window, dev)
    q, qd, u, up = b["q"], b["qd"], b["u"], b["u_prev"]
    qn, qdn = b["q_next"], b["qd_next"]
    B, L = q.shape[0], q.shape[1]
    terms = {"one": 0.0, "r10": 0.0, "r25": 0.0, "rate": 0.0,
             "hid": 0.0, "smooth": 0.0}
    tensors = {}

    # -- pass A: teacher-forced states, which is what inference sees ---------
    h = model.zero_hidden(B, dev)
    w = q[:, 0].clone()
    one = rate = hid = smooth = 0.0
    n = 0
    prev_c = None
    ueffs = []
    for t in range(L):
      u_eff, w, h, d = model.step(q[:, t], qd[:, t], u[:, t], up[:, t], w, h)
      ueffs.append(u_eff)
      c = torch.stack([d["alpha"], d["rate_pos"] / RATE_RANGE_RAD_S[1],
                       d["rate_neg"] / RATE_RANGE_RAD_S[1],
                       d["bias"] / BIAS_RANGE_RAD[1]], dim=-1)
      if t >= a.burn_in:
        yp = ueffs[t - 1] if t else u_eff
        pq, pqd = sur(q[:, t], qd[:, t], yp, u_eff)
        e = torch.cat([pq - qn[:, t], pqd - qdn[:, t]], dim=-1) / ystd
        one = one + e.pow(2).mean()
        rate = rate + (d["delta"] / rate_max).pow(2).mean()
        hid = hid + h.pow(2).mean()
        if prev_c is not None:
          smooth = smooth + (c - prev_c).pow(2).mean()
        n += 1
      prev_c = c
    one, rate, hid = one / n, rate / n, hid / n
    smooth = smooth / max(n - 1, 1)

    # -- passes B and C: K-step rollout through the frozen surrogate ---------
    rolls = {}
    for K, key in ((10, "r10"), (25, "r25")):
      h = model.zero_hidden(B, dev)
      w = q[:, 0].clone()
      pq, pqd = q[:, 0], qd[:, 0]
      prev_eff = u[:, 0]
      loss = 0.0
      m = 0
      for t in range(L):
        if t < a.burn_in or (t - a.burn_in) % K == 0:
          pq, pqd = q[:, t], qd[:, t]
        u_eff, w, h, _d = model.step(pq, pqd, u[:, t], up[:, t], w, h)
        pq, pqd = sur(pq, pqd, prev_eff, u_eff)
        prev_eff = u_eff
        if t >= a.burn_in:
          e = torch.cat([pq - qn[:, t], pqd - qdn[:, t]], dim=-1) / ystd
          loss = loss + e.pow(2).mean()
          m += 1
      rolls[key] = loss / max(m, 1)

    total = (one + rolls["r10"] + rolls["r25"]
             + a.w_rate * rate + a.w_hidden * hid + a.w_smooth * smooth)
    if train:
      opt.zero_grad(set_to_none=True)
      total.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      opt.step()
    for k, v in (("one", one), ("r10", rolls["r10"]), ("r25", rolls["r25"]),
                 ("rate", rate), ("hid", hid), ("smooth", smooth)):
      terms[k] = float(v)
    terms["total"] = float(total)
    return terms

  g = torch.Generator().manual_seed(a.seed)
  hist, best = [], None
  for ep in range(a.epochs):
    perm = tw[torch.randperm(tw.shape[0], generator=g)]
    acc = None
    nb = 0
    for s in range(0, min(perm.shape[0], a.batch * a.batches_per_epoch), a.batch):
      r = run_batch(trec, perm[s:s + a.batch], True)
      acc = r if acc is None else {k: acc[k] + v for k, v in r.items()}
      nb += 1
    tr = {k: v / nb for k, v in acc.items()}
    with torch.no_grad():
      vb = vw[torch.randperm(vw.shape[0], generator=g)[:a.batch * 2]]
      va = run_batch(vrec, vb, False)
    hist.append({"epoch": ep, "train": tr, "val": va})
    print(f"  ep {ep + 1:3d}/{a.epochs}  train one={tr['one']:.4f} "
          f"r10={tr['r10']:.4f} r25={tr['r25']:.4f} | val one={va['one']:.4f} "
          f"r10={va['r10']:.4f} r25={va['r25']:.4f}", flush=True)
    key = va["one"] + va["r10"] + va["r25"]
    if best is None or key < best[0]:
      best = (key, ep, {k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()})

  model.load_state_dict(best[2])
  ck = out / f"actuator_seed{a.seed}.pt"
  torch.save({"state_dict": model.state_dict(), "hidden": a.hidden, "dt": 0.02,
              "cmd_lo": lo.tolist(), "cmd_hi": hi.tolist(),
              "alpha_range": list(ALPHA_RANGE),
              "rate_range": list(RATE_RANGE_RAD_S),
              "bias_range": list(BIAS_RANGE_RAD),
              "meta": {"args": vars(a), "best_epoch": best[1],
                       "best_val": best[0], "history": hist,
                       "params": sum(p.numel() for p in model.parameters()),
                       "surrogate_sha256": sha256(Path(a.surrogate)),
                       "train_sha_prefix": sha256(Path(a.train))[:16],
                       "wall_clock_s": time.time() - t0}}, ck)
  info = {"checkpoint": str(ck), "sha256": sha256(ck), "best_epoch": best[1],
          "best_val": best[0], "wall_clock_s": time.time() - t0,
          "params": sum(p.numel() for p in model.parameters()),
          "history": hist, "args": vars(a)}
  (out / f"actuator_seed{a.seed}.json").write_text(json.dumps(info, indent=2))
  print(f"wrote {ck}  best epoch {best[1]}  val {best[0]:.5f}  "
        f"{time.time() - t0:.0f}s")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
