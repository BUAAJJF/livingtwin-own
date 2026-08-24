"""Fit the deployable risk head, and check it generalises off its own domains.

    python scripts/wm_risk.py --data results/wm1_damping/data \
        --out results/wm1_damping/model --device cuda:0

Three heads come out of this and the second two are what make the first
interpretable.

``risk``            trained on every candidate domain.  This is the one the
                    decision score uses: a practitioner would fit it across
                    their whole prior before deploying anything.
``risk_heldout``    trained on nominal and the counter-direction only, never on
                    the target.  The specification asks for validation on a
                    held-out *domain*, and this is it: if a head fitted without
                    the target still ranks risk inside it, the score does not
                    depend on having guessed the answer in advance.
``risk_shuffled``   the same architecture and the same number of updates with
                    the labels permuted.  Whatever this reaches is the floor.

Everything is fitted on simulator trip labels, which the phase specification
allows, and every head is frozen before a target session is read.  The head's
inputs are deployable channels only, enforced at construction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from piper_push import damping, risk, wm_data

LENGTH = 25          # the window the posterior stage uses, so the two agree
BURN_IN = 0


def build_index(sessions, stride: int, horizon: int = risk.HORIZON):
  """``(session_id, t0, env)`` for every window, with its label and domain."""
  idx, y, dom = [], [], []
  for si, (s, _) in enumerate(sessions):
    for env in range(s.n_envs):
      starts = [t0 for t0, b in wm_data.windows(
        s, length=LENGTH - BURN_IN, burn_in=BURN_IN, stride=stride,
        envs=torch.tensor([env]))]
      if not starts:
        continue
      lab = risk.labels_within_horizon(s.trip[:, env], starts, LENGTH,
                                       horizon)
      idx.extend((si, t0, env) for t0 in starts)
      y.append(lab)
      dom.extend([float(s.lag)] * len(starts))
  return idx, torch.cat(y), torch.tensor(dom)


def batch(sessions, idx, sel, device):
  return wm_data.batch_from(sessions, idx, sel, LENGTH, device,
                            keys=risk.RiskHead.CHANNELS)


@torch.no_grad()
def evaluate(model, sessions, idx, y, dom, device, limit=40000):
  order = list(range(len(idx)))
  if len(order) > limit:
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(order), generator=g)[:limit].tolist()
  probs = []
  for i in range(0, len(order), 512):
    sel = order[i:i + 512]
    probs.append(model.probability(batch(sessions, idx, sel, device)).cpu())
  p = torch.cat(probs)
  lab = y[order]
  d = dom[order]
  out = {"n": len(p), "base_rate": float(lab.mean()),
         "average_precision": risk.average_precision(p, lab),
         "roc_auc": risk.roc_auc(p, lab), "brier": risk.brier(p, lab),
         "calibration": risk.calibration(p, lab),
         "per_domain": {}}
  for v in damping.VALUES:
    m = (d - v).abs() < 1e-9
    if m.sum() == 0:
      continue
    out["per_domain"][str(v)] = {
      "n": int(m.sum()), "base_rate": float(lab[m].mean()),
      "average_precision": risk.average_precision(p[m], lab[m]),
      "roc_auc": risk.roc_auc(p[m], lab[m]),
      "mean_predicted": float(p[m].mean()),
    }
  return out


def train(model, sessions, idx, y, device, epochs, bs, lr, seed, shuffle=False,
          cap: float = 20.0):
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  g = torch.Generator().manual_seed(seed)
  labels = y[torch.randperm(len(y), generator=g)] if shuffle else y
  # Rare positives: without this the minimiser answers "never" and is right
  # 97% of the time.
  pos = float(labels.mean())
  pos_weight = torch.tensor(
    min((1.0 - pos) / max(pos, 1e-6), cap), device=device)
  hist = []
  for ep in range(epochs):
    perm = torch.randperm(len(idx), generator=g).tolist()
    run, nb = 0.0, 0
    for i in range(0, len(perm), bs):
      sel = perm[i:i + bs]
      logit = model(batch(sessions, idx, sel, device))
      loss = F.binary_cross_entropy_with_logits(
        logit, labels[sel].to(device), pos_weight=pos_weight)
      opt.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      opt.step()
      run += float(loss)
      nb += 1
    hist.append(run / nb)
    print(f"    epoch {ep}: loss {run / nb:.4f}", flush=True)
  return hist


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--data", default="results/wm1_damping/data")
  p.add_argument("--out", default="results/wm1_damping/model")
  p.add_argument("--epochs", type=int, default=4)
  p.add_argument("--batch", type=int, default=256)
  p.add_argument("--stride", type=int, default=13)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--hidden", type=int, default=96)
  p.add_argument("--horizon", type=int, default=risk.HORIZON)
  p.add_argument("--pos-weight-cap", type=float, default=20.0,
                 help="ceiling on the positive class weight.  Uncapped at a "
                      "0.8%% base rate it is 123, and the minimiser answers "
                      "0.5 for everything: calibration destroyed, ranking "
                      "unchanged, and every level-reading metric meaningless.")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()

  root, dev = Path(a.data), a.device
  train_s = wm_data.load_split(root, "train")
  val_s = wm_data.load_split(root, "val")
  valh_s = wm_data.load_split(root, "valh")
  for name, ss in (("train", train_s), ("val", val_s), ("valh", valh_s)):
    if any(s.trip is None for s, _ in ss):
      raise SystemExit(f"{name} has files without the trip channel; "
                       "regenerate them with the current collector")

  tr_idx, tr_y, tr_dom = build_index(train_s, a.stride, a.horizon)
  va_idx, va_y, va_dom = build_index(val_s, a.stride * 2, a.horizon)
  vh_idx, vh_y, vh_dom = build_index(valh_s, a.stride * 2, a.horizon)
  print(f"  windows: train {len(tr_idx):,} ({100 * float(tr_y.mean()):.2f}% "
        f"positive)  val {len(va_idx):,}  valh {len(vh_idx):,}")
  for v in damping.VALUES:
    m = (tr_dom - v).abs() < 1e-9
    print(f"    damping {v}: {int(m.sum()):,} windows, "
          f"{100 * float(tr_y[m].mean()):.2f}% positive")

  probe = batch(train_s, tr_idx, list(range(4)), dev)
  dims = {k: int(probe[k].shape[-1]) for k in risk.RiskHead.CHANNELS}
  print(f"  dims: {dims}")

  report = {"config": vars(a), "dims": dims, "horizon": a.horizon,
            "length": LENGTH,
            "n_windows": {"train": len(tr_idx), "val": len(va_idx),
                          "valh": len(vh_idx)},
            "train_positive_rate": float(tr_y.mean())}
  saved = {}
  t0 = time.time()

  # -- the head the score uses ----------------------------------------------
  # `risk_target_only` and its control are the decisive pair.  Trained across
  # all three domains a head can score by recognising which domain a window
  # came from -- the base rates differ a hundredfold -- and predict nothing
  # about which window trips.  Inside the target domain that shortcut is gone
  # and the base rate is three times higher, so if the label is learnable at
  # all this is where it shows, and if this pair is indistinguishable then the
  # answer is that it is not learnable at this lead time.
  for name, keep, shuffle in (
    ("risk", None, False),
    ("risk_heldout", [damping.NOMINAL, damping.COUNTER], False),
    ("risk_shuffled", None, True),
    ("risk_target_only", [damping.TARGET], False),
    ("risk_target_only_shuffled", [damping.TARGET], True),
  ):
    print(f"  -- {name}")
    if keep is None:
      idx, y = tr_idx, tr_y
      subset = train_s
    else:
      sel = [i for i, d in enumerate(tr_dom.tolist())
             if any(abs(d - k) < 1e-9 for k in keep)]
      idx, y = [tr_idx[i] for i in sel], tr_y[sel]
      subset = train_s
      print(f"     trained on {sorted(keep)} only: {len(idx):,} windows")
    torch.manual_seed(a.seed)
    m = risk.RiskHead(dims, hidden=a.hidden).to(dev)
    hist = train(m, subset, idx, y, dev, a.epochs, a.batch, a.lr,
                 a.seed + 5, shuffle=shuffle, cap=a.pos_weight_cap)
    ev = {"val": evaluate(m, val_s, va_idx, va_y, va_dom, dev),
          "valh": evaluate(m, valh_s, vh_idx, vh_y, vh_dom, dev)}
    # The headline is the WITHIN-DOMAIN number.  Pooled across domains the
    # base rate differs seventyfold, so a head that only knows which domain it
    # is in scores well and predicts nothing about which window trips.
    tgt = ev["valh"]["per_domain"].get(str(damping.TARGET), {})
    lift = (tgt.get("average_precision", 0.0)
            / max(tgt.get("base_rate", 1e-9), 1e-9))
    print(f"     val  AP {ev['val']['average_precision']:.3f}  "
          f"AUC {ev['val']['roc_auc']:.3f}  base {ev['val']['base_rate']:.4f}")
    print(f"     WITHIN target domain: AP "
          f"{tgt.get('average_precision', float('nan')):.3f} vs base "
          f"{tgt.get('base_rate', float('nan')):.4f} ({lift:.2f}x), AUC "
          f"{tgt.get('roc_auc', float('nan')):.3f}")
    print(f"     valh AP {ev['valh']['average_precision']:.3f}  "
          f"AUC {ev['valh']['roc_auc']:.3f}")
    for v, d in ev["valh"]["per_domain"].items():
      print(f"       damping {v}: AP {d['average_precision']:.3f} "
            f"base {d['base_rate']:.4f} mean p {d['mean_predicted']:.3f}")
    report[name] = {"history": hist, "eval": ev,
                    "within_target_domain": ev["valh"]["per_domain"].get(
                      str(damping.TARGET), {}),
                    "trained_on": keep or list(damping.VALUES)}
    st = m.state()
    st["horizon"] = a.horizon
    saved[name] = st

  outdir = Path(a.out)
  outdir.mkdir(parents=True, exist_ok=True)
  torch.save(saved, outdir / "risk_head.pt")
  report["wall_clock_s"] = time.time() - t0
  (outdir / "risk_report.json").write_text(json.dumps(report, indent=1))
  print(f"\n  wrote {outdir / 'risk_head.pt'} "
        f"({report['wall_clock_s'] / 60:.1f} min)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
