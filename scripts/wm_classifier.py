"""Train B2 -- the direct history classifier -- and both of its controls.

    python scripts/wm_classifier.py --data results/wm1_latency/data \
        --out results/wm1_latency/model

Three models come out of this, and the second two exist to make the first
interpretable:

``classifier``      the estimator: reward-free history -> q(theta).
``shuffled``        the same architecture, the same data, the same number of
                    updates, with the training labels permuted.  Whatever
                    accuracy this reaches on the test sessions is the floor
                    that the real one has to clear, and it is a tighter floor
                    than 1/5 because it absorbs any accidental structure in
                    how the windows were built.
``done_only``       a classifier that sees episode boundaries and nothing
                    else.  Reset cadence is a genuine consequence of the
                    domain and a robot can see it, but it says nothing about
                    the plant, so it is excluded from every feature set and
                    measured separately.  If it identifies the domain nearly
                    as well as the real classifier, the real classifier's
                    result is about how often the task restarts.

Model selection -- epochs, width, learning rate -- is by validation accuracy on
``val`` and ``valh``, both of which are simulation domains.  No target session
and no reward is read here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from piper_push import latency, wm_data, wm_infer

BURN_IN = 0
LENGTH = 25   # the window the posterior stage uses, so the two agree


def _batch(sessions, idx, sel, length, device):
  return wm_data.batch_from(sessions, idx, sel, length, device,
                            keys=("enc", "proprio", "action", "servo", "done"))


def _accuracy(model, sessions, idx, length, device, batch=512, limit=8000):
  order = list(range(len(idx)))
  if len(order) > limit:
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(order), generator=g)[:limit].tolist()
  right, n = 0, 0
  with torch.no_grad():
    for i in range(0, len(order), batch):
      sel = order[i:i + batch]
      b = _batch(sessions, idx, sel, length, device)
      right += int((model(b).argmax(-1) == b["_lag"]).sum())
      n += len(sel)
  return right / max(n, 1)


def train_one(model, sessions, idx, length, device, epochs, batch, lr, seed,
              shuffle_labels=False):
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  g = torch.Generator().manual_seed(seed)
  hist = []
  for ep in range(epochs):
    perm = torch.randperm(len(idx), generator=g).tolist()
    run, nb, right, n = 0.0, 0, 0, 0
    for i in range(0, len(perm), batch):
      sel = perm[i:i + batch]
      b = _batch(sessions, idx, sel, length, device)
      y = b["_lag"]
      if shuffle_labels:
        y = y[torch.randperm(len(y), generator=g).to(y.device)]
      logits = model(b)
      loss = F.cross_entropy(logits, y)
      opt.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      opt.step()
      run += float(loss)
      right += int((logits.argmax(-1) == y).sum())
      n += len(sel)
      nb += 1
    hist.append({"loss": run / nb, "train_acc": right / n})
    print(f"    epoch {ep}: loss {run / nb:.4f}  train acc {right / n:.3f}",
          flush=True)
  return hist


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--data", default="results/wm1_latency/data")
  p.add_argument("--out", default="results/wm1_latency/model")
  p.add_argument("--epochs", type=int, default=4)
  p.add_argument("--batch", type=int, default=256)
  p.add_argument("--stride", type=int, default=25)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--hidden", type=int, default=96)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()

  root, dev = Path(a.data), a.device
  train = wm_data.load_split(root, "train")
  val = wm_data.load_split(root, "val")
  valh = wm_data.load_split(root, "valh")
  tr, tr_lags = wm_data.make_index(train, a.stride, LENGTH, BURN_IN)
  va, _ = wm_data.make_index(val, a.stride * 2, LENGTH, BURN_IN)
  vh, _ = wm_data.make_index(valh, a.stride * 2, LENGTH, BURN_IN)
  print(f"  windows: train {len(tr):,}  val {len(va):,}  valh {len(vh):,}")
  print(f"  lag counts: "
        f"{torch.bincount(tr_lags, minlength=len(latency.LAGS)).tolist()}")

  probe = _batch(train, tr, list(range(4)), LENGTH, dev)
  dims = {k: int(probe[k].shape[-1]) for k in
          wm_infer.HistoryClassifier.CHANNELS}
  print(f"  dims: {dims}")

  report = {"config": vars(a), "dims": dims, "length": LENGTH,
            "n_windows": {"train": len(tr), "val": len(va), "valh": len(vh)}}
  saved = {}
  t0 = time.time()

  for name, shuffle, ctor in (
    ("classifier", False,
     lambda: wm_infer.HistoryClassifier(dims, len(latency.LAGS), a.hidden)),
    ("shuffled", True,
     lambda: wm_infer.HistoryClassifier(dims, len(latency.LAGS), a.hidden)),
    ("done_only", False,
     lambda: wm_infer.DoneOnlyClassifier(len(latency.LAGS))),
  ):
    print(f"  -- {name}")
    torch.manual_seed(a.seed)
    m = ctor().to(dev)
    hist = train_one(m, train, tr, LENGTH, dev, a.epochs, a.batch, a.lr,
                     a.seed + 7, shuffle_labels=shuffle)
    accs = {"val": _accuracy(m, val, va, LENGTH, dev),
            "valh": _accuracy(m, valh, vh, LENGTH, dev)}
    print(f"     window accuracy  val {accs['val']:.3f}  valh {accs['valh']:.3f}")
    report[name] = {"history": hist, "window_accuracy": accs}
    saved[name] = (m.state() if hasattr(m, "state")
                   else {"state_dict": m.state_dict(),
                         "n_theta": len(latency.LAGS)})

  outdir = Path(a.out)
  outdir.mkdir(parents=True, exist_ok=True)
  torch.save(saved, outdir / "classifier.pt")
  report["wall_clock_s"] = time.time() - t0
  (outdir / "classifier_report.json").write_text(json.dumps(report, indent=1))
  print(f"\n  wrote {outdir / 'classifier.pt'} "
        f"({report['wall_clock_s'] / 60:.1f} min)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
