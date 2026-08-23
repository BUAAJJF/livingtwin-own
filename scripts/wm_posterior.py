"""Infer the hidden observation delay from reward-free sessions, five ways.

    python scripts/wm_posterior.py --data results/wm1_latency/data \
        --model results/wm1_latency/model --out results/wm1_latency/posterior

Two stages, because the expensive one does not depend on the budget.

**Scores.**  For every session -- test, cal and calh -- and every window in it,
the per-candidate cost under each estimator.  Windows are non-overlapping and
never cross an episode boundary, so a budget is a *prefix* of the same session
and the 10 s result is literally the first twenty windows of the 300 s one.
Cached to disk: it is the only part that touches the ensemble.

**Posteriors.**  Score weights and temperature are fitted on ``cal`` -- a
simulation domain, held-out shapes checked separately on ``calh`` -- by
minimising posterior NLL, and then applied unchanged to the test sessions.  No
reward, no success rate, no objects per minute and no target label enters that
choice.  The only thing the target label is used for is scoring the answer
after it has been produced.

What is reported per budget: top-1 accuracy, balanced accuracy over the five
domains, posterior mass on the truth, entropy, NLL, Brier, ECE, the confusion
matrix, and wall-clock.  Plus three controls: a classifier trained on permuted
labels, an ensemble trained with the conditioning shuffled, and a classifier
that sees only episode boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from piper_push import latency, wm_data, wm_infer, wm_model

BURN_IN = 8
HORIZON = 16
LENGTH = BURN_IN + HORIZON + 1          # 25 control steps, 0.5 s
BUDGETS = (10.0, 30.0, 60.0, 180.0, 300.0)

# The components a session's score vector is built from.  Named here so that
# a method is a *subset of these names* plus weights, and adding a method
# cannot silently change what another one reads.
COMPONENTS = ("state", "latent", "action", "clf",
              "state_ctrl", "latent_ctrl", "action_ctrl", "clf_shuf", "clf_done")

METHODS: dict[str, dict] = {
  # name           components and weights (costs; lower is better)
  "B3_state":     {"state": 1.0},
  "B4_action":    {"latent": 1.0, "action": 1.0},
  "DA":           {"state": None, "latent": None, "action": None},  # fitted
  "B2_classifier": {"clf": 1.0},
  # Controls.
  "ctrl_wm_shuffled":  {"state_ctrl": 1.0, "latent_ctrl": 1.0,
                        "action_ctrl": 1.0},
  "ctrl_clf_shuffled": {"clf_shuf": 1.0},
  "ctrl_done_only":    {"clf_done": 1.0},
}


# ---------------------------------------------------------------------------
# Stage 1: per-window scores
# ---------------------------------------------------------------------------


@torch.no_grad()
def session_scores(v: wm_infer.SessionView, ens, ens_ctrl, head, clfs,
                   batch: int = 256) -> dict[str, torch.Tensor]:
  """``(n_windows, n_candidates)`` for every component."""
  starts = v.windows(LENGTH)
  out = {k: [] for k in COMPONENTS}
  for i in range(0, len(starts), batch):
    sel = starts[i:i + batch]
    b = v.stack(sel, LENGTH)                 # (LENGTH, B, C)
    for tag, e in (("", ens), ("_ctrl", ens_ctrl)):
      if e is None:
        for k in ("state", "latent", "action"):
          out[k + tag].append(torch.zeros(len(sel), len(latency.LAGS)))
        continue
      per = _model_scores_per_window(e, head, b)
      for k in ("state", "latent", "action"):
        out[k + tag].append(per[k].cpu())
    for name, key in (("classifier", "clf"), ("shuffled", "clf_shuf"),
                      ("done_only", "clf_done")):
      m = clfs.get(name)
      if m is None:
        out[key].append(torch.zeros(len(sel), len(latency.LAGS)))
        continue
      feed = dict(b)
      dev_ = v.done.device
      ts = (torch.tensor(sel, device=dev_).unsqueeze(0)
            + torch.arange(LENGTH, device=dev_).unsqueeze(1))
      feed["done"] = v.done[ts].float()
      # Cost, not logit: the rest of the pipeline minimises.
      out[key].append((-torch.log_softmax(m(feed), dim=-1)).cpu())
  return {k: (torch.cat(v_) if v_ else torch.zeros(0, len(latency.LAGS)))
          for k, v_ in out.items()}


@torch.no_grad()
def _model_scores_per_window(ens, head, b):
  """:func:`wm_infer.model_scores` without the average over windows."""
  norms = ens.norms
  z, p = norms["z"](b["enc"]), norms["p"](b["proprio"])
  a, e = norms["a"](b["action"]), norms["e"](b["servo"])
  lo, hi = BURN_IN, BURN_IN + HORIZON
  nb = z.shape[1]
  out = {k: torch.zeros(nb, len(latency.LAGS), device=z.device)
         for k in ("state", "latent", "action")}
  for ci, c in enumerate(latency.LAGS):
    theta = torch.full((nb,), c, dtype=torch.long, device=z.device)
    for m in ens.members:
      _, h = m.teacher_forced(z[:lo], p[:lo], a[:lo], theta, None)
      (z_mu, z_lv, dp, p_lv, e_mu, e_lv), _ = m.teacher_forced(
        z[lo:hi], p[lo:hi], a[lo:hi], theta, h)
      out["state"][:, ci] += (
        wm_model.gaussian_nll(p[lo + 1:hi + 1] - p[lo:hi], dp, p_lv).sum(-1).mean(0)
        + wm_model.gaussian_nll(e[lo + 1:hi + 1], e_mu, e_lv).sum(-1).mean(0))
      out["latent"][:, ci] += wm_model.gaussian_nll(
        z[lo + 1:hi + 1], z_mu, z_lv).sum(-1).mean(0)
      if head is not None:
        pred_enc = z_mu * norms["z"].std + norms["z"].mean
        hid = b["hidden"][lo + 1:hi + 1]
        act, _ = head(pred_enc.reshape(-1, pred_enc.shape[-1]),
                      hid.reshape(1, -1, hid.shape[-1]))
        real = b["action"][lo + 1:hi + 1]
        d = ((act.reshape(real.shape) - real) ** 2).sum(-1).mean(0)
        out["action"][:, ci] += d
  for k in out:
    out[k] /= len(ens.members)
  return out


# ---------------------------------------------------------------------------
# Stage 2: budgets, weights, posteriors
# ---------------------------------------------------------------------------


def prefix_windows(n_windows: int, arm_seconds: float) -> int:
  """How many windows fit in a budget.  Nested by construction."""
  return min(n_windows, int(arm_seconds * wm_infer.CONTROL_HZ) // LENGTH)


def combine(comp: dict[str, torch.Tensor], k: int,
            weights: dict[str, float]) -> list[float]:
  """Sum the first ``k`` windows of the weighted components."""
  total = None
  for name, w in weights.items():
    x = comp[name][:k]
    if x.numel() == 0:
      continue
    part = w * x.sum(0)
    total = part if total is None else total + part
  if total is None:
    return [0.0] * len(latency.LAGS)
  return [float(x) for x in total]


def fit_da_weights(cal_rows, budget_k, grid_steps: int = 6):
  """Choose (lambda_state, lambda_latent, lambda_action) and the temperature.

  A simplex grid, because three weights and one temperature over a few hundred
  precomputed score vectors is arithmetic, and a gradient method here would
  buy nothing except a way to end up in a local minimum without noticing.
  """
  best = (float("inf"), None, None)
  for i in range(grid_steps + 1):
    for j in range(grid_steps + 1 - i):
      ls, lz = i / grid_steps, j / grid_steps
      la = 1.0 - ls - lz
      w = {"state": ls, "latent": lz, "action": la}
      rows = [combine(c, budget_k(c), w) for c, _ in cal_rows]
      truths = [y for _, y in cal_rows]
      t = wm_infer.fit_temperature(rows, truths)
      total = sum(wm_infer.nll(wm_infer.posterior(r, t), y)
                  for r, y in zip(rows, truths))
      if total < best[0]:
        best = (total, w, t)
  return best[1], best[2], best[0]


def metrics(rows, truths, temperature, tag) -> dict:
  qs = [wm_infer.posterior(r, temperature) for r in rows]
  preds = [q.argmax for q in qs]
  n = len(qs)
  return {
    "method": tag,
    "n_sessions": n,
    "top1": sum(1 for q, y in zip(qs, truths) if q.argmax == y) / max(n, 1),
    "balanced_accuracy": wm_infer.balanced_accuracy(preds, truths),
    "mass_on_truth": sum(q.mass(y) for q, y in zip(qs, truths)) / max(n, 1),
    "entropy_bits": sum(q.entropy_bits for q in qs) / max(n, 1),
    "nll": sum(wm_infer.nll(q, y) for q, y in zip(qs, truths)) / max(n, 1),
    "brier": sum(wm_infer.brier(q, y) for q, y in zip(qs, truths)) / max(n, 1),
    "ece": wm_infer.ece(qs, truths),
    "confusion": wm_infer.confusion(preds, truths),
    "temperature": temperature,
    "target_only": _target_only(qs, truths),
  }


def _target_only(qs, truths) -> dict:
  """The hidden domain on its own: this is what the adaptation stage uses.

  A real deployment produces *one* session, so the per-session spread is
  reported next to the pooled answer.  ``session0_posterior`` is the one a
  practitioner with a single afternoon would have had; ``mean_posterior`` is
  what pooling every session gives.  If the two differ, the adaptation result
  depends on which session was collected, and that has to be visible.
  """
  sel = [i for i, y in enumerate(truths) if y == latency.TARGET_LAG]
  if not sel:
    return {}
  masses = [qs[i].mass(latency.TARGET_LAG) for i in sel]
  return {
    "n": len(sel),
    "top1": sum(1 for i in sel if qs[i].argmax == latency.TARGET_LAG) / len(sel),
    "mass_on_truth": sum(masses) / len(sel),
    "mass_min": min(masses), "mass_max": max(masses),
    "mean_posterior": [sum(qs[i].probs[k] for i in sel) / len(sel)
                       for k in range(len(latency.LAGS))],
    "session0_posterior": list(qs[sel[0]].probs),
  }


# ---------------------------------------------------------------------------


def collect_scores(sessions, ens, ens_ctrl, head, clfs, device, cache: Path,
                   tag: str, limit_envs: int | None = None):
  if cache.exists():
    d = torch.load(cache, map_location="cpu", weights_only=False)
    print(f"  {tag}: {len(d)} cached session score sets")
    return d
  rows = []
  t0 = time.time()
  for s, name in sessions:
    for env in range(min(s.n_envs, limit_envs or s.n_envs)):
      v = wm_infer.SessionView.from_session(s, env, 1e9, device=device)
      comp = session_scores(v, ens, ens_ctrl, head, clfs)
      rows.append({"file": name, "env": env, "lag": s.lag,
                   "n_windows": int(comp["state"].shape[0]),
                   "comp": comp})
    print(f"    {name}: {s.n_envs} sessions  "
          f"({time.time() - t0:.0f} s)", flush=True)
  cache.parent.mkdir(parents=True, exist_ok=True)
  torch.save(rows, cache)
  print(f"  {tag}: wrote {cache} ({time.time() - t0:.0f} s)")
  return rows


def analytic_rows(sessions, enc_split, device, limit_envs=None):
  """B1a and B1b at every budget: cheap, so recomputed rather than cached."""
  out = []
  for s, name in sessions:
    for env in range(min(s.n_envs, limit_envs or s.n_envs)):
      per_budget = {}
      for bud in BUDGETS:
        v = wm_infer.SessionView.from_session(s, env, bud, device=device)
        per_budget[bud] = {
          "b1a": wm_infer.xcorr_action_joint(v),
          "b1b": wm_infer.xcorr_latent_proprio(v, enc_split),
        }
      out.append({"file": name, "env": env, "lag": s.lag, "budgets": per_budget})
  return out


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--data", default="results/wm1_latency/data")
  p.add_argument("--model", default="results/wm1_latency/model")
  p.add_argument("--out", default="results/wm1_latency/posterior")
  p.add_argument("--checkpoint",
                 default="logs/rsl_rl/piperx_pick_place_vision/"
                         "2026-08-22_17-15-09_f3/model_1500.pt")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--recompute", action="store_true")
  p.add_argument("--limit-envs", type=int, default=None,
                 help="smoke testing only: sessions per file.  A run with this "
                      "set is marked in its report and is not a formal result.")
  a = p.parse_args()

  data, mdir, out = Path(a.data), Path(a.model), Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  dev = a.device

  ens = wm_model.Ensemble.load(mdir / "ensemble.pt").to_device(dev)
  ctrl_path = mdir / "ensemble_shuffled.pt"
  ens_ctrl = (wm_model.Ensemble.load(ctrl_path).to_device(dev)
              if ctrl_path.exists() else None)
  head = wm_model.ActorHead.from_checkpoint(a.checkpoint, map_location=dev).to(dev)
  saved = torch.load(mdir / "classifier.pt", map_location=dev, weights_only=False)
  clfs = {}
  for name in ("classifier", "shuffled"):
    if name in saved:
      clfs[name] = wm_infer.HistoryClassifier.load(saved[name], dev)
  if "done_only" in saved:
    m = wm_infer.DoneOnlyClassifier(saved["done_only"]["n_theta"])
    m.load_state_dict(saved["done_only"]["state_dict"])
    clfs["done_only"] = m.eval().to(dev)
  print(f"  ensemble {len(ens.members)} members; controls: "
        f"wm={'yes' if ens_ctrl else 'NO'}, clf={sorted(clfs)}")
  print(f"  encoder latent split at {head.obs_dim_1d} "
        f"(image encoding is the remaining columns)")

  splits = {}
  for s in ("test", "cal", "calh"):
    try:
      splits[s] = wm_data.load_split(data, s)
    except FileNotFoundError:
      print(f"  !!! no {s} split; it will be skipped")

  scores = {}
  for s, sess in splits.items():
    cache = out / f"scores_{s}.pt"
    if a.recompute and cache.exists():
      cache.unlink()
    scores[s] = collect_scores(sess, ens, ens_ctrl, head, clfs, dev, cache, s,
                               a.limit_envs)

  analytic = {s: analytic_rows(sess, head.obs_dim_1d, dev, a.limit_envs)
              for s, sess in splits.items()}
  torch.save(analytic, out / "analytic.pt")

  # -- what one session costs, at the budget the adaptation stage uses -------
  # G6 counts this against a deployment's online time, so it is measured on a
  # single 60 s session rather than divided out of a batch over hundreds.
  timing = {}
  if splits.get("test"):
    s0 = splits["test"][0][0]
    v60 = wm_infer.SessionView.from_session(s0, 0, 60.0, device=dev)
    for tag, fn in (
      ("world_model_scores_s",
       lambda: session_scores(v60, ens, None, head, {})),
      ("classifier_s",
       lambda: session_scores(v60, None, None, None,
                              {"classifier": clfs["classifier"]})
       if "classifier" in clfs else None),
      ("b1b_ridge_s",
       lambda: wm_infer.xcorr_latent_proprio(v60, head.obs_dim_1d)),
      ("b1a_xcorr_s", lambda: wm_infer.xcorr_action_joint(v60)),
    ):
      fn()                                   # warm the kernels
      t0 = time.time()
      for _ in range(3):
        fn()
      timing[tag] = (time.time() - t0) / 3
    timing["n_windows_60s"] = len(v60.windows(LENGTH))
    print("  inference wall-clock on one 60 s session: "
          + "  ".join(f"{k} {v:.3f}" for k, v in timing.items()
                      if k.endswith("_s")))
  (out / "inference_timing.json").write_text(json.dumps(timing, indent=1))

  # -- per budget -----------------------------------------------------------
  report = {"budgets": {}, "smoke": a.limit_envs is not None,
            "length": LENGTH, "burn_in": BURN_IN,
            "horizon": HORIZON, "enc_split": head.obs_dim_1d,
            "n_members": len(ens.members),
            "methods": {k: v for k, v in METHODS.items()}}
  for bud in BUDGETS:
    t0 = time.time()
    def k_of(comp, bud=bud):
      return prefix_windows(int(comp["state"].shape[0]), bud)

    entry = {"arm_seconds": bud, "methods": {}}
    # DA weights and temperature, on the calibration domains only.
    cal_rows = [(r["comp"], r["lag"]) for r in scores.get("cal", [])]
    if cal_rows:
      w_da, t_da, _ = fit_da_weights(cal_rows, k_of)
    else:
      w_da, t_da = {"state": 1 / 3, "latent": 1 / 3, "action": 1 / 3}, 1.0
    entry["da_weights"] = w_da

    for name, weights in METHODS.items():
      w = w_da if name == "DA" else weights
      cal_r = [combine(c, k_of(c), w) for c, _ in cal_rows]
      cal_y = [y for _, y in cal_rows]
      temp = (t_da if name == "DA" else
              (wm_infer.fit_temperature(cal_r, cal_y) if cal_r else 1.0))
      for split in ("test", "calh"):
        rows = [combine(r["comp"], k_of(r["comp"]), w) for r in scores.get(split, [])]
        truths = [r["lag"] for r in scores.get(split, [])]
        if not rows:
          continue
        entry["methods"].setdefault(split, {})[name] = metrics(
          rows, truths, temp, name)
      entry["methods"].setdefault("cal", {})[name] = metrics(
        cal_r, cal_y, temp, name) if cal_r else {}

    # The analytic baselines, which need no model and no cache.
    for split, rows in analytic.items():
      for key in ("b1a", "b1b"):
        sc = [r["budgets"][bud][key] for r in rows]
        truths = [r["lag"] for r in rows]
        temp = 1.0
        if split == "cal":
          temp = wm_infer.fit_temperature(sc, truths)
        else:
          cal_sc = [r["budgets"][bud][key] for r in analytic.get("cal", [])]
          cal_y = [r["lag"] for r in analytic.get("cal", [])]
          if cal_sc:
            temp = wm_infer.fit_temperature(cal_sc, cal_y)
        entry["methods"].setdefault(split, {})[
          {"b1a": "B1a_cmd_joint", "b1b": "B1b_img_proprio"}[key]] = metrics(
            sc, truths, temp, key)

    # B0: no target data at all.
    for split in ("test", "cal", "calh"):
      truths = [r["lag"] for r in analytic.get(split, [])]
      if truths:
        qs = [latency.P_SOURCE] * len(truths)
        entry["methods"].setdefault(split, {})["B0_prior"] = {
          "method": "B0_prior", "n_sessions": len(truths),
          "top1": sum(1 for y in truths if y == 0) / len(truths),
          "balanced_accuracy": wm_infer.balanced_accuracy(
            [0] * len(truths), truths),
          "mass_on_truth": sum(q.mass(y) for q, y in zip(qs, truths)) / len(truths),
          "entropy_bits": 0.0,
          "nll": sum(wm_infer.nll(q, y) for q, y in zip(qs, truths)) / len(truths),
          "brier": sum(wm_infer.brier(q, y) for q, y in zip(qs, truths)) / len(truths),
          "ece": wm_infer.ece(qs, truths),
          "confusion": wm_infer.confusion([0] * len(truths), truths),
          "temperature": float("nan"),
          "target_only": _target_only(qs, truths),
        }

    entry["wall_clock_s"] = time.time() - t0
    report["budgets"][str(bud)] = entry
    print(f"\n  == {bud:.0f} s budget "
          f"({k_of(scores['test'][0]['comp']) if scores.get('test') else 0} "
          f"windows) ==")
    print(f"     {'method':20s} {'top1':>6s} {'bal':>6s} {'mass':>6s} "
          f"{'H(bits)':>8s} {'ece':>6s}   target-only top1/mass")
    for name, m in sorted(entry["methods"].get("test", {}).items()):
      to = m.get("target_only") or {}
      print(f"     {name:20s} {m['top1']:6.3f} {m['balanced_accuracy']:6.3f} "
            f"{m['mass_on_truth']:6.3f} {m['entropy_bits']:8.3f} "
            f"{m['ece']:6.3f}   {to.get('top1', float('nan')):.3f} / "
            f"{to.get('mass_on_truth', float('nan')):.3f}")

  (out / "posterior_report.json").write_text(json.dumps(report, indent=1))

  # The posterior the adaptation stage will use, written on its own so that
  # nothing downstream has to parse the report to find it.
  chosen = {}
  for name in ("B1b_img_proprio", "B2_classifier", "B3_state", "B4_action",
               "DA"):
    e = report["budgets"]["60.0"]["methods"]["test"].get(name)
    if e and e.get("target_only"):
      probs = e["target_only"]["mean_posterior"]
      total = sum(probs)
      chosen[name] = latency.LatencyPrior(
        tuple(x / total for x in probs)).to_json()
  (out / "posteriors_60s.json").write_text(json.dumps(chosen, indent=1))
  print(f"\n  wrote {out / 'posterior_report.json'} and posteriors_60s.json")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
