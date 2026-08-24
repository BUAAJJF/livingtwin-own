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

from piper_push import (damping, latency, prior, risk, wm_data, wm_infer,
                        wm_model)

# Which simulator parameter this run is about.  Phase WM1-A's observation delay
# and Phase WM1-B's servo damping differ in their candidate set, their target
# and how a draw reaches the simulator, and in nothing else that this script
# does -- so the script takes the axis as an argument rather than being copied.
AXES = {
  "obs_latency_steps": (latency.LatencyPrior, latency.LAGS, latency.TARGET_LAG,
                        latency.P_SOURCE),
  "servo_damping_scale": (damping.DampingPrior, damping.VALUES, damping.TARGET,
                          damping.P_SOURCE),
}
AXIS = "obs_latency_steps"
PRIOR_CLS, VALUES, TARGET, P_SOURCE = AXES[AXIS]

SHIFT_BASELINES_APPLY = {"obs_latency_steps": True,
                         "servo_damping_scale": False}
"""Whether B1a and B1b mean anything on this axis.

Both are *shift-scanning* baselines.  B1a correlates the commanded step against
the joint's response at each candidate shift; B1b fits a ridge from the image
encoding to the joint state `c` steps earlier.  On the latency axis a candidate
IS a shift and both are the natural model-free thing to try.

On the damping axis a candidate is a multiplier on the servo's derivative gain.
There is no shift to scan.  `img[fit] -> q[fit - 0.75]` is not a weaker version
of the same idea; it is a fractional index into time, and the closest integer
reading of it would score every candidate identically while looking like a
measurement.

So they are deprecated here rather than adapted.  Phase WM1-B was asked not to
guess at an equivalence that does not exist, and the comparison it does need --
state, proprioception and servo-error trajectory matching -- is `B3_state`,
which compares a simulated rollout against the observed one and is indexed by
candidate rather than by shift, so it carries across unchanged.
"""


def set_axis(name: str) -> None:
  global AXIS, PRIOR_CLS, VALUES, TARGET, P_SOURCE
  AXIS = name
  PRIOR_CLS, VALUES, TARGET, P_SOURCE = AXES[name]

BURN_IN = 8
HORIZON = 16
LENGTH = BURN_IN + HORIZON + 1          # 25 control steps, 0.5 s
BUDGETS = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 180.0, 300.0)
"""The five the phase was asked for, plus three below them.

Everything model-based saturates at 10 s, so the ten-second row says only "at
least this easy".  Where it *stops* working is the number that matters for a
loop whose whole premise is an hour of robot time, and the smaller budgets cost
nothing: the per-window scores are already computed, and a budget is a prefix
of them.  One second is two non-overlapping windows."""

# The components a session's score vector is built from.  Named here so that
# a method is a *subset of these names* plus weights, and adding a method
# cannot silently change what another one reads.
COMPONENTS = ("state", "latent", "action", "risk", "clf",
              "state_ctrl", "latent_ctrl", "action_ctrl", "clf_shuf", "clf_done")

METHODS: dict[str, dict] = {
  # name           components and weights (costs; lower is better)
  "B3_state":     {"state": 1.0},
  "B4_action":    {"latent": 1.0, "action": 1.0},
  "DA":           {"state": None, "latent": None, "action": None},  # fitted
  "B2_classifier": {"clf": 1.0},
  # Ablations of B4, so that "the latent identifies it" and "the decision the
  # policy would have taken identifies it" are separable claims rather than one
  # number.  Free: the components are already cached.
  "abl_latent_only": {"latent": 1.0},
  "abl_action_only": {"action": 1.0},
  # Phase WM1-B.  `S_risk` asked what the *safety* consequence of the predicted
  # future would have been.  It is kept, and it is kept because it FAILED: the
  # head it depends on scores 1.02x its base rate inside the target domain
  # against a shuffled-label control at 1.16x, so the component is noise.  It
  # stays in the report as a measured negative rather than being deleted.
  "abl_risk_only": {"risk": 1.0},
  "M5_risk_score": {"latent": None, "action": None, "risk": None},  # fitted
  # Controls.
  "ctrl_wm_shuffled":  {"state_ctrl": 1.0, "latent_ctrl": 1.0,
                        "action_ctrl": 1.0},
  "ctrl_clf_shuffled": {"clf_shuf": 1.0},
  "ctrl_done_only":    {"clf_done": 1.0},
}


# Risk-awareness that does not depend on the failed head.  Each entry tilts
# another method's posterior by the simulated trip rate of each candidate:
#
#     q_risk(theta) ∝ q(theta) * exp(lambda * cost(theta))
#
# The tilt reads only simulator safety labels, which the specification allows,
# and never the target domain's reward, success, trip label or true damping.
# It is not an estimator and does not try to be -- it is the distribution a
# planner should train against when being wrong towards danger costs more than
# being wrong towards safety.
#
# `M2_broad` is tilted because a broad posterior is the one with mass left to
# move; `DA` is tilted as well so the report can show what the same operation
# does to a posterior that is already nearly a point mass, which is nothing.
TILTED = {
  "M5_risk_aware": ("M2_broad", "LAMBDA"),
  "M5_risk_aware_da": ("DA", "LAMBDA"),
}


# ---------------------------------------------------------------------------
# Stage 1: per-window scores
# ---------------------------------------------------------------------------


@torch.no_grad()
def session_scores(v: wm_infer.SessionView, ens, ens_ctrl, head, clfs,
                   batch: int = 256, risk_head=None) -> dict[str, torch.Tensor]:
  """``(n_windows, n_candidates)`` for every component."""
  starts = v.windows(LENGTH)
  out = {k: [] for k in COMPONENTS}
  for i in range(0, len(starts), batch):
    sel = starts[i:i + batch]
    b = v.stack(sel, LENGTH)                 # (LENGTH, B, C)
    for tag, e in (("", ens), ("_ctrl", ens_ctrl)):
      keys = ("state", "latent", "action", "risk") if not tag else (
        "state", "latent", "action")
      if e is None:
        for k in keys:
          out[k + tag].append(torch.zeros(len(sel), len(VALUES)))
        continue
      per = _model_scores_per_window(e, head, b,
                                     risk_head if not tag else None)
      for k in keys:
        out[k + tag].append(per[k].cpu())
    for name, key in (("classifier", "clf"), ("shuffled", "clf_shuf"),
                      ("done_only", "clf_done")):
      m = clfs.get(name)
      if m is None:
        out[key].append(torch.zeros(len(sel), len(VALUES)))
        continue
      feed = dict(b)
      dev_ = v.done.device
      ts = (torch.tensor(sel, device=dev_).unsqueeze(0)
            + torch.arange(LENGTH, device=dev_).unsqueeze(1))
      feed["done"] = v.done[ts].float()
      # Cost, not logit: the rest of the pipeline minimises.
      out[key].append((-torch.log_softmax(m(feed), dim=-1)).cpu())
  return {k: (torch.cat(v_) if v_ else torch.zeros(0, len(VALUES)))
          for k, v_ in out.items()}


@torch.no_grad()
def _model_scores_per_window(ens, head, b, risk_head=None):
  """:func:`wm_infer.model_scores` without the average over windows."""
  norms = ens.norms
  z, p = norms["z"](b["enc"]), norms["p"](b["proprio"])
  a, e = norms["a"](b["action"]), norms["e"](b["servo"])
  lo, hi = BURN_IN, BURN_IN + HORIZON
  nb = z.shape[1]
  out = {k: torch.zeros(nb, len(VALUES), device=z.device)
         for k in ("state", "latent", "action", "risk")}
  for ci, c in enumerate(VALUES):
    theta = torch.full((nb,), ci, dtype=torch.long, device=z.device)
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
      if risk_head is not None:
        # What the *safety* consequence of the predicted future would have
        # been, against the one that happened.  Same shape as the action score
        # with the risk head in place of the actor: replace the window's last
        # observed step with the candidate's prediction and ask C_obs both
        # ways.  A tail-only mismatch is exactly the case where the plant's
        # state trajectory and the policy's action barely move and the risk
        # does.
        real_w = {k: b[k] for k in risk.RiskHead.CHANNELS}
        pred_w = {k: v.clone() for k, v in real_w.items()}
        pred_w["enc"][-1] = pred_enc[-1]
        pred_w["proprio"][-1] = (
          b["proprio"][hi - 1]
          + dp[-1] * norms["p"].std)
        p_real = risk_head.probability(real_w)
        p_pred = risk_head.probability(pred_w)
        out["risk"][:, ci] += (p_real - p_pred).abs()
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
    return [0.0] * len(VALUES)
  return [float(x) for x in total]


def fit_da_weights(cal_rows, budget_k, grid_steps: int = 6,
                   names=("state", "latent", "action")):
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
      w = dict(zip(names, (ls, lz, la)))
      rows = [combine(c, budget_k(c), w) for c, _ in cal_rows]
      truths = [y for _, y in cal_rows]
      t = wm_infer.fit_temperature(rows, truths, cls=PRIOR_CLS)
      total = sum(wm_infer.nll(wm_infer.posterior(r, t, cls=PRIOR_CLS), y)
                  for r, y in zip(rows, truths))
      if total < best[0]:
        best = (total, w, t)
  return best[1], best[2], best[0]


def metrics(rows, truths, temperature, tag) -> dict:
  qs = [wm_infer.posterior(r, temperature, cls=PRIOR_CLS) for r in rows]
  preds = [q.argmax for q in qs]
  n = len(qs)
  return {
    "method": tag,
    "n_sessions": n,
    "top1": sum(1 for q, y in zip(qs, truths) if q.argmax == y) / max(n, 1),
    "balanced_accuracy": wm_infer.balanced_accuracy(preds, truths, VALUES),
    "mass_on_truth": sum(q.mass(y) for q, y in zip(qs, truths)) / max(n, 1),
    "entropy_bits": sum(q.entropy_bits for q in qs) / max(n, 1),
    "nll": sum(wm_infer.nll(q, y) for q, y in zip(qs, truths)) / max(n, 1),
    "brier": sum(wm_infer.brier(q, y) for q, y in zip(qs, truths)) / max(n, 1),
    "ece": wm_infer.ece(qs, truths),
    "confusion": wm_infer.confusion(preds, truths, VALUES),
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
  sel = [i for i, y in enumerate(truths) if y == TARGET]
  if not sel:
    return {}
  masses = [qs[i].mass(TARGET) for i in sel]
  return {
    "n": len(sel),
    "top1": sum(1 for i in sel if qs[i].argmax == TARGET) / len(sel),
    "mass_on_truth": sum(masses) / len(sel),
    "mass_min": min(masses), "mass_max": max(masses),
    "mean_posterior": [sum(qs[i].probs[k] for i in sel) / len(sel)
                       for k in range(len(VALUES))],
    "session0_posterior": list(qs[sel[0]].probs),
  }


# ---------------------------------------------------------------------------


def collect_scores(sessions, ens, ens_ctrl, head, clfs, device, cache: Path,
                   tag: str, limit_envs: int | None = None, risk_head=None):
  if cache.exists():
    d = torch.load(cache, map_location="cpu", weights_only=False)
    print(f"  {tag}: {len(d)} cached session score sets")
    return d
  rows = []
  t0 = time.time()
  for s, name in sessions:
    for env in range(min(s.n_envs, limit_envs or s.n_envs)):
      v = wm_infer.SessionView.from_session(s, env, 1e9, device=device)
      comp = session_scores(v, ens, ens_ctrl, head, clfs,
                            risk_head=risk_head)
      rows.append({"file": name, "env": env, "lag": s.lag,
                   "n_windows": int(comp["state"].shape[0]),
                   "comp": comp})
    print(f"    {name}: {min(s.n_envs, limit_envs or s.n_envs)} sessions  "
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
          "b1a": (wm_infer.xcorr_action_joint(v)
                  if SHIFT_BASELINES_APPLY[AXIS] else None),
          "b1b": (wm_infer.xcorr_latent_proprio(v, enc_split)
                  if SHIFT_BASELINES_APPLY[AXIS] else None),
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
  p.add_argument("--axis", default="obs_latency_steps", choices=sorted(AXES))
  p.add_argument("--risk-lambda", type=float, default=None,
                 help="tilt strength.  Default is the pre-registered rule in "
                      "`prior.risk_lambda`; passing a value overrides it and "
                      "override is recorded in the posterior's provenance.")
  p.add_argument("--risk-head", default=None,
                 help="path to risk_head.pt; enables the risk component")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--recompute", action="store_true")
  p.add_argument("--limit-envs", type=int, default=None,
                 help="smoke testing only: sessions per file.  A run with this "
                      "set is marked in its report and is not a formal result.")
  a = p.parse_args()

  set_axis(a.axis)
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
  risk_head = None
  if a.risk_head and Path(a.risk_head).exists():
    saved_r = torch.load(a.risk_head, map_location=dev, weights_only=False)
    risk_head = risk.RiskHead.load(saved_r["risk"], dev)
    print(f"  risk head: horizon {saved_r['risk']['horizon']} steps, "
          f"trained on {len(saved_r)} variants")
  print(f"  axis {AXIS}, candidates {VALUES}, target {TARGET}")
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
                               a.limit_envs, risk_head=risk_head)

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
      *((("b1b_ridge_s",
          lambda: wm_infer.xcorr_latent_proprio(v60, head.obs_dim_1d)),
         ("b1a_xcorr_s", lambda: wm_infer.xcorr_action_joint(v60)))
        if SHIFT_BASELINES_APPLY[AXIS] else ()),
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
  report = {"axis": AXIS, "candidates": list(VALUES), "target": TARGET,
            "budgets": {}, "smoke": a.limit_envs is not None,
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
      w_m5, t_m5, _ = fit_da_weights(cal_rows, k_of,
                                     names=("latent", "action", "risk"))
    else:
      w_da, t_da = {"state": 1 / 3, "latent": 1 / 3, "action": 1 / 3}, 1.0
      w_m5, t_m5 = {"latent": 1 / 3, "action": 1 / 3, "risk": 1 / 3}, 1.0
    entry["da_weights"] = w_da
    entry["m5_weights"] = w_m5
    if not SHIFT_BASELINES_APPLY[AXIS]:
      entry["deprecated_methods"] = {
        "B1a_cmd_joint": "shift-scanning; a damping candidate is not a shift",
        "B1b_img_proprio": "shift-scanning; a damping candidate is not a shift",
      }

    for name, weights in METHODS.items():
      w = (w_da if name == "DA"
           else w_m5 if name == "M5_risk_score" else weights)
      cal_r = [combine(c, k_of(c), w) for c, _ in cal_rows]
      cal_y = [y for _, y in cal_rows]
      temp = (t_da if name == "DA" else t_m5 if name == "M5_risk_score" else
              (wm_infer.fit_temperature(cal_r, cal_y, cls=PRIOR_CLS) if cal_r else 1.0))
      for split in ("test", "calh"):
        rows = [combine(r["comp"], k_of(r["comp"]), w) for r in scores.get(split, [])]
        truths = [r["lag"] for r in scores.get(split, [])]
        if not rows:
          continue
        entry["methods"].setdefault(split, {})[name] = metrics(
          rows, truths, temp, name)
      entry["methods"].setdefault("cal", {})[name] = metrics(
        cal_r, cal_y, temp, name) if cal_r else {}

    # The analytic baselines, which need no model and no cache -- where they
    # mean anything.  See SHIFT_BASELINES_APPLY.
    for split, rows in (analytic.items() if SHIFT_BASELINES_APPLY[AXIS]
                        else ()):
      for key in ("b1a", "b1b"):
        sc = [r["budgets"][bud][key] for r in rows]
        truths = [r["lag"] for r in rows]
        temp = 1.0
        if split == "cal":
          temp = wm_infer.fit_temperature(sc, truths, cls=PRIOR_CLS)
        else:
          cal_sc = [r["budgets"][bud][key] for r in analytic.get("cal", [])]
          cal_y = [r["lag"] for r in analytic.get("cal", [])]
          if cal_sc:
            temp = wm_infer.fit_temperature(cal_sc, cal_y, cls=PRIOR_CLS)
        entry["methods"].setdefault(split, {})[
          {"b1a": "B1a_cmd_joint", "b1b": "B1b_img_proprio"}[key]] = metrics(
            sc, truths, temp, key)

    # The broad-DR comparator: a fixed uniform posterior over the candidate
    # set, which is what "do not identify anything, randomise instead" means.
    # Not an estimator -- it reads nothing -- so it is constructed rather than
    # scored, exactly like B0.
    for split in ("test", "cal", "calh"):
      truths = [r["lag"] for r in scores.get(split, [])] or [
        r["lag"] for r in analytic.get(split, [])]
      if not truths:
        continue
      qs = [PRIOR_CLS.uniform()] * len(truths)
      preds = [q.argmax for q in qs]
      entry["methods"].setdefault(split, {})["M2_broad"] = {
        "method": "M2_broad", "n_sessions": len(truths),
        "top1": sum(1 for q, y in zip(qs, truths) if q.argmax == y) / len(truths),
        "balanced_accuracy": wm_infer.balanced_accuracy(preds, truths, VALUES),
        "mass_on_truth": sum(q.mass(y) for q, y in zip(qs, truths)) / len(truths),
        "entropy_bits": qs[0].entropy_bits,
        "nll": sum(wm_infer.nll(q, y) for q, y in zip(qs, truths)) / len(truths),
        "brier": sum(wm_infer.brier(q, y) for q, y in zip(qs, truths)) / len(truths),
        "ece": wm_infer.ece(qs, truths),
        "confusion": wm_infer.confusion(preds, truths, VALUES),
        "temperature": float("nan"),
        "target_only": _target_only(qs, truths),
      }

    # B0: no target data at all.
    for split in ("test", "cal", "calh"):
      truths = [r["lag"] for r in analytic.get(split, [])]
      if truths:
        qs = [P_SOURCE] * len(truths)
        entry["methods"].setdefault(split, {})["B0_prior"] = {
          "method": "B0_prior", "n_sessions": len(truths),
          "top1": sum(1 for y in truths if y == P_SOURCE.argmax) / len(truths),
          "balanced_accuracy": wm_infer.balanced_accuracy(
            [P_SOURCE.argmax] * len(truths), truths, VALUES),
          "mass_on_truth": sum(q.mass(y) for q, y in zip(qs, truths)) / len(truths),
          "entropy_bits": 0.0,
          "nll": sum(wm_infer.nll(q, y) for q, y in zip(qs, truths)) / len(truths),
          "brier": sum(wm_infer.brier(q, y) for q, y in zip(qs, truths)) / len(truths),
          "ece": wm_infer.ece(qs, truths),
          "confusion": wm_infer.confusion(
            [P_SOURCE.argmax] * len(truths), truths, VALUES),
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
  #
  # It is ONE session's posterior, not the mean over the thirty-two.  A real
  # deployment gets one afternoon with one arm, and averaging thirty-two
  # independent sessions is a variance reduction nobody will have.  The mean
  # and the spread are recorded beside it: if they agree the choice did not
  # matter and the report says so, and if they do not, the single session is
  # the honest one.
  chosen = {}
  for name in ("B1b_img_proprio", "B2_classifier", "B3_state", "B4_action",
               "DA", "abl_latent_only", "abl_action_only", "M2_broad",
               "M5_risk_score"):
    if name == "B1b_img_proprio" and not SHIFT_BASELINES_APPLY[AXIS]:
      continue
    e = report["budgets"]["60.0"]["methods"]["test"].get(name)
    t = (e or {}).get("target_only")
    if not t:
      continue
    def _norm(v):
      total = sum(v)
      return PRIOR_CLS(tuple(x / total for x in v))

    single = _norm(t["session0_posterior"])
    pooled = _norm(t["mean_posterior"])
    d = single.to_json()
    d["source"] = "session 0 of the target split, 60 s"
    d["pooled_over_32_sessions"] = pooled.to_json()
    d["total_variation_single_vs_pooled"] = single.total_variation(pooled)
    d["mass_on_truth_across_sessions"] = {
      "mean": t["mass_on_truth"], "min": t["mass_min"], "max": t["mass_max"]}
    chosen[name] = d

  # -- risk-aware tilts -----------------------------------------------------
  # Applied after the fact to a base posterior, so nothing above changes and a
  # tilted distribution can never be mistaken for a fitted one.
  if a.axis == "servo_damping_scale":
    costs = damping.trip_costs()
    lam = (a.risk_lambda if a.risk_lambda is not None
             else prior.risk_lambda(costs))
    for name, (base, _) in TILTED.items():
      if base not in chosen:
        print(f"  !! no `{base}` posterior; `{name}` is not written")
        continue
      q = PRIOR_CLS(tuple(chosen[base]["probs"]))
      try:
        tilted = q.tilt(costs, lam)
      except ValueError as exc:
        print(f"  !! {name}: {exc}")
        continue
      d = tilted.to_json()
      d["source"] = f"`{base}` tilted by exp({lam:.5f} * simulated trips/h)"
      d["risk_tilt"] = {
        "base_method": base, "lam": lam, "costs": list(costs),
        "cost_units": "safety-shell firings per arm-hour, in simulation",
        "base_probs": list(q.probs),
        "total_variation_from_base": tilted.total_variation(q),
        "reads_no_target_label": True,
      }
      chosen[name] = d
      print(f"  {name}: {base} {['%.3f' % x for x in q.probs]} -> "
            f"{['%.3f' % x for x in tilted.probs]} (TV "
            f"{tilted.total_variation(q):.3f})")

  (out / "posteriors_60s.json").write_text(json.dumps(chosen, indent=1))
  print(f"\n  wrote {out / 'posterior_report.json'} and posteriors_60s.json")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
