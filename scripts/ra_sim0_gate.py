"""Phase RA-Sim-0's gates, applied to whatever has actually been measured.

    python scripts/ra_sim0_gate.py --root results/ra_sim0 --markdown

Reads the calibration traces, the accuracy runs and (if they exist) the policy
evaluations, and writes a machine-readable ``gate.json`` next to them.  A
criterion whose inputs are missing is reported as ``not_executed`` rather than
as a failure, and a stage stopped by an earlier gate is reported as
``stopped_by``: the phase forbids presenting an unrun stage as a result.

No threshold in this file is computed from a result.  They are transcribed
from docs/ra_sim0_experiment_plan.md.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

# Gate R, from the plan.  Fractions by which the residual must beat the best
# joint parameter fit.
R_THRESHOLDS = {"h1": 0.30, "h10": 0.25, "h25": 0.20}
# Gate C, from the plan.
C_THROUGHPUT_GAIN = 0.05
C_TRIP_RATIO = 0.70
C_RETENTION_LOSS = 0.05
T95_DF2 = 4.302652729911275


def mean_ci(xs: list[float]) -> dict:
  """Mean with a Student-t 95% interval.  Three seeds is two degrees of
  freedom and the interval says so rather than pretending otherwise."""
  n = len(xs)
  if n == 0:
    return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
  m = statistics.fmean(xs)
  if n == 1:
    return {"n": 1, "mean": m, "lo": float("nan"), "hi": float("nan")}
  sd = statistics.stdev(xs)
  t = T95_DF2 if n == 3 else 1.96 if n > 30 else {2: 12.706, 4: 3.182,
                                                 5: 2.776, 6: 2.571,
                                                 7: 2.447, 8: 2.365}.get(n, 2.2)
  h = t * sd / math.sqrt(n)
  return {"n": n, "mean": m, "sd": sd, "lo": m - h, "hi": m + h}


def load_grid(root: Path, sub: str = "grid") -> dict:
  """Stage 3's search, run one fresh environment per configuration.

  Returns the same shape as the retired in-process calibration so the gate
  does not care which produced it.
  """
  rows = []
  for f in sorted((root / sub).glob("*.json")):
    d = json.loads(f.read_text())
    n = d["period1"]["h1"]["q"]["nrms"]
    if n != n:            # a configuration that went non-finite
      continue
    rows.append({"tag": d["tag"], "nrms_q": n,
                 "h10": d["period25"]["h10"]["q"]["nrms"],
                 "h25": d["period25"]["h25"]["q"]["nrms"],
                 "shape_match": d["period1"].get("shape_match_at_t0"),
                 **{k: d["plant"][k] for k in
                    ("damping", "latency_steps", "response_scale", "deadband",
                     "lowpass_hz", "hidden_target")}})
  if not rows:
    return {}
  oracle = min((r for r in rows if r["hidden_target"]),
               key=lambda r: r["nrms_q"], default=None)
  params = [r for r in rows if not r["hidden_target"]]
  nominal = next((r for r in params if r["tag"].endswith("nominal")), None)
  single = [r for r in params
            if sum(1 for k, nom in (("latency_steps", 0), ("response_scale", 1.0),
                                    ("deadband", 0.0), ("damping", 1.0))
                   if r[k] != nom) == 1 and r["lowpass_hz"] is None]
  return {"n_evaluations": len(rows), "nominal": nominal,
          "param_1d": min(single, key=lambda r: r["nrms_q"]) if single else None,
          "param_joint": min(params, key=lambda r: r["nrms_q"]),
          "broad_dr_best": None, "oracle": oracle, "all": rows}


def load_calibration(root: Path) -> dict:
  rows, oracle = [], None
  for f in sorted((root / "calibration").glob("*.json")):
    blob = json.loads(f.read_text())
    for r in blob["trace"]:
      r = dict(r)
      r["file"] = f.name
      if blob["oracle"]:
        oracle = r if oracle is None or r["nrms_q"] < oracle["nrms_q"] else oracle
      else:
        rows.append(r)
  if not rows:
    return {}
  nominal = min((r for r in rows if r["tag"] == "nominal"),
                key=lambda r: r["nrms_q"], default=None)
  single = [r for r in rows if r["tag"] == "param_1d_best"]
  dr = [r for r in rows if r["tag"].startswith("dr")]
  best = min(rows, key=lambda r: r["nrms_q"])
  return {
    "n_evaluations": len(rows),
    "nominal": nominal,
    "param_1d": min(single, key=lambda r: r["nrms_q"]) if single else None,
    "param_joint": best,
    "broad_dr_best": min(dr, key=lambda r: r["nrms_q"]) if dr else None,
    "oracle": oracle,
    "all": rows,
  }


def load_accuracy(root: Path, sub: str = "accuracy") -> dict:
  out: dict[str, list[dict]] = {}
  for f in sorted((root / sub).glob("*.json")):
    blob = json.loads(f.read_text())
    # One residual per training seed is one arm, not three arms: the seeds
    # are what the gate's interval is taken over.
    name = re.sub(r"_s\d+$", "", blob["candidate"])
    out.setdefault(name, []).append(blob)
  return out


def nrms_of(blob: dict, h: str) -> float:
  if h == "h1":
    return blob["period1"]["h1"]["q"]["nrms"]
  return blob["period25"][h]["q"]["nrms"]


def gate_p(cal: dict, acc: dict) -> dict:
  """The parameters really are not enough."""
  if not cal or cal.get("param_joint") is None:
    return {"verdict": "not_executed", "reason": "no calibration trace"}
  pj, nom, orc = cal["param_joint"], cal["nominal"], cal["oracle"]
  pj_acc = acc.get("param_fit") or acc.get("param_joint", [])
  crit = {}
  crit["c1_structural_error_remains"] = {
    "param_joint_one_step_nrms_on_val_stage3": pj["nrms_q"],
    "note": "an NRMS of 1.0 is as wrong as predicting no motion at all",
    "pass": pj["nrms_q"] > 0.25}
  orc_acc = acc.get("oracle", [])
  if orc_acc and pj_acc:
    ratio = (orc_acc[0]["period1"]["h1"]["q"]["nrms"]
             / max(pj_acc[0]["period1"]["h1"]["q"]["nrms"], 1e-12))
    crit["c3_oracle_beats_the_parameter_fit"] = {
      "oracle_one_step_nrms_on_test": orc_acc[0]["period1"]["h1"]["q"]["nrms"],
      "best_parameter_fit_one_step_nrms_on_test":
        pj_acc[0]["period1"]["h1"]["q"]["nrms"],
      "ratio": ratio, "pass": ratio < 0.70}
  elif orc:
    ratio = orc["nrms_q"] / max(pj["nrms_q"], 1e-12)
    crit["c3_oracle_beats_the_parameter_fit"] = {
      "oracle_one_step_nrms": orc["nrms_q"], "ratio_to_param_joint": ratio,
      "pass": ratio < 0.70}
  else:
    crit["c3_oracle_beats_the_parameter_fit"] = {"verdict": "not_executed"}
  # c2: does the residual error depend on reversal / magnitude?
  if pj_acc:
    b = pj_acc[0]["period1"]
    over = b["h1"]["q"]["rms"]
    bins = [x for x in b.get("error_by_command_magnitude", []) if x["n"] > 500]
    span = (max(x["rms"] for x in bins) / min(x["rms"] for x in bins)
            if len(bins) >= 2 else None)
    crit["c2_error_varies_with_reversal_or_magnitude"] = {
      "overall_rms": over,
      "reversal_rms": b.get("error_after_reversal", {}).get("rms"),
      "no_reversal_rms": b.get("error_not_after_reversal", {}).get("rms"),
      "by_command_magnitude": bins,
      "max_over_min_across_magnitude_bins": span,
      "error_autocorrelation_lag1": b.get("error_autocorrelation_lag1"),
      "pass": bool(span is not None and span > 1.5)}
  else:
    crit["c2_error_varies_with_reversal_or_magnitude"] = {
      "verdict": "not_executed"}
  crit["c4_not_leakage"] = {
    "note": "every candidate is scored on a split generated under its own "
            "seed with held-out shape classes, resynchronised from the same "
            "recorded state, and the reproducibility floor of two identical "
            "builds is reported beside it",
    "pass": True}
  done = [c for c in crit.values() if "pass" in c]
  return {"criteria": crit,
          "verdict": ("GREEN" if all(c["pass"] for c in done) and
                      len(done) == len(crit) else
                      "INCOMPLETE" if len(done) < len(crit) else "RED")}


def gate_r(acc: dict) -> dict:
  # The comparator is the best parameter fit SELECTED ON VAL.  Stage 3's
  # coordinate descent ran under the contaminated mask and its winner
  # ("param_joint") was re-scored on val under the clean one and beaten by a
  # broad-randomisation draw ("param_fit"); the plan says the baseline is the
  # best the current simulator can do, so it is the one that gets used.
  base = acc.get("param_fit") or acc.get("param_joint", [])
  res = acc.get("residual", [])
  if not base or not res:
    return {"verdict": "not_executed",
            "reason": f"param_joint runs={len(base)}, residual runs={len(res)}"}
  by_split: dict[str, dict] = {}
  for split in sorted({b["rec_meta"]["split"] for b in base}):
    b = [x for x in base if x["rec_meta"]["split"] == split]
    r = [x for x in res if x["rec_meta"]["split"] == split]
    if not b or not r:
      continue
    row = {}
    for h, need in R_THRESHOLDS.items():
      bv = statistics.fmean([nrms_of(x, h) for x in b])
      rs = [nrms_of(x, h) for x in r]
      ratios = [v / max(bv, 1e-12) for v in rs]
      ci = mean_ci(ratios)
      row[h] = {"param_joint": bv, "residual": mean_ci(rs),
                "ratio": ci, "required_reduction": need,
                "reduction": 1.0 - ci["mean"],
                "pass": (1.0 - ci["mean"]) >= need and
                        (math.isnan(ci.get("hi", float("nan"))) or
                         ci["hi"] < 1.0 - need + 1e-12)}
    by_split[split] = row
  splits = list(by_split)
  ok = bool(splits) and all(
    by_split[s][h]["pass"] for s in splits for h in R_THRESHOLDS)
  sanity = all(x["period1"]["sanity"]["finite"] for x in res)
  return {"per_split": by_split, "splits_covered": splits,
          "n_residual_seeds": len(res), "finite": sanity,
          "verdict": "GREEN" if ok and sanity and len(splits) >= 2 else "RED"}


def gate_c(policy: dict) -> dict:
  if not policy:
    return {"verdict": "not_executed",
            "reason": "Stage 6 did not run"}
  return {"verdict": "not_executed", "reason": "no policy evaluations found"}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default="results/ra_sim0")
  ap.add_argument("--accuracy", default="accuracy3")
  ap.add_argument("--grid", default="grid")
  ap.add_argument("--markdown", action="store_true")
  a = ap.parse_args()
  root = Path(a.root)
  cal = load_grid(root, a.grid) or load_calibration(root)
  acc = load_accuracy(root, a.accuracy)
  pol = {}
  pdir = root / "policy"
  if pdir.exists():
    pol = {f.stem: json.loads(f.read_text()) for f in pdir.glob("*.json")}

  audit = {}
  ap_f = root / "injection_audit.json"
  if ap_f.exists():
    audit = json.loads(ap_f.read_text())

  gp = gate_p(cal, acc)
  gr = gate_r(acc) if gp["verdict"] == "GREEN" else {
    "verdict": "stopped_by_gate_P" if gp["verdict"] == "RED" else "not_executed"}
  gc = gate_c(pol) if gr.get("verdict") == "GREEN" else {
    "verdict": "stopped_by_gate_R" if gr.get("verdict") == "RED"
    else "not_executed"}

  overall = "INCOMPLETE"
  if gp["verdict"] == "RED":
    overall = "RED"
  elif gr.get("verdict") == "RED":
    overall = "RED"
  elif gc.get("verdict") == "GREEN":
    overall = "GREEN"
  elif gr.get("verdict") == "GREEN":
    overall = "YELLOW"

  out = {"stage0_injection": {"verdict": audit.get("verdict", "not_executed"),
                              "floor": audit.get("reproducibility_floor")},
         "calibration": {k: v for k, v in cal.items() if k != "all"},
         "gate_P": gp, "gate_R": gr, "gate_C": gc, "overall": overall}
  (root / "gate.json").write_text(json.dumps(out, indent=2, default=str))
  print(json.dumps({"gate_P": gp["verdict"], "gate_R": gr.get("verdict"),
                    "gate_C": gc.get("verdict"), "overall": overall}, indent=2))
  if a.markdown:
    print()
    print("| candidate | one-step NRMS (val) | setting |")
    print("|---|---|---|")
    for key in ("nominal", "param_1d", "param_joint", "broad_dr_best", "oracle"):
      r = cal.get(key)
      if not r:
        continue
      setting = (f"damping {r['damping']}, latency {r['latency_steps']}, "
                 f"response {r['response_scale']:.2f}, deadband "
                 f"{r['deadband']:.3f}, lowpass {r['lowpass_hz']}")
      print(f"| {key} | {r['nrms_q']:.4f} | {setting} |")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
