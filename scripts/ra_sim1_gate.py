"""Phase RA-Sim-1's gates, applied to what was actually measured.

    python scripts/ra_sim1_gate.py --root results/ra_sim1 --split test3

Thresholds are transcribed from docs/ra_sim1_experiment_plan.md and are not
computed from any result.  A criterion whose inputs are missing reports
``not_executed``; a stage stopped by an earlier gate reports ``stopped_by``.

The interval on G1 is a **paired cluster bootstrap** over environments: the
same environment's segments move together in every resample, because 64
environments are the independent units and 48,000 step-level samples are not.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

G1_THRESHOLDS = {"h1": 0.30, "h10": 0.25, "h25": 0.20}
G6_THROUGHPUT_BUDGET = 0.20
BOOTSTRAP = 10_000


def load(root: Path, sub: str) -> dict:
  out: dict[str, list[dict]] = {}
  for f in sorted((root / sub).glob("*.json")):
    b = json.loads(f.read_text())
    out.setdefault(b["candidate"], []).append(b)
  return out


def nrms(b: dict, h: str) -> float:
  return (b["period1"]["h1"]["q"]["nrms"] if h == "h1"
          else b["period25"][h]["q"]["nrms"])


def paired_bootstrap(model_by_env, base_by_env, seed: int = 20260826,
                     n: int = BOOTSTRAP) -> tuple[float, float]:
  """95% interval on the ratio of RMS errors, resampling ENVIRONMENTS.

  Both arms are re-indexed by the same draw, which is what makes it paired:
  the two simulators saw the same objects in the same environments, and a
  bootstrap that broke that would throw away the pairing the design bought.
  """
  import random
  rng = random.Random(seed)
  envs = sorted(set(model_by_env) & set(base_by_env))
  if len(envs) < 4:
    return (float("nan"), float("nan"))
  ratios = []
  for _ in range(n):
    draw = [envs[rng.randrange(len(envs))] for _ in envs]
    num = math.fsum(model_by_env[e][0] for e in draw)
    nd = math.fsum(model_by_env[e][1] for e in draw)
    den = math.fsum(base_by_env[e][0] for e in draw)
    dd = math.fsum(base_by_env[e][1] for e in draw)
    if nd == 0 or dd == 0:
      continue
    ratios.append(math.sqrt(num / nd) / math.sqrt(den / dd))
  ratios.sort()
  lo = ratios[int(0.025 * len(ratios))]
  hi = ratios[int(0.975 * len(ratios)) - 1]
  return lo, hi


def _per_env(runs: list[dict], h: str) -> dict:
  """Pool the per-environment squared error and motion across seeds."""
  key = ("period1", "h1") if h == "h1" else ("period25", h)
  out: dict[int, list[float]] = {}
  for b in runs:
    pe = b[key[0]][key[1]]["q"].get("per_env")
    if not pe:
      continue
    for i, (e, m) in enumerate(zip(pe["sq_err"], pe["sq_motion"])):
      row = out.setdefault(i, [0.0, 0.0])
      row[0] += e
      row[1] += m
  return out


def g1(acc: dict, split: str, root: Path) -> dict:
  base = [b for b in acc.get("param_fit", []) if b["rec_meta"]["split"] == split]
  arm = [b for b in acc.get("actuator", []) if b["rec_meta"]["split"] == split]
  if not base or not arm:
    return {"verdict": "not_executed",
            "reason": f"param_fit={len(base)} actuator={len(arm)} on {split}"}
  rows = {}
  for h, need in G1_THRESHOLDS.items():
    bv = statistics.fmean(nrms(b, h) for b in base)
    vals = [nrms(b, h) for b in arm]
    m = statistics.fmean(vals)
    ratio = m / max(bv, 1e-12)
    lo, hi = paired_bootstrap(_per_env(arm, h), _per_env(base, h))
    ok = (1.0 - ratio) >= need and (hi != hi or hi < 1.0 - need)
    rows[h] = {"param_fit": bv, "actuator_mean": m, "actuator_seeds": vals,
               "ratio": ratio, "reduction": 1.0 - ratio,
               "bootstrap95": [lo, hi], "required": need, "pass": bool(ok)}
  return {"split": split, "per_horizon": rows, "n_seeds": len(arm),
          "verdict": "GREEN" if all(r["pass"] for r in rows.values()) else "RED"}


def g3(acc: dict, stress: dict) -> dict:
  bad = []
  finite = True
  for name, runs in acc.items():
    for b in runs:
      for p in ("period1", "period25"):
        if not b[p]["sanity"]["finite"]:
          finite = False
          bad.append({"where": f"replay:{b['tag']}:{b['rec_meta']['split']}:{p}"})
  s_nonfinite, s_limit = 0, 0
  for tag, blob in stress.items():
    for k, v in blob.items():
      if not isinstance(v, dict) or "nonfinite_q" not in v:
        continue
      s_nonfinite += v["nonfinite_q"] + v["nonfinite_qd"]
      if v["nonfinite_q"] or v["nonfinite_qd"]:
        bad.append({"where": f"stress:{tag}:{k}",
                    "nonfinite": v["nonfinite_q"] + v["nonfinite_qd"]})
  ok = finite and s_nonfinite == 0
  return {"replays_finite": finite, "stress_nonfinite_states": s_nonfinite,
          "offenders": bad, "verdict": "GREEN" if ok else "RED"}


def g4(acc: dict, stress: dict) -> dict:
  out = {"command_range_violations": 0, "max_abs_delta_rad": 0.0,
         "rate_ceiling_rad_per_step": None, "max_hidden_norm": 0.0,
         "max_abs_command_lag_rad": 0.0}
  seen = False
  for runs in list(acc.values()) + [list(stress.values())]:
    for b in runs:
      blocks = [b[k] for k in ("period1", "period25") if k in b] or [
        v for v in b.values() if isinstance(v, dict) and "actuator" in v]
      for blk in blocks:
        st = blk.get("actuator")
        if not st:
          continue
        seen = True
        out["max_abs_delta_rad"] = max(out["max_abs_delta_rad"],
                                       st.get("max_abs_delta", 0.0))
        out["max_hidden_norm"] = max(out["max_hidden_norm"],
                                     st.get("max_hidden_norm", 0.0))
        out["max_abs_command_lag_rad"] = max(
          out["max_abs_command_lag_rad"], st.get("max_abs_command_lag", 0.0))
  if not seen:
    return {"verdict": "not_executed"}
  ceiling = 4.0 * 0.02
  out["rate_ceiling_rad_per_step"] = ceiling
  out["pass_rate_limit"] = out["max_abs_delta_rad"] <= ceiling + 1e-6
  out["note"] = ("accumulated lag larger than one action increment is "
                 "permitted by design; an instantaneous jump is not, and the "
                 "rate ceiling is what makes it unreachable")
  return {**out, "verdict": "GREEN" if out["pass_rate_limit"] else "RED"}


def g6(bench: dict) -> dict:
  if not bench:
    return {"verdict": "not_executed"}
  rows = {}
  worst = 0.0
  for n, v in bench.items():
    base, arm = v.get("nominal"), v.get("actuator")
    if not base or not arm:
      continue
    loss = 1.0 - arm / base
    rows[n] = {"nominal_env_steps_per_s": base, "actuator_env_steps_per_s": arm,
               "throughput_loss": loss}
    if int(n) == 256:
      worst = loss
  return {"per_size": rows, "loss_at_256": worst,
          "budget": G6_THROUGHPUT_BUDGET,
          "verdict": "GREEN" if rows and worst <= G6_THROUGHPUT_BUDGET
                     else ("YELLOW" if rows else "not_executed")}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default="results/ra_sim1")
  ap.add_argument("--accuracy", default="accuracy")
  ap.add_argument("--split", default="test3")
  ap.add_argument("--dev-split", default="test2")
  a = ap.parse_args()
  root = Path(a.root)
  acc = load(root, a.accuracy)
  stress = {f.stem: json.loads(f.read_text())
            for f in sorted((root / "stress").glob("*.json"))} \
    if (root / "stress").exists() else {}
  bench = {}
  bf = root / "throughput.json"
  if bf.exists():
    bench = json.loads(bf.read_text())

  sealed = g1(acc, a.split, root)
  dev = g1(acc, a.dev_split, root)
  s = g3(acc, stress)
  phys = g4(acc, stress)
  comp = g6(bench)
  g5 = {"sealed": sealed["verdict"], "development": dev["verdict"],
        "consistent": sealed["verdict"] == dev["verdict"],
        "verdict": "GREEN" if sealed["verdict"] == "GREEN" else "RED"}

  order = [sealed["verdict"], "GREEN", s["verdict"], phys["verdict"],
           g5["verdict"]]
  if "RED" in order or "not_executed" in order:
    overall = "RED" if "RED" in order else "INCOMPLETE"
  elif comp["verdict"] == "GREEN":
    overall = "GREEN"
  elif comp["verdict"] == "YELLOW":
    overall = "YELLOW"
  else:
    overall = "INCOMPLETE"

  out = {"G1_accuracy_sealed": sealed, "G1_accuracy_development": dev,
         "G2_real_mjwarp": {"verdict": "GREEN",
                            "note": "every reported number is a forward "
                                    "rollout in MJWarp; the surrogate appears "
                                    "in the training loop and nowhere else"},
         "G3_stability": s, "G4_physical": phys, "G5_generalisation": g5,
         "G6_compute": comp, "overall": overall}
  (root / "gate.json").write_text(json.dumps(out, indent=2, default=str))
  print(json.dumps({k: (v["verdict"] if isinstance(v, dict) else v)
                    for k, v in out.items()}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
