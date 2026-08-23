"""Evaluate the Phase WM0 gate from the result files, mechanically.

The verdict decides whether to spend weeks building a world model, so it
should not be a judgement call made while looking at the numbers.  Each
criterion is checked against the machine-readable summaries and the verdict
falls out; where a criterion cannot be evaluated because a stage did not run,
it says so rather than defaulting either way.

    python scripts/wm0_gate.py
    python scripts/wm0_gate.py --json results/sim2real_sweep/gate.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "sim2real_sweep"

# A perception-side axis whose failure is a segmentation/masking failure rather
# than a physics mismatch.  Criterion 3 exists so that a Green verdict cannot
# rest on these alone: their fix is a perception fix, not a calibrated
# simulator.
PERCEPTION_ONLY = {"depth_dropout", "depth_dropout_blob"}

THROUGHPUT_GAP = 0.10   # criterion 2a
TAIL_FACTOR = 2.0       # criterion 2b, as a multiplier on the nominal rate
IDENT_MARGIN = 0.10     # criterion 4: accuracy over the shuffled control
RECOVERY_MIN = 0.30
"""Criterion 5: the fraction of the ZERO-SHOT-TO-NOMINAL GAP the oracle closes.

    recovery = (J_oracle - J_zero_shot) / (J_nominal - J_zero_shot)

Not the relative gain over zero-shot, which is what this first computed.  The
two differ a lot and the difference decides the verdict: on the latency domain
the oracle takes 41.85 to 49.79 objects/min, which is a 19% gain over
zero-shot but closes 57% of the 14.0-object gap to the unperturbed 55.86.  The
gap fraction is the quantity the README defines `recovery` as and the one a
learned method will later be scored on, so it is the one the gate must use."""


def _load(p: Path):
  try:
    return json.loads(p.read_text())
  except Exception:
    return None


def criterion_1_and_2(s2: dict) -> dict:
  """Factors whose effect clears its own uncertainty, and how large."""
  out = {"qualifying": [], "checked": bool(s2)}
  if not s2:
    return out
  for r in s2.get("points", []):
    e_t, e_p = r["effect_throughput"], r["effect_trips"]
    sep_t = bool(e_t.get("separated"))
    sep_p = bool(e_p.get("separated"))
    big_t = sep_t and not math.isnan(e_t["rel"]) and abs(e_t["rel"]) >= THROUGHPUT_GAP
    big_p = sep_p and not math.isnan(e_p["rel"]) and e_p["rel"] >= (TAIL_FACTOR - 1.0)
    if big_t or big_p:
      out["qualifying"].append({
        "axis": r["axis"], "value": r["value"],
        "throughput_rel": e_t["rel"], "trips_rel": e_p["rel"],
        "via": ("throughput" if big_t else "") + ("+tail" if big_p else ""),
        "perception_only": r["axis"] in PERCEPTION_ONLY,
      })
  axes = {q["axis"] for q in out["qualifying"]}
  phys = {a for a in axes if a not in PERCEPTION_ONLY}
  out["n_axes"] = len(axes)
  out["axes"] = sorted(axes)
  out["physics_axes"] = sorted(phys)
  out["c1_two_or_more_separated"] = len(axes) >= 2
  out["c2_size"] = len(out["qualifying"]) > 0
  out["c3_not_only_segmentation"] = len(phys) >= 1
  return out


def criterion_4(ident_dir: Path) -> dict:
  """Reward-free history beats its own permutation control."""
  files = sorted(ident_dir.glob("*.json")) if ident_dir.is_dir() else []
  out = {"checked": bool(files), "domains": []}
  for f in files:
    d = _load(f)
    if not d or not d.get("sets"):
      continue
    best = max(d["sets"].items(),
               key=lambda kv: kv[1]["accuracy"] - kv[1]["shuffled_control"])
    name, r = best
    out["domains"].append({
      "file": f.name, "axis": d["axis"], "holdout_shape": d["config"]["holdout_shape"],
      "best_set": name, "accuracy": r["accuracy"],
      "balanced_accuracy": r["balanced_accuracy"],
      "majority_chance": r["majority_chance"],
      "shuffled_control": r["shuffled_control"],
      "over_control": r["accuracy"] - r["shuffled_control"],
      "over_chance": r["accuracy"] - r["majority_chance"],
      "passes": (r["accuracy"] - r["shuffled_control"]) >= IDENT_MARGIN
                and (r["accuracy"] - r["majority_chance"]) >= IDENT_MARGIN,
    })
  passing = [d for d in out["domains"] if d["passes"]]
  held = [d for d in passing if d["holdout_shape"]]
  out["n_pass"] = len(passing)
  out["n_pass_heldout"] = len(held)
  out["c4_identifiable"] = len(passing) >= 1
  out["c4_generalises"] = len(held) >= 1
  return out


def criterion_5(oracle_dir: Path, j_nominal: float | None) -> dict:
  """Does knowing the parameter let simulation recover the loss?"""
  out = {"checked": oracle_dir.is_dir(), "j_nominal": j_nominal, "domains": []}
  if not oracle_dir.is_dir():
    return out
  tags = sorted({f.stem.split("_")[0] for f in oracle_dir.glob("*.json")})
  for tag in tags:
    def series(kind: str) -> list[float]:
      vals = []
      for f in sorted(oracle_dir.glob(f"{tag}_{kind}__r*.json")):
        d = _load(f)
        if d:
          vals.append(d["metrics"]["throughput_per_min"])
      return vals

    zs, orc, ret = series("zeroshot"), series("oracle"), series("retention")
    if not zs or not orc:
      continue
    mz, mo = sum(zs) / len(zs), sum(orc) / len(orc)
    mr = sum(ret) / len(ret) if ret else float("nan")
    gap = (j_nominal - mz) if j_nominal is not None else float("nan")
    # The fraction of the gap to the unperturbed policy that the oracle
    # closes.  This is what the README defines `recovery` as, and what a
    # learned method will later be scored against.
    frac = (mo - mz) / gap if gap and not math.isnan(gap) and gap > 0 else float("nan")
    out["domains"].append({
      "tag": tag, "n": [len(zs), len(orc), len(ret)],
      "zero_shot": mz, "oracle": mo, "retention": mr,
      "gap_to_nominal": gap,
      "absolute_gain": mo - mz,
      "relative_gain_over_zero_shot": (mo - mz) / mz if mz else float("nan"),
      "recovery_fraction_of_gap": frac,
      # What adapting cost back in the unperturbed domain.
      "retention_rel": ((mr - j_nominal) / j_nominal
                        if j_nominal and not math.isnan(mr) else float("nan")),
      "recovers": (not math.isnan(frac)) and frac >= RECOVERY_MIN,
    })
  out["c5_recoverable"] = any(d["recovers"] for d in out["domains"])
  return out


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--json", default=None)
  a = p.parse_args()

  s2 = _load(RESULTS / "s2_summary.json")
  c12 = criterion_1_and_2(s2)
  c4 = criterion_4(RESULTS / "ident")
  # The unperturbed reference the recovery fraction is measured against.
  j_nominal = None
  if s2 and s2.get("nominal"):
    j_nominal = s2["nominal"]["throughput_per_min"]["mean"]
  c5 = criterion_5(RESULTS / "oracle", j_nominal)

  checks = [
    ("1  >=2 factors separated from repeat uncertainty",
     c12.get("c1_two_or_more_separated"), c12["checked"]),
    (f"2  gap >={100 * THROUGHPUT_GAP:.0f}% throughput or >={TAIL_FACTOR:g}x tail",
     c12.get("c2_size"), c12["checked"]),
    ("3  not only segmentation/mask failure",
     c12.get("c3_not_only_segmentation"), c12["checked"]),
    ("4  reward-free history beats its permutation control",
     c4.get("c4_identifiable"), c4["checked"]),
    ("4b   ... and survives held-out object shapes",
     c4.get("c4_generalises"), c4["checked"]),
    (f"5  oracle closes >={100 * RECOVERY_MIN:.0f}% of the zero-shot gap",
     c5.get("c5_recoverable"), c5["checked"]),
  ]

  print()
  print("  Phase WM0 gate")
  print()
  for name, ok, checked in checks:
    mark = "NOT RUN" if not checked else ("PASS" if ok else "FAIL")
    print(f"    [{mark:^7s}] {name}")

  core = [c for _, c, ck in checks[:4] if ck]
  unrun = [n for n, _, ck in checks if not ck]
  gap_ok = all(x for x in [c12.get("c1_two_or_more_separated"),
                           c12.get("c2_size"),
                           c12.get("c3_not_only_segmentation")] if x is not None)

  if unrun:
    verdict = "INCOMPLETE"
    why = "not every criterion could be evaluated: " + "; ".join(unrun)
  elif gap_ok and c4.get("c4_identifiable") and c5.get("c5_recoverable"):
    verdict = "GREEN"
    why = ("a real gap on multiple physics axes, identifiable from reward-free "
           "data, and recoverable when the parameter is known")
  elif gap_ok and not c4.get("c4_identifiable"):
    verdict = "YELLOW"
    why = ("the gap is real but passive rollouts do not identify the domain; "
           "next step is more deployable observation, a short active probe, or "
           "a different parameterisation -- not a world model over this data")
  elif gap_ok and not c5.get("c5_recoverable"):
    verdict = "YELLOW"
    why = ("the gap is real and identifiable but simulation PPO does not "
           "recover it even knowing the target; the adaptation target itself "
           "is wrong, so calibrating it better will not help")
  elif not gap_ok:
    verdict = "RED"
    why = ("no persistent mismatch of consequence survives the repeat "
           "uncertainty, or what does is segmentation rather than physics; "
           "domain randomisation already covers the difference")
  else:
    verdict = "YELLOW"
    why = "mixed evidence; see the per-criterion table"

  print()
  print(f"  VERDICT: {verdict}")
  print(f"  {why}")
  print()

  out = {"verdict": verdict, "why": why,
         "thresholds": {"throughput_gap": THROUGHPUT_GAP,
                        "tail_factor": TAIL_FACTOR,
                        "ident_margin": IDENT_MARGIN,
                        "recovery_min": RECOVERY_MIN},
         "criteria": {n: {"pass": ok, "checked": ck}
                      for n, ok, ck in checks},
         "gap": c12, "identifiability": c4, "oracle": c5}
  if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"  wrote {a.json}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
