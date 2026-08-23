"""Turn the Phase WM0 sweep JSONs into tables, effect sizes and a verdict.

Nothing here contains a measured number and nothing recomputes a rollout.

Two statistical points shape the whole file.

**Repeats are the unit of uncertainty, not the bootstrap.**  Phase 0-2 found
the simulator is not reproducible: five identical commands spanned 2.9%
(novelty_validation_phase_0_2.md 7.5).  So each sweep point is run as three
independent processes with different seeds, and the interval quoted for a
point is a t-interval over those repeats.  The within-run bootstrap is still
computed, but it answers a narrower question and is not what an effect is
tested against.

**An effect is a difference between two such points**, so its uncertainty is
the two combined -- reported as a Welch interval on the difference, and as a
standardised effect size, rather than as a raw percentage that looks precise.

    python scripts/analyze_sweep.py --stage s1
    python scripts/analyze_sweep.py --stage s2 --json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "sim2real_sweep"

METRICS = [
  ("throughput_per_min", "obj/min", 1.0, "higher"),
  ("success", "%", 100.0, "higher"),
  ("drop_rate", "%", 100.0, "lower"),
  ("p95_s", "s", 1.0, "lower"),
  ("stuck_fraction", "%", 100.0, "lower"),
  ("trips_per_arm_hour", "/arm-h", 1.0, "lower"),
]

# Student-t 97.5% points; repeats are few, so the normal approximation is not
# good enough and reaching for scipy for one constant is not worth it.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _t(df: int) -> float:
  return T975.get(df, 1.96)


def load(stage: str) -> dict[str, list[dict]]:
  """Group runs by point label, dropping the repeat suffix."""
  d = RESULTS / stage
  out: dict[str, list[dict]] = defaultdict(list)
  if not d.is_dir():
    return out
  for f in sorted(d.glob("*.json")):
    try:
      data = json.loads(f.read_text())
    except json.JSONDecodeError:
      print(f"  ! {f.name} is not valid JSON, skipped")
      continue
    if "metrics" not in data:
      continue
    point = f.stem.rsplit("__r", 1)[0]
    data["_file"] = f.name
    out[point].append(data)
  return out


class Point:
  """One sweep point: several independent repeats of the same condition."""

  def __init__(self, label: str, runs: list[dict]):
    self.label = label
    self.runs = runs
    self.n = len(runs)
    self.axis, self.value = _parse(label, runs)

  def series(self, metric: str) -> list[float]:
    return [r["metrics"][metric] for r in self.runs
            if r["metrics"].get(metric) is not None]

  def mean(self, metric: str) -> float:
    s = self.series(metric)
    return sum(s) / len(s) if s else float("nan")

  def sd(self, metric: str) -> float:
    s = self.series(metric)
    if len(s) < 2:
      return float("nan")
    m = sum(s) / len(s)
    return math.sqrt(sum((x - m) ** 2 for x in s) / (len(s) - 1))

  def ci(self, metric: str) -> tuple[float, float]:
    s, m, sd = self.series(metric), self.mean(metric), self.sd(metric)
    if len(s) < 2:
      return (float("nan"), float("nan"))
    h = _t(len(s) - 1) * sd / math.sqrt(len(s))
    return (m - h, m + h)


def _parse(label: str, runs: list[dict]) -> tuple[str, float]:
  """Which axis this point perturbed, and to what.

  Read from the run's own recorded mismatch rather than from the label, so a
  mislabelled job cannot silently be plotted at the wrong x.
  """
  active = {}
  for r in runs:
    active = (r.get("mismatch") or {}).get("_active") or {}
    if active:
      break
  if not active:
    return ("nominal", 0.0)
  if len(active) == 1:
    k, v = next(iter(active.items()))
    return (k, float(v))
  return ("+".join(sorted(active)), float("nan"))


def effect(a: Point, b: Point, metric: str) -> dict:
  """b relative to a, with the uncertainty of BOTH points carried through."""
  ma, mb = a.mean(metric), b.mean(metric)
  sa, sb = a.sd(metric), b.sd(metric)
  na, nb = len(a.series(metric)), len(b.series(metric))
  diff = mb - ma
  rel = diff / ma if ma not in (0.0,) and not math.isnan(ma) else float("nan")
  out = {"delta": diff, "rel": rel, "mean_ref": ma, "mean": mb,
         "n_ref": na, "n": nb}
  if na < 2 or nb < 2 or math.isnan(sa) or math.isnan(sb):
    out.update(se=float("nan"), lo=float("nan"), hi=float("nan"),
               cohens_d=float("nan"), separated=None)
    return out
  se = math.sqrt(sa * sa / na + sb * sb / nb)
  # Welch-Satterthwaite, so a point whose repeats disagree is not given the
  # confidence of one whose repeats agree.
  num = (sa * sa / na + sb * sb / nb) ** 2
  den = ((sa * sa / na) ** 2 / max(na - 1, 1)
         + (sb * sb / nb) ** 2 / max(nb - 1, 1))
  df = int(num / den) if den > 0 else 1
  h = _t(max(df, 1)) * se
  pooled = math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb)
                     / max(na + nb - 2, 1))
  out.update(se=se, lo=diff - h, hi=diff + h, df=df,
             cohens_d=diff / pooled if pooled > 0 else float("nan"),
             separated=bool(abs(diff) > h))
  return out


def section(stage: str, as_json: bool) -> dict:
  pts = load(stage)
  if not pts:
    print(f"  (no runs in {RESULTS / stage})")
    return {}
  points = {k: Point(k, v) for k, v in pts.items()}
  nominal = points.get("nominal")
  if nominal is None:
    print("  ! no 'nominal' point -- every effect is measured against it")
    return {}

  print(f"  reference: nominal, {nominal.n} repeats")
  for m, unit, sc, _ in METRICS:
    print(f"    {m:22s} {sc * nominal.mean(m):8.2f} {unit:8s}"
          f" sd {sc * nominal.sd(m):6.3f}  "
          f"[{sc * nominal.ci(m)[0]:.2f}, {sc * nominal.ci(m)[1]:.2f}]")
  print()

  by_axis: dict[str, list[Point]] = defaultdict(list)
  for p in points.values():
    if p.label == "nominal":
      continue
    by_axis[p.axis].append(p)

  rows = []
  print(f"  {'axis':22s} {'level':>9s} {'n':>2s} "
        f"{'obj/min':>9s} {'Δ%':>7s} {'sep':>4s} "
        f"{'trips/h':>9s} {'Δ%':>8s} {'sep':>4s} {'p95':>6s} {'drop%':>6s}")
  for axis in sorted(by_axis):
    for p in sorted(by_axis[axis], key=lambda q: (math.isnan(q.value), q.value)):
      e_thr = effect(nominal, p, "throughput_per_min")
      e_trp = effect(nominal, p, "trips_per_arm_hour")
      e_drp = effect(nominal, p, "drop_rate")
      e_p95 = effect(nominal, p, "p95_s")
      sep_t = "yes" if e_thr["separated"] else ("no" if e_thr["separated"] is False else "?")
      sep_p = "yes" if e_trp["separated"] else ("no" if e_trp["separated"] is False else "?")
      print(f"  {axis:22s} {p.value:9.4g} {p.n:2d} "
            f"{p.mean('throughput_per_min'):9.2f} "
            f"{100 * e_thr['rel']:+7.1f} {sep_t:>4s} "
            f"{p.mean('trips_per_arm_hour'):9.2f} "
            f"{100 * e_trp['rel']:+8.1f} {sep_p:>4s} "
            f"{p.mean('p95_s'):6.2f} {100 * p.mean('drop_rate'):6.2f}")
      rows.append({
        "stage": stage, "point": p.label, "axis": axis, "value": p.value,
        "repeats": p.n,
        "files": [r["_file"] for r in p.runs],
        **{f"{m}_mean": p.mean(m) for m, *_ in METRICS},
        **{f"{m}_sd": p.sd(m) for m, *_ in METRICS},
        **{f"{m}_ci": list(p.ci(m)) for m, *_ in METRICS},
        "effect_throughput": e_thr,
        "effect_trips": e_trp,
        "effect_drop": e_drp,
        "effect_p95": e_p95,
      })

  # Which axes are worth taking to the next stage.
  print()
  print("  strongest effect per axis (by |Δ| that separates from nominal):")
  ranked = []
  for axis in sorted(by_axis):
    best_t = max((r for r in rows if r["axis"] == axis),
                 key=lambda r: abs(r["effect_throughput"]["rel"])
                 if r["effect_throughput"]["separated"] else -1, default=None)
    best_p = max((r for r in rows if r["axis"] == axis),
                 key=lambda r: abs(r["effect_trips"]["rel"])
                 if r["effect_trips"]["separated"] else -1, default=None)
    t_rel = best_t["effect_throughput"]["rel"] if best_t and best_t["effect_throughput"]["separated"] else 0.0
    p_rel = best_p["effect_trips"]["rel"] if best_p and best_p["effect_trips"]["separated"] else 0.0
    ranked.append((axis, t_rel, p_rel, best_t["value"] if best_t else None))
  ranked.sort(key=lambda x: -max(abs(x[1]), abs(x[2]) / 4))
  for axis, t_rel, p_rel, at in ranked:
    tag = ""
    if abs(t_rel) >= 0.10:
      tag = "  <- throughput gap >=10%"
    elif abs(p_rel) >= 1.0:
      tag = "  <- tail risk >=2x"
    print(f"    {axis:22s} throughput {100 * t_rel:+7.1f}%   "
          f"trips {100 * p_rel:+8.1f}%   at {at}{tag}")

  out = {"stage": stage, "nominal": {
    "repeats": nominal.n,
    **{m: {"mean": nominal.mean(m), "sd": nominal.sd(m),
           "ci": list(nominal.ci(m))} for m, *_ in METRICS}},
    "points": rows,
    "ranked": [{"axis": a, "throughput_rel": t, "trips_rel": p, "at": v}
               for a, t, p, v in ranked]}
  if as_json:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stage}_summary.json").write_text(json.dumps(out, indent=1))
    _csv(RESULTS / f"{stage}_points.csv", rows)
    print(f"\n  -> results/sim2real_sweep/{stage}_summary.json and _points.csv")
  return out


def _csv(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  keys = sorted({k for r in rows for k in r})
  lines = [",".join(keys)]
  for r in rows:
    lines.append(",".join(
      json.dumps(r.get(k), separators=(";", ":")) if isinstance(r.get(k), (dict, list))
      else ("" if r.get(k) is None else str(r.get(k)))
      for k in keys))
  path.write_text("\n".join(lines) + "\n")


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--stage", action="append",
                 help="s1, s2, s3 ... (default: all present)")
  p.add_argument("--json", action="store_true", help="write summary files")
  a = p.parse_args()
  stages = a.stage or [d.name for d in sorted(RESULTS.iterdir())
                       if d.is_dir()] if RESULTS.is_dir() else []
  for s in stages:
    print()
    print(f"=== Phase WM0 {s}")
    section(s, a.json)
  print()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
