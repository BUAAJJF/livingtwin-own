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


def section_interaction(stage: str, as_json: bool) -> dict:
  """A 3x3 grid: does calibrating one axis survive the other being wrong?

  The reference is the (nominal, nominal) cell, which in an interaction sweep
  is a grid point rather than a separate `nominal` run.  The question is not
  whether the corner is bad -- it will be -- but whether the corner is
  PREDICTABLE from the two edges.  If it is, single-axis calibration composes
  and each parameter can be estimated on its own.  If the corner is much worse
  than either composition rule predicts, the axes are entangled and a
  one-at-a-time posterior will be wrong in a way no amount of data fixes.
  """
  pts = load(stage)
  if not pts:
    print(f"  (no runs in {RESULTS / stage})")
    return {}
  points = {k: Point(k, v) for k, v in pts.items()}

  # Recover each cell's coordinates from the run's own recorded mismatch.
  cells: dict[tuple[float, float], Point] = {}
  names: list[str] = []
  for p in points.values():
    act = {}
    for r in p.runs:
      act = (r.get("mismatch") or {}).get("_active") or {}
      if act:
        break
    ax = sorted(act) if act else []
    for a in ax:
      if a not in names:
        names.append(a)
    key = tuple(float(act.get(n, _nominal_of(n))) for n in sorted(names or ax))
    cells[key] = p
  if len(names) != 2:
    print(f"  ! expected 2 interacting axes, found {names}")
    return {}
  a_name, b_name = sorted(names)
  ref_key = (_nominal_of(a_name), _nominal_of(b_name))
  ref = cells.get(ref_key)
  if ref is None:
    print(f"  ! no reference cell at {ref_key}")
    return {}

  la = sorted({k[0] for k in cells})
  lb = sorted({k[1] for k in cells})
  print(f"  reference cell: {a_name}={ref_key[0]:g}, {b_name}={ref_key[1]:g}"
        f"  ->  {ref.mean('throughput_per_min'):.2f} obj/min "
        f"(sd {ref.sd('throughput_per_min'):.2f}, n={ref.n})")
  print()
  print(f"  throughput (obj/min), rows {a_name}, cols {b_name}:")
  header = "         " + "".join(f"{v:>12g}" for v in lb)
  print(header)
  for va in la:
    row = f"  {va:7g}"
    for vb in lb:
      c = cells.get((va, vb))
      row += f"{c.mean('throughput_per_min'):12.2f}" if c else f"{'-':>12s}"
    print(row)

  rows = []
  print()
  print("  is the corner predictable from the edges?")
  print(f"  {'cell':>22s} {'observed':>9s} {'additive':>9s} {'multipl.':>9s} "
        f"{'obs-mult':>9s} {'beyond CI':>10s}")
  base = ref.mean("throughput_per_min")
  for va in la:
    for vb in lb:
      if va == ref_key[0] or vb == ref_key[1]:
        continue
      cell = cells.get((va, vb))
      ea = cells.get((va, ref_key[1]))
      eb = cells.get((ref_key[0], vb))
      if not (cell and ea and eb):
        continue
      da = ea.mean("throughput_per_min") - base
      db = eb.mean("throughput_per_min") - base
      obs = cell.mean("throughput_per_min")
      add = base + da + db
      mul = base * (1 + da / base) * (1 + db / base)
      e = effect(cell, cell, "throughput_per_min")  # for the shape
      # Uncertainty on the observed corner alone; the two edges enter the
      # prediction as means, so this is the conservative comparison.
      lo, hi = cell.ci("throughput_per_min")
      beyond = "yes" if (mul < lo or mul > hi) else "no"
      print(f"  {f'{va:g},{vb:g}':>22s} {obs:9.2f} {add:9.2f} {mul:9.2f} "
            f"{obs - mul:+9.2f} {beyond:>10s}")
      rows.append({
        "stage": stage, "a": a_name, "b": b_name, "a_value": va, "b_value": vb,
        "reference": base, "edge_a": ea.mean("throughput_per_min"),
        "edge_b": eb.mean("throughput_per_min"), "observed": obs,
        "additive_prediction": add, "multiplicative_prediction": mul,
        "obs_minus_multiplicative": obs - mul,
        "observed_ci": [lo, hi], "outside_ci": beyond == "yes",
        "repeats": cell.n,
      })
      del e

  grid = [{"a_value": k[0], "b_value": k[1], "repeats": c.n,
           **{f"{m}_mean": c.mean(m) for m, *_ in METRICS},
           **{f"{m}_sd": c.sd(m) for m, *_ in METRICS}}
          for k, c in sorted(cells.items())]
  out = {"stage": stage, "axes": [a_name, b_name], "reference_cell": list(ref_key),
         "grid": grid, "interaction": rows}
  if as_json:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stage}_summary.json").write_text(json.dumps(out, indent=1))
    _csv(RESULTS / f"{stage}_grid.csv", grid)
    print(f"\n  -> results/sim2real_sweep/{stage}_summary.json and _grid.csv")
  return out


def _nominal_of(axis: str) -> float:
  from piper_push.perturb import AXES
  return AXES[axis].nominal


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
  # With one repeat per point nothing can separate, so requiring separation
  # here would rank every axis at zero -- which is exactly what screening must
  # not do.  Screening ranks by raw effect and says so; S2 re-measures with
  # repeats and only then is separation meaningful.
  tested = nominal.n >= 2
  print()
  print("  strongest effect per axis, by |Δ| "
        + ("that separates from nominal:" if tested
           else "(RAW -- one repeat, nothing is tested):"))
  ranked = []
  for axis in sorted(by_axis):
    cand = [r for r in rows if r["axis"] == axis]

    def score(r, key):
      e = r[key]
      if math.isnan(e["rel"]):
        return -1.0
      return abs(e["rel"]) if (e["separated"] or not tested) else -1.0

    best_t = max(cand, key=lambda r: score(r, "effect_throughput"), default=None)
    best_p = max(cand, key=lambda r: score(r, "effect_trips"), default=None)
    t_rel = (best_t["effect_throughput"]["rel"]
             if best_t and score(best_t, "effect_throughput") >= 0 else 0.0)
    p_rel = (best_p["effect_trips"]["rel"]
             if best_p and score(best_p, "effect_trips") >= 0 else 0.0)
    ranked.append((axis, t_rel, p_rel,
                   best_t["value"] if best_t else None,
                   best_p["value"] if best_p else None))
  # Sorted on throughput, with tail risk discounted rather than ignored: an
  # axis that leaves throughput alone and quadruples the shell rate is a
  # calibration target, and sorting on throughput alone would bury it.
  ranked.sort(key=lambda x: -max(abs(x[1]), abs(x[2]) / 8))
  for axis, t_rel, p_rel, at_t, at_p in ranked:
    tags = []
    if abs(t_rel) >= 0.10:
      tags.append("throughput >=10%")
    if abs(p_rel) >= 1.0:
      tags.append("tail >=2x")
    tag = ("  <- " + ", ".join(tags)) if tags else ""
    print(f"    {axis:22s} throughput {100 * t_rel:+7.1f}% at {str(at_t):>7s}   "
          f"trips {100 * p_rel:+9.1f}% at {str(at_p):>7s}{tag}")

  out = {"stage": stage, "nominal": {
    "repeats": nominal.n,
    **{m: {"mean": nominal.mean(m), "sd": nominal.sd(m),
           "ci": list(nominal.ci(m))} for m, *_ in METRICS}},
    "points": rows,
    "separation_tested": tested,
    "ranked": [{"axis": a, "throughput_rel": t, "trips_rel": p,
                "at_throughput": vt, "at_trips": vp}
               for a, t, p, vt, vp in ranked]}
  if as_json:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stage}_summary.json").write_text(json.dumps(out, indent=1))
    _csv(RESULTS / f"{stage}_points.csv", rows)
    print(f"\n  -> results/sim2real_sweep/{stage}_summary.json and _points.csv")
    dose_response(stage, out)
  return out


def dose_response(stage: str, out: dict) -> None:
  """One small multiple per axis: throughput against level, with repeat spread.

  Hand-rolled SVG rather than matplotlib.  The figure is a dozen line charts
  and adding a plotting dependency to an environment that has to stay in step
  with mujoco-warp is a poor trade for that.  The CSV remains the source of
  truth; this is for seeing the shape of a dose-response at a glance, which is
  what separates "a real effect" from "one level that happened to be low".
  """
  rows = out.get("points") or []
  if not rows:
    return
  by_axis: dict[str, list[dict]] = defaultdict(list)
  for r in rows:
    if not math.isnan(r["value"]):
      by_axis[r["axis"]].append(r)
  if not by_axis:
    return
  nom = out["nominal"]["throughput_per_min"]["mean"]
  nom_sd = out["nominal"]["throughput_per_min"]["sd"]

  axes = sorted(by_axis, key=lambda a: min(
    (r["effect_throughput"]["rel"] for r in by_axis[a]), default=0.0))
  cols = 4
  rowsn = (len(axes) + cols - 1) // cols
  W, H = 260, 190
  pad_l, pad_b, pad_t, pad_r = 46, 34, 26, 12
  sw, sh = W * cols, H * rowsn

  parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{sw}" height="{sh}" '
    f'viewBox="0 0 {sw} {sh}" font-family="ui-sans-serif,system-ui,sans-serif">',
    f'<rect width="{sw}" height="{sh}" fill="#fbfbfa"/>',
  ]
  for i, axis in enumerate(axes):
    ox, oy = (i % cols) * W, (i // cols) * H
    pts = sorted(by_axis[axis], key=lambda r: r["value"])
    xs = [r["value"] for r in pts]
    ys = [r["throughput_per_min_mean"] for r in pts]
    x0, x1 = min(xs + [0.0]), max(xs + [0.0])
    if x1 - x0 < 1e-12:
      x1 = x0 + 1.0
    y0, y1 = 0.0, max(ys + [nom]) * 1.12

    def px(v): return ox + pad_l + (v - x0) / (x1 - x0) * (W - pad_l - pad_r)
    def py(v): return oy + H - pad_b - (v - y0) / (y1 - y0) * (H - pad_b - pad_t)

    parts.append(f'<line x1="{ox + pad_l}" y1="{oy + H - pad_b}" '
                 f'x2="{ox + W - pad_r}" y2="{oy + H - pad_b}" stroke="#444"/>')
    parts.append(f'<line x1="{ox + pad_l}" y1="{oy + pad_t}" '
                 f'x2="{ox + pad_l}" y2="{oy + H - pad_b}" stroke="#444"/>')
    # The unperturbed level, with its own repeat spread as a band.
    if not math.isnan(nom_sd):
      parts.append(f'<rect x="{ox + pad_l}" y="{py(nom + nom_sd):.1f}" '
                   f'width="{W - pad_l - pad_r}" '
                   f'height="{max(py(nom - nom_sd) - py(nom + nom_sd), 1):.1f}" '
                   f'fill="#b4453c" fill-opacity="0.12"/>')
    parts.append(f'<line x1="{ox + pad_l}" y1="{py(nom):.1f}" '
                 f'x2="{ox + W - pad_r}" y2="{py(nom):.1f}" '
                 f'stroke="#b4453c" stroke-dasharray="4 3"/>')

    d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    parts.append(f'<polyline points="{d}" fill="none" stroke="#2f6f8f" '
                 f'stroke-width="2"/>')
    for r, x, y in zip(pts, xs, ys):
      lo, hi = r["throughput_per_min_ci"]
      if not math.isnan(lo):
        parts.append(f'<line x1="{px(x):.1f}" y1="{py(lo):.1f}" '
                     f'x2="{px(x):.1f}" y2="{py(hi):.1f}" '
                     f'stroke="#2f6f8f" stroke-opacity="0.5" stroke-width="2"/>')
      parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.2" '
                   f'fill="#2f6f8f"/>')
    parts.append(f'<text x="{ox + pad_l}" y="{oy + 16}" font-size="11" '
                 f'fill="#222">{axis}</text>')
    for v in (x0, x1):
      parts.append(f'<text x="{px(v):.1f}" y="{oy + H - pad_b + 13}" '
                   f'font-size="9" fill="#666" text-anchor="middle">{v:g}</text>')
    parts.append(f'<text x="{ox + pad_l - 5}" y="{py(nom):.1f}" font-size="9" '
                 f'fill="#b4453c" text-anchor="end">{nom:.0f}</text>')
    parts.append(f'<text x="{ox + pad_l - 5}" y="{oy + H - pad_b}" '
                 f'font-size="9" fill="#666" text-anchor="end">0</text>')
  parts.append("</svg>")
  path = RESULTS / f"{stage}_dose_response.svg"
  path.write_text("\n".join(parts))
  print(f"  -> results/sim2real_sweep/{path.name}  (objects/min vs level)")


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
    # An interaction stage has no separate `nominal` run: its reference is the
    # (nominal, nominal) grid cell, and the question it answers is different.
    if s.startswith("s3"):
      section_interaction(s, a.json)
    else:
      section(s, a.json)
  print()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
