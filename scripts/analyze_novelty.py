"""Turn the evaluation JSONs into the tables and figures the report quotes.

Nothing here recomputes a rollout and nothing here contains a measured number.
Every figure is derived from ``results/novelty_validation/**/*.json``, which is
the point: a table typed by hand is a table that stops matching the runs behind
it the first time one is re-run.

Confidence intervals resample ENVIRONMENTS, not control steps.  Two consecutive
steps in one environment are about as independent as two consecutive frames of
a video; two different environments share only the policy and the parameter
distribution.  So the environment is the exchangeable unit and the per-env
totals in each JSON are what makes the bootstrap possible.

    python scripts/analyze_novelty.py                      # all sections
    python scripts/analyze_novelty.py --section baseline
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "novelty_validation"
N_BOOT = 2000
BOOT_SEED = 12345


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(section: str) -> dict[str, dict]:
  out = {}
  d = RESULTS / section
  if not d.is_dir():
    return out
  for f in sorted(d.glob("*.json")):
    try:
      out[f.stem] = json.loads(f.read_text())
    except json.JSONDecodeError:
      print(f"  ! {f} is not valid JSON, skipped")
  return out


# ---------------------------------------------------------------------------
# Bootstrap over environments
# ---------------------------------------------------------------------------


class _Rng:
  """A tiny LCG, so a resample does not depend on numpy or on global state."""

  def __init__(self, seed: int):
    self.s = seed & 0xFFFFFFFF

  def randint(self, n: int) -> int:
    self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
    return self.s % n


def bootstrap(per_env: dict[str, list], seconds_per_env: float,
              stat: str, n_boot: int = N_BOOT) -> tuple[float, float, float]:
  """(point estimate, lo, hi) at 95%, resampling environments with replacement.

  ``stat`` names a ratio of two per-environment totals, so the resample is of
  whole arms and the ratio is recomputed from the resampled totals rather than
  averaged over per-arm ratios -- an arm that placed nothing would otherwise
  contribute a 0/0.
  """
  num, den, scale = {
    "throughput_per_min": ("placed", "_seconds", 60.0),
    "trips_per_arm_hour": ("trips", "_seconds", 3600.0),
    "success": ("ok", "_instances", 1.0),
    "drop_rate": ("drops", "grasps", 1.0),
    "tables_per_arm_hour": ("clears", "_seconds", 3600.0),
  }[stat]

  n = len(per_env["placed"])

  def col(name: str) -> list[float]:
    if name == "_seconds":
      return [seconds_per_env] * n
    if name == "_instances":
      return [a + b for a, b in zip(per_env["ok"], per_env["fail"])]
    return per_env[name]

  a, b = col(num), col(den)
  point = scale * sum(a) / max(sum(b), 1e-9)

  rng = _Rng(BOOT_SEED)
  draws = []
  for _ in range(n_boot):
    sa = sb = 0.0
    for _ in range(n):
      i = rng.randint(n)
      sa += a[i]
      sb += b[i]
    draws.append(scale * sa / max(sb, 1e-9))
  draws.sort()
  lo = draws[int(0.025 * n_boot)]
  hi = draws[min(int(0.975 * n_boot), n_boot - 1)]
  return point, lo, hi


def ci(run: dict, stat: str) -> tuple[float, float, float]:
  return bootstrap(run["per_env"], run["per_env_seconds"], stat)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_baseline() -> None:
  runs = load("baseline")
  ref = json.loads((RESULTS / "reference.json").read_text())
  if not runs:
    print("  (no baseline runs yet)")
    return

  tol = ref["tolerance"]
  print(f"  {'run':26s} {'ref':>7s} {'got':>7s} {'95% CI':>16s} {'rel':>7s}  verdict")
  rows = []
  for name in sorted(runs):
    r = runs[name]
    base = name.replace("_seedB", "")
    exp = ref["runs"].get(base)
    got, lo, hi = ci(r, "throughput_per_min")
    if exp is None:
      print(f"  {name:26s} {'-':>7s} {got:7.1f} {f'[{lo:.1f},{hi:.1f}]':>16s}"
            f" {'-':>7s}  (no reference)")
      continue
    e = exp["throughput_per_min"]
    rel = (got - e) / e
    ok = abs(rel) <= tol["throughput_rel"]
    print(f"  {name:26s} {e:7.1f} {got:7.1f} {f'[{lo:.1f},{hi:.1f}]':>16s}"
          f" {100 * rel:+6.1f}%  {'PASS' if ok else 'FAIL'}")
    rows.append({
      "run": name, "reference": e, "reproduced": got, "ci_lo": lo, "ci_hi": hi,
      "rel_error": rel, "within_tolerance": ok,
      "success_ref": exp["success"], "success_got": r["metrics"]["success"],
      "p95_ref": exp["p95_s"], "p95_got": r["metrics"]["p95_s"],
      "trips_ref": exp["trips_per_arm_hour"],
      "trips_got": r["metrics"]["trips_per_arm_hour"],
      "cadence": r["config"].get("redraw_on_place"),
      "seed": r["config"].get("seed_effective"),
      "checkpoint_sha256": r["provenance"]["checkpoint_sha256"],
    })

  print()
  print(f"  {'run':26s} {'succ ref':>9s} {'succ got':>9s} {'p95 ref':>8s}"
        f" {'p95 got':>8s} {'trips ref':>10s} {'trips got':>10s}")
  for r in rows:
    print(f"  {r['run']:26s} {100 * r['success_ref']:8.1f}%"
          f" {100 * r['success_got']:8.1f}% {r['p95_ref']:8.2f}"
          f" {r['p95_got']:8.2f} {r['trips_ref']:10.1f} {r['trips_got']:10.1f}")
  _write("baseline_reproduction", rows)


def section_safety() -> None:
  all_runs = load("safety")
  if not all_runs:
    print("  (no safety runs yet)")
    return
  # Runs are named "<policy>__<filter>"; each policy gets its own baseline,
  # because a filter's worth is how far it moves THAT policy.
  policies: dict[str, dict[str, dict]] = {}
  for name, r in all_runs.items():
    pol, _, filt = name.rpartition("__")
    policies.setdefault(pol or "default", {})[filt or name] = r

  rows = []
  for pol in sorted(policies):
    print()
    print(f"  -- {pol}")
    rows += _safety_one(pol, policies[pol])

  print()
  print("  Safety Gate: a filter passes if it removes >=80% of the trips for")
  print("  <=2% of the throughput.")
  winners = [r for r in rows if r["filter"] != "none"
             and r["delta_trips"] <= -0.80 and r["delta_throughput"] >= -0.02]
  if winners:
    for w in winners:
      print(f"    PASS  {w['policy']}/{w['filter']}: "
            f"trips {100 * w['delta_trips']:+.1f}%, "
            f"throughput {100 * w['delta_throughput']:+.1f}%")
  else:
    print("    no filter met both conditions on any policy")
  _write("safety_pareto", rows)


def _safety_one(policy: str, runs: dict[str, dict]) -> list[dict]:
  base = runs.get("none")
  print(f"  {'filter':22s} {'obj/min':>8s} {'d thru':>7s} {'trips/h':>8s}"
        f" {'d trips':>8s} {'CVaR95':>7s} {'CVaR99':>7s} {'clipped':>8s}")
  rows = []
  for name in sorted(runs):
    r = runs[name]
    thr, tlo, thi = ci(r, "throughput_per_min")
    trp, plo, phi = ci(r, "trips_per_arm_hour")
    js = r["joint_speed"]
    c95, c99 = max(js["cvar95"]), max(js["cvar99"])
    sh = r["shaping"]
    clip = sh["frac_clipped_slew"] + sh["frac_clipped_accel"] + sh["frac_moved_lowpass"]
    d_thr = d_trp = float("nan")
    if base is not None and name != "none":
      b_thr = base["metrics"]["throughput_per_min"]
      b_trp = base["metrics"]["trips_per_arm_hour"]
      d_thr = (thr - b_thr) / b_thr
      d_trp = (trp - b_trp) / max(b_trp, 1e-9)
    print(f"  {name:22s} {thr:8.1f} {100 * d_thr:+6.1f}% {trp:8.1f}"
          f" {100 * d_trp:+7.1f}% {c95:7.3f} {c99:7.3f} {100 * clip:7.2f}%")
    rows.append({
      "policy": policy,
      "filter": name, "throughput": thr, "thr_lo": tlo, "thr_hi": thi,
      "trips_per_arm_hour": trp, "trips_lo": plo, "trips_hi": phi,
      "delta_throughput": d_thr, "delta_trips": d_trp,
      "speed_cvar95": c95, "speed_cvar99": c99,
      "frac_commands_modified": clip,
      "success": r["metrics"]["success"],
      "p95_s": r["metrics"]["p95_s"],
      "cmd_acc_cvar99": max(r["command"]["cmd_acc"]["cvar99"]),
      "cmd_jerk_cvar99": max(r["command"]["cmd_jerk"]["cvar99"]),
      "phase_trips": dict(zip(r["phase"]["names"], r["phase"]["trips"])),
      "shaping": sh,
    })

  if base is not None:
    ph = base["phase"]
    tot = max(sum(ph["trips"]), 1)
    share = ", ".join(f"{n} {100 * t / tot:.0f}%"
                      for n, t in zip(ph["names"], ph["trips"]) if t)
    print(f"    unfiltered trips by phase: {share}")
  return rows


def section_cadence() -> None:
  runs = load("cadence")
  if not runs:
    print("  (no cadence runs yet)")
    return
  print(f"  {'run':38s} {'cadence':22s} {'hid':4s} {'obj/min':>8s}"
        f" {'95% CI':>16s} {'trips/h':>8s} {'stuck':>7s}")
  rows = []
  for name in sorted(runs):
    r = runs[name]
    cfg = r["config"]
    cad = ",".join(cfg.get("redraw_on_place") or []) or "episode"
    hid = "zero" if cfg.get("reset_hidden_on_respawn") else "keep"
    thr, lo, hi = ci(r, "throughput_per_min")
    trp, _, _ = ci(r, "trips_per_arm_hour")
    print(f"  {name:38s} {cad:22s} {hid:4s} {thr:8.1f}"
          f" {f'[{lo:.1f},{hi:.1f}]':>16s} {trp:8.1f}"
          f" {100 * r['metrics']['stuck_fraction']:6.1f}%")
    rows.append({
      "run": name, "cadence": cad, "hidden": hid,
      "throughput": thr, "ci_lo": lo, "ci_hi": hi,
      "trips_per_arm_hour": trp,
      "success": r["metrics"]["success"],
      "stuck_fraction": r["metrics"]["stuck_fraction"],
      "p95_s": r["metrics"]["p95_s"],
      "recurrent": cfg.get("recurrent"),
      "seed": cfg.get("seed_effective"),
      "checkpoint_sha256": r["provenance"]["checkpoint_sha256"],
    })
  _write("cadence_matrix", rows)

  # The comparison the gate is about: same policy, same seed, cadence swapped.
  print()
  print("  EP-All minus OBJ-All, same policy and seed (positive = the leaky")
  print("  environment reads better, which is the effect under test):")
  by = {}
  for r in rows:
    key = (r["run"].rsplit("__", 1)[0], r["hidden"])
    by.setdefault(key, {})[r["cadence"]] = r
  gaps = []
  for (policy, hid), d in sorted(by.items()):
    obj = d.get("shape,mass,friction")
    ep = d.get("episode")
    if not obj or not ep:
      continue
    gap = (ep["throughput"] - obj["throughput"]) / obj["throughput"]
    sep = "disjoint" if (ep["ci_lo"] > obj["ci_hi"] or obj["ci_lo"] > ep["ci_hi"]) \
        else "overlap"
    print(f"    {policy:30s} hidden={hid:4s} {100 * gap:+6.2f}%   CI {sep}")
    gaps.append({"policy": policy, "hidden": hid, "ep_minus_obj_rel": gap,
                 "ci_disjoint": sep == "disjoint",
                 "throughput_obj": obj["throughput"],
                 "throughput_ep": ep["throughput"]})
  _write("cadence_gaps", gaps)


def section_probe() -> None:
  files = sorted((RESULTS / "probe").glob("*.json"))
  if not files:
    print("  (no probe results yet)")
    return
  for f in files:
    d = json.loads(f.read_text())
    print(f"  --- {f.stem}")
    for k, v in d.items():
      if k.startswith("_"):
        continue
      print(f"      {k}: {v}")


def _write(name: str, rows: list[dict]) -> None:
  if not rows:
    return
  out = RESULTS / f"{name}.json"
  out.write_text(json.dumps(rows, indent=1))
  # A CSV alongside, because half the point of a machine-readable result is
  # that somebody can open it without writing a parser.
  keys = sorted({k for r in rows for k in r})
  flat = []
  for r in rows:
    flat.append(",".join(
      json.dumps(r.get(k), separators=(";", ":")) if isinstance(r.get(k), (dict, list))
      else ("" if r.get(k) is None else str(r.get(k)))
      for k in keys))
  (RESULTS / f"{name}.csv").write_text(
    ",".join(keys) + "\n" + "\n".join(flat) + "\n")
  print(f"  -> {out.relative_to(ROOT)} and .csv")


def section_pareto_plot() -> None:
  """Throughput against trips, as an SVG built from the CSV.

  Hand-rolled rather than matplotlib: the figure is two axes and a dozen
  points, and adding a plotting dependency to an environment that has to stay
  in step with mujoco-warp is a poor trade for that.
  """
  f = RESULTS / "safety_pareto.json"
  if not f.exists():
    print("  (no safety results to plot)")
    return
  rows = json.loads(f.read_text())
  W, H, M = 640, 420, 70
  xs = [r["trips_per_arm_hour"] for r in rows]
  ys = [r["throughput"] for r in rows]
  x0, x1 = 0.0, max(xs + [1.0]) * 1.15
  y0, y1 = min(ys) * 0.97, max(ys) * 1.02

  def px(x): return M + (x - x0) / (x1 - x0) * (W - 2 * M)
  def py(y): return H - M - (y - y0) / (y1 - y0) * (H - 2 * M)

  parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">',
    f'<rect width="{W}" height="{H}" fill="#fbfbfa"/>',
    f'<line x1="{M}" y1="{H - M}" x2="{W - M}" y2="{H - M}" stroke="#333"/>',
    f'<line x1="{M}" y1="{M}" x2="{M}" y2="{H - M}" stroke="#333"/>',
    f'<text x="{W / 2}" y="{H - 22}" text-anchor="middle" font-size="13">'
    f'safety-shell trips per arm-hour  (lower is better)</text>',
    f'<text x="18" y="{H / 2}" text-anchor="middle" font-size="13" '
    f'transform="rotate(-90 18 {H / 2})">objects per minute</text>',
  ]
  for r in rows:
    x, y = px(r["trips_per_arm_hour"]), py(r["throughput"])
    lo, hi = px(r["trips_lo"]), px(r["trips_hi"])
    col = "#b4453c" if r["filter"] == "none" else "#2f6f8f"
    parts.append(f'<line x1="{lo:.1f}" y1="{y:.1f}" x2="{hi:.1f}" y2="{y:.1f}" '
                 f'stroke="{col}" stroke-opacity="0.45" stroke-width="2"/>')
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}"/>')
    parts.append(f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" font-size="11" '
                 f'fill="#222">{r["filter"]}</text>')
  parts.append("</svg>")
  out = RESULTS / "safety_pareto.svg"
  out.write_text("\n".join(parts))
  print(f"  -> {out.relative_to(ROOT)}")


SECTIONS = {
  "baseline": ("Phase 0.2  baseline reproduction", section_baseline),
  "safety": ("Phase 1  command-filter Pareto", section_safety),
  "pareto": ("Phase 1  Pareto figure", section_pareto_plot),
  "cadence": ("Phase 2  cadence matrix", section_cadence),
  "probe": ("Phase 2  hidden-state diagnostics", section_probe),
}


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--section", choices=list(SECTIONS), action="append")
  a = p.parse_args()
  for key in (a.section or list(SECTIONS)):
    title, fn = SECTIONS[key]
    print()
    print(f"=== {title}")
    fn()
  print()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
