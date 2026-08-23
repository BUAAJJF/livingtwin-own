"""Render the WM1-A result tables as markdown, from the JSON artefacts.

    python scripts/wm1_tables.py --section splits
    python scripts/wm1_tables.py --section identification
    python scripts/wm1_tables.py --section adaptation
    python scripts/wm1_tables.py --section all

The report is written by hand; its tables are not. Transcribing forty numbers
out of five JSON files is a way to publish a typo, and a table nobody can
regenerate is a table nobody can check.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

R = Path("results/wm1_latency")

PRETTY = {
  "B0_prior": "B0 prior",
  "B1a_cmd_joint": "B1a command→joint",
  "B1b_img_proprio": "B1b image→proprio",
  "B2_classifier": "B2 classifier",
  "B3_state": "B3 state matching",
  "B4_action": "B4 latent+action",
  "abl_latent_only": "  ablation: latent only",
  "abl_action_only": "  ablation: action only",
  "DA": "**DA** (fitted weights)",
  "ctrl_clf_shuffled": "*control*: shuffled labels",
  "ctrl_wm_shuffled": "*control*: shuffled θ",
  "ctrl_done_only": "*control*: episode boundaries only",
}
ORDER = list(PRETTY)


def _load(p: Path):
  try:
    return json.loads(p.read_text())
  except Exception:
    return None


def splits() -> str:
  m = _load(R / "data" / "manifest.json")
  if not m:
    return "_(no manifest yet)_"
  out = ["| split | files | sessions | length | arm-hours | GB | shapes | episodes |",
         "|---|---|---|---|---|---|---|---|"]
  for name, s in m["splits"].items():
    out.append(f"| `{name}` | {s['files']} | {s['sessions']} | "
               f"{s['steps']} steps ({s['steps'] / 50:.0f} s) | "
               f"{s['arm_hours']:.1f} | {s['gigabytes']:.2f} | "
               f"{','.join(s['shapes'])} | {s['episode_boundaries']} |")
  checks = m.get("leakage_checks", {})
  out.append("")
  out.append(f"All {len(checks)} leakage checks pass "
             f"({'no failures' if not m.get('failures') else m['failures']}).")
  return "\n".join(out)


def identification(budget: str = "60.0") -> str:
  rep = _load(R / "posterior" / "posterior_report.json")
  if not rep:
    return "_(no posterior report yet)_"
  b = rep["budgets"].get(budget)
  if not b:
    return f"_(no budget {budget})_"
  test = b["methods"]["test"]
  out = [f"Budget {float(budget):.0f} s of one arm "
         f"({int(float(budget) * 50 // rep['length'])} windows), "
         f"{test.get('DA', {}).get('n_sessions', '?')} sessions "
         f"across five domains.", "",
         "| method | top-1 | balanced | mass on truth | entropy (bits) | ECE | "
         "target top-1 | target mass |",
         "|---|---|---|---|---|---|---|---|"]
  for k in ORDER:
    m = test.get(k)
    if not m:
      continue
    t = m.get("target_only") or {}
    out.append(
      f"| {PRETTY[k]} | {m['top1']:.3f} | {m['balanced_accuracy']:.3f} | "
      f"{m['mass_on_truth']:.3f} | {m['entropy_bits']:.2f} | {m['ece']:.3f} | "
      f"{t.get('top1', float('nan')):.3f} | "
      f"{t.get('mass_on_truth', float('nan')):.3f} |")
  w = b.get("da_weights") or {}
  if w:
    out += ["", f"DA weights at this budget: state {w.get('state', 0):.2f}, "
                f"latent {w.get('latent', 0):.2f}, "
                f"action {w.get('action', 0):.2f}."]
  return "\n".join(out)


def budget_curve(metric: str = "balanced_accuracy") -> str:
  rep = _load(R / "posterior" / "posterior_report.json")
  if not rep:
    return "_(no posterior report yet)_"
  buds = sorted(rep["budgets"], key=float)
  out = ["| method | " + " | ".join(f"{float(b):.0f} s" for b in buds) + " |",
         "|---" * (len(buds) + 1) + "|"]
  for k in ORDER:
    row = []
    for b in buds:
      m = rep["budgets"][b]["methods"]["test"].get(k)
      row.append(f"{m[metric]:.3f}" if m else "—")
    if any(v != "—" for v in row):
      out.append(f"| {PRETTY[k]} | " + " | ".join(row) + " |")
  return "\n".join(out)


def confusion(method: str = "DA", budget: str = "60.0") -> str:
  rep = _load(R / "posterior" / "posterior_report.json")
  if not rep:
    return "_(no posterior report yet)_"
  m = rep["budgets"][budget]["methods"]["test"].get(method)
  if not m:
    return f"_(no {method})_"
  lags = [0, 1, 2, 3, 4]
  out = [f"`{method}`, {float(budget):.0f} s — rows are the true domain, "
         f"columns the answer.", "",
         "| true \\ said | " + " | ".join(str(x) for x in lags) + " |",
         "|---" * (len(lags) + 1) + "|"]
  for i, row in enumerate(m["confusion"]):
    cells = " | ".join(f"**{v}**" if j == i else str(v)
                       for j, v in enumerate(row))
    out.append(f"| **{lags[i]}** | {cells} |")
  return "\n".join(out)


def adaptation() -> str:
  an = _load(R / "analysis.json")
  plan = _load(R / "adapt" / "formal.json") or _load(R / "adapt" / "anchor.json")
  if not an:
    return "_(no analysis yet)_"
  by_cfg = {}
  for p in (_load(R / "adapt" / "anchor.json"), _load(R / "adapt" / "screen.json"),
            _load(R / "adapt" / "formal.json")):
    for job in (p or {}).get("jobs", []):
      cfg = re.sub(r"_s\d+$", "", job["tag"])
      e = by_cfg.setdefault(cfg, {"alpha": job["alpha"], "probs": job["probs"],
                                  "methods": set()})
      e["methods"].update(job["methods"])
  out = ["| configuration | α | p_adapt | methods sharing it | target obj/min | "
         "retention obj/min | target trips/h (events) |",
         "|---|---|---|---|---|---|---|"]
  # The two reference points first: what the deployed policy does in the
  # target domain without any adaptation, and what it does at home.  Every
  # row below is measured against these.
  for label, name in (("*no adaptation* (zero-shot)", "zeroshot"),
                      ("*no mismatch* (nominal)", "nominal")):
    g = None
    for k, v in an["groups"].items():
      if k.split("/")[-1] == name:
        g = v
    if not g:
      continue
    b, t = g["throughput"]["bootstrap"], g["trips"]
    cell = f"{b['mean']:.2f} [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}]"
    out.append(f"| {label} | — | — | — | "
               + (cell if name == "zeroshot" else "—") + " | "
               + ("—" if name == "zeroshot" else cell) + " | "
               + f"{t['rate']:.2f} ({int(t['events'])}) |")
  for cfg, e in sorted(by_cfg.items()):
    def g(kind):
      for name, gr in an["groups"].items():
        if name.endswith(f"/pooled:{cfg}_{kind}") or name.endswith(f"/{cfg}_{kind}"):
          return gr
      return None
    t, r = g("target"), g("retention")
    if not t:
      continue
    tb = t["throughput"]["bootstrap"]
    rb = r["throughput"]["bootstrap"] if r else None
    tr = t["trips"]
    probs = "[" + ", ".join(f"{x:.2f}" for x in e["probs"]) + "]"
    out.append(
      f"| `{cfg}` | {e['alpha']:.2f} | {probs} | "
      f"{', '.join(sorted(e['methods']))} | "
      f"{tb['mean']:.2f} [{tb['ci'][0]:.2f}, {tb['ci'][1]:.2f}] | "
      + (f"{rb['mean']:.2f} [{rb['ci'][0]:.2f}, {rb['ci'][1]:.2f}] | "
         if rb else "— | ")
      + f"{tr['rate']:.2f} ({int(tr['events'])}) |")
  del plan
  return "\n".join(out)


def anchors() -> str:
  an = _load(R / "analysis.json")
  if not an:
    return "_(no analysis yet)_"
  out = ["| condition | repeats | obj/min | 95% CI | trips/h | events | dispersion |",
         "|---|---|---|---|---|---|---|"]
  for name in sorted(an["groups"]):
    # The anchors are named for the condition, not the directory: the analysis
    # runs over the digest, so the group prefix is `runs/` and not
    # `equivalence/`.
    if name.split("/")[-1] not in ("nominal", "zeroshot"):
      continue
    g = an["groups"][name]
    b, t = g["throughput"]["bootstrap"], g["trips"]
    out.append(f"| `{name.split('/')[-1]}` | {g['n_repeats']} | "
               f"{b['mean']:.2f} | [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}] | "
               f"{t['rate']:.2f} | {int(t['events'])} | "
               f"{t['dispersion']:.2f} |")
  return "\n".join(out)


def gate() -> str:
  g = _load(R / "gate.json")
  if not g:
    return "_(no gate yet)_"
  out = ["| | criterion | verdict |", "|---|---|---|"]
  for name, c in g["criteria"].items():
    mark = "NOT RUN" if not c["checked"] else ("**PASS**" if c["pass"] else "**FAIL**")
    num, _, rest = name.partition("  ")
    out.append(f"| {num} | {rest} | {mark} |")
  out += ["", f"**{g['verdict']}** — {g['why']}"]
  return "\n".join(out)


SECTIONS = {
  "splits": splits,
  "identification": identification,
  "budget_curve": budget_curve,
  "confusion": confusion,
  "adaptation": adaptation,
  "anchors": anchors,
  "gate": gate,
}


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--section", default="all", choices=[*SECTIONS, "all"])
  a = p.parse_args()
  names = list(SECTIONS) if a.section == "all" else [a.section]
  for n in names:
    print(f"\n### {n}\n")
    print(SECTIONS[n]())
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
