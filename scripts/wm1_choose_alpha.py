"""Choose the mixing weight, by a rule written down before the runs finished.

    python scripts/wm1_choose_alpha.py --analysis results/wm1_latency/analysis.json

    alpha* = max { alpha : retention(alpha) >= 0.95 * J_nominal }

and, if no alpha qualifies, the alpha with the highest retention.

Two things about this rule matter more than the rule itself.

**It is deployable.**  Retention is measured in the *source* domain -- the one
the policy was trained in and the one a practitioner still has a simulator for
-- against a nominal figure they already know.  Nothing in it reads the hidden
target domain.  The threshold is the deployment requirement (lose no more than
5% of what the robot already does) rather than anything read off a result.

**The screening evaluates the domain the posterior *believes*, not the domain
that is true.**  Those coincide in this phase, because the posterior is
correct, and that is worth saying out loud: a practitioner choosing alpha this
way would be evaluating in whichever domain their inference pointed at, and if
inference were wrong the screening would be too.  The formal evaluation in the
real target domain, afterwards, is the measurement that is not circular.

The rule is committed before the screening finished so that "the alpha we
chose" and "the alpha that looked best" cannot be the same sentence written
twice.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

J_NOMINAL = 55.86
RETENTION_FRACTION = 0.95


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--analysis", default="results/wm1_latency/analysis.json")
  p.add_argument("--plan", default="results/wm1_latency/adapt/screen.json")
  p.add_argument("--json", default="results/wm1_latency/adapt/alpha.json")
  a = p.parse_args()

  analysis = json.loads(Path(a.analysis).read_text())
  plan = json.loads(Path(a.plan).read_text())

  threshold = RETENTION_FRACTION * J_NOMINAL
  rows = []
  for job in plan["jobs"]:
    cfg = re.sub(r"_s\d+$", "", job["tag"])
    ret = tgt = None
    for name, g in analysis.get("groups", {}).items():
      if name.endswith(f"/{cfg}_retention") or name.endswith(f"/pooled:{cfg}_retention"):
        ret = g
      if name.endswith(f"/{cfg}_target") or name.endswith(f"/pooled:{cfg}_target"):
        tgt = g
    if not ret:
      continue
    rows.append({
      "alpha": job["alpha"], "tag": job["tag"], "config": cfg,
      "probs": job["probs"],
      "retention": ret["throughput"]["bootstrap"]["mean"],
      "retention_ci": ret["throughput"]["bootstrap"]["ci"],
      "believed_domain": (tgt["throughput"]["bootstrap"]["mean"] if tgt
                          else None),
      "believed_domain_ci": (tgt["throughput"]["bootstrap"]["ci"] if tgt
                             else None),
      "qualifies": ret["throughput"]["bootstrap"]["mean"] >= threshold,
    })
  rows.sort(key=lambda r: r["alpha"])
  if not rows:
    raise SystemExit("no screening runs found in the analysis")

  ok = [r for r in rows if r["qualifies"]]
  chosen = (max(ok, key=lambda r: r["alpha"]) if ok
            else max(rows, key=lambda r: r["retention"]))
  why = ("largest alpha whose source-domain retention clears "
         f"{threshold:.2f} obj/min" if ok else
         "no alpha cleared the retention threshold; the one that retained most")

  print()
  print(f"  retention threshold: {RETENTION_FRACTION:.2f} x {J_NOMINAL} "
        f"= {threshold:.2f} obj/min")
  print(f"  {'alpha':>6s} {'retention':>10s} {'95% CI':>16s} "
        f"{'believed':>9s}  qualifies")
  for r in rows:
    ci = r["retention_ci"]
    print(f"  {r['alpha']:6.2f} {r['retention']:10.2f} "
          f"[{ci[0]:6.2f},{ci[1]:6.2f}] "
          f"{(r['believed_domain'] or float('nan')):9.2f}  "
          f"{'yes' if r['qualifies'] else 'no'}")
  print()
  print(f"  chosen alpha = {chosen['alpha']:.2f}  ({why})")

  out = {"rule": "max alpha subject to retention >= 0.95 * J_nominal",
         "threshold": threshold, "j_nominal": J_NOMINAL,
         "screening": rows, "chosen": chosen, "why": why}
  Path(a.json).parent.mkdir(parents=True, exist_ok=True)
  Path(a.json).write_text(json.dumps(out, indent=1))
  print(f"  wrote {a.json}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
