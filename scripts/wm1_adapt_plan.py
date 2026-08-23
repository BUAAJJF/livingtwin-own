"""Turn posteriors into a list of PPO adaptation runs, with duplicates merged.

    python scripts/wm1_adapt_plan.py --stage screen --out results/wm1_latency/adapt/screen.json
    python scripts/wm1_adapt_plan.py --stage formal --alpha 0.75 \
        --out results/wm1_latency/adapt/formal.json

The adaptation distribution is

    p_adapt(theta) = alpha * q(theta | D_target) + (1 - alpha) * p_source(theta)

with ``p_source = delta(0)``, because the deployed policy trained with no
observation delay at all.  Every run starts from the same checkpoint, gets the
same 600 iterations, the same hyper-parameters and the same environment
protocol; the only thing that differs between them is that distribution.

**Duplicates are merged, not run twice.**  The parameter here is one discrete
number with five values, so two methods that both return a confident answer
return the *same* distribution -- and then ``p_adapt`` is the same, the PPO run
is the same run, and reporting them as two agreeing results would be inventing
agreement out of a rounding tolerance.  Each job therefore lists every method
whose distribution it represents, and the report says which methods shared a
run.  The oracle is included in that merge: it is the special case
``q = delta(target)`` at ``alpha = 1``, so if a method recovers the target
exactly and is used at full weight, its adaptation *is* the oracle and there is
nothing to compare.  That is a real outcome and it is what G5 is there to
notice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from piper_push import latency

SEEDS = (42, 20260824, 31415927)
"""Training seeds.

Phase WM0's single oracle run used seed 42 too, but it is re-run rather than
reused: it went through the old ``--obs-latency-steps`` path and these go
through ``--latency-probs``, which is behaviourally the same domain and a
different consumption of the random stream.  Comparing a run from one code
path against runs from the other, and calling the difference a method effect,
is exactly the kind of thing that is invisible afterwards."""


def load_posteriors(path: Path) -> dict[str, latency.LatencyPrior]:
  raw = json.loads(path.read_text())
  return {k: latency.LatencyPrior(tuple(v["probs"])) for k, v in raw.items()}


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--posteriors",
                 default="results/wm1_latency/posterior/posteriors_60s.json")
  p.add_argument("--stage", choices=("screen", "anchor", "formal"),
                 required=True,
                 help="anchor: the two runs that do not depend on alpha -- the "
                      "oracle and a refit conditioned on the source prior -- "
                      "so they can share a wave with the alpha screening")
  p.add_argument("--alpha", type=float, default=None,
                 help="formal stage: the alpha chosen by screening")
  p.add_argument("--alphas", default="0.5,0.75,1.0",
                 help="screen stage: the alphas to try")
  p.add_argument("--methods", default="DA,B3_state,B4_action,B2_classifier,"
                                      "B1b_img_proprio")
  p.add_argument("--iterations", type=int, default=2100,
                 help="TARGET TOTAL, not additional: 2100 from 1500 is 600 more")
  p.add_argument("--out", required=True)
  a = p.parse_args()

  # The anchor stage is the two runs that do not depend on any posterior --
  # the oracle and a refit conditioned on the source prior -- so it can be
  # planned and launched before inference has finished.
  post = ({} if a.stage == "anchor"
          else load_posteriors(Path(a.posteriors)))
  want = [m for m in a.methods.split(",") if m in post]
  missing = [m for m in a.methods.split(",") if m not in post]
  if missing and a.stage != "anchor":
    print(f"  !! no posterior for {missing}; they are dropped from the plan")

  if a.stage == "screen":
    alphas = [float(x) for x in a.alphas.split(",")]
    seeds = SEEDS[:1]
    # Screening is about alpha, so it runs one method -- the decision-aware
    # one, which is the method the phase is about.
    want = ["DA"] if "DA" in post else want[:1]
  elif a.stage == "anchor":
    alphas, seeds, want = [], SEEDS, []
  else:
    if a.alpha is None:
      raise SystemExit("--alpha is required for the formal stage")
    alphas = [a.alpha]
    seeds = SEEDS

  jobs: dict[tuple, dict] = {}
  for m in want:
    for al in alphas:
      mixed = post[m].mix(latency.P_SOURCE, al)
      key = (mixed.fingerprint(), al)
      job = jobs.setdefault(key, {
        "probs": list(mixed.probs), "alpha": al, "methods": [],
        "prior": mixed.to_json()})
      job["methods"].append(m)

  # The oracle: the same budget, from the same checkpoint, told the answer.
  oracle = latency.LatencyPrior.point(latency.TARGET_LAG)
  key = (oracle.fingerprint(), 1.0)
  job = jobs.setdefault(key, {"probs": list(oracle.probs), "alpha": 1.0,
                              "methods": [], "prior": oracle.to_json()})
  job["methods"].append("ORACLE")

  # p_source itself: adapting to the *wrong* answer with the full budget, so
  # that "PPO for 600 iterations helps a bit whatever you condition on" is
  # measured rather than assumed.
  key = (latency.P_SOURCE.fingerprint(), 1.0)
  job = jobs.setdefault(key, {"probs": list(latency.P_SOURCE.probs),
                              "alpha": 1.0, "methods": [],
                              "prior": latency.P_SOURCE.to_json()})
  job["methods"].append("B0_prior_refit")

  # The tag is a function of the distribution and the seed and nothing else,
  # so a run that the alpha screening already did is not repeated by the formal
  # stage under a different name -- and its evaluations are not repeated
  # either.  The stage name is deliberately absent from it.
  out = []
  for _, job in sorted(jobs.items(), key=lambda kv: str(kv[0])):
    h = hashlib.sha256(
      ",".join(f"{x:.3f}" for x in job["probs"]).encode()).hexdigest()[:4]
    for seed in seeds:
      probs = ",".join(f"{x:.6f}" for x in job["probs"])
      tag = f"q{h}_a{job['alpha']:.2f}_s{seed}"
      out.append({**job, "tag": tag, "seed": seed, "probs_arg": probs,
                  "iterations": a.iterations})

  path = Path(a.out)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(
    {"stage": a.stage, "seeds": list(seeds), "alphas": alphas,
     "n_distinct": len(jobs), "n_runs": len(out), "jobs": out}, indent=1))

  print(f"  {len(want)} methods x {len(alphas)} alphas -> "
        f"{len(jobs)} distinct adaptation distributions, {len(out)} runs")
  for _, job in sorted(jobs.items(), key=lambda kv: str(kv[0])):
    print(f"    alpha {job['alpha']:.2f}  "
          f"{[round(x, 3) for x in job['probs']]}  <- "
          f"{', '.join(job['methods'])}")
  print(f"  wrote {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
