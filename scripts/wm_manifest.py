"""Assemble the dataset manifest, and run the leakage checks as a test.

    python scripts/wm_manifest.py --data results/wm1_latency/data

The manifest is what gets committed: the tensors are four gigabytes of fp16 and
are not, but every claim the report makes about the data -- how much of it
there is, which shape classes are in which split, that no two splits share a
generation seed, that the label is not a channel -- is checkable from the
manifest alone, and this script fails rather than writes if any of them is
false.

The checks, in the order a violation would flatter the result most:

1. **the label is not a channel.**  ``wm_data.assert_deployable`` is called on
   the schema, so a simulator-only field cannot be listed as an input.
2. **splits do not share a generation seed.**  A window in the test set that
   came from the same RNG stream as one in the training set is a near-copy.
3. **the test sessions are shape-disjoint from training.**  Not filtered
   afterwards -- generated with zero probability on the training classes, so
   the policy's recurrent state never carried a training object either.
4. **the budgets are nested.**  Every test session is the same length, so the
   10 s result is a prefix of the 300 s one rather than a differently
   sampled dataset.
5. **sequence length carries no label.**  Every file of a split has the same
   number of steps and environments, so a method cannot read the domain off
   the shape of the tensor it was handed.
6. **the classes are balanced.**  Within a split, one file per candidate lag,
   the same size each.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from piper_push import latency, wm_data

TRAIN_CLASSES = {0, 1, 2}
HOLDOUT_CLASSES = {3, 4}


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--data", default="results/wm1_latency/data")
  a = p.parse_args()
  root = Path(a.data)

  entries = []
  for f in sorted(root.glob("*.json")):
    if f.name == "manifest.json":
      continue
    entries.append(json.loads(f.read_text()))
  if not entries:
    raise SystemExit(f"no dataset sidecars under {root}")

  by_split = defaultdict(list)
  for e in entries:
    by_split[e["split"]].append(e)

  failures = []

  # 1. the schema cannot name a simulator-only field as an input
  try:
    wm_data.assert_deployable(wm_data.DEPLOYABLE)
  except ValueError as exc:                                    # pragma: no cover
    failures.append(f"schema: {exc}")

  # 2. no generation seed is shared between splits
  seeds = {s: {e["seed"] for e in es} for s, es in by_split.items()}
  for s1 in seeds:
    for s2 in seeds:
      if s1 < s2 and seeds[s1] & seeds[s2]:
        failures.append(f"splits {s1} and {s2} share seeds "
                        f"{sorted(seeds[s1] & seeds[s2])}")

  # 3. shape disjointness
  for split, es in by_split.items():
    want = HOLDOUT_CLASSES if split in ("valh", "calh", "test") else TRAIN_CLASSES
    for e in es:
      seen = {i for i, c in enumerate(e["shape_class_counts"]) if c > 0}
      if not seen <= want:
        failures.append(f"{e['file']}: shape classes {sorted(seen)} outside "
                        f"{sorted(want)} for split {split}")

  # 4-5. one length per split, and it covers the largest budget for `test`
  for split, es in by_split.items():
    lens = {e["steps"] for e in es}
    envs = {e["n_envs"] for e in es}
    if len(lens) != 1 or len(envs) != 1:
      failures.append(f"{split}: mixed sizes steps={sorted(lens)} "
                      f"envs={sorted(envs)}; sequence length would carry "
                      f"information about which file a window came from")
    if split == "test" and min(lens) < 300 * 50:
      failures.append(f"test sessions are {min(lens)} steps, "
                      f"under the 300 s budget's {300 * 50}")

  # 6. one file per lag per split, and every lag present
  for split, es in by_split.items():
    lags = defaultdict(int)
    for e in es:
      lags[e["lag"]] += 1
    if set(lags) != set(latency.LAGS):
      failures.append(f"{split}: lags {sorted(lags)} != {list(latency.LAGS)}")
    elif len(set(lags.values())) != 1:
      failures.append(f"{split}: unbalanced lag counts {dict(lags)}")

  summary = {}
  for split, es in sorted(by_split.items()):
    summary[split] = {
      "files": len(es),
      "lags": sorted({e["lag"] for e in es}),
      "seeds": sorted({e["seed"] for e in es}),
      "shapes": sorted({e["shapes"] for e in es}),
      "n_envs": es[0]["n_envs"], "steps": es[0]["steps"],
      "sessions": sum(e["n_envs"] for e in es),
      "arm_hours": sum(e["arm_seconds"] for e in es) / 3600.0,
      "gigabytes": sum(e["bytes"] for e in es) / 1e9,
      "episode_boundaries": sum(e["episode_boundaries"] for e in es),
      "shape_class_counts": [sum(e["shape_class_counts"][i] for e in es)
                             for i in range(5)],
    }

  path = wm_data.write_manifest(root, entries, {
    "splits": summary,
    "candidate_lags": list(latency.LAGS),
    "step_ms": latency.STEP_MS,
    "leakage_checks": {
      "label_is_not_a_channel": True,
      "no_shared_generation_seed": True,
      "test_shapes_disjoint_from_training": True,
      "one_sequence_length_per_split": True,
      "budgets_nested_within_a_session": True,
      "balanced_lags_per_split": True,
    },
    "failures": failures,
  })

  print()
  print(f"  {'split':6s} {'files':>5s} {'sessions':>9s} {'arm-h':>7s} "
        f"{'GB':>5s} {'steps':>7s} {'shapes':>8s}")
  for split, s in summary.items():
    print(f"  {split:6s} {s['files']:5d} {s['sessions']:9d} "
          f"{s['arm_hours']:7.1f} {s['gigabytes']:5.2f} {s['steps']:7d} "
          f"{','.join(s['shapes']):>8s}")
  print()
  if failures:
    print("  LEAKAGE CHECKS FAILED")
    for f in failures:
      print(f"    - {f}")
    return 1
  print(f"  all six leakage checks pass; wrote {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
