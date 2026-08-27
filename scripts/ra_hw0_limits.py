"""Generate configs/ra_hw0_safety_limits.json from its sources.

    python scripts/ra_hw0_limits.py [--audit results/ra_hw0/audit.json]

The table is derived, never typed: the absolute bounds are the intersection of
the vendor's parameter table and the envelope the policy was trained inside,
and every row records which of the two tightened it.  Rows the arm alone can
answer stay ``UNKNOWN`` until an audit with an attached arm supplies them, and
each one blocks motion by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--audit", default="results/ra_hw0/audit.json")
  ap.add_argument("--out", default="configs/ra_hw0_safety_limits.json")
  a = ap.parse_args()
  sys.path.insert(0, str(ROOT))
  sys.path.insert(0, str(ROOT / "src"))
  from hardware.ra_hw0 import limits as L
  from piper_push import robot as sim

  device = None
  p = Path(a.audit)
  if p.exists():
    audit = json.loads(p.read_text())
    if audit.get("device", {}).get("status") == "READ":
      # The parsing of the arm's own limit answer is deliberately left for the
      # run that actually has one: guessing the shape of a message this
      # machine has never received is how a units bug gets into a limit table.
      device = {}
  table = L.write(a.out, sim.SAFE_TARGET_CLIP, device)
  print(f"wrote {a.out}")
  print(f"joints: {len(table['joints'])}   "
        f"unknown blocking motion: {len(table['unknowns_blocking_motion'])}")
  for r in table["joints"]:
    print(f"  {r['joint']}  abs [{r['abs_min_rad']['value']:+.4f}, "
          f"{r['abs_max_rad']['value']:+.4f}]  H1 env "
          f"[{r['envelope_min_rad']['value']:+.4f}, "
          f"{r['envelope_max_rad']['value']:+.4f}]  "
          f"({r['abs_min_rad']['note']})")
  for reason in L.blocks_motion(table):
    print("BLOCKED:", reason)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
