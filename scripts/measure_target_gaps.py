"""How the rig actually loses its target, in time rather than on average.

``blind_when_held`` models the carrying phase as an 8% per-frame coin flip.
The marginal is right -- 155 of 1894 frames with the jaws closed -- and the
process is not: an independent draw re-exposes the object every ~12 frames,
while the rig can lose it for seconds at a stretch.  A GRU sees those as
completely different signals, and only one of them is what deployment does.

So this reads the recorded control logs, which store the tracker's chosen
label every control step, and measures the thing the marginal throws away:

  * how long the gaps are, as a distribution, split by jaw state
  * how often the label CHANGES between two nonzero values -- an identity
    swap, the failure where the policy is handed a confident mask of the
    wrong object
  * how much of that happens while the jaws are closed, when by construction
    the target cannot have changed

The swap rate is the number to design against.  It is currently unmeasured,
and both the deployment's ``lost_frames = 15`` and the simulator's dropout
model are set without it.

    python scripts/measure_target_gaps.py recordings/v4_*/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

# ``run.py``'s own default: below this the jaws are closed on something.
GRIPPER_CLOSED_M = 0.045
# ``proprio.CONTACT_EFFORT``: gripper load above which the drive is pushing
# against something.  Measured, not assumed -- see hardware/deploy/proprio.py.
CONTACT_EFFORT = 0.20


def runs_of(mask: np.ndarray) -> np.ndarray:
  """Lengths of the consecutive True runs in ``mask``."""
  if not mask.any():
    return np.zeros(0, dtype=int)
  d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
  return np.flatnonzero(d < 0) - np.flatnonzero(d > 0)


def load(session: pathlib.Path):
  f = session / "control.json"
  if not f.exists():
    return None
  rec = json.loads(f.read_text())
  rows = [r for r in rec if "label" in r and "joint_pos" in r]
  if len(rows) < 50:
    return None
  label = np.array([int(r.get("label") or 0) for r in rows])
  jp = [r["joint_pos"] for r in rows]
  gap_m = np.array([float(p[6]) if len(p) > 6 else np.nan for p in jp])
  effort = np.array([abs(float(r.get("gripper_effort") or 0.0)) for r in rows])
  t = np.array([float(r.get("t") or 0.0) for r in rows])
  return label, gap_m, effort, t


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("sessions", nargs="+")
  p.add_argument("--replay-lifecycle", action="store_true",
                 help="run the recorded labels back through "
                      "hardware.deploy.lifecycle and report how many identity "
                      "swaps it would have refused.  The bar is zero swaps "
                      "while the jaws are closed.")
  p.add_argument("--out", default=None)
  a = p.parse_args()

  agg = {"open": [], "closed": []}
  totals = dict(frames=0, closed=0, closed_seen=0, open_seen=0,
                swaps=0, swaps_held=0, held_spans=0)
  per_session = []

  for s in a.sessions:
    d = pathlib.Path(s)
    got = load(d)
    if got is None:
      continue
    label, gap_m, effort, t = got
    n = len(label)
    closed = np.isfinite(gap_m) & (gap_m < GRIPPER_CLOSED_M)
    seen = label > 0

    # An identity swap: two different nonzero labels back to back, with no
    # "no target" frame between them.  A gap in between is a re-acquisition,
    # which is a different (and expected) event.
    prev, cur = label[:-1], label[1:]
    swap = (prev > 0) & (cur > 0) & (prev != cur)
    swap_held = swap & closed[1:]

    for key, sel in (("open", ~closed), ("closed", closed)):
      lost = sel & ~seen
      agg[key].append(runs_of(lost))

    totals["frames"] += n
    totals["closed"] += int(closed.sum())
    totals["closed_seen"] += int((closed & seen).sum())
    totals["open_seen"] += int((~closed & seen).sum())
    totals["swaps"] += int(swap.sum())
    totals["swaps_held"] += int(swap_held.sum())
    totals["held_spans"] += int(len(runs_of(closed)))

    dt = float(np.median(np.diff(t))) if n > 2 else 0.02
    per_session.append({
      "session": d.name, "frames": n, "dt_s": round(dt, 4),
      "closed_frames": int(closed.sum()),
      "detect_rate_closed": round(float((closed & seen).sum() / max(closed.sum(), 1)), 4),
      "detect_rate_open": round(float(((~closed) & seen).sum() / max((~closed).sum(), 1)), 4),
      "swaps": int(swap.sum()), "swaps_while_held": int(swap_held.sum()),
    })

  if not per_session:
    print("no control.json with labels found", file=sys.stderr)
    return 2

  print(f"{'session':<34}{'frames':>8}{'closed':>8}{'seen|closed':>13}"
        f"{'seen|open':>11}{'swaps':>7}{'held':>6}")
  for r in per_session:
    print(f"{r['session']:<34}{r['frames']:>8}{r['closed_frames']:>8}"
          f"{100*r['detect_rate_closed']:>12.1f}%{100*r['detect_rate_open']:>10.1f}%"
          f"{r['swaps']:>7}{r['swaps_while_held']:>6}")

  if a.replay_lifecycle:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from hardware.deploy.lifecycle import TargetLifecycle

    print()
    print("replayed through TargetLifecycle -- swaps it would have refused")
    print(f"{'session':<34}{'swaps held (raw)':>18}{'with lifecycle':>16}"
          f"{'refused':>9}")
    raw_tot = new_tot = 0
    for s_dir in a.sessions:
      got = load(pathlib.Path(s_dir))
      if got is None:
        continue
      label, gap_m, effort, _ = got
      closed = np.isfinite(gap_m) & (gap_m < GRIPPER_CLOSED_M)
      # A plain threshold stands in for ``proprio._contact_bit``; the real
      # latch adds hysteresis, which can only make the carry MORE stable, so
      # this is the pessimistic version of the check.
      loaded = effort > CONTACT_EFFORT
      prev = label[:-1]
      raw = int((( prev > 0) & (label[1:] > 0) & (prev != label[1:])
                 & closed[1:]).sum())
      lc = TargetLifecycle()
      shown = np.array([lc.update(int(l), bool(c), bool(ld))
                        for l, c, ld in zip(label, closed, loaded)])
      sp, sc = shown[:-1], shown[1:]
      after = int(((sp > 0) & (sc > 0) & (sp != sc) & closed[1:]).sum())
      raw_tot += raw
      new_tot += after
      print(f"{pathlib.Path(s_dir).name:<34}{raw:>18}{after:>16}"
            f"{lc.refused:>9}")
    print(f"{'TOTAL':<34}{raw_tot:>18}{new_tot:>16}")
    if new_tot == 0:
      print("  bystander swap rate while held: 0 -- the bar is met")
    else:
      print(f"  {new_tot} swaps survived; the bar is zero")

  out = {"per_session": per_session, "gap_runs": {}}
  print()
  print("gap lengths -- consecutive control steps with NO target")
  for key in ("open", "closed"):
    r = np.concatenate(agg[key]) if agg[key] else np.zeros(0, int)
    if not r.size:
      print(f"  {key:<8} none")
      continue
    q = {f"p{p}": int(np.percentile(r, p)) for p in (50, 75, 90, 95, 99)}
    out["gap_runs"][key] = {"n": int(r.size), "mean": float(r.mean()),
                            "max": int(r.max()), **q}
    print(f"  {key:<8} n={r.size:<5} mean {r.mean():6.1f}  "
          f"p50 {q['p50']:>4}  p90 {q['p90']:>4}  p99 {q['p99']:>5}  "
          f"max {r.max():>5}   ({0.02*r.max():.1f} s at 50 Hz)")

  cs, cl = totals["closed_seen"], totals["closed"]
  print()
  print(f"aggregate: target present in {100*cs/max(cl,1):.1f}% of jaws-closed "
        f"steps ({cs}/{cl})")
  print(f"           identity swaps: {totals['swaps']} total, "
        f"{totals['swaps_held']} while the jaws were closed")
  print(f"           across {totals['held_spans']} closed-jaw spans")
  out["totals"] = totals
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
