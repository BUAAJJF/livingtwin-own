"""The first thing to run on a real arm: one joint, a few degrees, and a number.

Everything else in this directory has been checked against the simulator.  The
one thing that cannot be is the CAN boundary in ``robot.PiperArm`` -- the SDK
takes joint angles in thousandths of a degree and gripper travel in
micrometres, this code speaks radians and metres, and a missed conversion is a
factor of 57000.  A conversion error that large does not produce a slightly
wrong trajectory; it produces the arm attempting to reach a target ten thousand
radians away, at which point the only thing between it and the table is the
slew limiter.

So: move one joint, by a few degrees, and print what came back.

    python -m hardware.deploy.jointcheck --joint 1

What it checks, in order, and it stops at the first thing that looks wrong:

  scale      commanded 5 degrees, measured 5 degrees -- not 0.087, not 286
  direction  positive command, positive measurement
  tracking   the joint actually got there, within a degree

Every motion goes through ``ActionMapper``'s slew limit, so the arm can only
travel as fast as the trained command path allows it to, and the amplitude is
smaller than the safety envelope on every joint.  It is still a robot that
moves: clear the workspace and keep the estop in reach.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

from . import config, robot
from .proprio import SPEC_FILE

DEFAULT_DEG = 5.0
SETTLE_S = 1.5


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--joint", type=int, default=1, choices=range(1, 7),
                 help="which arm joint, 1-6")
  p.add_argument("--degrees", type=float, default=DEFAULT_DEG,
                 help="amplitude, both directions from where it is now")
  p.add_argument("--can", default=config.CAN_INTERFACE)
  p.add_argument("--dry-run", action="store_true",
                 help="run the whole thing against the stand-in arm, which "
                      "answers 'does this script work' and nothing about the "
                      "robot")
  p.add_argument("--yes", action="store_true",
                 help="skip the confirmation prompt")
  a = p.parse_args()

  if abs(a.degrees) > 15.0:
    raise SystemExit(
      f"{a.degrees} degrees is more than this is for.  It exists to check a "
      "unit conversion, and a conversion that is wrong is wrong at 5 degrees "
      "too."
    )

  spec = json.loads(pathlib.Path(SPEC_FILE).read_text())
  arm = robot.DryRunArm(spec) if a.dry_run else robot.PiperArm(a.can)
  # No policy is involved here; the mapper is used for the unit conversion only.
  mapper = robot.ActionMapper(spec, allow_legacy=True)

  if not a.dry_run and not a.yes:
    print(f"About to move joint {a.joint} by +-{a.degrees} degrees on "
          f"{a.can}.  Workspace clear?  estop in reach?")
    if input("type 'go' to continue: ").strip() != "go":
      return 1

  arm.connect()
  arm.enable()
  start = arm.read()
  mapper.reset(np.concatenate([start.q, [start.gripper]]))
  print(f"joint {a.joint} starts at {math.degrees(start.q[a.joint - 1]):+.2f} deg")

  # The action that moves this joint and nothing else.  Going through the
  # mapper rather than commanding a target directly is the point: it is the
  # same arithmetic the policy's commands go through, so if the mapper is
  # wrong this finds that too.
  idx = a.joint - 1
  amp = math.radians(a.degrees)
  worst = 0.0
  ok = True
  for sign in (+1.0, -1.0, 0.0):
    want = start.q[idx] + sign * amp
    action = np.zeros(7)
    action[idx] = (want - mapper.offset[idx]) / mapper.scale[idx]
    action[6] = (start.gripper - mapper.offset[6]) / mapper.scale[6]

    deadline = time.time() + SETTLE_S
    dt = 1.0 / config.CONTROL_HZ
    while time.time() < deadline:
      arm.command(mapper(action), dt)
      time.sleep(dt)

    got = arm.read()
    moved = math.degrees(got.q[idx] - start.q[idx])
    asked = math.degrees(want - start.q[idx])
    err = abs(moved - asked)
    worst = max(worst, err)
    verdict = "ok" if err < 1.0 else "WRONG"
    if err >= 1.0:
      ok = False
    print(f"  asked {asked:+6.2f} deg   measured {moved:+6.2f} deg   "
          f"error {err:5.2f} deg   {verdict}")
    if err > 5.0:
      print("\nSTOPPING.  That is not a tracking error, it is a units or a "
            "sign error.  Check RAD_TO_MDEG and the joint order in "
            "robot.PiperArm before moving anything else.")
      break

  arm.close()
  print()
  if a.dry_run:
    print("dry run: this says the script works, not that the arm does")
  print(f"worst error {worst:.2f} deg -- {'PASS' if ok else 'FAIL'}")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
