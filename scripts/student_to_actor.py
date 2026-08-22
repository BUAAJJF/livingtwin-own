"""Rewrite a distillation checkpoint as a policy checkpoint.

The two stages disagree about one word.  Distillation saves the network it
trained under ``student_state_dict``; everything that *runs* a policy -- the
S1 gate, ``play``, the PPO fine-tuning stage -- looks for ``actor_state_dict``.
The weights are the same weights: the distillation student is configured as the
vision PPO actor, deliberately, so that this stage's output is that stage's
initialisation.  Only the key differs.

    python scripts/student_to_actor.py distill/model_1500.pt actor.pt
    python scripts/accept_s1.py Mjlab-Pick-Place-PiperX-Vision actor.pt
"""

from __future__ import annotations

import argparse

from piper_push.checkpoints import as_actor_checkpoint


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("source", help="distillation checkpoint")
  p.add_argument("dest", help="where to write the policy checkpoint")
  a = p.parse_args()

  written = as_actor_checkpoint(a.source, a.dest)
  if written == a.source:
    print(f"{a.source} is already a policy checkpoint; nothing to write")
  else:
    print(f"wrote {written}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
