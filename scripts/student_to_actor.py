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
from pathlib import Path

import torch


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("source", help="distillation checkpoint")
  p.add_argument("dest", help="where to write the policy checkpoint")
  a = p.parse_args()

  loaded = torch.load(a.source, map_location="cpu", weights_only=False)
  if "student_state_dict" not in loaded:
    raise SystemExit(
      f"{a.source} has no student_state_dict; keys are {sorted(loaded)}. "
      "This converts distillation checkpoints, not policy checkpoints."
    )
  out = {
    "actor_state_dict": loaded["student_state_dict"],
    "iter": loaded.get("iter", 0),
    "infos": loaded.get("infos"),
  }
  Path(a.dest).parent.mkdir(parents=True, exist_ok=True)
  torch.save(out, a.dest)
  print(f"wrote {a.dest} from iteration {out['iter']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
