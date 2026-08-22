"""Moving weights between the stages that disagree about what to call them.

Deliberately free of mjlab imports.  Importing anything from mjlab runs its
entry-point loader, which imports ``piper_push.tasks``, which imports the task
package, which imports the distillation runner -- so a small utility that lives
next to that runner cannot be imported first without tripping over a
half-initialised module.  This one can be imported from anywhere.
"""

from __future__ import annotations

from pathlib import Path

import torch


def as_actor_checkpoint(source: str | Path, dest: str | Path) -> str:
  """Rewrite a distillation checkpoint so a policy loader will read it.

  The distillation student and the vision PPO actor are the same network by
  construction -- same convolutions, same recurrent layer, same head, same
  observation tuple -- so that this stage's output is the next stage's
  initialisation.  They disagree on one key name, and everything that *runs* a
  policy looks for the other one.

  Returns the path a loader should be given: ``source`` unchanged if it is
  already a policy checkpoint, otherwise ``dest``.
  """
  loaded = torch.load(source, map_location="cpu", weights_only=False)
  if "student_state_dict" not in loaded:
    if "actor_state_dict" not in loaded:
      raise ValueError(
        f"{source} holds neither a student nor an actor; keys are "
        f"{sorted(loaded)}."
      )
    return str(source)
  Path(dest).parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "actor_state_dict": loaded["student_state_dict"],
      "iter": loaded.get("iter", 0),
      "infos": loaded.get("infos"),
    },
    dest,
  )
  return str(dest)
