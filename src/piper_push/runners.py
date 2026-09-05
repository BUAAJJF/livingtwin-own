"""rsl_rl runners that stamp the action convention on save and check it on load."""

from __future__ import annotations

import torch
from mjlab.rl import MjlabOnPolicyRunner

from piper_push import action_api


class ActionApiRunnerMixin:
  """Every checkpoint this runner writes says which convention it was trained
  under; every checkpoint it reads is checked against the task it runs."""

  def expected_action_api(self) -> dict:
    return action_api.for_env(self.env)  # type: ignore[attr-defined]

  def save(self, path: str, infos=None) -> None:
    infos = action_api.stamp(infos, self.expected_action_api())
    return super().save(path, infos)  # type: ignore[misc]

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
           map_location: str | None = None) -> dict:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    action_api.check(loaded, self.expected_action_api(), where=path)
    return super().load(path, load_cfg, strict, map_location)  # type: ignore[misc]


class PickPlaceOnPolicyRunner(ActionApiRunnerMixin, MjlabOnPolicyRunner):
  pass
