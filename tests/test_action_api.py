"""The action convention is stamped on every checkpoint and checked by every loader."""

from __future__ import annotations

import os
from types import SimpleNamespace

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest
import torch

from piper_push import action_api, evalcfg
from piper_push import robot as piper

HOME = {j: piper.PICK_HOME_KEYFRAME.joint_pos.get(j, 0.0) for j in piper.ARM_JOINT_ORDER}


def _fake_env(bounded: bool):
  names = list(piper.ARM_JOINT_ORDER) + ["gripper_joint1", "gripper_joint2"]
  defaults = torch.tensor([[HOME[j] for j in piper.ARM_JOINT_ORDER] + [0.05, -0.05]])
  robot = SimpleNamespace(joint_names=names, data=SimpleNamespace(default_joint_pos=defaults))
  cfg = SimpleNamespace(actions={"arm": SimpleNamespace(bounded=bounded)})
  return SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg, scene={"robot": robot}))


@pytest.fixture(autouse=True)
def _no_legacy_env(monkeypatch):
  monkeypatch.delenv(action_api.ENV_FLAG, raising=False)


def test_versions_and_hashes_are_distinct_and_stable():
  b = action_api.for_convention("bounded")
  v = action_api.for_convention("v1", HOME)
  assert (b["version"], v["version"]) == (2, 1)
  assert b["spec_hash"] != v["spec_hash"]
  assert action_api.for_convention("bounded") == b
  assert action_api.for_env(_fake_env(True)) == b
  assert action_api.for_env(_fake_env(False)) == v


def test_unstamped_checkpoint_is_refused_unless_legacy_is_explicit(monkeypatch):
  loaded = {"actor_state_dict": {}, "iter": 3, "infos": None}
  with pytest.raises(action_api.ActionApiError, match="no action_api metadata"):
    action_api.check(loaded, action_api.for_convention("bounded"), where="x.pt")
  # even on a v1 task it needs the flag
  with pytest.raises(action_api.ActionApiError):
    action_api.check(loaded, action_api.for_convention("v1", HOME), where="x.pt")
  monkeypatch.setenv(action_api.ENV_FLAG, "1")
  assert action_api.check(loaded, action_api.for_convention("v1", HOME))["status"] == "legacy-unstamped"
  # the flag never makes an unstamped checkpoint acceptable on a bounded task
  with pytest.raises(action_api.ActionApiError):
    action_api.check(loaded, action_api.for_convention("bounded"))


def test_mismatched_stamp_is_refused_and_matching_one_passes():
  b = action_api.for_convention("bounded")
  v = action_api.for_convention("v1", HOME)
  stamped_v1 = {"actor_state_dict": {}, "infos": action_api.stamp(None, v)}
  with pytest.raises(action_api.ActionApiError, match="different joint"):
    action_api.check(stamped_v1, b)
  assert action_api.check(stamped_v1, v)["status"] == "ok"
  assert action_api.check({"infos": action_api.stamp({"k": 1}, b)}, b)["status"] == "ok"


def test_runner_mixin_stamps_on_save_and_checks_on_load(tmp_path):
  from piper_push.runners import ActionApiRunnerMixin

  class Base:
    def __init__(self):
      self.saved = None
    def save(self, path, infos=None):
      self.saved = infos
      torch.save({"actor_state_dict": {}, "iter": 0, "infos": infos}, path)
    def load(self, path, load_cfg=None, strict=True, map_location=None):
      return "loaded"

  class Runner(ActionApiRunnerMixin, Base):
    def __init__(self, env):
      super().__init__()
      self.env = env

  r = Runner(_fake_env(True))
  ck = tmp_path / "m.pt"
  r.save(str(ck), infos={"env_state": 1})
  assert r.saved["action_api"] == action_api.for_convention("bounded") and r.saved["env_state"] == 1
  assert r.load(str(ck)) == "loaded"
  with pytest.raises(action_api.ActionApiError):
    Runner(_fake_env(False)).load(str(ck))       # bounded checkpoint into a v1 task
  torch.save({"actor_state_dict": {}, "iter": 0, "infos": None}, ck)
  with pytest.raises(action_api.ActionApiError):
    r.load(str(ck))                              # unstamped into a bounded task


def test_evalcfg_loader_checks_the_stamp_when_the_runner_has_an_env(tmp_path, monkeypatch):
  net = torch.nn.Linear(2, 2)
  alg = SimpleNamespace(_raw_actor=net)
  runner = SimpleNamespace(alg=alg, env=_fake_env(True))
  ck = tmp_path / "a.pt"
  torch.save({"actor_state_dict": torch.nn.Linear(2, 2).state_dict(), "infos": None}, ck)
  with pytest.raises(action_api.ActionApiError):
    evalcfg.load_weights(runner, str(ck))
  torch.save({"actor_state_dict": torch.nn.Linear(2, 2).state_dict(),
              "infos": action_api.stamp(None, action_api.for_convention("bounded"))}, ck)
  assert evalcfg.load_weights(runner, str(ck))["action_api"]["status"] == "ok"
  # legacy: only with the flag, only on a v1 runner
  torch.save({"actor_state_dict": torch.nn.Linear(2, 2).state_dict(), "infos": None}, ck)
  runner_v1 = SimpleNamespace(alg=alg, env=_fake_env(False))
  with pytest.raises(action_api.ActionApiError):
    evalcfg.load_weights(runner_v1, str(ck))
  monkeypatch.setenv(action_api.ENV_FLAG, "0")   # so the fixture restores it afterwards
  a = SimpleNamespace(allow_legacy_action_api=True)
  evalcfg.apply_action_api_arg(a)
  assert os.environ[action_api.ENV_FLAG] == "1"
  assert evalcfg.load_weights(runner_v1, str(ck))["action_api"]["status"] == "legacy-unstamped"


def test_registered_runners_stamp():
  from mjlab.tasks.registry import load_runner_cls
  from piper_push.runners import ActionApiRunnerMixin
  for tid in ("Mjlab-Pick-Place-PiperX-Robust", "Mjlab-Pick-Place-PiperX-Vision-Robust",
              "Mjlab-Pick-Place-PiperX-Distill-Robust", "Mjlab-Pick-Place-PiperX-Robust-V1"):
    assert issubclass(load_runner_cls(tid), ActionApiRunnerMixin), tid
