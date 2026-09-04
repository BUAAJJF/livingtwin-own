"""The three things an evaluation has to get right, and once got wrong.

``piper_push.evalcfg`` exists because ``runner.load(load_cfg={"actor": True})``
loaded nothing on a distillation runner and nobody noticed for two days, and
because ``--sensor measured`` quietly downgraded a robust task's sensor.
"""

from __future__ import annotations

import re
from pathlib import Path

import mjlab.tasks  # noqa: F401  -- import mjlab before us: its entry-point
# loader imports piper_push.tasks, and if piper_push is already half-imported
# the task registration fails partway with a [WARN] and a registry that holds
# the state task but not the vision ones.
import pytest
import torch

from piper_push import evalcfg

ROOT = Path(__file__).resolve().parent.parent


# --- the knob list is the code's knob list -----------------------------------


def test_every_import_time_environ_read_is_recorded():
  seen = set()
  for f in (ROOT / "src/piper_push").rglob("*.py"):
    seen |= set(re.findall(r'os\.environ\.get\("([A-Z_0-9]+)"', f.read_text()))
  assert seen, "the scan found nothing; the pattern is wrong"
  assert seen <= set(evalcfg.ENV_KNOBS), sorted(seen - set(evalcfg.ENV_KNOBS))


def test_env_knobs_reads_the_given_environment_not_the_process():
  k = evalcfg.env_knobs({"TARGET_GAP_SCALE": "0.3"})
  assert k["TARGET_GAP_SCALE"] == "0.3"
  assert k["TARGET_VISIBLE_FLOOR"] is None
  assert evalcfg.env_knobs_prefix(k) == "TARGET_GAP_SCALE=0.3"
  assert evalcfg.env_knobs_prefix(evalcfg.env_knobs({})) == ""


# --- the weights arrive, or it is an error -----------------------------------


class _Net(torch.nn.Module):
  def __init__(self, seed):
    super().__init__()
    torch.manual_seed(seed)
    self.l = torch.nn.Linear(4, 3)
    self.gru = torch.nn.GRU(3, 5)


class _Alg:
  def __init__(self, attr, seed):
    setattr(self, attr, _Net(seed))


class _Runner:
  def __init__(self, attr, seed=0):
    self.alg = _Alg(attr, seed)

  def get_inference_policy(self, device=None):
    return evalcfg.evaluated_network(self)


def _save(tmp_path, key, seed=1):
  path = tmp_path / f"{key}.pt"
  torch.save({key: _Net(seed).state_dict(), "iter": 7, "infos": None}, path)
  return str(path)


@pytest.mark.parametrize("runner_attr", ["_raw_student", "_raw_actor"])
@pytest.mark.parametrize("ckpt_key", ["actor_state_dict", "student_state_dict"])
def test_weights_reach_whichever_network_the_runner_evaluates(
    tmp_path, runner_attr, ckpt_key):
  """An actor-only checkpoint into a distillation runner is the case that
  used to load nothing; every combination now lands, and says which key."""
  r = _Runner(runner_attr, seed=0)
  before = {k: v.clone() for k, v in evalcfg.evaluated_network(r).state_dict().items()}
  info = evalcfg.load_weights(r, _save(tmp_path, ckpt_key, seed=1))
  after = evalcfg.evaluated_network(r).state_dict()
  want = _Net(1).state_dict()
  assert info["key"] == ckpt_key and info["iter"] == 7
  assert info["n_tensors"] == len(want)
  assert all(torch.equal(after[k], want[k]) for k in want)
  assert any(not torch.equal(before[k], after[k]) for k in want), \
    "the test networks started equal; it would prove nothing"


def test_a_runner_with_nothing_to_load_into_is_an_error(tmp_path):
  class Empty:
    alg = object()
  with pytest.raises(TypeError, match="neither a student nor an actor"):
    evalcfg.load_weights(Empty(), _save(tmp_path, "actor_state_dict"))


def test_a_checkpoint_without_policy_weights_is_an_error(tmp_path):
  path = tmp_path / "critic_only.pt"
  torch.save({"critic_state_dict": {}, "iter": 0}, path)
  with pytest.raises(ValueError, match="holds none of"):
    evalcfg.load_weights(_Runner("_raw_actor"), str(path))


def test_a_shape_mismatch_is_an_error_not_a_partial_load(tmp_path):
  path = tmp_path / "wrong.pt"
  sd = _Net(1).state_dict()
  sd["l.weight"] = torch.zeros(3, 5)
  torch.save({"actor_state_dict": sd}, path)
  with pytest.raises(RuntimeError):
    evalcfg.load_weights(_Runner("_raw_actor"), str(path))


# --- the sensor means what it says -------------------------------------------

VISION = "Mjlab-Pick-Place-PiperX-Vision"
ROBUST = "Mjlab-Pick-Place-PiperX-Vision-Robust"
STATE = "Mjlab-Pick-Place-PiperX"


def _cfgs(task):
  from mjlab.tasks.registry import load_env_cfg
  return load_env_cfg(task, play=True), load_env_cfg(task, play=False)


def _noise(cfg):
  return cfg.observations["camera"].terms["scene"].params["noise_cfg"]


@pytest.mark.parametrize("task", [VISION, ROBUST])
def test_clean_is_clean_on_every_vision_task(task):
  play, _ = _cfgs(task)
  prov = evalcfg.apply_sensor(play, task, "clean")
  assert _noise(play).strength == 0.0
  assert play.observations["camera"].terms["scene"].params["mask_jitter_px"] == 0
  assert prov["camera"]["strength"] == 0.0


@pytest.mark.parametrize("task", [VISION, ROBUST])
def test_measured_is_the_sensor_the_task_trains_with(task):
  play, train = _cfgs(task)
  evalcfg.apply_sensor(play, task, "measured")
  got, want = _noise(play), _noise(train)
  assert got.strength == want.strength
  assert got.surface_fill == want.surface_fill
  assert got.texture_penalty == want.texture_penalty
  assert (play.observations["camera"].terms["scene"].params["mask_jitter_px"]
          == train.observations["camera"].terms["scene"].params["mask_jitter_px"])


def test_measured_on_the_robust_task_is_not_the_nominal_sensor():
  """The bug: ``replace(camera.DEPTH_NOISE, strength=1.0)`` on a -Robust
  task.  The robust profile is stronger than nominal and fills surfaces
  differently; 'measured' must keep it."""
  from piper_push import camera
  play, _ = _cfgs(ROBUST)
  evalcfg.apply_sensor(play, ROBUST, "measured")
  n = _noise(play)
  assert n.strength > 1.0
  assert (n.strength, n.surface_fill, n.texture_penalty) != (
    1.0, camera.DEPTH_NOISE.surface_fill, camera.DEPTH_NOISE.texture_penalty)


def test_task_setting_leaves_the_play_config_alone():
  play, _ = _cfgs(ROBUST)
  before = _noise(play)
  evalcfg.apply_sensor(play, ROBUST, "task")
  assert _noise(play) is before


def test_the_state_task_has_no_sensor_to_set():
  play, _ = _cfgs(STATE)
  prov = evalcfg.apply_sensor(play, STATE, "measured")
  assert prov["camera"] is None


def test_an_unknown_setting_is_rejected():
  play, _ = _cfgs(VISION)
  with pytest.raises(ValueError):
    evalcfg.apply_sensor(play, VISION, "real")


def test_provenance_carries_the_sensor_block_and_the_knobs():
  play, _ = _cfgs(ROBUST)
  sp = evalcfg.apply_sensor(play, ROBUST, "measured")
  prov = evalcfg.provenance(sensor=sp, argv=["x"])
  assert prov["sensor"] is sp and prov["sensor"]["camera"]["strength"] > 1.0
  assert set(prov["env_knobs"]) == set(evalcfg.ENV_KNOBS)
  assert prov["argv"] == ["x"]
