"""Static guards on the two rules Phase RA-Sim-0 could break without noticing.

Both are about what a file *reads*, which no numerical test can see: a residual
trained on the target's own effective command would look excellent and mean
nothing, and a surrogate fitted on target-domain transitions would smuggle the
mismatch in through the back door.  Greps are crude, and they are the only
thing that catches this class of error before it is a published number.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Names that exist only inside the hidden target or downstream of it.
FORBIDDEN = (
  "_prev_hooked",   # the effective command the servo actually received
  "_flank",         # the backlash state
  "hook.state",     # the accessor that returns both
  "reward_buf",
  "over_speed",
)

PIPELINE = (
  "scripts/ra_sim0_collect.py",
  "scripts/ra_sim0_train.py",
  "src/piper_push/residual.py",
  "src/piper_push/surrogate.py",
)


@pytest.mark.parametrize("rel", PIPELINE)
def test_no_stage_of_the_pipeline_reads_a_hidden_quantity(rel):
  text = (ROOT / rel).read_text()
  # `_lag` is checked separately: the word appears in prose about latency.
  for name in FORBIDDEN:
    assert name not in text, f"{rel} mentions {name}"


def test_the_collector_records_the_issued_command_not_the_delivered_one():
  """`_previous_target` is what the controller sent; `_prev_hooked` is what
  the plant produced from it.  A robot knows the first and not the second."""
  text = (ROOT / "scripts/ra_sim0_collect.py").read_text()
  assert "arm._previous_target" in text
  assert "_prev_hooked" not in text


def test_the_recording_carries_no_reward_success_or_safety_label():
  """Whatever else changes, these must never become channels."""
  text = (ROOT / "scripts/ra_sim0_collect.py").read_text()
  tree = ast.parse(text)
  keys = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
      keys.add(node.value)
  for banned in ("reward", "success", "trip", "rew", "return"):
    assert banned not in keys, f"a channel called {banned!r} appeared"


def test_the_surrogate_is_only_ever_fitted_on_nominal_data():
  """The bridge that carries the gradient must not have seen the target."""
  text = (ROOT / "scripts/ra_sim0_train.py").read_text()
  # the fit call takes the dataset built from the *nominal* recording
  assert "surrogate_dataset(nrec" in text
  assert "rp.Recording.load(a.nominal)" in text
  # and nothing builds a surrogate dataset from a target recording
  assert "surrogate_dataset(trecs" not in text
  assert "surrogate_dataset(vrec" not in text


def test_the_residual_feature_builder_has_exactly_four_inputs_everywhere():
  """One builder, four tensors, used identically offline and in-simulator."""
  for rel in ("scripts/ra_sim0_train.py", "src/piper_push/residual.py"):
    text = (ROOT / rel).read_text()
    for call in _calls_named(text, "build_features"):
      assert len(call.args) == 4, f"{rel}: build_features got {len(call.args)}"
      assert not call.keywords


def _calls_named(text: str, name: str):
  for node in ast.walk(ast.parse(text)):
    if isinstance(node, ast.Call):
      f = node.func
      if (isinstance(f, ast.Name) and f.id == name) or (
              isinstance(f, ast.Attribute) and f.attr == name):
        yield node


def test_every_new_feature_is_off_at_its_default():
  """`command_hooks` empty, residual absent, hidden target absent.  A phase
  whose features default on cannot claim its baseline is the old one."""
  from piper_push.actions import RateLimitedJointPositionActionCfg
  from piper_push import hidden_plant, residual

  cfg = RateLimitedJointPositionActionCfg(entity_name="robot",
                                          actuator_names=("joint1",))
  assert cfg.command_hooks == ()
  assert cfg.latency_steps == 0
  assert cfg.response_scale == 1.0
  assert cfg.deadband == 0.0

  class A:
    pass

  a = A()
  assert hidden_plant.hidden_from_args(a) is None
  assert residual.residual_from_args(a) is None
  a.hidden_target, a.residual = False, ""
  assert hidden_plant.hidden_from_args(a) is None
  assert residual.residual_from_args(a) is None


def test_the_replay_never_writes_state_after_a_physics_step():
  """The phase's STOP condition.  The harness writes state only at the top of
  a control step, before `env.step`, to re-anchor a teacher-forced comparison;
  a write after the step would be faking dynamics."""
  text = (ROOT / "src/piper_push/replay.py").read_text()
  tree = ast.parse(text)
  for node in ast.walk(tree):
    if not isinstance(node, ast.For):
      continue
    body = node.body
    step_at = None
    for i, stmt in enumerate(body):
      src = ast.dump(stmt)
      if "'step'" in src and "attr='step'" in src:
        step_at = i
      if "write_state" in src:
        assert step_at is None, "write_state follows env.step in the same loop"
