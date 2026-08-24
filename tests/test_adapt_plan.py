"""The plan has to name its own axis, or a run trains on one and is measured
on another.

`wm1_adapt.sh` used to hard-code `--latency-probs` for training and
`--obs-latency-steps 3` for evaluation. Run against a WM1-B plan that would
have trained every damping configuration at nominal damping and evaluated it
in the nominal domain, then reported the result as a recovery: a clean table
and a wrong conclusion, with nothing in the output to show it. So the flags
travel in the plan and these tests check they arrive.

Run with:  micromamba run -n mjlab python -m pytest tests -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mjlab.tasks  # noqa: F401  -- import mjlab before us
import pytest

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "scripts" / "wm1_adapt_plan.py"
RUNNER = ROOT / "scripts" / "wm1_adapt.sh"


def plan(tmp_path, *args) -> dict:
  out = tmp_path / "plan.json"
  r = subprocess.run([sys.executable, str(PLAN), "--out", str(out), *args],
                     cwd=ROOT, capture_output=True, text=True)
  assert r.returncode == 0, r.stderr
  return json.loads(out.read_text())


def test_the_damping_plan_carries_the_damping_flags(tmp_path):
  d = plan(tmp_path, "--axis", "servo_damping_scale", "--stage", "anchor")
  assert d["axis"] == "servo_damping_scale"
  assert d["probs_flag"] == "--damping-probs"
  assert d["eval_flag"] == "--servo-damping-scale"
  assert d["eval_target_value"] == 0.75
  for job in d["jobs"]:
    assert job["probs_flag"] == "--damping-probs"
    assert job["eval_flag"] == "--servo-damping-scale"
    assert job["eval_value"] == 0.75
    assert len(job["probs"]) == 3


def test_the_latency_plan_is_unchanged(tmp_path):
  """WM1-A's plans must keep producing exactly what they produced, or the
  seed extension is not comparable with the runs it extends."""
  d = plan(tmp_path, "--axis", "obs_latency_steps", "--stage", "anchor")
  assert d["probs_flag"] == "--latency-probs"
  assert d["eval_flag"] == "--obs-latency-steps"
  assert d["eval_target_value"] == 3
  for job in d["jobs"]:
    assert len(job["probs"]) == 5


def test_the_axis_defaults_to_latency(tmp_path):
  d = plan(tmp_path, "--stage", "anchor")
  assert d["axis"] == "obs_latency_steps"


def test_the_known_parameter_reference_is_a_point_mass_at_the_target(tmp_path):
  for axis, target, n in (("servo_damping_scale", 0.75, 3),
                          ("obs_latency_steps", 3, 5)):
    d = plan(tmp_path, "--axis", axis, "--stage", "anchor")
    jobs = [j for j in d["jobs"] if "KNOWN_PARAM" in j["methods"]]
    assert jobs, f"{axis}: no known-parameter reference in the plan"
    probs = jobs[0]["probs"]
    assert len(probs) == n
    assert probs[jobs[0]["prior"]["values"].index(target)] == 1.0


def test_the_source_refit_is_a_point_mass_at_the_source(tmp_path):
  """`B0_prior_refit` spends the whole budget on the *wrong* answer, so that
  "PPO for 600 iterations helps a bit whatever you condition on" is measured
  rather than assumed."""
  d = plan(tmp_path, "--axis", "servo_damping_scale", "--stage", "anchor")
  jobs = [j for j in d["jobs"] if "B0_prior_refit" in j["methods"]]
  assert jobs
  assert jobs[0]["probs"] == [0.0, 1.0, 0.0]        # nominal, not 0.75


def test_the_runner_reads_both_flags_from_the_plan():
  """A grep, deliberately.  The runner is shell and the failure it guards
  against is a literal flag reappearing in it."""
  src = RUNNER.read_text()
  body = src.split("for SEED_E in", 1)
  assert len(body) == 2, "the evaluation loop moved; update this test"
  assert '--obs-latency-steps 3' not in src, (
    "the evaluation domain is hard-coded again")
  assert '--latency-probs "$PROBS"' not in src, (
    "the training flag is hard-coded again")
  assert '"$FLAG" "$PROBS"' in src
  assert '"$EFLAG" "$EVAL"' in src


def test_a_job_that_does_not_parse_stops_the_shard():
  """Not a silent skip: a plan the runner cannot read means the results it
  would have produced are missing, and missing runs are how a comparison
  quietly becomes a comparison of different things."""
  src = RUNNER.read_text()
  assert "did not parse" in src and "exit 4" in src


@pytest.mark.parametrize("axis", ["servo_damping_scale", "obs_latency_steps"])
def test_seeds_are_settable_and_recorded(tmp_path, axis):
  d = plan(tmp_path, "--axis", axis, "--stage", "anchor",
           "--seeds", "7,13,101")
  assert d["seeds"] == [7, 13, 101]
  assert {j["seed"] for j in d["jobs"]} == {7, 13, 101}


# ---------------------------------------------------------------------------
# Where the results land
# ---------------------------------------------------------------------------


def test_the_runner_writes_beside_its_plan():
  """`OUT` was `results/wm1_latency/adapt`, a constant.

  Pointed at a WM1-B plan it put damping evaluations into WM1-A's directory,
  where `wm1_seeds.py` globs `*__r*.json` -- two phases' trip rates silently
  pooled into one number, with matching filenames and no error anywhere.
  """
  src = RUNNER.read_text()
  assert "OUT=results/wm1_latency/adapt" not in src, (
    "the output directory is hard-coded to WM1-A again")
  assert 'OUT=${OUT:-$(dirname "$PLAN")}' in src


def test_the_queue_writes_beside_its_plan():
  from pathlib import Path as _P

  src = (ROOT / "scripts" / "wm1_queue.py").read_text()
  assert "ADAPT = Path(a.plan).parent" in src, (
    "the queue's output directory no longer follows its plan")
  # and the default is only a default
  assert src.index("ADAPT = Path(a.plan).parent") > src.index(
    'ADAPT = Path("results/wm1_latency/adapt")')
  del _P
