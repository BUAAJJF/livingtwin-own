"""The gate's arithmetic, and its refusal to report an unrun stage as a result.

The phase's worst failure mode is not a wrong number; it is a gate that reads
GREEN because its inputs are missing, or one whose thresholds drifted towards
the result.  Both are testable.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
  "ra_gate", Path(__file__).resolve().parent.parent / "scripts" / "ra_sim0_gate.py")
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


def acc_blob(candidate, split, h1, h10, h25, *, finite=True, bins=None,
             rms=0.03):
  return {
    "candidate": candidate, "tag": candidate,
    "rec_meta": {"split": split},
    "period1": {"h1": {"q": {"nrms": h1, "rms": rms, "n": 40000},
                       "qd": {"nrms": h1 * 5}},
                "usable_fraction": 0.94,
                "sanity": {"finite": finite},
                "error_by_command_magnitude": bins or [],
                "error_after_reversal": {"rms": rms, "n": 1000},
                "error_not_after_reversal": {"rms": rms, "n": 1000},
                "error_autocorrelation_lag1": 0.98},
    "period25": {"h1": {"q": {"nrms": h1}}, "h5": {"q": {"nrms": h1}},
                 "h10": {"q": {"nrms": h10}}, "h25": {"q": {"nrms": h25}}},
  }


def test_the_thresholds_are_the_ones_the_plan_committed_to():
  """Transcribed, not computed.  If a later commit softens one, this fails."""
  assert G.R_THRESHOLDS == {"h1": 0.30, "h10": 0.25, "h25": 0.20}
  assert G.C_THROUGHPUT_GAIN == 0.05
  assert G.C_TRIP_RATIO == 0.70
  assert G.C_RETENTION_LOSS == 0.05


def test_a_missing_stage_is_not_executed_and_not_a_pass():
  assert G.gate_r({})["verdict"] == "not_executed"
  assert G.gate_c({})["verdict"] == "not_executed"
  assert G.gate_p({}, {})["verdict"] == "not_executed"


def test_a_residual_that_is_worse_fails_and_says_by_how_much():
  acc = {"param_fit": [acc_blob("param_fit", "test", 1.50, 1.30, 0.61)],
         "residual": [acc_blob("residual", "test", 1.90 + 0.01 * i, 1.47, 0.71)
                      for i in range(3)]}
  r = G.gate_r(acc)
  assert r["verdict"] == "RED"
  row = r["per_split"]["test"]["h1"]
  assert row["ratio"]["mean"] > 1.0
  assert row["reduction"] < 0.0
  assert not row["pass"]


def test_a_residual_that_clears_every_threshold_on_both_splits_passes():
  def arms(split):
    return ([acc_blob("param_fit", split, 1.50, 1.30, 0.61)],
            [acc_blob("residual", split, 1.00 + 0.001 * i, 0.90, 0.45)
             for i in range(3)])
  acc = {"param_fit": [], "residual": []}
  for split in ("test", "test_amp"):
    b, r = arms(split)
    acc["param_fit"] += b
    acc["residual"] += r
  out = G.gate_r(acc)
  assert out["verdict"] == "GREEN", out["per_split"]


def test_one_split_clearing_is_not_enough():
  """The plan asks for both the shape split and the action split."""
  acc = {"param_fit": [acc_blob("param_fit", "test", 1.50, 1.30, 0.61)],
         "residual": [acc_blob("residual", "test", 1.00, 0.90, 0.45)
                      for _ in range(3)]}
  assert G.gate_r(acc)["verdict"] == "RED"


def test_a_point_estimate_that_clears_with_an_interval_that_does_not_fails():
  """Three noisy seeds whose mean clears 30% but whose interval does not."""
  acc = {"param_fit": [], "residual": []}
  for split in ("test", "test_amp"):
    acc["param_fit"].append(acc_blob("param_fit", split, 1.00, 1.00, 1.00))
    for v in (0.50, 0.70, 0.90):
      acc["residual"].append(acc_blob("residual", split, v, 0.60, 0.60))
  out = G.gate_r(acc)
  assert out["per_split"]["test"]["h1"]["ratio"]["mean"] == pytest.approx(0.70)
  assert not out["per_split"]["test"]["h1"]["pass"]
  assert out["verdict"] == "RED"


def test_a_non_finite_run_cannot_pass_however_good_its_numbers_are():
  acc = {"param_fit": [], "residual": []}
  for split in ("test", "test_amp"):
    acc["param_fit"].append(acc_blob("param_fit", split, 1.50, 1.30, 0.61))
    for i in range(3):
      acc["residual"].append(
        acc_blob("residual", split, 0.10, 0.10, 0.10, finite=(i != 0)))
  out = G.gate_r(acc)
  assert not out["finite"]
  assert out["verdict"] == "RED"


def test_gate_c_is_stopped_by_not_silently_green_when_r_fails(tmp_path):
  root = tmp_path
  (root / "accuracy").mkdir()
  (root / "calibration").mkdir()
  (root / "calibration" / "d.json").write_text(json.dumps({
    "oracle": False, "trace": [
      {"tag": "nominal", "nrms_q": 2.2, "damping": 1.0, "latency_steps": 0,
       "response_scale": 1.0, "deadband": 0.0, "lowpass_hz": None},
      {"tag": "r0:x", "nrms_q": 1.5, "damping": 1.0, "latency_steps": 1,
       "response_scale": 0.6, "deadband": 0.008, "lowpass_hz": None}]}))
  bins = [{"lo": 0.0, "hi": 0.01, "n": 5000, "rms": 0.02, "mean_signed": 0.01},
          {"lo": 0.01, "hi": 1.0, "n": 5000, "rms": 0.05, "mean_signed": -0.002}]
  for name, blob in (("param_fit_test",
                      acc_blob("param_fit", "test", 1.5, 1.3, 0.6, bins=bins)),
                     ("residual_test", acc_blob("residual", "test", 1.9, 1.5, 0.7)),
                     ("oracle_test", acc_blob("oracle", "test", 0.13, 0.7, 0.3))):
    (root / "accuracy" / f"{name}.json").write_text(json.dumps(blob))
  import sys
  argv = sys.argv
  sys.argv = ["gate", "--root", str(root), "--accuracy", "accuracy"]
  try:
    G.main()
  finally:
    sys.argv = argv
  out = json.loads((root / "gate.json").read_text())
  assert out["gate_R"]["verdict"] == "RED"
  assert out["gate_C"]["verdict"] == "stopped_by_gate_R"
  assert out["overall"] == "RED"


def test_the_interval_widens_with_the_spread_and_names_its_n():
  tight = G.mean_ci([1.00, 1.01, 0.99])
  loose = G.mean_ci([0.50, 1.00, 1.50])
  assert tight["n"] == loose["n"] == 3
  assert (loose["hi"] - loose["lo"]) > (tight["hi"] - tight["lo"])
  # Two degrees of freedom, and the constant says so rather than borrowing 1.96.
  assert G.T95_DF2 == pytest.approx(4.3027, abs=1e-3)
  assert math.isnan(G.mean_ci([1.0])["lo"])
  assert G.mean_ci([])["n"] == 0
