"""The arithmetic of scripts/pc/eval_initiation.py, without a simulator.

Attempts, waits and stalls are what the second generation is judged on, and
each of them is an edge or a run length -- exactly where an off-by-one would
invert the reading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "pc"))

from eval_initiation import summarise  # noqa: E402

DT = 0.02


def _trace(T, B):
  z = lambda: np.zeros((T, B), np.float32)
  tr = {k: z() for k in ("grasped", "engaged", "on_table", "in_bin", "placed", "reset", "term", "fresh",
                         "target_full", "target_sampled", "jaw_cmd", "jaw_meas", "dist", "target_idx")}
  tr["on_table"][:] = 1.0
  tr["fresh"][:] = 1.0
  tr["dist"][:] = 0.3
  tr["term_names"] = ["object_lost", "over_speed"]
  return tr


def test_one_clean_cycle_is_one_attempt_one_grasp_one_placement_no_stall():
  T, B = 400, 1
  tr = _trace(T, B)
  # approach 0-49, engaged 50-79, carry 80-199, release/place 200-209, placement registers at 209
  tr["engaged"][50:80] = 1.0; tr["dist"][50:80] = 0.05
  tr["grasped"][80:200] = 1.0
  tr["on_table"][200:210] = 0.0
  tr["placed"][209] = 1.0
  # then the new object: approach 210-259, engaged from 260 (wait 51 steps = 1.02 s)
  tr["engaged"][260:300] = 1.0
  tr["grasped"][300:] = 1.0
  s = summarise(tr, DT, 100, idle_s=3.0)
  assert s["attempts_per_min"] * s["arm_minutes"] == 2
  assert s["grasps_per_min"] * s["arm_minutes"] == 2
  assert s["placed_per_min"] * s["arm_minutes"] == 1
  assert s["success_per_attempt"] == 0.5
  assert s["drops"] == 0
  assert s["stalls"]["n"] == 0
  w = s["wait_after_success_s"]
  assert w["n"] == 1 and abs(w["p50"] - 51 * DT) < 1e-9
  assert abs(s["time_to_first_attempt_s"]["p50"] - 50 * DT) < 1e-9
  ph = s["by_phase"]
  assert abs(ph["approach_first"]["steps_fraction"] - 50 / T) < 1e-9
  assert abs(ph["approach_re"]["steps_fraction"] - 50 / T) < 1e-9
  assert abs(ph["carry"]["steps_fraction"] - 220 / T) < 1e-9
  assert abs(ph["place"]["steps_fraction"] - 10 / T) < 1e-9


def test_a_release_without_a_placement_is_a_drop_and_its_wait_is_measured():
  T, B = 600, 1
  tr = _trace(T, B)
  tr["engaged"][10:20] = 1.0
  tr["grasped"][20:100] = 1.0           # released at step 100, nothing follows for > 3 s
  tr["engaged"][350:360] = 1.0          # the next attempt, 250 steps later
  s = summarise(tr, DT, 100, idle_s=3.0)
  assert s["drops"] == 1 and s["placed_per_min"] == 0
  assert s["wait_after_drop_s"]["n"] == 1 and abs(s["wait_after_drop_s"]["p50"] - 250 * DT) < 1e-9
  # the 250-step approach run (5 s) is a stall ended by the attempt; the 240
  # steps after it (4.8 s) are a second stall, cut off by the end of the run
  assert s["stalls"]["n"] == 2 and s["stalls"]["ended_by"] == {"attempt": 1, "reset": 0, "censored": 1}
  assert abs(s["stalls"]["length_s_max"] - 250 * DT) < 1e-9
  assert s["approach_run_s"]["n"] == 3


def test_a_reset_splits_runs_and_censors_waits():
  T, B = 500, 2
  tr = _trace(T, B)
  # env 0: a placement at 100, then nothing until a reset at 300, then an attempt at 320
  tr["grasped"][50:90, 0] = 1.0; tr["engaged"][40:50, 0] = 1.0
  tr["on_table"][90:101, 0] = 0.0; tr["placed"][100, 0] = 1.0
  tr["reset"][300, 0] = 1.0; tr["term"][300, 0] = 1
  tr["engaged"][320:330, 0] = 1.0
  # env 1: never does anything -> one censored stall, one censored first attempt
  s = summarise(tr, DT, 100, idle_s=3.0)
  w = s["wait_after_success_s"]
  assert w["n"] == 0 and w["censored"] == 1
  assert s["terminations"] == {"object_lost": 1, "over_speed": 0} and s["resets"] == 1
  st = s["stalls"]
  # env 0: 101..300 (200 steps, ended by the reset), 330..499 (170 steps, censored);
  # env 1: 0..499 (censored).  The 19 steps after the reset are not a stall.
  assert st["n"] == 3 and st["ended_by"] == {"attempt": 0, "reset": 1, "censored": 2}
  assert s["time_to_first_attempt_s"]["censored"] == 1
  # the attempt right after the reset counts from the new episode's start
  assert s["time_to_first_attempt_s"]["n"] == 2


def test_target_visibility_is_read_on_fresh_frames_by_phase():
  T, B = 200, 1
  tr = _trace(T, B)
  tr["engaged"][100:] = 1.0
  tr["target_sampled"][:100] = 6.0
  tr["target_sampled"][100:] = 20.0
  tr["fresh"][:] = 0.0
  tr["fresh"][::2] = 1.0
  s = summarise(tr, DT, 50, idle_s=3.0)
  assert s["by_phase"]["approach_first"]["target_points_mean"] == 6.0
  assert s["by_phase"]["engaged"]["target_points_mean"] == 20.0
  assert s["by_phase"]["approach_first"]["fresh_frames"] == 50
  # late/early on attempts: one attempt in the second half only
  assert s["early_attempts_per_min"] == 0.0 and s["late_attempts_per_min"] > 0


def test_a_stuck_object_is_not_a_stall_and_live_rates_exclude_it():
  T, B = 600, 2
  tr = _trace(T, B)

  def cycle(b, t0):
    tr["engaged"][t0 + 10:t0 + 20, b] = 1.0
    tr["grasped"][t0 + 20:t0 + 50, b] = 1.0
    tr["on_table"][t0 + 50:t0 + 60, b] = 0.0
    tr["placed"][t0 + 59, b] = 1.0

  # env 0: two cycles, then the object sits in the bin footprint unregistered from step 160 on
  cycle(0, 0); cycle(0, 100)
  tr["in_bin"][160:, 0] = 1.0
  # env 1: a cycle every 100 steps, steadily
  for k in range(6):
    cycle(1, 100 * k)
  s = summarise(tr, DT, 100, idle_s=3.0)
  assert s["stalls"]["n"] == 0                      # a stuck object never counts as a stall
  # the place run starts at the release (step 150) and runs to the end: 450 steps
  assert s["stuck_object"]["n"] == 1 and abs(s["stuck_object"]["length_s_max"] - 450 * DT) < 1e-9
  assert s["end_state_env_fraction"]["stuck_object"] == 0.5
  # raw late/early: 5 placements in the first half against 3 in the second
  assert abs(s["late_over_early_placed"] - 0.6) < 1e-9
  # over live time the steady environment is what is left, and the ratio says so
  assert s["late_over_early_placed_live"] >= 0.85
  assert s["placed_per_live_min"] > s["placed_per_min"]
