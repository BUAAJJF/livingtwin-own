#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from livingtwin_mujoco_rl.reporting import write_manifest


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/yaw_absorbing_diagnostic_summary"))
    parser.add_argument("--root-report", type=Path, default=Path("ABSORBING_DIAGNOSTIC_RESULT.md"))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    preflight = load_json(root / "results/yaw_absorbing_preflight/ABSORBING_PRECHECK_RESULTS.json")
    smoke = load_json(root / "results/yaw_absorbing_smoke_seed_1201/FINAL_REPORT.json")
    training = rows(root / "results/yaw_absorbing_smoke_seed_1201/training_metrics.csv")
    csv_paths = list((root / "results/yaw_absorbing_smoke_seed_1201").glob("*.csv"))
    invalid = []
    numeric_cells = 0
    for path in csv_paths:
        for line, row in enumerate(rows(path), start=2):
            for key, value in row.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                numeric_cells += 1
                if not math.isfinite(number):
                    invalid.append({"path": str(path), "line": line, "column": key, "value": value})
    audit = preflight["return_order_audit"]
    representative = audit["representatives"]
    common = {key: float(value) for key, value in smoke["final_common_evaluation"].items() if key != "global_step"}
    generalization = {key: float(value) for key, value in smoke["generalization_metrics"].items()}
    summary = {
        "schema_version": "yaw_absorbing_diagnostic_summary_v1",
        "status": "SMOKE_COMPLETE_STOPPED",
        "implementation_commit": smoke["code_commit"],
        "preflight_status": preflight["status"],
        "failure_reward": preflight["failure_reward_bound"]["frozen_failure_reward"],
        "failure_reward_basis": preflight["failure_reward_bound"],
        "original_return_order": {
            "success": {
                "undiscounted": representative["success"]["undiscounted_return"],
                "discounted_gamma_0_99": representative["success"]["discounted_return_gamma_0_99"],
            },
            "valid_timeout_meaningful_progress": {
                "undiscounted": representative["valid_timeout_meaningful_progress"]["undiscounted_return"],
                "discounted_gamma_0_99": representative["valid_timeout_meaningful_progress"]["discounted_return_gamma_0_99"],
            },
            "out_of_bounds": {
                "undiscounted": representative["out_of_bounds_failure"]["undiscounted_return"],
                "discounted_gamma_0_99": representative["out_of_bounds_failure"]["discounted_return_gamma_0_99"],
            },
        },
        "absorbing_fixed_out_of_bounds_return": {
            "undiscounted": representative["out_of_bounds_failure"]["absorbing_fixed_undiscounted_return"],
            "discounted_gamma_0_99": representative["out_of_bounds_failure"]["absorbing_fixed_discounted_return_gamma_0_99"],
        },
        "absorbing_semantics": preflight["absorbing_semantics"],
        "yaw_bin_reachability": preflight["yaw_bin_reachability"],
        "smoke": {
            "seed": 1201,
            "environment_steps": 65536,
            "common": common,
            "generalization": generalization,
            "maximum_training_failure_rate": max(float(row["completed_failure_rate"]) for row in training),
            "maximum_absorbing_state_fraction": max(float(row["absorbing_state_fraction"]) for row in training),
            "checkpoint_reload": smoke["checkpoint_reload"],
            "numeric_audit": {"passed": not invalid, "numeric_cells_checked": numeric_cells, "invalid": invalid},
            "replay": smoke["replay"],
        },
        "out_of_bounds_shortcut": {
            "structurally_removed": bool(audit["absorbing_fix_order_passed"]["discounted_gamma_0_99"]),
            "empirical_smoke_failure_rate": common["failure_rate"],
            "interpretation": "No out-of-bounds event occurred in either old or corrected 65,536-step smoke; the correction is therefore validated by return dominance and state-machine tests, while its training effect requires the longer formal budget where the old shortcut emerged.",
        },
        "recommend_formal_three_seed_run": True,
        "recommendation_boundary": "Run only a new preregistered absorbing-failure 3-seed Stage 1. Do not change reward weights, horizon, full-yaw distribution, architecture, or start Stage 2. Large-yaw completion remains unproven by the heuristic.",
        "formal_training_started": False,
    }
    (output / "ABSORBING_DIAGNOSTIC_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bins = preflight["yaw_bin_reachability"]["bins"]
    lines = [
        "# Absorbing-failure Stage 1 diagnostic",
        "",
        "**Status: smoke complete; stopped before formal training.**",
        "",
        "The original representative return order was success > out-of-bounds > valid timeout under both undiscounted and gamma=0.99 discounted return. With the frozen -1.262 absorbing reward, it becomes success > valid timeout > out-of-bounds.",
        "",
        "The absorbing state uses an all-zero 15-D observation, frozen MuJoCo state, ignored actions, no done mask through step 199, and done/zero bootstrap only at step 200.",
        "",
        "| Initial yaw error | Correct rotation | Simultaneous improvement | Heuristic success |",
        "|---|---:|---:|---:|",
    ]
    labels = (("0_45_deg", "0-45°"), ("45_90_deg", "45-90°"), ("90_135_deg", "90-135°"), ("135_180_deg", "135-180°"))
    for key, label in labels:
        value = bins[key]
        lines.append(f"| {label} | {value['correct_rotation_direction_fraction']:.1%} | {value['simultaneous_position_yaw_improvement_fraction']:.1%} | {value['success_rate']:.1%} |")
    lines.extend(
        [
            "",
            f"Smoke common: joint {common['joint_success_rate']:.1%}, position-only {common['position_success_rate']:.1%}, yaw-only {common['yaw_success_rate']:.1%}, failure {common['failure_rate']:.1%}, position error {common['mean_final_position_error_m']:.6f} m, yaw error {math.degrees(common['mean_final_yaw_error_rad']):.2f}°, discounted return {common['mean_discounted_return_gamma_0_99']:.3f}.",
            "",
            "No out-of-bounds event occurred during stochastic smoke collection or deterministic evaluation. The old smoke also had no out-of-bounds event, so identical task metrics are expected; the shortcut appeared only in longer old formal runs.",
            "",
            "Recommendation: a new preregistered three-seed absorbing-failure Stage 1 is justified to test the fix at the budget where the shortcut previously emerged. Do not modify any other task or PPO setting, and do not enter Stage 2 yet.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output / "ABSORBING_DIAGNOSTIC_REPORT.md").write_text(report, encoding="utf-8")
    (root / args.root_report).write_text(report, encoding="utf-8")
    write_manifest(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
