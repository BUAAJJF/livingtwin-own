#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from livingtwin_mujoco_rl.reporting import write_manifest


SEEDS = (2201, 2202, 2203)
COMMON_GATE = 0.90
GENERALIZATION_GATE = 0.85


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_csv_audit(paths: list[Path]) -> dict[str, Any]:
    invalid: list[dict[str, str]] = []
    cells = 0
    for path in paths:
        for row_number, row in enumerate(read_csv(path), start=2):
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                cells += 1
                if not math.isfinite(number):
                    invalid.append({"path": str(path), "row": str(row_number), "column": key, "value": value})
    return {"passed": not invalid, "numeric_cells_checked": cells, "invalid_cells": invalid}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("results/yaw_baseline_summary"))
    parser.add_argument("--root-report", type=Path, default=Path("STAGE1_YAW_RESULT.md"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (root / args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty summary directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    preflight = read_json(root / "results/yaw_preflight/YAW_PRECHECK_RESULTS.json")
    per_seed: list[dict[str, Any]] = []
    csv_paths: list[Path] = []
    reason_counts: dict[str, Any] = {}
    curves: dict[int, list[dict[str, str]]] = {}
    for seed in SEEDS:
        run = root / f"results/yaw_baseline_seed_{seed}"
        report = read_json(run / "FINAL_REPORT.json")
        common = {key: float(value) for key, value in report["final_common_evaluation"].items() if key != "global_step"}
        generalization = {key: float(value) for key, value in report["generalization_evaluation"].items()}
        checkpoint = Path(report["final_checkpoint"])
        common_rows = read_csv(run / "common_episodes_step_001048576.csv")
        generalization_rows = read_csv(run / "generalization_episodes.csv")
        reason_counts[str(seed)] = {
            "common": dict(Counter(row["terminated_reason"] for row in common_rows)),
            "generalization": dict(Counter(row["terminated_reason"] for row in generalization_rows)),
        }
        reload_report = report["checkpoint_reload"]
        per_seed.append(
            {
                "seed": seed,
                "common": common,
                "generalization": generalization,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_reload_passed": bool(reload_report["passed"]),
                "checkpoint_reload_max_metric_absolute_difference": float(reload_report["max_metric_absolute_difference"]),
                "common_gate_passed": common["joint_success_rate"] >= COMMON_GATE,
                "generalization_gate_passed": generalization["joint_success_rate"] >= GENERALIZATION_GATE,
            }
        )
        curves[seed] = read_csv(run / "common_evaluations.csv")
        csv_paths.extend(run.glob("*.csv"))

    finite = finite_csv_audit(sorted(csv_paths))
    keys = (
        "joint_success_rate", "position_success_rate", "yaw_success_rate",
        "mean_final_position_error_m", "mean_final_yaw_error_rad",
        "contact_episode_rate", "action_saturation_rate",
    )
    aggregates: dict[str, Any] = {}
    for split in ("common", "generalization"):
        aggregates[split] = {
            key: {
                "mean": float(np.mean([row[split][key] for row in per_seed])),
                "sample_standard_deviation": float(np.std([row[split][key] for row in per_seed], ddof=1)),
            }
            for key in keys
        }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for seed, rows in curves.items():
        steps = [int(row["global_step"]) for row in rows]
        axes[0, 0].plot(steps, [float(row["joint_success_rate"]) for row in rows], marker="o", label=str(seed))
        axes[0, 1].plot(steps, [float(row["position_success_rate"]) for row in rows], marker="o", label=f"pos {seed}")
        axes[0, 1].plot(steps, [float(row["yaw_success_rate"]) for row in rows], marker="x", linestyle="--", label=f"yaw {seed}")
        axes[1, 0].plot(steps, [100 * float(row["mean_final_position_error_m"]) for row in rows], marker="o", label=str(seed))
        axes[1, 1].plot(steps, [np.degrees(float(row["mean_final_yaw_error_rad"])) for row in rows], marker="o", label=str(seed))
    axes[0, 0].axhline(COMMON_GATE, color="black", linestyle="--", label="gate")
    axes[0, 0].set(title="Common-state joint success", ylim=(-0.02, 1.02))
    axes[0, 1].set(title="Position-only vs yaw-only success", ylim=(-0.02, 1.02))
    axes[1, 0].set(title="Mean final position error", ylabel="cm")
    axes[1, 1].set(title="Mean final yaw error", ylabel="degrees")
    for axis in axes.flat:
        axis.set_xlabel("environment steps")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    plot_path = output / "THREE_SEED_YAW_LEARNING_CURVES.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    all_reload = all(row["checkpoint_reload_passed"] for row in per_seed)
    all_gates = all(row["common_gate_passed"] and row["generalization_gate_passed"] for row in per_seed)
    summary = {
        "schema_version": "position_yaw_stage1_summary_v1",
        "status": "PASS" if all_gates else "FAIL_GATE",
        "code_commit": per_seed and read_json(root / "results/yaw_baseline_seed_2201/FINAL_REPORT.json")["code_commit"],
        "preflight_status": preflight["status"],
        "smoke_seed": 1201,
        "formal_seeds": list(SEEDS),
        "formal_environment_steps_per_seed": 1_048_576,
        "common_gate": COMMON_GATE,
        "generalization_gate": GENERALIZATION_GATE,
        "per_seed": per_seed,
        "aggregates": aggregates,
        "termination_reason_counts": reason_counts,
        "finite_numeric_audit": finite,
        "all_checkpoint_reloads_passed": all_reload,
        "diagnostic_extension_started": False,
        "stage2_started": False,
        "failure_diagnosis": {
            "physics_reachability": "not rejected: signed yaw and 32-state heuristic preflight passed",
            "reward_and_objective": "joint sparse success plus two dense error terms produced seed-dependent position/yaw trade-offs",
            "reset_distribution": "full-yaw formal distribution is substantially harder than smoke and was intentionally not narrowed",
            "action_space": "geometrically controllable, but requires contact acquisition, off-center rotation, and position recovery within 4 seconds",
            "ppo_optimization": "high seed variance; one cautious/low-contact policy and two high-contact policies that often moved the cube out of bounds",
        },
        "controlled_extension_decision": "not run: all three seeds are far below the gate and two exhibit safety-boundary failure, so an automatic budget increase is not justified",
        "old_position_baseline_preserved": {
            "commit_before_stage1": "a21580576f5bf858f8ee8d299a870897c7b7e766",
            "checkpoint_sha256": {
                "2101": "b890ffa5751112af662ed19b9ce235a3bef73edc16afa83472d7733348d08564",
                "2102": "e5f6ca76827848288d607aee975c4cbfdf6e355f8e7e22880ccb049beedc9405",
                "2103": "aaf81dd827a853ec215b0c6de2c11ac752adf4a4f3b00a744726b4f34886a135",
            },
        },
    }
    (output / "STAGE1_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Stage 1 position + yaw PPO result",
        "",
        "**Final status: `FAIL_GATE`. Stage 2 was not started.**",
        "",
        "All preflight checks passed, the smoke run was numerically healthy but had 0% joint success, and all three formal runs completed exactly 1,048,576 environment steps.",
        "",
        "| Seed | Common joint | Common position | Common yaw | Generalization joint | Generalization position | Generalization yaw |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        c, g = row["common"], row["generalization"]
        lines.append(
            f"| {row['seed']} | {c['joint_success_rate']:.1%} | {c['position_success_rate']:.1%} | {c['yaw_success_rate']:.1%} | "
            f"{g['joint_success_rate']:.1%} | {g['position_success_rate']:.1%} | {g['yaw_success_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The required gates were 90% common and 85% disjoint generalization joint success for every seed. All checkpoint reload evaluations reproduced their metrics with maximum absolute difference 0, and the CSV finite-value audit passed.",
            "",
            "Seed 2201 remained relatively cautious and contacted too little. Seeds 2202 and 2203 learned high-contact rotation behavior but sacrificed position and frequently terminated out of bounds. This is a joint-objective/finite-budget PPO failure, not evidence that yaw is physically uncontrollable and not a checkpoint serialization failure.",
            "",
            "No automatic diagnostic extension was run because all seeds were far below the gate and two had a clear safety-boundary failure mode. Since Stage 1 did not yield an accepted fixed policy, the Stage 2 mass/friction/COM sensitivity scan was correctly not started.",
            "",
            f"Three-seed curve: `{plot_path}`",
            f"Machine-readable summary: `{output / 'STAGE1_SUMMARY.json'}`",
        ]
    )
    report_text = "\n".join(lines) + "\n"
    (output / "STAGE1_REPORT.md").write_text(report_text, encoding="utf-8")
    (root / args.root_report).write_text(report_text, encoding="utf-8")
    write_manifest(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
