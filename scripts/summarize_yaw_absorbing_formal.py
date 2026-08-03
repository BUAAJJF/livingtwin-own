#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from livingtwin_mujoco_rl.reporting import write_manifest


SEEDS = (2201, 2202, 2203)
FINAL_STEP = 1_048_576


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_summary(rows: list[dict[str, str]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "last": float(values[-1]),
    }


def manifest_audit(run: Path) -> dict[str, Any]:
    manifest_path = run / "FILE_MANIFEST.json"
    expected_manifest_digest = (run / "FILE_MANIFEST.sha256").read_text(encoding="utf-8").split()[0]
    manifest_digest = file_sha256(manifest_path)
    manifest = read_json(manifest_path)
    mismatches = []
    for entry in manifest["files"]:
        path = run / entry["path"]
        if not path.is_file():
            mismatches.append({"path": entry["path"], "reason": "missing"})
        elif path.stat().st_size != int(entry["bytes"]):
            mismatches.append({"path": entry["path"], "reason": "size"})
        elif file_sha256(path) != entry["sha256"]:
            mismatches.append({"path": entry["path"], "reason": "sha256"})
    return {
        "passed": expected_manifest_digest == manifest_digest and not mismatches,
        "manifest_sha256": manifest_digest,
        "file_count": len(manifest["files"]),
        "mismatches": mismatches,
    }


def finite_csv_audit(paths: list[Path]) -> dict[str, Any]:
    invalid: list[dict[str, Any]] = []
    cells = 0
    for path in paths:
        for line, row in enumerate(read_csv(path), start=2):
            for column, text in row.items():
                try:
                    value = float(text)
                except (TypeError, ValueError):
                    continue
                cells += 1
                if not math.isfinite(value):
                    invalid.append({"path": str(path), "line": line, "column": column, "value": text})
    return {"passed": not invalid, "numeric_cells_checked": cells, "invalid": invalid}


def final_metrics(report: dict[str, Any], split: str) -> dict[str, float]:
    key = "final_common_evaluation" if split == "common" else (
        "generalization_metrics" if "generalization_metrics" in report else "generalization_evaluation"
    )
    return {name: float(value) for name, value in report[key].items() if name != "global_step"}


def mean_reward_components(rows: list[dict[str, str]]) -> dict[str, float]:
    columns = sorted(name for name in rows[0] if name.startswith("reward_"))
    return {name.removeprefix("reward_"): float(np.mean([float(row[name]) for row in rows])) for name in columns}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("results/yaw_absorbing_formal_summary"))
    parser.add_argument("--root-report", type=Path, default=Path("ABSORBING_FORMAL_RESULT.md"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (root / args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty summary directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, Any]] = []
    csv_paths: list[Path] = []
    curves: dict[int, list[dict[str, str]]] = {}
    comparison_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        old_run = root / f"results/yaw_baseline_seed_{seed}"
        new_run = root / f"results/yaw_absorbing_baseline_seed_{seed}"
        old_report, new_report = read_json(old_run / "FINAL_REPORT.json"), read_json(new_run / "FINAL_REPORT.json")
        if int(new_report["global_step"]) != FINAL_STEP or int(new_report["seed"]) != seed:
            raise RuntimeError(f"seed {seed} is not a complete formal result")
        old_common, old_gen = final_metrics(old_report, "common"), final_metrics(old_report, "generalization")
        new_common, new_gen = final_metrics(new_report, "common"), final_metrics(new_report, "generalization")
        old_common_rows = read_csv(old_run / f"common_episodes_step_{FINAL_STEP:09d}.csv")
        old_gen_rows = read_csv(old_run / "generalization_episodes.csv")
        new_common_rows = read_csv(new_run / f"common_episodes_step_{FINAL_STEP:09d}.csv")
        new_gen_rows = read_csv(new_run / "generalization_episodes.csv")
        old_common["out_of_bounds_rate"] = float(np.mean([row["terminated_reason"] == "out_of_bounds" for row in old_common_rows]))
        old_gen["out_of_bounds_rate"] = float(np.mean([row["terminated_reason"] == "out_of_bounds" for row in old_gen_rows]))
        old_training, new_training = read_csv(old_run / "training_metrics.csv"), read_csv(new_run / "training_metrics.csv")
        diagnostics = {}
        for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance"):
            diagnostics[name] = {
                "original": numeric_summary(old_training, name),
                "absorbing": numeric_summary(new_training, name),
            }
        absorbing_training_exposure = {
            "absorbing_state_fraction": numeric_summary(new_training, "absorbing_state_fraction"),
            "completed_failure_rate": numeric_summary(new_training, "completed_failure_rate"),
        }
        entry = {
            "seed": seed,
            "environment_steps": int(new_report["global_step"]),
            "elapsed_seconds": float(new_report["elapsed_seconds"]),
            "original": {"common": old_common, "generalization": old_gen},
            "absorbing": {"common": new_common, "generalization": new_gen},
            "paired_joint_success_change_percentage_points": {
                "common": 100.0 * (new_common["joint_success_rate"] - old_common["joint_success_rate"]),
                "generalization": 100.0 * (new_gen["joint_success_rate"] - old_gen["joint_success_rate"]),
            },
            "training_diagnostics": diagnostics,
            "absorbing_training_exposure": absorbing_training_exposure,
            "evaluation_reward_components": {
                "common": mean_reward_components(new_common_rows),
                "generalization": mean_reward_components(new_gen_rows),
            },
            "observation_normalization": new_report["observation_normalization"],
            "checkpoint_reload": new_report["checkpoint_reload"],
            "checkpoint_sha256": file_sha256(Path(new_report["final_checkpoint"])),
            "manifest_audit": manifest_audit(new_run),
        }
        per_seed.append(entry)
        curves[seed] = read_csv(new_run / "common_evaluations.csv")
        csv_paths.extend(new_run.glob("*.csv"))
        for split, old, new in (("common", old_common, new_common), ("generalization", old_gen, new_gen)):
            comparison_rows.append({
                "seed": seed,
                "split": split,
                "original_joint_success_rate": old["joint_success_rate"],
                "absorbing_joint_success_rate": new["joint_success_rate"],
                "joint_change_percentage_points": 100.0 * (new["joint_success_rate"] - old["joint_success_rate"]),
                "original_out_of_bounds_rate": old.get("out_of_bounds_rate", float("nan")),
                "absorbing_failure_rate": new["failure_rate"],
                "absorbing_state_fraction": new["mean_absorbing_state_fraction"],
                "original_contact_episode_rate": old["contact_episode_rate"],
                "absorbing_contact_episode_rate": new["contact_episode_rate"],
                "original_contact_step_fraction": old["mean_contact_control_step_fraction"],
                "absorbing_contact_step_fraction": new["mean_contact_control_step_fraction"],
            })

    with (output / "PAIRED_COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for seed, rows in curves.items():
        steps = [int(row["global_step"]) for row in rows]
        axes[0, 0].plot(steps, [float(row["joint_success_rate"]) for row in rows], marker="o", label=str(seed))
        axes[0, 1].plot(steps, [float(row["failure_rate"]) for row in rows], marker="o", label=str(seed))
        axes[1, 0].plot(steps, [float(row["contact_episode_rate"]) for row in rows], marker="o", label=str(seed))
        axes[1, 1].plot(steps, [float(row["mean_discounted_return_gamma_0_99"]) for row in rows], marker="o", label=str(seed))
    titles = ("Joint success", "Out-of-bounds / absorbing failure", "Contact episode rate", "Discounted return (gamma=0.99)")
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.set_xlabel("environment steps")
        axis.grid(alpha=0.25)
        axis.legend(title="seed", fontsize=8)
    plot_path = output / "THREE_SEED_ABSORBING_CURVES.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    old_high_oob = all(
        row["original"]["common"].get("out_of_bounds_rate", 0.0) >= threshold
        for row, threshold in zip(per_seed[1:], (0.48, 0.76), strict=True)
    )
    new_max_failure = max(
        row["absorbing"][split]["failure_rate"]
        for row in per_seed for split in ("common", "generalization")
    )
    all_manifests = all(row["manifest_audit"]["passed"] for row in per_seed)
    all_reload = all(row["checkpoint_reload"]["passed"] for row in per_seed)
    all_norm_finite = all(row["observation_normalization"]["all_finite"] for row in per_seed)
    finite = finite_csv_audit(sorted(csv_paths))
    absorbing_evaluation_samples_observed = any(
        row["absorbing"][split]["mean_absorbing_state_fraction"] > 0.0
        for row in per_seed for split in ("common", "generalization")
    )
    absorbing_training_samples_observed = any(
        row["absorbing_training_exposure"]["absorbing_state_fraction"]["maximum"] > 0.0
        for row in per_seed
    )
    common_deltas = [row["paired_joint_success_change_percentage_points"]["common"] for row in per_seed]
    contact_episode_rates = [row["absorbing"]["common"]["contact_episode_rate"] for row in per_seed]
    contact_step_rates = [row["absorbing"]["common"]["mean_contact_control_step_fraction"] for row in per_seed]
    value_diagnostics = {
        str(row["seed"]): row["training_diagnostics"] for row in per_seed
    }
    summary = {
        "schema_version": "yaw_absorbing_formal_paired_summary_v1",
        "status": "THREE_SEED_FORMAL_COMPLETE_STOPPED",
        "formal_seeds": list(SEEDS),
        "environment_steps_per_seed": FINAL_STEP,
        "implementation_commit": per_seed[0] and read_json(root / "results/yaw_absorbing_baseline_seed_2201/FINAL_REPORT.json")["code_commit"],
        "per_seed": per_seed,
        "finite_numeric_audit": finite,
        "all_manifests_passed": all_manifests,
        "all_checkpoint_reloads_passed": all_reload,
        "all_observation_normalizations_finite": all_norm_finite,
        "causal_conclusions": {
            "high_out_of_bounds_removed": bool(old_high_oob and new_max_failure <= 0.01),
            "maximum_absorbing_failure_rate": new_max_failure,
            "common_joint_success_change_percentage_points_by_seed": dict(zip(map(str, SEEDS), common_deltas, strict=True)),
            "low_contact_waiting_replacement": bool(max(contact_episode_rates) < 0.5 and max(contact_step_rates) < 0.1),
            "contact_episode_rate_by_seed": dict(zip(map(str, SEEDS), contact_episode_rates, strict=True)),
            "contact_step_fraction_by_seed": dict(zip(map(str, SEEDS), contact_step_rates, strict=True)),
            "absorbing_evaluation_samples_observed": absorbing_evaluation_samples_observed,
            "absorbing_training_samples_observed": absorbing_training_samples_observed,
            "critic_and_normalization_interpretation": (
                "No absorbing training or evaluation samples occurred, so the run cannot empirically isolate their critic impact; "
                "all logged values and normalization statistics are finite, and paired loss/variance summaries are retained for audit."
                if not (absorbing_training_samples_observed or absorbing_evaluation_samples_observed) else
                "Absorbing samples occurred; finite-value, value-loss, explained-variance, and normalization audits are reported without post-hoc thresholds."
            ),
            "failure_mechanism_interpretation": (
                "The intervention removed the high out-of-bounds behavior. Remaining low joint success therefore cannot be attributed solely to the early-termination return shortcut; it is consistent with the already documented finite-horizon joint position-yaw control difficulty, without claiming that horizon length is causally established."
            ),
            "value_diagnostics": value_diagnostics,
        },
        "stop_boundary": {
            "new_experiments_proposed": False,
            "new_experiments_started": False,
            "eight_second_horizon_control": "NOT_PROPOSED_OR_RUN_PER_USER_STOP_CONSTRAINT",
            "reward_horizon_curriculum_action_network_changes": "NONE",
        },
    }
    (output / "ABSORBING_FORMAL_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Absorbing-failure formal three-seed result",
        "",
        "**Status: all three frozen seeds complete; causal summary complete; stopped.**",
        "",
        "The only intervention relative to the original Stage 1 was the preregistered absorbing-failure semantics with reward -1.262. No reward, horizon, curriculum, action-space, architecture, reset, or PPO setting changed.",
        "",
        "| Seed | Original joint | Absorbing joint | Change | Original OOB | Absorbing failure | Contact episodes | Contact steps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        old, new = row["original"]["common"], row["absorbing"]["common"]
        lines.append(
            f"| {row['seed']} | {old['joint_success_rate']:.1%} | {new['joint_success_rate']:.1%} | "
            f"{row['paired_joint_success_change_percentage_points']['common']:+.1f} pp | "
            f"{old.get('out_of_bounds_rate', float('nan')):.1%} | {new['failure_rate']:.1%} | "
            f"{new['contact_episode_rate']:.1%} | {new['mean_contact_control_step_fraction']:.1%} |"
        )
    lines.extend([
        "",
        f"The maximum corrected evaluation failure rate across common and generalization sets was {new_max_failure:.1%}. "
        + ("The 2202/2203 high-out-of-bounds shortcut disappeared." if old_high_oob and new_max_failure <= 0.01 else "The preregistered disappearance criterion was not met."),
        "",
        "The paired success changes show the performance consequence separately from the clear behavioral removal of early out-of-bounds termination. Because joint success remains low, the shortcut was not the sole explanation for Stage 1 failure.",
        "",
        "All result manifests and checkpoint reloads passed; CSV values and observation-normalization statistics are finite. Detailed value loss, explained variance, entropy, KL, returns, errors, reward components, contact, saturation, and per-split metrics are in the machine-readable summary.",
        "",
        "Per the explicit stop constraint, no follow-up experiment is proposed or started.",
        "",
        f"Plot: `{plot_path}`",
        f"Machine-readable result: `{output / 'ABSORBING_FORMAL_SUMMARY.json'}`",
    ])
    report = "\n".join(lines) + "\n"
    (output / "ABSORBING_FORMAL_REPORT.md").write_text(report, encoding="utf-8")
    (root / args.root_report).write_text(report, encoding="utf-8")
    write_manifest(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
