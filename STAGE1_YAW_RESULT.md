# Stage 1 position + yaw PPO result

**Final status: `FAIL_GATE`. Stage 2 was not started.**

All preflight checks passed, the smoke run was numerically healthy but had 0% joint success, and all three formal runs completed exactly 1,048,576 environment steps.

| Seed | Common joint | Common position | Common yaw | Generalization joint | Generalization position | Generalization yaw |
|---:|---:|---:|---:|---:|---:|---:|
| 2201 | 4.0% | 12.0% | 20.0% | 2.8% | 20.8% | 11.2% |
| 2202 | 11.0% | 14.0% | 35.0% | 6.0% | 8.8% | 26.8% |
| 2203 | 9.0% | 9.0% | 40.0% | 6.6% | 6.6% | 35.2% |

The required gates were 90% common and 85% disjoint generalization joint success for every seed. All checkpoint reload evaluations reproduced their metrics with maximum absolute difference 0, and the CSV finite-value audit passed.

Seed 2201 remained relatively cautious and contacted too little. Seeds 2202 and 2203 learned high-contact rotation behavior but sacrificed position and frequently terminated out of bounds. This is a joint-objective/finite-budget PPO failure, not evidence that yaw is physically uncontrollable and not a checkpoint serialization failure.

No automatic diagnostic extension was run because all seeds were far below the gate and two had a clear safety-boundary failure mode. Since Stage 1 did not yield an accepted fixed policy, the Stage 2 mass/friction/COM sensitivity scan was correctly not started.

Three-seed curve: `/home/wenjing/projects/livingtwin_mujoco_rl/results/yaw_baseline_summary/THREE_SEED_YAW_LEARNING_CURVES.png`
Machine-readable summary: `/home/wenjing/projects/livingtwin_mujoco_rl/results/yaw_baseline_summary/STAGE1_SUMMARY.json`
