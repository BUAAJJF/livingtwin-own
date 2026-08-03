# Absorbing-failure formal three-seed result

**Status: all three frozen seeds complete; causal summary complete; stopped.**

The only intervention relative to the original Stage 1 was the preregistered absorbing-failure semantics with reward -1.262. No reward, horizon, curriculum, action-space, architecture, reset, or PPO setting changed.

| Seed | Original joint | Absorbing joint | Change | Original OOB | Absorbing failure | Contact episodes | Contact steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2201 | 4.0% | 3.0% | -1.0 pp | 0.0% | 0.0% | 48.0% | 8.3% |
| 2202 | 11.0% | 15.0% | +4.0 pp | 48.0% | 0.0% | 100.0% | 28.1% |
| 2203 | 9.0% | 3.0% | -6.0 pp | 76.0% | 0.0% | 69.0% | 12.8% |

The maximum corrected evaluation failure rate across common and generalization sets was 0.2%. The 2202/2203 high-out-of-bounds shortcut disappeared.

The paired success changes show the performance consequence separately from the clear behavioral removal of early out-of-bounds termination. Because joint success remains low, the shortcut was not the sole explanation for Stage 1 failure.

All result manifests and checkpoint reloads passed; CSV values and observation-normalization statistics are finite. Detailed value loss, explained variance, entropy, KL, returns, errors, reward components, contact, saturation, and per-split metrics are in the machine-readable summary.

Per the explicit stop constraint, no follow-up experiment is proposed or started.

Plot: `/home/wenjing/projects/livingtwin_mujoco_rl/results/yaw_absorbing_formal_summary/THREE_SEED_ABSORBING_CURVES.png`
Machine-readable result: `/home/wenjing/projects/livingtwin_mujoco_rl/results/yaw_absorbing_formal_summary/ABSORBING_FORMAL_SUMMARY.json`
