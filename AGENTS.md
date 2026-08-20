# Codex project map and operating rules

## Mainline

This branch is the Piper X + 50 mm cube goal-conditioned continuous-pushing
sim-to-real probe. The current baseline is the committed cube Policy V2
baseline; the immediate engineering goals are a stable pushing primitive,
Policy V2 evaluation, and sim-to-real preparation.

Current branch: `codex/piperx-policy-v2-ik-localization`.

## Project map

- PPO train: `scripts/train_piperx_goal_push.py`
- PPO evaluation: `scripts/evaluate_piperx_goal_push.py`
- Policy V2 environment: `src/livingtwin_mujoco_rl/piperx_goal_push_env.py`
- Push execution/controller: `src/livingtwin_mujoco_rl/piperx_continuous_selfplay.py`
- IK/engineering: `src/livingtwin_mujoco_rl/piperx_engineering.py`
- Policy viewer: `scripts/run_piperx_gate1_live_viewer.py`
- Best checkpoint: `results/piperx_goalpush_ppo_policy_v2_nominal_20260818T190900Z/checkpoints/step_000049152.pt`
- Frozen run config: the sibling `FROZEN_CONFIG.json`
- Piper X assets: `piperx-mjlab/` submodule, pinned by the superproject

See `README.md` and `CLAUDE.md` for commands and detailed context.

## Frozen task semantics

- 12-D observation;
- 2-D direction plus continuous after-touch travel action;
- 50 mm cube;
- initial goal distance 60–140 mm;
- success threshold at or below 5 mm.

Do not change reward, action, observation, task distribution, or termination
semantics without an explicit decision and a read-only audit first.

## Known issue and operating rules

Some directions still have final contact-acquisition or pose-constraint
failures. Do not respond by casually adding route/waypoint/tolerance hacks.

- Read `README.md` and `CLAUDE.md` before acting.
- Never reset, clean, or discard a dirty worktree.
- Preserve untracked audit/diagnostic scripts.
- Perform a read-only check before changing experiment behavior.
- Do not overwrite or silently replace checkpoints.
