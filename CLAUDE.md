# LivingTwin / Piper X quick handoff

## Current scope

This repository contains the Piper X goal-conditioned continuous-pushing
simulation probe: a 50 mm cube, nominal MuJoCo execution, and Policy V2 PPO.
Do not infer unverified behavior from historical diagnostics.

Policy V2 uses a 12-D observation and a 2-D action: goal-relative push
direction plus continuous after-touch travel/magnitude.

## Important entry points

- Environment: `src/livingtwin_mujoco_rl/piperx_goal_push_env.py`
- Execution/controller: `src/livingtwin_mujoco_rl/piperx_continuous_selfplay.py`
- IK/engineering: `src/livingtwin_mujoco_rl/piperx_engineering.py`
- PPO/network: `src/livingtwin_mujoco_rl/piperx_ppo.py`, `src/livingtwin_mujoco_rl/networks.py`
- Train: `scripts/train_piperx_goal_push.py`
- Evaluate: `scripts/evaluate_piperx_goal_push.py`
- Replay: `scripts/replay_piperx_policy_v2_ik_localization.py`
- Viewer: `scripts/run_piperx_gate1_live_viewer.py`

The controller's nominal flow is approach/transport, contact acquisition,
measured sustained push, retract, and settle/reobserve. Some directions still
have known final contact-acquisition or pose-constraint failures; this is an
engineering limitation under investigation, not a reason to change Policy V2
semantics here.

## Asset and checkpoint prerequisites

`piperx-mjlab` is intended to be a git submodule at the pinned lab asset
commit `71e490c46ae8ffd7c5c70b62ac900f90425e2d1b`. Clone with
`--recurse-submodules`. The Policy V2 viewer/evaluation checkpoint is:

`results/piperx_goalpush_ppo_policy_v2_nominal_20260818T190900Z/checkpoints/step_000049152.pt`

with its sibling `FROZEN_CONFIG.json`.

## Short commands

```bash
git clone --recurse-submodules https://github.com/orcabotics/LivingTwin.git
cd LivingTwin
python3.10 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python scripts/train_piperx_goal_push.py --config configs/piperx_goal_push_dev.yaml --seed 20260818 --output results/piperx_goalpush_ppo_policy_v2_new
.venv/bin/python scripts/evaluate_piperx_goal_push.py --config configs/piperx_goal_push_dev.yaml --checkpoint results/piperx_goalpush_ppo_policy_v2_nominal_20260818T190900Z/checkpoints/step_000049152.pt --episodes 64 --seed-start 910000
PYTHONPATH=src .venv/bin/python scripts/run_piperx_gate1_live_viewer.py --policy-run results/piperx_goalpush_ppo_policy_v2_nominal_20260818T190900Z --checkpoint results/piperx_goalpush_ppo_policy_v2_nominal_20260818T190900Z/checkpoints/step_000049152.pt --host 0.0.0.0 --port 8799
```
