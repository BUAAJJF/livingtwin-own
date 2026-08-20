# MuJoCo single-cube planar pushing PPO baseline

This repository is an independent, state-based, single-object PPO baseline. It
does not import LivingTwin calibration code, checkpoints, posterior models, or
planning code.

The scientific protocol is frozen in `PROTOCOL.md` and the YAML files under
`configs/`. Formal training is forbidden until `scripts/run_preflight.py`
reports PASS for checks A--F.

Typical commands:

```bash
.venv/bin/pip install -e .
.venv/bin/python scripts/run_preflight.py --config configs/smoke.yaml
.venv/bin/python scripts/train.py --config configs/smoke.yaml --seed 1101 --output results/smoke_seed_1101
```

## Piper X goal-conditioned continuous pushing

The Piper X task uses the laboratory `orcabotics/piperx-mjlab` repository as
a pinned git submodule. Clone this repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/orcabotics/LivingTwin.git
cd LivingTwin
git submodule update --init --recursive
python3.10 -m venv .venv
.venv/bin/pip install -e .
```

The Piper X asset path in `configs/piperx_goal_push_dev.yaml` and the frozen
Policy V2 config is the repository-relative `piperx-mjlab` directory. The
submodule must be checked out at the commit recorded by the superproject.

Train a nominal Policy V2 run:

```bash
.venv/bin/python scripts/train_piperx_goal_push.py \
  --config configs/piperx_goal_push_dev.yaml \
  --seed 20260818 \
  --output results/piperx_goalpush_ppo_policy_v2_new
```

Evaluate a checkpoint:

```bash
.venv/bin/python scripts/evaluate_piperx_goal_push.py \
  --config configs/piperx_goal_push_dev.yaml \
  --checkpoint results/piperx_goalpush_ppo_policy_v2_nominal_20260818T190900Z/checkpoints/step_000049152.pt \
  --episodes 64 --seed-start 910000
```

Run the live Viser viewer (the checkpoint and `FROZEN_CONFIG.json` are kept at
the same paths under `results/`):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_piperx_gate1_live_viewer.py \
  --policy-run results/piperx_goalpush_ppo_policy_v2_nominal_20260818T190900Z \
  --checkpoint results/piperx_goalpush_ppo_policy_v2_nominal_20260818T190900Z/checkpoints/step_000049152.pt \
  --host 0.0.0.0 --port 8799
```

The viewer includes the known-success single episode, continuous Policy V2
play, and the frozen staged-controller inspection cases.
