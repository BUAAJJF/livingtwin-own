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

