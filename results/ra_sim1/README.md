# `results/ra_sim1/` — Phase RA-Sim-1

| path | what it is |
|---|---|
| `model/` | five stable-actuator checkpoints, one per training seed, plus `manifest.json` with their SHA-256 and validation losses |
| `frozen_sha256.txt` | the five checkpoints **and the sealed split**, hashed before `test3` was opened |
| `accuracy/` | 20 replay runs: 10 candidates × `test2` (development) and `test3` (sealed) |
| `stress/` | the five pre-registered stresses, for four arms |
| `throughput.json` | 64 / 256 / 512 environments, with and without the model |
| `gate.json` | G1–G6, machine-readable |

Raw trajectories are not in git. `test3` (seed 7601) was collected after
`docs/ra_sim1_experiment_plan.md` was committed and opened once, after the
checkpoints were frozen; `results/ra_sim0/data/manifest.json` carries its
seed, budget, composition and hash alongside the other splits.

The verdict is **RED** on G1 and G5. Phase RA-Sim-0's own RED verdict and its
`gate.json` are untouched.
