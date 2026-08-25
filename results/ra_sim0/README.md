# `results/ra_sim0/` — what is in each directory

Raw trajectories are **not** in git; `data/manifest.json` carries every split's
seed, budget, composition and SHA-256, and `scripts/ra_sim0_collect.py`
reproduces them. `model/manifest.json` lists all 26 residual checkpoints with
their bounds and hashes; ten of them are in the repository and sixteen are
points on a validation curve.

| directory | what it is | used by |
|---|---|---|
| `injection_audit.json` | Stage 0, the five injection layers and the reproducibility floor | `docs/residual_injection_audit.md` |
| `grid/` | **Stage 3.** 19 parameter configurations on `val`, one fresh environment each | §3 of the results |
| `accuracy3/` | **Stage 5.** The gate's accuracy runs on `test`, `test_amp` and `valh` | §4, Gate R |
| `bounds/`, `budget/` | exploratory sweeps on `val` | §6, §7 |
| `fresh/` | **RA-Sim-0b.** `test2`, collected after §6 and opened once | §8 |
| `gate.json` | every criterion, machine-readable | — |
| `model/` | the surrogate and the residual checkpoints | — |

## Superseded, kept deliberately

`accuracy/`, `calibration/`, `calibration2/` and `selection/` are the first
pass, produced before the three method bugs in §2 of the results were found.
**No number in the report comes from them.** They are kept because §2 is about
them, and because deleting the run that taught you something is how the lesson
gets lost:

* `accuracy/` was scored with terminations enabled, so the worst simulators
  kept 0.9% of their steps and the oracle 88.6% — a comparison between one
  arm's whole trajectory and another arm's luckiest environments.
* `calibration/` and `calibration2/` swept many configurations inside one
  environment, which only draws the recording's objects on its *first* reset.
  Under that, a 30-configuration coordinate descent found nothing better than
  changing nothing.
* `selection/` re-scored the survivors with fresh environments per
  configuration and is correct at one step, but its 10- and 25-step numbers
  came from a second reset in the same process and are not.
