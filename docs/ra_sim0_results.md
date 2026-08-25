# Phase RA-Sim-0 — results

**Verdict: RED.** The parameter axes really cannot express the mismatch
(Gate P GREEN, and by a wide margin). The residual, as this phase specified
it, is **27% worse** than the best parameter fit at one step rather than 30%
better (Gate R RED). Stage 6 was therefore **not run** — the phase says stop,
and the gate was not moved.

There is a specific, measured reason, and it is not "residual learning does not
work": the correction's output bound was saturated at its 99th percentile in
every evaluation. A post-hoc diagnostic at three times the bound, run on
validation data only and reported below as a diagnostic and **not** as a gate
result, halves the training loss and overtakes the best parameter fit. The
next phase's design follows from that; this phase's verdict does not.

*Run 2026-08-25 16:21 UTC to 23:55 UTC, 7 h 34 m, from commit `623e2a9`
(301 tests) to this commit (344 tests). GPUs 4–7 of `shen-teacher` only; GPUs
0–3 were never touched. mjlab 1.6.0 / rsl_rl 5.4.2 / MuJoCo 3.11.0 /
mujoco-warp 3.11.0 / warp 1.16.0 / torch 2.13.0+cu13.0.*

---

## 1. What was executed, and what was not

| stage | status |
|---|---|
| 0 — injection audit | **formal**, `docs/residual_injection_audit.md` |
| 1 — freeze the hidden target | **formal**, `docs/ra_sim0_experiment_plan.md`, committed before any result run |
| 2 — reward-free data | **formal**, 6 splits, 1.03 GiB, manifest in `results/ra_sim0/data/manifest.json` |
| 3 — parameter calibration | **formal but truncated** — see the deviations below |
| 4 — residual training | **formal**, 3 seeds |
| 5 — accuracy gate | **formal**, 2 held-out splits, real MJWarp |
| 5b — bound diagnostic | **exploratory**, validation only, chosen after seeing Stage 5 |
| 6 — policy | **stopped by Gate R**. Not run, not estimated, not reported. |

### Deviations from the pre-registration, all recorded before the report

1. **The second training-data seed (7102) was killed** at 2,000 of 15,000
   steps. Four collectors and a calibration were sharing the four cards and it
   had slowed to 487 env-steps/s, which would have cost 28 minutes for a
   second copy of a split that already carries 960k transitions. The residual
   is trained on seed 7101 alone.
2. **`valh`, `test` and `test_amp` were collected at 6,000 steps (120 s) rather
   than 15,000 (300 s)**, for wall-clock, decided before any of them was read.
   The accuracy gate scores 800 of those steps.
3. **The damping sweep was truncated.** `damping = 1.0` completed all 30
   evaluations; 0.75, 1.25 and 1.5 were killed part-way and their traces are
   lost, because the script writes its JSON only at the end. Damping is not an
   axis the hidden target touches, and the 1.0 trace already shows the fit's
   ceiling.
4. **The coordinate descent ran one round, not three**, at 400 replay steps
   rather than 600.
5. **The data-budget ladder (10 / 30 / 60 / 180 / 300 s) was not executed.**
   Stage 4's `--budget-steps` exists and is tested, and the four smaller
   budgets would have cost about 20 minutes; the phase's 7:30 rule — no new
   training — arrived first. Gate R fails at the *largest* budget, so no
   smaller one could have passed it, but that is an argument and not a
   measurement and is not reported as one.
6. **Stage 3's winner was re-selected.** Its coordinate descent ran under a
   mask that was later found to be wrong (§2.2); its winner was re-scored on
   validation under the corrected mask and beaten by a broad-randomisation
   draw. The Gate R comparator is the re-selected one, which is the *stronger*
   baseline — see §4.

---

## 2. Two bugs the phase found in its own method

Both were found by looking at numbers that should have been boring. Both would
have produced a publishable-looking result.

### 2.1 MJWarp is not bit-reproducible, so "unchanged" needs a floor

Two builds of the same environment, same seed, same command stream, disagree
by a mean of 3.6 × 10⁻⁷ rad after eight steps and 7.6 × 10⁻³ rad after sixty;
the *maximum* after sixty steps is 1.37 rad, which is a different grasp. Every
"nothing changed" claim in the audit is therefore a ratio against that floor.
The residual at identity initialisation sits at **1.03×** it.

### 2.2 A replay that lets its candidates die scores the worst ones on survivors

The first accuracy run looked like this:

| candidate | one-step NRMS | **usable fraction** | n |
|---|---|---|---|
| nominal | 1.479 | **0.009** | 457 |
| param_joint | 0.848 | **0.014** | 719 |
| broad_dr | 0.660 | **0.022** | 1,123 |
| oracle | 0.123 | **0.886** | 45,319 |

A candidate whose command path is badly wrong trips the safety shell within a
few steps, resets, and draws a fresh object — so it was being scored on the
handful of environments that happened to survive, and the oracle on all of
them. It made the *worse* simulators look better. Replays now run with no
termination terms at all, and the window is bounded at both ends by the
recording's own first reset, because that draws it a new object too. Every
candidate in §3 and §4 is scored on the identical window: **usable fraction
0.945, n = 48,340** one-step samples, for all of them.

Both fixes are pinned by tests (`tests/test_replay.py`).

---

## 3. Gate P — the parameters really are not enough

**GREEN, on all four criteria.**

Thirty configurations were searched on validation — transport delay in
{0,1,2,3} steps, response scale in [0.30, 1.00], deadband in [0, 0.030] rad,
low-pass in {off, 40, 20, 10, 5} Hz, gripper rate in [0.5, 1.5], damping at
1.0 — plus three broad-randomisation draws.

| simulator | one-step NRMS on `test` | what it is |
|---|---|---|
| nominal | **2.228** | as shipped |
| best single axis | 2.028 | one step of transport delay |
| Stage 3's joint winner | 1.886 | delay 1, response 0.60, deadband 0.008 |
| **best parameter fit** | **1.502** | delay 3, response 0.377, deadband 0.0175 |
| **oracle** | **0.137** | the hidden target itself |

An NRMS of 1.0 is *as wrong as predicting that nothing moved*. The whole
parameter family gets from 2.23 to 1.50 — a third of the way — and the oracle
gets to 0.137. **The oracle removes 91% of what the best parameter fit leaves
behind.** Ratio 0.091 against a criterion of 0.70.

The error that survives the fit is structured, not noisy:

* **lag-1 autocorrelation of the one-step error: 0.985.** White residuals
  would mean the simulator is merely imprecise. This is a simulator that is
  wrong in a way a state variable could predict.
* **RMS error varies 1.82× across commanded-step-magnitude bins**, and
  non-monotonically: 0.0368 rad below 2 mrad of command, 0.0202 at 2–5 mrad,
  0.0330 at 20–40 mrad. A constant response scale cannot bend like that.
* The signed mean error flips sign between the smallest bin (+0.0167 rad) and
  every other bin (−0.002 to −0.003). That sign flip is the backlash band: for
  commands smaller than the band the target does not move at all and the fit
  overshoots; for larger ones it lags and the fit undershoots. One scalar
  cannot be on both sides of zero.
* Error after a command reversal, 0.0266 rad, against 0.0302 without one —
  the reversal split as defined does *not* separate them, and is reported that
  way rather than dropped.

Criterion 4, leakage: every candidate is scored on a split generated under its
own seed with held-out shape classes (3–4, never trained on), resynchronised
from the same recorded state, in the same window, against the same recording.
The env's own draw was verified to reproduce the recording's objects exactly —
64 of 64 shape classes and `q₀` to 0.0 — before any of it ran.

---

## 4. Gate R — the residual is not more accurate

**RED.** Three residual training seeds, two held-out splits, real MJWarp
rollouts, the same window for every candidate.

| split | horizon | best parameter fit | residual (3 seeds) | ratio, 95% CI | required |
|---|---|---|---|---|---|
| `test` | 1 step | 1.5023 | 1.9136 | **1.274** [1.254, 1.294] | ≤ 0.70 |
| `test` | 10 steps | 1.2976 | 1.4677 | **1.131** [1.124, 1.138] | ≤ 0.75 |
| `test` | 25 steps | 0.6133 | 0.7099 | **1.158** [1.148, 1.167] | ≤ 0.80 |
| `test_amp` | 1 step | 1.5186 | 1.9306 | **1.271** [1.251, 1.291] | ≤ 0.70 |
| `test_amp` | 10 steps | 1.3204 | 1.4945 | **1.132** [1.124, 1.140] | ≤ 0.75 |
| `test_amp` | 25 steps | 0.6207 | 0.7142 | **1.151** [1.141, 1.160] | ≤ 0.80 |

Every interval is on the wrong side of 1.0. The two splits — one held out by
object shape, one by action amplitude and reversal rate — agree to the third
decimal place, so this is not a split that went badly.

Against the **nominal** simulator the residual does help: one-step NRMS 2.228
→ 1.913, a 14% reduction; RMS error 0.0435 → 0.0373 rad; and in the reversal
region where the backlash actually lives, 0.0403 → 0.0344 rad. It is learning
the right kind of thing. It does not reach the best parameter fit *anywhere*,
including in that region, where the fit is at 0.0266.

One number separates the oracle from everything else and is worth recording:
the lag-1 autocorrelation of the one-step error is 0.98 for nominal, for both
parameter fits and for all three residuals, and **0.837** for the oracle; the
magnitude-bin swing is 1.6–1.8× for all of them and **1.36×** for the oracle.
The oracle is the only simulator whose remaining error looks like noise rather
than like an unmodelled mechanism. Neither the parameter fit nor the residual
has touched the mechanism — they have only made it smaller.

**Physical sanity.** All finite, no state jumps, peak joint velocity 17.1 rad/s
against the nominal simulator's 17.6 and the oracle's 19.0 — the residual is
the *quietest* candidate, not one that injects energy. Throughput 2,347
env-steps/s against 2,454 nominal (4.4% at 64 environments; 15.8% at 256, from
the audit). GPU 146 MiB against 118.

**The ensemble's uncertainty is not calibrated.** Correlation between the
members' disagreement and the actual error: **−0.28**, the same on both splits
and all three seeds. Where the ensemble disagrees most, the error is
*smallest*. A spread that anti-predicts error is worse than no spread at all,
and nothing downstream should have used it.

### Why: the bound was saturated the whole time

`DELTA_MAX` was frozen at **0.05 rad** in the plan, argued from the target's
backlash band. The measurement:

| | mean \|Δ\| | **p99 \|Δ\|** | bound |
|---|---|---|---|
| residual, all seeds, both splits | 0.038 rad | **0.0499–0.0500 rad** | 0.05 |

The 99th percentile is the bound, to four decimal places, in all six runs. The
correction spends a large fraction of its steps clipped.

Arithmetic that was in the plan and should have been taken more seriously: at
a commanded step of `s`, the target completes `0.85 / (1 + 1.6 s / 0.04)` of
it. The command path's slew ceiling permits steps up to about 0.06 rad, and at
0.06 rad the target's shortfall is 0.045 rad *before* the backlash band of up
to 0.022 rad is added. The worst case is about 0.067 rad and the bound is
0.05.

---

## 5. Diagnostic, not a gate: what a wider bound does

**Selected after seeing the Stage 5 result, trained on the same data, scored on
`val` only. It is reported here because the phase requires deviations and
diagnoses to be recorded, and it does not change any verdict above. `test` was
opened once, in §4, and is not reopened.**

One residual, same architecture, same three loss terms, same 30 epochs, same
seed, `DELTA_MAX = 0.15` instead of 0.05:

| | one-step | 10-step | 25-step | surrogate val loss |
|---|---|---|---|---|
| best parameter fit (val) | 1.4907 | 1.2603 | 0.6011 | — |
| residual, bound 0.05 (val) | 1.9176 | 1.4314 | 0.6990 | 14.99 |
| **residual, bound 0.15 (val)** | **1.3981** | **1.1926** | **0.5949** | **7.54** |
| oracle (val) | 0.1155 | 0.7150 | 0.2930 | — |

At three times the bound the residual **overtakes the best parameter fit on
validation** — 1.398 against 1.491 — and its training loss halves. Its p99
\|Δ\| is 0.1499: **still exactly the bound.** The constraint has not been
relieved, only moved.

This says the phase's negative result is a statement about one hyper-parameter
and not about the method. It does not say the method would pass Gate R, and
nothing here should be quoted as if it did: this is one seed, on validation,
chosen after the fact, and the thresholds were set for `test`.

---

## 6. What was measured, in numbers

**Data.** Six splits, 1.03 GiB, none in git; `results/ra_sim0/data/manifest.json`
carries every shape, seed, budget and SHA-256.

| split | seed | domain | shapes | envs × steps | budget | SHA-256 (12) |
|---|---|---|---|---|---|---|
| `train` | 7101 | target | 0–2 | 64 × 15000 | 300 s | `8d5715fa3d80` |
| `val` | 7201 | target | 0–2 | 64 × 15000 | 300 s | `53f3d2a33fb8` |
| `valh` | 7301 | target | 3–4 | 64 × 6000 | 120 s | `1fb75bfff068` |
| `test` | 7401 | target | 3–4 | 64 × 6000 | 120 s | `290993e30b26` |
| `test_amp` | 7402 | target | 3–4 | 64 × 6000 | 120 s | `25192397d3f6` |
| `nominal` | 7501 | **nominal** | 0–2 | 64 × 15000 | 300 s | `9c8b5f2f0614` |

Each is 70% frozen-policy rollout, 20% the same policy under a pre-registered
action perturbation, 10% a scripted multi-frequency sweep with the gripper
cycling. The frozen policy is
`piperx_pick_place_vision/2026-08-22_17-15-09_f3/model_1500.pt`, SHA-256
`fe9ecd12f5895192…`, the same checkpoint Phases WM1-A and WM1-B used.

**Models.**

| | size | data | training |
|---|---|---|---|
| nominal transition surrogate | 75,276 params | 949,029 nominal transitions | 60 epochs, val NMSE **0.0128** |
| residual, one member | 18,822 params | 16,297 windows of 80 steps | 30 epochs |
| residual ensemble (4) | **75,288 params** | same | 194–215 s per seed on one GPU |

The surrogate is fitted on nominal-simulator transitions only, frozen the
moment it is fitted, and appears in no result: every number in §3, §4 and §5
comes from a real MJWarp rollout.

**Compute.** 6 data collections, 30 + 3 calibration evaluations, 4 residual
trainings, 34 accuracy runs. About 6 GPU-hours across four cards.

---

## 7. Summary

1. **Injection point.** L2, the action/command wrapper — the stateful,
   batched, per-environment stage `piper_push.actions` already owned between
   the policy and the servo. It gained a `command_hooks` tuple. Pre-step
   generalized force (L4) was tested and works but was not needed; no post-step
   state overwrite is used anywhere.
2. **Why the simulator was not rewritten.** It did not need to be. MuJoCo keeps
   integrating, resolving contact and enforcing limits; the only thing that
   changes is the number the servo is asked to hold.
3. **The hidden target.** Per-joint asymmetric gear backlash (0.006–0.022 rad
   flanks, different flank wider on different joints) composed with a
   current-limited lag whose completed fraction falls with the size of the
   commanded step. Frozen and committed before any result run; a test pins
   the constants.
4. **How much the parameters explain.** A third. Thirty configurations get
   one-step NRMS from 2.228 to 1.502; the oracle reaches 0.137, so the best
   parameter fit leaves **91%** of the explainable error on the table, with a
   lag-1 autocorrelation of 0.985 and a 1.82× swing across command-magnitude
   bins.
5. **Did the residual reduce held-out multi-step error?** **No.** 1.274×,
   1.131× and 1.158× the best parameter fit at 1, 10 and 25 steps, every
   interval on the wrong side of 1.0, identically on both held-out splits. It
   *does* beat the nominal simulator by 14% and is the only candidate that
   improves the reversal and deadband regions.
6. **Does it hold in a real MJWarp rollout?** Yes — that is the only way
   anything here was measured. The surrogate exists solely to carry a gradient
   and reports nothing.
7. **Cost.** 75,288 parameters, 960k target transitions, 194–215 s of offline
   training per seed, 4.4% throughput at 64 environments and 15.8% at 256.
8. **Policy training.** **Not run.** Gate R failed and the phase says stop.
   The gate was not moved, and G3's arithmetic was not re-derived to make it
   pass.
9. **Target policy, safety, retention.** Not measured. Stage 6 did not run.
10. **Total run time.** 7 h 34 m of the 8 h budget.
11. **Verdict: RED.**
12. **Is residual-augmented MuJoCo worth a real robot?** **Not yet, and not on
    this evidence.** But the failure is diagnosed rather than mysterious: the
    correction was clipped at its 99th percentile in every run, and at three
    times the bound it overtakes the best parameter fit on validation while
    *still* saturating. The next phase's first experiment is the bound sweep
    this one did not have time for — pre-registered, on validation, with `test`
    kept shut — and only then a fresh held-out gate. Nothing about a camera, a
    real robot or a policy should be attempted before that number exists.

---

## 8. What would have to be different

**Re-select the bound properly.** Sweep `DELTA_MAX` over {0.05, 0.10, 0.15,
0.25, 0.40} on validation, pick by validation loss, and only then open a fresh
test split. The saturation statistic — p99 \|Δ\| against the bound — is the
thing to watch, and it should be reported for every candidate.

**Fix the ensemble or drop it.** A spread that correlates −0.28 with error is
not uncertainty. Four members trained on one data order and differing only in
initialisation is too little diversity; bootstrap the windows per member, or
replace the ensemble with a single model and a learned variance head.

**Run the data-budget ladder.** It costs 20 minutes and it answers a question
this phase asked and did not answer: how much target-domain time a structural
correction needs.

**Then, and only then, Stage 6.** A simulator that is not yet more accurate
than the parameter fit has nothing to offer a policy, and running the policy
comparison now would only measure PPO seed noise — which is exactly what
Phase WM1-A's eight-seed extension has already shown this project can mistake
for a result.
