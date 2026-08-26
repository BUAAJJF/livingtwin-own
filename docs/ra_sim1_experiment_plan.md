# Phase RA-Sim-1 — pre-registration

*Written and committed before any result run of this phase. Every constant,
range, split, seed and threshold below is fixed at that commit. Deviations are
recorded in `docs/ra_sim1_results.md` under their own heading; this text is
not edited to match what happened.*

Start 2026-08-26 06:08 UTC. Budget 4 hours. Baseline commit `27ecb34`,
**364 tests passing**. GPUs 6 and 7 only — 4 and 5 are occupied by another of
the user's jobs, 0–3 are out of bounds.

**Phase RA-Sim-0 stands at RED and nothing here reopens it.** Its Gate R was
run and failed; its hidden target, its constants, its thresholds and its
`gate.json` are untouched.

---

## 1. The question

> Was RA-Sim-0's failure a property of *residual learning*, or of the
> particular parameterisation it used — a small, memoryless, additive
> correction on the command? Can a model with internal state, physical
> limits and a stable recursion reach the accuracy of the large-bound
> additive residual **and** the stability of the small-bound one?

RA-Sim-0 measured the thing this phase is reacting to: the additive residual's
99th-percentile correction sat exactly on its bound at every setting from 0.05
to 0.30 rad, it only beat a parameter fit once the correction averaged **3.3×
the commanded step**, and one validation run in fourteen went non-finite.
"Bigger bound" is not an answer; a different shape might be.

## 2. The model

`piper_push.actuator.StableActuator`. Per environment and per joint the model
keeps the actuator's own position command `w`, and learns how it chases the
controller's:

    error_t      = u_t - w_t
    a, r+, r-, b = net(GRU state, deployable history)
    delta_t      = clip( a * (error_t - b), -r- * dt, +r+ * dt )
    u_eff,t      = clip( w_t + delta_t, joint_lo, joint_hi )
    w_{t+1}      = u_eff,t

**Ordering.** The phase's specification writes `u_eff,t+1 = u_eff,t + delta_t`
— emit, then update. That form cannot be identity-initialised: with `a = 1`
and no rate limit it emits `u_{t-1}`, which is a one-step transport delay and
a change to the simulator. The update is therefore done inside the step and
its result emitted. This is the same recursion with the index shifted and it
is the only ordering under which "model off" and "model at initialisation" are
the same simulator. It is recorded here as a deliberate deviation and measured
in the audit rather than asserted.

**Architecture, frozen.**

| | |
|---|---|
| features | `[q, qdot, u, u_prev, q − u, u − w]`, 36 numbers |
| recurrent core | one `GRUCell`, hidden **64** |
| head | `Linear(64 → 24)`, weights `N(0, 0.05)`, zero bias, plus a `4 × 6` **gate** initialised to zero |
| parameters | **21,168**, one model (no ensemble) |
| control period `dt` | 0.02 s |

No ensemble, deliberately: RA-Sim-0 measured its four-member ensemble's spread
at **−0.28** correlation with the actual error, so the spread was worse than
useless and paying 4× for it is not justified. Uncertainty is out of scope for
this phase and is not claimed.

**Physical ranges, frozen.** Every coefficient is a sigmoid mapped into a
range chosen from the hardware, not from a fit:

| coefficient | range | why that range |
|---|---|---|
| `alpha` | [0.02, 1.0] | fraction of the tracking error closed per 20 ms; 1.0 lands on target within the period, 0.02 takes 2.5 s |
| `rate_pos`, `rate_neg` | [0.05, 4.0] rad/s | an actuator cannot slew its own target faster than the joint can turn; this arm's safety-shell trip speeds are 3.14–3.93 rad/s, rounded up. Learned **separately per direction**, which is what lets an asymmetry exist |
| `bias` | [−0.05, +0.05] rad | a directional offset in the error being chased. Backlash presents this way; nothing tells the model that |
| `u_eff` | `piper.SAFE_TARGET_CLIP` | the joint command range the action term already enforces |

**Identity at initialisation.** Each coefficient is written as *the identity
minus a gated deviation*:

    alpha = 1        − (1 − alpha_min) · g_a · sigmoid(z_a)
    rate  = rate_max − (rate_max − rate_min) · g_r · sigmoid(z_r)
    bias  =            bias_max · g_b · tanh(z_b)

with `g` a learnable scalar per coefficient and joint, initialised to zero. At
`g = 0` the coefficients are **exactly** 1, `rate_max` and 0, so the deviation
from a pure pass-through is **exactly 0.0 rad** — not 8.3 × 10⁻⁷ — while
`dL/dg` is proportional to `sigmoid(0) = 0.5` and is alive from the first
step. The coefficients are re-clamped into their registered ranges after the
gate, so a gate that overshoots cannot take one outside its window.

*This replaces the first version of this section, which set a head bias of 14
and is recorded in §11 as a deviation: it was exactly the trap it was trying to
avoid.*

**What it may see.** `q`, `qdot`, the commands the controller issued, their
difference, and the model's own state `w` — which is not a hidden quantity
because the model produced it. Forbidden and structurally unreachable: the
hidden target's effective command, its flank or lag state, its formula, its
constants, reward, success, the safety label, the object's pose, the critic.
Enforced by a single feature builder with a fixed signature and checked
statically in `tests/test_ra_sim1_actuator.py`.

**Not a re-encoding of the answer.** The model has no backlash term, no
deadband, no magnitude-dependent gain and no knowledge that the target is two
composed effects. It has one gain, two rate limits and one offset, all
state-dependent. If it works it works because a chasing recursion with memory
is the right general shape, not because the answer was written into it.

## 3. Training

| | |
|---|---|
| data | `train` (seed 7101) truncated to **3,000 control steps = 60 s** of arm time — the budget RA-Sim-0's ladder showed saturating; more data is not allowed to hide a model problem |
| validation | `val` (seed 7201), same truncation, for early stopping and nothing else |
| gradient bridge | RA-Sim-0's **frozen** nominal surrogate, `results/ra_sim0/model/surrogate.pt`, val NMSE 0.0128. Fitted on nominal-simulator transitions only, never on target data, and it appears in no result |
| windows | 80 steps, 16 of them burn-in, stride 40 |
| optimiser | Adam, lr 3e-4, grad-norm clip 1.0, 40 epochs, 12 batches of 256 windows per epoch |
| **seeds** | **0, 1, 2, 3, 4** — five model-training seeds, all reported |

**Loss**, with the weights frozen here:

    L = 1.0 * one_step
      + 1.0 * rollout_10
      + 1.0 * rollout_25
      + 1e-3 * effective_command_rate
      + 1e-4 * hidden_state_norm
      + 1e-3 * coefficient_smoothness

Transition terms are MSE in the surrogate's normalised label space. The rate
penalty is `(delta / (rate_max * dt))^2`, the hidden term `|h|^2 / hidden`,
the smoothness term the squared step-to-step change of the four coefficients
in their own normalised units. Supervision comes only from observable
transitions; the hidden target's effective command is never a label.

## 4. Data

| split | seed | shapes | envs × steps | role |
|---|---|---|---|---|
| `train` | 7101 | 0–2 | 64 × 3000 used | fit |
| `val` | 7201 | 0–2 | 64 × 3000 used | early stopping, nothing else |
| `test`, `test_amp`, `test2`, `valh` | 7401/7402/7403/7301 | 3–4 | existing | **development evidence only** — already looked at in RA-Sim-0 and no longer capable of supporting a claim |
| **`test3`** | **7601** | **3–4** | 64 × 6000 | **sealed.** Collected after this document is committed, opened **once**, after the checkpoints and configs are frozen and their SHA-256 recorded |

`test3` uses a new rollout seed, held-out shape classes 3–4, and an action
perturbation of σ = 0.09 clipped at ±0.25 — between `train`'s 0.05/0.15 and
`test_amp`'s 0.12/0.35 — plus the standard 10% scripted probe, whose
multi-frequency sweep is where the direction reversals live.

**Rules for opening it.** After `test3` is scored, no retraining and no
hyper-parameter change. The **only** permitted debugging is to the measurement
and plumbing code, and if any such fix is made, `test3` is **voided**, the
deviation is recorded, and a new sealed set is registered and collected. There
is no silent re-run.

## 5. Replay protocol

Unchanged from RA-Sim-0's corrected protocol, which is the thing worth keeping
from that phase:

* one fresh environment per replay pass — only its first reset draws the
  recording's own objects, and `shape_match_at_t0` is recorded and must be
  1.000;
* no termination terms during a replay, so no candidate is scored on its
  luckiest survivors;
* the window is bounded at both ends by the recording's own first reset;
* every candidate scored on the identical usable set;
* the cross-build reproducibility floor measured and reported beside the
  result rather than subtracted silently.

## 6. Candidates

1. `nominal` — the simulator as shipped
2. `param_fit` — RA-Sim-0's best parameter fit: delay 3, response 0.377,
   deadband 0.0175, damping 1.25
3. `residual_b005` — RA-Sim-0's additive residual at its own bound
4. `residual_b040` — the same at bound 0.40, the accurate-but-suspect control
5. **`actuator`** — this phase's model, five seeds
6. `oracle` — the hidden target installed; an upper bound, not a method

Failed and non-finite runs are reported, not deleted.

## 7. Stress tests, pre-registered

Beyond the paired replays, each run at 64 environments in the augmented
simulator with terminations **enabled** (unlike the replay), so the safety
shell is a measurement rather than an obstacle:

| id | what |
|---|---|
| S1 | 3,000-step closed-loop rollout of the frozen vision policy — long-horizon hidden-state behaviour |
| S2 | scripted actions at the action space's full ±1.0 amplitude |
| S3 | direction reversal every 2 control steps (12.5 Hz square wave) |
| S4 | scripted ramps that drive every joint onto its command limit and hold |
| S5 | half the batch reset asynchronously every 137 steps, the other half never |

Recorded for each: NaN/Inf count, command-limit violations, effective-command
rate distribution, `qvel`/`qacc` tails, safety-shell (`over_speed`)
terminations, hidden-state norm, and throughput/memory.

## 8. Gates

**G1 — accuracy.** Against `param_fit`, on the sealed `test3`: one-step NRMS
down ≥ **30%**, 10-step ≥ **25%**, 25-step ≥ **20%**; improvement on the
held-out shape and direction-reversal content; a paired cluster bootstrap 95%
interval (clustered on environment, 10,000 resamples) supporting each.

**G2 — real MJWarp.** Every reported number from a forward rollout in the real
simulator. The surrogate appears in the training loop and nowhere else.

**G3 — stability.** Across every formal and stress rollout: **zero** NaN,
**zero** Inf, **zero** command-limit violations beyond the model's own clip
(which is a design feature, counted separately), no illegal state jump, a
bounded hidden state, and `qvel`/`qacc` tails not significantly worse than
`param_fit`. A single non-finite formal seed fails G3; the seed is not deleted.

**G4 — physical plausibility.** *Not* "the correction must be smaller than one
action increment" — RA-Sim-0 showed that constraint is what made the problem
unsolvable. Instead: the effective command stays inside the joint command
range; its rate of change stays inside the pre-registered actuator range;
accumulated lag larger than a single action increment **is allowed** provided
it is produced by the stable internal state; and no accuracy may come from an
instantaneous large reverse command, which the rate clip makes unreachable and
which is measured as `max |delta|` per step.

**G5 — generalisation.** G1–G4 hold on the sealed `test3`, not only on the
development splits.

**G6 — compute.** Throughput reported at 64, 256 and 512 environments. Target:
≤ **20%** loss at 256. Exceeding it is YELLOW even if accuracy passes.

## 9. Verdict rules

* **GREEN** — G1–G6 all pass. A policy phase becomes permissible.
* **YELLOW** — accuracy and stability pass, compute or part of generalisation
  does not. One engineering-optimisation pass allowed; no PPO.
* **RED** — accuracy fails, non-finite states appear, or it only works on data
  already looked at. The synthetic-actuator-residual line stops and real
  hardware measurement takes priority.
* **INCOMPLETE** — no trustworthy formal result inside the budget.

No gate is relaxed, the hidden target is not changed, and RA-Sim-0 is not
reinterpreted as a success.

---

## 11. Deviations

**D1 — the initialisation, found during training and before any result.**
The plan originally reached identity with a head bias of 14, so that
`sigmoid(14) = 1 − 8.3e-7`. That is identity to within the simulator's own
noise and it is also **untrainable**: `sigmoid'(14)` is 8.3 × 10⁻⁷ too, so
every gradient reaching the head is scaled by it. Measured on the first
launch: forty epochs moved the one-step training loss from 7.71 to 7.65, and
7.6 is the nominal simulator's own number — the model never left the identity.

The gated form in §2 replaces it: exactly identity at `g = 0`, with a live
gradient on `g`. Nothing else changed — the architecture, the four ranges, the
loss, the seeds, the budget and every gate threshold stand as written. The
five training seeds were restarted from scratch under the new
parameterisation, and no result from the first launch is used anywhere.
