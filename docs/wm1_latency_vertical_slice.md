# Phase WM1-A — a reward-free calibration loop for 60 ms of observation delay

*Status: in progress. Sections marked TODO are waiting on runs that are still
going; nothing in the finished sections will be rewritten to match them.*

## 0. What this phase asked

Phase WM0 established that a persistent simulator mismatch on this task is
real, identifiable from reward-free rollouts, and recoverable by simulation PPO
when the parameter is *known*. It did not close a loop. This phase does, on one
axis and one target value:

> Freeze the deployed vision policy. Hide a target domain with
> `obs_latency_steps = 3` — 60 ms at the task's 50 Hz. Infer the simulator
> parameter from reward-free observation–action history alone, then fine-tune
> in simulation under the inferred distribution and measure whether the policy
> recovers in the target domain without losing the domain it came from.

The three questions the gate turns on are separable and are reported
separately: **can the domain be identified at all** from data a robot could
record (G1), **does adaptation guided by that inference recover the loss**
(G2, G3, G4), and **does the decision-aware part of the method earn its place**
against a classical trajectory-matching alternative (G5).

Everything here is simulation. There is no real robot in this phase and no
claim about one.

## 1. Setup and provenance

| | |
|---|---|
| repository | `yf/bolt`, frozen at tag `wm0-green` = `4351a8a` |
| policy | `logs/rsl_rl/piperx_pick_place_vision/2026-08-22_17-15-09_f3/model_1500.pt` |
| sha256 | `fe9ecd12f5895192538fec1a3df163d7147dab881a35f6db00790da365f00636` |
| task | `Mjlab-Pick-Place-PiperX-Vision`, 50 Hz control, 500 Hz physics |
| stack | mjlab 1.6.0, mujoco 3.11.0, mujoco-warp 3.11, rsl_rl 5.4.2, torch 2.13.0+cu130 |
| evaluation protocol | 512 environments × 2400 control steps = 6.83 arm-hours, three process repeats |

Every result file carries its own commit, checkpoint sha, argv and library
versions; the tables below are generated from those files rather than
transcribed.

### 1.1 One delay implementation, and what changed when the other was deleted

Phase WM0 measured `obs_latency_steps` through a ring buffer inside the camera
observation term, and found only afterwards that mjlab already ships the same
thing — `ObservationTermCfg.delay_min_lag` / `delay_max_lag`, which build a
`DelayBuffer` and serve the term's output from `t − lag`. WM0 recorded that in
a comment rather than acting on it, because the sweep was already running.

WM1 needs a *distribution* over the lag rather than a constant, so keeping two
implementations stopped being merely untidy. The ring buffer is gone;
`perturb.apply_session_mismatch` now sets the native fields, and
`src/piper_push/latency.py` adds the per-environment categorical assignment on
top of mjlab's buffer.

`tests/test_latency.py` pins the two to the same output sequence for every lag
WM0 measured. They differ in one place a unit test cannot price: for three
control steps after an episode boundary, mjlab ramps the served lag up as its
history refills, where the ring buffer held the frame from the reset. So the
target domain was re-measured under the new implementation, same protocol, same
three evaluation seeds.

| | WM0 (ring buffer), 3 repeats | WM1 (mjlab DelayBuffer), 6 repeats | rate ratio |
|---|---|---|---|
| nominal, obj/min | 55.86 | **55.94** [55.73, 56.15] | |
| zero-shot at lag 3, obj/min | 42.19 | **42.12** [41.79, 42.45] | |
| nominal, trips/arm-hour | 2.29 (47 events / 20.5 h) | 3.12 (128 / 41.0 h) | 1.36 [0.92, 2.01], p = 0.12 |
| zero-shot, trips/arm-hour | 8.69 (178 / 20.5 h) | 9.64 (395 / 41.0 h) | 1.11 [0.90, 1.37], p = 0.34 |

Throughput reproduces to within a tenth of an object per minute in both
domains, so the WM0 gate anchors carry forward. The trip rates run higher in
this phase in both conditions and neither difference is significant once the
counts' overdispersion is accounted for; section 8.2 works out what that is and
is not evidence of, and the gate reports its safety criterion against both the
inherited threshold and one re-derived from this phase's own anchors.

The nominal condition is the control on the whole comparison: its code path is
byte-identical across the change — `apply_session_mismatch` returns before
touching anything when no axis is active, and the evaluator never calls
`apply_latency_prior` — so any difference in it is not the delay
implementation.

### 1.2 What counts as a result, and what does not

Everything in `results/wm1_latency/` is a formal run at the protocol above,
with one exception that is labelled as such in its own file. Nothing else in
this report is a number from a shortened run.

**Formal.** Every evaluation is 512 environments × 2400 control steps with
three process repeats at the fixed seed bank `20260823, 31415926, 27182818`,
and every adaptation configuration is measured at three training seeds
(`42, 20260824, 31415927`) and reported both per-seed and pooled. The dataset
is 35 full-length rollouts; the model fits are the full ones; the scoring pass
covers all 320 sessions of the test and calibration splits.

**Plumbing, not a result.** `results/wm1_latency/plumbing_check.json` is 24
environments × 40 steps and answers one yes/no question about whether the
delayed observation is the frame it should be (section 2.4). It is not a
measurement of anything and no number in this report comes from it.

**Smoke runs that produced no committed file.** A 8-environment × 120-step
dataset generation, a one-member one-epoch dynamics fit, and a two-sessions-
per-domain scoring pass, all written outside the repository and deleted. They
exist in this account because they are how three crashes were found before
they could cost a wave of GPU time — a keyword argument the observation
manager passes that the camera term did not accept, a hidden state left on the
CPU, and a score vector with seven entries reaching a five-way posterior.

## 2. The dataset

One session is one environment's contiguous timeline: the unit a real
deployment gives you, one arm for one stretch of time. Files are
`{split}__lag{L}__seed{S}.pt`; the tensors are gitignored (about four gigabytes
of fp16) and `results/wm1_latency/data/manifest.json` is not.

### 2.1 Channels

Recorded, and all of them computable on a real robot from its own sensors and
its own policy weights:

| channel | shape | what it is |
|---|---|---|
| `enc` | (T, N, 100) | the actor's encoder output — the tensor its GRU consumes. The first 36 columns are the normalised one-dimensional observation group, the last 64 the spatial-softmax encoding of the (delayed) depth image. |
| `hidden` | (T, N, 256) | the actor's GRU hidden state **before** the step, so `(enc_t, hidden_t)` reproduces `a_t` exactly |
| `proprio` | (T, N, 13) | six joint positions, six joint velocities, gripper position |
| `action` | (T, N, 7) | the commanded joint targets as the policy emitted them |
| `servo` | (T, N, 1) | gripper position minus its target — what a real drive reports as current |
| `done` | (T, N) | episode boundary |

Not recorded, and not nameable: reward, return, success, placement counts,
safety-shell trips, object pose, mass, friction, contact flags, the privileged
critic's value, and the latency itself.
`wm_data.assert_deployable` is called by the loaders and by `wm_data.gather`,
so a batch containing the label cannot be assembled at all, and
`tests/test_wm_data.py` checks that eight privileged names raise rather than
resolving to an empty feature.

`done` is deployable — a robot knows when the task restarts — but it is
excluded from every feature set anyway, because reset cadence is a consequence
of the domain rather than evidence about the plant. It is measured separately
instead, by a classifier that sees episode boundaries and nothing else
(section 5.3).

### 2.2 Splits

| split | files | sessions | length | arm-hours | GB | shapes | episodes |
|---|---|---|---|---|---|---|---|
| `cal` | 5 | 80 | 15000 steps (300 s) | 6.7 | 0.96 | train | 689 |
| `calh` | 5 | 80 | 15000 steps (300 s) | 6.7 | 0.96 | holdout | 635 |
| `test` | 5 | 160 | 15000 steps (300 s) | 13.3 | 1.92 | holdout | 1287 |
| `train` | 10 | 1280 | 1500 steps (30 s) | 10.7 | 1.53 | train | 294 |
| `val` | 5 | 320 | 1500 steps (30 s) | 2.7 | 0.38 | train | 77 |
| `valh` | 5 | 320 | 1500 steps (30 s) | 2.7 | 0.38 | holdout | 69 |

All 6 leakage checks pass (no failures).

### 2.3 Leakage controls

Six checks, run by `scripts/wm_manifest.py`, which fails rather than writes:

1. the label is not a listed channel;
2. no two splits share a generation seed;
3. the test sessions' object shape classes are disjoint from training — by
   construction, with zero probability on the training classes at generation
   time, not by a filter applied to the log afterwards, so the policy's
   recurrent state never carried a training object either;
4. every file within a split has the same length and environment count, so the
   shape of the tensor a method is handed says nothing about which domain it
   came from;
5. the five data budgets are nested prefixes of one 300 s session rather than
   five separately sampled datasets;
6. the five candidate lags are balanced within every split.

Windows are 25 control steps (0.5 s), never cross an episode boundary, and are
non-overlapping at inference time — overlapping windows would count the same
control step several times and sharpen a posterior without adding evidence.

### 2.4 Is the camera actually late?

Everything above assumes the observation a running environment hands the
policy is the frame from `lag` control steps ago. Unit tests cannot check
that: they see the buffer's arithmetic and the term's bookkeeping, not the
observation the policy receives.

`scripts/check_latency.py` installs a second, *undelayed* copy of the camera
term beside the delayed one in the same environment, so the comparison is
between two observations of one simulation rather than two runs of a simulator
that is not bitwise reproducible. It then asks two questions per environment,
and the second is the one that matters: does the delayed group match the
reference at its own assigned lag, and does it match at **no other** shift? A
term that delayed everything by a constant, ignoring its per-environment
assignment, would pass the first and fail the second.

With 24 environments drawn from a mixture over lags {0, 3, 4}
(assignment [7, 0, 0, 8, 9]) over 36 comparable steps:
**24/24** environments match at their own lag and at no other.

## 3. The estimators

All of them read the same channels and get the same data budget. Their
hyper-parameters, weights and temperatures are chosen on simulation validation
domains and never on the target, on reward, or on task performance.

| | what it is | what it needs offline |
|---|---|---|
| **B0** | the prior. No target data: the answer is the source domain, `δ(0)`. The policy that ships if calibration is not worth doing. | nothing |
| **B1a** | cross-correlation between the commanded action and the joint response — the textbook latency estimator. Included *because* it should fail: an observation delay does not move the actuator's response to a command. | nothing |
| **B1b** | ridge regression from the **image** half of the encoder latent to joint positions `θ` steps earlier, scored on a held-out half of the same session. The depth image contains the arm, so a delayed image is a picture of where the arm was. | nothing |
| **B2** | a small GRU classifier from raw history to `q(θ)`. Amortised system identification: all the simulator knowledge is spent offline and inference is one forward pass. | one classifier fit |
| **B3** | trajectory matching: proprioception and servo-error NLL under the parameter-conditioned dynamics model. The channels a classical system-identification pipeline uses. | the dynamics ensemble |
| **B4** | the actor latent's NLL plus the action the frozen policy would have taken had the next observation been the predicted one. | the dynamics ensemble |
| **DA** | `λs·S_state + λz·S_latent + λa·S_action`, weights and temperature fitted on the calibration split. | the dynamics ensemble |

B1b is the baseline the rest of the pipeline has to justify itself against: it
needs no simulator, no training and no world model.

A note on B1b's construction that decides whether it measures anything: the
encoder latent's first 36 columns *are* the normalised proprioception the
policy was handed. Regressing joint positions on the whole latent fits
perfectly at every candidate lag. Only the image half is used.

## 4. The parameter-conditioned dynamics model

Nothing here reconstructs a depth image. The policy's encoder already turns the
image into a 100-wide latent and that latent is what the policy's decision is a
function of, so that is what is predicted:

```
F_ψ(z_t, p_t, a_t, θ, h_t) → N(z_{t+1}), N(p_{t+1} − p_t), N(e_{t+1})
```

Four ensemble members, each a two-layer MLP into a 192-wide GRU with three
Gaussian heads, 8-dimensional embedding of `θ`. Residual heads, so an untrained
model is the identity rather than the origin. Learned, clamped log-variances,
because a homoscedastic model ranks candidate domains by squared error in
whichever channel has the largest units. Normalisation statistics from the
training split alone. Members differ in initialisation *and* in which 80% of
the windows they see.

Loss: one-step NLL on all three heads over a 16-step horizon after an 8-step
burn-in, plus an 8-step open-loop rollout on the latent at weight 0.5. The
open-loop term feeds the model's own predictions back; the actions stay the
real ones, because they are in the log and a robot has them.

118,602 training windows from ten rollouts (two generation seeds × five
candidate lags, 128 environments × 1500 steps each, training shape classes).
Fitting takes 5.8 minutes on one RTX 6000D.

### 4.1 Held-out prediction

| | z (latent) | p (proprio) | e (servo) | multi-step z |
|---|---|---|---|---|
| `val` (training shapes, unseen seed) | −221.30 | −32.05 | −1.36 | −146.07 |
| `valh` (held-out shapes) | −222.19 | −31.75 | −1.31 | −146.34 |

Per-window NLL in normalised units, summed over channel dimensions, constant
dropped. The two rows agreeing says the model did not fit the objects.

### 4.2 Which channel carries the domain

The table that decides what the decision-aware weighting has left to do: for
validation windows of known latency, the NLL under each candidate. The diagonal
has to be the cheapest entry in its row or no posterior built on this model can
work.

`val`, latent NLL — **5/5 correct**:

| true \ candidate | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **0** | **−220.9** | −203.8 | −155.5 | −86.4 | −29.7 |
| **1** | −211.4 | **−223.6** | −206.2 | −161.3 | −113.1 |
| **2** | −175.6 | −208.9 | **−223.2** | −208.6 | −177.9 |
| **3** | −129.7 | −172.0 | −209.4 | **−222.5** | −212.1 |
| **4** | −83.1 | −123.6 | −172.4 | −205.4 | **−214.8** |

`val`, multi-step latent rollout — **5/5 correct**, and with wider margins
still (lag 0 costs −141.5 under the right candidate and **+240.7** under lag 4).

`val`, proprioception — **3/5 correct**, and the margins are about half a nat
out of thirty-two:

| true \ candidate | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **0** | −32.00 | **−32.08** | −31.72 | −31.29 | −30.68 |
| **1** | −31.98 | **−32.31** | −32.16 | −31.87 | −31.34 |
| **2** | −31.43 | −31.96 | **−32.05** | −31.96 | −31.57 |
| **3** | −30.74 | −31.41 | −31.66 | **−31.72** | −31.46 |
| **4** | −29.80 | −30.52 | −30.85 | **−31.02** | −30.87 |

`val`, servo error — **1/5 correct**. There is no signal here at all.

Every one of these repeats on `valh`, the held-out shape classes, with the same
counts: 5/5, 5/5, 3/5, 1/5.

**This is the phase's clearest result, and it is a result about what
trajectory matching can see.** An observation-side mismatch leaves almost no
trace in the plant's state trajectory — the arm still moves where it is told —
and is written all over the perceptual channel that the policy's decision
actually depends on. A calibration objective built on proprioception and servo
error, which is what a classical system-identification pipeline would build, is
reading the one part of the log that does not know about this domain.

### 4.3 The permutation control

The same architecture, the same data, the same number of updates, with `θ`
shuffled within each batch so the conditioning carries no information. Its
candidate grid is flat to four significant figures:

| true \ candidate | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **0** | −249.94 | −250.01 | −249.87 | −250.02 | −249.99 |
| **3** | −251.02 | −251.06 | −250.95 | −250.97 | −251.14 |

0/5 argmins correct on both `val` and `valh`; the spread across a row is
0.1 nats against the real model's 100-plus.

## 5. Reward-free identification

160 test sessions — 32 per domain, five domains, every one of them an object
shape class the models never saw. The estimator is handed the deployable
channels and nothing else; the label is read afterwards, by the scoring code,
to grade an answer that has already been produced.

### 5.1 At the budget the adaptation stage uses

Budget 60 s of one arm (120 windows), 160 sessions across five domains.

| method | top-1 | balanced | mass on truth | entropy (bits) | ECE | target top-1 | target mass |
|---|---|---|---|---|---|---|---|
| B0 prior | 0.200 | 0.200 | 0.200 | 0.00 | 0.800 | 0.000 | 0.000 |
| B1a command→joint | 0.156 | 0.156 | 0.200 | 2.32 | 0.044 | 0.000 | 0.200 |
| B1b image→proprio | 0.438 | 0.438 | 0.301 | 2.01 | 0.122 | 0.219 | 0.281 |
| B2 classifier | 0.981 | 0.981 | 0.959 | 0.15 | 0.018 | 0.969 | 0.915 |
| B3 state matching | 0.600 | 0.600 | 0.421 | 1.47 | 0.213 | 0.969 | 0.505 |
| B4 latent+action | 1.000 | 1.000 | 1.000 | 0.00 | 0.000 | 1.000 | 1.000 |
|   ablation: latent only | 1.000 | 1.000 | 1.000 | 0.00 | 0.000 | 1.000 | 1.000 |
|   ablation: action only | 0.544 | 0.544 | 0.495 | 1.11 | 0.148 | 0.844 | 0.651 |
| **DA** (fitted weights) | 1.000 | 1.000 | 1.000 | 0.00 | 0.000 | 1.000 | 1.000 |
| *control*: shuffled labels | 0.194 | 0.194 | 0.200 | 2.32 | 0.006 | 0.000 | 0.200 |
| *control*: shuffled θ | 0.050 | 0.050 | 0.200 | 2.32 | 0.150 | 0.094 | 0.200 |
| *control*: episode boundaries only | 0.200 | 0.200 | 0.200 | 2.32 | 0.000 | 1.000 | 0.200 |

DA weights at this budget: state 0.00, latent 0.17, action 0.83.

Read the **balanced** column, not the two on the right. A method that answers
"3" for every session scores 1.000 on target top-1 and 0.200 balanced, and the
episode-boundary control does exactly that — which is the reason the control is
there.

Five things in that table.

**B1a is at chance, as it should be.** The classical latency estimator —
cross-correlate the command against the joint response — scores 0.156, below
1/5. An observation delay does not move the actuator's response to a command,
so the quantity it measures is the same in all five domains. This is a result
about the coverage of a standard tool, not a failure of the implementation:
the same estimator would be the right one for `action_latency_steps`, which
Phase WM0 measured as the single most damaging axis on this task.

**The applicable analytic baseline works, and is not enough.** B1b — ridge
from the image half of the encoder latent to joint positions `θ` steps earlier
— reaches 0.438 balanced with no simulator, no training and 3 ms of compute.
That is well clear of every control and it is a long way from solved: its
confusion matrix is diffuse and biased towards long lags, and it puts only
0.281 of its mass on the truth in the target domain.

**Trajectory matching resolves the middle of the range and collapses the
ends.** B3 — proprioception and servo-error NLL under the same dynamics model
the other methods use — scores 0.600, which is exactly 3 of 5:

| B3, true \ said | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **0** | 1 | **31** | 0 | 0 | 0 |
| **1** | 0 | **32** | 0 | 0 | 0 |
| **2** | 0 | 0 | **32** | 0 | 0 |
| **3** | 0 | 0 | 0 | **31** | 1 |
| **4** | 0 | 0 | 0 | **32** | 0 |

It reads lag 0 as lag 1 and lag 4 as lag 3. The second of those is the one
that matters: on 32 of 32 sessions from a domain with *80 ms* of delay, a
state-matching posterior reports the 60 ms target. Its benign-domain false
positive rate for the target class is 25%, against 0% for the latent-based
methods. This is section 4.2 showing up at session level — the plant's state
trajectory barely knows about an observation-side mismatch.

**The latent settles it, and the action-consequence term does not.** The
decomposition is the point:

| | balanced accuracy at 60 s |
|---|---|
| `S_state` alone (B3) | 0.600 |
| `S_action` alone | 0.544 |
| `S_latent` alone | **1.000** |
| `S_latent + S_action` (B4) | 1.000 |
| fitted combination (DA) | 1.000 |

DA's fitted weights are state 0.00, latent 0.17, action 0.83 — and those
numbers are *not* importance scores. The three components are not on a common
scale: the latent NLL spans a hundred nats between candidates where the action
score spans a fraction of one, so a weight of 0.17 on the latent still
dominates the sum. The ablations are the honest decomposition, and they say
the identification comes from predicting the policy's **perceptual latent**,
not from the action the policy would have taken given that latent.

That is a partial result for the decision-aware framing, and it should be
stated as one. Scoring a candidate domain *in the policy's own representation*
rather than in the plant's state is what beats trajectory matching here, by
0.400 balanced accuracy and by 25 points of benign false-positive rate. The
specific `S_action` term — push the predicted latent through the frozen actor
and compare the action — adds nothing on this axis, because there is nothing
left to add once the latent is exact.

**A single half-second window is already most of the way there.** The B2
classifier reads one 25-step window and gets 69% of them right (section 5.3);
by 10 seconds of pooled evidence it is at 0.988 balanced, and the model-based
methods are at 1.000. Identification is not the expensive part of this loop.

### 5.2 Data budget

Balanced accuracy over five domains, by seconds of one arm. The phase was
asked for 10 s and up; those all saturate, so three smaller budgets are added.
They cost nothing — the per-window scores are cached and a budget is a prefix
of them — and they are the numbers that matter for a loop whose premise is an
hour of robot time.

| method | 1 s | 2 s | 5 s | 10 s | 30 s | 60 s | 180 s | 300 s |
|---|---|---|---|---|---|---|---|---|
| B0 prior | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 |
| B1a command→joint | 0.206 | 0.169 | 0.150 | 0.181 | 0.169 | 0.156 | 0.188 | 0.194 |
| B1b image→proprio | 0.225 | 0.250 | 0.300 | 0.431 | 0.475 | 0.438 | 0.463 | 0.556 |
| B2 classifier | 0.800 | 0.919 | 0.975 | 0.988 | 0.963 | 0.981 | 1.000 | 1.000 |
| B3 state matching | 0.406 | 0.431 | 0.506 | 0.575 | 0.594 | 0.600 | 0.600 | 0.600 |
| B4 latent+action | 0.975 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
|   ablation: latent only | 0.975 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
|   ablation: action only | 0.287 | 0.312 | 0.325 | 0.475 | 0.519 | 0.544 | 0.544 | 0.569 |
| **DA** (fitted weights) | 0.975 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| *control*: shuffled labels | 0.200 | 0.181 | 0.138 | 0.175 | 0.206 | 0.194 | 0.200 | 0.200 |
| *control*: shuffled θ | 0.119 | 0.106 | 0.094 | 0.050 | 0.031 | 0.050 | 0.013 | 0.000 |
| *control*: episode boundaries only | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 |

**One second of arm time — two half-second windows — puts the decision-aware
posterior at 0.975 balanced accuracy and 0.972 of its mass on the truth.** By
five seconds it is exactly right on all 160 sessions. Identification is not
what makes this loop expensive; section 6 is.

Two smaller readings from the same table. The action-consequence term does add
something, but only where the latent has not yet settled: at 1 s, DA carries
0.972 of its mass on the truth against latent-only's 0.968, and by 5 s the
difference is gone because there is nothing left to improve. And B1a does not
improve with any amount of data — 0.206 at 1 s, 0.194 at 300 s — which is what
an absent signal looks like next to B1b's weak-but-real 0.225 → 0.556.

### 5.3 Controls

Three, each measuring a different way the result could be an artefact.

| control | what it removes | balanced accuracy at 60 s |
|---|---|---|
| shuffled training labels (B2) | any information linking history to domain, keeping the architecture and the update count | 0.194 |
| shuffled `θ` (dynamics ensemble) | the conditioning, keeping the model's ability to predict | 0.050 |
| episode boundaries only | everything except reset cadence | 0.200 |

The first two are at or below chance at every budget. The shuffled-`θ` ensemble
is *below* chance and falls further with more data (0.050 at 10 s, 0.000 at
300 s), which is what a consistently-wrong-and-increasingly-confident ranking
looks like when a flat score vector is amplified by hundreds of windows; its
candidate grid is flat to four significant figures (section 4.3).

The same three, measured one window at a time rather than pooled over a
session, on the simulation validation splits the classifier was selected on:

| | `val` (training shapes) | `valh` (held-out shapes) |
|---|---|---|
| B2 classifier, single 0.5 s window | **0.694** | **0.685** |
| trained on shuffled labels | 0.194 | 0.191 |
| episode boundaries only | 0.197 | 0.200 |

A single half-second of history is right about seven times in ten across five
domains, and neither control is distinguishable from the 0.200 floor.

The third control answers a question the others cannot. Reset cadence *is* a
consequence of the domain — a policy that fails more often restarts more often
— and a robot can see it, so it would have been an easy accidental win. A
classifier that sees the episode-boundary channel and nothing else scores
0.200: it learns to answer "3" for everything. None of the identification in
this section is reset cadence.

### 5.4 Calibration, and whether it transfers to unseen objects

Temperature and weights are fitted on `cal` — 80 sessions of *training* shape
classes. Applied unchanged to `calh`, which is the same simulation domains on
*held-out* shape classes:

| method | balanced (test, held-out shapes) | balanced (calh) | ECE (calh) |
|---|---|---|---|
| DA / B4 / latent-only | 1.000 | 1.000 | 0.000 |
| B2 classifier | 0.981 | 1.000 | 0.019 |
| B3 state matching | 0.600 | 0.600 | 0.185 |
| B1b image→proprio | 0.438 | 0.400 | 0.157 |

Nothing degrades across the shape populations. The methods that are
well-calibrated on the calibration split stay well-calibrated on objects it
never contained.

### 5.5 What one session costs

Measured on a single 60 s session, three timed repetitions after a warm-up,
one RTX 6000D:

| | seconds |
|---|---|
| world-model scores, 5 candidates × 4 ensemble members × 120 windows | 0.152 |
| B2 classifier | 0.0024 |
| B1b ridge | 0.0030 |
| B1a cross-correlation | 0.0005 |

Inference is not a cost. The online budget is dominated by the 60 s of arm
time and by the PPO adaptation that follows.

## 6. Posterior-guided adaptation

TODO.

## 7. Gate

TODO.

## 8. Statistics

### 8.1 Two metrics, two treatments

Throughput is a ratio of two large totals — objects placed over arm-minutes,
over 512 environments and three process repeats per condition, nine when the
three training seeds are pooled. Its interval is a **hierarchical bootstrap**:
resample repeats, then resample environments within each resampled repeat, and
recompute the pooled ratio. That carries both the process-level spread — which
is what differs between repeats of an identical command, because MuJoCo-Warp is
not run-to-run reproducible ([mujoco_warp#562]) — and the environment-level
spread. The statistic is the pooled ratio and not the mean of per-environment
rates, which would weight an environment that lived four seconds like one that
ran the whole rollout.

Safety-shell trips are **counts**, and small ones. A nominal 512 × 2400 run
produces about twenty events in 6.83 arm-hours. Reporting a standard deviation
over three repeats of a rate built from twenty events is reporting Poisson
noise as a measurement. So trips are pooled as counts over exposure, given an
exact Poisson interval (bisection on the closed-form chi-square CDF for even
degrees of freedom — Wilson–Hilferty is off by a factor of two at the two
degrees of freedom a zero-trip condition would need), and checked for
overdispersion with Pearson's statistic across repeats. Where the repeats
disagree by more than a common Poisson rate allows, the interval is widened by
the square root of the dispersion rather than the disagreement being averaged
away.

### 8.2 The trip rate did not regress, and reading it as a mean would have said it did

This matters enough to state on its own, because G3 is a threshold on exactly
this quantity.

The nominal condition's code path did not change between Phase WM0 and this
phase — `apply_session_mismatch` returns before touching anything when no axis
is active, and `apply_latency_prior` is not called by the evaluator at all. Its
throughput reproduced to within a tenth of an object per minute. Its trip rate
did not: WM0 measured 2.29/h and the re-measurement 3.15/h, which reads as a
37% regression.

Read as counts, over six repeats:

| | events | exposure | rate | per-repeat rates |
|---|---|---|---|---|
| WM0 nominal, 3 repeats | 47 | 20.5 h | 2.29/h | 2.49, 2.05, 2.34 |
| WM1 nominal, same command, 6 repeats | 128 | 41.0 h | 3.12/h | 2.93, 3.37, 2.20, 3.66, 4.25, 2.34 |
| rate ratio | | | **1.36, 95% CI [0.92, 2.01], p = 0.12** | |

Nothing that can be distinguished from noise happened. Two things are worth
taking from it rather than one.

First, three repeats of a twenty-event count cannot resolve a 40% change, and
WM0's reported standard deviation on this metric (±0.22 on 2.29) understated
its spread by a lot: repeating the same protocol six times gives per-repeat
rates from 2.20 to 4.25 per arm-hour, a factor of 1.9.

Second, that spread is **larger than Poisson**. The dispersion — Pearson's
statistic over the repeats divided by its degrees of freedom — is 1.36 for
nominal and 1.44 for zero-shot, so the repeats disagree by about 20% more in
standard deviation than counting noise alone allows. Intervals on this metric
are therefore quasi-Poisson throughout, and it is the dispersion correction
that turns the nominal comparison from p = 0.036 into p = 0.12. Reporting the
uncorrected number would have been reporting a regression that is not there.

The consequences: the anchors in this phase get six repeats rather than three;
the gate's trip criterion is evaluated on nine evaluations pooled across
training seeds — about 61 arm-hours — rather than on three; and G3 is reported
against both the inherited threshold and one re-derived from anchors measured
in this phase, since a threshold built from WM0's trip rates and applied to
this phase's is comparing across the thing that moved.

### 8.3 Comparisons

Conditions are compared **paired on the evaluation seed**: the run-to-run term
is shared between a pair and cancels, which is what makes three repeats usable
at all. Where several comparisons are made against one baseline the p-values
are Holm-corrected. The primary evidence is the interval, not the p-value;
p-values appear only to order the comparisons before correction.

Session-level identification results are over 32 independent sessions per
domain, and the reported figure is **balanced accuracy** over the five domains
rather than raw accuracy, so that a method which always answers with one
domain cannot be read as being at chance when it is not.

## 9. Failure diagnosis

The phase specification listed the failure modes in the order they should be
checked, because each one points at a different thing to fix and running the
next stage before the previous one is sound wastes GPU-days. Where this phase
landed on each:

| symptom | what it would mean | this phase |
|---|---|---|
| reward-free history cannot separate the domains | the data is not exciting enough, or the model is wrong; do not run PPO | **not hit.** 1.000 balanced accuracy on 160 held-out-shape sessions from 5 s of arm time, against three controls at or below chance |
| posterior wrong where cross-correlation is right | the learned inference is broken | **not hit**, and the reverse: B1a is at chance and B1b reaches 0.438 where the learned posterior reaches 1.000 |
| posterior right, target does not recover | simulator adaptation is the failure, not inference | TODO |
| target recovers, retention collapses | the posterior is too narrow; mix more source prior | TODO |
| trajectory matching indistinguishable from decision-aware | the decision-aware novelty is not established | **not hit at the identification level** — 0.600 against 1.000 balanced accuracy, and a 25-point benign false-positive gap — but see section 5.1 on *which* part of the decision-aware score is doing it |
| the oracle is unstable across training seeds | fix the oracle before evaluating anything against it | TODO |

## 10. Limitations

### 10.1 What I expected and did not find

Five things this phase went in expecting, and what happened to them.

**The action-consequence term was supposed to be the discriminative one.** It
is the part that makes the score "decision-aware" in the sense the direction is
named for: push the predicted latent through the frozen actor and ask what the
policy would have done. It is not what identifies the domain here. Latent NLL
alone is at ceiling; the action term alone reaches 0.544 balanced accuracy,
barely above state matching's 0.600. The claim that survives is narrower than
the one I set out to test, and section 5.1 states the narrow one.

**The fitted weights looked like they said the opposite.** DA's calibration fit
puts 0.83 on the action term and 0.17 on the latent, which reads as an
importance ranking and is not one — the components are on wildly different
scales, and 0.17 of a hundred-nat spread dominates 0.83 of a fractional one.
The ablations are the decomposition; the weights are not. I nearly wrote the
weights up as the finding.

**B1a was specified as "the strong simple baseline for a latency problem",**
and for an *action* latency it would be. For an observation latency it is at
chance, because the actuator's response to a command does not know that the
camera is late. Included and reported flat rather than quietly dropped.

**The Phase WM0 trip anchors did not reproduce as tightly as the throughput
anchors.** Nominal ran 36% high on a code path that did not change. Six repeats
and a dispersion correction put that at p = 0.12 and it is not a regression —
but WM0's reported ±0.22 on 2.29 trips/hour was an understatement of that
metric's spread, and this phase reports the safety criterion against two
threshold sets because of it.

**One "oracle" was the unadapted policy, and its exit status was clean.** The
adaptation runner picked its evaluation checkpoint with `ls -1v | tail -1` over
every timestamped directory matching the run's tag. A duplicate of the seed-42
oracle — started by an overlapping plan before the run lock existed, killed
after it had written its first checkpoint — left a directory whose name sorts
*after* the real one and whose newest file is `model_1500.pt`, which is where
fine-tuning starts. The run reported 42.80 objects/min in the target domain and
56.13 in the nominal one, which are the zero-shot and nominal figures to two
decimal places, because it was the zero-shot policy. It was caught by reading
those two numbers rather than by anything in the pipeline. The checkpoint is
now selected by parsed iteration number and asserted against the requested
budget, the affected evaluations are in `quarantine/`, and the run was redone.

**I nearly published that regression.** Without the overdispersion correction
the same comparison is p = 0.036. The repeats disagree by 20% more in standard
deviation than Poisson allows, so the quasi-Poisson interval is the correct
one; the uncorrected p-value would have been a false alarm about a code change
that did not happen.

### 10.2 Scope

The honest boundaries of what this phase establishes.

**One axis, one value, one direction.** `obs_latency_steps = 3` and nothing
else. Phase WM0 found five axes that clear their own uncertainty; the reason to
start with this one is that it is the largest and the cleanest, not that it is
representative. Section 4.2 in particular — proprioception barely identifying
the domain while the latent identifies it decisively — is a statement about an
*observation-side* mismatch. A plant-side mismatch such as servo damping would
be expected to come out the other way round, and WM1-B is where that gets
tested rather than assumed.

**Simulation only.** There is no real robot in this phase. The "target domain"
is a simulator configured differently from the one the policy trained in, and
every claim about recovery is a claim about that. Nothing here shows that a
real 60 ms camera pipeline produces the same signature, and the calibration
loop has never been run on data from hardware.

**The identification depends on the policy that collected the data.** The
signature the world model reads is not the plant's alone; it is the plant as
driven by *this* frozen policy, whose behaviour under delay is part of what
makes the domain visible. A different policy would leave a different
signature, and the dynamics model is conditioned on this one's statistics. That
is fine for the deployment story — the policy that will be adapted is the
policy that collects the data — but it means the model is not a reusable model
of the robot.

**A risk head was specified and is not here.** The optional deployable safety
score `C_obs(history)` was not implemented. The reason is that on this axis it
could not have been measured: the latent and action scores already identify the
domain at ceiling on every test session, so a third score cannot move the
posterior, and a head trained on simulator safety labels would add a way to be
wrong without adding a way to notice. It is the right tool for a *tail-only*
mismatch — WM0's `servo_damping_scale = 0.75` costs 4.5% of throughput and
multiplies trips by 89 — and that is where it belongs.

**Held-out shapes, not held-out everything.** Test sessions use object shape
classes the models never saw, generated with zero probability on the training
classes. They do not vary the camera, the table, the bin geometry, or the
task. A method that survives an unseen object is not thereby a method that
survives an unseen scene.

## 11. Exact commands

Every command below was run from the repository root on the training server,
with `MUJOCO_GL=disable` and the mjlab environment's `lib` on
`LD_LIBRARY_PATH`. Provenance — commit, checkpoint sha256, argv, library
versions — is written into each result file by the script that produced it.

```bash
# 0. freeze
git tag -a wm0-green -m "..."          # 4351a8a

# 1. the delay implementation change, re-measured
scripts/wm1_equivalence.sh 0
scripts/wm1_anchor_extra.sh 3

# 2. the dataset: 25 rollouts, six shards
for i in 0 1 2 3 4 5; do scripts/wm1_collect.sh $GPU $i 6 & done
for i in 0 1 2;       do scripts/wm1_collect_cal.sh $GPU $i 3 & done
python scripts/wm_manifest.py

# 3. the models
python scripts/wm_train.py --members 4 --epochs 6 --stride 16 --device cuda:7
python scripts/wm_train.py --members 4 --epochs 6 --stride 16 --shuffle-theta \
    --device cuda:7
python scripts/wm_classifier.py --device cuda:7

# 4. inference
python scripts/wm_posterior.py --device cuda:7

# 5. adaptation
python scripts/wm1_adapt_plan.py --stage anchor --out .../anchor.json
python scripts/wm1_adapt_plan.py --stage screen --out .../screen.json
for i in 0 1 2 3 4 5; do scripts/wm1_adapt.sh $GPU $i 6 .../anchor.json & done
python scripts/analyze_wm1.py --dirs results/wm1_latency/adapt ...
python scripts/wm1_choose_alpha.py
python scripts/wm1_adapt_plan.py --stage formal --alpha $ALPHA --out .../formal.json
for i in ...; do scripts/wm1_adapt.sh $GPU $i N .../formal.json & done

# 6. the verdict
python scripts/analyze_wm1.py --dirs results/wm1_latency/adapt \
    results/wm1_latency/equivalence --json results/wm1_latency/analysis.json
python scripts/wm1_timings.py
python scripts/wm1_gate.py --json results/wm1_latency/gate.json
```

[mujoco_warp#562]: https://github.com/google-deepmind/mujoco_warp/issues/562
