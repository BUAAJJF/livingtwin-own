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

TODO — generated from the manifest.

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

TODO.

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

## 9. Limitations

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

## 10. Exact commands

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
