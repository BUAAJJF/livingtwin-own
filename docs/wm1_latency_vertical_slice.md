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

| | WM0 (ring buffer) | WM1 (mjlab DelayBuffer) |
|---|---|---|
| nominal, obj/min | 55.86 (55.90, 55.54, 56.15) | TODO |
| zero-shot at lag 3, obj/min | 42.19 (41.85, 42.32, 42.40) | 42.22 (42.54, 42.14, 41.98) |
| nominal, trips/arm-hour | 2.29 (47 events / 20.5 h) | TODO |
| zero-shot, trips/arm-hour | 8.69 (178 events / 20.5 h) | 10.30 (211 events / 20.5 h) |

Throughput reproduces to within a tenth of an object per minute in the domain
that matters, so the WM0 gate anchors carry forward. The trip rates do not
reproduce as tightly, and section 8 works out what that is and is not evidence
of.

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
so a batch containing the label cannot be assembled at all; `--
tests/test_wm_data.py` checks that eight privileged names raise.

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

TODO.

## 9. Limitations

TODO.

## 10. Exact commands

TODO.
