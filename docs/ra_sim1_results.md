# Phase RA-Sim-1 — results

**Verdict: RED**, on G1 and G5, with G2, G3, G4 and G6 green.

The phase asked whether RA-Sim-0's failure came from the *parameterisation*
rather than from residual learning. The answer is **partly, and the part that
is left over is now located precisely**:

* **The structure fixed the multi-step problem.** On the sealed split the
  stateful actuator is **22.5% better than the best parameter fit at 10 steps
  and 21.8% better at 25**, where RA-Sim-0's small additive residual was 30%
  and 28% *worse*.
* **It did not fix the one-step problem.** It is **9.2% worse than the
  parameter fit at one step**, and no seed of five is better. G1 requires 30%
  better, so G1 fails on the horizon it fails on and misses the 10-step
  threshold by 2.5 points.
* **It bought the stability, and it cost nothing.** Zero non-finite states in
  twenty formal replays and the stress suite, every effective command inside
  the joint range, no single step larger than the pre-registered rate ceiling
  — and **no measurable throughput cost at 64, 256 or 512 environments**. Over
  the same period RA-Sim-0's additive residual at its own bound of 0.05 rad
  went **non-finite on the sealed split**.
* **The mechanism is identified.** The rate ceiling binds on 6–30% of
  joint-steps depending on seed. A one-step error is exactly the horizon at
  which an instantaneous adjustment is needed, and an instantaneous adjustment
  is exactly what this parameterisation forbids.

*Run 2026-08-26 06:08 UTC onward, from commit `27ecb34` (364 tests). GPUs 6
and 7 only — 4 and 5 were occupied by another of the user's jobs and 0–3 are
out of bounds. Phase RA-Sim-0 stands at RED and nothing here reopens it.*

---

## 1. The model

`piper_push.actuator.StableActuator`. Per environment and per joint it keeps
the actuator's own position command `w` and learns how it chases the
controller's:

    error = u - w
    a, r+, r-, b = net(GRU state, deployable history)
    delta        = clip( a * (error - b), -r- * dt, +r+ * dt )
    u_eff        = clip( w + delta, joint_lo, joint_hi )
    w'           = u_eff

**21,168 parameters**: one `GRUCell` (36 → 64), a `Linear(64 → 24)` head, and
a `4 × 6` gate. One model per seed, no ensemble — RA-Sim-0 measured its
ensemble's spread at −0.28 correlation with the actual error, so paying four
times for it was not justified and no uncertainty is claimed here.

Three properties come from the shape rather than from a penalty, and the tests
push on all three with the head and the gate driven to `N(0, 50)`:

* the effective command **cannot leave the joint range**, because the last
  line clips it there;
* **no single step can exceed** `rate_max · dt` = 0.08 rad, because the rate
  coefficients are mapped into rad/s and multiplied by the control period;
* a **large accumulated lag stays reachable** — the property RA-Sim-0's bound
  forbade and the reason that phase could not express the target at all.
  Measured here: up to 2.06 rad of standing lag, 42× a single commanded step.

**Deployable inputs only.** `[q, qdot, u, u_prev, q − u, u − w]`, where `w` is
the model's own state and therefore not a hidden quantity. A test walks the
module's syntax tree and asserts that its *code* — not its docstring, which
names them precisely because it is forbidding them — never reaches the hidden
target's flank state, its constants, the safety label, reward or success.

**Identity at initialisation is exact.** Each coefficient is *the identity
minus a gated deviation*, with the gate initialised to zero, so an untrained
model returns the command bit-for-bit: measured deviation **0.0 rad**, against
the 3.6 × 10⁻⁷ rad two identical MJWarp builds disagree by.

## 2. The gates

| gate | verdict | the number that decided it |
|---|---|---|
| **G1 accuracy (sealed `test3`)** | **RED** | 1.092× / 0.775× / 0.782× at 1, 10, 25 steps against required 0.70 / 0.75 / 0.80 |
| G2 real MJWarp | GREEN | every number is a forward rollout; the surrogate appears in the training loop and nowhere else |
| G3 stability | GREEN | **zero** non-finite states in the model under test, across 20 formal replays and the stress suite |
| G4 physical plausibility | GREEN | max single step 0.0800 rad against a 0.0800 ceiling; every effective command inside the joint range |
| **G5 generalisation** | **RED** | sealed and development agree closely — and both fail |
| G6 compute | GREEN | **−0.09%** throughput at 256 environments, against a 20% budget |

**Overall: RED.**

### G1 — accuracy, on the sealed split

`test3`: seed 7601, held-out shape classes 3–4, 64 × 6000 steps, action
perturbation σ 0.09 clipped at ±0.25. Collected after the plan was committed,
opened **once**, after the five checkpoints and the split itself were frozen
and their SHA-256 written to `results/ra_sim1/frozen_sha256.txt`. Five model
seeds; the interval is a **paired cluster bootstrap over environments**,
10,000 resamples, both arms re-indexed by the same draw.

| horizon | best parameter fit | actuator (5 seeds) | ratio | 95% CI | reduction | required |
|---|---|---|---|---|---|---|
| 1 step | 1.2847 | 1.4025 | **1.092** | [1.074, 1.118] | **−9.2%** | ≥ 30% |
| 10 steps | 0.9979 | 0.7732 | **0.775** | [0.761, 0.796] | **+22.5%** | ≥ 25% |
| 25 steps | 0.5071 | 0.3963 | **0.782** | [0.768, 0.803] | **+21.8%** | ≥ 20% |

The 25-step row is the interesting failure: its point estimate clears the 20%
threshold and its interval does not (upper bound 0.803 against 0.800). It is
reported as failing because that is what the pre-registered rule says, and the
margin is 3 parts in a thousand.

Every candidate on the same window, `shape_match_at_t0 = 1.000`, 47,574
one-step samples each:

| candidate | 1 step | 10 steps | 25 steps |
|---|---|---|---|
| nominal | 2.227 | 1.432 | 0.707 |
| RA-Sim-0 additive residual, bound 0.05 | 1.929 | *non-finite* | *non-finite* |
| **best parameter fit** | **1.285** | **0.998** | **0.507** |
| **stateful actuator** (mean of 5) | 1.403 | **0.773** | **0.396** |
| RA-Sim-0 additive residual, bound 0.40 | **0.648** | **0.560** | **0.315** |
| oracle | 0.123 | 0.033 | 0.022 |

Per seed at one step: 1.199, 1.366, 1.441, 1.451, 1.556. **None** is better
than the parameter fit's 1.285, so the one-step deficit is systematic and not
a bad seed.

### G5 — generalisation

The development split `test2` and the sealed `test3` agree to about one point:

| horizon | `test2` ratio | `test3` ratio |
|---|---|---|
| 1 step | 1.114 | 1.092 |
| 10 steps | 0.796 | 0.775 |
| 25 steps | 0.801 | 0.782 |

So the model generalises; it simply does not clear the bar on either. G5 is
red because it inherits G1's verdict on the sealed split, and the honest
reading is that this is a **capability** result and not a generalisation one.

### G3 — stability

**Zero non-finite states in the model under test**: five seeds × two splits ×
two replay passes, plus the five stress tests. Every stress ran at 64
environments in the augmented simulator with terminations **enabled**, so the
safety shell is a measurement rather than an obstacle.

| stress | what it is | non-finite | max \|qd\| | p99.9 \|qd\| | max \|qacc\| | shell events / arm-h |
|---|---|---|---|---|---|---|
| S1 | 3,000-step closed-loop rollout of the frozen policy | **0** | 3.90 | 2.95 | 285 | 436 |
| S2 | full ±1.0 action amplitude | **0** | 3.93 | 3.35 | 302 | 1,552 |
| S3 | direction reversal every two steps (12.5 Hz) | **0** | 2.39 | 0.57 | 59 | 0 |
| S4 | ramp onto every command limit and hold | **0** | 3.47 | 2.18 | 174 | 11 |
| S5 | half the batch reset every 137 steps, half never | **0** | 3.93 | 3.31 | 303 | 1,457 |

S5 is the cross-talk test and it returns the same numbers as S2 to two decimal
places, which is what "no state leaks between environments" looks like from
outside.

**The comparator was not stable.** RA-Sim-0's additive residual at its own
bound of 0.05 rad produced **non-finite states on the sealed split** in the
25-step pass. It is reported here and not counted against this phase's model:
a comparator's instability is a finding about the comparator. It is also the
strongest single piece of evidence that the parameterisation, not the size of
the correction, was RA-Sim-0's problem — the *small* residual is the one that
diverged.

### G4 — physical plausibility

The plan replaced RA-Sim-0's "the correction must be smaller than one action
increment" — the constraint that made the problem unsolvable — with four
conditions on the *shape* of the command:

| condition | measured |
|---|---|
| effective command inside the joint range | yes, by construction; the clip engaged 16,329 times in S4, which is the test that drives onto the limit and holds |
| single-step change within the actuator range | **max 0.0800 rad** against a 0.0800 ceiling, over 21 recorded blocks |
| accumulated lag may exceed one action increment | up to **2.06 rad**, 42× a commanded step, and it is produced by the state rather than by a jump |
| no accuracy from an instantaneous large reverse command | unreachable: the rate clip is what bounds the row above, and it binds on **6–30%** of joint-steps depending on seed |
| hidden state bounded | \|h\| = 8.00 = √64 in every run, which is a GRU's structural bound and not a measurement of restraint |

The last row deserves its own sentence. A GRU's hidden units live in [−1, 1],
so `|h| ≤ √hidden` whatever it learns; the state is bounded *because of the
cell*, and reporting 8.00 as evidence of good behaviour would be reporting the
architecture back to itself. What the number does say is that the state is
saturated — it sits at its ceiling in every run — which is a hint that 64
units are being used hard.

The coefficients the model actually chose, on the sealed split:

| seed | mean α | α range | bias range (rad) | rate-clipped | mean \|Δ\| (rad) |
|---|---|---|---|---|---|
| 0 | 0.298 | [0.020, 0.802] | [−0.048, +0.050] | 9.9% | 0.0232 |
| 1 | 0.175 | [0.020, 0.687] | [−0.043, +0.034] | 6.2% | 0.0183 |
| 2 | 0.465 | [0.020, 0.868] | [−0.050, +0.050] | 30.1% | 0.0253 |
| 3 | 0.333 | [0.020, 0.703] | [−0.050, +0.050] | 20.6% | 0.0230 |
| 4 | 0.152 | [0.020, 0.680] | [−0.040, +0.035] | 11.3% | 0.0168 |

Every seed learned a strongly lagging servo — a mean gain of 0.15–0.47 against
1.0 for a servo that lands on its target within the control period — which is
what the hidden target is, arrived at without being told. The mean correction
is 0.017–0.025 rad against a commanded step of 0.049: on average the effective
command moves *less* than the command, which is a lag rather than an override.
Three of five seeds pin the bias at its ±0.05 rad edge, so **that range binds**
and is the first thing a follow-up should widen.

### G6 — compute

| environments | nominal | with the actuator | loss |
|---|---|---|---|
| 64 | 2,004 env-steps/s | 2,021 | **−0.9%** |
| 256 | 4,949 | 4,954 | **−0.1%** |
| 512 | 7,959 | 7,971 | **−0.1%** |

Against a 20% budget. Twenty-one thousand parameters evaluated once per
control step do not register next to MuJoCo's solver; all three numbers are
inside run-to-run noise and two of them are nominally negative. This is the
one gate that passes with room to spare, and it passes the same way at every
size.

### The stress suite, every arm

Zero non-finite states everywhere, including the comparators — the additive
residual's divergence was in the *replay*, on the sealed split, not here.

| test | arm | max \|qd\| | p99.9 \|qd\| | max \|qacc\| | shell / arm-h |
|---|---|---|---|---|---|
| S1 | nominal | 3.92 | 2.91 | 269 | 2.8 |
| S1 | param_fit | 3.92 | 2.86 | 220 | 140 |
| S1 | **actuator** | 3.90 | 2.95 | 285 | 436 |
| S1 | residual b040 | 3.85 | 2.69 | 186 | 2.8 |
| S2 | param_fit | 3.92 | 3.54 | 361 | 1,149 |
| S2 | **actuator** | 3.93 | **3.35** | **302** | 1,552 |
| S4 | param_fit | 3.42 | 2.04 | 153 | 7.5 |
| S4 | **actuator** | 3.47 | 2.18 | 174 | 11 |
| S5 | param_fit | 3.93 | 3.45 | 357 | 1,037 |
| S5 | **actuator** | 3.93 | **3.31** | **303** | 1,457 |

G3 requires the `qvel`/`qacc` tails to be *not significantly worse* than the
parameter fit's. They are not: at full amplitude (S2) and under asynchronous
resets (S5) the actuator's p99.9 velocity and peak acceleration are **lower**
than the parameter fit's, and at the command limit (S4) they are within 14%.

The safety-shell column is not a stability number and should not be read as
one. Under the frozen policy (S1) the augmented simulator fires the shell 436
times per arm-hour against the parameter fit's 140 and the nominal
simulator's 2.8 — because a more strongly lagging simulator is a harder one to
fly, which is the point of building it. Zero of those events involved a
non-finite state.

---

## 3. What this says about RA-Sim-0

The phase's question was whether the earlier failure was about *residual
learning* or about *that residual's parameterisation*. Three measurements
separate the two:

1. **The stateful model beats the small additive residual by a wide margin at
   every horizon** — 1.40 against 1.93 at one step, 0.77 against non-finite at
   ten. Same data, same budget, same gradient bridge, same 60 s of arm time.
   The difference is the shape.
2. **The small additive residual is the one that diverged.** RA-Sim-0's bound
   was justified as a safety property; on a split it had never seen, the
   bounded model produced non-finite states and the rate-limited one did not.
   "Small" and "stable" turned out not to be the same word.
3. **But the rate limit is now the binding constraint**, exactly as the bound
   was before: max \|Δ\| sits on the ceiling, 6–30% of joint-steps are
   clipped, and the horizon the model fails on — one step — is precisely the
   one where an instantaneous adjustment is what is needed.

So the parameterisation was *part* of the problem and structure recovered
most of the multi-step gap, at no compute cost and with better stability than
either alternative. What is left is not a bug: it is that reproducing this
target's instantaneous response requires moving the effective command further
in one step than any physically-argued actuator rate allows. The unconstrained
0.40 rad residual still wins on accuracy at every horizon, and it wins by
doing the thing the hardware could not.

---

## 4. Deviations

**D1 — the plan's initialisation was untrainable, found before any result.**
Identity was originally reached with a head bias of 14, so that
`sigmoid(14) = 1 − 8.3e-7`. `sigmoid'(14)` is 8.3e-7 as well, so every
gradient reaching the head was scaled by it: forty epochs moved the one-step
training loss from 7.71 to 7.65, and 7.6 is the nominal simulator's own
number. Replaced by *the identity minus a gated deviation*, with the gate
initialised to zero — exactly identity, and `dL/dg` alive from the first step.

**D2 — the gate needs its own learning rate.** Adam moves a parameter by about
its learning rate per step; the gate must travel O(1) and the budget is 480
steps, so at 3e-4 it can move 0.14. Ten epochs took the loss from 8.21 to 8.12
with the gate under 0.1. The gate is its own parameter group at 5e-2. No
threshold, range, seed count or data budget moved, and the five seeds were
retrained from scratch under the final configuration.

**D3 — the evaluator's `--actuator` flag reached the provenance block and not
the simulator.** The first smoke run reported the nominal simulator's numbers
under the model's name: `build()` accepted the argument and never called
`apply_actuator`. Caught by the smoke run before any formal result, and now
pinned by a test that walks `build()`'s syntax tree and asserts that every
hook its signature accepts has its installer invoked.

**D4 — stress S5 had no command stream** and raised after its environment was
built, which the queue recorded as FAIL and which a careless reading would
have recorded as nothing. Four of eight stress runs did not run on the first
pass. Fixed, rerun in full, and a test now asks every pre-registered stress
for its stream before a GPU is involved.

**D5 — stress step counts.** S1 ran at the pre-registered 3,000 steps; S2–S5
ran at 1,500. Decided for wall-clock before any of them was read.

None of D1–D5 touched a gate threshold, the hidden target, the data budget or
the sealed split.

---

## 5. What would have to be different

**The bias range binds and should be widened first.** Three of five seeds pin
it at ±0.05 rad. It is the cheapest thing to test and it is the coefficient
that carries directional memory, which is where the hidden target's backlash
lives.

**The one-step horizon needs a different mechanism, not a looser rate.**
Raising the rate ceiling would recover one-step accuracy the same way
RA-Sim-0's bigger bound did, and it would give back the property this phase
was built to keep. A model that must be accurate *and* rate-limited needs the
instantaneous part of the response somewhere else — a feed-forward term on the
command that is itself bounded by the *joint's* range rather than by a slew,
or an actuator state that leads rather than lags.

**Do not collect more data.** RA-Sim-0's ladder saturated at 60 seconds of arm
time and this phase used exactly that. Nothing here is data-limited.

**Keep the protocol.** One environment per replay, no terminations during a
replay, the reproducibility floor measured beside every claim, per-environment
clustering for every interval, and a sealed split opened once. Two of the four
deviations above were found by that protocol rather than by luck.

**No policy training.** G1 failed and the plan says stop. The augmented
simulator is better than a parameter fit over ten and twenty-five steps and
worse over one; a policy comparison run now would measure which of those two
matters for PPO, which is a real question and not this phase's.

---

## 6. Summary

1. **Model.** A stateful actuator that chases the command instead of adding to
   it: one `GRUCell` (36 → 64), a `Linear(64 → 24)` head and a `4 × 6` gate,
   **21,168 parameters**, one model per seed, no ensemble.
2. **Deployable inputs only** — `[q, qdot, u, u_prev, q − u, u − w]`, where
   `w` is the model's own state. A test walks the module's syntax tree to
   assert its code never reaches anything else.
3. **No hidden-target labels.** Supervision is observable transitions through
   RA-Sim-0's frozen nominal surrogate, which reports nothing.
4. **Sealed `test3` against the best parameter fit:** **+9.2% worse** at one
   step, **−22.5%** at ten, **−21.8%** at twenty-five. Required 30 / 25 / 20.
5. **`test2` and `test3` agree** to about one point at every horizon
   (1.114/1.092, 0.796/0.775, 0.801/0.782). The model generalises; it does not
   clear the bar on either.
6. **Non-finite states: zero** — five seeds × two splits × two passes, plus
   five stress tests. RA-Sim-0's *small* additive residual went non-finite on
   the same sealed split.
7. **Effective command** never left the joint range; max single step **0.0800
   rad** against a 0.0800 ceiling; standing lag up to **2.06 rad**, produced by
   the state. Hidden-state norm 8.00, which is √64 and therefore the cell's
   own bound rather than a measurement of restraint.
8. **Held-out shapes and reversals:** `test3` is held-out classes 3–4 at σ 0.09
   / ±0.25; the 12.5 Hz reversal stress is the calmest run in the suite — max
   \|Δ\| 0.0616, no rate clipping, no shell events.
9. **Compute:** −0.9% / −0.1% / −0.1% at 64 / 256 / 512 environments, against
   a 20% budget. Inside noise at every size.
10. **RA-Sim-1: RED**, on G1 and G5. G2, G3, G4 and G6 green.
11. **No policy training.** G1 failed; the plan says stop and the gate was not
    moved.
12. **What caused it: expressiveness, not stability, generalisation or cost.**
    The rate ceiling binds on 6–30% of joint-steps and the horizon that fails
    is the one where an instantaneous adjustment is what is needed. The three
    gates that could have failed for the other three reasons all passed, and
    passed with room.
