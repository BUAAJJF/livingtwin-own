# Phase RA-Sim-0 — results

**Verdict: RED.**

* **Gate P — GREEN, decisively.** The existing parameter axes cannot express
  the mismatch. Nineteen configurations get the one-step error from 2.214 to
  1.267 NRMS; the oracle reaches 0.116, so **89% of what the best fit leaves
  behind is still the hidden mechanism** — with a lag-1 error autocorrelation
  of 0.983 and a 1.87× swing across command-magnitude bins.
* **Gate R — RED, on all three held-out splits.** The residual is **1.50×,
  1.30× and 1.28×** the best parameter fit at 1, 10 and 25 steps, where the
  plan required 0.70×, 0.75× and 0.80×. Every interval is on the wrong side
  of 1.0 and the three splits agree to two decimal places.
* **Gate C — not run.** Gate R failed and the phase says stop. No policy was
  trained, no threshold was moved, and no number in this document was produced
  by a stage that did not execute.

The failure has a measured cause and it is not "residual learning does not
work here": **the correction's 99th percentile was its output bound, to four
decimal places, in all nine runs.** The bound was frozen in the plan at
0.05 rad. A validation-only sweep (§6) shows what happens when it is not.

*Run 2026-08-25 16:21 UTC onward, from commit `623e2a9` (301 tests). GPUs 4–7
of `shen-teacher`, and from 18:30 UTC only 6 and 7, because another of the
user's jobs took 4 and 5; GPUs 0–3 were never touched. mjlab 1.6.0 /
rsl_rl 5.4.2 / MuJoCo 3.11.0 / mujoco-warp 3.11.0 / warp 1.16.0 /
torch 2.13.0+cu13.0.*

---

## 1. What ran

| stage | status |
|---|---|
| 0 — injection audit | **formal** — `docs/residual_injection_audit.md` |
| 1 — freeze the hidden target | **formal** — `docs/ra_sim0_experiment_plan.md`, committed before any result run |
| 2 — reward-free data | **formal** — 6 splits, 1.03 GiB, `results/ra_sim0/data/manifest.json` |
| 3 — parameter calibration | **formal** — 19 configurations, `results/ra_sim0/grid/` |
| 4 — residual training | **formal** — 3 seeds |
| 5 — accuracy gate | **formal** — 3 held-out splits, real MJWarp, `results/ra_sim0/accuracy3/` |
| 6 — policy | **stopped by Gate R** — not run, not estimated, not reported |
| — bound sweep | **exploratory**, validation only, chosen after Stage 5 (§6) |
| — data-budget ladder | **exploratory**, validation only (§7) |

### Superseded results, kept

`results/ra_sim0/accuracy/` and `results/ra_sim0/calibration/` are the first
pass, produced before the two method bugs in §2 were found. They are left in
the tree because §2 is about them and because deleting the run that taught you
something is how the lesson is lost. **Nothing in §3–§5 comes from them.**

---

## 2. Three bugs in the method, found by looking at boring numbers

Each of these would have produced a publishable-looking result. None of them
shows up as an implausible number; two of them make the *wrong* answer look
better.

### 2.1 MJWarp is not bit-reproducible, so "unchanged" needs a floor

Two builds of the same environment, same seed, same command stream, disagree
by a mean of **3.6 × 10⁻⁷ rad** after eight control steps and 7.6 × 10⁻³ after
sixty; the maximum after sixty is 1.37 rad, which is a different grasp. Every
"this changed nothing" claim in the audit is a ratio against that floor. An
identity-initialised residual sits at **1.03×** it; the hidden target at
**122,572×**.

### 2.2 A replay that lets its candidates die scores the worst ones on survivors

The first accuracy run, before this was found:

| candidate | one-step NRMS | **usable fraction** | n |
|---|---|---|---|
| nominal | 1.479 | **0.009** | 457 |
| Stage 3 winner | 0.848 | **0.014** | 719 |
| oracle | 0.123 | **0.886** | 45,319 |

A candidate whose command path is badly wrong trips the safety shell within a
few steps, resets, and draws a fresh object. So the *worse* a simulator was,
the more selectively it was scored — on the handful of environments that
happened to survive. Replays now run with **no termination terms at all**.

### 2.3 One environment, one replay: only the first reset draws the recording's objects

The scene's object draw comes from a global RNG that every reset advances. A
second replay inside one process — the 25-step pass, or the next configuration
in a search — resets into a *different* set of objects, held at the recorded
poses. Under that, a 30-configuration coordinate descent found **nothing
better than changing nothing**: its first evaluation was the only valid one.

Re-seeding before the reset does not fix it. Construction consumes RNG before
its own first reset, so seeding to the same value and resetting again lands at
**0.44** of the recording's objects, against 1.00 for a fresh build. So every
replay now builds its own environment, and **every result file carries
`shape_match_at_t0`**. It is 1.000 for all 39 runs in §3–§5.

This is why the accuracy table was recomputed and why Stage 3's search moved
out of `ra_sim0_calibrate.py` (which now refuses to score below 0.999) and
into the evaluator.

All three fixes are pinned by tests in `tests/test_replay.py`.

---

## 3. Gate P — the parameters really are not enough

**GREEN, all four criteria.**

Nineteen configurations, one fresh environment each, scored on `val` by
one-step NRMS over 44,346 samples, `shape_match_at_t0 = 1.000` throughout.
The axes are the pre-registered ones: transport delay {0,1,2,3}, response
scale [0.30,1.00], deadband [0,0.030] rad, low-pass {off,10,20} Hz, damping
{0.75,1.0,1.25}.

| simulator | one-step | 10-step | 25-step | what it is |
|---|---|---|---|---|
| nominal | **2.2137** | 1.4015 | 0.7027 | as shipped |
| best single axis | 1.7071 | 1.1696 | 0.5917 | 3 steps of transport delay |
| Stage 3's joint winner | **1.2666** | 0.9707 | 0.5084 | delay 3, response 0.377, deadband 0.0175, damping 1.25 |
| **oracle** | **0.1155** | **0.0441** | **0.0279** | the hidden target itself |

An NRMS of 1.0 is *as wrong as predicting that nothing moved*. The whole
parameter family gets **45%** of the way from nominal to the oracle at one
step, and the oracle removes **89%** of what the best fit still leaves
(ratio 0.107 against a criterion of 0.70).

**Four of the nineteen configurations went non-finite** and are recorded as
NaN rather than dropped: delay 1 alone, delay 1 + response 0.60 + deadband
0.008, a 10 Hz low-pass alone, and response 0.50 alone. With the safety shell
removed for the replay (§2.2) and the state re-anchored every step, a command
path that lags without enough damping diverges. That is a real property of
those settings and not a bug in the measurement; it is also why the winner
carries `damping 1.25`.

The error that survives the best fit is **structured, not noisy**:

| statistic | best parameter fit | oracle |
|---|---|---|
| lag-1 autocorrelation of the one-step error | **0.983** | 0.837 |
| max/min RMS across command-magnitude bins | **1.87×** | 1.36× |

Binned by the size of the commanded step, the best fit's RMS error is 0.0368
rad below 2 mrad of command, 0.0202 at 2–5 mrad and 0.0330 at 20–40 mrad —
non-monotone, which no constant response scale can be. The **signed** mean
error flips sign between the smallest bin (+0.0167 rad) and every other bin
(−0.002 to −0.003): below the backlash band the target does not move at all
and the fit overshoots; above it the target lags and the fit undershoots. One
scalar cannot sit on both sides of zero.

The oracle is the only simulator whose remaining error looks like noise rather
than like an unmodelled mechanism. Neither the parameter fit nor the residual
has touched the mechanism — they have only made it smaller.

Criterion 4, leakage: every candidate is scored on a split generated under its
own seed with held-out shape classes, re-anchored from the same recorded
state, in the same window, against the same recording, with the objects
verified identical (`shape_match_at_t0 = 1.000`) and the reproducibility floor
of §2.1 reported beside it.

---

## 4. Gate R — the residual is not more accurate

**RED.** Three residual training seeds, three held-out splits, real MJWarp
rollouts, 800 control steps at 64 environments each, `shape_match 1.000`,
45–48k one-step samples per candidate.

| split | horizon | best parameter fit | residual (3 seeds) | ratio, 95% CI | required |
|---|---|---|---|---|---|
| `test` | 1 | 1.2768 | 1.9136 | **1.499** [1.475, 1.523] | ≤ 0.70 |
| `test` | 10 | 0.9866 | 1.2855 | **1.303** [1.292, 1.314] | ≤ 0.75 |
| `test` | 25 | 0.5063 | 0.6465 | **1.277** [1.265, 1.289] | ≤ 0.80 |
| `test_amp` | 1 | 1.2906 | 1.9306 | **1.496** [1.472, 1.520] | ≤ 0.70 |
| `test_amp` | 10 | 0.9907 | 1.2968 | **1.309** [1.298, 1.320] | ≤ 0.75 |
| `test_amp` | 25 | 0.5108 | 0.6482 | **1.269** [1.257, 1.281] | ≤ 0.80 |
| `valh` | 1 | 1.2888 | 1.9220 | **1.491** [1.468, 1.515] | ≤ 0.70 |
| `valh` | 10 | 0.9974 | 1.2924 | **1.296** [1.285, 1.307] | ≤ 0.75 |
| `valh` | 25 | 0.5177 | 0.6575 | **1.270** [1.258, 1.283] | ≤ 0.80 |

`test` is held out by object shape, `test_amp` by object shape *and* action
amplitude and reversal rate (perturbation σ 0.12 clipped at 0.35, against
0.05/0.15 in training), `valh` is a third independent draw of held-out shapes.
They agree to two decimal places, so this is not a split that went badly.

Full table, one-step / 10-step / 25-step NRMS on `test`:

| candidate | 1 | 10 | 25 |
|---|---|---|---|
| nominal | 2.228 | 1.415 | 0.705 |
| best single axis | 1.721 | 1.188 | 0.590 |
| broad DR (best draw) | 1.502 | 1.072 | 0.532 |
| **best parameter fit** | **1.277** | **0.987** | **0.506** |
| residual (mean of 3) | 1.913 | 1.285 | 0.646 |
| **oracle** | **0.137** | **0.034** | **0.022** |

Against the **nominal** simulator the residual does help — 2.228 → 1.913 at
one step (−14%), RMS 0.0435 → 0.0373 rad, and 0.0403 → 0.0344 rad in the
reversal region where the backlash lives. It is learning the right kind of
thing. It does not reach the parameter fit anywhere, including in that region,
where the fit is at 0.0225.

**Physical sanity.** All finite. Peak joint velocity 17.1 rad/s against the
nominal simulator's 17.6 and the oracle's 19.0 — the residual is the
*quietest* candidate, not one that injects energy. No state jumps. Throughput
2,347 env-steps/s against 2,454 nominal at 64 environments (4.4%; 15.8% at 256
from the audit). GPU 146 MiB against 118.

**The ensemble's uncertainty is anti-calibrated.** Correlation between the
four members' disagreement and the actual error: **−0.28**, the same on all
three splits and all three seeds. Where the ensemble disagrees most, the error
is *smallest*. A spread that anti-predicts error is worse than no spread, and
nothing downstream should have used it.

### The bound was saturated the whole time

| | mean \|Δ\| | **p99 \|Δ\|** | bound |
|---|---|---|---|
| residual, 3 seeds × 3 splits | 0.038 rad | **0.0499–0.0500 rad** | 0.05 |

Nine runs, and the 99th percentile is the bound to four decimal places in
every one. Arithmetic that was in the plan and deserved more weight: at a
commanded step `s` the target completes `0.85/(1 + 1.6 s/0.04)` of it; the
command path's slew ceiling permits steps to about 0.06 rad; at 0.06 rad the
shortfall is 0.045 rad **before** a backlash band of up to 0.022 rad. Worst
case ≈ 0.067 rad against a bound of 0.05.

---

## 5. What was measured, in numbers

**Data.** Six splits, 1.03 GiB, none in git;
`results/ra_sim0/data/manifest.json` carries every seed, budget, composition
and SHA-256.

| split | seed | domain | shapes | envs × steps | budget | SHA-256 (12) |
|---|---|---|---|---|---|---|
| `train` | 7101 | target | 0–2 | 64 × 15000 | 300 s | `8d5715fa3d80` |
| `val` | 7201 | target | 0–2 | 64 × 15000 | 300 s | `53f3d2a33fb8` |
| `valh` | 7301 | target | 3–4 | 64 × 6000 | 120 s | `1fb75bfff068` |
| `test` | 7401 | target | 3–4 | 64 × 6000 | 120 s | `290993e30b26` |
| `test_amp` | 7402 | target | 3–4 | 64 × 6000 | 120 s | `25192397d3f6` |
| `nominal` | 7501 | **nominal** | 0–2 | 64 × 15000 | 300 s | `9c8b5f2f0614` |
| `test2` | 7403 | target | 3–4 | 64 × 6000 | 120 s | `a8185370c785` |

Each is 70% frozen-policy rollout, 20% the same policy under a pre-registered
action perturbation, 10% a scripted multi-frequency sweep with the gripper
cycling. The frozen policy is
`piperx_pick_place_vision/2026-08-22_17-15-09_f3/model_1500.pt`, SHA-256
`fe9ecd12f5895192…` — the checkpoint Phases WM1-A and WM1-B used.

**Models.**

| | parameters | data | training |
|---|---|---|---|
| nominal transition surrogate | 75,276 | 949,029 nominal transitions | 60 epochs, val NMSE **0.0128** |
| residual, one member | 18,822 | 16,297 windows of 80 steps | 30 epochs |
| residual ensemble (4 members) | **75,288** | same | 194–215 s per seed, one GPU |

The surrogate is fitted on nominal-simulator transitions only, frozen the
moment it is fitted, and reports nothing: every number in §3, §4, §6 and §7
comes from a real MJWarp rollout. `tests/test_ra_sim0_discipline.py` checks
statically that the residual's feature builder takes exactly four deployable
tensors everywhere it is called, that the collector records the command the
controller *issued* rather than the one the plant delivered, that no stage of
the pipeline mentions the hidden plant's state, and that the replay harness
never writes state after a physics step.

**Compute.** 7 data collections, 19 grid evaluations, 20 accuracy runs, 26
residual trainings, 16 bound/budget evaluations, 12 fresh-split evaluations,
plus the superseded first pass. About 11 GPU-hours across four cards, then
two.

### Deviations from the pre-registration

1. **The second training-data seed (7102) was killed** at 2,000 of 15,000
   steps; four collectors and a calibration were sharing the cards and it had
   slowed to 487 env-steps/s. The residual trains on seed 7101 alone —
   960k transitions.
2. **`valh`, `test` and `test_amp` were collected at 6,000 steps (120 s)
   rather than 15,000**, decided for wall-clock before any of them was read.
   The accuracy gate scores 800 of those steps.
3. **Stage 3's search moved from a coordinate descent to a 19-configuration
   grid**, because §2.3 makes an in-process descent invalid. The grid covers
   the same pre-registered axes and is scored under the corrected protocol.
   The descent's own winner is in the grid and is not the best of it.
4. **Damping 1.5 was not searched.** Three of the four pre-registered values
   were, and the winner sits at 1.25, interior to the range.
5. **GPUs 4 and 5 became unavailable at 18:30 UTC** when another of the user's
   jobs took them. Everything after that ran on 6 and 7.
6. **The data-budget ladder and the bound sweep, both listed in the plan and
   both missing from the first pass, were run** — on validation, and reported
   in §7 and §6 as exploratory. §8's fresh split was collected afterwards and
   is the only place a selected bound meets data it had not seen.

---

## 6. Why it failed: the bound, swept on validation

**Exploratory. Chosen after seeing Stage 5, scored on `val` only, and it does
not change the verdict in §4.** `DELTA_MAX` was frozen in the plan at 0.05 rad
and argued for from the target's backlash band; it was not selected on
validation, which is the mistake this section measures.

Two training seeds at each bound, everything else identical — same
architecture, same 30 epochs, same four loss terms, same data, same frozen
surrogate.

| bound (rad) | one-step | 10-step | 25-step | mean \|Δ\| | **p99 \|Δ\|** | mean \|Δ\| ÷ commanded step | ratio to the parameter fit |
|---|---|---|---|---|---|---|---|
| **0.05** (the plan's) | 1.9071 | 1.2725 | 0.6452 | 0.038 | **0.0499** | 0.79× | 1.506 |
| 0.10 | 1.6198 | 1.1270 | 0.5769 | 0.074 | **0.0999** | 1.53× | 1.279 |
| 0.15 | 1.4001 | 1.0006 | 0.5190 | 0.104 | **0.1499** | 2.13× | 1.105 |
| 0.20 | 1.2150 | 0.8909 | 0.4705 | 0.127 | **0.1998** | 2.60× | **0.959** |
| 0.25 | 1.0503 | 0.7905 | 0.4260 | 0.142 | **0.2495** | 2.92× | 0.829 |
| 0.30 | *one seed non-finite* | 0.6987 | 0.3848 | 0.152 | **0.2990** | 3.13× | — |
| 0.40 | **0.6294** | **0.5336** | **0.3055** | 0.164 | **0.3970** | 3.37× | **0.497** |
| *best parameter fit* | *1.2666* | *0.9707* | *0.5084* | — | — | — | 1.000 |
| *oracle* | *0.1155* | *0.0441* | *0.0279* | — | — | — | 0.091 |

Two seeds per bound except where noted; the seeds agree to three decimal
places at every level, so none of this is noise.

Four things to read off it.

**The bound was the binding constraint at every level.** The 99th percentile
of \|Δ\| is the bound to three or four decimal places at every setting from
0.05 to 0.30, and only at 0.40 does it begin to come off it (0.397 of 0.400).
Nothing in this sweep found the residual's natural size.

**The improvement is monotone, and the crossings can be located.** From 0.05
to 0.40 the one-step error falls by a factor of three. The residual first
*matches* the best parameter fit at a bound of **0.20**, where its mean
correction is **2.6× the commanded step**; it first clears Gate R's 30%
margin between 0.25 and 0.40, at roughly **3.2×**. Those two numbers are the
useful output of this sweep.

**But by then it is no longer a residual.** The commanded step in this task
sits on its slew ceiling almost always: median 0.0390 rad, 99.9th percentile
0.0487 rad. A correction whose mean magnitude is 0.164 rad is **3.4× the step
it is correcting**; at that size the network
is not adjusting the servo target, it *is* the servo target. That is a
different object from the one the phase set out to test, with different safety
properties, and the plan bounded the correction precisely to prevent it.

So the honest reading is not simply "the bound was too small". It is: **on
this axis, matching the hidden target requires holding the servo target
further from the command than the command itself ever moves.** The hidden
lag completes as little as 17% of a large step, so a simulator that reproduces
it has to keep the target far behind — and a correction constrained to be
small cannot express that, however it is parameterised.

**And a large correction starts to destabilise the simulator.** One of the
fourteen validation runs — bound 0.30, seed 0 — produced non-finite states in
the one-step pass and is recorded as NaN rather than dropped. That is the
first appearance of exactly what the bound was there to prevent, and it
appears inside the range where the method starts to win.

The ensemble's anti-calibration softens as the bound grows (−0.28 → −0.13) but
never becomes informative. Four members differing only in initialisation and
batch order are not a posterior.

**None of this is a Gate R result and none of it may be quoted as one.** The
bound was chosen after the fact, the numbers are on validation, and the
thresholds in §4 were set against `test`.

---

## 7. How much target data it needed: the budget ladder

**Exploratory, validation only.** One seed per budget, everything else
identical, the bound at the plan's 0.05.

| budget | control steps | one-step | 10-step | 25-step | mean \|Δ\| |
|---|---|---|---|---|---|
| 10 s | 500 | 2.111 | 1.332 | 0.671 | 0.020 |
| 30 s | 1,500 | 1.966 | 1.285 | 0.654 | 0.035 |
| 60 s | 3,000 | **1.895** | **1.269** | **0.643** | 0.040 |
| 180 s | 9,000 | 1.901 | 1.271 | 0.644 | 0.039 |
| 300 s | 15,000 | 1.918 | 1.276 | 0.647 | 0.038 |

**It saturates at 60 seconds of arm time**, and the last three rows are within
1% of one another — the 300 s row is fractionally *worse* than the 60 s one,
which is the size of the run-to-run spread. Ten seconds is clearly too little;
a minute is enough for everything this residual is going to learn.

Put beside §6, that settles the diagnosis. The residual is **not data-limited**
— five times more target data changes nothing — and it **is** bound-limited,
where three times the bound changes the answer by a factor of three. Whatever
is wrong with the pre-registered instantiation, collecting more reward-free
trajectories would not have fixed it.

---

## 8. RA-Sim-0b — the same question, on data nobody has looked at

**Pre-registered here, before the runs finished.** §6 is a validation sweep
chosen after seeing `test`, so nothing in it can be quoted as a gate. This
section is the clean version of the question it raises, and it is written down
before its numbers exist.

**Design.**

* A **fresh target-domain split**, `test2`, seed 7403, held-out shape classes
  3–4, 64 × 6000 steps, collected after §6 and used for nothing else. No
  model, hyper-parameter or threshold on this page was chosen with it.
* Two bounds carried forward, both **selected on `val`**: **0.40**, the sweep's
  minimum, and **0.15**, a moderate value carried alongside precisely because
  0.40 sits at the edge of the swept range and an argmin at an edge is a
  warning, not a result. The plan's **0.05** is carried as the control.
* Three training seeds at each bound, the same three the gate uses.
* Comparator: the same best parameter fit as §4 (delay 3, response 0.377,
  deadband 0.0175, damping 1.25), and the same nominal and oracle anchors.
* The same thresholds as Gate R — **30% / 25% / 20%** reductions at 1, 10 and
  25 steps, with the 95% interval over seeds on the right side.

**What each outcome would mean, decided now.**

* If **0.40 clears the thresholds and 0.15 does not**: the effect is real but
  needs a correction larger than the command it corrects, which is a
  replacement for the command path rather than a residual on it. That is a
  finding about what this axis demands, not a licence to call Gate R passed.
* If **0.15 clears them too**: the pre-registered bound was simply mis-chosen
  and the method works at a magnitude that is still a correction. That is the
  strongest outcome available and it still does not retro-fit Gate R — §4
  stands as run.
* If **neither clears them**: §6's validation gain does not transfer, and the
  bound is not the explanation after all.

Whatever it returns, **Phase RA-Sim-0's verdict stays RED** and Stage 6 stays
unrun. A gate is what you registered before you looked.

### 8.1 What it returned

`test2`, seed 7403, held-out shapes, 64 × 6000 steps, collected after §6 and
opened once. Three training seeds per bound; anchors measured on the same
split, `shape_match_at_t0 = 1.000` for all twelve runs.

| | one-step | 10-step | 25-step |
|---|---|---|---|
| nominal | 2.2105 | 1.4051 | 0.6974 |
| best parameter fit | **1.2683** | **0.9674** | **0.4969** |
| residual, bound 0.05 (the plan's) | 1.8971 | 1.2746 | 0.6386 |
| residual, bound 0.15 | 1.3949 | 1.0023 | 0.5136 |
| residual, bound 0.40 | **0.6365** | **0.5338** | **0.2976** |
| oracle | 0.1321 | 0.0314 | 0.0230 |

Against the parameter fit, with the gate's own thresholds and a 95% interval
over three seeds:

| bound | mean \|Δ\| ÷ commanded step | 1 step | 10 steps | 25 steps | verdict |
|---|---|---|---|---|---|
| **0.05** | 0.79× | 1.496 [1.472, 1.520] | 1.318 [1.306, 1.329] | 1.285 [1.273, 1.298] | fails all three |
| **0.15** | 2.11× | 1.100 [1.095, 1.105] | 1.036 [1.031, 1.041] | 1.034 [1.032, 1.035] | fails all three |
| **0.40** | 3.32× | **0.502** [0.487, 0.517] | **0.552** [0.542, 0.562] | **0.599** [0.582, 0.616] | **clears all three** |

Reductions at bound 0.40: **−49.8%, −44.8%, −40.1%** against required 30%,
25% and 20%, with every interval clear of the threshold.

**This is the first branch of the three registered in §8, and its reading was
fixed before the numbers existed:** the effect is real, and it needs a
correction larger than the command it is correcting.

How much larger is now measured rather than argued. The commanded step in this
task sits on its slew ceiling almost always — median 0.0390 rad, 99.9th
percentile 0.0487 rad. The winning residual's mean correction is 0.162 rad,
**3.3× the step it is correcting**. The moderate bound, at 2.1×, is already
past "adjusting the command" and still loses to the parameter fit; only at
3.3× does it win.

So the sentence that survives contact with fresh data is not *"a bigger bound
fixes it"*. It is:

> On this axis, a correction that only nudges the command cannot reproduce the
> target at all, and one that can is no longer a residual on the command path
> — it has replaced it.

That is a real and reportable property of structural actuator mismatch of this
severity, and it is a different claim from the one Phase RA-Sim-0 set out to
test. **The phase's verdict stands at RED**, Gate R stands as run in §4, and
Stage 6 stays unrun. What §8 buys is a well-posed next question rather than a
retro-fitted pass.

---

## 9. Summary

1. **Injection point: L2, the action/command wrapper.** `piper_push.actions`
   already owned a stateful, batched, per-environment stage between the policy
   and the servo; it gained a `command_hooks` tuple, empty by default. Pre-step
   generalized force (L4) was tested and works; no post-step state overwrite is
   used anywhere, which is the phase's STOP condition.
2. **Why the simulator was not rewritten.** It did not need to be. MuJoCo
   keeps integrating, resolving contact and enforcing limits; only the number
   the servo is asked to hold changes.
3. **The hidden target.** Per-joint asymmetric gear backlash (0.006–0.022 rad
   flanks, a different flank wider on different joints) composed with a
   current-limited lag whose completed fraction falls with the size of the
   commanded step. Frozen and committed before any result run; a test pins the
   constants.
4. **How much the parameters explain.** 45% of the gap. Nineteen
   configurations take the one-step NRMS from 2.214 to 1.267; the oracle
   reaches 0.116, so the best fit still leaves 89% of the explainable error,
   with a lag-1 error autocorrelation of 0.983 and a 1.87× swing across
   command-magnitude bins.
5. **Did the residual reduce held-out multi-step error?** **No.** 1.50×,
   1.30× and 1.28× the best parameter fit at 1, 10 and 25 steps, on three
   held-out splits that agree to two decimals. It beats the *nominal*
   simulator by 14% and never reaches the parameter fit.
6. **Does it hold in real MJWarp?** Yes — every number here is a real MJWarp
   rollout, with the recording's own objects (`shape_match_at_t0 = 1.000` in
   all 51 runs). The surrogate exists only to carry a gradient.
7. **Cost.** 75,288 parameters, 960k target transitions, 194–215 s of offline
   training per seed, 4.4% throughput at 64 environments and 15.8% at 256.
8. **Policy training: not run.** Gate R failed; the phase says stop. No
   threshold was moved.
9. **Target policy, safety, retention: not measured.** Stage 6 did not run.
10. **Run time.** Under the 8-hour budget, GPUs 4–7 and then 6–7 only.
11. **Verdict: RED**, with Gate P GREEN and Gate R RED.
12. **Worth a real robot? Not yet** — and the reason is now a measured
    quantity rather than a mystery. The correction has to be **3.3× the
    commanded step** before it beats a parameter fit; at 2.1× it still loses.
    A correction that large is not a residual on the command path, it is a
    replacement for it. Before any camera, policy or hardware work, the next
    phase has to decide whether that object is the one it wants — and if it
    is, it is a different design with different safety arguments, not a
    bigger `DELTA_MAX`.

---

## 10. What would have to be different

**Decide what the correction is allowed to be, first.** The phase treated
`DELTA_MAX` as a safety bound and picked it from the target's backlash width.
The measurement says the target's *lag* — not its backlash — is what needs
expressing, and that needs a correction several times the commanded step. Any
follow-up has to choose deliberately between (a) a genuinely small correction,
which on this axis cannot work, and (b) a learned command-path replacement,
which can, and which needs its own safety story: what bounds it, what happens
when it is wrong, and what the arm does if it saturates.

**Fix the ensemble or drop it.** A spread that correlates **−0.28** with error
is not uncertainty. Four members differing only in initialisation and batch
order are not a posterior. Bootstrap the windows per member, or replace the
ensemble with one model and a learned variance head, and re-check the
correlation before anything downstream consumes it.

**Do not collect more data.** The budget ladder saturates at 60 seconds of arm
time and the 300 s row is fractionally worse than the 60 s one. Five times the
data changes nothing. This is not a data problem.

**Keep the three protocol fixes.** They are worth more than this phase's
result: measure the simulator's own reproducibility floor before claiming
anything is unchanged; never let a replay's candidates terminate, or the worst
simulators get scored on their luckiest environments; and build one
environment per replay, because only its first reset draws the recording's own
objects.

**Then, and only then, Stage 6.** A simulator that is not yet more accurate
than a parameter fit has nothing to offer a policy, and running the policy
comparison now would measure PPO seed noise — which Phase WM1-A's eight-seed
extension has already shown this project can mistake for a result.
