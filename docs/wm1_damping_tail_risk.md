# Phase WM1-B — a reward-free calibration loop for a tail-only mismatch

**Status: in progress.** Identification, the risk head and the posteriors are
complete and reported below. The adaptation stage is running; §6 onward is
marked `PENDING` where it has no numbers yet, rather than left out.

---

## 0. What this phase asked, and the one-line answer to each part

The axis is `servo_damping_scale`. The target is **0.75**, nominal is **1.0**,
and **1.5** is a counter-direction control. What makes it a different question
from WM1-A is where the cost sits. Phase WM0 measured, at 0.75, a throughput
fall of 4.5% and a safety-shell trip rate multiplied by **89**. Almost the
whole cost is in the tail, so every score built on throughput, on reward, or on
how faithfully a simulator reproduces a state trajectory reads this domain as
nearly harmless.

| # | asked | answer |
|---|---|---|
| 1 | restore the engineering baseline | done — the suite was green before any experiment ran, and is at **293 tests** now (§1) |
| 2 | correct WM1-A's wording | done — `oracle` is `known-parameter target-only adaptation` everywhere, in the gate's constants as well as the prose, and the safety gate is reported as failed (§1.3) |
| 3 | extend WM1-A to ≥8 training seeds with a count model | 8 seeds on the known-parameter reference, negative binomial and cluster-robust Poisson, both clustered on the training seed (§1.4) |
| 4 | target 0.75, nominal 1.0, counter 1.5 | verified against WM0's config path to bitwise-identical `kd` (§2.2) |
| 5 | a deployable risk head `C_obs` | **built, trained, and it does not work.** It loses to its own shuffled-label control. Reported as a measured negative (§4) |
| 6 | compare seven approaches | five of them return the *same distribution*; the comparison that survives is three-way (§5.4) |
| 7 | does a risk-aware posterior cut safety events at equal data and PPO budget? | PENDING — §6 |
| 8 | 512 envs × 2400 steps, ≥8 seeds, ≥3 repeats, count model, ≤5% throughput and retention loss | running |
| 9 | do not extend to camera, residual or real robot | not extended |

---

## 1. Restoring the baseline, and correcting the record

### 1.1 The test suite

`perturb.py` referenced `DEPTH_DROPOUT`, which a camera refactor in the working
tree had removed; the server could not import `piper_push` at all. The
follow-through was already written in the working tree and was committed as-is
with attribution rather than rewritten.

Two further breakages were found only because something tried to use them,
which is the argument for running the pipeline rather than reading it:

* `wm_posterior.py` raised on import — `AXES` referenced a bare `TARGET` bound
  on the line below it. Broken since the axis refactor, invisible because
  WM1-A's posteriors predate that commit.
* `wm1_adapt.sh` hard-coded `--latency-probs` for training, `--obs-latency-steps
  3` for evaluation, and `OUT=results/wm1_latency/adapt` for results. Pointed at
  a WM1-B plan, it would have trained every damping configuration at *nominal*
  damping, evaluated it in the *nominal* domain, written the results into
  *WM1-A's* directory under filenames `wm1_seeds.py` globs — and reported the
  difference as a recovery. A clean table and a wrong conclusion, with nothing
  in the output to show it.

All three now travel in the plan, and tests grep for each constant returning.

### 1.2 Where no equivalence exists, nothing was guessed

`B1a_cmd_joint` correlates the commanded step against the joint's response at
each candidate **shift**. `B1b_img_proprio` fits a ridge from the image encoding
to the joint state `c` steps earlier. On the latency axis a candidate *is* a
shift and both are the natural model-free thing to try.

On the damping axis a candidate is a multiplier on the servo's derivative gain.
There is no shift to scan. `img[fit] -> q[fit - 0.75]` is not a weaker version
of the same idea; it is a fractional index into time, and the nearest integer
reading of it would score every candidate identically while looking like a
measurement. As written it crashed — five scores into a three-way posterior.

Both are therefore **deprecated by name on this axis**, and the report records
why. The trajectory-matching comparison the phase does need is `B3_state`,
which scores a simulated rollout against the observed one, is indexed by
candidate rather than by shift, and carries across unchanged.

### 1.3 `oracle` was the wrong word

`J_ORACLE` is now `J_KNOWN_PARAM`, `TRIPS_ORACLE` is `TRIPS_KNOWN_PARAM`, and
the plans emit `KNOWN_PARAM`. The name asserted an upper bound the numbers
contradict: at α = 0.75 WM1-A's posterior-guided run reached **49.89** obj/min
in the target domain against the known-parameter run's **49.72**, in the same
domain, *without being told the parameter*, and retained **54.32** against its
**49.79**.

What §8 of the WM0 report measures is what one fine-tuning recipe achieves when
the inference problem is removed. That is a reference point. `recovery` is
measured against it in exactly that sense and can exceed 1.

**WM1-A's safety gate G3 failed and this report does not claim otherwise.**

### 1.4 The seed extension, and what more seeds did to WM1-A

WM1-A reported trips with a quasi-Poisson interval scaled by the dispersion
across *process repeats*. That is the wrong denominator: repeats inside a
training seed agree closely, and seeds do not — 3.22, 8.01 and 4.49 trips per
arm-hour for one configuration. An interval built from within-seed variation
says how precisely one PPO run was measured, not how precisely the method was.

`piper_push.count` re-answers every trip question with the seed as the unit:
repeats summed inside a seed, a negative binomial whose `alpha` *is* the
seed-to-seed heterogeneity, and a cluster-robust Poisson sandwich beside every
rate ratio so the report can say which assumption carries a conclusion.

Three things the tests caught while building it. The likelihood is checked term
by term against `torch.distributions.NegativeBinomial` at float64 — at float32
the reference is only good to 1e-6, which would not catch a `p`/`1-p` swap. A
method that never trips used to raise, because the log rate is minus infinity
and the information matrix is singular; that is the outcome this phase is
*hoping* for, so it now gets the exact rule-of-three bound, and a zero-event arm
gets a conditional binomial test. And an all-clusters-agree case must give a
visibly narrower interval than an all-clusters-disagree case with identical
totals, which is the whole point.

The extension is still running. What it has already changed, at eight seeds
against the original three, is not only the width of the intervals but the
point estimates:

| configuration | seeds | trips/arm-h | NB 95% CI | recovery |
|---|---|---|---|---|
| known-parameter target-only | 3 → **8** | 4.56 → **8.54** | [5.83, 12.50] | 0.55 → **0.32** |
| decision-aware, α = 0.75 | 3 → **5** | 5.24 → **6.21** | [4.51, 8.53] | 0.56 → **0.42** |
| broad posterior (B1b), α = 0.75 | 3 | 4.43 | [3.93, 4.99] | 0.55 |

WM1-A's three seeds were an optimistic draw on safety, and the pattern is
consistent across configurations rather than isolated to one.

**A consequence for G3 that is recorded and not acted on.** G3's ceiling is
derived from `TRIPS_KNOWN_PARAM = 4.83`, measured on three seeds. Pooled over
eight the same configuration trips at **7.22**/arm-hour, which would put the
threshold at **7.59** instead of **6.00** and turn several current failures into
passes. **The threshold is not being recomputed.** That is repairing a result by
moving the gate. It stays at 6.00; the constant carries the arithmetic so a
reader can see the size of the effect without the gate having moved.

---

## 2. The axis

### 2.1 How a draw reaches the simulator

`randomize_servo_damping` is a per-world event under
`@requires_model_fields("actuator_biasprm")`. A `BuiltinPositionActuator` has
`biasprm = (0, -kp, -kd)`, so the event scales column 2 of `actuator_biasprm`
for the six arm actuators, from the *default* field rather than the current one
so repeated resets do not compound. The gripper is skipped.

A point mass at nominal returns `{}` and installs nothing, so every
latency-only run is byte-identical to what it was before this axis existed.

### 2.2 Verified against WM0's path

`check_damping.py` compares the per-environment event against the whole-session
config path WM0 used:

| check | result |
|---|---|
| `kd` from the event vs from the config path | **bitwise identical**, max abs difference 0.000e+00 |
| `kd` ratio on the six arm actuators | exactly 0.750 |
| gripper actuator | untouched |
| `kp` | unchanged |
| mixture proportions realised vs asked (0.5 / 0.25 / 0.25) | 0.508 / 0.266 / 0.227 |

### 2.3 The cost vector

Measured over the dataset's 2496 sessions and 35.2 arm-hours:

| damping | trips per arm-hour |
|---|---|
| **0.75** (target) | 183 – 230 |
| 1.0 (nominal) | 1.9 – 3.8 |
| 1.5 (counter) | 0 – 1.4 |

confirming WM0's ×89. The counter-direction candidate is *safer* than nominal,
which is why 1.5 is a control and not a second target.

### 2.4 The anchors, measured rather than reconstructed

Phase WM0 ran the known-parameter fine-tune for the latency axis only. For
damping it recorded the zero-shot penalty as "-4.5% throughput, ×89 trips" and
left the ceiling command in its report unrun, so recovery and the safety gate
had nothing to be measured against except a percentage borrowed from a
different axis. `scripts/wm1b_anchor.sh` measures both the same way every
adapted run is measured -- 512 environments, 2400 steps, the same three
evaluation seeds -- so the comparison is paired rather than assembled from two
recipes:

| anchor | obj/min | trips/arm-hour |
|---|---|---|
| `nominal` — deployed policy, source domain | 55.78 | 2.78 |
| `zeroshot` — same weights, damping 0.75, no adaptation | **53.39** | **209.08** |

A throughput fall of **4.3%** and a trip multiplier of **75×**, against WM0's
4.5% and ×89 measured independently a phase earlier. The axis is what it was
said to be.

**And this is what makes WM1-B a different problem from WM1-A.** There, the
zero-shot policy lost 24% of its throughput and adaptation's job was to win it
back. Here it loses 4.3%, so there is very little throughput to recover and
essentially the entire mismatch is the tail. The first completed run makes the
consequence concrete — broad domain randomisation, one training seed:

| | obj/min | trips/arm-hour |
|---|---|---|
| zero-shot in target | 53.39 | 209.08 |
| broad DR adapted, in target | **50.20** | **92.82** |

Adaptation on this axis **buys safety with throughput** rather than recovering
throughput: trips more than halve and objects per minute fall by 6%. So `C4`
in §7 is not "did it recover throughput" — there is none to recover — but "did
it pay less than 5% more than the comparator it is being judged against". A
method could otherwise pass the safety criteria by learning to move slowly, and
a policy that never moves trips nothing.

---

## 3. The dataset

21 files, **2496 sessions**, **35.2 arm-hours**, 5.05 GB of fp16. All six
leakage checks pass: the label is not a channel, no generation seed is shared
between splits, held-out splits are shape-disjoint from training, one sequence
length per split, budgets nested within a session, balanced candidates per
split.

---

## 4. The risk head — a measured negative

The phase asked for `C_obs(observation/action history) -> P(a simulated safety
event within the next H steps)`, trained on simulator safety labels, reading
only deployable channels. It was built exactly so: `RiskHead.CHANNELS` passes
through `wm_data.assert_deployable` at construction, so a head naming the trip
label, the damping or the object pose cannot be instantiated at all. The label
is strictly in the future of everything the head is shown.

**It does not work.**

### 4.1 The first attempt lost to its own control

| variant | val AP | AUC | base rate |
|---|---|---|---|
| `risk` | 0.012 | 0.577 | 0.0083 |
| `risk_shuffled` | **0.017** | **0.597** | 0.0083 |

Mean prediction 0.486 everywhere — a constant. The uncapped positive class
weight at a 0.8% base rate is 123, and the weighted minimiser answers 0.5 for
everything: ranking unchanged, calibration destroyed.

### 4.2 What any head could reach

Before retraining anything, the ceiling was measured directly — history length
{25, 50, 100} against horizon {2, 5, 10, 25}, scoring the best simple velocity
feature **inside the target domain**, so the question is which *windows* trip
rather than which *domain* they came from:

| horizon | lead time | AUC | AP ÷ base |
|---|---|---|---|
| 2 steps | 40 ms | 0.79 – 0.83 | 1.5 – 3.7× |
| 5 steps | 100 ms | 0.63 – 0.67 | 1.4 – 2.3× |
| **10 steps** | **200 ms** | 0.60 – 0.66 | **2.4 – 2.7×** |
| 25 steps | 500 ms | 0.57 – 0.60 | 1.5 – 1.9× |

Two steps is 40 ms — the joint is already at the threshold, so that is
detection wearing a horizon. Twenty-five is what the specification suggests and
is barely above chance. `HORIZON` was moved from 25 to **10** by this
measurement, and the sweep is recorded in the constant's docstring so moving it
again requires editing the evidence.

### 4.3 The decisive control

Trained across all three domains a head can score well by recognising *which
domain* a window came from — the base rates are 0.99% / 0.01% / 0.01%, a
hundredfold apart — while predicting nothing about which window trips. So the
headline is the **within-target-domain** number, where that shortcut is gone
and the base rate is three times higher:

| variant | AP within target | ÷ base | AUC |
|---|---|---|---|
| `risk` (all domains) | 0.007 | 0.74× | 0.383 |
| `risk_heldout` (target unseen in training) | 0.009 | 0.97× | 0.473 |
| `risk_shuffled` | 0.008 | 0.88× | 0.439 |
| **`risk_target_only`** | 0.009 | **1.02×** | 0.464 |
| **`risk_target_only_shuffled`** | 0.011 | **1.16×** | **0.522** |

**The shuffled-label control matches or beats the real head in every pairing**,
including inside the target domain where there is no shortcut to exploit. Two
independent measurements agree. `C_obs` as the phase defines it is not
learnable from deployable observations on this axis at any lead time tested.

The head is kept. `M5_risk_score` in §5 keeps `S_risk` and reports what a score
built on noise scores — the negative stays in the record rather than being
deleted.

### 4.4 What was done instead

Risk-awareness moved from a learned head into the **decision rule**. What is
available without any target label is §2.3's cost vector: how often the shell
fires under each candidate *in simulation*.

```
q_risk(θ) ∝ q(θ) · exp(λ · cost(θ))
```

This is not an estimator and does not try to be. It is the distribution a
planner should train against when being wrong towards danger costs more than
being wrong towards safety. `λ` comes from a rule fixed before any adaptation
ran — `ln(9) / (cost range)`, nine-to-one odds between the extremes from an
undecided start — stated as a rule rather than a number so it cannot be tuned
towards the answer, and scale-free, so trips-per-hour and trips-per-minute give
the same tilt.

**Its limitation is in the docstring, in a test, and here: a point mass cannot
be tilted.** Risk-awareness has nothing to do when there is no surviving
candidate to move mass towards. It can only help a posterior that is genuinely
uncertain — which, as §5.4 shows, is exactly the comparison this phase is about.
A candidate the data excluded stays excluded: the tilt reweights, it does not
resurrect.

---

## 5. Reward-free identification

### 5.1 The dynamics model

3/3 correct on every component (`z`, `p`, `e`, multi, sum) on both validation
and held-out-shape validation. Unlike WM1-A, proprioception and servo error
**do** identify damping, because it is plant-side rather than sensor-side. The
shuffled-θ control is flat — a spread of 0.09–0.16 nats out of ~250.

### 5.2 The classifier

99.8% on both validation splits; shuffled 32.8%, done-only 33.9%, against a
chance of 1/3.

### 5.3 Posteriors, 60 s budget, 96 held-out-shape test sessions

| method | top-1 | balanced | mass on truth | H (bits) | ECE |
|---|---|---|---|---|---|
| `B2_classifier` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `B3_state` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `B4_action` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `DA` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `abl_latent_only` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `M5_risk_score` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `abl_action_only` | 0.990 | 0.990 | 0.980 | 0.045 | 0.012 |
| `abl_risk_only` | 0.771 | 0.771 | 0.409 | 1.540 | 0.357 |
| `M2_broad` | 0.333 | 0.333 | 0.333 | 1.585 | 0.000 |
| `B0_prior` | 0.333 | 0.333 | 0.333 | 0.000 | 0.667 |
| `ctrl_wm_shuffled` | 0.490 | 0.490 | 0.390 | 1.474 | 0.111 |
| `ctrl_clf_shuffled` | 0.333 | 0.333 | 0.333 | 1.585 | 0.004 |
| `ctrl_done_only` | 0.333 | 0.333 | 0.333 | 1.585 | 0.000 |

Saturated from the **1 s** budget onward for the model-based methods.

**One control is not at chance and it is not glossed.** `ctrl_wm_shuffled` — the
world model trained with permuted θ labels — reaches 0.490 balanced accuracy
against a chance of 0.333. Its confusion matrix is `[[15,14,3],[5,0,27],
[0,0,32]]`: it calls almost everything the counter-direction candidate and
still gets 15 of 32 target sessions right. The reason is that the *magnitude*
of world-model prediction error differs by domain even without conditioning —
0.75 makes the arm ring, and ringing is harder to predict — so some domain
information survives the permutation. Its temperature had to be fitted to 17.8
and its entropy is 1.474 of a possible 1.585 bits, so the posterior is nearly
uniform and the accuracy comes from tiny residual structure.

The margin over it is still large (1.000 against 0.490) and the classifier and
done-only controls are exactly at chance, so "the model identifies damping"
survives. But on this axis the shuffled-world-model control is **not** a
chance-level control, and any future claim resting on it should use the
classifier's.

### 5.4 Every method that works returns the same distribution

| distribution | methods |
|---|---|
| `[1.000, 0.000, 0.000]` | `B2_classifier`, `B3_state`, `B4_action`, `DA`, `abl_latent_only`, `M5_risk_score`, `M5_risk_aware_da` — **and the known-parameter reference** |
| `[0.333, 0.333, 0.333]` | `M2_broad` |
| `[0.816, 0.093, 0.091]` | `M5_risk_aware` (`M2_broad`, tilted) |

This is the phase's most consequential structural finding, and it is what
item 7 predicted: **the question of who estimates damping most accurately is
not merely uninteresting here, it is degenerate.** Six methods are one PPO run,
not six. The known-parameter reference is not a separate arm either — it is the
same point mass.

`M5_risk_aware_da` is reported precisely because it does nothing: tilting `DA`
moves total variation **0.000**, which is the point-mass limitation of §4.4
demonstrated rather than asserted.

What is left to compare is three-way, and it is exactly the phase's central
question: trajectory matching (= known-parameter) against broad domain
randomisation against the same broad prior risk-tilted.

### 5.5 What one session costs

| stage | seconds |
|---|---|
| world-model scores | 0.182 |
| classifier | 0.094 |
| **total inference on 60 s of arm time** | **≈ 0.28** |

---

## 6. Posterior-guided adaptation — PENDING

Every checkpoint is evaluated at 512 environments × 2400 steps, three process
repeats, in both the target and the source domain.

| arm | training distribution | seeds |
|---|---|---|
| trajectory matching = known-parameter | `[1.000, 0.000, 0.000]` | 8 |
| broad domain randomisation | `[0.333, 0.333, 0.333]` | 8 |
| risk-aware (broad, tilted) | `[0.816, 0.093, 0.091]` | 8 |
| **mixture, untilted** (α = 0.5) | `[0.500, 0.500, 0.000]` | 8 |
| **mixture, risk-tilted** (α = 0.5) | `[0.898, 0.102, 0.000]` | 8 |
| source-prior refit | `[0.000, 1.000, 0.000]` | 3 |

### 6.1 What is expected, written down before the runs finished

This section was written while the queue was still running and has not been
edited since. It is here so that §7's outcome cannot be dressed up afterwards.

**C2 — risk-aware against trajectory matching — is expected to FAIL, and the
reason is structural rather than a fact about risk-awareness.** Every estimator
on this axis returns a point mass at the target (§5.4). Trajectory matching
therefore spends **100%** of its PPO budget in the dangerous domain and the
risk-aware arm spends **81.6%** of its budget there. On target-domain safety
the arm with more exposure to the target should win. A point mass cannot be
tilted, so there was no uncertainty for risk-awareness to act on and C2 asks a
question with no mechanism behind it. It is still reported, because the phase
specification asks for it in those words.

**C3 — risk-aware against broad domain randomisation — is expected to PASS**,
for the same structural reason read the other way: 81.6% of the budget in the
target domain against 33.3%. This is a real effect and it is also a weak claim.
It says the tilt moved mass towards the domain that turned out to be the true
one, which on a three-value axis where the truth is also the most dangerous
candidate is close to unfalsifiable.

**C4 is the comparison that can actually fail.** The α = 0.5 pair holds
everything constant except whether the surviving uncertainty is resolved
towards danger: same posterior, same mixing, same PPO budget, same amount of
uncertainty, `[0.898, 0.102, 0]` against `[0.500, 0.500, 0]`. If risk-awareness
buys safety anywhere in this phase, it is here. If it does not buy it here, the
mechanism does not work and C3 was measuring the tilt's direction rather than
its value.

**A caveat that applies to C3 and C4 both, and that no amount of seeds fixes.**
The cost vector the tilt reads is a *simulator* quantity, and on this axis the
most dangerous candidate happens to be the true one. A tilt towards danger is
therefore also a tilt towards truth, and the two cannot be separated by this
experiment. The counter-direction candidate at 1.5 is in the design partly to
limit this — it is *safer* than nominal, so the tilt actively moves mass away
from it — but a clean separation would need an axis whose dangerous candidate
is not the target, which this phase was told not to open.

---

## 7. Gate — PENDING

---

## 8. Statistics

Trips: negative binomial with the **training seed** as the observation, and a
cluster-robust Poisson sandwich beside every rate ratio (§1.4). Throughput and
retention: t intervals over seed means, so a configuration with many repeats of
few seeds cannot look better resolved than one with many seeds. Holm correction
across the pre-registered comparisons. With eight clusters the sandwich is
anticonservative even with the `G/(G-1) · (N-1)/(N-p)` correction, and the
number of clusters is printed next to every interval.

---

## 9. Reproducing this

```bash
# data, models, risk head
scripts/wm1b_collect.sh
python scripts/wm_manifest.py --data results/wm1_damping/data
python scripts/wm_train.py      --axis servo_damping_scale --data results/wm1_damping/data
python scripts/wm_classifier.py --axis servo_damping_scale --data results/wm1_damping/data
python scripts/wm_risk.py --horizon 10 --stride 10 --epochs 6 --device cuda:0

# posteriors and the risk tilt
python scripts/wm_posterior.py --axis servo_damping_scale \
    --data results/wm1_damping/data --model results/wm1_damping/model \
    --out results/wm1_damping/posterior \
    --risk-head results/wm1_damping/model/risk_head.pt --device cuda:0

# adaptation
python scripts/wm1_adapt_plan.py --axis servo_damping_scale --stage formal \
    --alpha 1.0 --posteriors results/wm1_damping/posterior/posteriors_60s.json \
    --methods B3_state,M2_broad,M5_risk_aware \
    --only B3_state,M2_broad,M5_risk_aware,KNOWN_PARAM \
    --seeds 42,20260824,31415927,7,13,101,2718,31415 \
    --out results/wm1_damping/adapt/formal.json
python scripts/wm1_queue.py --plan results/wm1_damping/adapt/formal.json --gpus 4,5,6,7

# analysis
python scripts/wm1_seeds.py --adapt results/wm1_damping/adapt

# checks
python scripts/check_damping.py
micromamba run -n mjlab python -m pytest tests -q
```
