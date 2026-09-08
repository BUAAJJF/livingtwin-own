# Object-Level Physical Memory for Manipulation
## High-Level Research Route for Codex / Claude

> This document describes the current high-level research route and research objectives only.
> It is a research extension of the current repository, not a description of features that
> already exist. The repository is a PiPER-X continuous tabletop pick-and-place system
> with a privileged state teacher, student-rollout DAgGER, vision/point-cloud PPO, and
> D455 deployment.
> The first goal is to make the complete story work end to end before adding more complex extensions.

---

## 1. Research Goal

We want to study the following question:

> **Can a robot persist object-specific physical experience acquired from one interaction and reuse it when the same object reappears, before sufficient new physical interaction is available, in order to improve manipulation performance?**

The first version focuses on one explicit physical property:

```text
object mass
```

The long-term direction is:

```text
mass
→ mass + friction + COM + compliance + ...
→ implicit object-level physical latent z_obj
```

The core contribution is not simply mass prediction.

The target closed loop is:

```text
Recognize
→ Retrieve
→ Act
→ Interact
→ Estimate
→ Verify
→ Update
```

## Repository Baseline

The current code already randomizes object shape, mass, friction, and the derived
centre of mass in simulation. Its actor sees proprioception plus object state (or
the D455/depth/point-cloud route); `object_physics` is currently critic-only and
contains mass, friction, and COM. There is no object identity, mass predictor, or
persistent Memory Bank yet.

The recurrent GRU in the vision and point-cloud policies is temporal control state.
It is reset at episode boundaries and must not be treated as object memory. The
first implementation therefore needs a small actor-observation change (`mass_gt`,
normally normalized) and a separate identity/memory evaluation harness; it does
not require replacing the existing teacher → DAgger → PPO pipeline.

---

## 2. Core Difference from Existing Work

### RMA / RMA²

The core idea is:

```text
current interaction history
→ adaptation / physics representation
→ adaptive policy
```

It mainly answers:

> How can the robot rapidly adapt to hidden physical properties from the current interaction?

Our target is one step further:

```text
interaction
→ physical estimate
→ bind estimate to object identity
→ persistent memory
→ retrieve at the next encounter
```

In short:

> **RMA² learns how to adapt from the current interaction; we want to remember that adaptation for the next encounter.**

---

### Dynamic Pick-and-Place with Explicit Mass Estimation

Recent work has shown that the following route is feasible:

```text
interaction history
→ mass estimate
→ mass-adaptive RL policy
```

Our additional research question is:

> If the same physical object has already been manipulated before, why should the robot re-estimate its mass from scratch every time it reappears?

---

### Phys2Real

We mainly borrow the following ideas:

```text
Phase 1:
ground-truth physical parameter
→ conditioned RL policy

Phase 1.5:
noisy physical parameter
→ policy robustness fine-tuning

Phase 2:
interaction history
→ physical parameter estimate
```

For the current mass-only V0:

- the teacher directly receives `mass_gt`;
- no additional learned mass embedding `e` is required;
- Phase 1.5 should not be inserted into the initial teacher training;
- after the mass predictor is trained, the final vision policy can be fine-tuned with predicted / noisy / stale mass values.

This is cleaner because the robustness distribution can be based on the actual prediction errors of the learned mass estimator instead of arbitrary synthetic noise.

If the method later expands to multiple physical properties, we may introduce:

```text
[mass, friction, COM, compliance, ...]
→ physics encoder
→ compact physical embedding
```

---

### CoRAL

CoRAL is useful mainly for future cold-start estimation:

```text
unseen object
→ visual / VLM estimate
→ initial physical prior
```

This is not the core of the first Memory Bank implementation.

---

### PhysMem

We mainly borrow the following ideas:

```text
memory confidence
verification before trust
stale-memory detection
update / overwrite / versioning
```

A key principle is:

> **Identity determines where to retrieve; interaction determines whether the retrieved physics should still be trusted.**

---

## 3. Overall Architecture

```text
                   RGB-D / Object Observation
                             ↓
                 Object Recognition / Key
                             ↓
                       Memory Search
                             ↓
             ┌───────────────┴───────────────┐
             │                               │
        Known Object                    Unknown Object
             │                               │
             ↓                               ↓
    retrieve remembered mass          initial mass prior
    + confidence                     (simple prior in V0;
             │                        VLM / similar memory later)
             └───────────────┬───────────────┘
                             ↓
                    mass_used + confidence
                             ↓
                 Mass-Conditioned PPO
                             ↓
                          Action
                             ↓
                       Interaction
                             ↓
                    Recent History
                             ↓
                    History Encoder
                             ↓
                         z_last
                             ↓
                       Mass Head
                             ↓
              mass_fast + uncertainty
                             ↓
                Verify / Fuse / Update
                             ↓
                       Memory Bank
```

---

# 4. Training Route

## Stage A — Mass-Conditioned State Teacher

Build on the existing state-teacher pipeline. The current state actor is not given
the full privileged physics vector, so `mass_gt` should be added as a separate
scalar actor observation while friction and COM remain critic-only for V0.

Train:

```text
state observation (proprio + object state)
+
mass_gt
↓
PPO Teacher
↓
action
```

Formally:

```math
 a_t = \pi_T(s_t^{state}, m_{gt})
```

Object mass is randomized in simulation.

### Important

The same object geometry / appearance must be allowed to appear with different masses.

Avoid a shortcut such as:

```text
Object A appearance
→ fixed mass
```

### Why no learned mass embedding in V0?

The first version only uses one scalar physical variable.

Directly conditioning the policy on normalized mass is simpler and more interpretable:

```text
mass_gt
→ policy
```

A learned physics embedding can be introduced later when multiple physical properties are modeled jointly.

---

## Stage B — Vision Student with DAgger

Preserve the current training structure:

```text
Mass-conditioned State Teacher
→ DAgger
→ Vision Student
```

At this stage, both teacher and student still receive `mass_gt`.

Teacher (after the Stage-A change):

```text
state observation + mass_gt
```

Student:

```text
vision / depth / point cloud + proprioception + mass_gt
```

For the current point-cloud line, keep `vision_meta` and the existing GRU; the
new scalar is simply another 1D actor input.

This stage should solve only:

```text
privileged-state policy
→ deployable vision policy
```

It should not simultaneously learn mass inference from interaction history.

---

## Stage C — Mass-Conditioned Vision PPO

After DAgger, continue with vision PPO fine-tuning.

Actor:

```text
vision / depth / point-cloud observation
+ proprioception
+ mass_gt
```

Critic:

```text
full state observation + privileged physics
```

The existing `object_physics` term already exposes mass, friction, and COM to the
critic. Only mass is promoted to the actor in this V0.

The goal is to obtain a strong:

```text
mass-conditioned vision manipulation policy
```

This stage also provides the Oracle-Mass upper bound:

> How much can manipulation improve if the object mass is perfectly known?

---

## Stage D — History Mass Predictor

After the mass-conditioned policy is stable, collect interaction trajectories under different masses.

Useful data sources include:

```text
final policy
intermediate checkpoints
action perturbations
gripper perturbations
successes
failures
slip
recovery
```

Train:

```text
recent observation / action history
+
optional visual or depth feature
↓
History Encoder
↓
z_last
↓
Mass Head
↓
mass_fast
```

The current policy GRU only produces actions. Reuse its temporal inputs if useful,
but add an explicit mass head (or a separate history model) rather than reading a
mass estimate from the hidden state by convention. Available first-version signals
are the existing proprioception, actions, pad/gripper signals, and D455/depth or
point-cloud observations; do not assume force/torque sensing that is not in the
current environment.

Formally:

```math
z_{last} = E(H_t)
```

```math
m_{fast} = D_m(z_{last})
```

The supervision directly uses simulator ground-truth mass:

```math
L_{mass} = \operatorname{Huber}(m_{fast}, m_{gt})
```

Future versions may predict uncertainty as well:

```text
mass_mean
mass_std
```

### Important

The first version should not introduce:

- a full world model;
- nominal-vs-randomized dynamics residual learning;
- long-horizon future-state prediction.

The first objective is simply:

```text
interaction history
→ useful mass estimate
```

---

## Stage E — Estimated-Mass PPO Fine-Tuning

Earlier policy stages use:

```text
mass_gt
```

Deployment will use:

```text
predicted mass
retrieved mass
fused mass
```

Therefore the final actor should be fine-tuned for imperfect physical estimates.

This stage borrows the idea of Phys2Real Phase 1.5, but should preferably use:

```text
actual predictor error distribution
+
controlled synthetic noise
+
occasional stale / incorrect memory
```

instead of adding arbitrary noise during the initial teacher training.

Final actor input:

```text
vision
+ proprioception
+ mass_used
+ optional confidence
```

The policy does not need to know the source of `mass_used`.

---

# 5. Physical Memory Bank

The first version does not require a complex latent memory.

A single object-memory entry may be:

```text
ObjectMemory {
    object_key / object_id
    mass_mean
    mass_uncertainty
    confidence
    interaction_count
    last_update
    optional representative z_last
}
```

Here:

```text
object_key
```

answers:

> Which object is this?

while:

```text
mass_mean / future z_obj
```

answers:

> What physical property do I remember about this object?

These two concepts should remain separate.

The bank must live outside the episode-local policy state so it survives resets.
`PickCommand.target` is current task bookkeeping, not a persistent identity. A V0
identity can be an injected simulator `object_id`; learned visual recognition is a
later replacement.

---

# 6. V0 Retrieval Strategy

The first complete implementation should use:

```text
Oracle Object ID
```

This is a new controlled benchmark signal, not an existing observation. In
particular, the point-cloud route `P1BT` carries an oracle per-frame target label
for diagnostics and is explicitly not deployable; it is not a persistent object
identity and must not be used as the memory key.

The first question to answer is:

> If memory retrieval is always correct, does persistent physical memory actually improve manipulation?

Only after this is validated should visual instance recognition be introduced.

---

# 7. Runtime Logic

## Case 1 — Never-Seen Object

V0:

```text
no memory
↓
default / global mass prior
↓
mass-conditioned policy
↓
interaction
↓
history mass predictor
↓
mass_fast
↓
store into memory
```

Future extensions may use:

```text
similar-object retrieval
VLM mass prior
visual mass regression
```

---

## Case 2 — Previously Seen Object

```text
recognize Object A
↓
retrieve Memory[A]
↓
mass_obj + confidence
↓
before new physical interaction:
mass_used = mass_obj
↓
policy acts immediately using remembered physics
```

This is the central capability:

```text
pre-contact physical recall
```

---

## Case 3 — Memory Verification

After interaction, obtain:

```text
mass_fast
```

and compare it with:

```text
mass_obj
```

If they are consistent:

```text
memory confidence ↑
fuse new evidence
```

If they are significantly inconsistent:

```text
memory confidence ↓
trust current interaction more
collect more evidence
update / overwrite / version memory
```

---

# 8. Main Experimental Protocol

The current continuous task redraws shape/mass/friction on successful placement
by default (`reshape_on_place`); that is useful for robustness training but does
not preserve an object's identity. Recall experiments therefore need a small
sequence harness that assigns stable IDs and explicitly controls redraw cadence
(`redraw_on_place`) across encounters. Keep this benchmark separate from the
existing throughput ruler.

## Positive Recall

```text
A_200g
→ B
→ C
→ A_200g
```

Compare the second encounter with A using:

```text
Online-only adaptation
vs
Persistent memory
vs
Oracle mass
```

Main question:

> Does memory improve first-attempt manipulation performance on a returning object?

---

## Stale Memory

```text
A_200g
→ B
→ C
→ A_700g
```

Object identity and appearance remain unchanged while mass changes.

Main question:

> Can the system detect that old physical memory is stale and recover quickly?

---

## Future Generalization

Later extensions may include:

```text
New-Similar
New-Novel
```

for studying:

```text
similar-memory interpolation
+
VLM / visual physical prior
```

---

# 9. Primary Metrics

The main contribution should not be evaluated by mass reconstruction accuracy alone.

Primary manipulation metrics:

```text
1. First-attempt success rate
2. Pick-to-place / time-to-success
3. Throughput (objects/min)
4. Re-grasp / drop / recovery count
5. Control effort / energy proxy
6. Stale-memory recovery time
7. Negative-transfer rate
```

The mass predictor should also be evaluated separately:

```text
mass MAE / RMSE
mass error vs time after informative contact
uncertainty calibration, if used
```

For the existing deployment line, retain its fixed 36 s episodes, three evaluation
seeds, median plus spread, and late/early throughput ratio. The memory benchmark
adds first-attempt and stale-memory metrics; it should not replace that ruler.

---

# 10. Main Baselines

At minimum:

```text
1. Mass-Agnostic PPO
2. Oracle-Mass PPO
3. Online-Only Mass Adapter
   (RMA² / interaction-only style)
4. Naive Persistent Mass Memory
   (no verification)
5. Ours:
   Persistent Memory + Confidence + Verification
```

The current repository supplies the mass-agnostic actor baseline (with a
privileged critic). Oracle-mass, online adaptation, and persistent-memory actors
are new experimental variants, not existing checkpoints.

Future baselines may include:

```text
visual / VLM prior
similar-object retrieval
learned object recognition
```

---

# 11. z_last → z_obj

This is an important future extension.

Current:

```text
history
→ z_last
→ mass_fast
```

`z_last` is the short-term physical representation extracted from the most recent interaction.

Future:

```text
z_last^1
z_last^2
z_last^3
...
↓
Memory Aggregator
↓
z_obj
```

`z_obj` is intended to represent a persistent object-level physical representation across interactions and episodes.

V0 does not need to solve this immediately.

The first complete system can treat:

```text
z_obj
≈ persistent explicit mass belief
```

and validate the full closed loop first.

---

# 12. Current Non-Goals

The first version should not make the following components core dependencies:

```text
full world model
VND-style nominal dynamics residual
MPPI
trajectory optimization
HALO-style long-history retrieval
Force-VAE
online PPO weight updates on hardware
complex VLM reasoning
learned visual instance retrieval
multi-property implicit z_obj
```

These can be added after the basic physical-memory story is validated.

---

# 13. References and What We Borrow

## RMA²
Rapid Motor Adaptation for Robotic Manipulator Arms

Borrow:

```text
privileged physics-conditioned policy
+
history-based adaptation
```

Difference:

```text
RMA²:
current interaction adaptation

Ours:
persistent object-specific physical memory across encounters
```

---

## Learning Dynamic Pick-and-Place for a Legged Manipulator
https://arxiv.org/pdf/2605.15713

Borrow:

```text
explicit mass estimation
+
mass-aware RL manipulation
```

---

## Phys2Real
https://arxiv.org/pdf/2510.11689

Borrow:

```text
direct interpretable physical-parameter conditioning
optional noisy-parameter policy fine-tuning
uncertainty-aware physical estimate
visual prior + interaction estimate
```

Current decision:

```text
No extra mass latent e in V0.
Use GT mass for the teacher.
Use Phase-1.5-style robustness only after the predictor exists.
```

---

## CoRAL

Borrow later:

```text
VLM-based cold-start physical prior
```

---

## PhysMem

Borrow:

```text
confidence
verification
stale-memory handling
persistent update
```

---

# 14. Instructions to Codex / Claude

Before proposing implementation changes:

1. Read the current repository and existing handoff / CLAUDE documentation.
2. Do not assume the code structure from this document.
3. Map the stages above onto the actual current training pipeline.
4. Identify the smallest changes required to first achieve:
   ```text
   Mass-Conditioned State Teacher
   ```
5. Preserve the existing:
   ```text
   teacher → DAgger → vision PPO
   ```
   pipeline unless code inspection shows a compelling reason otherwise.
6. Do not implement the entire memory system in one pass.
7. After reading the codebase, produce a concrete implementation plan and list uncertainties before large architectural changes.

Recommended implementation order:

```text
A. Add normalized mass_gt to the state actor and train the mass-conditioned teacher
B. Propagate mass_gt through DAgger and the vision / point-cloud actor
C. History mass predictor
D. Estimated-mass PPO fine-tuning
E. Oracle-ID sequence benchmark and persistent memory
F. Confidence / stale-memory verification
G. Learned retrieval / VLM prior
H. Implicit z_obj
```

---

## One-Sentence Summary

> **Learn how to manipulate with known mass, infer mass from interaction, remember that estimate for a specific object, reuse it before the next interaction, and revise it when new evidence shows that the object's physics has changed.**
