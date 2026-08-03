# Frozen protocol: minimal MuJoCo planar pushing PPO baseline

## Scope

One cube, one cylindrical mocap pusher, fixed physics, privileged low-dimensional
state, position-only target, and PPO trained from random initialization. There is
no vision, orientation target, recurrent memory, parameter randomization,
identification, multi-object logic, planner, or LivingTwin source/checkpoint use.

## Simulation

- MuJoCo 3.3.7, physics timestep 0.002 s.
- Policy frame skip 10, therefore control timestep 0.020 s (50 Hz).
- Cube: 0.05 m side length, fixed mass/contact parameters.
- Pusher: vertical cylinder, radius 0.012 m, controlled as a mocap body.
- Episode: 200 policy steps (4.0 s maximum).

## Observation

The raw 13-dimensional float vector is:

```text
[pusher_x, pusher_y, pusher_vx, pusher_vy,
 cube_x, cube_y, sin(cube_yaw), cos(cube_yaw),
 cube_vx, cube_vy, cube_omega_z,
 target_x-cube_x, target_y-cube_y]
```

PPO uses a RunningMeanStd normalizer. Evaluation uses the saved statistics in
read-only mode. Training and evaluation call the same environment and action
mapping.

## Action

The policy produces a tanh-squashed two-dimensional action in `[-1, 1]^2`.
The environment clips once more for safety and maps it to pusher x/y velocity:

```text
velocity_mps = clip(action, -1, 1) * 0.25
```

The PPO log probability includes the tanh change-of-variables correction. The
action used in the likelihood is therefore the action executed by the environment.

## Reward

Only three terms are used:

```text
reward = -2.0 * cube_target_distance_m
         + 10.0 * first_success_event
         - 0.001 * sum(normalized_action ** 2)
```

There is no yaw reward, pusher shaping, contact reward, curriculum, or planner
cost.

## Success and termination

- Success: cube-target Euclidean distance <= 0.025 m for 5 consecutive steps.
- Terminate on success, 200-step timeout, or cube leaving the frozen workspace.
- Target orientation is absent.

## PPO

- Separate actor and critic MLPs: 13 -> 128 -> 128 -> outputs, tanh hidden units.
- From-scratch orthogonal initialization for every seed.
- Tanh-squashed diagonal Gaussian actor.
- GAE, clipped PPO objective, value loss, entropy diagnostic, gradient clipping.
- Checkpoints contain actor, critic, log standard deviation, optimizer,
  observation normalization, RNG states, config, and global step.

## Stage gates

1. A--F preflight must all pass.
2. 64-environment smoke must have finite metrics, non-zero parameter changes,
   bounded actions, contacts, and no checkpoint/evaluation inconsistency.
3. Only then run three frozen baseline seeds.
4. Target: each seed reaches >=90% deterministic success. If not, stop at the
   baseline and diagnose the minimal PPO pipeline; do not expand scope.

