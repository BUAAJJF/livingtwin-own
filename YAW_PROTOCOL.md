# Frozen protocol: single-cube position + yaw PPO baseline

This stage extends the accepted position-only baseline without modifying its
environment, configuration, checkpoints, or results. It uses a separate MuJoCo
asset, environment class, configuration namespace, scripts, and result paths.

## Fixed scope

- One 0.05 m cube and one cylindrical mocap pusher.
- Fixed nominal physics; no training randomization.
- Privileged state, feed-forward PPO, position and yaw target only.
- 50 Hz control, 0.25 m/s maximum pusher speed, 200 steps / 4 seconds.
- No arm, vision, recurrence, identification, planner, or LivingTwin code.

## Observation (15 dimensions)

```text
0  pusher_x_m                 workspace [-0.16, 0.20]
1  pusher_y_m                 workspace [-0.16, 0.16]
2  pusher_vx_mps              [-0.25, 0.25]
3  pusher_vy_mps              [-0.25, 0.25]
4  cube_x_m                   safety bounds [-0.20, 0.24]
5  cube_y_m                   safety bounds [-0.18, 0.18]
6  sin_cube_yaw               [-1, 1]
7  cos_cube_yaw               [-1, 1]
8  cube_vx_mps                finite, not hard-clipped
9  cube_vy_mps                finite, not hard-clipped
10 cube_omega_z_radps          finite, not hard-clipped
11 target_minus_cube_x_m       approximately [-0.165, 0.32]
12 target_minus_cube_y_m       approximately [-0.23, 0.23]
13 sin_delta_yaw              [-1, 1]
14 cos_delta_yaw              [-1, 1]
```

`delta_yaw = wrap_to_pi(target_yaw - cube_yaw)`. RunningMeanStd is saved in
every checkpoint and frozen during deterministic evaluation.

## Reset distributions

- Existing cube, pusher, and target-position distributions are unchanged.
- Smoke target yaw: `[-pi/4, pi/4]`.
- Formal target yaw: `[-pi, pi)`.
- Initial cube yaw: unchanged `[-0.10, 0.10]` rad.

## Action, reward, success, and termination

The tanh-squashed 2-D action maps to pusher x/y velocity at 0.25 m/s maximum.
The PPO log probability includes the tanh Jacobian correction.

```text
reward = -2.0 * position_error_m
         -0.15 * abs(wrap_to_pi(delta_yaw))
         +10.0 * first_sustained_success
         -0.001 * sum(action ** 2)
```

Success is position error <= 0.025 m AND absolute yaw error <= 10 degrees for
five consecutive control steps. The episode terminates on sustained success,
the unchanged safety boundary, or the unchanged 200-step horizon. Training and
evaluation use identical definitions.

## Stage gates

Before PPO: signed yaw controllability, 32-state heuristic improvement, reward
direction, angle wrap continuity, LR=0, and full checkpoint round-trip must pass.
Smoke uses 64 environments, 65,536 steps, seed 1201. Formal seeds are 2201,
2202, and 2203 with 1,048,576 steps each. Common evaluation uses 100 frozen
states; final generalization uses 500 disjoint states.

Stage 1 passes only if every seed reaches >=90% common joint success and >=85%
generalization joint success, with no NaN/Inf, persistent saturation, metric
collapse, checkpoint mismatch, or collision/boundary exploit.
