# PiPER-X cube pushing — end-to-end joint-space PPO

An AgileX PiPER-X with its gripper shut pushes a 50 mm cube to randomly placed
goals on a table. Everything is end to end: the policy reads proprioception,
its own end-effector pose, the cube's pose and the goal, and writes six joint
position targets at 50 Hz. There is no IK, no scripted push primitive, and no
learned dynamics model.

Built on [mjlab](https://pypi.org/project/mjlab/) (MuJoCo-Warp physics on GPU)
with rsl_rl PPO, so thousands of environments run in parallel and a usable
policy takes minutes rather than days.

> Earlier work on this problem — an IK-driven staged push primitive, a learned
> local dynamics twin, and a CEM planner — lives on the `piperx/cube-policy-v2`
> and `piperx/puck-sustained-push` branches.

## Task

| | |
|---|---|
| Observation (45-D) | 6 joint positions, 6 joint velocities, end-effector pose (position + 6-D rotation), cube pose (position + 6-D rotation), cube linear velocity, EE→cube, cube→goal, goal phase, last action |
| Action (6-D) | joint position targets for `joint1`…`joint6`, offset from the home pose |
| Gripper | permanently shut — its actuator is never written, so it holds `ctrl = 0` |
| Goal | a point on the table 7–16 cm from the cube; resamples on a timer **and** the moment it is reached |
| Objective | reach as many goals as possible per episode, with minimum mechanical work and no jitter |

Episodes are a fixed 8 s and never end on success, so "fast" is rewarded
directly: more goals fit in the same wall clock.

## Setup

One shared micromamba environment is used for every mjlab project on the
machine, so torch and Warp are installed once rather than per repository.

```bash
git clone --recurse-submodules <this repo> && cd LivingTwin
git submodule update --init              # PiPER-X URDF + meshes

micromamba create -y -n mjlab -c conda-forge python=3.11 pip
micromamba run -n mjlab pip install "mjlab==1.6.0"
micromamba run -n mjlab pip install -e .

micromamba run -n mjlab list-envs --keyword Push   # -> Mjlab-Push-Cube-PiperX
```

## Look before you train

```bash
scripts/play.sh          # zero-action policy, 4 envs
```

Check that the gripper is shut and stays shut, that the cube rests flat, that
the goal marker sits on the table, and that the arm is not sweeping through the
floor.

## Train

```bash
tmux new -s piperpush
NUM_ENVS=8192 ITERS=20  RUN_NAME=smoke  scripts/train.sh   # measure throughput
NUM_ENVS=8192 ITERS=500 RUN_NAME=v1     scripts/train.sh   # the real run
```

Size the real run from the smoke run's reported steps/s rather than guessing.
For more speed, `GPUS="[0, 1, 2, 3]"` fans out over GPUs with torchrunx.

Metrics worth watching in wandb: `Metrics/push_goal/goals_completed` (goals per
episode — the headline number), `Metrics/push_goal/planar_error`,
`Episode_Reward/mech_power`, and `Episode_Reward/action_rate`.

Checkpoints land in
`logs/rsl_rl/piperx_push_cube/<timestamp>_<run_name>/model_<iter>.pt`.

## Replay

```bash
scripts/play.sh logs/rsl_rl/piperx_push_cube/<run>/model_500.pt
```

Add `--video True --video-length 1000` to record instead of render, which is
what you want on a headless box.

## Layout

```
src/piper_push/
├── robot.py                     PiPER-X as a closed-gripper pusher
├── cube.py                      the 50 mm cube
└── tasks/push_cube/
    ├── env_cfg.py               scene, observations, rewards, terminations
    ├── mdp.py                   push goal command + the terms mjlab lacks
    └── rl_cfg.py                PPO hyperparameters
piperx-mjlab/                    submodule, pinned — URDF and meshes only
```
