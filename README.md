# PiPER-X tabletop tidying — vision policies on a D455 + PiPER-X

An AgileX PiPER-X clears objects from a table into a bin, continuously: pick
one up, put it in, the table refills. The deployed policy sees the scene
through a single fixed third-person RealSense D455 and writes six joint
position targets plus a gripper command at 50 Hz. There is no IK, no scripted
primitive, and nothing on the robot that reads object state.

Built on [mjlab](https://pypi.org/project/mjlab/) (MuJoCo-Warp on GPU) with
rsl_rl, so thousands of environments run in parallel.

## Where things stand

- **Hardware.** On 2026-09-01 the mask-based policy `d455_v4_final` picked
  objects off the table and placed them in the bin from the calibrated D455
  alone (`recordings/v4_stereo_try3`). It is Action API v1 and remains the
  rollback for the arm. What that day taught: [`docs/hardware_lessons.md`](docs/hardware_lessons.md).
- **Current line (`yf/pc`, 2026-09-06).** Mask-free students that see a
  workspace point cloud (or raw metric depth) instead of a segmented target,
  distilled from the qualified Action-API-v2 state teacher `v11_nosight_36s`.
  Four routes were trained and evaluated on one ruler; the best (P1B,
  point-patch transformer) reaches a third of the teacher's throughput and
  decays within an episode, so the first generation is **NO-GO for real
  motion and GO for shadow runs**. Report: [`results/pc/REPORT.md`](results/pc/REPORT.md);
  runbook: [`docs/pc_shadow_runbook.md`](docs/pc_shadow_runbook.md).
- **Two facts that constrain every number here.** The simulator is not
  reproducible run to run (identical commands span ~3%), so quote a median
  over three evaluation seeds with the spread. And a throughput mean hides a
  within-episode collapse, so quote the late/early ratio beside it.

## The task

| | |
|---|---|
| Actor observation | proprioception (joint positions and velocities, end-effector pose, gripper opening, pad contacts, gripper servo error, last action) plus the camera: for the mask line a 3-channel 224×168 image (scene depth, target mask, masked depth); for the point-cloud line 512 base-frame points with a validity flag, or the raw depth, plus `vision_meta` (frame age, fresh, valid) |
| Critic observation | the above, uncorrupted, plus object pose/velocity/shape and privileged physics — training only |
| Action | 6 joint targets + gripper. The policy emits `u`, the action term applies `a = tanh(u)`; `a = ±1` is the safe target clip on every joint and 0–50 mm on the jaw. Rate-limited to 0.62 × the safety-shell trip speed |
| Objects | five shape classes, 25–45 mm wide, 24–90 mm tall, 50–400 g, friction 0.4–1.0, redrawn **per object**; the `capped` class is held out for the point-cloud routes |
| Episode | fixed length (36 s for the current line); success never ends it, so throughput is rewarded directly |
| Camera timing | 30 Hz vision through the 50 Hz loop (3 fresh frames per 5 steps, per-episode phase), 0–4 control steps of processing latency, measured D455 noise, holes and dropout |

**Action convention (Action API v2, since 2026-09-05).** Checkpoints and
exported specs carry an `action_api` stamp (`piper_push.action_api`) that every
loader checks. There is no `atanh` anywhere and no v1 → v2 translation; a
pre-2026-09-05 checkpoint (v3–v9, `d455_v4_final`) loads only on the `-V1`
task ids with `--allow-legacy-action-api`, and the deploy mapper follows the
spec's `action_spec` block so v4 keeps driving the arm it was trained on.

## The pipeline

```text
state teacher (PPO, capability-gated cold-start curriculum, heavy DR)   scripts/run_v10c.sh, scripts/pc/continue_teacher.sh
   -> vision student (student-rollout DAgger, MSE on tanh(u))            scripts/distill.py
   -> vision actor + privileged state critic (PPO fine-tune)             scripts/finetune.py
   -> evaluation on one ruler                                            scripts/accept_s1.py, eval_endurance.py, eval_occlusion.py, pc/eval_actions.py
   -> export + bundle (ONNX / TorchScript, explicit GRU state)           scripts/check_export.py, scripts/pc/bundle.py
   -> shadow run, then guarded motion                                    hardware/deploy/pc_run.py, hardware/deploy/run.py
```

`scripts/pc/run_route.sh` runs one route end to end (smoke → distill →
distill eval → finetune → 3-seed eval + held-out + actions + occlusion →
export) into `results/pc/routes/<tag>/` with a manifest; `scripts/pc/report_routes.py`
tabulates the routes and applies the deployment gate.

## Setup

```bash
git clone --recurse-submodules <this repo> && cd LivingTwin
micromamba create -y -n mjlab -c conda-forge python=3.11 pip
micromamba run -n mjlab pip install "mjlab==1.6.0"
micromamba run -n mjlab pip install -e .
micromamba run -n mjlab list-envs --keyword Pick
```

Two settings any non-interactive run needs (both baked into `scripts/eval.sh`
and the `scripts/pc/*.sh` drivers):

```bash
export MUJOCO_GL=disable                       # mujoco initialises a GL backend it never uses
export LD_LIBRARY_PATH=$MAMBA_ROOT/envs/mjlab/lib:$LD_LIBRARY_PATH   # libicui18n.so.78 needs CXXABI_1.3.15
```

## Evaluate, test

```bash
scripts/eval.sh Mjlab-Pick-Place-PiperX-PC-P1B-Vision <checkpoint.pt> my_run   # accept_s1, 512 x 2400, provenance JSON
micromamba run -n mjlab python -m pytest tests -q                                # 408 tests, ~35 s
```

Every evaluation goes through `piper_push.evalcfg.load_weights` (raises if the
weights did not arrive), takes `--sensor measured` (the task's own trained
sensor; a student evaluated with the sensor off is out of distribution), and
resets the GRU on episode boundaries. Where the numbers go and what is
tracked: [`results/README.md`](results/README.md).

## Running on the arm

Read [`hardware/deploy/README.md`](hardware/deploy/README.md) (calibration,
bring-up order, guards) and, for the point-cloud policies,
[`docs/pc_shadow_runbook.md`](docs/pc_shadow_runbook.md). Every real-motion run
needs the physical emergency stop in hand, a clear workspace, a **fresh**
`--record` directory and the typed word `move`. Restore the table first:

```bash
micromamba run -n mjlab python -m hardware.deploy.scene --like recordings/v4_stereo_try3
```

The rollback, the only policy that has placed objects on the arm (mask
pipeline, Action API v1):

```bash
micromamba run -n mjlab python -m hardware.deploy.run \
    --policy hardware/deploy/policies/d455_v4_final \
    --camera d455 --mask depth --policy-device cpu \
    --record recordings/<fresh dir> --seconds 20 \
    --home-first --command-rate-scale 0.6 \
    --depth-source stereo --allow-legacy-action-api
```

`--home-first` because every training episode began at the home pose;
`--command-rate-scale 0.6` because occlusion scales with speed; no
`--min-grasp-height` or `--min-table-clearance` because the task allows light
fingertip contact and a setpoint-based floor stopped healthy runs. A
point-cloud bundle runs through the same guarded loop with `--obs pc` (the
mask-only flags are refused):

```bash
micromamba run -n mjlab python -m hardware.deploy.run --obs pc \
    --policy hardware/deploy/policies/pc_P1B_20260906T0319 --camera d455 --no-arm \
    --seconds 20 --record recordings/pc_noarm_<try>          # camera + dry arm first
```

Afterwards: `python -m hardware.deploy.review recordings/<session>` renders the
session; `scripts/logview.sh` serves every recording at full frame rate;
`scripts/pc/viewer.py` produces a frame-by-frame page of a simulated rollout
with the cloud, the camera's RGB and the corrupted depth.

## Layout

```
src/piper_push/
├── robot.py, objects.py, shapes.py, camera.py       scene, object distribution, camera model
├── depth_noise.py, d455_noise.py                    the measured D455, in simulation
├── actions.py, squashed.py, action_api.py           bounded action term, tanh head, the API stamp
├── latency.py                                       per-environment observation delay (robust task)
├── models.py, distill.py, checkpoints.py, evalcfg.py  recurrent model, DAgger runner, conversion, the evaluation ruler
├── pc/                                              point-cloud line: cloud.py (capture, cadence, ring), encoders.py, models.py, grasp.py
└── tasks/pick_place/                                env_cfg, mdp, rl_cfg, robust_cfg (heavy DR), cold_curriculum, pc_cfg, registrations
scripts/
├── accept_s1.py, eval.sh, eval_endurance.py, eval_occlusion.py   the ruler
├── distill.py, finetune.py, train.sh, run_v10c.sh, v10c_verdict.py  training
├── check_export.py, export_obs_spec.py, student_to_actor.py        export
├── pc/                                              run_route, continue_teacher, eval_teacher, eval_actions, smoke, reports, bundle, viewer
└── sight_viewer.py, sim_perception_check.py, record_vision.py, rig_to_sim.py   mask-line diagnostics, rig-to-sim check
hardware/
├── deploy/                                          D455 + PiPER-X -> the policy (run.py, calibration, perception, pc_*)
├── depth_bench/                                     how the camera was chosen and its noise fitted
└── objects/                                         the measured real objects
results/                                             the numbers (see results/README.md); results/pc/REPORT.md is current
docs/                                                pc_shadow_runbook.md, hardware_lessons.md, rig build sheet, history/
```

Retired lines live in git history and on branches: the cube-pushing task
(`piperx/cube-policy-v2`), the reward-free sim-to-real calibration research
(`yf/wm-ra-research`), and everything removed on 2026-09-06 (campaign scripts
v3–v10, YOLO labelling tools, HITL/sim-replay, the Odin 1 / ZED X bench
backends) in the commits before that date.
