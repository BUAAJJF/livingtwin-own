# LivingTwin repository and current-task handoff

This file is a handoff for cross-checking the repository in its **current working-tree state**. It is not a claim that every historical document is current. In particular, the root `README.md` and the opening paragraphs of `hardware/deploy/README.md` still say that the deployment stack has not run on hardware; that statement is obsolete. D405 and D455 calibration, perception, guarded policy motion, HITL, and real-arm trajectory replay have now been exercised.

## Before changing anything

- Repository: `/home/yunfan/Project/PiperPush/LivingTwin`
- Current branch: `yf/bolt`
- Last checked commit: `f1dab6e` (`RA-HW-0: the design, the limits and the dry-run stack -- and no way to move an arm`)
- The working tree is intentionally very dirty. The D455/calibration/deployment/DR work described below is mostly uncommitted. Do not reset, clean, replace `.orig` files, or discard unrelated changes.
- Inspect with `git status --short` and review the actual diff before editing. Treat this file as orientation, then verify every important claim in source or an artifact.
- Python environment is `mjlab` under micromamba. Non-interactive MuJoCo runs normally need:

  ```bash
  export MUJOCO_GL=disable
  export LD_LIBRARY_PATH=/home/yunfan/micromamba/envs/mjlab/lib:${LD_LIBRARY_PATH:-}
  ```

- Run the local test suite with:

  ```bash
  micromamba run -n mjlab python -m pytest tests -q
  ```

  The last full run after removing simulated table-contact reward/termination was **437 passed, 36 warnings**.

## What the project does

The main task is continuous tabletop pick-and-place/tidying with an AgileX PiPER-X. A policy observes a fixed third-person depth camera and robot proprioception, then directly commands six joint-position targets plus the gripper at 50 Hz. There is no deployment-time object pose, IK planner, scripted grasp primitive, or privileged grasp flag.

The principal training sequence is:

```text
privileged state teacher (PPO)
    -> vision student (student-rollout DAgger/distillation)
    -> vision actor + state critic (PPO fine-tuning)
    -> deterministic evaluation / export
    -> calibrated D455 + segmentation + PiPER-X deployment
```

The state teacher sees object state and privileged physics. The deployed actor sees proprioception plus a 3-channel 224x168 policy image: scene depth, target mask, and masked target depth. Its visual encoder is a two-layer CNN with spatial softmax followed by a 256-wide GRU. The asymmetric critic remains privileged during PPO and is not deployed.

The reward-free, decision-aware sim-to-real calibration research track (WM0/WM1/RA-Sim/RA-HW) was moved to the branch `yf/wm-ra-research` on 2026-09-04 and is not on this branch. What remains here (`perturb.py`, `latency.py`, `damping.py`, `prior.py`, `hidden_plant.py`, `residual.py`) is shared simulator-mismatch infrastructure the hardware line imports. Do not conflate simulation-only WM/RA claims with the current hardware campaign; the README carries the true per-phase verdicts (all RED or BLOCKED after WM0).

## Code map

### Simulation and learning

- `src/piper_push/tasks/pick_place/env_cfg.py`: scene, observations, commands, rewards, events, curricula, and terminations for pick/place.
- `src/piper_push/tasks/pick_place/mdp.py`: custom observations, commands, metrics, rewards, diagnostic contact functions, and reset logic.
- `src/piper_push/tasks/pick_place/rl_cfg.py`: state MLP, vision CNN/spatial-softmax/GRU, PPO, and distillation configurations.
- `src/piper_push/tasks/pick_place/robust_cfg.py`: versioned broad D455 domain randomization (`HEAVY_DR_PROFILE`, currently named `d455_heavy_dr_v2`).
- `src/piper_push/tasks/pick_place/__init__.py`: all mjlab task registrations.
- `src/piper_push/robot.py`, `objects.py`, `shapes.py`, `camera.py`: robot/scene geometry, object distribution, camera model, and cadence.
- `src/piper_push/tasks/pick_place/cold_curriculum.py` + `cold_cfg.py`: the cold-start teacher tasks (`-Robust-Cold`, `-Robust-Cold-NoSight`): nominal DR and near-zero penalties at random init, stages opened by rolling capability gates (grasp attempts, placements, safe episodes) with persistence and no regression, ending at the full heavy profile.  `scripts/run_v10c.sh` trains one from scratch, evaluates it on `-Robust` with three seeds and freezes a baseline only against the criteria declared in its `config_snapshot.json` (`scripts/v10c_verdict.py`).
- `src/piper_push/actions.py`: policy action scaling, clipping, slew limiting, latency/hold/response/deadband plant hooks.  Since 2026-09-05 the convention is bounded: the policy emits u (`piper_push.squashed.PreSquashGaussianDistribution`), the action term applies a = tanh(u), and a = ±1 is the safe clip (`robot.BOUNDED_ARM_SCALE/OFFSET`); PPO's density is on the stored u, never atanh of a float32 a; checkpoints and exported specs carry an `action_api` stamp (`piper_push.action_api`) that every loader checks, and unstamped (pre-2026-09-05) checkpoints load only on `-V1` ids with `--allow-legacy-action-api`; pre-2026-09-05 checkpoints evaluate only on the `-V1` task ids.
- `src/piper_push/depth_noise.py`, `d455_noise.py`: active-stereo depth corruption and fitted D455 parameters.
- `src/piper_push/layout.py`: calibrated layout convention, including the requested +90-degree workspace/goal rotation.
- `src/piper_push/models.py`, `distill.py`, `checkpoints.py`: recurrent visual model, distillation runner, and checkpoint conversion.
- `scripts/train.py`: state PPO entry point (invoked by `scripts/train.sh`).
- `scripts/distill.py`: vision distillation. Its `--iterations` value is an absolute target when resuming.
- `scripts/finetune.py`: vision PPO fine-tuning with distilled actor, state critic, low exploration, and critic-only warm-up. Its `--iterations` is also an absolute target.
- `scripts/eval.sh` / `scripts/accept_s1.py`: deterministic acceptance evaluation and provenance JSON.
- `scripts/audit_table_contact.py`: contact-only diagnostic audit; contact is no longer a training objective or termination.

Registered task IDs relevant to this campaign:

| Task | Purpose | Log experiment |
|---|---|---|
| `Mjlab-Pick-Place-PiperX` | nominal state teacher | `piperx_pick_place` |
| `Mjlab-Pick-Place-PiperX-Robust` | broad-DR state teacher | `piperx_pick_place_robust` |
| `Mjlab-Pick-Place-PiperX-Vision` | nominal vision PPO/eval | `piperx_pick_place_vision` |
| `Mjlab-Pick-Place-PiperX-Vision-Robust` | broad-DR vision PPO/eval | `piperx_pick_place_vision_robust` |
| `Mjlab-Pick-Place-PiperX-Distill` | nominal vision distillation | `piperx_pick_place_distill` |
| `Mjlab-Pick-Place-PiperX-Distill-Robust` | broad-DR vision distillation | `piperx_pick_place_distill_robust` |

### Camera characterization and calibration

- `hardware/depth_bench/capture/`: D405, D455, ZED X, and other capture backends.
- `hardware/depth_bench/model/fit_d455_noise.py`: fits the D455 active-stereo corruption model from local recordings.
- `hardware/depth_bench/model/d455_noise.json`: fitted D455 model and source provenance.
- `hardware/depth_bench/targets/make_calib_compact_v2.py`: compact white-dominant ChArUco target generator, including crop/reference frame.
- `hardware/deploy/calibgui.py` + `calibgui.html`: browser GUI for guided hand-eye and table-board collection, pose coverage, next-view proposal, solve/reset/save.
- `hardware/deploy/calibrate.py`: raw high-resolution grayscale detection, multi-frame corner fusion, robust outlier rejection, AX=XB solve, and joint reprojection refinement; also fits the tabletop from a moved checkerboard.
- `hardware/deploy/config.py`: shared camera/workspace/rig conventions.
- `hardware/deploy/rig_d455.json`: active D455 eye-to-hand calibration and table plane.

The installed compact white calibration target is 180x152 mm finished size, with a 168x140 mm 6x5 ChArUco pattern using 28 mm squares; printing/cutting details are in `hardware/deploy/README.md`. The table helper checkerboard uses 25 mm squares. A 3.5 mm physical board thickness/offset was accounted for during the table workflow; verify the saved pose metadata before recomputing.

Current D455 calibration artifact (`hardware/deploy/rig_d455.json`):

- serial `262822300638`
- 44 hand-eye poses / 1056 fused frames in `calib_poses_d455.json`
- hand-eye residual 3.916 mm
- table board: 6 poses / 104 fused frames in `calib_table_poses_d455.json`
- table `z = -3.784 mm` in base frame
- table normal `[-0.000760, 0.015389, 0.999881]`
- table tilt 0.883 degrees; fitted flatness 0.411 mm

The fitted D455 noise model combines 800 frames from `recordings/d455_yolo/session01` and 100 manually accepted frames from `recordings/d455_yolo/manual_20260826_163148`. It includes static and temporal range error, spatial correlation, edge/dropout behavior, depth bias, and stereo disparity quantization. See the JSON rather than copying constants into new code.

### Perception and deployment

- `hardware/deploy/sensor.py`: D405/D455 reader abstraction and frame timestamps.
- `hardware/deploy/rectify.py`: calibrated camera geometry to the policy view.
- `hardware/deploy/mask.py`: table/depth segmentation, target tracking, optional YOLO and fused segmentation.
- `hardware/deploy/yolo_backend.py`: YOLO segmentation backend.
- `hardware/deploy/collect_yolo_gui.py` + HTML: click-to-save data collection after manually moving four objects.
- `hardware/deploy/autolabel.py`, `autolabel_manual.py`, `train_yolo.py`, `yolo_eval.py`: dataset creation/training/evaluation.
- `hardware/deploy/policygui.py` + HTML: browser view of exactly what the policy sees; it is visualization, not a separate controller.
- `hardware/deploy/policy.py`, `obs.py`, `proprio.py`: recurrent policy execution and simulator-compatible observation packing.
- `hardware/deploy/robot.py`: PiPER SDK connection, feedback, gripper range, command mapping, and hold behavior.
- `hardware/deploy/run.py`: real-time 50 Hz control loop, mandatory recording for real motion, stale/no-target hold, speed/action/timing guards, optional height/table guards, and final hold.
- `hardware/deploy/hitl.py`: simulated camera/object input with real-arm motion for dynamics-gap experiments.
- `hardware/deploy/sim_replay.py`: generate a simulated trajectory and replay it 1:1 on the real arm.

D455 YOLO artifacts are under `hardware/deploy/yolo_d455/`. The current `yolo26n-seg` model used 90 accepted frames (76 train, 14 validation, four instances each). Reported validation mask metrics are precision 0.902, recall 0.826, mAP50 0.931, mAP50-95 0.696. CUDA PyTorch inference was 4.5 ms p50; CPU ONNX was 44.2 ms p50 and is too slow for the 30 Hz camera. Recommended deployment mode is **fused depth + YOLO**, CUDA, confidence 0.25. Pure YOLO is explicitly not yet considered ready.

The only locally exported policy currently under `hardware/deploy/policies/` is the older D405-trained, rotated-layout model `calib_rot90_d405_finetune_v1_model1400`. It has been used for guarded D455 bring-up, but it is not the final D455-DR model now being trained.

## Hardware facts and safety invariants

Real-arm tests exposed two serious issues: a table touch/downward probe and visible jitter. A prior failure mode also suddenly disabled/powered off the arm. Current code was changed to hold measured pose on stop/exception and not deliberately power off the drives, but hardware motion remains hazardous.

Do not run real motion merely to test a code hypothesis. For any real motion:

- Keep the physical emergency stop in hand and clear the whole workspace.
- Require an interactive terminal and type the explicit `move` confirmation.
- Pass a **fresh** `--record` directory. `run.py` refuses real motion without it; every attempt must preserve depth, gray, feedback, actions, timing, rig, events, final feedback, and stop reason.
- Start with reduced `--command-rate-scale`, acceleration limits, and joint-speed fraction. Do not silently relax them.
- A stale observation, no target, malformed/non-finite action, feedback fault, repeated timing overrun, or configured geometry violation must hold/resync or stop.
- Do not restore the previous behavior of powering down on ordinary exit; holding is safer than a sudden loss of torque for this setup.
- Do not infer force/contact from kinematics. There is no external table force sensor.

Table behavior needs careful interpretation:

- The current training task has **no table-safety reward, no table-contact termination, and no table-safety curriculum**. Contact sensors remain for offline audit only.
- This change was deliberate. A 5 mm proximity shell was terminating safe near-table motion and drove the robust teacher toward inactivity. The policy should learn successful picks without being trained against a discontinuous, poorly matched contact event.
- Deployment `--min-table-clearance` defaults to `None`; omitting it allows light fingertip/table contact. General command, speed, stale-input, timing, and communication protections still remain.
- `--min-grasp-height` is a separate optional hard floor; initial guarded tests used values around 0.05-0.065 m.
- If `--min-table-clearance` is requested, `run.py` checks both measured and predicted moving robot geometry against the calibrated plane and requires `table_normal_base` in the rig file.
- HITL and trajectory replay tools may retain a conservative 40 mm default because they are dynamics/replay tools, not the normal vision-policy deployment path.
- A contact audit of the strong nominal state teacher over 256 envs x 600 steps observed pad contact in all environments and grasps in 254/256, but **zero actual robot-table contacts**. Thus the successful nominal teacher does not depend on table collision.

One recorded guarded real-policy attempt is `recordings/d455_policy_motion_grasp50_try6/run.json`: it stopped as designed when predicted grasp-site height reached 49.4 mm below a configured 50 mm floor. The 60 s simulation-plan-to-real replay provenance is in `recordings/sim_replay_d455_60s_try1/run.json`.

## Current D455 robust-training campaign

Goal: retrain the complete teacher -> student -> PPO stack for the measured D455/calibrated +90-degree layout using deliberately broad DR, prioritizing first-deployment success rate over peak speed. D455 recordings from YOLO collection and calibration were used to fit the simulated sensor noise.

`src/piper_push/tasks/pick_place/robust_cfg.py` currently randomizes:

- arm and gripper action delay, command holding, response scale, and deadband
- arm/gripper PD gains, joint friction, link pseudo-inertia/mass, and COM shift
- table z and tilt, object mass/friction, and finger-pad friction
- calibrated camera pose, fitted D455 depth effects, mask jitter, and observation latency

This is intentionally a broad first-pass distribution. It is not evidence that sim-to-real is guaranteed, and real gripper dynamics/system identification is still weak. Final reporting must quantify the nominal-performance degradation caused by DR.

### Previous checkpoints and diagnostic results

Known remote checkpoints under `/home/yunfan/work/piper-push/LivingTwin`:

```text
nominal state teacher:
logs/rsl_rl/piperx_pick_place/2026-08-27_15-50-47_d455_heavy_dr_v2_nominal_teacher/model_3499.pt

old nominal vision final:
logs/rsl_rl/piperx_pick_place_vision/2026-08-27_22-05-10_d455_heavy_dr_v2_nominal_finetune/model_2999.pt

old robust state teacher (collapsed/inactive):
logs/rsl_rl/piperx_pick_place_robust/2026-08-27_16-32-13_d455_heavy_dr_v2_robust_from_nominal600/model_4099.pt

old robust vision final (collapsed/inactive):
logs/rsl_rl/piperx_pick_place_vision_robust/2026-08-27_22-05-10_d455_heavy_dr_v2_robust_finetune/model_2999.pt
```

Quick evaluation after removing table-contact termination, 256 envs x 1200 steps:

| Checkpoint/eval domain | Result |
|---|---|
| nominal teacher / nominal | PASS; 35.996 objects/min, success 0.9924, drops 0.0115, p95 2.08 s |
| old robust teacher / robust | FAIL; zero throughput |
| old nominal vision / nominal | improved but FAIL; 9.297 objects/min, success 0.7535, p95 9.52 s |
| old robust vision / robust | FAIL; zero throughput |

Interpretation: the obsolete 5 mm termination shell was a major cause, but removing it only at evaluation cannot revive a checkpoint that already learned inactivity. A fresh robust adaptation is required.

### Active run at last update

Timestamp checked: **2026-08-28 11:14 Asia/Shanghai** (`03:14 UTC`).

- Host: `ssh shen-teacher`
- Remote repo: `/home/yunfan/work/piper-push/LivingTwin`
- Command wrapper: `scripts/train_d455_no_contact_v3.sh`
- Campaign tag: `d455_heavy_dr_v3_no_contact`
- GPU: 0, 8192 environments
- Bootstrap: copied nominal `model_3499.pt` into `piperx_pick_place_robust/d455_heavy_dr_v3_no_contact_bootstrap/`
- Run: 2000 **additional** robust teacher iterations, displayed as iteration 3499 -> 5499
- PID file: `results/d455_heavy_dr/d455_heavy_dr_v3_no_contact/robust_teacher.pid`
- Log: `results/d455_heavy_dr/d455_heavy_dr_v3_no_contact/robust_teacher.log`
- At the check: iteration 3562/5499, iteration time 3.33 s, log ETA about 4 h 08 min (ETA has varied substantially)
- Recent placed-object metric had recovered from 0 at startup to roughly 0.9-1.3 per rollout; over-speed terminations fell from about 19-31 to roughly 4-7. This is recovery, not yet a passed final result.

There was one incorrect launch with `END_ITER=5500`; because rsl_rl resume semantics add iterations, it would have run 3499 -> 8999. It was interrupted after only a few iterations. Its log is retained at:

```text
results/d455_heavy_dr/d455_heavy_dr_v3_no_contact/robust_teacher_bad_iteration_count.log
```

Do not mistake it for the active run.

### Important incomplete work

Only the v3 robust **teacher** is currently launched automatically. There is no v3 continuation watcher yet. After the teacher finishes, the following still must be deliberately launched and monitored:

1. Find the newest completed robust-teacher run matching `d455_heavy_dr_v3_no_contact_robust_teacher` and select its highest `model_*.pt` numerically.
2. Distill with `Mjlab-Pick-Place-PiperX-Distill-Robust`, normally 512 envs and 3000 target iterations.
3. PPO fine-tune with `Mjlab-Pick-Place-PiperX-Vision-Robust`, using the distilled student as actor and the new robust teacher as critic, normally 512 envs and 3000 target iterations.
4. Evaluate the new final policy on both nominal and robust/stress tasks using the standard 512 x 2400 protocol, preferably multiple independent seeds/processes because this simulator is not exactly run-to-run reproducible.
5. Run `scripts/report_dr_degrade.py` against the nominal control and `scripts/audit_table_contact.py` for diagnostic contact counts.
6. Export/check the final recurrent policy and run dry/replay/no-arm validation before any guarded hardware motion.

Previous observed durations provide only a planning estimate: robust teacher roughly 2-4 h depending on iteration speed, distillation about 2 h 10 min, PPO fine-tuning about 3 h 30 min, and evaluation/audit 20-40 min. Check GPU availability before assigning continuation stages; at the last inspection GPU 1 was occupied by an unrelated process and must not be disturbed.

`scripts/continue_d455_heavy_dr_v2.sh` is a useful template but is not safe to run unchanged: it assumes the old v2 paired nominal/robust PIDs, old run names, and fixed GPUs. A v3 continuation should be robust-only, wait on the current PID, validate process exit/checkpoint freshness, choose currently free GPUs, and fail loudly if a stage or expected artifact is missing.

Useful read-only status command:

```bash
ssh shen-teacher '
  cd /home/yunfan/work/piper-push/LivingTwin
  out=results/d455_heavy_dr/d455_heavy_dr_v3_no_contact
  pid=$(cat "$out/robust_teacher.pid")
  ps -p "$pid" -o pid=,etime=,stat=,cmd=
  ps --ppid "$pid" -o pid=,etime=,%cpu=,%mem=,stat=,cmd=
  tail -n 100 "$out/robust_teacher.log"
'
```

The remote host does not currently have `rg` in its non-interactive PATH; use `grep`/`find` there unless the environment is activated.

## What to cross-check next

Please prioritize review in this order:

1. **No hidden table termination:** confirm `env_cfg.py`, `mdp.py`, robust task construction, tests, and CLI overrides contain no remaining table-contact reward/termination/curriculum that can teach inactivity. Keep contact metrics diagnostic.
2. **Teacher adaptation semantics:** confirm `scripts/train_d455_no_contact_v3.sh` truly loads nominal `model_3499.pt` and performs exactly 2000 additional iterations, ending at 5499, without silently choosing a stale run.
3. **DR correctness and reachability:** verify each `HEAVY_DR_PROFILE` axis actually reaches MuJoCo/action/camera observations, is sampled at the intended cadence, and does not create impossible dynamics or invalid reset states. Pay special attention to action hold/latency composition and gripper randomization.
4. **D455 geometry consistency:** trace `rig_d455.json` -> rectification/masking -> simulation camera/layout. Check OpenCV vs MuJoCo frame conversion and the +90-degree layout exactly once, not twice.
5. **Sensor model validity:** confirm fitted D455 noise parameters are used by the robust vision task and that high-resolution gray/RGB used for segmentation/calibration is not accidentally fed to a depth-only policy as an undocumented fourth modality.
6. **Student/critic checkpoint compatibility:** verify strict state-teacher load in distillation, student-to-actor conversion, state critic observation group identity, GRU state reset/export, and absolute-vs-additional iteration semantics.
7. **Evaluation integrity:** reject results based on one short rollout, wrong task/checkpoint pair, stale JSON, or the old collapsed robust checkpoints. Preserve commit/diff/checkpoint hashes and quote DR degradation explicitly.
8. **Deployment control timing:** inspect the 50 Hz command schedule, mapping `dt`, stale frame behavior, resync after hold, inference/perception threading, logging cost, and the no-catch-up rule. The prior 18 Hz impression was partly visualization/CPU segmentation; the control loop and camera clocks must not be conflated.
9. **Physical safety:** confirm all exception/stop paths hold measured pose, real motion always requires a new log directory, no normal path unexpectedly disables torque, and optional table/height guards cannot create a discontinuous unsafe command. Do not test fixes on the arm until dry/replay checks pass.
10. **Repository hygiene:** identify what should be committed versus retained as local recordings/calibration/model artifacts. Do not commit large or machine-specific data blindly, but do not lose the only copies of D455 calibration/provenance.

## Current definition of done

The immediate task is not done when the teacher merely finishes. It is done only when the new no-contact robust teacher, vision distillation, and vision PPO stages complete; the final policy is evaluated on nominal and stress domains with contact diagnostics and DR-degradation reporting; export/replay/no-arm validation succeeds; and a conservative, fully logged real-arm trial can be proposed with evidence-based safety settings. Hardware success must be reported separately from simulation success.
