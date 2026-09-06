# LivingTwin — orientation for an assistant session

Slimmed on 2026-09-06. Verify anything important in source or in a result JSON;
history is in git and in `results/README.md`, not here.

## What this is

Continuous tabletop pick-and-place with an AgileX PiPER-X: a policy sees a fixed
third-person RealSense D455 plus proprioception and writes six joint targets and
a gripper command at 50 Hz. mjlab 1.6.0 (MuJoCo-Warp) + rsl_rl. Pipeline:
privileged state teacher (PPO) → vision student (student-rollout DAgger) →
vision actor + state critic (PPO fine-tune) → evaluation → ONNX/TorchScript
export → D455 + PiPER-X deployment. Root `README.md` has the status, the task
table and the layout; read it first.

Current branch `yf/pc`: mask-free point-cloud/depth students from the qualified
teacher `v11_nosight_36s` (`results/pc/REPORT.md`, `docs/pc_shadow_runbook.md`).
First generation is NO-GO for real motion, GO for shadow. The mask-based
`d455_v4_final` (Action API v1) is the rollback and the only policy that has
placed objects on the arm.

## Environment and tests

```bash
export MUJOCO_GL=disable
export LD_LIBRARY_PATH=/home/yunfan/micromamba/envs/mjlab/lib:${LD_LIBRARY_PATH:-}
micromamba run -n mjlab python -m pytest tests -q      # 408 passed on 2026-09-06, ~35 s
```

Local machine: one RTX 5090 (smoke, export, viewers, deployment). Training
host `shen-teacher`, repo `/home/yunfan/work/piper-push/LivingTwin`, GPUs 4–7
free by convention, 0–3 ask first. Sync code with `rsync` of named paths;
never `git reset`/`checkout` the remote tree. Launch with `nohup setsid`, a
stage-marker script and a watcher log; `scripts/pull_results.sh <campaign>`
brings the numbers back.

## Conventions that override defaults

- **Action API v2 (bounded).** The policy emits `u`; the action term applies
  `a = tanh(u)`; `a = ±1` is the safe clip. PPO's density is on the stored `u`,
  never `atanh` of a float32 `a`. Distillation is `MSE(tanh(u_s), tanh(u_t))`,
  exactly one tanh anywhere. Checkpoints and specs carry an `action_api` stamp
  (`piper_push.action_api`, spec hash `51c919a9dd2e92db`) that every loader
  checks. No v1→v2 translation, no bypassing the hash; pre-2026-09-05
  checkpoints load only on `-V1` task ids with `--allow-legacy-action-api`.
  Never gate on |u| magnitude (a |u|>10 guard killed five runs).
- **The evaluation ruler.** `piper_push.evalcfg.load_weights` (raises if the
  weights did not arrive; `runner.load(load_cfg={"actor": True})` on a Distill
  runner loads nothing), `--sensor measured` (a student with the sensor off is
  out of distribution), GRU reset on dones inside `inference_mode`
  (`eval_occlusion.reset_recurrent`), 256 envs, 3 seeds (101/202/303), 2400
  steps for accept and 1200 no-reset steps for endurance. Quote the median
  with the spread and the late/early ratio beside every placed/min. Read
  `provenance.env_knobs`, `provenance.sensor` and `validity` before comparing
  two JSONs. Perception numbers from one environment are anecdote (6–92%
  across seeds); sweep ≥6.
- **Results and logs.** `results/<campaign>/<run_tag>/` holds the numbers
  (JSON + manifest + README, tracked); transcripts, markers, exports and
  weights beside them are ignored; `logs/` is trainer output; `recordings/`
  is every deployment session; `checkpoints/` and `hardware/deploy/policies/`
  are untracked artefacts. A run directory is written once: a fix is a new
  commit and a new tag, never an edit in place. Formal runs start with a
  smoke run and a fresh UTC-stamped directory.
- **Teachers and students.** Distil only from a teacher that passed the gate
  (all three seeds place, late/early ≥ 0.85, no jaw latch/drift/NaN, no
  single-seed fluke). Observation dropout goes in the PPO stage, never in
  DAgger. `v5`/`v9` are history, not label sources.
- **Perception fixes ship with a frame-by-frame page** (`scripts/pc/viewer.py`,
  `scripts/sight_viewer.py`, `hardware/deploy/graspview.py`), frame id burned
  into the image, fixed crop. Aggregates have hidden the failure structure
  repeatedly.
- **Real motion** only with the physical e-stop in hand, a clear workspace, a
  fresh `--record` directory, the typed `move`, `--home-first`, reduced
  `--command-rate-scale`, and a human at the arm. Never to test a code
  hypothesis; never lower a gate to make a run happen tonight. Exception/stop
  paths hold the measured pose; nothing powers the drives off on exit. Do not
  open the live camera or CAN on this machine unasked.

## Code map

- `src/piper_push/tasks/pick_place/`: `env_cfg.py` (scene, observations, rewards, events, terminations), `mdp.py` (terms, `CameraScene`), `rl_cfg.py`, `robust_cfg.py` (`HEAVY_DR_PROFILE`, measured D455 sensor, latency mixture, `make_robust_env_cfg`), `cold_curriculum.py` + `cold_cfg.py` (capability-gated cold-start teacher, `PIPER_COLD_START_STAGE`), `pc_cfg.py` (point-cloud tasks), `__init__.py` (registrations).
- `src/piper_push/pc/`: `cloud.py` (measured sensor → base-frame workspace cloud or metric depth, 3-of-5 cadence, 0–4 step latency ring, `vision_meta`), `encoders.py` (PointNet, point-patch transformer, set MLP, depth ResNet-lite), `models.py` (`SetRecurrentModel`, export wrappers), `grasp.py` (analytic top-K grasp candidates + lock for P2).
- `src/piper_push/`: `robot.py`, `objects.py`, `shapes.py`, `camera.py`, `depth_noise.py`, `d455_noise.py`, `actions.py`, `squashed.py`, `action_api.py`, `latency.py`, `models.py`, `distill.py`, `checkpoints.py`, `evalcfg.py`, `layout.py` (+90° calibrated layout), `target_process.py`, `runners.py`.
- `scripts/`: `accept_s1.py`, `eval.sh`, `eval_endurance.py`, `eval_occlusion.py`, `distill.py`, `finetune.py` (`--iterations` is an absolute target when resuming), `train.sh`, `run_v10c.sh` + `v10c_verdict.py` (cold-start teacher from scratch with a declared verdict), `check_export.py`, `export_obs_spec.py`, `student_to_actor.py`, `pc/` (`run_route.sh`, `continue_teacher.sh`, `eval_teacher.sh`, `eval_actions.py`, `smoke.py`, `report_routes.py`, `report_teacher.py`, `bundle.py`, `viewer.py`), mask-line diagnostics (`sight_viewer.py`, `sim_perception_check.py`, `record_vision.py`, `measure_target_gaps.py`, `check_cadence.py`), `rig_to_sim.py`, `fit_object_distribution.py`, `logview.sh`, `pull_results.sh`, `bringup_d455.sh`, `accept_student.sh`.
- `hardware/deploy/`: `run.py` (the guarded 50 Hz loop; `--obs pc` for point-cloud bundles), `config.py` + `rig_d455.json` (calibration, hand-eye residual 3.9 mm, table plane), `rectify.py`, `mask.py`, `lifecycle.py`, `target_mask.py`, `sam_tracker.py`/`sam2_predictor.py`/`rgbmap.py`, `yolo_backend.py`, `stereo.py`, `obs.py`, `proprio.py`, `policy.py`, `robot.py` (MIT response flag 0xAD; `piper_sdk.JointMitCtrl` does not move this arm), `sensor.py`, `calibrate.py` + `calibgui.py`, `jointcheck.py`, `scene.py`, `review.py`, `logview.py`, `graspview.py`, `gripcal.py`, `mit.py`, `sysid.py`, `selftest.py`, `pc_obs.py`, `pc_perception.py`, `pc_run.py`. README there is the bring-up order.
- `hardware/depth_bench/`: the D405/D455 bench and `model/fit_d455_noise.py`.
- `tests/`: 408 tests; `test_deploy.py` is the deployment stack against the simulator.

Task ids: `Mjlab-Pick-Place-PiperX{,-Robust,-Vision,-Vision-Robust,-Distill,-Distill-Robust}` (+ `-V1` for legacy checkpoints, `-Wrist`, `-Half`/`-ThreeQ` dropout ramps, `-Robust-Cold[-NoSight]`, `-Robust-Cold2[-NoSight]`), and `Mjlab-Pick-Place-PiperX-PC-{P0,P1A,P1B,P2}-{Distill,Vision,Vision-Heldout}`.

## Where history lives

`results/README.md` indexes every past campaign and which of its numbers
stand (`results/VALIDITY.md`). `docs/history/` has the D405-era sensor write-up
and the early nominal results. Retired code: cube pushing on
`piperx/cube-policy-v2`, the WM/RA calibration research on
`yf/wm-ra-research`, and the 2026-09-06 removals (campaign scripts v3–v10,
YOLO tooling, HITL/sim-replay, Odin/ZED backends) in the commits before that
date. The assistant's own lessons are in its memory directory, not in this file.
