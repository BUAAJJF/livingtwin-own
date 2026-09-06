# Point-cloud routes: shadow test and first-motion runbook (branch `yf/pc`)

Everything below is for a policy that passed the deployment gate in
`results/pc/routes/<tag>/` (`scripts/pc/report_routes.py` prints `GO` only
when every criterion holds).  A NO-GO candidate may be shadow-run for timing
and never moves the arm.

## What a bundle is

`scripts/pc/bundle.py <route dir> --spec results/pc/specs/obs_spec_<route>.json --out hardware/deploy/policies/<name>`
writes one directory:

| file | what | checked by |
|---|---|---|
| `policy.onnx`, `policy.pt` | the exported recurrent policy, explicit GRU state, emits pre-squash `u` | `scripts/check_export.py` (torch vs onnx vs jit over 8 steps, < 1e-4) |
| `checkpoint.pt` | the training checkpoint the graphs came from | `infos.action_api` = v2 bounded, spec `51c919a9dd2e92db` |
| `obs_spec.json` | actor groups, term order and widths, `action_spec` + `action_api` | `ProprioBuilder` refuses an unknown term; `ActionMapper` refuses a spec whose hash does not match |
| `rig_d455.json` | the calibration the policy will be run with (snapshot) | hand-eye residual 3.9 mm, table plane |
| `manifest.json` | route, teacher and its sha256, training and bundle commits, every file's sha256, rollback | read by `pc_run.py` |

The action convention is the bounded one on every path: the graph emits `u`,
`ActionMapper` applies `tanh` exactly once (`action_spec.squashed`), the
simulator's action term did the same, and the `actions` observation is
`tanh(u)` in both.  There is no `atanh` anywhere and no v1 translation.

## Shadow mode (tonight's only allowed real-camera mode)

No drive is ever enabled by `pc_run.py`.  Three joint sources:

```bash
M="micromamba run -n mjlab python"
# 1. recorded D455 session + its recorded joints, on the recording's own 30 Hz clock
$M -m hardware.deploy.pc_run --policy hardware/deploy/policies/<bundle> \
    --replay recordings/v4_stereo_try3 --seconds 30 --record recordings/pc_shadow_replay_<try>
# 2. live D455, dry joints (default pose), for perception timing on the rig
$M -m hardware.deploy.pc_run --policy hardware/deploy/policies/<bundle> \
    --camera d455 --seconds 30 --record recordings/pc_shadow_live_<try>
# 3. live D455 and the arm's real joint feedback, drives OFF (CAN opened for reading only)
$M -m hardware.deploy.pc_run --policy hardware/deploy/policies/<bundle> \
    --camera d455 --arm-read --seconds 30 --record recordings/pc_shadow_armread_<try>
```

Each run writes `shadow.jsonl` (per control step: vision age in steps, fresh,
valid, hold state, `u`, `tanh(u)`, the joint target that WOULD have been sent,
loop and perception ms) and `summary.json`.  Read the summary before anything
else; the numbers that gate a real run:

| field | must be |
|---|---|
| `control_ms.p95` | < 20 ms and `overruns` a handful at most (the first steps warm the GPU) |
| `perception_ms.p95` | well under 33 ms, or the effective rate drops below 30 Hz |
| `effective_vision_hz` | ~30 (3 fresh frames per 5 control steps) |
| `frame_age_s.p95` | < 0.10 s; the policy trained on 0-80 ms |
| `holds` | only `hold_no_vision` in the first steps; `hold_stale` / `hold_no_depth` / `hold_no_candidate` must be rare and explained |
| `action_api_status` | `ok` |

Measured on 2026-09-06 with a P1A smoke export against `recordings/v4_stereo_try3`:
control 1.2 / 1.9 ms (p50 / p95), perception 1.0 / 1.4 ms, 29.7 Hz effective,
frame age 16 / 30 ms, CUDA provider.

## Logs and replay

```bash
python - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("recordings/pc_shadow_replay_try1/shadow.jsonl")]
print(collections.Counter(r["hold"] for r in rows))
print("age steps p95", sorted(r["age_steps"] for r in rows)[int(0.95 * len(rows))])
PY
```

The training-side evaluations of the same policy are in the route directory:
`accept_final_s{101,202,303}.json`, `endurance_final_s*.json`,
`accept_heldout_s101.json`, `actions_final_s101.json`,
`occlusion_final_s101.json`, `export.log`, and the smoke's timing in
`smoke.json`.  `scripts/pc/report_routes.py <dirs> --teacher-placed <n>`
tabulates them and applies the gate.

## Rollback

The only policy that has placed objects on the real arm is
`hardware/deploy/policies/d455_v4_final` (2026-09-01).  It is Action API v1:
it runs through `hardware/deploy/run.py` on the mask pipeline with
`--allow-legacy-action-api`, not through `pc_run.py`, and its command is in
the root README.  It is the rollback in the sense of "the arm can still be
driven by something that worked", not a checkpoint of this pipeline.

## First real motion (tomorrow, human-confirmed)

Do not start this without the physical emergency stop in hand, the workspace
clear, and ONE textured object of the trained size class on the mat.

1. `python -m hardware.deploy.scene --like recordings/v4_stereo_try3` -- the table matches a known-good scene.
2. `python -m hardware.deploy.jointcheck --joint 1` (then 2..6): five degrees each, the arm moves the joint the number says.
3. Shadow run 3 above for 30 s with the object in place: `holds` empty after the first steps, the logged targets stay inside `SAFE_TARGET_CLIP`, the gripper target opens on approach (read `target[6]` in `shadow.jsonl`).
4. Real motion goes through `hardware/deploy/run.py --obs pc`: the same guards as every mask-policy run (fresh `--record`, the typed `move`, `--home-first`, `--command-rate-scale`, `--max-joint-speed-fraction`, feedback faults, stale-observation and empty-workspace holds, the final hold), with `PcPerception` in place of the segmenter and the bundle's own obs spec.  Verified on 2026-09-06 with the P1B bundle: a `--dry-run` (no camera) and a `--replay recordings/v4_stereo_try3` (dry arm) both run the loop at 50 Hz with perception at 30 Hz, 5.0 / 7.5 ms compute, no holds beyond the first frame.

   ```bash
   # 4a. the loop on the rig with the real camera and the DRY arm first (no CAN, nothing moves)
   micromamba run -n mjlab python -m hardware.deploy.run --obs pc \
       --policy hardware/deploy/policies/pc_P1B_20260906T0319 --camera d455 --no-arm \
       --seconds 20 --record recordings/pc_noarm_<try>
   # 4b. first motion: e-stop in hand, one object, low rate, guarded.  The run asks for the word `move`.
   micromamba run -n mjlab python -m hardware.deploy.run --obs pc \
       --policy hardware/deploy/policies/pc_P1B_20260906T0319 --camera d455 --policy-device cuda \
       --home-first --command-rate-scale 0.5 --max-joint-speed-fraction 0.6 \
       --seconds 20 --record recordings/pc_motion_<try>
   ```

   What `--obs pc` refuses on purpose: `--view`, `--mask`, `--target-lifecycle`, `--target-tracker`, `--held-target-radius`, `--flatten-scene` -- they all belong to the mask observation.  `--depth-source stereo` still works (the perception thread owns the TensorRT engine).  The recorded session has the usual `control.json` / `meta.json` / frames plus `mask_state = pc:<route>` and the workspace point count per frame in `detections`.
5. Stop conditions, whichever comes first: any hold state for more than 1 s, a joint-speed trip, the object leaving the sector, 20 s on the first run, 60 s afterwards.
6. Two objects only after three clean single-object cycles.

## GO / NO-GO rule

GO for a shadow test on the rig: the route's `report_routes.py` line reads
`GO`, its bundle's `check_export` says OK, and a `--replay` shadow run meets
the timing table above.  GO for real motion additionally needs a driven runner
(step 4) and a human at the e-stop.  Anything else is NO-GO, and the reason is
the first line of the report.
