# results/ — the numbers, and where everything else goes

## The convention (since 2026-09-06)

| where | what | tracked |
|---|---|---|
| `logs/rsl_rl/<experiment>/<stamp>_<run>/` | trainer output: `model_*.pt`, tensorboard, `params/`. Written only by `train`, `distill.py`, `finetune.py`. Never hand-edited, never the source of a quoted number. | no |
| `recordings/<name>/` | every deployment session: `run.py --record`, `pc_run.py --record`, shadow runs, the calibration captures. One directory per run, fresh, never reused (`run.py` refuses a non-empty one). | no |
| `checkpoints/<line>/` | checkpoints copied off the training host for evaluation, export or the viewer. | no |
| `hardware/deploy/policies/<bundle>/` | exported policies with their `obs_spec.json`, `manifest.json` and rig snapshot; rebuilt by `scripts/check_export.py` / `scripts/pc/bundle.py`. | no |
| `results/<campaign>/<run_tag>/` | **the numbers**: every evaluation JSON with its provenance (commit, dirty diff, checkpoint sha256, sensor, env knobs), the run's `manifest.json`, and a README or REPORT.md that says what was concluded. | **yes** |

Inside a `results/<campaign>/<run_tag>/`, a pipeline script (`scripts/pc/run_route.sh`,
`scripts/pc/continue_teacher.sh`, `scripts/run_v10c.sh`) writes:

- `manifest.json` — tag, route, teacher + sha256, commit, seeds, budgets, host, start time. Written before the first stage.
- `stage_<name>.done`, `all.done`, `FAILED` — markers so a relaunch resumes. Ignored.
- `<stage>.log`, `watcher.log` — transcripts. Ignored; the JSON beside each carries the number.
- `accept_*.json`, `endurance_*.json`, `actions_*.json`, `occlusion_*.json`, `smoke.json` — one per evaluation, named `<what>_<stage>_s<seed>`. Tracked.
- `export/` — the graphs; `scripts/pc/bundle.py` copies them into a bundle. Ignored.
- `*_checkpoint.txt`, `*_run.txt` — the paths on the training host the stage used. Tracked.

Environment epochs (read `provenance.env_knobs` before comparing across them):

- before 2026-09-05: Action API v1 (`-V1` task ids);
- 2026-09-05 → `ad49e04` (2026-09-06 evening): Action API v2, an object knocked out of the
  spawn sector stays there for the rest of the episode;
- from `ad49e04`: the `object_astray` termination ends such an episode after 1 s
  (`OBJECT_ASTRAY_TERMINATE=0` reproduces the epoch before).  The same checkpoint reads
  differently across the last two epochs (`results/pc/gen2/REPORT.md` section 5).

Rules that came from being bitten:

- A run directory is written once. A fix is a new commit and a new tag, never an edit in place.
- Quote a median over three evaluation seeds with the spread, and the late/early ratio beside any placed/min (`report-late-over-early`); the simulator is not reproducible run to run.
- Read `provenance.env_knobs`, `provenance.sensor` and `validity` in a JSON before comparing two of them. `VALIDITY.md` lists the files written under a defect and what replaced them.
- `scripts/pull_results.sh <campaign>` rsyncs a campaign back from the training host (excludes `*.pt`); the server side is the source of truth for anything trained there.

## What is here

Current line (branch `yf/pc`, point-cloud students from the v11 teacher):

| dir | what | read |
|---|---|---|
| `pc/` | teacher qualification (`teacher_eval/`, `teacher/v11_nosight_36s/`), the four routes (`routes/pc_final_*`, screens in `routes/pc_screen_*`), observation specs (`specs/`) | `pc/REPORT.md` |

History, read-only. Each has a README saying what it measured and, since the 2026-09-04 audit, which numbers stand:

| dir | campaign |
|---|---|
| `audit_20260904/` | the evaluation ruler re-measured on `d455_v4_final` after the no-op-load / GRU-reset / sensor fixes; `v4_endurance_fixed.json` |
| `accept_compare/` | v4 against v8 on one ruler |
| `teacher_pick/` | six v7-era state teachers, three seeds each |
| `v7_students/` | why every v7 student grasps once and stops (the gripper latch) |
| `v8_students/` | the v8 depth vs SAM2 observation A/B (`*_fixed_*` are the valid re-measurements) |
| `d455_heavy_dr/` | v8 horizon, v8 remote teachers (the `.pt` files are untracked since 2026-09-06), v9 full-reset |
| `decay/` | within-episode decay measurements |
| `sam_eval/` | SAM2.1 in the sim perception evaluation, against the renderer's segmentation buffer |
| `segbench/` | segmentation backends on the recording `v4_stereo_repro_scene2`, per-frame JSONL |
| `lifecycle/` | `target_gaps.json`, the measured target-loss gaps behind `hardware/deploy/lifecycle.py` |
| `sysid/` | the plant fit (`plant_fit*.json`) behind the MIT gains |
| `depth_sensor/` | D405 sensor-model measurements (the D405 era; see `docs/history/`) |
| `calibration/` | old-vs-calibrated rig figures |
| `VALIDITY.md` | the 2026-09-04 validity audit |

Everything under `pages/` and `ra_sim0/` is untracked local material (rendered viewer pages; a research-branch leftover).
