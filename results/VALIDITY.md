# Result validity, audited 2026-09-04

Files below are kept as they were written; each carries a top-level `validity`
key saying which evaluation defect it was measured under and what replaces it.
`invalid` = the number does not measure the policy (random network, or a
recurrent state the policy never has); `superseded` = it measures the policy
but in a domain the protocol did not intend, and must not be quoted until
re-measured.  Any `results/**` JSON without a `validity` key and without a
`provenance.sensor` block predates the audit and was checked by
a scan of every evaluation JSON under results/ on 2026-09-04: 701 files, 19 affected.

| file | status | defect(s) | superseded by |
|---|---|---|---|
| `results/accept_compare/v4_robust.json` | **superseded** | sensor_downgrade | re-measure pending |
| `results/accept_compare/v8s_depth.json` | **superseded** | sensor_downgrade | re-measure pending |
| `results/accept_compare/v8s_sam2.json` | **superseded** | sensor_downgrade | re-measure pending |
| `results/decay/v4_final.json` | **invalid** | noop_load, no_gru_reset | `results/audit_20260904/v4_distill_robust_fixedload_sensor-task.json` |
| `results/decay/v4_final_nominal.json` | **invalid** | noop_load, no_gru_reset, clean_sensor | `results/audit_20260904/v4_vision_measured.json` |
| `results/decay/v4_final_reset.json` | **invalid** | no_gru_reset, clean_sensor | `results/audit_20260904/v4_vision_measured.json` |
| `results/decay/v4_final_vision.json` | **invalid** | no_gru_reset, clean_sensor | `results/audit_20260904/v4_vision_measured.json` |
| `results/audit_20260904/v4_endurance_fixed.json` | **superseded** | clean_sensor | `results/audit_20260904/v4_vision_measured.json` |
| `results/v7_students/heldproxy_5000.json` | **invalid** | no_gru_reset | re-measure pending |
| `results/v7_students/iid_control_8000.json` | **invalid** | no_gru_reset | re-measure pending |
| `results/v7_students/statemachine_5000.json` | **invalid** | no_gru_reset | re-measure pending |
| `results/v7_students/student_5000.json` | **invalid** | no_gru_reset | re-measure pending |
| `results/v7_students/v7b_strong_student_700.json` | **invalid** | no_gru_reset | re-measure pending |
| `results/v8_students/v8_occ_v8s_depth.json` | **invalid** | no_gru_reset | `results/v8_students/v8_occ_fixed_v8s_depth.json` |
| `results/v8_students/v8_occ_v8s_sam2.json` | **invalid** | no_gru_reset | `results/v8_students/v8_occ_fixed_v8s_sam2.json` |
| `results/v8_students/v8_students_v8s_depth-0.json` | **invalid** | no_gru_reset | `results/v8_students/v8_fixed_v8s_depth.json` |
| `results/v8_students/v8_students_v8s_depth-300.json` | **invalid** | no_gru_reset | `results/v8_students/v8_fixed_v8s_depth.json` |
| `results/v8_students/v8_students_v8s_sam2-0.json` | **invalid** | no_gru_reset | `results/v8_students/v8_fixed_v8s_sam2.json` |
| `results/v8_students/v8_students_v8s_sam2-300.json` | **invalid** | no_gru_reset | `results/v8_students/v8_fixed_v8s_sam2.json` |

Re-measuring anything listed here: every checkpoint on this page predates the
bounded action convention of 2026-09-05, so it has to be run on the matching
`-V1` task id (`Mjlab-Pick-Place-PiperX-Vision-Robust-V1`, `...-Distill-Robust-V1`,
...); the default ids refuse it at the first step.

Defects:

* **noop_load** -- runner.load(load_cfg={'actor': True}) on a -Distill* task loads nothing (rsl_rl Distillation.load ignores the key): the numbers are a randomly initialised student.  Fixed in d94d944 (piper_push.evalcfg.load_weights).
* **no_gru_reset** -- eval_endurance/eval_occlusion did not reset the GRU hidden state on episode boundaries before 2026-09-04 01:26; a recurrent student carried the last episode's state into the next.  Fixed in 4cf9800 (reset_recurrent).
* **sensor_downgrade** -- accept_s1 --sensor measured replaced the task's noise model with the NOMINAL profile at strength 1.0; on a -Robust task that is a downgrade from the robust profile (strength 1.35, wider surface_fill/texture_penalty).  Fixed in d94d944 (piper_push.evalcfg.apply_sensor).
* **clean_sensor** -- measured with the depth noise and mask jitter off (the play-mode default on a nominal task).  A student distilled under the fitted sensor collapses under a clean one (results/audit_20260904: 4/min -> 0.2 vs ~32/min with the sensor on), so this is out-of-distribution, not a baseline.  Default changed to --sensor measured on 2026-09-04.
