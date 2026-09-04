# Audit 2026-09-04: the ruler, re-measured on v4

Three runs of `scripts/eval_endurance.py` on the one checkpoint that has worked
on the arm, `hardware/deploy/policies/d455_v4_final/checkpoint.pt`, after the
fixes in commit `d94d944` (loader that proves the weights arrived; `--sensor`
that means what it says; provenance that records the domain).  1200 steps,
seed 101, no reset, 100-step windows.  Single seed each: these settle *which
mechanism* was at work, not a number to quote.

| run | task | sensor | envs | placed/min first → last window | early / late | late/early | started / stopped | why this condition |
|---|---|---|---|---|---|---|---|---|
| `v4_distill_robust_fixedload_sensor-task` | Distill-Robust | task (robust profile, strength 1.35) | 48 | 8.1 → 10.6 | 12.8 / 12.0 | 0.93 | 13 / 9 | the exact condition of results/decay/v4_final.json (twelve 0.0 windows) |
| `v4_vision_clean` | Vision | clean (strength 0, no jitter) | 128 | 4.0 → 0.2 | 1.2 / 0.2 | 0.16 | 17 / 17 | the default of eval_endurance/eval_occlusion |
| `v4_vision_measured` | Vision | measured (nominal D455, strength 1.0) | 128 | 18.5 → 31.6 | 32.4 / 31.5 | 0.97 | 79 / 16 | what accept_s1 --sensor measured meant on a nominal task |

Window sequences (placed/min):

```
distill_robust, sensor=task     8.1 15.6 17.5 13.1 11.2 11.2 11.9 10.6 15.6 7.5 15.6 10.6
vision, sensor=clean            4.0 2.1 0.7 0.0 0.2 0.2 0.5 0.0 0.5 0.0 0.0 0.2
vision, sensor=measured         18.5 34.7 38.0 35.9 35.2 32.1 32.3 32.1 34.2 28.8 30.0 31.6
```

## What this settles

1. **`results/decay/v4_final.json` was a random network.**  Same checkpoint,
   task, env count, steps and seed; the only change is that the weights are
   now loaded (the old `runner.load(load_cfg={"actor": True})` loads nothing
   on a `-Distill*` runner).  Twelve windows of 0.0 became a policy that
   places at 10-16/min to the end of the episode.  The "v4 only scores under
   accept_s1" rule, and the `--hold-shim` explanation in
   `results/accept_compare/README.md`, are both retired.

2. **The clean sensor kills v4.**  On the nominal `-Vision` task with the
   depth noise and mask jitter turned off -- the *default* of
   `eval_endurance.py` and `eval_occlusion.py`, and what `play=True` gives
   every script that does not ask -- v4 starts at 4/min and is below 0.3/min from
   the fourth window on, with every one of its starters dead by the end
   (late/early 0.16; the first pass of this run, before the provenance
   re-run, read 0.00).  Turn the
   sensor it trained with back on and the same run reads ~30/min flat for
   1200 steps, late/early 1.00, which is the accept_s1 number (29.95/min).
   A student distilled under the fitted D455 model has never seen a clean
   depth image; evaluating it on one is out-of-distribution, not "easier".
   Every within-episode decay measured on a student under the clean default
   before today has this confound in it.

3. **`--sensor measured` must be the task's own sensor.**  On the robust task
   the same policy reads 12/min under the robust profile versus 30/min under
   nominal noise on the nominal task -- the robust sensor plus robust
   dynamics cost v4 about 60%, which is the number the accept_compare
   `robust` rows were supposed to be measuring and were not (they ran the
   nominal sensor at strength 1.0 under the robust dynamics).

## What it does not settle

* Whether the v8/v9 students are worse than v4 -- they have not been re-run
  on the fixed ruler yet, and their runs need their training-domain
  environment variables (`domain.env`) as well as `--sensor measured`.
* Anything about the arm.  These are simulation numbers under the fitted
  sensor model.
* Seeds.  One each; the project's own rule is a median of three with the
  spread quoted before any number is compared to another.
