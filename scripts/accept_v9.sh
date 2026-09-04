#!/usr/bin/env bash
# Score the v9 teachers the way v9 was asked for, and not the other way.
#
# The trap this exists to avoid: `RESET_FULL_RANGE=1` is a training-time env
# var, so an evaluation that forgets it scores a start-from-anywhere teacher on
# the narrow reset -- i.e. measures the part that already worked.  Every row
# below is run BOTH ways, and the pair is the answer to "can it now start from
# anywhere", which the single number cannot give.
set -Eeuo pipefail
OUT=${OUT:-results/d455_heavy_dr/v9_fullreset}
DEV=${DEV:-cuda:0}
ENVS=${ENVS:-256}
MM=${MM:-micromamba}
export MUJOCO_GL=disable PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/home/yunfan/micromamba/envs/mjlab/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$OUT"

for ck in "$@"; do
  name=$(basename "${ck%.pt}")
  for mode in narrow full; do
    if [ "$mode" = full ]; then export RESET_FULL_RANGE=1; else unset RESET_FULL_RANGE; fi
    $MM run -n mjlab python scripts/eval_endurance.py --checkpoint "$ck" \
      --num-envs "$ENVS" --steps 1200 --device "$DEV" --seed 101 \
      --out "$OUT/${name}_end_$mode.json" >/dev/null 2>&1 || true
    $MM run -n mjlab python scripts/eval_occlusion.py --checkpoint "$ck" \
      --num-envs "$ENVS" --steps 300 --device "$DEV" --seed 101 \
      --out "$OUT/${name}_occ_$mode.json" >/dev/null 2>&1 || true
  done
  unset RESET_FULL_RANGE
done

$MM run -n mjlab python - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
names = sorted({os.path.basename(f).rsplit("_end_", 1)[0]
                for f in glob.glob(f"{out}/*_end_*.json")})
print(f"{'teacher':<22}{'reset':<8}{'placed/min':>11}{'late/early':>12}"
      f"{'engaged blk':>13}{'|da|':>8}")
for n in names:
  for mode in ("narrow", "full"):
    e = f"{out}/{n}_end_{mode}.json"; o = f"{out}/{n}_occ_{mode}.json"
    if not (os.path.exists(e) and os.path.exists(o)):
      continue
    d, c = json.load(open(e)), json.load(open(o))
    print(f"{n:<22}{mode:<8}{c['placed_per_min']:>11.1f}"
          f"{d['late_over_early']:>12.2f}"
          f"{100*c['engaged']['blocked_rate']:>12.1f}%{c['action_rate']:>8.3f}")
print("\nnarrow = the reset every earlier result was measured on;")
print("full   = RESET_FULL_RANGE=1, which is what v9 was trained for.")
print("A teacher that only works on the narrow row has not learned to start")
print("from anywhere; it has learned to be evaluated leniently.")
PY
