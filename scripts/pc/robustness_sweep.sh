#!/usr/bin/env bash
# The simulator's sim-to-real proxy: the initiation ruler under fixed perturbations the
# policy was not trained for.  One JSON per level; scripts/pc/report_robustness.py tabulates.
#
#   scripts/pc/robustness_sweep.sh <checkpoint> <task> <out dir> [device]
set -Eeuo pipefail
CK=$1; TASK=$2; OUT=$3; DEV=${4:-cuda:0}
mkdir -p "$OUT"
export MUJOCO_GL=disable PYTHONUNBUFFERED=1
PY="micromamba run -n mjlab python -u"
LEVELS=${LEVELS:-"none plane=0.004 plane=-0.004 plane=0.008 drop=0.3 drop=0.6 offset=0.01 offset=0.02 jitter=0.005 noise=1.5 noise=2.0 campos=0.02,camrot=2 campos=0.04,camrot=4"}
for lv in $LEVELS; do
  name=$(echo "$lv" | tr '=,' '_-')
  [ -f "$OUT/ini_$name.json" ] && continue
  if [ "$lv" = none ]; then arg=""; else arg="--perturb $lv"; fi
  $PY scripts/pc/eval_initiation.py --checkpoint "$CK" --task "$TASK" --num-envs 256 --steps 1800 --seed 101 --device "$DEV" \
    --sensor measured $arg --out "$OUT/ini_$name.json" >"$OUT/ini_$name.log" 2>&1 || echo "level $lv failed"
done
echo "sweep done" >>"$OUT/sweep.log"
