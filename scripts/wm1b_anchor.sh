#!/usr/bin/env bash
# The two anchors WM1-B needs and Phase WM0 never measured.
#
# WM0 ran the known-parameter fine-tune for the latency axis only; for damping
# it recorded the zero-shot penalty as "-4.5% throughput, x89 trips" and left
# the ceiling command in the report unrun.  Recovery and the safety gate both
# need the numbers themselves, measured the same way every adapted run is
# measured, so they are measured here rather than reconstructed from a
# percentage.
#
#   scripts/wm1b_anchor.sh <cuda_device>
#
# `zeroshot` is the deployed policy in the target domain with no adaptation at
# all; `nominal` is the same weights in the source domain.  Three repeats each,
# same evaluation seeds as every adapted run, so the comparison is paired.
set -Eeuo pipefail

GPU=${1:?usage: wm1b_anchor.sh <cuda_device>}
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-disable}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MAMBA=${MAMBA:-$(command -v micromamba || true)}
for c in "$HOME/.local/bin/micromamba" /usr/local/bin/micromamba; do
  if [ -z "$MAMBA" ] && [ -x "$c" ]; then MAMBA="$c"; fi
done
[ -n "$MAMBA" ] || { echo "!!! micromamba not found; set MAMBA" >&2; exit 3; }
PREFIX="$("$MAMBA" env list | awk '$1=="mjlab" {print $NF}')"
[ -n "$PREFIX" ] && export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Hide every card but this shard's own.
#
# `--device cuda:5` does NOT stop a process from touching cuda:0.  torch and
# warp both initialise a context on device 0 during startup regardless of which
# device the work runs on, and each of those contexts is ~464 MiB that stays
# for the life of the process.  Nine concurrent jobs put 4.2 GiB on a card this
# project was told to stay off entirely -- all of that card's used memory, in
# fact, so it looked occupied while being idle.
#
# CUDA_VISIBLE_DEVICES makes it impossible rather than unlikely: the process
# cannot enumerate the other cards at all.  Inside the process the one visible
# card is renumbered to 0, so every --device below is cuda:0 and the physical
# card is chosen here and only here.  `nvidia-smi -i` keeps using the global
# index, which is what the memory wait wants.
export CUDA_VISIBLE_DEVICES="$GPU"
DEV=cuda:0

BASE=logs/rsl_rl/piperx_pick_place_vision/2026-08-22_17-15-09_f3/model_1500.pt
TASK=Mjlab-Pick-Place-PiperX-Vision
OUT=results/wm1_damping/adapt
mkdir -p "$OUT"

r=0
for SEED_E in 20260823 31415926 27182818; do
  for KIND in zeroshot nominal; do
    J="$OUT/anchor_${KIND}__r$r.json"
    [ -e "$J" ] && { echo "=== $J exists"; continue; }
    EXTRA=()
    [ "$KIND" = zeroshot ] && EXTRA=(--servo-damping-scale 0.75)
    echo "=== [gpu $GPU] anchor $KIND repeat $r"
    "$MAMBA" run -n mjlab python scripts/accept_s1.py "$TASK" "$BASE" \
      --num-envs 512 --steps 2400 --seed "$SEED_E" --device "$DEV" \
      "${EXTRA[@]}" \
      --label "anchor_${KIND}__r$r" --json "$J" \
      > "$OUT/anchor_${KIND}__r$r.log" 2>&1
  done
  r=$((r + 1))
done
echo "=== WM1-B ANCHORS DONE ==="
