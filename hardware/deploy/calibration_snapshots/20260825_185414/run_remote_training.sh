#!/usr/bin/env bash
set -euo pipefail

WT=/home/yunfan/work/piper-push/LivingTwin-calib-20260825
MAIN=/home/yunfan/work/piper-push/LivingTwin
MAMBA=/home/yunfan/.local/bin/micromamba
URDF="$MAIN/piperx-mjlab/src/piper_mjlab/assets/agilex_piper_x/piper_x.urdf"
CRITIC="$MAIN/logs/rsl_rl/piperx_pick_place/2026-08-22_07-55-30_h_full/model_3400.pt"

cd "$WT"
export PYTHONPATH="$WT/src:$WT"
export PIPER_X_URDF="$URDF"
export MUJOCO_GL=disable
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/home/yunfan/micromamba/envs/mjlab/lib:${LD_LIBRARY_PATH:-}"

"$MAMBA" run -a '' -n mjlab python -u scripts/distill.py \
  --teacher "$CRITIC" \
  --num-envs 1024 \
  --iterations 1500 \
  --run-name calib_d405_distill_v1 \
  --device cuda:0 \
  --seed 42 \
  --cadence object \
  --sensor measured \
  --logger tensorboard

DISTILL_RUN=$(find "$MAIN/logs/rsl_rl/piperx_pick_place_distill" \
  -maxdepth 1 -type d -name '*_calib_d405_distill_v1' -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)
STUDENT="$DISTILL_RUN/model_1499.pt"
test -f "$STUDENT"

"$MAMBA" run -a '' -n mjlab python -u scripts/finetune.py \
  --student "$STUDENT" \
  --critic "$CRITIC" \
  --num-envs 1024 \
  --iterations 1500 \
  --run-name calib_d405_finetune_v1 \
  --device cuda:0 \
  --seed 42 \
  --cadence object \
  --logger tensorboard
