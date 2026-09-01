#!/usr/bin/env bash
set -euo pipefail

# Calibrated camera/table plus the physical rig's +90 degree task layout.
# The state teacher is retrained first: shifting the workspace also shifts
# joint1 by pi/2, so the old state policy is not a valid oracle for this task.
WT=/home/yunfan/work/piper-push/LivingTwin-calib-20260825
MAIN=/home/yunfan/work/piper-push/LivingTwin
MAMBA=/home/yunfan/.local/bin/micromamba
URDF="$MAIN/piperx-mjlab/src/piper_mjlab/assets/agilex_piper_x/piper_x.urdf"
RESULTS="$WT/results/calibrated_rot90"

cd "$WT"
mkdir -p "$RESULTS"
export PYTHONPATH="$WT/src:$WT"
export PIPER_X_URDF="$URDF"
export MUJOCO_GL=disable
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/home/yunfan/micromamba/envs/mjlab/lib:${LD_LIBRARY_PATH:-}"

echo "[$(date -Is)] training rotated state teacher"
"$MAMBA" run -a '' -n mjlab train Mjlab-Pick-Place-PiperX \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 2000 \
  --agent.run-name calib_rot90_teacher_v1 \
  --agent.seed 42 \
  --agent.logger tensorboard \
  --gpu-ids '[0]'

TEACHER_RUN=$(find "$WT/logs/rsl_rl/piperx_pick_place" \
  -maxdepth 1 -type d -name '*_calib_rot90_teacher_v1' -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)
TEACHER="$TEACHER_RUN/model_1999.pt"
test -f "$TEACHER"

echo "[$(date -Is)] evaluating rotated state teacher: $TEACHER"
"$MAMBA" run -a '' -n mjlab python -u scripts/accept_s1.py \
  Mjlab-Pick-Place-PiperX "$TEACHER" \
  --num-envs 512 --steps 3000 --seed 4201 \
  --json "$RESULTS/teacher.json" --label calib_rot90_teacher_v1

echo "[$(date -Is)] D405 distillation"
"$MAMBA" run -a '' -n mjlab python -u scripts/distill.py \
  --teacher "$TEACHER" \
  --num-envs 1024 \
  --iterations 1500 \
  --run-name calib_rot90_d405_distill_v1 \
  --device cuda:0 \
  --seed 42 \
  --cadence object \
  --sensor measured \
  --logger tensorboard

DISTILL_RUN=$(find "$WT/logs/rsl_rl/piperx_pick_place_distill" \
  -maxdepth 1 -type d -name '*_calib_rot90_d405_distill_v1' -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)
STUDENT="$DISTILL_RUN/model_1499.pt"
test -f "$STUDENT"

echo "[$(date -Is)] PPO fine-tuning"
"$MAMBA" run -a '' -n mjlab python -u scripts/finetune.py \
  --student "$STUDENT" \
  --critic "$TEACHER" \
  --num-envs 1024 \
  --iterations 1500 \
  --run-name calib_rot90_d405_finetune_v1 \
  --device cuda:0 \
  --seed 42 \
  --cadence object \
  --logger tensorboard

FINETUNE_RUN=$(find "$WT/logs/rsl_rl/piperx_pick_place_vision" \
  -maxdepth 1 -type d -name '*_calib_rot90_d405_finetune_v1' -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)
POLICY="$FINETUNE_RUN/model_1499.pt"
test -f "$POLICY"

echo "[$(date -Is)] evaluating measured-D405 vision policy: $POLICY"
"$MAMBA" run -a '' -n mjlab python -u scripts/accept_s1.py \
  Mjlab-Pick-Place-PiperX-Vision "$POLICY" \
  --num-envs 512 --steps 3000 --seed 4202 \
  --sensor measured --cadence object \
  --json "$RESULTS/vision_measured.json" \
  --label calib_rot90_d405_finetune_v1

echo "[$(date -Is)] rotated calibrated pipeline complete"
