#!/usr/bin/env bash
# 说明：让系统使用 bash 来解释本脚本。
# v10c: a teacher trained from random weights under the cold-start curriculum,
# evaluated on the full -Robust domain with three seeds, and frozen as a
# baseline only if it meets the criteria declared in config_snapshot.json
# BEFORE training starts.
#
#   TAG=v10c_coldstart_sight   SIGHT=1 GPU=0 nohup setsid bash scripts/run_v10c.sh >results/d455_heavy_dr/v10c_coldstart_sight/watcher.log 2>&1 &
#   TAG=v10c_coldstart_nosight SIGHT=0 GPU=0 nohup setsid bash scripts/run_v10c.sh >results/d455_heavy_dr/v10c_coldstart_nosight/watcher.log 2>&1 &
#
# No resume, no warm start, no checkpoint is loaded before training: the only
# checkpoint this script ever loads is the one it produced, for evaluation.
# If the GPU is busy the script waits (and says so) rather than kill anything.
# The result directory is never overwritten: an existing one is an error.
set -Eeuo pipefail
# 说明：遇到错误、未定义变量或管道中任一命令失败时立即停止。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 说明：取得脚本所在目录的绝对路径，避免从其他工作目录启动时找不到文件。
ROOT=${ROOT:-$SCRIPT_DIR/..}
# 说明：允许调用者覆盖仓库根目录，默认是 scripts/ 的上一级。
# CONDA_ENV is activated by scripts/conda_env.sh below.  MJLAB_ENV remains an
# alias so old local command lines fail less surprisingly.
CONDA_ENV=${CONDA_ENV:-${MJLAB_ENV:-livingtwin}}
# 说明：选择 Conda 环境；没有 CONDA_ENV 时兼容旧的 MJLAB_ENV，最终默认 livingtwin。
TAG=${TAG:?set TAG}
# 说明：要求提供本次实验标签；标签同时决定结果目录和训练 run 名称。
GPU=${GPU:?set GPU (physical index as nvidia-smi lists it)}
# 说明：要求提供物理 GPU 编号，编号必须与 nvidia-smi 显示的一致。
SIGHT=${SIGHT:?set SIGHT=1 or 0}
# 说明：要求明确是否启用 sight 视觉奖励/观测分支，1 是启用，0 是 NoSight。
MASS_GT=${MASS_GT:-0}
# 说明：是否训练带目标物体质量真值的 state teacher；默认关闭以保留 v10c 基线。
export MASS_GT
OUT=${OUT:-results/d455_heavy_dr/$TAG}
# 说明：设置总输出目录；调用者可用 OUT 指定其他目录。
SEED=${SEED:-42}
# 说明：设置训练随机种子，默认 42。
TEACHER_ENVS=${TEACHER_ENVS:-8192}
# 说明：设置并行仿真环境数量，默认 8192。
TEACHER_ITERS=${TEACHER_ITERS:-9000}
# 说明：设置 PPO 最大迭代次数，默认 9000。
# v10c evaluates on the plain -Robust domain.  v10d trains under a tighter slew
# ceiling (0.35 x trip) and is evaluated under it: the Cold2 play config is
# -Robust plus that ceiling plus the approach terms at zero weight.
# 说明：指定评估任务；留空时按 APPROACH/SIGHT 自动选择。
EVAL_TASK=${EVAL_TASK:-}
# 说明：设置 accept_s1 评估使用的并行环境数量，默认 256。
EVAL_ENVS=${EVAL_ENVS:-256}
# 说明：设置每个评估 seed 的控制步数，默认 2400。
EVAL_STEPS=${EVAL_STEPS:-2400}
# 说明：固定三个评估 seed，用空格分隔。
EVAL_SEEDS=${EVAL_SEEDS:-"101 202 303"}
# 说明：APPROACH=1 使用 v10d 的 Cold2 任务；默认 0 表示 v10c。
APPROACH=${APPROACH:-0}   # 1: the v10d variant with the approach terms (-Cold2 ids)
# 说明：根据 APPROACH 选择冷启动任务的基础 task id。
if [ "$APPROACH" = "1" ]; then BASE_TASK=Mjlab-Pick-Place-PiperX-Robust-Cold2; else BASE_TASK=Mjlab-Pick-Place-PiperX-Robust-Cold; fi
# 说明：质量真值只改变 teacher actor 的观测，因此使用独立的注册 task id。
if [ "$MASS_GT" = "1" ]; then BASE_TASK=${BASE_TASK}-Mass; fi
# 说明：根据 SIGHT 选择 Sight 或 NoSight 任务。
if [ "$SIGHT" = "1" ]; then TASK=$BASE_TASK; else TASK=${BASE_TASK}-NoSight; fi
# 说明：未显式指定评估任务时，v10c 训练任务在普通 Robust 域评估，Cold2 则在自身任务评估。
if [ -z "$EVAL_TASK" ]; then
  if [ "$APPROACH" = "1" ]; then EVAL_TASK=$TASK
  elif [ "$MASS_GT" = "1" ]; then EVAL_TASK=Mjlab-Pick-Place-PiperX-Robust-Mass
  else EVAL_TASK=Mjlab-Pick-Place-PiperX-Robust
  fi
fi

# 说明：切换到仓库根目录，保证相对路径都以仓库为基准。
cd "$ROOT"
# 说明：激活 Conda livingtwin；远端无 Conda 时由该脚本兼容 micromamba。
source "$ROOT/scripts/conda_env.sh"
# 说明：关闭 wandb 在线同步，训练记录保存在本地。
export WANDB_MODE=offline
# 说明：允许 PyTorch 使用可扩展 CUDA 内存段，降低显存碎片导致的 OOM 风险。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# 说明：统一使用无缓冲 Python，日志能够及时写入 watcher.log。
PY="python -u"
# 说明：输出带 ISO 时间戳的普通日志。
say()  { printf '[%s] %s\n' "$(date -Is)" "$*"; }
# 说明：输出错误日志并以失败状态退出。
fail() { printf '[%s] FAILED: %s\n' "$(date -Is)" "$*" >&2; exit 1; }
# 说明：返回某个阶段对应的完成标记文件路径。
marker() { printf '%s\n' "$OUT/stage_$1.done"; }
# 说明：检查阶段标记是否存在，用于安全地跳过已完成阶段。
stage_done() { [ -f "$(marker "$1")" ]; }
# 说明：写入当前时间，表示一个阶段已经完成。
mark_done()  { date -Is >"$(marker "$1")"; }
latest_checkpoint() {
  # 说明：函数接收训练 run 目录，并声明局部变量。
  local run=$1 file
  # 说明：找出该目录中版本号最大的 model_*.pt。
  file=$(find "$run" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -n 1)
  # 说明：没有 checkpoint 就终止，避免后续评估空路径。
  [ -n "$file" ] || fail "no model_*.pt under $run"
  # 说明：输出最终 checkpoint 的完整相对路径。
  printf '%s/%s\n' "$run" "$file"
}
run_dir_from_log() {
  # 说明：函数接收训练日志路径，并声明局部变量。
  local log=$1 dir
  # 说明：从 mjlab 日志中提取实际创建的实验目录。
  dir=$(grep -aE 'Logging experiment in directory: |\[INFO\] logging to ' "$log" | tail -n 1 | sed -E 's/.*(directory: |logging to )//')
  # 说明：日志没有记录 run 目录时终止。
  [ -n "$dir" ] || fail "no run directory recorded in $log"
  # 说明：去掉仓库根目录前缀，让后续路径与本地工作目录一致。
  dir=${dir#$ROOT/}
  # 说明：确认提取出的 run 目录确实存在。
  [ -d "$dir" ] || fail "run directory recorded in $log does not exist: $dir"
  # 说明：输出 run 目录。
  printf '%s\n' "$dir"
}
assert_complete() {
  # 说明：函数接收训练日志，并声明迭代计数变量。
  local log=$1 line reached target
  # 说明：取得日志中最后一条 Learning iteration 记录。
  line=$(grep -a 'Learning iteration' "$log" | tail -1)
  # 说明：从“当前/总数”文本中解析已经完成的迭代编号。
  reached=$(printf '%s' "$line" | sed -E 's#.*iteration ([0-9]+)/([0-9]+).*#\1#')
  # 说明：从同一行解析计划中的总迭代数。
  target=$(printf '%s' "$line" | sed -E 's#.*iteration ([0-9]+)/([0-9]+).*#\2#')
  # 说明：解析失败表示训练日志不完整，立即报错。
  [ "$reached" -ge 0 ] 2>/dev/null && [ "$target" -ge 1 ] 2>/dev/null || fail "cannot read the iteration counter out of $log"
  # 说明：要求最后一次记录已达到总迭代数，防止把中途崩溃当成完成。
  [ "$((reached + 1))" -ge "$target" ] || fail "$log stopped at $reached of $target -- killed or crashed"
}
# 说明：只有显存占用低于 5 GiB 时才认为目标 GPU 可用；这样允许少量后台进程。
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -lt 5120 ]; }

# 说明：SELFCHECK 模式只打印配置并做静态检查，不启动训练。
if [ -n "${SELFCHECK:-}" ]; then
  # 说明：声明当前为自检模式。
  say "SELFCHECK: nothing is launched"
  # 说明：打印本次运行将使用的关键参数。
  say "tag $TAG  task $TASK  mass_gt $MASS_GT  gpu $GPU  seed $SEED  envs $TEACHER_ENVS  iters $TEACHER_ITERS"
  # 说明：打印 GPU 名称和当前显存占用。
  say "gpu $GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i "$GPU")"
  # tests/test_cold_curriculum.py asserts this script carries no resume/warm-start flag
  # 说明：让 verdict 工具只生成配置快照，并截取可读摘要。
  $PY scripts/v10c_verdict.py --snapshot-only --task "$TASK" --sight "$SIGHT" --seed "$SEED" \
    --envs "$TEACHER_ENVS" --iters "$TEACHER_ITERS" --gpu "$GPU" --out /dev/stdout 2>&1 | grep -v "^\[" | head -60
  # 说明：自检成功，退出且不训练。
  say "SELFCHECK passed"
  exit 0
fi

# 说明：拒绝覆盖已有结果目录；只有显式允许时才可恢复旧目录。
[ -e "$OUT" ] && [ -z "${ALLOW_RESUME_OF_THIS_RUN:-}" ] && fail "$OUT exists; a v10c run is never overwritten (set a new TAG)"
# 说明：创建本次实验的结果目录。
mkdir -p "$OUT"
# 说明：记录遥测文件位置和当前阶段标签。
export PIPER_U_TELEMETRY="$OUT/u_telemetry.jsonl" PIPER_U_TELEMETRY_TAG=teacher
# 说明：记录课程阶段切换信息。
export PIPER_CURRICULUM_LOG="$OUT/curriculum.jsonl"
# 说明：输出实际训练配置摘要。
say "v10c ($TAG) task $TASK (mass_gt=$MASS_GT) on physical GPU $GPU, seed $SEED, budget $TEACHER_ITERS iterations"

# --- the declaration, before anything trains --------------------------------
# 说明：把训练命令完整保存下来，供配置快照和日志复现。
TRAIN_CMD="MASS_GT=$MASS_GT TASK=$TASK NUM_ENVS=$TEACHER_ENVS ITERS=$TEACHER_ITERS GPUS=[$GPU] RUN_NAME=${TAG}_teacher bash scripts/train.sh --agent.logger tensorboard --agent.seed $SEED"
# 说明：训练前先冻结 task、seed、环境数、迭代数和评估任务等声明。
$PY scripts/v10c_verdict.py --snapshot-only --task "$TASK" --sight "$SIGHT" --seed "$SEED" \
  --envs "$TEACHER_ENVS" --iters "$TEACHER_ITERS" --gpu "$GPU" --command "$TRAIN_CMD" --eval-task "$EVAL_TASK" --out "$OUT/config_snapshot.json" \
  >"$OUT/config_snapshot.log" 2>&1 || fail "config snapshot -- see $OUT/config_snapshot.log"
# 说明：提示配置快照已经写入。
say "wrote $OUT/config_snapshot.json"

# --- queue for the GPU, never kill ------------------------------------------
# 说明：循环等待目标 GPU 空闲，不杀掉其他任务。
until gpu_free; do
  # 说明：打印当前占用并等待两分钟后重试。
  say "GPU $GPU is busy ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU") used); queued, checking again in 120 s"
  sleep 120
done

# --- 1. the teacher, from random weights --------------------------------------
# 说明：如果 teacher 阶段没有完成标记，则从随机权重启动冷训练。
if ! stage_done teacher; then
  # 说明：打印即将执行的训练命令。
  say "teacher: $TRAIN_CMD"
  # 说明：在 livingtwin 中运行 train.sh，并把 stdout/stderr 保存到 teacher.log。
  if ! TASK="$TASK" NUM_ENVS="$TEACHER_ENVS" ITERS="$TEACHER_ITERS" GPUS="[$GPU]" RUN_NAME="${TAG}_teacher" \
       bash scripts/train.sh --agent.logger tensorboard --agent.seed "$SEED" >"$OUT/teacher.log" 2>&1; then
    # 说明：训练进程非零退出时指出日志位置并终止。
    fail "teacher -- see $OUT/teacher.log"
  fi
  # 说明：训练成功后写入 teacher 阶段完成标记。
  mark_done teacher
fi
# 说明：从 teacher.log 找到 mjlab 实际生成的训练目录。
TEACHER_RUN=$(run_dir_from_log "$OUT/teacher.log")
# 说明：确认日志显示训练达到了目标迭代数。
assert_complete "$OUT/teacher.log"
# 说明：选择该 run 中编号最大的最终 checkpoint，不挑选中间模型。
RT=$(latest_checkpoint "$TEACHER_RUN")   # declared: the FINAL checkpoint, never a picked one
# 说明：把 teacher checkpoint 路径写入结果目录供后续脚本读取。
printf '%s\n' "$RT" >"$OUT/teacher_checkpoint.txt"
# 说明：输出最终 teacher 路径。
say "teacher $RT"

# --- 2. fixed three-seed evaluation on the full -Robust domain ---------------
# 说明：如果评估阶段尚未完成，则运行固定的三个 seed。
if ! stage_done eval; then
  # 说明：遍历 101、202、303 三个评估 seed。
  for s in $EVAL_SEEDS; do
    # 说明：运行 accept_s1，通过统一尺子评估吞吐和成功率。
    $PY scripts/accept_s1.py "$EVAL_TASK" "$RT" \
      --num-envs "$EVAL_ENVS" --steps "$EVAL_STEPS" --seed "$s" --device "cuda:$GPU" --sensor measured \
      --label "teacher_robust_s$s" --json "$OUT/accept_teacher_robust_s$s.json" >"$OUT/accept_teacher_robust_s$s.log" 2>&1 \
      || say "accept seed $s failed (see log)"
    # 说明：运行无 reset 的 endurance 评估，检查 episode 后半段是否衰退。
    $PY scripts/eval_endurance.py --checkpoint "$RT" --task "$EVAL_TASK" \
      --num-envs 128 --steps 1200 --seed "$s" --device "cuda:$GPU" --sensor measured \
      --out "$OUT/endurance_teacher_s$s.json" >"$OUT/endurance_teacher_s$s.log" 2>&1 \
      || say "endurance seed $s failed (see log)"
  done
  # 说明：运行 300 步遮挡评估，检查手臂/夹爪是否遮挡目标。
  $PY scripts/eval_occlusion.py --checkpoint "$RT" --task "$EVAL_TASK" \
    --num-envs 128 --steps 300 --seed 101 --device "cuda:$GPU" --sensor measured \
    --out "$OUT/occlusion_teacher.json" >"$OUT/occlusion_teacher.log" 2>&1 || say "occlusion failed (see log)"
  # 说明：三个 seed 和遮挡评估都已尝试，写入评估阶段标记。
  mark_done eval
fi

# --- 3. the verdict against the declaration, and the freeze ----------------------
# 说明：把评估结果与训练前声明的门槛进行汇总，生成 verdict.json。
$PY scripts/v10c_verdict.py --out-dir "$OUT" --checkpoint "$RT" \
  >"$OUT/verdict.log" 2>&1 || true
# 说明：在终端显示结论日志的最后 30 行。
cat "$OUT/verdict.log" | tail -30
# 说明：将结果目录设为只读，防止评估后被意外修改。
chmod -R a-w "$OUT" 2>/dev/null || true
# 说明：提示整个 v10c pipeline 完成以及结论文件位置。
say "v10c ($TAG) done; verdict in $OUT/verdict.json (directory is now read-only)"
