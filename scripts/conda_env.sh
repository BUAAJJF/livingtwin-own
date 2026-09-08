#!/usr/bin/env bash
# Shared environment bootstrap for LivingTwin scripts.
#
# Local runs use Conda by default.  The remote launchers in scripts/pc/ still
# copy these scripts to the laboratory host, whose only supported runtime is
# micromamba; the fallback below keeps those SSH-launched jobs working.

CONDA_ENV=${CONDA_ENV:-${MJLAB_ENV:-livingtwin}}
CONDA_EXE=${CONDA_EXE:-conda}
PIPER_ENV_BACKEND=${PIPER_ENV_BACKEND:-auto}

conda_ready=0
if [ "$PIPER_ENV_BACKEND" != "micromamba" ] && command -v "$CONDA_EXE" >/dev/null 2>&1; then
  CONDA_BASE=$($CONDA_EXE info --base 2>/dev/null || true)
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] \
      && "$CONDA_EXE" env list 2>/dev/null | awk -v e="$CONDA_ENV" '$1 == e {found=1} END {exit !found}'; then
    conda_ready=1
  fi
fi

if [ "$conda_ready" = 1 ]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  # Remote laboratory jobs use this path/name; local jobs never reach this
  # branch while the requested Conda environment exists.
  MICROMAMBA_BIN=${MICROMAMBA_BIN:-${MM:-}}
  if [ -z "$MICROMAMBA_BIN" ] && command -v micromamba >/dev/null 2>&1; then
    MICROMAMBA_BIN=$(command -v micromamba)
  fi
  if [ -z "$MICROMAMBA_BIN" ] && [ -x /home/yunfan/.local/bin/micromamba ]; then
    MICROMAMBA_BIN=/home/yunfan/.local/bin/micromamba
  fi
  MICROMAMBA_ENV=${MICROMAMBA_ENV:-${ENV_NAME:-mjlab}}
  if [ "$PIPER_ENV_BACKEND" = "conda" ] || [ -z "$MICROMAMBA_BIN" ] || [ ! -x "$MICROMAMBA_BIN" ]; then
    echo "Conda environment '$CONDA_ENV' is unavailable" >&2
    echo "Run 'conda env list' and activate livingtwin, or set CONDA_ENV." >&2
    return 2 2>/dev/null || exit 2
  fi
  if ! "$MICROMAMBA_BIN" env list 2>/dev/null | awk -v e="$MICROMAMBA_ENV" '$1 == e {found=1} END {exit !found}'; then
    echo "micromamba environment '$MICROMAMBA_ENV' is unavailable" >&2
    return 2 2>/dev/null || exit 2
  fi
  eval "$("$MICROMAMBA_BIN" shell hook --shell bash)"
  micromamba activate "$MICROMAMBA_ENV"
fi

# mujoco_warp renders headlessly, but importing MuJoCo still probes GL.  The
# Conda runtime also supplies the libstdc++/ICU pair required by mjlab.
export MUJOCO_GL=${MUJOCO_GL:-disable}
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
