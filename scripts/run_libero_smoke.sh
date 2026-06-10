#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

python_bin="${LIBERO_PYTHON:-python}"

export EVO1_SERVER_URI="${EVO1_SERVER_URI:-ws://127.0.0.1:9000}"
export EVO1_MUJOCO_GL="${EVO1_MUJOCO_GL:-osmesa}"
export EVO1_LIBERO_EPISODES="${EVO1_LIBERO_EPISODES:-1}"
export EVO1_LIBERO_TASK_SUITES="${EVO1_LIBERO_TASK_SUITES:-libero_spatial}"
export EVO1_LIBERO_TASK_LIMIT="${EVO1_LIBERO_TASK_LIMIT:-1}"
export EVO1_LIBERO_MAX_STEPS="${EVO1_LIBERO_MAX_STEPS:-1}"
export EVO1_LIBERO_HORIZON="${EVO1_LIBERO_HORIZON:-1}"
export EVO1_LIBERO_CKPT_NAME="${EVO1_LIBERO_CKPT_NAME:-Evo1_libero_smoke}"
export EVO1_LIBERO_LOG_DIR="${EVO1_LIBERO_LOG_DIR:-$repo_root/LIBERO_evaluation/log_file}"
export EVO1_LIBERO_VIDEO_DIR="${EVO1_LIBERO_VIDEO_DIR:-$repo_root/LIBERO_evaluation/video_log_file/$EVO1_LIBERO_CKPT_NAME}"
export EVO1_LIBERO_LOG_FILE="${EVO1_LIBERO_LOG_FILE:-$EVO1_LIBERO_LOG_DIR/$EVO1_LIBERO_CKPT_NAME.txt}"

if [ "$EVO1_MUJOCO_GL" = "egl" ]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

cd "$repo_root/LIBERO_evaluation"
exec "$python_bin" libero_client_4tasks.py
