#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

source "$script_dir/libero_profile.sh"
load_libero_profile "$repo_root"

python_bin="${LIBERO_PYTHON:-python}"
run_dir="${EVO1_LIBERO_RUN_DIR:-}"

if [ -n "$run_dir" ]; then
  case "$run_dir" in
    /*) ;;
    *) run_dir="$repo_root/$run_dir" ;;
  esac
  export EVO1_LIBERO_RUN_DIR="$run_dir"
fi

export EVO1_SERVER_URI="${EVO1_SERVER_URI:-ws://127.0.0.1:9000}"
export EVO1_MUJOCO_GL="${EVO1_MUJOCO_GL:-osmesa}"
export EVO1_LIBERO_EPISODES="${EVO1_LIBERO_EPISODES:-1}"
export EVO1_LIBERO_TASK_SUITES="${EVO1_LIBERO_TASK_SUITES:-libero_spatial}"
export EVO1_LIBERO_TASK_LIMIT="${EVO1_LIBERO_TASK_LIMIT:-1}"
export EVO1_LIBERO_MAX_STEPS="${EVO1_LIBERO_MAX_STEPS:-1}"
export EVO1_LIBERO_HORIZON="${EVO1_LIBERO_HORIZON:-1}"
export EVO1_LIBERO_CKPT_NAME="${EVO1_LIBERO_CKPT_NAME:-Evo1_libero_smoke}"

if [ -n "$run_dir" ]; then
  export EVO1_LIBERO_LOG_DIR="${EVO1_LIBERO_LOG_DIR:-$run_dir/logs}"
  export EVO1_LIBERO_VIDEO_DIR="${EVO1_LIBERO_VIDEO_DIR:-$run_dir/videos}"
  export EVO1_LIBERO_RESULT_FILE="${EVO1_LIBERO_RESULT_FILE:-$run_dir/results/${EVO1_LIBERO_CKPT_NAME}_results.json}"
  export EVO1_LIBERO_MANIFEST_FILE="${EVO1_LIBERO_MANIFEST_FILE:-$run_dir/run_manifest.json}"
else
  export EVO1_LIBERO_LOG_DIR="${EVO1_LIBERO_LOG_DIR:-$repo_root/LIBERO_evaluation/log_file}"
  export EVO1_LIBERO_VIDEO_DIR="${EVO1_LIBERO_VIDEO_DIR:-$repo_root/LIBERO_evaluation/video_log_file/$EVO1_LIBERO_CKPT_NAME}"
  export EVO1_LIBERO_RESULT_FILE="${EVO1_LIBERO_RESULT_FILE:-$EVO1_LIBERO_LOG_DIR/${EVO1_LIBERO_CKPT_NAME}_results.json}"
  export EVO1_LIBERO_MANIFEST_FILE="${EVO1_LIBERO_MANIFEST_FILE:-$EVO1_LIBERO_LOG_DIR/${EVO1_LIBERO_CKPT_NAME}_run_manifest.json}"
fi
export EVO1_LIBERO_LOG_FILE="${EVO1_LIBERO_LOG_FILE:-$EVO1_LIBERO_LOG_DIR/$EVO1_LIBERO_CKPT_NAME.txt}"

if [ "$EVO1_MUJOCO_GL" = "egl" ]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

if [ "${EVO1_LIBERO_DRY_RUN:-0}" = "1" ]; then
  env | sort | grep '^EVO1_'
  exit 0
fi

mkdir -p \
  "$EVO1_LIBERO_LOG_DIR" \
  "$EVO1_LIBERO_VIDEO_DIR" \
  "$(dirname "$EVO1_LIBERO_RESULT_FILE")" \
  "$(dirname "$EVO1_LIBERO_MANIFEST_FILE")"

"$repo_root/scripts/write_libero_run_manifest.py" \
  --output "$EVO1_LIBERO_MANIFEST_FILE" \
  --run-kind smoke \
  --repo-root "$repo_root"

cd "$repo_root/LIBERO_evaluation"
exec "$python_bin" libero_client_4tasks.py
