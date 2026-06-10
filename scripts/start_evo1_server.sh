#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

python_bin="${EVO1_PYTHON:-python}"
preflight_python="${EVO1_PREFLIGHT_PYTHON:-$python_bin}"
ckpt_dir="${EVO1_CKPT_DIR:-${1:-}}"
host="${EVO1_HOST:-127.0.0.1}"
port="${EVO1_PORT:-9000}"
device="${EVO1_DEVICE:-cuda:0}"
inference_steps="${EVO1_INFERENCE_STEPS:-1}"
skip_preflight="${EVO1_SKIP_PREFLIGHT:-0}"

if [ -z "$ckpt_dir" ]; then
  printf 'Usage: EVO1_PYTHON=/path/to/python %s /path/to/Evo1_LIBERO_checkpoint\n' "$0" >&2
  printf 'Or set EVO1_CKPT_DIR=/path/to/checkpoint.\n' >&2
  exit 2
fi

if [ ! -d "$ckpt_dir" ]; then
  printf 'Checkpoint directory does not exist: %s\n' "$ckpt_dir" >&2
  exit 2
fi

if [ "$skip_preflight" != "1" ]; then
  "$preflight_python" "$repo_root/scripts/preflight.py" \
    --dataset-config "" \
    --checkpoint "$ckpt_dir" \
    --skip-shell-syntax
fi

cd "$repo_root/Evo_1"
exec "$python_bin" scripts/Evo1_server.py \
  --ckpt_dir "$ckpt_dir" \
  --host "$host" \
  --port "$port" \
  --device "$device" \
  --inference_steps "$inference_steps"
