#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[check-repo] %s\n' "$*" >&2
}

run_step() {
  local name=$1
  shift

  if [ "${EVO1_CHECK_DRY_RUN:-0}" = "1" ]; then
    printf '[check-repo] DRY-RUN %s:' "$name"
    printf ' %q' "$@"
    printf '\n'
    return
  fi

  log "$name"
  "$@"
}

check_shell_syntax() {
  local script
  if [ "${EVO1_CHECK_DRY_RUN:-0}" = "1" ]; then
    printf '[check-repo] DRY-RUN shell syntax: bash -n scripts/*.sh\n'
    return
  fi

  log "Shell script syntax"
  while IFS= read -r -d '' script; do
    bash -n "$script"
  done < <(find scripts -name "*.sh" -print0)
}

run_ruff() {
  if [ "${EVO1_CHECK_SKIP_RUFF:-0}" = "1" ]; then
    log "Skipping ruff because EVO1_CHECK_SKIP_RUFF=1"
    return
  fi

  if [ "${EVO1_CHECK_DRY_RUN:-0}" = "1" ]; then
    run_step "Ruff lint" "$python_bin" -m ruff check .
    return
  fi

  if "$python_bin" -m ruff --version >/dev/null 2>&1; then
    run_step "Ruff lint" "$python_bin" -m ruff check .
    return
  fi

  if [ "${EVO1_CHECK_REQUIRE_RUFF:-0}" = "1" ]; then
    log "ERROR: ruff is required but not importable by $python_bin"
    exit 1
  fi

  log "WARN: ruff is not installed for $python_bin; skipped lint"
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON:-python3}"

cd "$repo_root"

run_step "Requirements policy audit" "$python_bin" scripts/audit_requirements.py

if [ "${EVO1_CHECK_SKIP_PYTEST:-0}" = "1" ]; then
  log "Skipping pytest because EVO1_CHECK_SKIP_PYTEST=1"
else
  run_step "Unit tests" "$python_bin" -m pytest
fi

run_ruff
check_shell_syntax
run_step "Repository preflight" "$python_bin" scripts/preflight.py
run_step "LIBERO setup dry-run" env EVO1_SETUP_LIBERO_DRY_RUN=1 "$script_dir/setup_libero_env.sh"
run_step \
  "LIBERO checkpoint download dry-run" \
  env EVO1_DOWNLOAD_LIBERO_CHECKPOINT_DRY_RUN=1 "$script_dir/download_libero_checkpoint.sh"
run_step \
  "LIBERO smoke profile dry-run" \
  env EVO1_LIBERO_DRY_RUN=1 EVO1_LIBERO_PROFILE=configs/libero_profiles/smoke.env \
  "$script_dir/run_libero_smoke.sh"
run_step \
  "LIBERO eval profile dry-run" \
  env EVO1_LIBERO_DRY_RUN=1 EVO1_LIBERO_PROFILE=configs/libero_profiles/full_eval.env \
  "$script_dir/run_libero_eval.sh"

if [ "${EVO1_CHECK_SKIP_COMPILE:-0}" = "1" ]; then
  log "Skipping compileall because EVO1_CHECK_SKIP_COMPILE=1"
else
  run_step \
    "Python compileall" \
    env PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/evo1_pycache}" \
    "$python_bin" -m compileall -q Evo_1 MetaWorld_evaluation LIBERO_evaluation scripts tests
fi

run_step "Git whitespace check" git diff --check
log "All requested checks passed"
