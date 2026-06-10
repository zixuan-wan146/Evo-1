#!/usr/bin/env bash

libero_profile_allowed_key() {
  case "$1" in
    EVO1_SERVER_URI | \
      EVO1_MUJOCO_GL | \
      EVO1_LIBERO_EPISODES | \
      EVO1_LIBERO_TASK_SUITES | \
      EVO1_LIBERO_TASK_LIMIT | \
      EVO1_LIBERO_MAX_STEPS | \
      EVO1_LIBERO_HORIZON | \
      EVO1_LIBERO_CKPT_NAME | \
      EVO1_LIBERO_RUN_DIR | \
      EVO1_LIBERO_LOG_DIR | \
      EVO1_LIBERO_VIDEO_DIR | \
      EVO1_LIBERO_LOG_FILE | \
      EVO1_LIBERO_RESULT_FILE | \
      EVO1_LIBERO_MANIFEST_FILE | \
      LIBERO_PYTHON | \
      PYOPENGL_PLATFORM)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

load_libero_profile() {
  local repo_root=$1
  local profile=${EVO1_LIBERO_PROFILE:-}
  local line key value

  if [ -z "$profile" ]; then
    return 0
  fi

  case "$profile" in
    /*) ;;
    *) profile="$repo_root/$profile" ;;
  esac

  if [ ! -f "$profile" ]; then
    printf '[libero-profile] ERROR: profile file does not exist: %s\n' "$profile" >&2
    return 1
  fi

  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      "" | "#"*) continue ;;
    esac
    case "$line" in
      *=*) ;;
      *)
        printf '[libero-profile] ERROR: invalid line in %s: %s\n' "$profile" "$line" >&2
        return 1
        ;;
    esac

    key=${line%%=*}
    value=${line#*=}
    if ! libero_profile_allowed_key "$key"; then
      printf '[libero-profile] ERROR: unsupported key in %s: %s\n' "$profile" "$key" >&2
      return 1
    fi
    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$profile"

  export EVO1_LIBERO_PROFILE="$profile"
}
