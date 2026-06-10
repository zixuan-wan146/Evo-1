from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_libero_eval.sh"


def run_eval_script(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "EVO1_LIBERO_DRY_RUN": "1",
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def parse_env_output(stdout: str) -> dict[str, str]:
    result = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def test_run_libero_eval_script_uses_full_eval_defaults():
    result = run_eval_script()

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["EVO1_LIBERO_EPISODES"] == "10"
    assert env["EVO1_LIBERO_TASK_SUITES"] == "libero_spatial,libero_object,libero_goal,libero_10"
    assert env["EVO1_LIBERO_TASK_LIMIT"] == "0"
    assert env["EVO1_LIBERO_MAX_STEPS"] == "25,25,25,95"
    assert env["EVO1_LIBERO_HORIZON"] == "14"
    assert env["EVO1_LIBERO_CKPT_NAME"] == "Evo1_libero_eval"
    assert env["EVO1_LIBERO_RESULT_FILE"].endswith("Evo1_libero_eval_results.json")


def test_run_libero_eval_script_preserves_explicit_overrides():
    result = run_eval_script(
        {
            "EVO1_LIBERO_EPISODES": "2",
            "EVO1_LIBERO_TASK_SUITES": "libero_spatial",
            "EVO1_LIBERO_MAX_STEPS": "3",
            "EVO1_LIBERO_CKPT_NAME": "custom_eval",
        }
    )

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["EVO1_LIBERO_EPISODES"] == "2"
    assert env["EVO1_LIBERO_TASK_SUITES"] == "libero_spatial"
    assert env["EVO1_LIBERO_MAX_STEPS"] == "3"
    assert env["EVO1_LIBERO_CKPT_NAME"] == "custom_eval"
    assert env["EVO1_LIBERO_RESULT_FILE"].endswith("custom_eval_results.json")
