from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_libero_smoke.sh"


def run_smoke_script(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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


def test_run_libero_smoke_script_uses_minimal_smoke_defaults():
    result = run_smoke_script()

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["EVO1_LIBERO_EPISODES"] == "1"
    assert env["EVO1_LIBERO_TASK_SUITES"] == "libero_spatial"
    assert env["EVO1_LIBERO_TASK_LIMIT"] == "1"
    assert env["EVO1_LIBERO_MAX_STEPS"] == "1"
    assert env["EVO1_LIBERO_HORIZON"] == "1"
    assert env["EVO1_LIBERO_CKPT_NAME"] == "Evo1_libero_smoke"
    assert env["EVO1_LIBERO_RESULT_FILE"].endswith("Evo1_libero_smoke_results.json")
    assert env["EVO1_LIBERO_MANIFEST_FILE"].endswith("Evo1_libero_smoke_run_manifest.json")


def test_run_libero_smoke_script_can_group_outputs_under_run_dir(tmp_path):
    run_dir = tmp_path / "libero_smoke_run"
    result = run_smoke_script({"EVO1_LIBERO_RUN_DIR": str(run_dir)})

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["EVO1_LIBERO_RUN_DIR"] == str(run_dir)
    assert env["EVO1_LIBERO_LOG_DIR"] == str(run_dir / "logs")
    assert env["EVO1_LIBERO_VIDEO_DIR"] == str(run_dir / "videos")
    assert env["EVO1_LIBERO_LOG_FILE"] == str(run_dir / "logs" / "Evo1_libero_smoke.txt")
    assert env["EVO1_LIBERO_RESULT_FILE"] == str(
        run_dir / "results" / "Evo1_libero_smoke_results.json"
    )
    assert env["EVO1_LIBERO_MANIFEST_FILE"] == str(run_dir / "run_manifest.json")
