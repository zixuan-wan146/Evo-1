from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "write_libero_run_manifest.py"


def test_write_libero_run_manifest_records_run_context(tmp_path: Path):
    output_path = tmp_path / "run" / "run_manifest.json"
    env = {
        **os.environ,
        "EVO1_LIBERO_RUN_DIR": str(tmp_path / "run"),
        "EVO1_LIBERO_RESULT_FILE": str(tmp_path / "run" / "results" / "smoke_results.json"),
        "EVO1_LIBERO_CKPT_NAME": "smoke",
        "EVO1_SERVER_URI": "ws://127.0.0.1:9000",
        "EVO1_TOKEN": "should-not-be-written",
    }

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output",
            str(output_path),
            "--run-kind",
            "smoke",
            "--repo-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(output_path)
    payload = json.loads(output_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["run_kind"] == "smoke"
    assert payload["libero"]["EVO1_LIBERO_CKPT_NAME"] == "smoke"
    assert payload["libero"]["EVO1_LIBERO_RUN_DIR"] == str(tmp_path / "run")
    assert payload["libero"]["EVO1_SERVER_URI"] == "ws://127.0.0.1:9000"
    assert payload["metadata"]["git"]["repo_root"] == str(REPO_ROOT)
    assert "EVO1_TOKEN" not in payload["metadata"]["environment"]
    assert "EVO1_TOKEN" not in payload["libero"]
