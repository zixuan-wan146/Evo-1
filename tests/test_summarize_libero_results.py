from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


def load_summary_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_libero_results.py"
    spec = importlib.util.spec_from_file_location("summarize_libero_results", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_result_file(path: Path, ckpt_name: str = "run_a") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": {"ckpt_name": ckpt_name},
                "metadata": {
                    "created_at_utc": "2026-06-11T00:00:00Z",
                    "git": {"commit": "abc123", "is_dirty": False},
                },
                "summary": {
                    "total_episodes": 3,
                    "successful_episodes": 2,
                    "failed_episodes": 1,
                    "success_rate": 2 / 3,
                    "average_decision_steps": 4.5,
                    "average_control_steps": 60.0,
                    "average_success_decision_steps": 3.0,
                    "suites": {
                        "libero_spatial": {
                            "total_episodes": 1,
                            "successful_episodes": 1,
                            "failed_episodes": 0,
                            "success_rate": 1.0,
                            "average_decision_steps": 2.0,
                            "average_control_steps": 28.0,
                            "average_success_decision_steps": 2.0,
                        },
                        "libero_goal": {
                            "total_episodes": 2,
                            "successful_episodes": 1,
                            "failed_episodes": 1,
                            "success_rate": 0.5,
                            "average_decision_steps": 5.75,
                            "average_control_steps": 76.0,
                            "average_success_decision_steps": 4.0,
                        },
                    },
                },
                "episodes": [],
            }
        )
    )
    return path


def test_discover_result_files_accepts_directories_and_globs(tmp_path):
    module = load_summary_module()
    first = write_result_file(tmp_path / "run_a_results.json", "run_a")
    second = write_result_file(tmp_path / "nested" / "run_b_results.json", "run_b")

    by_directory = module.discover_result_files([str(tmp_path)])
    by_glob = module.discover_result_files([str(tmp_path / "**" / "*_results.json")])

    assert set(by_directory) == {first.resolve(), second.resolve()}
    assert set(by_glob) == {first.resolve(), second.resolve()}


def test_load_result_rows_expands_overall_and_suite_rows(tmp_path):
    module = load_summary_module()
    result_file = write_result_file(tmp_path / "run_a_results.json")

    rows = module.load_result_rows(result_file)

    assert [row["scope"] for row in rows] == ["overall", "suite:libero_goal", "suite:libero_spatial"]
    assert rows[0]["run_name"] == "run_a"
    assert rows[0]["created_at_utc"] == "2026-06-11T00:00:00Z"
    assert rows[0]["git_commit"] == "abc123"
    assert rows[0]["git_dirty"] is False
    assert rows[0]["success_rate"] == 2 / 3
    assert rows[1]["successful_episodes"] == 1


def test_write_markdown_and_csv_tables(tmp_path):
    module = load_summary_module()
    rows = module.load_result_rows(write_result_file(tmp_path / "run_a_results.json"))
    markdown_path = tmp_path / "summary.md"
    csv_path = tmp_path / "summary.csv"

    module.write_markdown(rows, markdown_path)
    module.write_csv(rows, csv_path)

    markdown = markdown_path.read_text()
    assert "| result_file | run_name | created_at_utc | git_commit | git_dirty | scope |" in markdown
    assert "suite:libero_spatial" in markdown

    csv_rows = list(csv.DictReader(csv_path.open()))
    assert csv_rows[0]["scope"] == "overall"
    assert csv_rows[0]["run_name"] == "run_a"
    assert csv_rows[0]["git_commit"] == "abc123"
    assert csv_rows[0]["total_episodes"] == "3"


def test_main_reports_missing_inputs_without_traceback(capsys):
    module = load_summary_module()

    exit_code = module.main(["/tmp/definitely_missing_libero_results.json"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("ERROR: LIBERO result path not found:")
    assert "Traceback" not in captured.err
