from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def load_preflight_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
    spec = importlib.util.spec_from_file_location("evo1_preflight", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result_levels(report):
    return [result.level for result in report.results]


def valid_libero_result_payload() -> dict:
    return {
        "config": {"ckpt_name": "smoke"},
        "metadata": {"git": {"commit": "abc123", "is_dirty": False}},
        "summary": {
            "total_episodes": 2,
            "successful_episodes": 1,
            "failed_episodes": 1,
            "success_rate": 0.5,
            "average_decision_steps": 2.5,
            "average_control_steps": 10.0,
            "average_success_decision_steps": 2.0,
            "suites": {
                "libero_spatial": {
                    "total_episodes": 2,
                    "successful_episodes": 1,
                    "failed_episodes": 1,
                    "success_rate": 0.5,
                    "average_decision_steps": 2.5,
                    "average_control_steps": 10.0,
                    "average_success_decision_steps": 2.0,
                }
            },
        },
        "episodes": [
            {
                "task_suite": "libero_spatial",
                "task_id": 0,
                "episode_id": 0,
                "task_description": "pick up the object",
                "success": True,
                "decision_steps": 2,
                "control_steps": 14,
                "failure_reason": "",
                "video_path": "task1_episode1.mp4",
            },
            {
                "task_suite": "libero_spatial",
                "task_id": 0,
                "episode_id": 1,
                "task_description": "pick up the object",
                "success": False,
                "decision_steps": 3,
                "control_steps": 6,
                "failure_reason": "max_steps_exhausted",
                "video_path": "task1_episode2.mp4",
            },
        ],
    }


def valid_checkpoint_config() -> dict:
    return {
        "horizon": 14,
        "per_action_dim": 7,
        "state_dim": 7,
        "action_dim": 98,
    }


def valid_norm_stats() -> dict:
    return {
        "libero": {
            "observation.state": {
                "min": [0.0, -1.0, -2.0],
                "max": [1.0, 2.0, 3.0],
            },
            "action": {
                "min": [-1.0] * 7,
                "max": [1.0] * 7,
            },
        }
    }


def test_checkpoint_validation_accepts_required_files(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(json.dumps(valid_checkpoint_config()))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(valid_norm_stats()))
    (ckpt_dir / "mp_rank_00_model_states.pt").write_bytes(b"checkpoint")

    report = preflight.Report()
    preflight.check_checkpoint_dir(ckpt_dir, report)

    assert result_levels(report) == ["OK"]


def test_checkpoint_validation_rejects_missing_weight_file(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text("{}")
    (ckpt_dir / "norm_stats.json").write_text("{}")

    report = preflight.Report()
    preflight.check_checkpoint_dir(ckpt_dir, report)

    assert report.has_failures
    assert "mp_rank_00_model_states.pt" in report.results[-1].message


def test_checkpoint_validation_rejects_inconsistent_action_dim(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    config = valid_checkpoint_config()
    config["action_dim"] = 99
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(valid_norm_stats()))
    (ckpt_dir / "mp_rank_00_model_states.pt").write_bytes(b"checkpoint")

    report = preflight.Report()
    preflight.check_checkpoint_dir(ckpt_dir, report)

    assert report.has_failures
    assert "action_dim" in report.results[-1].message


def test_checkpoint_validation_rejects_invalid_norm_stats(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    stats = valid_norm_stats()
    stats["libero"]["action"]["max"] = [1.0, 2.0]
    (ckpt_dir / "config.json").write_text(json.dumps(valid_checkpoint_config()))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps(stats))
    (ckpt_dir / "mp_rank_00_model_states.pt").write_bytes(b"checkpoint")

    report = preflight.Report()
    preflight.check_checkpoint_dir(ckpt_dir, report)

    assert report.has_failures
    assert "same length" in report.results[-1].message


def test_libero_result_validation_accepts_valid_result_file(tmp_path):
    preflight = load_preflight_module()
    result_file = tmp_path / "smoke_results.json"
    result_file.write_text(json.dumps(valid_libero_result_payload()))

    report = preflight.Report()
    preflight.check_libero_result_file(result_file, report)

    assert result_levels(report) == ["OK"]


def test_libero_result_validation_requires_failure_reason_for_failed_episode(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_result_payload()
    payload["episodes"][1]["failure_reason"] = ""
    result_file = tmp_path / "smoke_results.json"
    result_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_result_file(result_file, report)

    assert report.has_failures
    assert "failure_reason" in report.results[-1].message


def test_libero_result_validation_rejects_summary_episode_count_mismatch(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_result_payload()
    payload["summary"]["total_episodes"] = 3
    payload["summary"]["successful_episodes"] = 2
    result_file = tmp_path / "smoke_results.json"
    result_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_result_file(result_file, report)

    assert report.has_failures
    assert "does not match episodes length" in report.results[-1].message


def test_libero_result_validation_rejects_summary_metric_mismatch(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_result_payload()
    payload["summary"]["success_rate"] = 1.0
    result_file = tmp_path / "smoke_results.json"
    result_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_result_file(result_file, report)

    assert report.has_failures
    assert "summary.success_rate" in report.results[-1].message


def test_libero_result_validation_rejects_suite_keys_mismatch(tmp_path):
    preflight = load_preflight_module()
    payload = valid_libero_result_payload()
    payload["summary"]["suites"] = {}
    result_file = tmp_path / "smoke_results.json"
    result_file.write_text(json.dumps(payload))

    report = preflight.Report()
    preflight.check_libero_result_file(result_file, report)

    assert report.has_failures
    assert "summary.suites keys" in report.results[-1].message


def test_run_preflight_accepts_libero_result_directory(tmp_path):
    preflight = load_preflight_module()
    repo_root = Path(__file__).resolve().parents[1]
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "smoke_results.json").write_text(json.dumps(valid_libero_result_payload()))
    args = argparse.Namespace(
        repo_root=str(repo_root),
        checkpoint=None,
        dataset_config="",
        dataset_base_dir="Evo_1",
        strict_data=False,
        check_imports="none",
        libero_result=[str(result_dir)],
        skip_shell_syntax=True,
    )

    report = preflight.run_preflight(args)

    assert not report.has_failures
    assert any(result.name == "libero-result" and result.level == "OK" for result in report.results)


def test_dataset_config_validation_accepts_repo_default_without_strict_data():
    preflight = load_preflight_module()
    repo_root = Path(__file__).resolve().parents[1]
    report = preflight.Report()

    preflight.check_dataset_config(
        repo_root / "Evo_1" / "dataset" / "config.yaml",
        repo_root / "Evo_1",
        strict_data=False,
        report=report,
    )

    assert not report.has_failures


def test_run_preflight_default_has_no_failures():
    preflight = load_preflight_module()
    repo_root = Path(__file__).resolve().parents[1]
    args = argparse.Namespace(
        repo_root=str(repo_root),
        checkpoint=None,
        dataset_config="Evo_1/dataset/config.yaml",
        dataset_base_dir="Evo_1",
        strict_data=False,
        check_imports="none",
        libero_result=[],
        skip_shell_syntax=False,
    )

    report = preflight.run_preflight(args)

    assert not report.has_failures
