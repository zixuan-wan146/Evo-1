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


def test_checkpoint_validation_accepts_required_files(tmp_path):
    preflight = load_preflight_module()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(json.dumps({"horizon": 14}))
    (ckpt_dir / "norm_stats.json").write_text(json.dumps({"state": {}}))
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
        skip_shell_syntax=False,
    )

    report = preflight.run_preflight(args)

    assert not report.has_failures
