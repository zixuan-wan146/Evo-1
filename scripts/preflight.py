#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


REQUIRED_REPO_FILES = (
    "README.md",
    "Evo_1/scripts/Evo1_server.py",
    "Evo_1/scripts/train.py",
    "Evo_1/dataset/config.yaml",
    "LIBERO_evaluation/libero_client_config.py",
    "LIBERO_evaluation/libero_client_4tasks.py",
    "scripts/preflight.py",
    "scripts/setup_libero_env.sh",
    "scripts/start_evo1_server.sh",
    "scripts/run_libero_smoke.sh",
)

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "norm_stats.json",
    "mp_rank_00_model_states.pt",
)

EVO1_RUNTIME_IMPORTS = (
    "torch",
    "websockets",
    "cv2",
    "PIL",
    "torchvision",
)

LIBERO_RUNTIME_IMPORTS = (
    "libero",
    "robosuite",
    "mujoco",
    "websockets",
    "imageio",
)


@dataclass(frozen=True)
class CheckResult:
    level: str
    name: str
    message: str


class Report:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def ok(self, name: str, message: str) -> None:
        self.results.append(CheckResult("OK", name, message))

    def warn(self, name: str, message: str) -> None:
        self.results.append(CheckResult("WARN", name, message))

    def fail(self, name: str, message: str) -> None:
        self.results.append(CheckResult("FAIL", name, message))

    @property
    def has_failures(self) -> bool:
        return any(result.level == "FAIL" for result in self.results)

    def print(self) -> None:
        for result in self.results:
            print(f"[{result.level}] {result.name}: {result.message}")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def check_repo_layout(repo_root: Path, report: Report) -> None:
    if not repo_root.exists():
        report.fail("repo", f"repository root does not exist: {repo_root}")
        return

    missing = [path for path in REQUIRED_REPO_FILES if not (repo_root / path).exists()]
    if missing:
        report.fail("repo", f"missing required files: {', '.join(missing)}")
    else:
        report.ok("repo", f"required files present under {repo_root}")

    git_dir = repo_root / ".git"
    if git_dir.exists():
        report.ok("git", ".git directory present")
    else:
        report.warn("git", "repository root has no .git directory")


def check_script_permissions(repo_root: Path, report: Report) -> None:
    scripts_dir = repo_root / "scripts"
    scripts = sorted(scripts_dir.glob("*.sh"))
    if not scripts:
        report.fail("scripts", f"no shell scripts found under {scripts_dir}")
        return

    non_executable = [str(path.relative_to(repo_root)) for path in scripts if not os.access(path, os.X_OK)]
    if non_executable:
        report.fail("scripts", f"not executable: {', '.join(non_executable)}")
    else:
        report.ok("scripts", f"{len(scripts)} shell scripts are executable")


def check_shell_syntax(repo_root: Path, report: Report) -> None:
    scripts = sorted((repo_root / "scripts").glob("*.sh"))
    if not scripts:
        return

    bash = importlib.util.find_spec("subprocess")
    if bash is None:
        report.warn("shell", "subprocess module is unavailable; skipped bash -n")
        return

    for script in scripts:
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        rel = script.relative_to(repo_root)
        if result.returncode == 0:
            report.ok("shell", f"{rel} syntax OK")
        else:
            stderr = result.stderr.strip() or result.stdout.strip()
            report.fail("shell", f"{rel} syntax error: {stderr}")


def check_imports(import_names: tuple[str, ...], report: Report, group_name: str) -> None:
    missing = [name for name in import_names if importlib.util.find_spec(name) is None]
    if missing:
        report.fail(group_name, f"missing Python packages: {', '.join(missing)}")
    else:
        report.ok(group_name, f"required Python packages importable: {', '.join(import_names)}")


def check_checkpoint_dir(ckpt_dir: Path, report: Report) -> None:
    if not ckpt_dir.exists():
        report.fail("checkpoint", f"directory does not exist: {ckpt_dir}")
        return
    if not ckpt_dir.is_dir():
        report.fail("checkpoint", f"path is not a directory: {ckpt_dir}")
        return

    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (ckpt_dir / name).exists()]
    if missing:
        report.fail("checkpoint", f"missing required files in {ckpt_dir}: {', '.join(missing)}")
        return

    for json_name in ("config.json", "norm_stats.json"):
        path = ckpt_dir / json_name
        try:
            with path.open("r") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            report.fail("checkpoint", f"{json_name} is not valid JSON: {exc}")
            return
        if not isinstance(payload, dict):
            report.fail("checkpoint", f"{json_name} must contain a JSON object")
            return

    weight_path = ckpt_dir / "mp_rank_00_model_states.pt"
    if weight_path.stat().st_size == 0:
        report.fail("checkpoint", f"checkpoint weight file is empty: {weight_path}")
        return

    report.ok("checkpoint", f"required checkpoint files are present in {ckpt_dir}")


def load_yaml_if_available(path: Path, report: Report) -> dict[str, Any] | None:
    spec = importlib.util.find_spec("yaml")
    if spec is None:
        report.warn("dataset", "PyYAML is not installed; skipped structured dataset config validation")
        return None

    import yaml  # type: ignore[import-not-found]

    try:
        with path.open("r") as f:
            loaded = yaml.safe_load(f)
    except Exception as exc:
        report.fail("dataset", f"failed to parse dataset config {path}: {exc}")
        return None

    if not isinstance(loaded, dict):
        report.fail("dataset", f"dataset config must be a mapping: {path}")
        return None
    return loaded


def resolve_dataset_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def check_dataset_config(config_path: Path, base_dir: Path, strict_data: bool, report: Report) -> None:
    if not config_path.exists():
        report.fail("dataset", f"dataset config does not exist: {config_path}")
        return

    config = load_yaml_if_available(config_path, report)
    if config is None:
        return

    for key in ("max_action_dim", "max_state_dim", "max_views"):
        value = config.get(key)
        if not isinstance(value, int) or value <= 0:
            report.fail("dataset", f"{key} must be a positive integer")
            return

    data_groups = config.get("data_groups")
    if not isinstance(data_groups, dict) or not data_groups:
        report.fail("dataset", "data_groups must be a non-empty mapping")
        return

    dataset_count = 0
    missing_paths: list[str] = []
    missing_required_data: list[str] = []
    for group_name, group_config in data_groups.items():
        if not isinstance(group_config, dict) or not group_config:
            report.fail("dataset", f"data group {group_name!r} must contain datasets")
            return
        for dataset_name, dataset_config in group_config.items():
            dataset_count += 1
            if not isinstance(dataset_config, dict):
                report.fail("dataset", f"dataset {group_name}/{dataset_name} must be a mapping")
                return
            raw_path = dataset_config.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                report.fail("dataset", f"dataset {group_name}/{dataset_name} has no path")
                return
            dataset_path = resolve_dataset_path(raw_path, base_dir)
            if not dataset_path.exists():
                missing_paths.append(f"{group_name}/{dataset_name}: {dataset_path}")
                continue
            if strict_data:
                required_paths = (
                    dataset_path / "meta" / "tasks.jsonl",
                    dataset_path / "meta" / "episodes.jsonl",
                )
                for required_path in required_paths:
                    if not required_path.exists():
                        missing_required_data.append(str(required_path))
                has_stats = (dataset_path / "meta" / "stats.json").exists() or (
                    dataset_path / "meta" / "episodes_stats.jsonl"
                ).exists()
                if not has_stats:
                    missing_required_data.append(str(dataset_path / "meta" / "stats.json"))
                if not list(dataset_path.glob("data/*/*.parquet")):
                    missing_required_data.append(str(dataset_path / "data/*/*.parquet"))

    if missing_paths:
        message = "configured dataset paths do not exist: " + "; ".join(missing_paths)
        if strict_data:
            report.fail("dataset", message)
        else:
            report.warn("dataset", message)
    elif missing_required_data:
        report.fail("dataset", "missing required dataset files: " + "; ".join(missing_required_data))
    else:
        report.ok("dataset", f"dataset config describes {dataset_count} dataset(s)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight Evo-1 repository preflight checks.")
    parser.add_argument("--repo-root", default=str(repo_root_from_script()), help="Repository root to check.")
    parser.add_argument("--checkpoint", help="Optional Evo-1 checkpoint directory to validate.")
    parser.add_argument(
        "--dataset-config",
        default="Evo_1/dataset/config.yaml",
        help="Dataset config to validate. Set to empty string to skip.",
    )
    parser.add_argument(
        "--dataset-base-dir",
        default="Evo_1",
        help="Base directory for relative dataset paths in the dataset config.",
    )
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Fail when configured dataset paths or required dataset files are missing.",
    )
    parser.add_argument(
        "--check-imports",
        choices=("none", "evo1", "libero", "all"),
        default="none",
        help="Optionally require runtime packages to be importable.",
    )
    parser.add_argument("--skip-shell-syntax", action="store_true", help="Skip bash -n checks for scripts/*.sh.")
    return parser.parse_args(argv)


def run_preflight(args: argparse.Namespace) -> Report:
    report = Report()
    repo_root = Path(args.repo_root).expanduser().resolve()

    check_repo_layout(repo_root, report)
    check_script_permissions(repo_root, report)
    if not args.skip_shell_syntax:
        check_shell_syntax(repo_root, report)

    if args.dataset_config:
        dataset_config = resolve_repo_path(repo_root, args.dataset_config)
        dataset_base_dir = resolve_repo_path(repo_root, args.dataset_base_dir)
        check_dataset_config(dataset_config, dataset_base_dir, bool(args.strict_data), report)

    if args.checkpoint:
        check_checkpoint_dir(Path(args.checkpoint).expanduser().resolve(), report)

    if args.check_imports in ("evo1", "all"):
        check_imports(EVO1_RUNTIME_IMPORTS, report, "evo1-imports")
    if args.check_imports in ("libero", "all"):
        check_imports(LIBERO_RUNTIME_IMPORTS, report, "libero-imports")

    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_preflight(args)
    report.print()
    return 1 if report.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
