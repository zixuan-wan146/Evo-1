#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from Evo_1.dataset.config_utils import (  # noqa: E402
    iter_dataset_entries,
    resolve_dataset_config_paths,
    validate_dataset_config_structure,
)
from Evo_1.dataset.validation import validate_configured_datasets  # noqa: E402
from Evo_1.runtime_config import TARGET_STATE_DIM  # noqa: E402


REQUIRED_REPO_FILES = (
    "README.md",
    "requirements-policy.json",
    "requirements-libero.txt",
    "Evo_1/training_config.py",
    "Evo_1/server_protocol.py",
    "Evo_1/scripts/Evo1_server.py",
    "Evo_1/scripts/train.py",
    "Evo_1/dataset/config.yaml",
    "Evo_1/dataset/config_utils.py",
    "Evo_1/dataset/validation.py",
    "LIBERO_evaluation/libero_action_protocol.py",
    "LIBERO_evaluation/libero_client_config.py",
    "LIBERO_evaluation/libero_eval_summary.py",
    "LIBERO_evaluation/libero_client_4tasks.py",
    "scripts/preflight.py",
    "scripts/audit_requirements.py",
    "scripts/check_repo.sh",
    "scripts/validate_training_dataset.py",
    "scripts/setup_libero_env.sh",
    "scripts/start_evo1_server.sh",
    "scripts/export_unpushed_commits.sh",
    "scripts/run_libero_smoke.sh",
    "scripts/run_libero_eval.sh",
    "scripts/summarize_libero_results.py",
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

LIBERO_SUMMARY_COUNT_FIELDS = (
    "total_episodes",
    "successful_episodes",
    "failed_episodes",
)

LIBERO_SUMMARY_FLOAT_FIELDS = (
    "success_rate",
    "average_decision_steps",
    "average_control_steps",
    "average_success_decision_steps",
)

LIBERO_EPISODE_REQUIRED_FIELDS = (
    "task_suite",
    "task_id",
    "episode_id",
    "task_description",
    "success",
    "decision_steps",
    "control_steps",
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

    payloads = {}
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
        payloads[json_name] = payload

    config_error = validate_checkpoint_config(payloads["config.json"])
    if config_error:
        report.fail("checkpoint", f"config.json: {config_error}")
        return

    stats_error = validate_norm_stats(payloads["norm_stats.json"])
    if stats_error:
        report.fail("checkpoint", f"norm_stats.json: {stats_error}")
        return

    weight_path = ckpt_dir / "mp_rank_00_model_states.pt"
    if weight_path.stat().st_size == 0:
        report.fail("checkpoint", f"checkpoint weight file is empty: {weight_path}")
        return

    report.ok("checkpoint", f"required checkpoint files are present in {ckpt_dir}")


def validate_checkpoint_config(config: dict[str, Any], target_dim: int = TARGET_STATE_DIM) -> str | None:
    positive_ints = {}
    for key in ("horizon", "per_action_dim", "state_dim"):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return f"{key} must be a positive integer when present"
        positive_ints[key] = value

    per_action_dim = positive_ints.get("per_action_dim")
    if per_action_dim is not None and per_action_dim > target_dim:
        return f"per_action_dim {per_action_dim} exceeds server target dimension {target_dim}"

    state_dim = positive_ints.get("state_dim")
    if state_dim is not None and state_dim > target_dim:
        return f"state_dim {state_dim} exceeds server target dimension {target_dim}"

    action_dim = config.get("action_dim")
    if action_dim is not None:
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim <= 0:
            return "action_dim must be a positive integer when present"
        horizon = positive_ints.get("horizon")
        if horizon is not None and per_action_dim is not None and action_dim != horizon * per_action_dim:
            return (
                f"action_dim {action_dim} must equal horizon {horizon} "
                f"* per_action_dim {per_action_dim}"
            )

    return None


def validate_norm_stats(stats: dict[str, Any], target_dim: int = TARGET_STATE_DIM) -> str | None:
    if len(stats) != 1:
        return f"expected one robot stats entry, got {len(stats)}"

    robot_name, robot_stats = next(iter(stats.items()))
    if not isinstance(robot_stats, dict):
        return f"{robot_name} stats must be an object"

    for stat_name in ("observation.state", "action"):
        stat_error = _validate_minmax_stat(robot_stats, stat_name, target_dim)
        if stat_error:
            return f"{robot_name}.{stat_error}"

    return None


def _validate_minmax_stat(robot_stats: dict[str, Any], stat_name: str, target_dim: int) -> str | None:
    stat = robot_stats.get(stat_name)
    if not isinstance(stat, dict):
        return f"{stat_name} must be an object with min/max"

    mins = stat.get("min")
    maxs = stat.get("max")
    min_error = _validate_numeric_vector(mins, f"{stat_name}.min", target_dim)
    if min_error:
        return min_error
    max_error = _validate_numeric_vector(maxs, f"{stat_name}.max", target_dim)
    if max_error:
        return max_error
    if len(mins) != len(maxs):
        return f"{stat_name}.min and max must have the same length"
    for index, (min_value, max_value) in enumerate(zip(mins, maxs)):
        if float(min_value) > float(max_value):
            return f"{stat_name}.min[{index}] must be <= max[{index}]"
    return None


def _validate_numeric_vector(value: Any, label: str, target_dim: int) -> str | None:
    if not isinstance(value, list) or not value:
        return f"{label} must be a non-empty list"
    if len(value) > target_dim:
        return f"{label} length {len(value)} exceeds server target dimension {target_dim}"
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return f"{label}[{index}] must be numeric"
        if not math.isfinite(float(item)):
            return f"{label}[{index}] must be finite"
    return None


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


def check_dataset_config(config_path: Path, base_dir: Path, strict_data: bool, report: Report) -> None:
    if not config_path.exists():
        report.fail("dataset", f"dataset config does not exist: {config_path}")
        return

    config = load_yaml_if_available(config_path, report)
    if config is None:
        return

    try:
        dataset_count = validate_dataset_config_structure(config)
        resolved_config = resolve_dataset_config_paths(config, base_dir)
    except (TypeError, ValueError) as exc:
        report.fail("dataset", str(exc))
        return

    if strict_data:
        issues = validate_configured_datasets(config, base_dir, require_videos=True)
        if issues:
            for issue in issues:
                if issue.level == "FAIL":
                    report.fail("dataset", f"{issue.path}: {issue.message}")
                else:
                    report.warn("dataset", f"{issue.path}: {issue.message}")
            return
        report.ok("dataset", f"dataset config describes {dataset_count} dataset(s) with valid training data")
        return

    missing_paths: list[str] = []
    for group_name, dataset_name, dataset_config in iter_dataset_entries(resolved_config):
        dataset_path = Path(str(dataset_config["path"]))
        if not dataset_path.exists():
            missing_paths.append(f"{group_name}/{dataset_name}: {dataset_path}")
            continue

    if missing_paths:
        message = "configured dataset paths do not exist: " + "; ".join(missing_paths)
        report.warn("dataset", message)
    else:
        report.ok("dataset", f"dataset config describes {dataset_count} dataset(s)")


def resolve_libero_result_paths(raw_inputs: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for raw_input in raw_inputs:
        matches = glob.glob(raw_input, recursive=True)
        candidate_paths = matches if matches else [raw_input]
        for candidate in candidate_paths:
            path = Path(candidate).expanduser()
            if path.is_dir():
                paths.update(path.rglob("*_results.json"))
            elif path.is_file():
                paths.add(path)
            else:
                raise FileNotFoundError(f"LIBERO result path not found: {raw_input}")
    return sorted(path.resolve() for path in paths)


def check_libero_result_file(result_path: Path, report: Report) -> None:
    if not result_path.exists():
        report.fail("libero-result", f"result file does not exist: {result_path}")
        return
    if not result_path.is_file():
        report.fail("libero-result", f"result path is not a file: {result_path}")
        return

    try:
        with result_path.open("r") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        report.fail("libero-result", f"{result_path} is not valid JSON: {exc}")
        return

    if not isinstance(payload, dict):
        report.fail("libero-result", f"{result_path} must contain a JSON object")
        return

    config = payload.get("config")
    if config is not None and not isinstance(config, dict):
        report.fail("libero-result", f"{result_path} config must be an object when present")
        return

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        report.fail("libero-result", f"{result_path} metadata must be an object when present")
        return

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        report.fail("libero-result", f"{result_path} has no summary object")
        return
    summary_error = _validate_libero_summary(summary, "summary")
    if summary_error:
        report.fail("libero-result", f"{result_path}: {summary_error}")
        return

    suites = summary.get("suites", {})
    if suites is not None and not isinstance(suites, dict):
        report.fail("libero-result", f"{result_path} summary.suites must be an object")
        return
    for suite_name, suite_summary in (suites or {}).items():
        if not isinstance(suite_summary, dict):
            report.fail("libero-result", f"{result_path} suite {suite_name!r} summary must be an object")
            return
        suite_error = _validate_libero_summary(suite_summary, f"summary.suites.{suite_name}")
        if suite_error:
            report.fail("libero-result", f"{result_path}: {suite_error}")
            return

    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        report.fail("libero-result", f"{result_path} episodes must be a list")
        return
    for index, episode in enumerate(episodes):
        episode_error = _validate_libero_episode(episode, index)
        if episode_error:
            report.fail("libero-result", f"{result_path}: {episode_error}")
            return

    summary_total = int(summary["total_episodes"])
    if summary_total != len(episodes):
        report.fail(
            "libero-result",
            f"{result_path} summary total_episodes={summary_total} does not match episodes length={len(episodes)}",
        )
        return
    consistency_error = _validate_libero_summary_matches_episodes(summary, episodes, "summary")
    if consistency_error:
        report.fail("libero-result", f"{result_path}: {consistency_error}")
        return

    episode_suite_names = {episode["task_suite"] for episode in episodes}
    summary_suite_names = set((suites or {}).keys())
    if episode_suite_names != summary_suite_names:
        report.fail(
            "libero-result",
            f"{result_path} summary.suites keys {sorted(summary_suite_names)} "
            f"do not match episode task suites {sorted(episode_suite_names)}",
        )
        return
    for suite_name in sorted(episode_suite_names):
        suite_episodes = [episode for episode in episodes if episode["task_suite"] == suite_name]
        suite_consistency_error = _validate_libero_summary_matches_episodes(
            suites[suite_name],
            suite_episodes,
            f"summary.suites.{suite_name}",
        )
        if suite_consistency_error:
            report.fail("libero-result", f"{result_path}: {suite_consistency_error}")
            return

    report.ok("libero-result", f"{result_path} describes {len(episodes)} episode(s)")


def _validate_libero_summary(summary: dict[str, Any], label: str) -> str | None:
    required_fields = (*LIBERO_SUMMARY_COUNT_FIELDS, *LIBERO_SUMMARY_FLOAT_FIELDS)
    missing = [field for field in required_fields if field not in summary]
    if missing:
        return f"{label} missing fields: {', '.join(missing)}"
    for field in LIBERO_SUMMARY_COUNT_FIELDS:
        value = summary[field]
        if not isinstance(value, int) or isinstance(value, bool):
            return f"{label}.{field} must be an integer"
        if value < 0:
            return f"{label}.{field} must be non-negative"
    for field in LIBERO_SUMMARY_FLOAT_FIELDS:
        value = summary[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{label}.{field} must be numeric"
        if value < 0:
            return f"{label}.{field} must be non-negative"
    if not 0 <= float(summary["success_rate"]) <= 1:
        return f"{label}.success_rate must be between 0 and 1"
    total = int(summary["total_episodes"])
    successful = int(summary["successful_episodes"])
    failed = int(summary["failed_episodes"])
    if successful + failed != total:
        return f"{label} successful_episodes + failed_episodes must equal total_episodes"
    return None


def _validate_libero_episode(episode: Any, index: int) -> str | None:
    if not isinstance(episode, dict):
        return f"episodes[{index}] must be an object"
    missing = [field for field in LIBERO_EPISODE_REQUIRED_FIELDS if field not in episode]
    if missing:
        return f"episodes[{index}] missing fields: {', '.join(missing)}"
    if not isinstance(episode["task_suite"], str) or not episode["task_suite"]:
        return f"episodes[{index}].task_suite must be a non-empty string"
    if not isinstance(episode["task_description"], str):
        return f"episodes[{index}].task_description must be a string"
    if not isinstance(episode["success"], bool):
        return f"episodes[{index}].success must be a boolean"
    for field in ("task_id", "episode_id", "decision_steps", "control_steps"):
        value = episode[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return f"episodes[{index}].{field} must be a non-negative integer"
    if not episode["success"] and not str(episode.get("failure_reason", "")).strip():
        return f"episodes[{index}].failure_reason is required for failed episodes"
    return None


def _validate_libero_summary_matches_episodes(
    summary: dict[str, Any],
    episodes: list[dict[str, Any]],
    label: str,
) -> str | None:
    expected = _compute_episode_summary(episodes)
    for field in LIBERO_SUMMARY_COUNT_FIELDS:
        if int(summary[field]) != expected[field]:
            return f"{label}.{field}={summary[field]} does not match episode-derived value {expected[field]}"
    for field in LIBERO_SUMMARY_FLOAT_FIELDS:
        if not math.isclose(float(summary[field]), expected[field], rel_tol=1e-9, abs_tol=1e-9):
            return f"{label}.{field}={summary[field]} does not match episode-derived value {expected[field]}"
    return None


def _compute_episode_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(episodes)
    successful = sum(1 for episode in episodes if episode["success"])
    failed = total - successful
    decision_steps = [int(episode["decision_steps"]) for episode in episodes]
    control_steps = [int(episode["control_steps"]) for episode in episodes]
    success_decision_steps = [
        int(episode["decision_steps"]) for episode in episodes if episode["success"]
    ]
    return {
        "total_episodes": total,
        "successful_episodes": successful,
        "failed_episodes": failed,
        "success_rate": successful / total if total else 0.0,
        "average_decision_steps": _mean(decision_steps),
        "average_control_steps": _mean(control_steps),
        "average_success_decision_steps": _mean(success_decision_steps),
    }


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


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
    parser.add_argument(
        "--libero-result",
        action="append",
        default=[],
        help="Optional LIBERO result JSON file, directory, or glob to validate. Can be passed multiple times.",
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

    libero_result_inputs = getattr(args, "libero_result", [])
    if libero_result_inputs:
        try:
            libero_result_paths = resolve_libero_result_paths(libero_result_inputs)
        except FileNotFoundError as exc:
            report.fail("libero-result", str(exc))
        else:
            if not libero_result_paths:
                report.fail("libero-result", "no LIBERO result files matched")
            for result_path in libero_result_paths:
                check_libero_result_file(result_path, report)

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
