from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EpisodeResult:
    task_suite: str
    task_id: int
    episode_id: int
    task_description: str
    success: bool
    decision_steps: int
    control_steps: int
    failure_reason: str = ""
    video_path: str = ""


def summarize_episode_results(results: Sequence[EpisodeResult | Mapping[str, Any]]) -> dict[str, Any]:
    episodes = [_episode_to_dict(result) for result in results]
    successful = [episode for episode in episodes if episode["success"]]

    suite_names = sorted({episode["task_suite"] for episode in episodes})
    suite_summaries = {
        suite_name: _summarize_subset(
            [episode for episode in episodes if episode["task_suite"] == suite_name]
        )
        for suite_name in suite_names
    }

    summary = _summarize_subset(episodes)
    summary["suites"] = suite_summaries
    summary["successful_episode_ids"] = [
        {
            "task_suite": episode["task_suite"],
            "task_id": episode["task_id"],
            "episode_id": episode["episode_id"],
        }
        for episode in successful
    ]
    return summary


def write_result_summary(
    path: str | Path,
    *,
    config: Any,
    results: Sequence[EpisodeResult | Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    result_path = Path(path).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [_episode_to_dict(result) for result in results]
    payload = {
        "config": _serialize_config(config),
        "metadata": dict(metadata) if metadata is not None else build_run_metadata(),
        "summary": summarize_episode_results(episodes),
        "episodes": episodes,
    }
    with result_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return result_path


def build_run_metadata(
    *,
    repo_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    repo_path = Path(repo_root).expanduser().resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    environ = os.environ if environ is None else environ
    argv = sys.argv if argv is None else argv
    created_at_utc = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "created_at_utc": created_at_utc,
        "cwd": str(Path.cwd()),
        "argv": [str(item) for item in argv],
        "command": " ".join(str(item) for item in argv),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git": {
            "repo_root": str(repo_path),
            "commit": _git_output(repo_path, "rev-parse", "HEAD"),
            "branch": _git_output(repo_path, "rev-parse", "--abbrev-ref", "HEAD"),
            "is_dirty": _git_is_dirty(repo_path),
        },
        "environment": _safe_environment(environ),
    }


def _summarize_subset(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_episodes = len(episodes)
    successful_episodes = sum(1 for episode in episodes if episode["success"])
    success_decision_steps = [
        int(episode["decision_steps"]) for episode in episodes if episode["success"]
    ]
    all_decision_steps = [int(episode["decision_steps"]) for episode in episodes]
    all_control_steps = [int(episode["control_steps"]) for episode in episodes]

    return {
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "failed_episodes": total_episodes - successful_episodes,
        "success_rate": successful_episodes / total_episodes if total_episodes else 0.0,
        "average_decision_steps": _mean(all_decision_steps),
        "average_control_steps": _mean(all_control_steps),
        "average_success_decision_steps": _mean(success_decision_steps),
    }


def _mean(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _episode_to_dict(result: EpisodeResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, EpisodeResult):
        payload = asdict(result)
    else:
        payload = dict(result)
    payload["success"] = bool(payload["success"])
    payload["task_id"] = int(payload["task_id"])
    payload["episode_id"] = int(payload["episode_id"])
    payload["decision_steps"] = int(payload["decision_steps"])
    payload["control_steps"] = int(payload["control_steps"])
    payload["failure_reason"] = str(payload.get("failure_reason") or "")
    payload["video_path"] = str(payload.get("video_path") or "")
    return payload


def _serialize_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    return {"repr": repr(config)}


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_is_dirty(repo_root: Path) -> bool | None:
    status = _git_output(repo_root, "status", "--porcelain")
    if status is None:
        return None
    return bool(status)


def _safe_environment(environ: Mapping[str, str]) -> dict[str, str]:
    allowed_exact = {
        "HF_ENDPOINT",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "LIBERO_DATASETS_DIR",
        "LIBERO_ENV_PREFIX",
        "LIBERO_PYTHON",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
    }
    allowed_prefixes = ("EVO1_",)
    blocked_fragments = ("TOKEN", "SECRET", "PASSWORD", "KEY")

    safe_items = {}
    for key, value in environ.items():
        if any(fragment in key.upper() for fragment in blocked_fragments):
            continue
        if key in allowed_exact or any(key.startswith(prefix) for prefix in allowed_prefixes):
            safe_items[key] = str(value)
    return dict(sorted(safe_items.items()))
