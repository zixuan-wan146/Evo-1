#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


TABLE_COLUMNS = (
    "result_file",
    "run_name",
    "scope",
    "total_episodes",
    "successful_episodes",
    "failed_episodes",
    "success_rate",
    "average_decision_steps",
    "average_control_steps",
    "average_success_decision_steps",
)


def discover_result_files(inputs: Iterable[str]) -> list[Path]:
    discovered: set[Path] = set()
    for raw_input in inputs:
        matches = glob.glob(raw_input, recursive=True)
        candidate_paths = matches if matches else [raw_input]
        for candidate in candidate_paths:
            path = Path(candidate).expanduser()
            if path.is_dir():
                discovered.update(path.rglob("*_results.json"))
            elif path.is_file():
                discovered.add(path)
            else:
                raise FileNotFoundError(f"LIBERO result path not found: {raw_input}")
    return sorted(path.resolve() for path in discovered)


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        payload = json.load(f)

    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")

    config = payload.get("config", {})
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"{path} has no summary object")

    run_name = _run_name(path, config)
    rows = [_summary_row(path, run_name, "overall", summary)]

    suites = summary.get("suites", {})
    if suites is not None and not isinstance(suites, Mapping):
        raise ValueError(f"{path} summary.suites must be an object")
    for suite_name, suite_summary in sorted((suites or {}).items()):
        if not isinstance(suite_summary, Mapping):
            raise ValueError(f"{path} suite summary {suite_name!r} must be an object")
        rows.append(_summary_row(path, run_name, f"suite:{suite_name}", suite_summary))

    return rows


def collect_result_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_result_rows(path))
    return rows


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in TABLE_COLUMNS})
    return path


def write_markdown(rows: list[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_markdown_table(rows))
    return path


def format_markdown_table(rows: list[Mapping[str, Any]]) -> str:
    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |"
    body = [
        "| " + " | ".join(_markdown_cell(row.get(column, "")) for column in TABLE_COLUMNS) + " |"
        for row in rows
    ]
    return "\n".join([header, separator] + body) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one or more LIBERO *_results.json files.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Result JSON files, directories, or glob patterns such as LIBERO_evaluation/log_file/*_results.json.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output table format.",
    )
    parser.add_argument("--output", help="Optional output path. Defaults to stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = discover_result_files(args.inputs)
        rows = collect_result_rows(paths)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output).expanduser()
        if args.format == "csv":
            write_csv(rows, output_path)
        else:
            write_markdown(rows, output_path)
        print(f"Wrote {len(rows)} row(s) from {len(paths)} result file(s) to {output_path}")
    else:
        if args.format == "csv":
            writer = csv.DictWriter(_StdoutWriter(), fieldnames=TABLE_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in TABLE_COLUMNS})
        else:
            print(format_markdown_table(rows), end="")

    return 0


def _summary_row(path: Path, run_name: str, scope: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "result_file": str(path),
        "run_name": run_name,
        "scope": scope,
        "total_episodes": int(summary.get("total_episodes", 0)),
        "successful_episodes": int(summary.get("successful_episodes", 0)),
        "failed_episodes": int(summary.get("failed_episodes", 0)),
        "success_rate": _float(summary.get("success_rate", 0.0)),
        "average_decision_steps": _float(summary.get("average_decision_steps", 0.0)),
        "average_control_steps": _float(summary.get("average_control_steps", 0.0)),
        "average_success_decision_steps": _float(summary.get("average_success_decision_steps", 0.0)),
    }


def _run_name(path: Path, config: Any) -> str:
    if isinstance(config, Mapping):
        for key in ("ckpt_name", "run_name", "checkpoint", "ckpt_dir"):
            value = config.get(key)
            if value:
                return str(value)
    return path.stem.replace("_results", "")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        rendered = f"{value:.4f}"
    else:
        rendered = str(value)
    return rendered.replace("|", "\\|").replace("\n", " ")


class _StdoutWriter:
    def write(self, value: str) -> int:
        print(value, end="")
        return len(value)


if __name__ == "__main__":
    raise SystemExit(main())
