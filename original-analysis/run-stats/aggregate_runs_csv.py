from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_NAME = "run-stats-summary.json"
DEFAULT_OUTPUT_NAME = "run-stats-summary.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-run run-stats summaries into one CSV for a root directory."
        )
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory containing many run result directories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Optional output CSV path. Default: <root-dir>/{DEFAULT_OUTPUT_NAME}",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        signed = stripped[1:] if stripped[0] in {"+", "-"} else stripped
        if signed.isdigit():
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def discover_run_stats_summary_paths(root_dir: Path) -> list[Path]:
    summary_paths: set[Path] = set()
    for summary_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name != "run-stats":
            continue
        summary_paths.add(summary_path.resolve())
    return sorted(summary_paths)


def _run_dir_from_summary_path(summary_path: Path) -> Path:
    # <run-dir>/run-stats/run-stats-summary.json
    return summary_path.parent.parent


def _path_sort_key(relative_path: str) -> tuple[tuple[Any, ...], ...]:
    key_parts: list[tuple[Any, ...]] = []
    for part in relative_path.split("/"):
        if part.isdigit():
            key_parts.append((0, int(part), part))
        else:
            key_parts.append((1, part))
    return tuple(key_parts)


def build_rows(root_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in discover_run_stats_summary_paths(root_dir):
        payload = _load_json(summary_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Summary payload must be a JSON object: {summary_path}")

        run_dir = _run_dir_from_summary_path(summary_path)
        relative_path = run_dir.relative_to(root_dir).as_posix()
        jobs_payload = payload.get("jobs")
        jobs = jobs_payload if isinstance(jobs_payload, list) else []

        job_count = _int_or_none(payload.get("job_count"))
        if job_count is None and jobs:
            job_count = len([entry for entry in jobs if isinstance(entry, dict)])

        max_job_max_request_length = _float_or_none(
            payload.get("max_job_max_request_length")
        )
        if max_job_max_request_length is None:
            raw_lengths = payload.get("job_max_request_lengths")
            if isinstance(raw_lengths, list):
                numeric_lengths = [value for value in (_int_or_none(raw) for raw in raw_lengths) if value is not None]
                if numeric_lengths:
                    max_job_max_request_length = float(max(numeric_lengths))

        avg_turns_per_run = _float_or_none(payload.get("avg_turns_per_run"))
        if avg_turns_per_run is None and jobs:
            request_counts = [
                request_count
                for request_count in (
                    _int_or_none(entry.get("request_count"))
                    for entry in jobs
                    if isinstance(entry, dict)
                )
                if request_count is not None
            ]
            if request_counts:
                avg_turns_per_run = sum(request_counts) / len(request_counts)
        max_turns_per_run = _float_or_none(payload.get("max_turns_per_run"))
        if max_turns_per_run is None and jobs:
            request_counts = [
                request_count
                for request_count in (
                    _int_or_none(entry.get("request_count"))
                    for entry in jobs
                    if isinstance(entry, dict)
                )
                if request_count is not None
            ]
            if request_counts:
                max_turns_per_run = float(max(request_counts))

        agent_type = _string_or_none(payload.get("agent_type"))
        if agent_type is None:
            # Expected run path shape: <dataset>/<agent_type>/<run-id>.
            candidate = run_dir.parent.name.strip()
            if candidate:
                agent_type = candidate

        rows.append(
            {
                "run_path": relative_path,
                "agent_type": agent_type,
                "dataset": payload.get("dataset"),
                "score": _float_or_none(payload.get("score")),
                "job_count": job_count,
                "avg_job_max_request_length": _float_or_none(
                    payload.get("avg_job_max_request_length")
                ),
                "max_job_max_request_length": max_job_max_request_length,
                "avg_turns_per_run": avg_turns_per_run,
                "max_turns_per_run": max_turns_per_run,
                "avg_run_prompt_tokens_per_request": _float_or_none(
                    payload.get("avg_run_prompt_tokens_per_request")
                ),
                "avg_run_generation_tokens_per_request": _float_or_none(
                    payload.get("avg_run_generation_tokens_per_request")
                ),
            }
        )

    rows.sort(key=lambda item: _path_sort_key(item["run_path"]))
    return rows


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    rows = build_rows(root_dir)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (root_dir / DEFAULT_OUTPUT_NAME).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_path",
        "agent_type",
        "dataset",
        "score",
        "job_count",
        "avg_job_max_request_length",
        "max_job_max_request_length",
        "avg_turns_per_run",
        "max_turns_per_run",
        "avg_run_prompt_tokens_per_request",
        "avg_run_generation_tokens_per_request",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
