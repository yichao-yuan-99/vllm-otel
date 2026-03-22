#!/usr/bin/env python3
"""Compute 1-gram/2-gram token frequency stats for replay plans.

For each replay-plan JSON file under a root directory, this script extracts
all request-level ``forced_token_ids`` sequences, counts unigram and bigram
occurrences, and writes per-plan aggregate stats to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "plan_path",
    "worker_count",
    "request_count",
    "request_with_forced_token_ids",
    "missing_forced_token_ids_count",
    "total_token_count",
    "unigram_unique_count",
    "unigram_total_occurrences",
    "unigram_freq_avg",
    "unigram_freq_min",
    "unigram_freq_max",
    "unigram_freq_std",
    "bigram_unique_count",
    "bigram_total_occurrences",
    "bigram_freq_avg",
    "bigram_freq_min",
    "bigram_freq_max",
    "bigram_freq_std",
    "error",
]


def _base_row(plan_path: Path) -> dict[str, Any]:
    return {
        "plan_path": str(plan_path),
        "worker_count": 0,
        "request_count": 0,
        "request_with_forced_token_ids": 0,
        "missing_forced_token_ids_count": 0,
        "total_token_count": 0,
        "unigram_unique_count": 0,
        "unigram_total_occurrences": 0,
        "unigram_freq_avg": 0.0,
        "unigram_freq_min": 0.0,
        "unigram_freq_max": 0.0,
        "unigram_freq_std": 0.0,
        "bigram_unique_count": 0,
        "bigram_total_occurrences": 0,
        "bigram_freq_avg": 0.0,
        "bigram_freq_min": 0.0,
        "bigram_freq_max": 0.0,
        "bigram_freq_std": 0.0,
        "error": "",
    }


def _render_progress_bar(completed: int, total: int) -> None:
    if total <= 0:
        return
    ratio = float(completed) / float(total)
    ratio = max(0.0, min(1.0, ratio))
    width = 32
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rAnalyzing replay plans [{bar}] {completed}/{total}",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _compute_counter_stats(counter: Counter[Any]) -> dict[str, float | int]:
    if not counter:
        return {
            "unique_count": 0,
            "total_occurrences": 0,
            "freq_avg": 0.0,
            "freq_min": 0.0,
            "freq_max": 0.0,
            "freq_std": 0.0,
        }

    values = list(counter.values())
    unique_count = len(values)
    total_occurrences = sum(values)
    freq_avg = float(total_occurrences) / float(unique_count)
    freq_min = float(min(values))
    freq_max = float(max(values))
    variance = sum((float(value) - freq_avg) ** 2 for value in values) / float(unique_count)
    freq_std = math.sqrt(variance)

    return {
        "unique_count": unique_count,
        "total_occurrences": total_occurrences,
        "freq_avg": freq_avg,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "freq_std": freq_std,
    }


def _is_int_token(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_token_sequence(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    if not all(_is_int_token(item) for item in value):
        return None
    return [int(item) for item in value]


def _extract_forced_token_ids(request: dict[str, Any]) -> list[int] | None:
    direct = _normalize_token_sequence(request.get("forced_token_ids"))
    if direct is not None:
        return direct

    body = request.get("body")
    if not isinstance(body, dict):
        return None
    vllm_xargs = body.get("vllm_xargs")
    if not isinstance(vllm_xargs, dict):
        return None
    return _normalize_token_sequence(vllm_xargs.get("forced_token_ids"))


def _analyze_plan(plan_path_str: str) -> dict[str, Any]:
    plan_path = Path(plan_path_str)
    row = _base_row(plan_path)

    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        row["error"] = f"json_read_error: {exc}"
        return row

    if not isinstance(payload, dict):
        row["error"] = "invalid_payload: root is not an object"
        return row

    workers = payload.get("workers")
    if not isinstance(workers, list):
        row["error"] = "invalid_payload: workers is not a list"
        return row

    row["worker_count"] = len(workers)

    unigram_counter: Counter[int] = Counter()
    bigram_counter: Counter[tuple[int, int]] = Counter()

    for worker in workers:
        if not isinstance(worker, dict):
            continue
        requests = worker.get("requests")
        if not isinstance(requests, list):
            continue

        for request in requests:
            row["request_count"] += 1
            if not isinstance(request, dict):
                row["missing_forced_token_ids_count"] += 1
                continue

            forced_token_ids = _extract_forced_token_ids(request)
            if forced_token_ids is None:
                row["missing_forced_token_ids_count"] += 1
                continue

            row["request_with_forced_token_ids"] += 1
            row["total_token_count"] += len(forced_token_ids)

            for token in forced_token_ids:
                unigram_counter[token] += 1

            for first, second in zip(forced_token_ids, forced_token_ids[1:]):
                bigram_counter[(first, second)] += 1

    unigram_stats = _compute_counter_stats(unigram_counter)
    bigram_stats = _compute_counter_stats(bigram_counter)

    row["unigram_unique_count"] = unigram_stats["unique_count"]
    row["unigram_total_occurrences"] = unigram_stats["total_occurrences"]
    row["unigram_freq_avg"] = unigram_stats["freq_avg"]
    row["unigram_freq_min"] = unigram_stats["freq_min"]
    row["unigram_freq_max"] = unigram_stats["freq_max"]
    row["unigram_freq_std"] = unigram_stats["freq_std"]

    row["bigram_unique_count"] = bigram_stats["unique_count"]
    row["bigram_total_occurrences"] = bigram_stats["total_occurrences"]
    row["bigram_freq_avg"] = bigram_stats["freq_avg"]
    row["bigram_freq_min"] = bigram_stats["freq_min"]
    row["bigram_freq_max"] = bigram_stats["freq_max"]
    row["bigram_freq_std"] = bigram_stats["freq_std"]

    return row


def _iter_replay_plan_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("replay-plan*.json"):
        if path.is_file() and path.name.startswith("replay-plan"):
            paths.append(path.resolve())
    paths.sort()
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compute_replay_plan_gram_freq.py",
        description=(
            "Recursively scan a root directory for replay-plan*.json files and "
            "compute per-file unigram/bigram frequency stats from forced_token_ids."
        ),
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to recursively scan for replay plans.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Default: "
            "<analysis/gram-freq>/output/replay-plan-gram-freq-summary.csv"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker process count for per-file analysis (default: CPU count).",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute plan paths in CSV (default: paths relative to --root).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the interactive progress bar.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        output_path = (script_dir / "output" / "replay-plan-gram-freq-summary.csv").resolve()

    if not root.is_dir():
        print(f"error: --root is not a directory: {root}", file=sys.stderr)
        return 2

    jobs = max(1, int(args.jobs))
    plan_paths = _iter_replay_plan_paths(root)

    rows: list[dict[str, Any]] = []
    if plan_paths:
        show_progress = sys.stderr.isatty() and not args.no_progress
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_path = {
                executor.submit(_analyze_plan, str(path)): path for path in plan_paths
            }
            completed = 0
            if show_progress:
                _render_progress_bar(completed, len(plan_paths))
            for future in as_completed(future_to_path):
                source_path = future_to_path[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    error_row = _base_row(source_path)
                    error_row["error"] = f"worker_error: {exc}"
                    rows.append(error_row)
                completed += 1
                if show_progress:
                    _render_progress_bar(completed, len(plan_paths))
        if show_progress:
            print(file=sys.stderr)

    for row in rows:
        raw_path = Path(row["plan_path"])
        if args.absolute_paths:
            row["plan_path"] = str(raw_path)
            continue
        try:
            row["plan_path"] = str(raw_path.relative_to(root))
        except ValueError:
            row["plan_path"] = str(raw_path)

    rows.sort(key=lambda row: str(row["plan_path"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    errored = sum(1 for row in rows if str(row.get("error", "")).strip())
    print(
        json.dumps(
            {
                "status": "ok",
                "root": str(root),
                "output_csv": str(output_path),
                "plan_count": len(plan_paths),
                "errored_plan_count": errored,
                "jobs": jobs,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
