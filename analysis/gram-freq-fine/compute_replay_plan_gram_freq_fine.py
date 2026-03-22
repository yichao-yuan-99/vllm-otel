#!/usr/bin/env python3
"""Compute fine-grained windowed n-gram frequency stats for replay plans.

For each replay-plan JSON file found under a root directory, this script computes
windowed n-gram frequency statistics for every trace (worker) and every step
(request index). Window bounds are configured by --window-mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WINDOW_MODE_CENTERED = "centered"
WINDOW_MODE_TRAILING = "trailing"
WINDOW_MODE_CUMULATIVE = "cumulative"
WINDOW_MODE_CHOICES = (
    WINDOW_MODE_CENTERED,
    WINDOW_MODE_TRAILING,
    WINDOW_MODE_CUMULATIVE,
)


def _utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_n_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid n value: {item}") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"n must be positive: {parsed}")
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("At least one n value is required.")
    values.sort()
    return values


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


def _iter_replay_plan_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("replay-plan*.json"):
        if path.is_file() and path.name.startswith("replay-plan"):
            paths.append(path.resolve())
    paths.sort()
    return paths


def _compute_counter_stats(counter: Counter[Any]) -> dict[str, float | int]:
    if not counter:
        return {
            "avg": 0.0,
            "std": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "min": 0.0,
            "max": 0.0,
            "unique_count": 0,
            "total_occurrences": 0,
        }

    values = list(counter.values())
    sorted_values = sorted(values)
    unique_count = len(values)
    total_occurrences = sum(values)
    avg = float(total_occurrences) / float(unique_count)
    variance = sum((float(value) - avg) ** 2 for value in values) / float(unique_count)
    std = math.sqrt(variance)
    p25 = _percentile_from_sorted(sorted_values, 0.25)
    p75 = _percentile_from_sorted(sorted_values, 0.75)
    return {
        "avg": avg,
        "std": std,
        "p25": p25,
        "p75": p75,
        "min": float(min(values)),
        "max": float(max(values)),
        "unique_count": unique_count,
        "total_occurrences": total_occurrences,
    }


def _iter_ngrams(tokens: list[int], n: int):
    for i in range(0, len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def _percentile_from_sorted(sorted_values: list[int], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if quantile <= 0.0:
        return float(sorted_values[0])
    if quantile >= 1.0:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower]) * (1.0 - fraction) + float(sorted_values[upper]) * fraction


def _build_per_step_counter(tokens: list[int] | None, n: int) -> Counter[Any]:
    counter: Counter[Any] = Counter()
    if tokens is None or len(tokens) < n:
        return counter

    if n == 1:
        counter.update(tokens)
        return counter

    for gram in _iter_ngrams(tokens, n):
        counter[gram] += 1
    return counter


def _window_bounds(
    *,
    step_index: int,
    step_count: int,
    window_size: int,
    window_mode: str,
) -> tuple[int, int]:
    if step_count <= 0:
        return 0, -1
    if window_mode == WINDOW_MODE_CUMULATIVE:
        return 0, step_index
    if window_mode == WINDOW_MODE_TRAILING:
        return max(0, step_index - window_size), step_index
    return max(0, step_index - window_size), min(step_count - 1, step_index + window_size)


def _analyze_trace(
    *,
    worker_index: int,
    worker: dict[str, Any],
    window_size: int,
    window_mode: str,
    n_values: list[int],
) -> dict[str, Any]:
    requests_raw = worker.get("requests")
    requests = requests_raw if isinstance(requests_raw, list) else []

    request_ids: list[str | None] = []
    token_sequences: list[list[int] | None] = []
    for request in requests:
        if not isinstance(request, dict):
            request_ids.append(None)
            token_sequences.append(None)
            continue
        request_id = request.get("request_id")
        request_ids.append(request_id if isinstance(request_id, str) else None)
        token_sequences.append(_extract_forced_token_ids(request))

    per_step_ngram_counters: dict[int, list[Counter[Any]]] = {}
    for n in n_values:
        per_step_ngram_counters[n] = [_build_per_step_counter(tokens, n) for tokens in token_sequences]

    steps: list[dict[str, Any]] = []
    for step_index in range(len(token_sequences)):
        window_start, window_end = _window_bounds(
            step_index=step_index,
            step_count=len(token_sequences),
            window_size=window_size,
            window_mode=window_mode,
        )

        window_n_gram_stats: dict[str, dict[str, float | int]] = {}
        for n in n_values:
            window_counter: Counter[Any] = Counter()
            for i in range(window_start, window_end + 1):
                window_counter.update(per_step_ngram_counters[n][i])
            window_n_gram_stats[str(n)] = _compute_counter_stats(window_counter)

        window_tokens = token_sequences[window_start : window_end + 1]
        steps.append(
            {
                "step_index": step_index,
                "request_id": request_ids[step_index],
                "window_start_step": window_start,
                "window_end_step": window_end,
                "window_request_count": window_end - window_start + 1,
                "window_requests_with_forced_token_ids": sum(
                    1 for tokens in window_tokens if tokens is not None
                ),
                "n_gram_stats": window_n_gram_stats,
            }
        )

    return {
        "trace_index": worker_index,
        "worker_id": worker.get("worker_id") if isinstance(worker.get("worker_id"), str) else None,
        "request_count": len(token_sequences),
        "request_with_forced_token_ids": sum(1 for tokens in token_sequences if tokens is not None),
        "steps": steps,
    }


def _analyze_plan(
    *,
    plan_path_str: str,
    root_str: str,
    window_size: int,
    window_mode: str,
    n_values: list[int],
) -> dict[str, Any]:
    plan_path = Path(plan_path_str)
    root = Path(root_str)

    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        return {
            "valid": False,
            "plan_path": str(plan_path),
            "error": f"json_read_error: {exc}",
        }

    if not isinstance(payload, dict):
        return {
            "valid": False,
            "plan_path": str(plan_path),
            "error": "invalid_payload: root is not an object",
        }

    workers = payload.get("workers")
    if not isinstance(workers, list):
        return {
            "valid": False,
            "plan_path": str(plan_path),
            "error": "invalid_payload: workers is not a list",
        }

    try:
        relative_plan_path = str(plan_path.relative_to(root))
    except ValueError:
        relative_plan_path = plan_path.name

    trace_results: list[dict[str, Any]] = []
    for worker_index, worker in enumerate(workers):
        if not isinstance(worker, dict):
            continue
        trace_results.append(
            _analyze_trace(
                worker_index=worker_index,
                worker=worker,
                window_size=window_size,
                window_mode=window_mode,
                n_values=n_values,
            )
        )

    effective_window_size: int | None = (
        None if window_mode == WINDOW_MODE_CUMULATIVE else int(window_size)
    )

    output_payload = {
        "schema_version": "gram-freq-fine.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_plan_path": str(plan_path),
        "relative_plan_path": relative_plan_path,
        "window_size": effective_window_size,
        "window_mode": window_mode,
        "n_values": n_values,
        "trace_count": len(trace_results),
        "traces": trace_results,
    }

    return {
        "valid": True,
        "plan_path": str(plan_path),
        "relative_plan_path": relative_plan_path,
        "payload": output_payload,
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


def _build_output_rel_path(
    relative_plan_path: str,
    plan_path: str,
    window_size: int | None,
    window_mode: str,
) -> Path:
    window_marker = f".w{window_size}" if window_size is not None else ""
    mode_marker = f".{window_mode}"
    rel = Path(relative_plan_path)
    if not rel.is_absolute() and ".." not in rel.parts:
        return rel.parent / f"{rel.stem}{window_marker}{mode_marker}.gram-freq-fine.json"

    digest = hashlib.sha1(plan_path.encode("utf-8")).hexdigest()[:12]
    stem = Path(plan_path).stem
    return Path(f"{stem}.{digest}{window_marker}{mode_marker}.gram-freq-fine.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compute_replay_plan_gram_freq_fine.py",
        description=(
            "Recursively scan replay-plan*.json files and compute per-trace, "
            "per-step windowed n-gram frequency stats."
        ),
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to recursively scan for replay plans.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help=(
            "Window size w (default: 2). Ignored when --window-mode=cumulative."
        ),
    )
    parser.add_argument(
        "--window-mode",
        choices=WINDOW_MODE_CHOICES,
        default=WINDOW_MODE_CENTERED,
        help=(
            "Window mode: centered=[t-w, t+w], trailing=[t-w, t], "
            "cumulative=[0, t] (default: centered)."
        ),
    )
    parser.add_argument(
        "--n-values",
        type=_parse_n_values,
        default=[1, 2],
        help="Comma-separated n values (default: 1,2).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for per-plan JSON files. "
            "Default: <analysis/gram-freq-fine>/output/run-<UTC timestamp>"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker process count for per-file analysis (default: CPU count).",
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
    if not root.is_dir():
        print(f"error: --root is not a directory: {root}", file=sys.stderr)
        return 2

    if args.window_size < 0:
        print("error: --window-size must be >= 0", file=sys.stderr)
        return 2

    jobs = max(1, int(args.jobs))
    script_dir = Path(__file__).resolve().parent
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (script_dir / "output" / f"run-{_utc_now_timestamp()}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_paths = _iter_replay_plan_paths(root)
    show_progress = sys.stderr.isatty() and not args.no_progress

    valid_plan_count = 0
    invalid_plan_count = 0
    invalid_plans: list[dict[str, str]] = []

    if plan_paths:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_plan = {
                executor.submit(
                    _analyze_plan,
                    plan_path_str=str(plan_path),
                    root_str=str(root),
                    window_size=args.window_size,
                    window_mode=str(args.window_mode),
                    n_values=args.n_values,
                ): plan_path
                for plan_path in plan_paths
            }

            completed = 0
            if show_progress:
                _render_progress_bar(completed, len(plan_paths))

            for future in as_completed(future_to_plan):
                source_path = future_to_plan[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - best effort diagnostics
                    invalid_plan_count += 1
                    invalid_plans.append(
                        {"plan_path": str(source_path), "error": f"worker_error: {exc}"}
                    )
                else:
                    if result.get("valid"):
                        output_window_size: int | None = (
                            None
                            if str(args.window_mode) == WINDOW_MODE_CUMULATIVE
                            else int(args.window_size)
                        )
                        output_rel_path = _build_output_rel_path(
                            relative_plan_path=str(result["relative_plan_path"]),
                            plan_path=str(result["plan_path"]),
                            window_size=output_window_size,
                            window_mode=str(args.window_mode),
                        )
                        output_path = output_dir / output_rel_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_text(
                            json.dumps(result["payload"], indent=2, ensure_ascii=True) + "\n",
                            encoding="utf-8",
                        )
                        valid_plan_count += 1
                    else:
                        invalid_plan_count += 1
                        invalid_plans.append(
                            {
                                "plan_path": str(result.get("plan_path", source_path)),
                                "error": str(result.get("error", "unknown error")),
                            }
                        )

                completed += 1
                if show_progress:
                    _render_progress_bar(completed, len(plan_paths))

        if show_progress:
            print(file=sys.stderr)

    print(
        json.dumps(
            {
                "status": "ok",
                "root": str(root),
                "output_dir": str(output_dir),
                "window_size": (
                    None
                    if str(args.window_mode) == WINDOW_MODE_CUMULATIVE
                    else args.window_size
                ),
                "window_mode": args.window_mode,
                "n_values": args.n_values,
                "plan_count": len(plan_paths),
                "valid_plan_count": valid_plan_count,
                "invalid_plan_count": invalid_plan_count,
                "invalid_plans": invalid_plans,
                "jobs": jobs,
            },
            ensure_ascii=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
