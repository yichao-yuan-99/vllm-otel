from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from typing import Iterable


DEFAULT_OUTPUT_NAME = "duration-split-summary.json"
DEFAULT_SPLIT_COUNT = 10
METRIC_NAMES = (
    "duration_s",
    "turn_count",
    "prompt_tokens",
    "decode_tokens",
    "cached_prompt_tokens",
)


def _default_max_procs() -> int:
    max_procs_env = os.getenv("MAX_PROCS")
    if max_procs_env:
        try:
            parsed = int(max_procs_env)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        return 1
    return cpu_count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split worker jobs by context-usage rank and compute per-bin averages "
            "for duration, turns, and token counts."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing gateway-output/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with a direct gateway-output/ child will be processed."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/post-processed/split/duration/"
            f"{DEFAULT_OUTPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--split-count",
        type=int,
        default=DEFAULT_SPLIT_COUNT,
        help=f"Number of rank-percentile bins (default: {DEFAULT_SPLIT_COUNT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered run directories and exit (only for --root-dir).",
    )
    parser.add_argument(
        "--max-procs",
        type=int,
        default=_default_max_procs(),
        help=(
            "Number of worker processes for --root-dir mode. "
            "Default: MAX_PROCS env var, else CPU count."
        ),
    )
    return parser.parse_args(argv)


def _profile_id_from_name(name: str) -> int | None:
    prefix = "profile-"
    if not name.startswith(prefix):
        return None
    raw = name[len(prefix) :]
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def discover_gateway_run_dirs(gateway_output_dir: Path) -> list[tuple[Path, int | None]]:
    run_dirs: list[tuple[Path, int | None]] = []

    for run_dir in sorted(gateway_output_dir.glob("run_*")):
        if run_dir.is_dir():
            run_dirs.append((run_dir, None))

    for child in sorted(gateway_output_dir.iterdir()):
        if not child.is_dir():
            continue
        profile_id = _profile_id_from_name(child.name)
        if profile_id is None:
            continue
        for run_dir in sorted(child.glob("run_*")):
            if run_dir.is_dir():
                run_dirs.append((run_dir, profile_id))

    return run_dirs


def discover_run_dirs_with_gateway_output(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for gateway_output_dir in root_dir.rglob("gateway-output"):
        if not gateway_output_dir.is_dir():
            continue
        run_dirs.add(gateway_output_dir.parent.resolve())
    return sorted(run_dirs)


def _iter_jsonl_dict_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


_INTEGER_PATTERN = re.compile(r"[-+]?\d+")


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
        if _INTEGER_PATTERN.fullmatch(stripped):
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_seconds(start_dt: datetime | None, end_dt: datetime | None) -> float | None:
    if start_dt is None or end_dt is None:
        return None
    return round((end_dt - start_dt).total_seconds(), 6)


def _extract_usage_tokens(record: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    response = record.get("response")
    if not isinstance(response, dict):
        return None, None, None

    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None, None, None

    prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
    completion_tokens = _int_or_none(usage.get("completion_tokens"))
    cached_tokens = None
    prompt_tokens_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_tokens_details, dict):
        cached_tokens = _int_or_none(prompt_tokens_details.get("cached_tokens"))
    return prompt_tokens, completion_tokens, cached_tokens


def _extract_request_window(record: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    start_dt = _parse_iso8601(record.get("request_start_time"))
    end_dt = _parse_iso8601(record.get("request_end_time"))
    return start_dt, end_dt


def _load_lifecycle_windows(
    lifecycle_path: Path,
) -> tuple[datetime | None, datetime | None, datetime | None, datetime | None]:
    job_start: datetime | None = None
    job_end: datetime | None = None
    agent_start: datetime | None = None
    agent_end: datetime | None = None
    for record in _iter_jsonl_dict_records(lifecycle_path):
        event_type = record.get("event_type")
        timestamp = _parse_iso8601(record.get("timestamp"))
        if timestamp is None:
            continue
        if event_type == "job_start" and job_start is None:
            job_start = timestamp
        if event_type == "job_end":
            job_end = timestamp
        if event_type == "agent_start" and agent_start is None:
            agent_start = timestamp
        if event_type == "agent_end":
            agent_end = timestamp
    return agent_start, agent_end, job_start, job_end


def _extract_job_metrics(
    gateway_run_dir: Path,
    *,
    profile_id: int | None,
) -> dict[str, Any]:
    requests_path = gateway_run_dir / "requests" / "model_inference.jsonl"
    if not requests_path.is_file():
        raise ValueError(f"Missing required file: {requests_path}")

    request_count = 0
    request_count_with_usage = 0
    prompt_tokens_total = 0
    decode_tokens_total = 0
    cached_prompt_tokens_total = 0
    max_request_length: int | None = None
    first_valid_request_start: datetime | None = None
    last_valid_request_end: datetime | None = None

    for record in _iter_jsonl_dict_records(requests_path):
        request_count += 1
        prompt_tokens, decode_tokens, cached_prompt_tokens = _extract_usage_tokens(record)

        request_start, request_end = _extract_request_window(record)
        if prompt_tokens is not None and decode_tokens is not None:
            request_count_with_usage += 1
            prompt_tokens_total += prompt_tokens
            decode_tokens_total += decode_tokens
            if cached_prompt_tokens is not None:
                cached_prompt_tokens_total += cached_prompt_tokens

            request_length = prompt_tokens + decode_tokens
            if max_request_length is None:
                max_request_length = request_length
            else:
                max_request_length = max(max_request_length, request_length)

            if request_start is not None:
                if (
                    first_valid_request_start is None
                    or request_start < first_valid_request_start
                ):
                    first_valid_request_start = request_start
            if request_end is not None:
                if (
                    last_valid_request_end is None
                    or request_end > last_valid_request_end
                ):
                    last_valid_request_end = request_end

    duration_s: float | None = None
    lifecycle_path = gateway_run_dir / "events" / "lifecycle.jsonl"
    if lifecycle_path.is_file():
        agent_start, agent_end, job_start, job_end = _load_lifecycle_windows(
            lifecycle_path
        )
        duration_s = _duration_seconds(agent_start, agent_end)
        if duration_s is None:
            duration_s = _duration_seconds(job_start, job_end)
    if duration_s is None:
        duration_s = _duration_seconds(first_valid_request_start, last_valid_request_end)

    return {
        "gateway_run_id": gateway_run_dir.name,
        "gateway_profile_id": profile_id,
        "max_request_length": max_request_length,
        "duration_s": duration_s,
        "turn_count": request_count_with_usage,
        "request_count_total": request_count,
        "request_count_with_usage": request_count_with_usage,
        "prompt_tokens": prompt_tokens_total,
        "decode_tokens": decode_tokens_total,
        "cached_prompt_tokens": cached_prompt_tokens_total,
    }


def _bin_labels(split_count: int) -> list[str]:
    labels: list[str] = []
    for index in range(split_count):
        lower = int((100 * index) / split_count)
        upper = int((100 * (index + 1)) / split_count)
        labels.append(f"{lower}-{upper}%")
    return labels


def _rank_jobs_by_max_request_length(jobs: list[dict[str, Any]], *, split_count: int) -> list[dict[str, Any]]:
    sorted_jobs = sorted(
        jobs,
        key=lambda job: (
            _int_or_none(job.get("max_request_length"))
            if _int_or_none(job.get("max_request_length")) is not None
            else -1,
            str(job.get("gateway_run_id") or ""),
        ),
    )
    total_jobs = len(sorted_jobs)
    labels = _bin_labels(split_count)
    for index, job in enumerate(sorted_jobs):
        if total_jobs <= 0:
            bin_index = 0
        else:
            bin_index = min(split_count - 1, int(index * split_count / total_jobs))
        job["split_bin"] = labels[bin_index]
        job["split_bin_index"] = bin_index
    return sorted_jobs


def _init_metric_table(labels: list[str]) -> dict[str, dict[str, int | float | None]]:
    return {
        label: {
            "avg": None,
            "count": 0,
        }
        for label in labels
    }


def _build_tables(
    jobs: list[dict[str, Any]],
    *,
    split_count: int,
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    labels = _bin_labels(split_count)
    sums: dict[str, dict[str, float]] = {
        metric_name: {label: 0.0 for label in labels}
        for metric_name in METRIC_NAMES
    }
    counts: dict[str, dict[str, int]] = {
        metric_name: {label: 0 for label in labels}
        for metric_name in METRIC_NAMES
    }

    for job in jobs:
        split_bin = job.get("split_bin")
        if not isinstance(split_bin, str) or split_bin not in sums["duration_s"]:
            continue
        for metric_name in METRIC_NAMES:
            metric_value = _float_or_none(job.get(metric_name))
            if metric_value is None:
                continue
            sums[metric_name][split_bin] += metric_value
            counts[metric_name][split_bin] += 1

    tables: dict[str, dict[str, dict[str, int | float | None]]] = {}
    for metric_name in METRIC_NAMES:
        metric_table = _init_metric_table(labels)
        for label in labels:
            count = counts[metric_name][label]
            avg_value = None
            if count > 0:
                avg_value = sums[metric_name][label] / count
            metric_table[label]["avg"] = avg_value
            metric_table[label]["count"] = count
        tables[metric_name] = metric_table
    return tables


def extract_split_duration_from_run_dir(
    run_dir: Path,
    *,
    split_count: int = DEFAULT_SPLIT_COUNT,
) -> dict[str, Any]:
    if split_count <= 0:
        raise ValueError(f"split_count must be a positive integer: {split_count}")

    resolved_run_dir = run_dir.expanduser().resolve()
    gateway_output_dir = resolved_run_dir / "gateway-output"
    if not gateway_output_dir.is_dir():
        raise ValueError(f"Missing required directory: {gateway_output_dir}")

    discovered_run_dirs = discover_gateway_run_dirs(gateway_output_dir)
    if not discovered_run_dirs:
        raise ValueError(
            "No run_* artifacts found under gateway-output. "
            "Expected either gateway-output/run_* or gateway-output/profile-*/run_*."
        )

    jobs: list[dict[str, Any]] = []
    for gateway_run_dir, profile_id in discovered_run_dirs:
        jobs.append(_extract_job_metrics(gateway_run_dir, profile_id=profile_id))

    ranked_jobs = [
        job for job in jobs if _int_or_none(job.get("max_request_length")) is not None
    ]
    excluded_jobs_no_token_usage = [
        job for job in jobs if _int_or_none(job.get("max_request_length")) is None
    ]
    ranked_jobs = _rank_jobs_by_max_request_length(ranked_jobs, split_count=split_count)
    tables = _build_tables(ranked_jobs, split_count=split_count)

    return {
        "source_run_dir": str(resolved_run_dir),
        "source_gateway_output_dir": str(gateway_output_dir.resolve()),
        "split_count": split_count,
        "bin_labels": _bin_labels(split_count),
        "job_count_total": len(jobs),
        "job_count": len(ranked_jobs),
        "job_count_excluded_no_token_usage": len(excluded_jobs_no_token_usage),
        "metrics": METRIC_NAMES,
        "tables": tables,
        "jobs": ranked_jobs,
        "excluded_jobs_no_token_usage": excluded_jobs_no_token_usage,
    }


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "split" / "duration" / DEFAULT_OUTPUT_NAME).resolve()


def extract_run_dir(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    split_count: int = DEFAULT_SPLIT_COUNT,
) -> Path:
    resolved_output_path = (output_path or _default_output_path_for_run(run_dir)).expanduser().resolve()
    result = extract_split_duration_from_run_dir(run_dir, split_count=split_count)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def _extract_run_dir_worker(job: tuple[str, int]) -> tuple[str, str | None, str | None]:
    run_dir_text, split_count = job
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = extract_run_dir(run_dir, split_count=split_count)
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(run_dirs: list[Path], *, split_count: int) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = extract_run_dir(run_dir, split_count=split_count)
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int, split_count: int) -> int:
    failure_count = 0
    jobs = [(str(run_dir), split_count) for run_dir in run_dirs]
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_path_text, error_text in executor.map(
            _extract_run_dir_worker,
            jobs,
        ):
            if error_text is None:
                print(f"[done] {run_dir_text} -> {output_path_text}")
            else:
                failure_count += 1
                print(f"[error] {run_dir_text}: {error_text}", file=sys.stderr)
    return failure_count


def _main_run_dir(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")
    if args.split_count <= 0:
        raise ValueError(f"--split-count must be a positive integer: {args.split_count}")
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    resolved_output_path = extract_run_dir(
        run_dir,
        output_path=output_path,
        split_count=args.split_count,
    )
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.split_count <= 0:
        raise ValueError(f"--split-count must be a positive integer: {args.split_count}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_gateway_output(root_dir)
    print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
    if not run_dirs:
        return 0
    if args.dry_run:
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    worker_count = min(args.max_procs, len(run_dirs))
    print(f"Running extraction with {worker_count} worker process(es)")

    if worker_count <= 1:
        failure_count = _run_root_dir_sequential(run_dirs, split_count=args.split_count)
    else:
        try:
            failure_count = _run_root_dir_parallel(
                run_dirs,
                max_procs=worker_count,
                split_count=args.split_count,
            )
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(run_dirs, split_count=args.split_count)

    if failure_count:
        print(
            f"Completed with {failure_count} failure(s) out of {len(run_dirs)} run directories.",
            file=sys.stderr,
        )
        return 1
    print(f"Completed extraction for {len(run_dirs)} run directories.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_dir:
        return _main_run_dir(args)
    return _main_root_dir(args)


if __name__ == "__main__":
    raise SystemExit(main())
