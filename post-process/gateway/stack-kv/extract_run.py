from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


DEFAULT_LLM_REQUESTS_OUTPUT_NAME = "llm-requests.json"
RANGES_OUTPUT_NAME = "kv-usage-ranges.json"
HISTOGRAM_OUTPUT_NAME = "kv-usage-stacked-histogram.json"

_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


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
            "Recover stacked request KV usage over time from gateway "
            "llm-requests.json."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/gateway/llm-requests/llm-requests.json will be processed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. Default: "
            "<run-dir>/post-processed/gateway/stack-kv/"
        ),
    )
    parser.add_argument(
        "--llm-requests",
        default=None,
        help=(
            "Optional input path to llm-requests.json (only for --run-dir). "
            "Default: <run-dir>/post-processed/gateway/llm-requests/"
            f"{DEFAULT_LLM_REQUESTS_OUTPUT_NAME}"
        ),
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


def discover_run_dirs_with_llm_requests(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for llm_requests_path in root_dir.rglob(DEFAULT_LLM_REQUESTS_OUTPUT_NAME):
        if not llm_requests_path.is_file():
            continue
        if llm_requests_path.parent.name != "llm-requests":
            continue
        if llm_requests_path.parent.parent.name != "gateway":
            continue
        if llm_requests_path.parent.parent.parent.name != "post-processed":
            continue
        run_dirs.add(llm_requests_path.parent.parent.parent.parent.resolve())
    return sorted(run_dirs)


def _default_llm_requests_path_for_run(run_dir: Path) -> Path:
    return (
        run_dir
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / DEFAULT_LLM_REQUESTS_OUTPUT_NAME
    ).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "gateway" / "stack-kv").resolve()


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if _FLOAT_PATTERN.fullmatch(stripped) is None:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        if math.isfinite(parsed):
            return parsed
    return None


def _load_llm_request_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing required file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object in llm request file: {path}")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        raise ValueError(f"Missing 'requests' list in llm request file: {path}")
    request_records = [item for item in raw_requests if isinstance(item, dict)]
    return request_records, payload


def _resolve_gateway_output_dir(
    run_dir: Path,
    llm_requests_payload: dict[str, Any],
) -> Path:
    payload_path = llm_requests_payload.get("source_gateway_output_dir")
    if isinstance(payload_path, str) and payload_path.strip():
        candidate = Path(payload_path).expanduser().resolve()
        if candidate.is_dir():
            return candidate
    return (run_dir / "gateway-output").resolve()


def _build_kv_usage_range(
    request_record: dict[str, Any],
    *,
    range_start_s: float,
    range_end_s: float,
    kv_usage_tokens: float,
) -> dict[str, Any]:
    duration_s = range_end_s - range_start_s
    return {
        "gateway_run_id": request_record.get("gateway_run_id"),
        "gateway_profile_id": request_record.get("gateway_profile_id"),
        "request_id": request_record.get("request_id"),
        "trace_id": request_record.get("trace_id"),
        "range_start_s": round(range_start_s, 6),
        "range_end_s": round(range_end_s, 6),
        "range_duration_s": round(duration_s, 6),
        "kv_usage_tokens": round(kv_usage_tokens, 6),
        "total_value": round(kv_usage_tokens * duration_s, 6),
        "avg_value_per_s": round(kv_usage_tokens, 6),
    }


def _build_kv_usage_ranges(
    request_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []

    for request_record in request_records:
        range_start_s = _float_or_none(request_record.get("request_start_offset_s"))
        range_end_s = _float_or_none(request_record.get("request_end_offset_s"))
        if range_start_s is None or range_end_s is None:
            continue
        if range_end_s <= range_start_s:
            continue

        prompt_tokens = _float_or_none(request_record.get("prompt_tokens"))
        if prompt_tokens is None:
            continue
        completion_tokens = _float_or_none(request_record.get("completion_tokens"))
        if completion_tokens is None:
            completion_tokens = 0.0

        kv_usage_tokens = prompt_tokens + completion_tokens
        ranges.append(
            _build_kv_usage_range(
                request_record,
                range_start_s=range_start_s,
                range_end_s=range_end_s,
                kv_usage_tokens=kv_usage_tokens,
            )
        )

    ranges.sort(
        key=lambda entry: (
            _float_or_none(entry.get("range_start_s"))
            if _float_or_none(entry.get("range_start_s")) is not None
            else float("inf"),
            str(entry.get("request_id") or ""),
        )
    )
    return ranges


def build_stacked_histogram(range_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals: list[tuple[float, float, float]] = []
    max_second = 0

    for entry in range_entries:
        range_start_s = _float_or_none(entry.get("range_start_s"))
        range_end_s = _float_or_none(entry.get("range_end_s"))
        avg_value_per_s = _float_or_none(entry.get("avg_value_per_s"))
        if range_start_s is None or range_end_s is None or avg_value_per_s is None:
            continue
        if range_end_s <= range_start_s:
            continue
        if avg_value_per_s == 0:
            continue

        clipped_start_s = max(0.0, range_start_s)
        clipped_end_s = max(0.0, range_end_s)
        if clipped_end_s <= clipped_start_s:
            continue

        intervals.append((clipped_start_s, clipped_end_s, avg_value_per_s))
        max_second = max(max_second, int(math.ceil(clipped_end_s)))

    if max_second <= 0:
        return []

    full_bin_rate_deltas = [0.0] * (max_second + 1)
    partial_values: dict[int, float] = {}

    for interval_start_s, interval_end_s, avg_value_per_s in intervals:
        start_bin = int(math.floor(interval_start_s))
        end_floor = int(math.floor(interval_end_s))

        if start_bin == end_floor:
            partial_values[start_bin] = partial_values.get(start_bin, 0.0) + (
                avg_value_per_s * (interval_end_s - interval_start_s)
            )
            continue

        partial_values[start_bin] = partial_values.get(start_bin, 0.0) + (
            avg_value_per_s * ((start_bin + 1) - interval_start_s)
        )

        end_partial_width = interval_end_s - end_floor
        if end_partial_width > 0.0:
            partial_values[end_floor] = partial_values.get(end_floor, 0.0) + (
                avg_value_per_s * end_partial_width
            )

        full_start_bin = start_bin + 1
        full_end_bin = end_floor - 1
        if full_start_bin <= full_end_bin:
            full_bin_rate_deltas[full_start_bin] += avg_value_per_s
            full_bin_rate_deltas[full_end_bin + 1] -= avg_value_per_s

    points: list[dict[str, Any]] = []
    running_full_bin_rate = 0.0
    for second in range(max_second):
        running_full_bin_rate += full_bin_rate_deltas[second]
        value = running_full_bin_rate + partial_values.get(second, 0.0)
        points.append(
            {
                "second": second,
                "accumulated_value": value,
            }
        )
    return points


def extract_gateway_stack_kv_from_run_dir(
    run_dir: Path,
    *,
    llm_requests_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_llm_requests_path = (
        llm_requests_path or _default_llm_requests_path_for_run(resolved_run_dir)
    ).expanduser().resolve()

    request_records, llm_requests_payload = _load_llm_request_records(resolved_llm_requests_path)
    ranges = _build_kv_usage_ranges(request_records)
    histogram_points = build_stacked_histogram(ranges)

    source_gateway_output_dir = str(
        _resolve_gateway_output_dir(resolved_run_dir, llm_requests_payload)
    )
    source_run_dir = str(resolved_run_dir)
    input_request_count = llm_requests_payload.get("request_count")
    if not isinstance(input_request_count, int):
        input_request_count = len(request_records)
    service_failure_detected = bool(
        llm_requests_payload.get("service_failure_detected", False)
    )
    service_failure_cutoff_time_utc = llm_requests_payload.get("service_failure_cutoff_time_utc")

    ranges_payload = {
        "source_run_dir": source_run_dir,
        "source_gateway_output_dir": source_gateway_output_dir,
        "source_llm_requests_path": str(resolved_llm_requests_path),
        "service_failure_detected": service_failure_detected,
        "service_failure_cutoff_time_utc": service_failure_cutoff_time_utc,
        "input_request_count": input_request_count,
        "metric": "kv_usage_tokens",
        "phase": "request_lifetime",
        "entry_count": len(ranges),
        "entries": ranges,
    }
    histogram_payload = {
        "source_run_dir": source_run_dir,
        "source_gateway_output_dir": source_gateway_output_dir,
        "source_llm_requests_path": str(resolved_llm_requests_path),
        "service_failure_detected": service_failure_detected,
        "service_failure_cutoff_time_utc": service_failure_cutoff_time_utc,
        "input_request_count": input_request_count,
        "metric": "kv_usage_tokens",
        "bucket_width_s": 1,
        "point_count": len(histogram_points),
        "points": histogram_points,
    }
    return ranges_payload, histogram_payload


def extract_run_dir(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    llm_requests_path: Path | None = None,
) -> list[Path]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_dir = (
        output_dir or _default_output_dir_for_run(resolved_run_dir)
    ).expanduser().resolve()
    ranges_payload, histogram_payload = extract_gateway_stack_kv_from_run_dir(
        resolved_run_dir,
        llm_requests_path=llm_requests_path,
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    ranges_output_path = resolved_output_dir / RANGES_OUTPUT_NAME
    histogram_output_path = resolved_output_dir / HISTOGRAM_OUTPUT_NAME
    ranges_output_path.write_text(
        json.dumps(ranges_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    histogram_output_path.write_text(
        json.dumps(histogram_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return [ranges_output_path, histogram_output_path]


def _extract_run_dir_worker(run_dir_text: str) -> tuple[str, list[str] | None, str | None]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_paths = extract_run_dir(run_dir)
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), [str(path) for path in output_paths], None)


def _run_root_dir_sequential(run_dirs: list[Path]) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_paths = extract_run_dir(run_dir)
            output_dir = output_paths[0].parent if output_paths else _default_output_dir_for_run(run_dir)
            print(f"[done] {run_dir} -> {output_dir}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int) -> int:
    failure_count = 0
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_paths_text, error_text in executor.map(
            _extract_run_dir_worker,
            [str(run_dir) for run_dir in run_dirs],
        ):
            if error_text is None:
                output_dir_text = (
                    str(Path(output_paths_text[0]).parent)
                    if output_paths_text
                    else "<unknown-output-dir>"
                )
                print(f"[done] {run_dir_text} -> {output_dir_text}")
            else:
                failure_count += 1
                print(f"[error] {run_dir_text}: {error_text}", file=sys.stderr)
    return failure_count


def _main_run_dir(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    llm_requests_path = (
        Path(args.llm_requests).expanduser().resolve() if args.llm_requests else None
    )
    output_paths = extract_run_dir(
        run_dir,
        output_dir=output_dir,
        llm_requests_path=llm_requests_path,
    )
    for output_path in output_paths:
        print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.llm_requests:
        raise ValueError("--llm-requests can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_llm_requests(root_dir)
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
        failure_count = _run_root_dir_sequential(run_dirs)
    else:
        try:
            failure_count = _run_root_dir_parallel(run_dirs, max_procs=worker_count)
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(run_dirs)

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
