from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pp_common.profile_id import int_or_none
from pp_common.profile_id import profile_ids_from_payload
from pp_common.profile_id import profile_label


DEFAULT_LLM_REQUESTS_INPUT_NAME = "llm-requests.json"
DEFAULT_ACTIVITY_OUTPUT_NAME = "prefill-activities.json"
DEFAULT_TIMESERIES_OUTPUT_NAME = "prefill-concurrency-timeseries.json"
DEFAULT_STATS_OUTPUT_NAME = "prefill-concurrency-stats.json"
DEFAULT_TICK_MS = 10


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
            "Extract prefill activity ranges and 10ms prefill concurrency series "
            "from gateway llm-requests output."
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
            "<run-dir>/post-processed/prefill-concurrency/"
        ),
    )
    parser.add_argument(
        "--llm-requests",
        default=None,
        help=(
            "Optional input path to llm-requests.json (only for --run-dir). "
            "Default: <run-dir>/post-processed/gateway/llm-requests/"
            f"{DEFAULT_LLM_REQUESTS_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--tick-ms",
        type=int,
        default=DEFAULT_TICK_MS,
        help=f"Tick size in milliseconds for concurrency series (default: {DEFAULT_TICK_MS}).",
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


def _default_llm_requests_path_for_run(run_dir: Path) -> Path:
    return (
        run_dir
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / DEFAULT_LLM_REQUESTS_INPUT_NAME
    ).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "prefill-concurrency").resolve()


def discover_run_dirs_with_llm_requests(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for llm_requests_path in root_dir.rglob(DEFAULT_LLM_REQUESTS_INPUT_NAME):
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


def _load_llm_requests_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing required file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object in llm requests file: {path}")
    return payload


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _gateway_profile_id_or_none(value: Any) -> int | None:
    return int_or_none(value)


def _prefill_activity_sort_key(activity: dict[str, Any]) -> tuple[float, float, str]:
    prefill_start_offset_s = _float_or_none(activity.get("prefill_start_offset_s"))
    prefill_end_offset_s = _float_or_none(activity.get("prefill_end_offset_s"))
    return (
        prefill_start_offset_s if prefill_start_offset_s is not None else float("inf"),
        prefill_end_offset_s if prefill_end_offset_s is not None else float("inf"),
        str(activity.get("request_id") or ""),
    )


def _build_prefill_activities(request_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    activities: list[dict[str, Any]] = []

    for request in request_records:
        request_start_offset_s = _float_or_none(request.get("request_start_offset_s"))
        time_in_prefill_s = _float_or_none(request.get("gen_ai.latency.time_in_model_prefill"))
        if request_start_offset_s is None or time_in_prefill_s is None:
            continue
        if time_in_prefill_s <= 0:
            continue

        time_in_queue_s = _float_or_none(request.get("gen_ai.latency.time_in_queue"))
        if time_in_queue_s is None:
            time_in_queue_s = 0.0

        prefill_start_offset_s = request_start_offset_s + time_in_queue_s
        prefill_end_offset_s = prefill_start_offset_s + time_in_prefill_s
        if prefill_end_offset_s <= prefill_start_offset_s:
            continue

        activities.append(
            {
                "gateway_run_id": request.get("gateway_run_id"),
                "gateway_profile_id": request.get("gateway_profile_id"),
                "trace_id": request.get("trace_id"),
                "request_id": request.get("request_id"),
                "status_code": _int_or_none(request.get("status_code")),
                "request_start_offset_s": round(request_start_offset_s, 6),
                "request_end_offset_s": _float_or_none(request.get("request_end_offset_s")),
                "time_in_queue_s": round(time_in_queue_s, 6),
                "prefill_duration_s": round(time_in_prefill_s, 6),
                "prefill_start_offset_s": round(prefill_start_offset_s, 6),
                "prefill_end_offset_s": round(prefill_end_offset_s, 6),
            }
        )

    activities.sort(key=_prefill_activity_sort_key)
    return activities


def _resolve_total_duration_s(
    request_records: list[dict[str, Any]],
    prefill_activities: list[dict[str, Any]],
) -> float:
    candidates: list[float] = []

    for request in request_records:
        request_end_offset_s = _float_or_none(request.get("request_end_offset_s"))
        request_end_to_run_end_s = _float_or_none(request.get("request_end_to_run_end_s"))
        if request_end_offset_s is not None:
            candidates.append(max(request_end_offset_s, 0.0))
            if request_end_to_run_end_s is not None:
                candidates.append(max(request_end_offset_s + request_end_to_run_end_s, 0.0))

    for activity in prefill_activities:
        prefill_end_offset_s = _float_or_none(activity.get("prefill_end_offset_s"))
        if prefill_end_offset_s is not None:
            candidates.append(max(prefill_end_offset_s, 0.0))

    if not candidates:
        return 0.0
    return round(max(candidates), 6)


def _sample_count(total_duration_s: float, *, tick_s: float) -> int:
    if total_duration_s <= 0:
        return 0
    return max(1, int(math.ceil((total_duration_s / tick_s) - 1e-12)))


def _build_prefill_concurrency_points(
    prefill_activities: list[dict[str, Any]],
    *,
    total_duration_s: float,
    tick_s: float,
) -> list[dict[str, float | int]]:
    samples = _sample_count(total_duration_s, tick_s=tick_s)
    if samples <= 0:
        return []

    deltas = [0] * (samples + 1)
    for activity in prefill_activities:
        prefill_start_offset_s = _float_or_none(activity.get("prefill_start_offset_s"))
        prefill_end_offset_s = _float_or_none(activity.get("prefill_end_offset_s"))
        if prefill_start_offset_s is None or prefill_end_offset_s is None:
            continue

        clipped_start_s = min(max(prefill_start_offset_s, 0.0), total_duration_s)
        clipped_end_s = min(max(prefill_end_offset_s, 0.0), total_duration_s)
        if clipped_end_s <= clipped_start_s:
            continue

        start_tick = int(math.floor(clipped_start_s / tick_s))
        end_tick = int(math.ceil(clipped_end_s / tick_s))
        start_tick = min(max(start_tick, 0), samples)
        end_tick = min(max(end_tick, 0), samples)
        if end_tick <= start_tick:
            continue

        deltas[start_tick] += 1
        deltas[end_tick] -= 1

    points: list[dict[str, float | int]] = []
    active = 0
    for tick_index in range(samples):
        active += deltas[tick_index]
        points.append(
            {
                "tick_index": tick_index,
                "time_offset_s": round(tick_index * tick_s, 6),
                "concurrency": active,
            }
        )
    return points


def _build_interval_length_stats(
    interval_lengths_ticks: list[int],
    *,
    tick_s: float,
) -> dict[str, int | float]:
    if not interval_lengths_ticks:
        return {
            "interval_count": 0,
            "avg_interval_length_ticks": 0.0,
            "min_interval_length_ticks": 0,
            "max_interval_length_ticks": 0,
            "std_interval_length_ticks": 0.0,
            "avg_interval_length_s": 0.0,
            "min_interval_length_s": 0.0,
            "max_interval_length_s": 0.0,
            "std_interval_length_s": 0.0,
        }

    interval_count = len(interval_lengths_ticks)
    avg_interval_length_ticks = sum(interval_lengths_ticks) / interval_count
    variance_ticks = (
        sum(
            (interval_length_ticks - avg_interval_length_ticks) ** 2
            for interval_length_ticks in interval_lengths_ticks
        )
        / interval_count
    )
    std_interval_length_ticks = math.sqrt(variance_ticks)
    min_interval_length_ticks = min(interval_lengths_ticks)
    max_interval_length_ticks = max(interval_lengths_ticks)

    return {
        "interval_count": interval_count,
        "avg_interval_length_ticks": round(avg_interval_length_ticks, 6),
        "min_interval_length_ticks": min_interval_length_ticks,
        "max_interval_length_ticks": max_interval_length_ticks,
        "std_interval_length_ticks": round(std_interval_length_ticks, 6),
        "avg_interval_length_s": round(avg_interval_length_ticks * tick_s, 6),
        "min_interval_length_s": round(min_interval_length_ticks * tick_s, 6),
        "max_interval_length_s": round(max_interval_length_ticks * tick_s, 6),
        "std_interval_length_s": round(std_interval_length_ticks * tick_s, 6),
    }


def _build_concurrency_interval_length_stats(
    prefill_concurrency_points: list[dict[str, float | int]],
    *,
    tick_s: float,
) -> dict[str, dict[str, int | float]]:
    values = [
        int(point["concurrency"])
        for point in prefill_concurrency_points
        if isinstance(point.get("concurrency"), int)
    ]
    if not values:
        return {}

    interval_lengths_by_concurrency: dict[int, list[int]] = {}
    current_value = values[0]
    current_interval_length = 1
    for value in values[1:]:
        if value == current_value:
            current_interval_length += 1
            continue
        interval_lengths_by_concurrency.setdefault(current_value, []).append(
            current_interval_length
        )
        current_value = value
        current_interval_length = 1
    interval_lengths_by_concurrency.setdefault(current_value, []).append(current_interval_length)

    concurrency_interval_length_stats: dict[str, dict[str, int | float]] = {}
    for concurrency in sorted(interval_lengths_by_concurrency):
        interval_lengths_ticks = interval_lengths_by_concurrency[concurrency]
        concurrency_interval_length_stats[str(concurrency)] = {
            "concurrency": concurrency,
            **_build_interval_length_stats(interval_lengths_ticks, tick_s=tick_s),
        }
    return concurrency_interval_length_stats


def _build_prefill_concurrency_stats(
    prefill_concurrency_points: list[dict[str, float | int]],
) -> tuple[int, int, float]:
    if not prefill_concurrency_points:
        return 0, 0, 0.0
    values = [
        int(point["concurrency"])
        for point in prefill_concurrency_points
        if isinstance(point.get("concurrency"), int)
    ]
    if not values:
        return 0, 0, 0.0
    return min(values), max(values), round(sum(values) / len(values), 6)


def _build_prefill_timeseries_and_stats_payloads(
    request_records: list[dict[str, Any]],
    prefill_activities: list[dict[str, Any]],
    *,
    total_duration_s: float,
    tick_ms: int,
    tick_s: float,
    gateway_profile_id: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prefill_concurrency_points = _build_prefill_concurrency_points(
        prefill_activities,
        total_duration_s=total_duration_s,
        tick_s=tick_s,
    )
    concurrency_interval_length_stats = _build_concurrency_interval_length_stats(
        prefill_concurrency_points,
        tick_s=tick_s,
    )
    min_concurrency, max_concurrency, avg_concurrency = _build_prefill_concurrency_stats(
        prefill_concurrency_points
    )

    common = {
        "request_count": len(request_records),
        "prefill_activity_count": len(prefill_activities),
        "total_duration_s": total_duration_s,
        "tick_ms": tick_ms,
        "tick_s": tick_s,
    }
    if gateway_profile_id is not None:
        common["gateway_profile_id"] = gateway_profile_id

    timeseries_payload = {
        **common,
        "sample_count": len(prefill_concurrency_points),
        "concurrency_points": prefill_concurrency_points,
    }
    stats_payload = {
        **common,
        "sample_count": len(prefill_concurrency_points),
        "min_concurrency": min_concurrency,
        "max_concurrency": max_concurrency,
        "avg_concurrency": avg_concurrency,
        "concurrency_interval_length_stats": concurrency_interval_length_stats,
    }
    return timeseries_payload, stats_payload


def extract_prefill_concurrency_from_run_dir(
    run_dir: Path,
    *,
    llm_requests_path: Path | None = None,
    tick_ms: int = DEFAULT_TICK_MS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if tick_ms <= 0:
        raise ValueError(f"tick_ms must be a positive integer: {tick_ms}")
    tick_s = round(tick_ms / 1000.0, 6)

    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_llm_requests_path = (
        llm_requests_path or _default_llm_requests_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    llm_requests_payload = _load_llm_requests_payload(resolved_llm_requests_path)
    request_records = llm_requests_payload.get("requests")
    if not isinstance(request_records, list):
        raise ValueError(
            "llm-requests payload is missing list field 'requests': "
            f"{resolved_llm_requests_path}"
        )
    request_records = [item for item in request_records if isinstance(item, dict)]

    prefill_activities = _build_prefill_activities(request_records)
    total_duration_s = _resolve_total_duration_s(request_records, prefill_activities)
    aggregate_timeseries_payload, aggregate_stats_payload = (
        _build_prefill_timeseries_and_stats_payloads(
            request_records,
            prefill_activities,
            total_duration_s=total_duration_s,
            tick_ms=tick_ms,
            tick_s=tick_s,
        )
    )
    port_profile_ids = sorted(
        set(profile_ids_from_payload(llm_requests_payload))
        | {
            profile_id
            for profile_id in (
                _gateway_profile_id_or_none(record.get("gateway_profile_id"))
                for record in request_records
            )
            if profile_id is not None
        }
        | {
            profile_id
            for profile_id in (
                _gateway_profile_id_or_none(activity.get("gateway_profile_id"))
                for activity in prefill_activities
            )
            if profile_id is not None
        }
    )

    common = {
        "source_run_dir": str(resolved_run_dir),
        "source_llm_requests_path": str(resolved_llm_requests_path),
        "source_gateway_output_dir": llm_requests_payload.get("source_gateway_output_dir"),
        "service_failure_detected": bool(
            llm_requests_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": llm_requests_payload.get(
            "service_failure_cutoff_time_utc"
        ),
        "multi_profile": len(port_profile_ids) > 1,
        "port_profile_ids": port_profile_ids,
        "series_keys": [profile_label(profile_id) for profile_id in port_profile_ids],
    }

    activities_by_profile: dict[str, dict[str, Any]] = {}
    timeseries_by_profile: dict[str, dict[str, Any]] = {}
    stats_by_profile: dict[str, dict[str, Any]] = {}
    for gateway_profile_id in port_profile_ids:
        series_key = profile_label(gateway_profile_id)
        profile_request_records = [
            record
            for record in request_records
            if _gateway_profile_id_or_none(record.get("gateway_profile_id")) == gateway_profile_id
        ]
        profile_prefill_activities = [
            activity
            for activity in prefill_activities
            if _gateway_profile_id_or_none(activity.get("gateway_profile_id")) == gateway_profile_id
        ]
        profile_timeseries_payload, profile_stats_payload = (
            _build_prefill_timeseries_and_stats_payloads(
                profile_request_records,
                profile_prefill_activities,
                total_duration_s=total_duration_s,
                tick_ms=tick_ms,
                tick_s=tick_s,
                gateway_profile_id=gateway_profile_id,
            )
        )
        activities_by_profile[series_key] = {
            "gateway_profile_id": gateway_profile_id,
            "request_count": len(profile_request_records),
            "prefill_activity_count": len(profile_prefill_activities),
            "activities": profile_prefill_activities,
        }
        timeseries_by_profile[series_key] = {
            **common,
            **profile_timeseries_payload,
        }
        stats_by_profile[series_key] = {
            **common,
            **profile_stats_payload,
        }

    activities_payload = {
        **common,
        "request_count": len(request_records),
        "prefill_activity_count": len(prefill_activities),
        "total_duration_s": total_duration_s,
        "tick_ms": tick_ms,
        "tick_s": tick_s,
        "activities": prefill_activities,
        "activities_by_profile": activities_by_profile,
    }
    timeseries_payload = {
        **common,
        **aggregate_timeseries_payload,
        "series_by_profile": timeseries_by_profile,
    }
    stats_payload = {
        **common,
        **aggregate_stats_payload,
        "series_by_profile": stats_by_profile,
    }
    return activities_payload, timeseries_payload, stats_payload


def extract_run_dir(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    llm_requests_path: Path | None = None,
    tick_ms: int = DEFAULT_TICK_MS,
) -> list[Path]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_dir = (
        output_dir or _default_output_dir_for_run(resolved_run_dir)
    ).expanduser().resolve()
    activities_payload, timeseries_payload, stats_payload = (
        extract_prefill_concurrency_from_run_dir(
            resolved_run_dir,
            llm_requests_path=llm_requests_path,
            tick_ms=tick_ms,
        )
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    activity_path = resolved_output_dir / DEFAULT_ACTIVITY_OUTPUT_NAME
    timeseries_path = resolved_output_dir / DEFAULT_TIMESERIES_OUTPUT_NAME
    stats_path = resolved_output_dir / DEFAULT_STATS_OUTPUT_NAME
    activity_path.write_text(
        json.dumps(activities_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    timeseries_path.write_text(
        json.dumps(timeseries_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return [activity_path, timeseries_path, stats_path]


def _extract_run_dir_worker(job: tuple[str, int]) -> tuple[str, list[str] | None, str | None]:
    run_dir_text, tick_ms = job
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_paths = extract_run_dir(run_dir, tick_ms=tick_ms)
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), [str(path) for path in output_paths], None)


def _run_root_dir_sequential(run_dirs: list[Path], *, tick_ms: int) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_paths = extract_run_dir(run_dir, tick_ms=tick_ms)
            output_dir = output_paths[0].parent if output_paths else _default_output_dir_for_run(run_dir)
            print(f"[done] {run_dir} -> {output_dir}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int, tick_ms: int) -> int:
    failure_count = 0
    jobs = [(str(run_dir), tick_ms) for run_dir in run_dirs]
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_paths_text, error_text in executor.map(
            _extract_run_dir_worker,
            jobs,
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
    if args.tick_ms <= 0:
        raise ValueError(f"--tick-ms must be a positive integer: {args.tick_ms}")
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    llm_requests_path = (
        Path(args.llm_requests).expanduser().resolve() if args.llm_requests else None
    )
    output_paths = extract_run_dir(
        run_dir,
        output_dir=output_dir,
        llm_requests_path=llm_requests_path,
        tick_ms=args.tick_ms,
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
    if args.tick_ms <= 0:
        raise ValueError(f"--tick-ms must be a positive integer: {args.tick_ms}")
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
        failure_count = _run_root_dir_sequential(run_dirs, tick_ms=args.tick_ms)
    else:
        try:
            failure_count = _run_root_dir_parallel(
                run_dirs,
                max_procs=worker_count,
                tick_ms=args.tick_ms,
            )
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(run_dirs, tick_ms=args.tick_ms)

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
