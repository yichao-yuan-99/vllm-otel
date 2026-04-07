from __future__ import annotations

import argparse
from bisect import bisect_left
from bisect import bisect_right
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


DEFAULT_OUTPUT_NAME = "request-throughput-timeseries.json"
DEFAULT_TIMEPOINT_FREQ_HZ = 1.0
DEFAULT_WINDOW_SIZE_S = 600.0
DEFAULT_LLM_REQUESTS_REL_PATH = Path("post-processed/gateway/llm-requests/llm-requests.json")


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
            "Extract moving request throughput (requests per second) from "
            "gateway llm-requests output for one run or for all runs under a root directory."
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
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/post-processed/request-throughput/"
            f"{DEFAULT_OUTPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--llm-requests",
        default=None,
        help=(
            "Optional llm-requests input path (only for --run-dir). Default: "
            f"<run-dir>/{DEFAULT_LLM_REQUESTS_REL_PATH.as_posix()}"
        ),
    )
    parser.add_argument(
        "--timepoint-freq-hz",
        type=float,
        default=DEFAULT_TIMEPOINT_FREQ_HZ,
        help=(
            "Sampling frequency for throughput time points "
            f"(default: {DEFAULT_TIMEPOINT_FREQ_HZ})."
        ),
    )
    parser.add_argument(
        "--window-size-s",
        type=float,
        default=DEFAULT_WINDOW_SIZE_S,
        help=(
            "Half-width of the throughput window around each time point in seconds "
            f"(default: {DEFAULT_WINDOW_SIZE_S})."
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _status_code_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer() and math.isfinite(value):
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def _default_llm_requests_path_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_LLM_REQUESTS_REL_PATH).resolve()


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "request-throughput" / DEFAULT_OUTPUT_NAME).resolve()


def discover_run_dirs_with_llm_requests(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for llm_requests_path in root_dir.rglob("llm-requests.json"):
        if not llm_requests_path.is_file():
            continue
        if llm_requests_path.parent.name != "llm-requests":
            continue
        gateway_dir = llm_requests_path.parent.parent
        if gateway_dir.name != "gateway":
            continue
        post_processed_dir = gateway_dir.parent
        if post_processed_dir.name != "post-processed":
            continue
        run_dirs.add(post_processed_dir.parent.resolve())
    return sorted(run_dirs)


def _sample_count(total_duration_s: float, *, timepoint_freq_hz: float) -> int:
    if total_duration_s <= 0:
        return 0
    return max(1, int(math.ceil((total_duration_s * timepoint_freq_hz) - 1e-12)))


def _sampled_total_duration_s(
    observed_offsets_s: list[float],
    *,
    timepoint_freq_hz: float,
) -> float:
    if not observed_offsets_s:
        return 0.0
    sample_interval_s = 1.0 / timepoint_freq_hz
    last_observed_offset_s = max(offset for offset in observed_offsets_s if offset >= 0.0)
    last_sample_index = max(0, int(math.floor((last_observed_offset_s * timepoint_freq_hz) + 1e-12)))
    return round(max(sample_interval_s, (last_sample_index + 1) * sample_interval_s), 6)


def _build_throughput_points(
    *,
    completion_offsets_s: list[float],
    total_duration_s: float,
    timepoint_freq_hz: float,
    window_size_s: float,
) -> list[dict[str, float]]:
    sorted_offsets = sorted(offset for offset in completion_offsets_s if offset >= 0.0)
    sample_interval_s = 1.0 / timepoint_freq_hz
    points: list[dict[str, float]] = []

    for sample_index in range(_sample_count(total_duration_s, timepoint_freq_hz=timepoint_freq_hz)):
        time_s = round(sample_index * sample_interval_s, 6)
        window_start_s = max(0.0, time_s - window_size_s)
        window_end_s = min(total_duration_s, time_s + window_size_s)
        window_duration_s = round(window_end_s - window_start_s, 6)

        throughput_requests_per_s = 0.0
        if window_duration_s > 0:
            start_index = bisect_left(sorted_offsets, window_start_s)
            end_index = bisect_right(sorted_offsets, window_end_s)
            throughput_requests_per_s = round(
                (end_index - start_index) / window_duration_s,
                6,
            )

        points.append(
            {
                "time_s": time_s,
                "throughput_requests_per_s": throughput_requests_per_s,
            }
        )

    return points


def _load_request_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing llm-requests file: {path}")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"llm-requests payload must be a JSON object: {path}")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        raise ValueError(f"llm-requests payload missing 'requests' list: {path}")
    return [item for item in raw_requests if isinstance(item, dict)], payload


def _gateway_profile_id_or_none(record: dict[str, Any]) -> int | None:
    return int_or_none(record.get("gateway_profile_id"))


def _summarize_request_records(
    request_records: list[dict[str, Any]],
    *,
    total_duration_s: float,
    timepoint_freq_hz: float,
    window_size_s: float,
) -> dict[str, Any]:
    completed_offsets_s: list[float] = []
    completed_offsets_s_status_200: list[float] = []
    first_request_start_s: float | None = None
    last_request_end_s: float | None = None

    for record in request_records:
        request_start_offset_s = _float_or_none(record.get("request_start_offset_s"))
        request_end_offset_s = _float_or_none(record.get("request_end_offset_s"))
        if request_start_offset_s is not None and request_start_offset_s >= 0.0:
            if first_request_start_s is None or request_start_offset_s < first_request_start_s:
                first_request_start_s = request_start_offset_s
        if request_end_offset_s is None or request_end_offset_s < 0.0:
            continue
        completed_offsets_s.append(request_end_offset_s)
        if last_request_end_s is None or request_end_offset_s > last_request_end_s:
            last_request_end_s = request_end_offset_s
        if _status_code_int_or_none(record.get("status_code")) == 200:
            completed_offsets_s_status_200.append(request_end_offset_s)

    filtered_completed_offsets_s = [
        offset for offset in completed_offsets_s if 0.0 <= offset <= total_duration_s
    ]
    filtered_completed_offsets_s_status_200 = [
        offset
        for offset in completed_offsets_s_status_200
        if 0.0 <= offset <= total_duration_s
    ]
    throughput_points = _build_throughput_points(
        completion_offsets_s=filtered_completed_offsets_s,
        total_duration_s=total_duration_s,
        timepoint_freq_hz=timepoint_freq_hz,
        window_size_s=window_size_s,
    )
    throughput_points_status_200 = _build_throughput_points(
        completion_offsets_s=filtered_completed_offsets_s_status_200,
        total_duration_s=total_duration_s,
        timepoint_freq_hz=timepoint_freq_hz,
        window_size_s=window_size_s,
    )

    return {
        "request_count": len(request_records),
        "finished_request_count": len(filtered_completed_offsets_s),
        "finished_request_count_status_200": len(filtered_completed_offsets_s_status_200),
        "non_200_finished_request_count": (
            len(filtered_completed_offsets_s) - len(filtered_completed_offsets_s_status_200)
        ),
        "first_request_start_s": first_request_start_s,
        "last_request_end_s": last_request_end_s,
        "sample_count": len(throughput_points),
        "throughput_points": throughput_points,
        "throughput_points_status_200": throughput_points_status_200,
    }


def extract_request_throughput_from_run_dir(
    run_dir: Path,
    *,
    llm_requests_path: Path | None = None,
    timepoint_freq_hz: float = DEFAULT_TIMEPOINT_FREQ_HZ,
    window_size_s: float = DEFAULT_WINDOW_SIZE_S,
) -> dict[str, Any]:
    if timepoint_freq_hz <= 0:
        raise ValueError(
            f"timepoint_freq_hz must be a positive number: {timepoint_freq_hz}"
        )
    if window_size_s <= 0:
        raise ValueError(f"window_size_s must be a positive number: {window_size_s}")

    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_llm_requests_path = (
        llm_requests_path or _default_llm_requests_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    request_records, payload = _load_request_records(resolved_llm_requests_path)

    observed_offsets_s: list[float] = []

    for record in request_records:
        request_start_offset_s = _float_or_none(record.get("request_start_offset_s"))
        request_end_offset_s = _float_or_none(record.get("request_end_offset_s"))
        if request_start_offset_s is not None and request_start_offset_s >= 0.0:
            observed_offsets_s.append(request_start_offset_s)
        if request_end_offset_s is not None and request_end_offset_s >= 0.0:
            observed_offsets_s.append(request_end_offset_s)

    total_duration_s = _sampled_total_duration_s(
        observed_offsets_s,
        timepoint_freq_hz=timepoint_freq_hz,
    )

    source_gateway_output_dir = payload.get("source_gateway_output_dir")
    if not isinstance(source_gateway_output_dir, str) or not source_gateway_output_dir.strip():
        source_gateway_output_dir = str((resolved_run_dir / "gateway-output").resolve())

    common = {
        "source_run_dir": str(resolved_run_dir),
        "source_llm_requests_path": str(resolved_llm_requests_path),
        "source_gateway_output_dir": source_gateway_output_dir,
        "service_failure_detected": bool(payload.get("service_failure_detected", False)),
        "service_failure_cutoff_time_utc": payload.get("service_failure_cutoff_time_utc"),
        "total_duration_s": total_duration_s,
        "timepoint_frequency_hz": timepoint_freq_hz,
        "timepoint_interval_s": round(1.0 / timepoint_freq_hz, 6),
        "window_size_s": window_size_s,
        "window_width_s": round(window_size_s * 2.0, 6),
    }
    aggregate_summary = _summarize_request_records(
        request_records,
        total_duration_s=total_duration_s,
        timepoint_freq_hz=timepoint_freq_hz,
        window_size_s=window_size_s,
    )
    port_profile_ids = sorted(
        set(profile_ids_from_payload(payload))
        | {
            profile_id
            for profile_id in (_gateway_profile_id_or_none(record) for record in request_records)
            if profile_id is not None
        }
    )
    series_by_profile: dict[str, dict[str, Any]] = {}
    for gateway_profile_id in port_profile_ids:
        series_key = profile_label(gateway_profile_id)
        profile_records = [
            record
            for record in request_records
            if _gateway_profile_id_or_none(record) == gateway_profile_id
        ]
        series_by_profile[series_key] = {
            **common,
            "gateway_profile_id": gateway_profile_id,
            **_summarize_request_records(
                profile_records,
                total_duration_s=total_duration_s,
                timepoint_freq_hz=timepoint_freq_hz,
                window_size_s=window_size_s,
            ),
        }

    return {
        **common,
        **aggregate_summary,
        "multi_profile": len(port_profile_ids) > 1,
        "port_profile_ids": port_profile_ids,
        "series_keys": list(series_by_profile.keys()),
        "series_by_profile": series_by_profile,
    }


def extract_run_dir(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    llm_requests_path: Path | None = None,
    timepoint_freq_hz: float = DEFAULT_TIMEPOINT_FREQ_HZ,
    window_size_s: float = DEFAULT_WINDOW_SIZE_S,
) -> Path:
    resolved_output_path = (output_path or _default_output_path_for_run(run_dir)).expanduser().resolve()
    result = extract_request_throughput_from_run_dir(
        run_dir,
        llm_requests_path=llm_requests_path,
        timepoint_freq_hz=timepoint_freq_hz,
        window_size_s=window_size_s,
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def _extract_run_dir_worker(job: tuple[str, float, float]) -> tuple[str, str | None, str | None]:
    run_dir_text, timepoint_freq_hz, window_size_s = job
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = extract_run_dir(
            run_dir,
            timepoint_freq_hz=timepoint_freq_hz,
            window_size_s=window_size_s,
        )
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(
    run_dirs: list[Path],
    *,
    timepoint_freq_hz: float,
    window_size_s: float,
) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = extract_run_dir(
                run_dir,
                timepoint_freq_hz=timepoint_freq_hz,
                window_size_s=window_size_s,
            )
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(
    run_dirs: list[Path],
    *,
    max_procs: int,
    timepoint_freq_hz: float,
    window_size_s: float,
) -> int:
    failure_count = 0
    jobs = [
        (str(run_dir), timepoint_freq_hz, window_size_s)
        for run_dir in run_dirs
    ]
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
    if args.timepoint_freq_hz <= 0:
        raise ValueError(
            f"--timepoint-freq-hz must be a positive number: {args.timepoint_freq_hz}"
        )
    if args.window_size_s <= 0:
        raise ValueError(f"--window-size-s must be a positive number: {args.window_size_s}")
    run_path = Path(args.run_dir).expanduser().resolve()
    if not run_path.exists():
        raise ValueError(f"Run directory not found: {run_path}")
    if not run_path.is_dir():
        raise ValueError(
            f"--run-dir must point to a directory, got file: {run_path}. "
            f"If you want to process many runs, use --root-dir {run_path.parent}"
        )
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    llm_requests_path = Path(args.llm_requests).expanduser().resolve() if args.llm_requests else None
    resolved_output_path = extract_run_dir(
        run_path,
        output_path=output_path,
        llm_requests_path=llm_requests_path,
        timepoint_freq_hz=args.timepoint_freq_hz,
        window_size_s=args.window_size_s,
    )
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.llm_requests:
        raise ValueError("--llm-requests can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.timepoint_freq_hz <= 0:
        raise ValueError(
            f"--timepoint-freq-hz must be a positive number: {args.timepoint_freq_hz}"
        )
    if args.window_size_s <= 0:
        raise ValueError(f"--window-size-s must be a positive number: {args.window_size_s}")
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
        failure_count = _run_root_dir_sequential(
            run_dirs,
            timepoint_freq_hz=args.timepoint_freq_hz,
            window_size_s=args.window_size_s,
        )
    else:
        try:
            failure_count = _run_root_dir_parallel(
                run_dirs,
                max_procs=worker_count,
                timepoint_freq_hz=args.timepoint_freq_hz,
                window_size_s=args.window_size_s,
            )
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(
                run_dirs,
                timepoint_freq_hz=args.timepoint_freq_hz,
                window_size_s=args.window_size_s,
            )

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
