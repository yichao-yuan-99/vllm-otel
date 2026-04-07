from __future__ import annotations

import argparse
from bisect import bisect_left
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
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

from pp_common.service_failure import cutoff_datetime_utc_from_payload
from pp_common.service_failure import ensure_service_failure_payload
from pp_common.service_failure import parse_iso8601_to_utc
from pp_common.profile_id import api_token_sha256
from pp_common.profile_id import build_gateway_run_profile_id_by_api_token_hash
from pp_common.profile_id import profile_ids_from_payload
from pp_common.profile_id import profile_label


DEFAULT_OUTPUT_NAME = "job-throughput-timeseries.json"
DEFAULT_TIMEPOINT_FREQ_HZ = 1.0
DEFAULT_WINDOW_SIZE_S = 600.0


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
            "Extract moving replay/job throughput (jobs per second) for one run "
            "or for all runs under a root directory."
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
            "with replay/summary.json or meta/results.json + meta/run_manifest.json "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/post-processed/job-throughput/"
            f"{DEFAULT_OUTPUT_NAME}"
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
        return float(value)
    return None


def _non_negative_float_or_none(value: Any) -> float | None:
    numeric = _float_or_none(value)
    if numeric is None or numeric < 0:
        return None
    return numeric


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


def _is_cancelled_status(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"cancelled", "canceled"}


def _gateway_profile_id_from_job_payload(
    payload: Any,
    *,
    profile_id_by_api_token_hash: dict[str, int],
) -> int | None:
    declared_profile_ids = profile_ids_from_payload(payload)
    if declared_profile_ids:
        return declared_profile_ids[0]
    if not isinstance(payload, dict):
        return None
    api_token_hash = api_token_sha256(payload.get("api_token"))
    if api_token_hash is None:
        return None
    return profile_id_by_api_token_hash.get(api_token_hash)


def _resolved_total_duration_s(
    total_duration_s: float | None,
    *,
    completion_offsets_s: list[float],
    time_constraint_s: float | None,
) -> float:
    resolved_total_duration_s = max(total_duration_s or 0.0, 0.0)
    if completion_offsets_s:
        resolved_total_duration_s = max(resolved_total_duration_s, max(completion_offsets_s))
    if time_constraint_s is not None:
        resolved_total_duration_s = min(resolved_total_duration_s, time_constraint_s)
    return round(resolved_total_duration_s, 6)


def _extract_completion_offsets_from_replay_run(
    run_dir: Path,
    *,
    cutoff_time_utc: datetime | None = None,
    profile_id_by_api_token_hash: dict[str, int] | None = None,
) -> tuple[str, Any, Any, float | None, int, list[dict[str, Any]], float, list[int], dict[int, int]]:
    summary_path = run_dir / "replay" / "summary.json"
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Replay summary must be a JSON object: {summary_path}")

    experiment_started_at = payload.get("started_at")
    experiment_finished_at = payload.get("finished_at")
    experiment_start_dt = _parse_iso8601(experiment_started_at)
    experiment_start_dt_utc = parse_iso8601_to_utc(experiment_started_at)
    experiment_finish_dt = _parse_iso8601(experiment_finished_at)
    if experiment_start_dt is None:
        raise ValueError(f"Invalid or missing started_at in replay summary: {summary_path}")

    worker_results = payload.get("worker_results")
    if not isinstance(worker_results, dict):
        raise ValueError(f"Replay summary is missing object field worker_results: {summary_path}")

    replay_count = 0
    finish_events: list[dict[str, Any]] = []
    replay_count_by_profile: dict[int, int] = {}
    for worker_payload in worker_results.values():
        if not isinstance(worker_payload, dict):
            continue
        replay_count += 1
        gateway_profile_id = _gateway_profile_id_from_job_payload(
            worker_payload,
            profile_id_by_api_token_hash=profile_id_by_api_token_hash or {},
        )
        if gateway_profile_id is not None:
            replay_count_by_profile[gateway_profile_id] = (
                replay_count_by_profile.get(gateway_profile_id, 0) + 1
            )
        finish_dt = _parse_iso8601(worker_payload.get("finished_at"))
        finish_dt_utc = parse_iso8601_to_utc(worker_payload.get("finished_at"))
        if cutoff_time_utc is not None and finish_dt_utc is not None and finish_dt_utc > cutoff_time_utc:
            continue
        finish_offset_s = _duration_seconds(experiment_start_dt, finish_dt)
        if finish_offset_s is not None:
            finish_events.append(
                {
                    "finish_offset_s": finish_offset_s,
                    "status": worker_payload.get("status"),
                    "gateway_profile_id": gateway_profile_id,
                }
            )

    total_duration_s = _float_or_none(payload.get("run_duration_s"))
    if total_duration_s is None:
        total_duration_s = _duration_seconds(experiment_start_dt, experiment_finish_dt)
    cutoff_offset_s = _duration_seconds(experiment_start_dt_utc, cutoff_time_utc)
    if cutoff_offset_s is not None:
        cutoff_offset_s = max(cutoff_offset_s, 0.0)
    time_constraint_s = _non_negative_float_or_none(payload.get("time_constraint_s"))
    if cutoff_offset_s is not None:
        if time_constraint_s is None:
            time_constraint_s = cutoff_offset_s
        else:
            time_constraint_s = min(time_constraint_s, cutoff_offset_s)
    completion_offsets_s = [
        event["finish_offset_s"]
        for event in finish_events
        if isinstance(event.get("finish_offset_s"), (int, float))
    ]
    return (
        "replay",
        experiment_started_at,
        experiment_finished_at,
        time_constraint_s,
        replay_count,
        finish_events,
        _resolved_total_duration_s(
            total_duration_s,
            completion_offsets_s=completion_offsets_s,
            time_constraint_s=time_constraint_s,
        ),
        profile_ids_from_payload(payload),
        replay_count_by_profile,
    )


def _extract_completion_offsets_from_con_driver_run(
    run_dir: Path,
    *,
    cutoff_time_utc: datetime | None = None,
    profile_id_by_api_token_hash: dict[str, int] | None = None,
) -> tuple[str, Any, Any, float | None, int, list[dict[str, Any]], float, list[int], dict[int, int]]:
    manifest_path = run_dir / "meta" / "run_manifest.json"
    results_path = run_dir / "meta" / "results.json"

    manifest_payload = _load_json(manifest_path)
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"Con-driver run_manifest must be a JSON object: {manifest_path}")
    results_payload = _load_json(results_path)
    if not isinstance(results_payload, list):
        raise ValueError(f"Con-driver results must be a JSON array: {results_path}")

    experiment_started_at = manifest_payload.get("started_at")
    experiment_finished_at = manifest_payload.get("finished_at")
    experiment_start_dt = _parse_iso8601(experiment_started_at)
    experiment_start_dt_utc = parse_iso8601_to_utc(experiment_started_at)
    experiment_finish_dt = _parse_iso8601(experiment_finished_at)
    if experiment_start_dt is None:
        raise ValueError(f"Invalid or missing started_at in run_manifest: {manifest_path}")

    replay_count = 0
    finish_events: list[dict[str, Any]] = []
    replay_count_by_profile: dict[int, int] = {}
    for entry in results_payload:
        if not isinstance(entry, dict):
            continue
        replay_count += 1
        gateway_profile_id = _gateway_profile_id_from_job_payload(
            entry,
            profile_id_by_api_token_hash=profile_id_by_api_token_hash or {},
        )
        if gateway_profile_id is not None:
            replay_count_by_profile[gateway_profile_id] = (
                replay_count_by_profile.get(gateway_profile_id, 0) + 1
            )
        finish_dt = _parse_iso8601(entry.get("finished_at"))
        finish_dt_utc = parse_iso8601_to_utc(entry.get("finished_at"))
        if cutoff_time_utc is not None and finish_dt_utc is not None and finish_dt_utc > cutoff_time_utc:
            continue
        finish_offset_s = _duration_seconds(experiment_start_dt, finish_dt)
        if finish_offset_s is not None:
            finish_events.append(
                {
                    "finish_offset_s": finish_offset_s,
                    "status": entry.get("status"),
                    "gateway_profile_id": gateway_profile_id,
                }
            )

    total_duration_s = _float_or_none(manifest_payload.get("run_duration_s"))
    if total_duration_s is None:
        total_duration_s = _duration_seconds(experiment_start_dt, experiment_finish_dt)
    cutoff_offset_s = _duration_seconds(experiment_start_dt_utc, cutoff_time_utc)
    if cutoff_offset_s is not None:
        cutoff_offset_s = max(cutoff_offset_s, 0.0)
    time_constraint_s = _non_negative_float_or_none(manifest_payload.get("time_constraint_s"))
    if cutoff_offset_s is not None:
        if time_constraint_s is None:
            time_constraint_s = cutoff_offset_s
        else:
            time_constraint_s = min(time_constraint_s, cutoff_offset_s)
    completion_offsets_s = [
        event["finish_offset_s"]
        for event in finish_events
        if isinstance(event.get("finish_offset_s"), (int, float))
    ]
    return (
        "con-driver",
        experiment_started_at,
        experiment_finished_at,
        time_constraint_s,
        replay_count,
        finish_events,
        _resolved_total_duration_s(
            total_duration_s,
            completion_offsets_s=completion_offsets_s,
            time_constraint_s=time_constraint_s,
        ),
        profile_ids_from_payload(manifest_payload),
        replay_count_by_profile,
    )


def _sample_count(total_duration_s: float, *, timepoint_freq_hz: float) -> int:
    if total_duration_s <= 0:
        return 0
    return max(1, int(math.ceil((total_duration_s * timepoint_freq_hz) - 1e-12)))


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

        throughput_jobs_per_s = 0.0
        if window_duration_s > 0:
            start_index = bisect_left(sorted_offsets, window_start_s)
            end_index = bisect_right(sorted_offsets, window_end_s)
            throughput_jobs_per_s = round(
                (end_index - start_index) / window_duration_s,
                6,
            )

        points.append(
            {
                "time_s": time_s,
                "throughput_jobs_per_s": throughput_jobs_per_s,
            }
        )

    return points


def _build_job_throughput_summary(
    finish_events: list[dict[str, Any]],
    *,
    total_duration_s: float,
    timepoint_freq_hz: float,
    window_size_s: float,
) -> dict[str, Any]:
    filtered_finish_events = [
        event
        for event in finish_events
        if isinstance(event, dict)
        and isinstance(event.get("finish_offset_s"), (int, float))
        and 0.0 <= float(event["finish_offset_s"]) <= total_duration_s
    ]
    completion_offsets_s = [
        float(event["finish_offset_s"])
        for event in filtered_finish_events
    ]
    completion_offsets_s_excluding_cancelled = [
        float(event["finish_offset_s"])
        for event in filtered_finish_events
        if not _is_cancelled_status(event.get("status"))
    ]
    throughput_points = _build_throughput_points(
        completion_offsets_s=completion_offsets_s,
        total_duration_s=total_duration_s,
        timepoint_freq_hz=timepoint_freq_hz,
        window_size_s=window_size_s,
    )
    throughput_points_excluding_cancelled = _build_throughput_points(
        completion_offsets_s=completion_offsets_s_excluding_cancelled,
        total_duration_s=total_duration_s,
        timepoint_freq_hz=timepoint_freq_hz,
        window_size_s=window_size_s,
    )
    return {
        "finished_replay_count": len(completion_offsets_s),
        "finished_replay_count_excluding_cancelled": len(
            completion_offsets_s_excluding_cancelled
        ),
        "cancelled_finished_replay_count": (
            len(completion_offsets_s) - len(completion_offsets_s_excluding_cancelled)
        ),
        "sample_count": len(throughput_points),
        "throughput_points": throughput_points,
        "throughput_points_excluding_cancelled": throughput_points_excluding_cancelled,
    }


def extract_job_throughput_from_run_dir(
    run_dir: Path,
    *,
    timepoint_freq_hz: float = DEFAULT_TIMEPOINT_FREQ_HZ,
    window_size_s: float = DEFAULT_WINDOW_SIZE_S,
) -> dict[str, Any]:
    if timepoint_freq_hz <= 0:
        raise ValueError(
            f"timepoint_freq_hz must be a positive number: {timepoint_freq_hz}"
        )
    if window_size_s <= 0:
        raise ValueError(f"window_size_s must be a positive number: {window_size_s}")

    service_failure_payload = ensure_service_failure_payload(run_dir)
    cutoff_time_utc = cutoff_datetime_utc_from_payload(service_failure_payload)
    profile_id_by_api_token_hash = build_gateway_run_profile_id_by_api_token_hash(run_dir)

    replay_summary_path = run_dir / "replay" / "summary.json"
    con_driver_results_path = run_dir / "meta" / "results.json"
    con_driver_manifest_path = run_dir / "meta" / "run_manifest.json"

    if replay_summary_path.is_file():
        (
            source_type,
            experiment_started_at,
            experiment_finished_at,
            time_constraint_s,
            replay_count,
            finish_events,
            total_duration_s,
            declared_port_profile_ids,
            replay_count_by_profile,
        ) = _extract_completion_offsets_from_replay_run(
            run_dir,
            cutoff_time_utc=cutoff_time_utc,
            profile_id_by_api_token_hash=profile_id_by_api_token_hash,
        )
    elif con_driver_results_path.is_file() and con_driver_manifest_path.is_file():
        (
            source_type,
            experiment_started_at,
            experiment_finished_at,
            time_constraint_s,
            replay_count,
            finish_events,
            total_duration_s,
            declared_port_profile_ids,
            replay_count_by_profile,
        ) = _extract_completion_offsets_from_con_driver_run(
            run_dir,
            cutoff_time_utc=cutoff_time_utc,
            profile_id_by_api_token_hash=profile_id_by_api_token_hash,
        )
    else:
        raise ValueError(
            "Unrecognized run layout. Expected either replay/summary.json "
            "or meta/results.json + meta/run_manifest.json."
        )

    common = {
        "source_run_dir": str(run_dir.resolve()),
        "source_type": source_type,
        "experiment_started_at": experiment_started_at,
        "experiment_finished_at": experiment_finished_at,
        "time_constraint_s": time_constraint_s,
        "total_duration_s": total_duration_s,
        "timepoint_frequency_hz": timepoint_freq_hz,
        "timepoint_interval_s": round(1.0 / timepoint_freq_hz, 6),
        "window_size_s": window_size_s,
        "window_width_s": round(window_size_s * 2.0, 6),
        "service_failure_detected": bool(
            service_failure_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": service_failure_payload.get("cutoff_time_utc"),
    }
    aggregate_summary = _build_job_throughput_summary(
        finish_events,
        total_duration_s=total_duration_s,
        timepoint_freq_hz=timepoint_freq_hz,
        window_size_s=window_size_s,
    )
    port_profile_ids = sorted(
        set(declared_port_profile_ids)
        | {
            int(event["gateway_profile_id"])
            for event in finish_events
            if isinstance(event, dict) and isinstance(event.get("gateway_profile_id"), int)
        }
    )
    series_by_profile: dict[str, dict[str, Any]] = {}
    for gateway_profile_id in port_profile_ids:
        profile_finish_events = [
            event
            for event in finish_events
            if isinstance(event, dict) and event.get("gateway_profile_id") == gateway_profile_id
        ]
        series_by_profile[profile_label(gateway_profile_id)] = {
            **common,
            "gateway_profile_id": gateway_profile_id,
            "replay_count": replay_count_by_profile.get(gateway_profile_id, 0),
            **_build_job_throughput_summary(
                profile_finish_events,
                total_duration_s=total_duration_s,
                timepoint_freq_hz=timepoint_freq_hz,
                window_size_s=window_size_s,
            ),
        }
    return {
        **common,
        "replay_count": replay_count,
        **aggregate_summary,
        "multi_profile": len(port_profile_ids) > 1,
        "port_profile_ids": port_profile_ids,
        "series_keys": list(series_by_profile.keys()),
        "series_by_profile": series_by_profile,
    }


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "job-throughput" / DEFAULT_OUTPUT_NAME).resolve()


def discover_run_dirs_with_job_throughput_sources(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()

    for summary_path in root_dir.rglob("summary.json"):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name != "replay":
            continue
        run_dirs.add(summary_path.parent.parent.resolve())

    for results_path in root_dir.rglob("results.json"):
        if not results_path.is_file():
            continue
        if results_path.parent.name != "meta":
            continue
        manifest_path = results_path.parent / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        run_dirs.add(results_path.parent.parent.resolve())

    return sorted(run_dirs)


def extract_run_dir(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    timepoint_freq_hz: float = DEFAULT_TIMEPOINT_FREQ_HZ,
    window_size_s: float = DEFAULT_WINDOW_SIZE_S,
) -> Path:
    resolved_output_path = (output_path or _default_output_path_for_run(run_dir)).expanduser().resolve()
    result = extract_job_throughput_from_run_dir(
        run_dir,
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
    resolved_output_path = extract_run_dir(
        run_path,
        output_path=output_path,
        timepoint_freq_hz=args.timepoint_freq_hz,
        window_size_s=args.window_size_s,
    )
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
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

    run_dirs = discover_run_dirs_with_job_throughput_sources(root_dir)
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
