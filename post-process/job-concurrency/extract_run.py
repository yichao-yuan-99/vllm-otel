from __future__ import annotations

import argparse
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


DEFAULT_OUTPUT_NAME = "job-concurrency-timeseries.json"


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
            "Extract per-second job concurrency from trial start/end times for one run "
            "or all runs under a root directory."
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
            "Optional output path. Default: <run-dir>/post-processed/job-concurrency/"
            f"{DEFAULT_OUTPUT_NAME}"
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


def _resolved_total_duration_s(
    total_duration_s: float | None,
    *,
    end_offsets_s: list[float],
    time_constraint_s: float | None,
) -> float:
    resolved_total_duration_s = max(total_duration_s or 0.0, 0.0)
    if end_offsets_s:
        resolved_total_duration_s = max(resolved_total_duration_s, max(end_offsets_s))
    if time_constraint_s is not None:
        resolved_total_duration_s = min(resolved_total_duration_s, time_constraint_s)
    return round(resolved_total_duration_s, 6)


def _clip_time_constraint_s(
    time_constraint_s: float | None,
    *,
    experiment_started_at: Any,
    cutoff_time_utc: datetime | None,
) -> float | None:
    experiment_start_dt_utc = parse_iso8601_to_utc(experiment_started_at)
    cutoff_offset_s = _duration_seconds(experiment_start_dt_utc, cutoff_time_utc)
    if cutoff_offset_s is not None:
        cutoff_offset_s = max(cutoff_offset_s, 0.0)
    if cutoff_offset_s is None:
        return time_constraint_s
    if time_constraint_s is None:
        return cutoff_offset_s
    return min(time_constraint_s, cutoff_offset_s)


def _build_interval_entry(
    *,
    job_id: Any,
    status: Any,
    start_offset_s: float,
    end_offset_s: float,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": status,
        "start_offset_s": start_offset_s,
        "end_offset_s": end_offset_s,
        "duration_s": round(end_offset_s - start_offset_s, 6),
    }


def _extract_job_intervals_from_replay_run(
    run_dir: Path,
    *,
    cutoff_time_utc: datetime | None = None,
) -> tuple[str, Any, Any, float | None, int, list[dict[str, Any]], float]:
    summary_path = run_dir / "replay" / "summary.json"
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Replay summary must be a JSON object: {summary_path}")

    experiment_started_at = payload.get("started_at")
    experiment_finished_at = payload.get("finished_at")
    experiment_start_dt = _parse_iso8601(experiment_started_at)
    if experiment_start_dt is None:
        raise ValueError(f"Invalid or missing started_at in replay summary: {summary_path}")
    experiment_finish_dt = _parse_iso8601(experiment_finished_at)

    worker_results = payload.get("worker_results")
    if not isinstance(worker_results, dict):
        raise ValueError(f"Replay summary is missing object field worker_results: {summary_path}")

    replay_count = 0
    intervals: list[dict[str, Any]] = []
    for worker_id, worker_payload in worker_results.items():
        if not isinstance(worker_payload, dict):
            continue
        replay_count += 1

        start_dt = _parse_iso8601(worker_payload.get("started_at"))
        finish_dt = _parse_iso8601(worker_payload.get("finished_at"))
        start_offset_s = _duration_seconds(experiment_start_dt, start_dt)
        end_offset_s = _duration_seconds(experiment_start_dt, finish_dt)
        if start_offset_s is None or end_offset_s is None:
            continue
        intervals.append(
            _build_interval_entry(
                job_id=worker_payload.get("worker_id") or worker_id,
                status=worker_payload.get("status"),
                start_offset_s=start_offset_s,
                end_offset_s=end_offset_s,
            )
        )

    total_duration_s = _float_or_none(payload.get("run_duration_s"))
    if total_duration_s is None:
        total_duration_s = _duration_seconds(experiment_start_dt, experiment_finish_dt)

    time_constraint_s = _non_negative_float_or_none(payload.get("time_constraint_s"))
    time_constraint_s = _clip_time_constraint_s(
        time_constraint_s,
        experiment_started_at=experiment_started_at,
        cutoff_time_utc=cutoff_time_utc,
    )
    end_offsets_s = [
        float(interval["end_offset_s"])
        for interval in intervals
        if isinstance(interval.get("end_offset_s"), (int, float))
    ]
    return (
        "replay",
        experiment_started_at,
        experiment_finished_at,
        time_constraint_s,
        replay_count,
        intervals,
        _resolved_total_duration_s(
            total_duration_s,
            end_offsets_s=end_offsets_s,
            time_constraint_s=time_constraint_s,
        ),
    )


def _extract_job_intervals_from_con_driver_run(
    run_dir: Path,
    *,
    cutoff_time_utc: datetime | None = None,
) -> tuple[str, Any, Any, float | None, int, list[dict[str, Any]], float]:
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
    if experiment_start_dt is None:
        raise ValueError(f"Invalid or missing started_at in run_manifest: {manifest_path}")
    experiment_finish_dt = _parse_iso8601(experiment_finished_at)

    replay_count = 0
    intervals: list[dict[str, Any]] = []
    for entry in results_payload:
        if not isinstance(entry, dict):
            continue
        replay_count += 1
        start_dt = _parse_iso8601(entry.get("started_at"))
        finish_dt = _parse_iso8601(entry.get("finished_at"))
        start_offset_s = _duration_seconds(experiment_start_dt, start_dt)
        end_offset_s = _duration_seconds(experiment_start_dt, finish_dt)
        if start_offset_s is None or end_offset_s is None:
            continue
        intervals.append(
            _build_interval_entry(
                job_id=entry.get("trial_id") or entry.get("task_path"),
                status=entry.get("status"),
                start_offset_s=start_offset_s,
                end_offset_s=end_offset_s,
            )
        )

    total_duration_s = _float_or_none(manifest_payload.get("run_duration_s"))
    if total_duration_s is None:
        total_duration_s = _duration_seconds(experiment_start_dt, experiment_finish_dt)

    time_constraint_s = _non_negative_float_or_none(manifest_payload.get("time_constraint_s"))
    time_constraint_s = _clip_time_constraint_s(
        time_constraint_s,
        experiment_started_at=experiment_started_at,
        cutoff_time_utc=cutoff_time_utc,
    )
    end_offsets_s = [
        float(interval["end_offset_s"])
        for interval in intervals
        if isinstance(interval.get("end_offset_s"), (int, float))
    ]
    return (
        "con-driver",
        experiment_started_at,
        experiment_finished_at,
        time_constraint_s,
        replay_count,
        intervals,
        _resolved_total_duration_s(
            total_duration_s,
            end_offsets_s=end_offsets_s,
            time_constraint_s=time_constraint_s,
        ),
    )


def _interval_sort_key(interval: dict[str, Any]) -> tuple[float, float, str]:
    start_offset_s = _float_or_none(interval.get("start_offset_s"))
    end_offset_s = _float_or_none(interval.get("end_offset_s"))
    return (
        start_offset_s if start_offset_s is not None else float("inf"),
        end_offset_s if end_offset_s is not None else float("inf"),
        str(interval.get("job_id") or ""),
    )


def _clip_job_intervals(
    intervals: list[dict[str, Any]],
    *,
    total_duration_s: float,
) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for interval in intervals:
        start_offset_s = _float_or_none(interval.get("start_offset_s"))
        end_offset_s = _float_or_none(interval.get("end_offset_s"))
        if start_offset_s is None or end_offset_s is None:
            continue
        clipped_start_s = min(max(start_offset_s, 0.0), total_duration_s)
        clipped_end_s = min(max(end_offset_s, 0.0), total_duration_s)
        if clipped_end_s <= clipped_start_s:
            continue
        clipped.append(
            {
                **interval,
                "start_offset_s": round(clipped_start_s, 6),
                "end_offset_s": round(clipped_end_s, 6),
                "duration_s": round(clipped_end_s - clipped_start_s, 6),
            }
        )
    clipped.sort(key=_interval_sort_key)
    return clipped


def _sample_count(total_duration_s: float) -> int:
    if total_duration_s <= 0:
        return 0
    return max(1, int(math.ceil(total_duration_s - 1e-12)))


def _build_concurrency_points(
    *,
    job_intervals: list[dict[str, Any]],
    total_duration_s: float,
) -> list[dict[str, int]]:
    samples = _sample_count(total_duration_s)
    if samples <= 0:
        return []

    deltas = [0] * (samples + 1)
    for interval in job_intervals:
        start_offset_s = _float_or_none(interval.get("start_offset_s"))
        end_offset_s = _float_or_none(interval.get("end_offset_s"))
        if start_offset_s is None or end_offset_s is None:
            continue
        start_second = int(math.floor(start_offset_s))
        end_second = int(math.ceil(end_offset_s))
        start_second = min(max(start_second, 0), samples)
        end_second = min(max(end_second, 0), samples)
        if end_second <= start_second:
            continue
        deltas[start_second] += 1
        deltas[end_second] -= 1

    points: list[dict[str, int]] = []
    active = 0
    for second in range(samples):
        active += deltas[second]
        points.append({"second": second, "concurrency": active})
    return points


def extract_job_concurrency_from_run_dir(run_dir: Path) -> dict[str, Any]:
    service_failure_payload = ensure_service_failure_payload(run_dir)
    cutoff_time_utc = cutoff_datetime_utc_from_payload(service_failure_payload)

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
            raw_intervals,
            total_duration_s,
        ) = _extract_job_intervals_from_replay_run(
            run_dir,
            cutoff_time_utc=cutoff_time_utc,
        )
    elif con_driver_results_path.is_file() and con_driver_manifest_path.is_file():
        (
            source_type,
            experiment_started_at,
            experiment_finished_at,
            time_constraint_s,
            replay_count,
            raw_intervals,
            total_duration_s,
        ) = _extract_job_intervals_from_con_driver_run(
            run_dir,
            cutoff_time_utc=cutoff_time_utc,
        )
    else:
        raise ValueError(
            "Unrecognized run layout. Expected either replay/summary.json "
            "or meta/results.json + meta/run_manifest.json."
        )

    job_intervals = _clip_job_intervals(raw_intervals, total_duration_s=total_duration_s)
    concurrency_points = _build_concurrency_points(
        job_intervals=job_intervals,
        total_duration_s=total_duration_s,
    )
    max_concurrency = max(
        (point["concurrency"] for point in concurrency_points),
        default=0,
    )
    avg_concurrency = (
        round(
            sum(point["concurrency"] for point in concurrency_points) / len(concurrency_points),
            6,
        )
        if concurrency_points
        else 0.0
    )

    return {
        "source_run_dir": str(run_dir.resolve()),
        "source_type": source_type,
        "experiment_started_at": experiment_started_at,
        "experiment_finished_at": experiment_finished_at,
        "time_constraint_s": time_constraint_s,
        "service_failure_detected": bool(
            service_failure_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": service_failure_payload.get("cutoff_time_utc"),
        "replay_count": replay_count,
        "jobs_with_valid_range_count": len(job_intervals),
        "total_duration_s": total_duration_s,
        "sample_count": len(concurrency_points),
        "max_concurrency": max_concurrency,
        "avg_concurrency": avg_concurrency,
        "concurrency_points": concurrency_points,
        "job_intervals_preview": job_intervals[:20],
    }


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "job-concurrency" / DEFAULT_OUTPUT_NAME).resolve()


def discover_run_dirs_with_job_concurrency_sources(root_dir: Path) -> list[Path]:
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
) -> Path:
    resolved_output_path = (output_path or _default_output_path_for_run(run_dir)).expanduser().resolve()
    result = extract_job_concurrency_from_run_dir(run_dir)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def _extract_run_dir_worker(run_dir_text: str) -> tuple[str, str | None, str | None]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = extract_run_dir(run_dir)
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(run_dirs: list[Path]) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = extract_run_dir(run_dir)
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int) -> int:
    failure_count = 0
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_path_text, error_text in executor.map(
            _extract_run_dir_worker,
            [str(run_dir) for run_dir in run_dirs],
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
    run_path = Path(args.run_dir).expanduser().resolve()
    if not run_path.exists():
        raise ValueError(f"Run directory not found: {run_path}")
    if not run_path.is_dir():
        raise ValueError(
            f"--run-dir must point to a directory, got file: {run_path}. "
            f"If you want to process many runs, use --root-dir {run_path.parent}"
        )
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    resolved_output_path = extract_run_dir(run_path, output_path=output_path)
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_job_concurrency_sources(root_dir)
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
