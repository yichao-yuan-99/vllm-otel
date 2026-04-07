from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import json
import os
from pathlib import Path
import sys
from typing import Any
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pp_common.service_failure import cutoff_datetime_utc_from_payload
from pp_common.service_failure import ensure_service_failure_payload
from pp_common.service_failure import parse_iso8601_to_utc


DEFAULT_OUTPUT_NAME = "power-summary.json"


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
            "Extract GPU power summary and power-over-time points from "
            "power/power-log.jsonl for one run or all runs under a root directory."
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
            "Optional output path. Default: <run-dir>/post-processed/power/"
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


def _isoformat_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _iter_jsonl_dict_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


def _record_timestamp_utc(record: dict[str, Any]) -> datetime | None:
    payload = record.get("payload")
    candidate_timestamp_s: list[float] = []
    if isinstance(payload, dict):
        for endpoint_payload in payload.values():
            if not isinstance(endpoint_payload, dict):
                continue
            timestamp_s = _float_or_none(endpoint_payload.get("timestamp_s"))
            if timestamp_s is None:
                continue
            candidate_timestamp_s.append(timestamp_s)
    if candidate_timestamp_s:
        return datetime.fromtimestamp(min(candidate_timestamp_s), tz=timezone.utc)
    return parse_iso8601_to_utc(record.get("timestamp"))


def _iter_record_gpu_power_readings(
    record: dict[str, Any],
) -> Iterable[tuple[str, str, float]]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return
    for endpoint_name, endpoint_payload in payload.items():
        if not isinstance(endpoint_payload, dict):
            continue
        gpu_power_w = endpoint_payload.get("gpu_power_w")
        if not isinstance(gpu_power_w, dict):
            continue
        endpoint_text = str(endpoint_name)
        for gpu_id, power_value in gpu_power_w.items():
            numeric_power = _float_or_none(power_value)
            if numeric_power is None:
                continue
            yield (endpoint_text, str(gpu_id), numeric_power)


def _resolve_experiment_window(
    run_dir: Path,
    *,
    cutoff_time_utc: datetime | None = None,
) -> tuple[str, Any, Any, float | None, datetime, datetime | None]:
    replay_summary_path = run_dir / "replay" / "summary.json"
    con_driver_results_path = run_dir / "meta" / "results.json"
    con_driver_manifest_path = run_dir / "meta" / "run_manifest.json"

    source_type: str
    experiment_started_at: Any
    experiment_finished_at: Any
    run_start_utc: datetime
    run_finish_utc: datetime | None
    time_constraint_s: float | None

    if replay_summary_path.is_file():
        payload = _load_json(replay_summary_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Replay summary must be a JSON object: {replay_summary_path}")
        source_type = "replay"
        experiment_started_at = payload.get("started_at")
        experiment_finished_at = payload.get("finished_at")
        run_start_utc = parse_iso8601_to_utc(experiment_started_at)  # type: ignore[assignment]
        run_finish_utc = parse_iso8601_to_utc(experiment_finished_at)
        time_constraint_s = _non_negative_float_or_none(payload.get("time_constraint_s"))
    elif con_driver_results_path.is_file() and con_driver_manifest_path.is_file():
        payload = _load_json(con_driver_manifest_path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Con-driver run_manifest must be a JSON object: {con_driver_manifest_path}"
            )
        source_type = "con-driver"
        experiment_started_at = payload.get("started_at")
        experiment_finished_at = payload.get("finished_at")
        run_start_utc = parse_iso8601_to_utc(experiment_started_at)  # type: ignore[assignment]
        run_finish_utc = parse_iso8601_to_utc(experiment_finished_at)
        time_constraint_s = _non_negative_float_or_none(payload.get("time_constraint_s"))
    else:
        raise ValueError(
            "Unrecognized run layout. Expected either replay/summary.json "
            "or meta/results.json + meta/run_manifest.json."
        )

    if run_start_utc is None:
        raise ValueError(
            "Invalid or missing experiment start timestamp. "
            "Expected started_at in replay/summary.json or meta/run_manifest.json."
        )

    run_end_candidates: list[datetime] = []
    if run_finish_utc is not None:
        run_end_candidates.append(run_finish_utc)
    if time_constraint_s is not None:
        run_end_candidates.append(run_start_utc + timedelta(seconds=time_constraint_s))
    if cutoff_time_utc is not None:
        run_end_candidates.append(cutoff_time_utc)
    run_end_utc = min(run_end_candidates) if run_end_candidates else None

    return (
        source_type,
        experiment_started_at,
        experiment_finished_at,
        time_constraint_s,
        run_start_utc,
        run_end_utc,
    )


def _integrate_energy_j(points: list[tuple[datetime, float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    energy_j = 0.0
    previous_timestamp, _, previous_power_w = points[0]
    for timestamp_utc, _, power_w in points[1:]:
        delta_s = (timestamp_utc - previous_timestamp).total_seconds()
        if delta_s > 0:
            energy_j += ((previous_power_w + power_w) / 2.0) * delta_s
        previous_timestamp = timestamp_utc
        previous_power_w = power_w
    return energy_j


def _build_power_stats(power_values: list[float]) -> dict[str, float | None]:
    if not power_values:
        return {
            "avg": None,
            "min": None,
            "max": None,
        }
    return {
        "avg": round(sum(power_values) / len(power_values), 6),
        "min": round(min(power_values), 6),
        "max": round(max(power_values), 6),
    }


def _summarize_power_points(points: list[tuple[datetime, float, float]]) -> dict[str, Any]:
    sorted_points = sorted(points, key=lambda point: point[0])
    power_values = [point[2] for point in sorted_points]
    total_energy_j = round(_integrate_energy_j(sorted_points), 6)
    return {
        "power_sample_count": len(sorted_points),
        "power_stats_w": _build_power_stats(power_values),
        "total_energy_j": total_energy_j,
        "total_energy_kwh": round(total_energy_j / 3_600_000.0, 12),
        "power_points": [
            {
                "time_offset_s": time_offset_s,
                "power_w": power_w,
            }
            for _, time_offset_s, power_w in sorted_points
        ],
    }


def _gpu_sort_key(gpu_id: str, source_endpoint: str, gpu_key: str) -> tuple[int, str, str, str]:
    try:
        numeric_gpu_id = int(gpu_id)
    except ValueError:
        numeric_gpu_id = sys.maxsize
    return (numeric_gpu_id, gpu_id, source_endpoint, gpu_key)


def _build_per_gpu_power_entries(
    points_by_gpu_key: dict[str, list[tuple[datetime, float, float]]],
    gpu_meta_by_key: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    gpu_id_counts: dict[str, int] = {}
    for metadata in gpu_meta_by_key.values():
        gpu_id = metadata["gpu_id"]
        gpu_id_counts[gpu_id] = gpu_id_counts.get(gpu_id, 0) + 1

    entries: list[dict[str, Any]] = []
    for raw_gpu_key, points in points_by_gpu_key.items():
        metadata = gpu_meta_by_key[raw_gpu_key]
        gpu_id = metadata["gpu_id"]
        source_endpoint = metadata["source_endpoint"]
        duplicate_gpu_id = gpu_id_counts.get(gpu_id, 0) > 1
        gpu_key = raw_gpu_key if duplicate_gpu_id else gpu_id
        display_label = (
            f"GPU {gpu_id} ({source_endpoint})" if duplicate_gpu_id else f"GPU {gpu_id}"
        )
        entries.append(
            {
                "gpu_key": gpu_key,
                "gpu_id": gpu_id,
                "display_label": display_label,
                "source_endpoint": source_endpoint,
                **_summarize_power_points(points),
            }
        )

    entries.sort(
        key=lambda entry: _gpu_sort_key(
            str(entry["gpu_id"]),
            str(entry["source_endpoint"]),
            str(entry["gpu_key"]),
        )
    )
    return entries


def extract_power_summary_from_run_dir(run_dir: Path) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    service_failure_payload = ensure_service_failure_payload(resolved_run_dir)
    cutoff_time_utc = cutoff_datetime_utc_from_payload(service_failure_payload)

    (
        source_type,
        experiment_started_at,
        experiment_finished_at,
        time_constraint_s,
        run_start_utc,
        run_end_utc,
    ) = _resolve_experiment_window(resolved_run_dir, cutoff_time_utc=cutoff_time_utc)

    power_log_path = resolved_run_dir / "power" / "power-log.jsonl"
    if not power_log_path.is_file():
        return {
            "source_run_dir": str(resolved_run_dir),
            "source_type": source_type,
            "source_power_log_path": str(power_log_path.resolve()),
            "experiment_started_at": experiment_started_at,
            "experiment_finished_at": experiment_finished_at,
            "time_constraint_s": time_constraint_s,
            "analysis_window_start_utc": _isoformat_utc(run_start_utc),
            "analysis_window_end_utc": _isoformat_utc(run_end_utc),
            "service_failure_detected": bool(
                service_failure_payload.get("service_failure_detected", False)
            ),
            "service_failure_cutoff_time_utc": service_failure_payload.get("cutoff_time_utc"),
            "power_log_found": False,
            "power_sample_count": 0,
            "power_stats_w": {
                "avg": None,
                "min": None,
                "max": None,
            },
            "total_energy_j": 0.0,
            "total_energy_kwh": 0.0,
            "power_points": [],
            "per_gpu_power": [],
        }

    points: list[tuple[datetime, float, float]] = []
    per_gpu_points_by_key: dict[str, list[tuple[datetime, float, float]]] = {}
    per_gpu_meta_by_key: dict[str, dict[str, str]] = {}
    for record in _iter_jsonl_dict_records(power_log_path):
        timestamp_utc = _record_timestamp_utc(record)
        gpu_power_readings = list(_iter_record_gpu_power_readings(record))
        if timestamp_utc is None or not gpu_power_readings:
            continue
        if timestamp_utc < run_start_utc:
            continue
        if run_end_utc is not None and timestamp_utc > run_end_utc:
            continue
        time_offset_s = round((timestamp_utc - run_start_utc).total_seconds(), 6)
        total_power_w = round(
            sum(power_w for _, _, power_w in gpu_power_readings),
            6,
        )
        points.append((timestamp_utc, time_offset_s, total_power_w))
        for source_endpoint, gpu_id, power_w in gpu_power_readings:
            raw_gpu_key = f"{source_endpoint}::{gpu_id}"
            per_gpu_meta_by_key.setdefault(
                raw_gpu_key,
                {
                    "gpu_id": gpu_id,
                    "source_endpoint": source_endpoint,
                },
            )
            per_gpu_points_by_key.setdefault(raw_gpu_key, []).append(
                (timestamp_utc, time_offset_s, round(power_w, 6))
            )

    aggregate_summary = _summarize_power_points(points)
    per_gpu_power = _build_per_gpu_power_entries(
        per_gpu_points_by_key,
        per_gpu_meta_by_key,
    )

    return {
        "source_run_dir": str(resolved_run_dir),
        "source_type": source_type,
        "source_power_log_path": str(power_log_path.resolve()),
        "experiment_started_at": experiment_started_at,
        "experiment_finished_at": experiment_finished_at,
        "time_constraint_s": time_constraint_s,
        "analysis_window_start_utc": _isoformat_utc(run_start_utc),
        "analysis_window_end_utc": _isoformat_utc(run_end_utc),
        "service_failure_detected": bool(
            service_failure_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": service_failure_payload.get("cutoff_time_utc"),
        "power_log_found": True,
        "power_sample_count": aggregate_summary["power_sample_count"],
        "power_stats_w": aggregate_summary["power_stats_w"],
        "total_energy_j": aggregate_summary["total_energy_j"],
        "total_energy_kwh": aggregate_summary["total_energy_kwh"],
        "power_points": aggregate_summary["power_points"],
        "per_gpu_power": per_gpu_power,
    }


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "power" / DEFAULT_OUTPUT_NAME).resolve()


def discover_run_dirs_with_power_sources(root_dir: Path) -> list[Path]:
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


def extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_path = (output_path or _default_output_path_for_run(resolved_run_dir)).expanduser().resolve()
    result = extract_power_summary_from_run_dir(resolved_run_dir)
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
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    resolved_output_path = extract_run_dir(run_dir, output_path=output_path)
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

    run_dirs = discover_run_dirs_with_power_sources(root_dir)
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
