from __future__ import annotations

import argparse
from collections import Counter
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
MODULE_ROOT = THIS_DIR.parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pp_common.service_failure import cutoff_datetime_utc_from_payload
from pp_common.service_failure import ensure_service_failure_payload
from pp_common.service_failure import parse_iso8601_to_utc


DEFAULT_SLO_AWARE_LOG_GLOB = "slo_aware_decisions_*.jsonl"
DEFAULT_OUTPUT_NAME = "slo-aware-events.json"


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
            "Extract gateway SLO-aware decision logs into a normalized event summary "
            "for one run or for all runs under a root directory."
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
            "with gateway-output/job/slo_aware_decisions_*.jsonl will be processed."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/post-processed/gateway/"
            f"slo-aware-log/{DEFAULT_OUTPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--slo-aware-log",
        default=None,
        help=(
            "Optional input path to a single gateway SLO-aware decision log "
            "(only for --run-dir). Default: all "
            "<run-dir>/gateway-output/job/slo_aware_decisions_*.jsonl files"
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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _non_empty_str_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


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


def _resolve_experiment_window(
    run_dir: Path,
    *,
    cutoff_time_utc: datetime | None = None,
) -> tuple[str, Any, Any, float | None, datetime, datetime | None]:
    replay_summary_path = run_dir / "replay" / "summary.json"
    con_driver_results_path = run_dir / "meta" / "results.json"
    con_driver_manifest_path = run_dir / "meta" / "run_manifest.json"

    if replay_summary_path.is_file():
        payload = _load_json(replay_summary_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Replay summary must be a JSON object: {replay_summary_path}")
        source_type = "replay"
        experiment_started_at = payload.get("started_at")
        experiment_finished_at = payload.get("finished_at")
        run_start_utc = parse_iso8601_to_utc(experiment_started_at)
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
        run_start_utc = parse_iso8601_to_utc(experiment_started_at)
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


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (
        run_dir
        / "post-processed"
        / "gateway"
        / "slo-aware-log"
        / DEFAULT_OUTPUT_NAME
    ).resolve()


def _default_slo_aware_log_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / "gateway-output" / "job").resolve()


def _discover_slo_aware_logs_for_run(run_dir: Path) -> list[Path]:
    log_dir = _default_slo_aware_log_dir_for_run(run_dir)
    if not log_dir.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in log_dir.glob(DEFAULT_SLO_AWARE_LOG_GLOB)
        if path.is_file()
    )


def discover_run_dirs_with_slo_aware_log(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for log_path in root_dir.rglob(DEFAULT_SLO_AWARE_LOG_GLOB):
        if not log_path.is_file():
            continue
        if log_path.parent.name != "job":
            continue
        if log_path.parent.parent.name != "gateway-output":
            continue
        run_dirs.add(log_path.parent.parent.parent.resolve())
    return sorted(run_dirs)


def _resolve_slo_aware_log_paths(
    run_dir: Path,
    *,
    slo_aware_log_path: Path | None = None,
) -> list[Path]:
    if slo_aware_log_path is not None:
        return [slo_aware_log_path.expanduser().resolve()]
    return _discover_slo_aware_logs_for_run(run_dir)


def _normalize_slo_aware_event(
    record: dict[str, Any],
    *,
    run_start_utc: datetime,
    run_end_utc: datetime | None,
) -> dict[str, Any] | None:
    timestamp_utc = parse_iso8601_to_utc(record.get("timestamp"))
    if timestamp_utc is None:
        return None
    if run_end_utc is not None and timestamp_utc > run_end_utc:
        return None

    ralexation_until_utc = parse_iso8601_to_utc(record.get("ralexation_until"))
    return {
        "timestamp_utc": _isoformat_utc(timestamp_utc),
        "time_offset_s": round((timestamp_utc - run_start_utc).total_seconds(), 6),
        "event_type": _non_empty_str_or_none(record.get("event_type")),
        "api_token_hash": _non_empty_str_or_none(record.get("api_token_hash")),
        "trace_id": _non_empty_str_or_none(record.get("trace_id")),
        "schedule_state": _non_empty_str_or_none(record.get("schedule_state")),
        "from_schedule_state": _non_empty_str_or_none(record.get("from_schedule_state")),
        "to_schedule_state": _non_empty_str_or_none(record.get("to_schedule_state")),
        "wake_reason": _non_empty_str_or_none(record.get("wake_reason")),
        "resume_disposition": _non_empty_str_or_none(record.get("resume_disposition")),
        "policy_mode": _non_empty_str_or_none(record.get("policy_mode")),
        "context_tokens": _int_or_none(record.get("context_tokens")),
        "effective_context_tokens": _int_or_none(record.get("effective_context_tokens")),
        "output_tokens_per_s": _float_or_none(record.get("output_tokens_per_s")),
        "slo_slack_s": _float_or_none(record.get("slo_slack_s")),
        "slo_target_tokens_per_s": _float_or_none(record.get("slo_target_tokens_per_s")),
        "min_output_tokens_per_s": _float_or_none(record.get("min_output_tokens_per_s")),
        "avg_output_tokens_per_s": _float_or_none(record.get("avg_output_tokens_per_s")),
        "ralexation_duration_s": _float_or_none(record.get("ralexation_duration_s")),
        "ralexation_until_utc": _isoformat_utc(ralexation_until_utc),
    }


def _first_non_none(items: Iterable[Any]) -> Any:
    for item in items:
        if item is not None:
            return item
    return None


def _min_or_none(items: Iterable[float | int | None]) -> float | int | None:
    values = [value for value in items if value is not None]
    if not values:
        return None
    return min(values)


def _max_or_none(items: Iterable[float | int | None]) -> float | int | None:
    values = [value for value in items if value is not None]
    if not values:
        return None
    return max(values)


def _counts_by_value(items: Iterable[str | None]) -> dict[str, int]:
    counts = Counter(item for item in items if isinstance(item, str) and item)
    return dict(sorted(counts.items()))


def extract_slo_aware_log_from_run_dir(
    run_dir: Path,
    *,
    slo_aware_log_path: Path | None = None,
) -> dict[str, Any]:
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

    log_paths = _resolve_slo_aware_log_paths(
        resolved_run_dir,
        slo_aware_log_path=slo_aware_log_path,
    )

    events: list[dict[str, Any]] = []
    for log_path in log_paths:
        for record in _iter_jsonl_dict_records(log_path):
            event = _normalize_slo_aware_event(
                record,
                run_start_utc=run_start_utc,
                run_end_utc=run_end_utc,
            )
            if event is not None:
                events.append(event)

    events.sort(
        key=lambda event: (
            _float_or_none(event.get("time_offset_s")) or 0.0,
            _non_empty_str_or_none(event.get("timestamp_utc")) or "",
            _non_empty_str_or_none(event.get("event_type")) or "",
            _non_empty_str_or_none(event.get("api_token_hash")) or "",
            _non_empty_str_or_none(event.get("trace_id")) or "",
        )
    )

    api_token_hashes = sorted(
        {
            api_token_hash
            for api_token_hash in (
                _non_empty_str_or_none(event.get("api_token_hash")) for event in events
            )
            if api_token_hash is not None
        }
    )

    return {
        "source_run_dir": str(resolved_run_dir),
        "source_type": source_type,
        "source_slo_aware_log_paths": [str(path) for path in log_paths],
        "experiment_started_at": experiment_started_at,
        "experiment_finished_at": experiment_finished_at,
        "time_constraint_s": time_constraint_s,
        "analysis_window_start_utc": _isoformat_utc(run_start_utc),
        "analysis_window_end_utc": _isoformat_utc(run_end_utc),
        "service_failure_detected": bool(
            service_failure_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": service_failure_payload.get("cutoff_time_utc"),
        "slo_aware_log_found": bool(log_paths),
        "slo_aware_event_count": len(events),
        "unique_agent_count": len(api_token_hashes),
        "unique_api_token_hashes": api_token_hashes,
        "first_slo_aware_event_time_offset_s": _first_non_none(
            _float_or_none(event.get("time_offset_s")) for event in events
        ),
        "target_output_throughput_tokens_per_s": _first_non_none(
            _float_or_none(event.get("slo_target_tokens_per_s")) for event in events
        ),
        "event_type_counts": _counts_by_value(
            _non_empty_str_or_none(event.get("event_type")) for event in events
        ),
        "wake_reason_counts": _counts_by_value(
            _non_empty_str_or_none(event.get("wake_reason")) for event in events
        ),
        "resume_disposition_counts": _counts_by_value(
            _non_empty_str_or_none(event.get("resume_disposition")) for event in events
        ),
        "min_output_tokens_per_s_at_events": _min_or_none(
            _float_or_none(event.get("output_tokens_per_s")) for event in events
        ),
        "max_output_tokens_per_s_at_events": _max_or_none(
            _float_or_none(event.get("output_tokens_per_s")) for event in events
        ),
        "min_slo_slack_s": _min_or_none(
            _float_or_none(event.get("slo_slack_s")) for event in events
        ),
        "max_slo_slack_s": _max_or_none(
            _float_or_none(event.get("slo_slack_s")) for event in events
        ),
        "min_ralexation_duration_s": _min_or_none(
            _float_or_none(event.get("ralexation_duration_s")) for event in events
        ),
        "max_ralexation_duration_s": _max_or_none(
            _float_or_none(event.get("ralexation_duration_s")) for event in events
        ),
        "min_observed_min_output_tokens_per_s": _min_or_none(
            _float_or_none(event.get("min_output_tokens_per_s")) for event in events
        ),
        "max_observed_min_output_tokens_per_s": _max_or_none(
            _float_or_none(event.get("min_output_tokens_per_s")) for event in events
        ),
        "min_observed_avg_output_tokens_per_s": _min_or_none(
            _float_or_none(event.get("avg_output_tokens_per_s")) for event in events
        ),
        "max_observed_avg_output_tokens_per_s": _max_or_none(
            _float_or_none(event.get("avg_output_tokens_per_s")) for event in events
        ),
        "events": events,
    }


def extract_run_dir(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    slo_aware_log_path: Path | None = None,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_path = (
        output_path or _default_output_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    result = extract_slo_aware_log_from_run_dir(
        resolved_run_dir,
        slo_aware_log_path=slo_aware_log_path,
    )
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
    slo_aware_log_path = (
        Path(args.slo_aware_log).expanduser().resolve() if args.slo_aware_log else None
    )
    result_path = extract_run_dir(
        run_dir,
        output_path=output_path,
        slo_aware_log_path=slo_aware_log_path,
    )
    print(str(result_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.slo_aware_log:
        raise ValueError("--slo-aware-log can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_slo_aware_log(root_dir)
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
        failure_count = _run_root_dir_parallel(run_dirs, max_procs=worker_count)
    return 0 if failure_count == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_dir is not None:
        return _main_run_dir(args)
    return _main_root_dir(args)


if __name__ == "__main__":
    raise SystemExit(main())
