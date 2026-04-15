from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import json
import os
from pathlib import Path
import re
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


DEFAULT_OUTPUT_NAME = "freq-control-summary.json"
DEFAULT_FREQ_CONTROL_LOG_DIR_NAME = "freq-control"
SEGMENTED_FREQ_CONTROL_LOG_DIR_NAME = "freq-control-seg"
LINESPACE_FREQ_CONTROL_LOG_DIR_NAME = "freq-control-linespace"
INSTANCE_SLO_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME = (
    "freq-control-linespace-instance-slo"
)
INSTANCE_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME = "freq-control-linespace-instance"
AMD_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME = "freq-control-linespace-amd"
MULTI_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME = "freq-control-linespace-multi"
QUERY_LOG_GLOBS = (
    "freq-controller.query.*.jsonl",
    "freq-controller-ls.query.*.jsonl",
    "freq-controller-ls-instance-slo.query.*.jsonl",
    "freq-controller-ls-instance.query.*.jsonl",
    "freq-controller-ls-amd.query.*.jsonl",
    "freq-controller-ls-multi.query.*.jsonl",
)
DECISION_LOG_GLOBS = (
    "freq-controller.decision.*.jsonl",
    "freq-controller-ls.decision.*.jsonl",
    "freq-controller-ls-instance-slo.decision.*.jsonl",
    "freq-controller-ls-instance.decision.*.jsonl",
    "freq-controller-ls-amd.decision.*.jsonl",
    "freq-controller-ls-multi.decision.*.jsonl",
)
CONTROL_ERROR_LOG_GLOBS = (
    "freq-controller.control-error.*.jsonl",
    "freq-controller-ls.control-error.*.jsonl",
    "freq-controller-ls-instance-slo.control-error.*.jsonl",
    "freq-controller-ls-instance.control-error.*.jsonl",
    "freq-controller-ls-amd.control-error.*.jsonl",
    "freq-controller-ls-multi.control-error.*.jsonl",
)
LINESPACE_LOG_PREFIX = "freq-controller-ls."
INSTANCE_SLO_LINESPACE_LOG_PREFIX = "freq-controller-ls-instance-slo."
INSTANCE_LINESPACE_LOG_PREFIX = "freq-controller-ls-instance."
AMD_LINESPACE_LOG_PREFIX = "freq-controller-ls-amd."
MULTI_LINESPACE_LOG_PREFIX = "freq-controller-ls-multi."
PROFILE_DIR_RE = re.compile(r"^profile-(\d+)$")


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
            "Extract freq-controller query and decision timeseries for one run "
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
            "Optional output path. Default matches the detected log layout: "
            f"<run-dir>/post-processed/freq-control/{DEFAULT_OUTPUT_NAME} or "
            f"<run-dir>/post-processed/freq-control-seg/{DEFAULT_OUTPUT_NAME} or "
            f"<run-dir>/post-processed/freq-control-linespace/{DEFAULT_OUTPUT_NAME} or "
            f"<run-dir>/post-processed/freq-control-linespace-instance-slo/{DEFAULT_OUTPUT_NAME} or "
            f"<run-dir>/post-processed/freq-control-linespace-instance/{DEFAULT_OUTPUT_NAME} or "
            f"<run-dir>/post-processed/freq-control-linespace-amd/{DEFAULT_OUTPUT_NAME} or "
            f"<run-dir>/post-processed/freq-control-linespace-multi/{DEFAULT_OUTPUT_NAME}"
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


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
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


def _detect_freq_control_layout_name(run_dir: Path) -> str:
    segmented_dir = (run_dir / SEGMENTED_FREQ_CONTROL_LOG_DIR_NAME).resolve()
    if segmented_dir.exists():
        return SEGMENTED_FREQ_CONTROL_LOG_DIR_NAME
    multi_linespace_dir = (run_dir / MULTI_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME).resolve()
    if multi_linespace_dir.exists():
        return MULTI_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    instance_slo_linespace_dir = (
        run_dir / INSTANCE_SLO_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    ).resolve()
    if instance_slo_linespace_dir.exists():
        return INSTANCE_SLO_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    instance_linespace_dir = (
        run_dir / INSTANCE_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    ).resolve()
    if instance_linespace_dir.exists():
        return INSTANCE_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    amd_linespace_dir = (run_dir / AMD_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME).resolve()
    if amd_linespace_dir.exists():
        return AMD_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    linespace_dir = (run_dir / LINESPACE_FREQ_CONTROL_LOG_DIR_NAME).resolve()
    if linespace_dir.exists():
        return LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    multi_linespace_log_paths, _ = _discover_log_paths(
        run_dir,
        (
            "freq-controller-ls-multi.query.*.jsonl",
            "freq-controller-ls-multi.decision.*.jsonl",
            "freq-controller-ls-multi.control-error.*.jsonl",
        ),
    )
    if multi_linespace_log_paths:
        return MULTI_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    instance_slo_linespace_log_paths, _ = _discover_log_paths(
        run_dir,
        (
            "freq-controller-ls-instance-slo.query.*.jsonl",
            "freq-controller-ls-instance-slo.decision.*.jsonl",
            "freq-controller-ls-instance-slo.control-error.*.jsonl",
        ),
    )
    if instance_slo_linespace_log_paths:
        return INSTANCE_SLO_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    instance_linespace_log_paths, _ = _discover_log_paths(
        run_dir,
        (
            "freq-controller-ls-instance.query.*.jsonl",
            "freq-controller-ls-instance.decision.*.jsonl",
            "freq-controller-ls-instance.control-error.*.jsonl",
        ),
    )
    if instance_linespace_log_paths:
        return INSTANCE_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    amd_linespace_log_paths, _ = _discover_log_paths(
        run_dir,
        (
            "freq-controller-ls-amd.query.*.jsonl",
            "freq-controller-ls-amd.decision.*.jsonl",
            "freq-controller-ls-amd.control-error.*.jsonl",
        ),
    )
    if amd_linespace_log_paths:
        return AMD_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    linespace_log_paths, _ = _discover_log_paths(
        run_dir,
        (
            "freq-controller-ls.query.*.jsonl",
            "freq-controller-ls.decision.*.jsonl",
            "freq-controller-ls.control-error.*.jsonl",
        ),
    )
    if linespace_log_paths:
        return LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
    baseline_dir = (run_dir / DEFAULT_FREQ_CONTROL_LOG_DIR_NAME).resolve()
    if baseline_dir.exists():
        return DEFAULT_FREQ_CONTROL_LOG_DIR_NAME
    return DEFAULT_FREQ_CONTROL_LOG_DIR_NAME


def _resolve_log_dir_candidates(run_dir: Path) -> list[tuple[str | None, Path]]:
    candidates: list[tuple[str | None, Path]] = []
    for log_dir_name in (
        SEGMENTED_FREQ_CONTROL_LOG_DIR_NAME,
        MULTI_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME,
        INSTANCE_SLO_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME,
        INSTANCE_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME,
        AMD_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME,
        LINESPACE_FREQ_CONTROL_LOG_DIR_NAME,
        DEFAULT_FREQ_CONTROL_LOG_DIR_NAME,
    ):
        log_dir = (run_dir / log_dir_name).resolve()
        if all(existing_path != log_dir for _, existing_path in candidates):
            candidates.append((log_dir_name, log_dir))
    resolved_run_dir = run_dir.resolve()
    if all(existing_path != resolved_run_dir for _, existing_path in candidates):
        candidates.append((None, resolved_run_dir))
    return candidates


def _discover_log_paths(
    run_dir: Path,
    glob_patterns: tuple[str, ...],
) -> tuple[list[Path], str | None]:
    for log_dir_name, log_dir in _resolve_log_dir_candidates(run_dir):
        # Actual freq-control directories may contain nested profile-specific logs
        # (for example freq-control-linespace/profile-2/*.jsonl). Keep the
        # legacy run-dir fallback non-recursive so we do not scan unrelated
        # parts of the run tree when probing for root-level historical layouts.
        matcher = log_dir.glob if log_dir_name is None else log_dir.rglob
        matches = sorted(
            {
                path.resolve()
                for glob_pattern in glob_patterns
                for path in matcher(glob_pattern)
                if path.is_file()
            }
        )
        if matches:
            return matches, log_dir_name
    return [], None


def _port_profile_id_from_log_path(path: Path, *, run_dir: Path) -> int | None:
    try:
        relative_parts = path.resolve().relative_to(run_dir.resolve()).parts
    except ValueError:
        relative_parts = path.resolve().parts
    for part in relative_parts:
        match = PROFILE_DIR_RE.fullmatch(part)
        if match is not None:
            return int(match.group(1))
    return None


def _profile_label(profile_id: int) -> str:
    return f"profile-{profile_id}"


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


def _normalize_query_point(
    record: dict[str, Any],
    *,
    run_start_utc: datetime,
    run_end_utc: datetime | None,
    port_profile_id: int | None,
) -> dict[str, Any] | None:
    timestamp_utc = parse_iso8601_to_utc(record.get("timestamp"))
    if timestamp_utc is None:
        return None
    if run_end_utc is not None and timestamp_utc > run_end_utc:
        return None
    result = {
        "timestamp_utc": _isoformat_utc(timestamp_utc),
        "time_offset_s": round((timestamp_utc - run_start_utc).total_seconds(), 6),
        "phase": _non_empty_str_or_none(record.get("phase")),
        "context_usage": _float_or_none(record.get("context_usage")),
        "job_active": _bool_or_none(record.get("job_active")),
        "agent_count": _int_or_none(record.get("agent_count")),
        "sample_count_window": _int_or_none(record.get("sample_count_window")),
        "error": _non_empty_str_or_none(record.get("error")),
    }
    if port_profile_id is not None:
        result["port_profile_id"] = port_profile_id
    return result


def _normalize_decision_point(
    record: dict[str, Any],
    *,
    run_start_utc: datetime,
    run_end_utc: datetime | None,
    port_profile_id: int | None,
) -> dict[str, Any] | None:
    timestamp_utc = parse_iso8601_to_utc(record.get("timestamp"))
    if timestamp_utc is None:
        return None
    if run_end_utc is not None and timestamp_utc > run_end_utc:
        return None
    result = {
        "timestamp_utc": _isoformat_utc(timestamp_utc),
        "time_offset_s": round((timestamp_utc - run_start_utc).total_seconds(), 6),
        "action": _non_empty_str_or_none(record.get("action")),
        "changed": _bool_or_none(record.get("changed")),
        "current_frequency_mhz": _int_or_none(record.get("current_frequency_mhz")),
        "target_frequency_mhz": _int_or_none(record.get("target_frequency_mhz")),
        "window_context_usage": _float_or_none(record.get("window_context_usage")),
        "sample_count": _int_or_none(record.get("sample_count")),
        "lower_bound": _float_or_none(record.get("lower_bound")),
        "upper_bound": _float_or_none(record.get("upper_bound")),
        "target_context_usage_threshold": _float_or_none(
            record.get("target_context_usage_threshold")
        ),
        "segment_count": _int_or_none(record.get("segment_count")),
        "segment_width_context_usage": _float_or_none(
            record.get("segment_width_context_usage")
        ),
        "target_frequency_index": _int_or_none(record.get("target_frequency_index")),
        "low_freq_threshold": _float_or_none(record.get("low_freq_threshold")),
        "low_freq_cap_mhz": _int_or_none(record.get("low_freq_cap_mhz")),
        "effective_min_frequency_mhz": _int_or_none(
            record.get("effective_min_frequency_mhz")
        ),
    }
    if port_profile_id is not None:
        result["port_profile_id"] = port_profile_id
    return result


def _normalize_control_error_point(
    record: dict[str, Any],
    *,
    run_start_utc: datetime,
    run_end_utc: datetime | None,
    port_profile_id: int | None,
) -> dict[str, Any] | None:
    timestamp_utc = parse_iso8601_to_utc(record.get("timestamp"))
    if timestamp_utc is None:
        return None
    if run_end_utc is not None and timestamp_utc > run_end_utc:
        return None
    result = {
        "timestamp_utc": _isoformat_utc(timestamp_utc),
        "time_offset_s": round((timestamp_utc - run_start_utc).total_seconds(), 6),
        "reason": _non_empty_str_or_none(record.get("reason")),
        "action": _non_empty_str_or_none(record.get("action")),
        "error": _non_empty_str_or_none(record.get("error")),
        "attempted_frequency_index": _int_or_none(
            record.get("attempted_frequency_index")
        ),
        "attempted_frequency_mhz": _int_or_none(record.get("attempted_frequency_mhz")),
        "current_frequency_index": _int_or_none(record.get("current_frequency_index")),
        "current_frequency_mhz": _int_or_none(record.get("current_frequency_mhz")),
        "moving_average_context_usage": _float_or_none(
            record.get("moving_average_context_usage")
        ),
        "sample_count": _int_or_none(record.get("sample_count")),
    }
    if port_profile_id is not None:
        result["port_profile_id"] = port_profile_id
    return result


def _first_non_none(items: Iterable[Any]) -> Any:
    for item in items:
        if item is not None:
            return item
    return None


def _max_or_none(items: Iterable[float | None]) -> float | None:
    values = [value for value in items if value is not None]
    if not values:
        return None
    return max(values)


def _min_or_none(items: Iterable[int | None]) -> int | None:
    values = [value for value in items if value is not None]
    if not values:
        return None
    return min(values)


def _max_int_or_none(items: Iterable[int | None]) -> int | None:
    values = [value for value in items if value is not None]
    if not values:
        return None
    return max(values)


def _build_summary_from_log_paths(
    *,
    run_dir: Path,
    source_type: str,
    source_freq_control_log_dir_name: str,
    query_log_paths: list[Path],
    decision_log_paths: list[Path],
    control_error_log_paths: list[Path],
    experiment_started_at: Any,
    experiment_finished_at: Any,
    time_constraint_s: float | None,
    run_start_utc: datetime,
    run_end_utc: datetime | None,
    service_failure_payload: dict[str, Any],
) -> dict[str, Any]:
    query_points: list[dict[str, Any]] = []
    for query_log_path in query_log_paths:
        port_profile_id = _port_profile_id_from_log_path(query_log_path, run_dir=run_dir)
        for record in _iter_jsonl_dict_records(query_log_path):
            point = _normalize_query_point(
                record,
                run_start_utc=run_start_utc,
                run_end_utc=run_end_utc,
                port_profile_id=port_profile_id,
            )
            if point is not None:
                query_points.append(point)
    query_points.sort(
        key=lambda point: (
            _float_or_none(point.get("time_offset_s")) or 0.0,
            _non_empty_str_or_none(point.get("timestamp_utc")) or "",
        )
    )

    decision_points: list[dict[str, Any]] = []
    for decision_log_path in decision_log_paths:
        port_profile_id = _port_profile_id_from_log_path(decision_log_path, run_dir=run_dir)
        for record in _iter_jsonl_dict_records(decision_log_path):
            point = _normalize_decision_point(
                record,
                run_start_utc=run_start_utc,
                run_end_utc=run_end_utc,
                port_profile_id=port_profile_id,
            )
            if point is not None:
                decision_points.append(point)
    decision_points.sort(
        key=lambda point: (
            _float_or_none(point.get("time_offset_s")) or 0.0,
            _non_empty_str_or_none(point.get("timestamp_utc")) or "",
        )
    )

    control_error_points: list[dict[str, Any]] = []
    for control_error_log_path in control_error_log_paths:
        port_profile_id = _port_profile_id_from_log_path(
            control_error_log_path,
            run_dir=run_dir,
        )
        for record in _iter_jsonl_dict_records(control_error_log_path):
            point = _normalize_control_error_point(
                record,
                run_start_utc=run_start_utc,
                run_end_utc=run_end_utc,
                port_profile_id=port_profile_id,
            )
            if point is not None:
                control_error_points.append(point)
    control_error_points.sort(
        key=lambda point: (
            _float_or_none(point.get("time_offset_s")) or 0.0,
            _non_empty_str_or_none(point.get("timestamp_utc")) or "",
        )
    )

    pending_query_point_count = sum(
        1 for point in query_points if point.get("phase") == "pending"
    )
    active_query_point_count = sum(
        1 for point in query_points if point.get("phase") == "active"
    )
    query_error_count = sum(1 for point in query_points if point.get("error") is not None)
    control_error_point_count = len(control_error_points)
    decision_change_count = sum(
        1 for point in decision_points if point.get("changed") is True
    )
    lower_bound = _first_non_none(
        point.get("lower_bound") for point in decision_points
    )
    upper_bound = _first_non_none(
        point.get("upper_bound") for point in decision_points
    )
    target_context_usage_threshold = _first_non_none(
        point.get("target_context_usage_threshold") for point in decision_points
    )
    segment_count = _first_non_none(
        point.get("segment_count") for point in decision_points
    )
    segment_width_context_usage = _first_non_none(
        point.get("segment_width_context_usage") for point in decision_points
    )
    low_freq_threshold = _first_non_none(
        point.get("low_freq_threshold") for point in decision_points
    )
    low_freq_cap_mhz = _first_non_none(
        point.get("low_freq_cap_mhz") for point in decision_points
    )
    max_context_usage = _max_or_none(
        _float_or_none(point.get("context_usage")) for point in query_points
    )
    max_window_context_usage = _max_or_none(
        _float_or_none(point.get("window_context_usage")) for point in decision_points
    )
    min_frequency_mhz = _min_or_none(
        value
        for point in decision_points
        for value in (
            _int_or_none(point.get("current_frequency_mhz")),
            _int_or_none(point.get("target_frequency_mhz")),
        )
    )
    max_frequency_mhz = _max_int_or_none(
        value
        for point in decision_points
        for value in (
            _int_or_none(point.get("current_frequency_mhz")),
            _int_or_none(point.get("target_frequency_mhz")),
        )
    )
    min_effective_min_frequency_mhz = _min_or_none(
        _int_or_none(point.get("effective_min_frequency_mhz")) for point in decision_points
    )
    max_effective_min_frequency_mhz = _max_int_or_none(
        _int_or_none(point.get("effective_min_frequency_mhz")) for point in decision_points
    )
    first_job_active_time_offset_s = _first_non_none(
        _float_or_none(point.get("time_offset_s"))
        for point in query_points
        if point.get("job_active") is True
    )
    first_control_error_time_offset_s = _first_non_none(
        _float_or_none(point.get("time_offset_s")) for point in control_error_points
    )
    segmented_policy_detected = (
        low_freq_threshold is not None
        or low_freq_cap_mhz is not None
        or min_effective_min_frequency_mhz is not None
        or max_effective_min_frequency_mhz is not None
        or source_freq_control_log_dir_name == SEGMENTED_FREQ_CONTROL_LOG_DIR_NAME
    )
    multi_linespace_log_detected = any(
        path.name.startswith(MULTI_LINESPACE_LOG_PREFIX) for path in query_log_paths
    ) or any(
        path.name.startswith(MULTI_LINESPACE_LOG_PREFIX) for path in decision_log_paths
    ) or any(
        path.name.startswith(MULTI_LINESPACE_LOG_PREFIX)
        for path in control_error_log_paths
    )
    instance_slo_linespace_log_detected = any(
        path.name.startswith(INSTANCE_SLO_LINESPACE_LOG_PREFIX)
        for path in query_log_paths
    ) or any(
        path.name.startswith(INSTANCE_SLO_LINESPACE_LOG_PREFIX)
        for path in decision_log_paths
    ) or any(
        path.name.startswith(INSTANCE_SLO_LINESPACE_LOG_PREFIX)
        for path in control_error_log_paths
    )
    instance_linespace_log_detected = any(
        path.name.startswith(INSTANCE_LINESPACE_LOG_PREFIX) for path in query_log_paths
    ) or any(
        path.name.startswith(INSTANCE_LINESPACE_LOG_PREFIX) for path in decision_log_paths
    ) or any(
        path.name.startswith(INSTANCE_LINESPACE_LOG_PREFIX)
        for path in control_error_log_paths
    )
    amd_linespace_log_detected = any(
        path.name.startswith(AMD_LINESPACE_LOG_PREFIX) for path in query_log_paths
    ) or any(
        path.name.startswith(AMD_LINESPACE_LOG_PREFIX) for path in decision_log_paths
    ) or any(
        path.name.startswith(AMD_LINESPACE_LOG_PREFIX)
        for path in control_error_log_paths
    )
    linespace_log_detected = any(
        path.name.startswith(LINESPACE_LOG_PREFIX) for path in query_log_paths
    ) or any(
        path.name.startswith(LINESPACE_LOG_PREFIX) for path in decision_log_paths
    ) or any(
        path.name.startswith(LINESPACE_LOG_PREFIX)
        for path in control_error_log_paths
    )
    linespace_policy_detected = (
        target_context_usage_threshold is not None
        or segment_count is not None
        or segment_width_context_usage is not None
        or source_freq_control_log_dir_name == LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
        or source_freq_control_log_dir_name
        == INSTANCE_SLO_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
        or source_freq_control_log_dir_name == INSTANCE_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
        or source_freq_control_log_dir_name == AMD_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
        or source_freq_control_log_dir_name == MULTI_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
        or linespace_log_detected
        or instance_slo_linespace_log_detected
        or instance_linespace_log_detected
        or amd_linespace_log_detected
        or multi_linespace_log_detected
    )
    effective_source_freq_control_log_dir_name = source_freq_control_log_dir_name
    if (
        linespace_policy_detected
        and effective_source_freq_control_log_dir_name == DEFAULT_FREQ_CONTROL_LOG_DIR_NAME
    ):
        effective_source_freq_control_log_dir_name = (
            MULTI_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
            if multi_linespace_log_detected
            else INSTANCE_SLO_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
            if instance_slo_linespace_log_detected
            else INSTANCE_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
            if instance_linespace_log_detected
            else AMD_LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
            if amd_linespace_log_detected
            else LINESPACE_FREQ_CONTROL_LOG_DIR_NAME
        )

    port_profile_ids = sorted(
        {
            profile_id
            for profile_id in (
                _port_profile_id_from_log_path(path, run_dir=run_dir)
                for path in (
                    query_log_paths + decision_log_paths + control_error_log_paths
                )
            )
            if profile_id is not None
        }
    )

    return {
        "source_run_dir": str(run_dir),
        "source_type": source_type,
        "source_freq_control_log_dir_name": effective_source_freq_control_log_dir_name,
        "source_query_log_paths": [str(path) for path in query_log_paths],
        "source_decision_log_paths": [str(path) for path in decision_log_paths],
        "source_control_error_log_paths": [
            str(path) for path in control_error_log_paths
        ],
        "experiment_started_at": experiment_started_at,
        "experiment_finished_at": experiment_finished_at,
        "time_constraint_s": time_constraint_s,
        "analysis_window_start_utc": _isoformat_utc(run_start_utc),
        "analysis_window_end_utc": _isoformat_utc(run_end_utc),
        "service_failure_detected": bool(
            service_failure_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": service_failure_payload.get("cutoff_time_utc"),
        "freq_control_log_found": bool(
            query_log_paths or decision_log_paths or control_error_log_paths
        ),
        "query_log_found": bool(query_log_paths),
        "decision_log_found": bool(decision_log_paths),
        "control_error_log_found": bool(control_error_log_paths),
        "multi_profile": len(port_profile_ids) > 1,
        "port_profile_ids": port_profile_ids,
        "series_keys": [_profile_label(profile_id) for profile_id in port_profile_ids],
        "query_point_count": len(query_points),
        "pending_query_point_count": pending_query_point_count,
        "active_query_point_count": active_query_point_count,
        "query_error_count": query_error_count,
        "control_error_point_count": control_error_point_count,
        "decision_point_count": len(decision_points),
        "decision_change_count": decision_change_count,
        "first_job_active_time_offset_s": first_job_active_time_offset_s,
        "first_control_error_time_offset_s": first_control_error_time_offset_s,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "linespace_policy_detected": linespace_policy_detected,
        "target_context_usage_threshold": target_context_usage_threshold,
        "segment_count": segment_count,
        "segment_width_context_usage": segment_width_context_usage,
        "segmented_policy_detected": segmented_policy_detected,
        "low_freq_threshold": low_freq_threshold,
        "low_freq_cap_mhz": low_freq_cap_mhz,
        "min_effective_min_frequency_mhz": min_effective_min_frequency_mhz,
        "max_effective_min_frequency_mhz": max_effective_min_frequency_mhz,
        "max_context_usage": max_context_usage,
        "max_window_context_usage": max_window_context_usage,
        "min_frequency_mhz": min_frequency_mhz,
        "max_frequency_mhz": max_frequency_mhz,
        "query_points": query_points,
        "decision_points": decision_points,
        "control_error_points": control_error_points,
    }


def extract_freq_control_summary_from_run_dir(run_dir: Path) -> dict[str, Any]:
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

    default_log_dir_name = _detect_freq_control_layout_name(resolved_run_dir)
    query_log_paths, query_log_dir_name = _discover_log_paths(
        resolved_run_dir,
        QUERY_LOG_GLOBS,
    )
    decision_log_paths, decision_log_dir_name = _discover_log_paths(
        resolved_run_dir,
        DECISION_LOG_GLOBS,
    )
    control_error_log_paths, control_error_log_dir_name = _discover_log_paths(
        resolved_run_dir,
        CONTROL_ERROR_LOG_GLOBS,
    )
    source_freq_control_log_dir_name = (
        decision_log_dir_name
        or query_log_dir_name
        or control_error_log_dir_name
        or default_log_dir_name
    )

    return _build_summary_from_log_paths(
        run_dir=resolved_run_dir,
        source_type=source_type,
        source_freq_control_log_dir_name=source_freq_control_log_dir_name,
        query_log_paths=query_log_paths,
        decision_log_paths=decision_log_paths,
        control_error_log_paths=control_error_log_paths,
        experiment_started_at=experiment_started_at,
        experiment_finished_at=experiment_finished_at,
        time_constraint_s=time_constraint_s,
        run_start_utc=run_start_utc,
        run_end_utc=run_end_utc,
        service_failure_payload=service_failure_payload,
    )


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (
        run_dir
        / "post-processed"
        / _detect_freq_control_layout_name(run_dir)
        / DEFAULT_OUTPUT_NAME
    ).resolve()


def discover_run_dirs_with_freq_control_sources(root_dir: Path) -> list[Path]:
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
    resolved_output_path = (
        output_path or _default_output_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    result = extract_freq_control_summary_from_run_dir(resolved_run_dir)
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

    run_dirs = discover_run_dirs_with_freq_control_sources(root_dir)
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
