from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from datetime import timezone
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


DEFAULT_CTX_AWARE_LOG_GLOB = "ctx_aware_*.jsonl"
DEFAULT_OUTPUT_NAME = "ctx-aware-timeseries.json"

STATE_METRIC_FIELDS = (
    "ongoing_agent_count",
    "pending_agent_count",
    "ongoing_effective_context_tokens",
    "pending_effective_context_tokens",
)
EVENT_METRIC_FIELDS = (
    "agents_turned_pending_due_to_context_threshold",
    "agents_turned_ongoing",
    "new_agents_added_as_pending",
    "new_agents_added_as_ongoing",
)
METRIC_FIELDS = STATE_METRIC_FIELDS + EVENT_METRIC_FIELDS

_PROFILE_SUFFIX_RE = re.compile(r"_profile-(\d+)\.jsonl$")


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
            "Extract gateway_multi ctx-aware job log JSONL into a stable timeseries "
            "JSON for one run or for all runs under a root directory."
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
            "with gateway-output/job/ctx_aware_*.jsonl will be processed."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/post-processed/gateway/"
            f"ctx-aware-log/{DEFAULT_OUTPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--ctx-aware-log",
        default=None,
        help=(
            "Optional input path to a single ctx-aware JSONL log (only for --run-dir). "
            "Default: all <run-dir>/gateway-output/job/ctx_aware_*.jsonl files"
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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        if value.is_integer() and math.isfinite(value):
            return max(int(value), 0)
    return 0


def _parse_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _isoformat_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    text = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if not text.endswith("Z"):
        return text
    body = text[:-1]
    if "." not in body:
        return text
    head, fraction = body.split(".", 1)
    trimmed_fraction = fraction.rstrip("0")
    if not trimmed_fraction:
        return f"{head}Z"
    return f"{head}.{trimmed_fraction}Z"


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing ctx-aware log file: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object on line {line_number} in {path}")
        records.append(payload)
    return records


def _metric_summary(points: list[dict[str, Any]], field_name: str) -> dict[str, float | int | None]:
    values = [
        _float_or_none(point.get(field_name))
        for point in points
        if _float_or_none(point.get(field_name)) is not None
    ]
    if not values:
        return {
            "sample_count": 0,
            "avg": None,
            "min": None,
            "max": None,
            "total": None,
        }
    return {
        "sample_count": len(values),
        "avg": round(sum(values) / len(values), 6),
        "min": min(values),
        "max": max(values),
        "total": round(sum(values), 6),
    }


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (
        run_dir
        / "post-processed"
        / "gateway"
        / "ctx-aware-log"
        / DEFAULT_OUTPUT_NAME
    ).resolve()


def _default_ctx_aware_log_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / "gateway-output" / "job").resolve()


def _profile_id_from_log_name(path: Path) -> int | None:
    match = _PROFILE_SUFFIX_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _series_key_for_log_path(path: Path) -> str:
    profile_id = _profile_id_from_log_name(path)
    if profile_id is not None:
        return f"profile-{profile_id}"
    return path.stem


def _display_label_for_log_path(path: Path) -> str:
    profile_id = _profile_id_from_log_name(path)
    if profile_id is not None:
        return f"Profile {profile_id}"
    return path.name


def _discover_ctx_aware_logs_for_run(run_dir: Path) -> list[Path]:
    log_dir = _default_ctx_aware_log_dir_for_run(run_dir)
    if not log_dir.is_dir():
        return []
    candidates = [
        path.resolve()
        for path in log_dir.glob(DEFAULT_CTX_AWARE_LOG_GLOB)
        if path.is_file()
    ]
    return sorted(
        candidates,
        key=lambda path: (
            _profile_id_from_log_name(path) is None,
            _profile_id_from_log_name(path) if _profile_id_from_log_name(path) is not None else sys.maxsize,
            path.name,
        ),
    )


def discover_run_dirs_with_ctx_aware_log(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for log_path in root_dir.rglob(DEFAULT_CTX_AWARE_LOG_GLOB):
        if not log_path.is_file():
            continue
        if log_path.parent.name != "job":
            continue
        if log_path.parent.parent.name != "gateway-output":
            continue
        run_dirs.add(log_path.parent.parent.parent.resolve())
    return sorted(run_dirs)


def _resolve_ctx_aware_log_paths(
    run_dir: Path,
    *,
    ctx_aware_log_path: Path | None = None,
) -> list[Path]:
    if ctx_aware_log_path is not None:
        return [ctx_aware_log_path.expanduser().resolve()]

    candidates = _discover_ctx_aware_logs_for_run(run_dir)
    if not candidates:
        raise ValueError(
            "Missing ctx-aware log under "
            f"{_default_ctx_aware_log_dir_for_run(run_dir) / DEFAULT_CTX_AWARE_LOG_GLOB}"
        )
    return candidates


def _build_series_payload(
    run_dir: Path,
    ctx_aware_log_path: Path,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_log_path = ctx_aware_log_path.expanduser().resolve()
    raw_records = _load_jsonl_records(resolved_log_path)

    dated_samples: list[tuple[datetime, str, dict[str, Any]]] = []
    for record in raw_records:
        timestamp = record.get("timestamp")
        timestamp_dt = _parse_iso8601(timestamp)
        if timestamp_dt is None or not isinstance(timestamp, str):
            continue
        normalized = {
            "timestamp": timestamp,
        }
        for field_name in METRIC_FIELDS:
            normalized[field_name] = _non_negative_int(record.get(field_name))
        dated_samples.append((timestamp_dt, timestamp, normalized))

    dated_samples.sort(key=lambda item: item[0])
    sample_points: list[dict[str, Any]] = []
    started_at: str | None = None
    ended_at: str | None = None
    avg_sample_interval_s: float | None = None

    if dated_samples:
        base_dt = dated_samples[0][0]
        started_at = dated_samples[0][1]
        ended_at = dated_samples[-1][1]

        for timestamp_dt, _timestamp_text, normalized in dated_samples:
            sample_points.append(
                {
                    "second": round((timestamp_dt - base_dt).total_seconds(), 6),
                    **normalized,
                }
            )

        if len(sample_points) > 1:
            intervals = [
                float(sample_points[index]["second"]) - float(sample_points[index - 1]["second"])
                for index in range(1, len(sample_points))
            ]
            if intervals:
                avg_sample_interval_s = round(sum(intervals) / len(intervals), 6)

    metric_summaries = {
        field_name: _metric_summary(sample_points, field_name)
        for field_name in METRIC_FIELDS
    }
    duration_s = round(float(sample_points[-1]["second"]), 6) if sample_points else 0.0
    profile_id = _profile_id_from_log_name(resolved_log_path)

    return {
        "source_run_dir": str(resolved_run_dir),
        "source_ctx_aware_log_path": str(resolved_log_path),
        "selected_ctx_aware_log_file_name": resolved_log_path.name,
        "port_profile_id": profile_id,
        "series_key": _series_key_for_log_path(resolved_log_path),
        "display_label": _display_label_for_log_path(resolved_log_path),
        "started_at": started_at,
        "ended_at": ended_at,
        "sample_count": len(sample_points),
        "duration_s": duration_s,
        "avg_sample_interval_s": avg_sample_interval_s,
        "metric_summaries": metric_summaries,
        "samples": sample_points,
    }


def _build_combined_payload(series_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    parsed_series: list[list[tuple[datetime, dict[str, Any]]]] = []
    all_timestamps: set[datetime] = set()

    for payload in series_payloads:
        parsed_samples: list[tuple[datetime, dict[str, Any]]] = []
        raw_samples = payload.get("samples")
        if not isinstance(raw_samples, list):
            parsed_series.append(parsed_samples)
            continue
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            timestamp_dt = _parse_iso8601(sample.get("timestamp"))
            if timestamp_dt is None:
                continue
            parsed_samples.append((timestamp_dt, sample))
            all_timestamps.add(timestamp_dt)
        parsed_samples.sort(key=lambda item: item[0])
        parsed_series.append(parsed_samples)

    if not all_timestamps:
        metric_summaries = {
            field_name: _metric_summary([], field_name)
            for field_name in METRIC_FIELDS
        }
        return {
            "started_at": None,
            "ended_at": None,
            "sample_count": 0,
            "duration_s": 0.0,
            "avg_sample_interval_s": None,
            "metric_summaries": metric_summaries,
            "samples": [],
        }

    sorted_timestamps = sorted(all_timestamps)
    base_dt = sorted_timestamps[0]
    sample_points: list[dict[str, Any]] = []
    series_indices = [0 for _ in parsed_series]
    series_state: list[dict[str, Any] | None] = [None for _ in parsed_series]

    for timestamp_dt in sorted_timestamps:
        combined_sample = {
            "second": round((timestamp_dt - base_dt).total_seconds(), 6),
            "timestamp": _isoformat_utc(timestamp_dt),
        }
        for field_name in METRIC_FIELDS:
            combined_sample[field_name] = 0

        for series_index, series_samples in enumerate(parsed_series):
            event_totals = {field_name: 0 for field_name in EVENT_METRIC_FIELDS}
            while (
                series_indices[series_index] < len(series_samples)
                and series_samples[series_indices[series_index]][0] <= timestamp_dt
            ):
                sample_dt, sample_payload = series_samples[series_indices[series_index]]
                series_state[series_index] = sample_payload
                if sample_dt == timestamp_dt:
                    for field_name in EVENT_METRIC_FIELDS:
                        event_totals[field_name] += _non_negative_int(sample_payload.get(field_name))
                series_indices[series_index] += 1

            latest_state = series_state[series_index]
            if latest_state is not None:
                for field_name in STATE_METRIC_FIELDS:
                    combined_sample[field_name] += _non_negative_int(latest_state.get(field_name))
            for field_name in EVENT_METRIC_FIELDS:
                combined_sample[field_name] += event_totals[field_name]

        sample_points.append(combined_sample)

    avg_sample_interval_s: float | None = None
    if len(sample_points) > 1:
        intervals = [
            float(sample_points[index]["second"]) - float(sample_points[index - 1]["second"])
            for index in range(1, len(sample_points))
        ]
        if intervals:
            avg_sample_interval_s = round(sum(intervals) / len(intervals), 6)

    metric_summaries = {
        field_name: _metric_summary(sample_points, field_name)
        for field_name in METRIC_FIELDS
    }
    return {
        "started_at": _isoformat_utc(sorted_timestamps[0]),
        "ended_at": _isoformat_utc(sorted_timestamps[-1]),
        "sample_count": len(sample_points),
        "duration_s": round(float(sample_points[-1]["second"]), 6) if sample_points else 0.0,
        "avg_sample_interval_s": avg_sample_interval_s,
        "metric_summaries": metric_summaries,
        "samples": sample_points,
    }


def extract_ctx_aware_log_from_run_dir(
    run_dir: Path,
    *,
    ctx_aware_log_path: Path | None = None,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_log_paths = _resolve_ctx_aware_log_paths(
        resolved_run_dir,
        ctx_aware_log_path=ctx_aware_log_path,
    )
    series_payloads = [
        _build_series_payload(resolved_run_dir, log_path)
        for log_path in resolved_log_paths
    ]
    combined_payload = _build_combined_payload(series_payloads)
    port_profile_ids = sorted(
        payload["port_profile_id"]
        for payload in series_payloads
        if isinstance(payload.get("port_profile_id"), int)
    )
    multi_profile = len(series_payloads) > 1

    result: dict[str, Any] = {
        "source_run_dir": str(resolved_run_dir),
        "source_ctx_aware_log_dir": str(_default_ctx_aware_log_dir_for_run(resolved_run_dir)),
        "source_ctx_aware_log_paths": [str(path) for path in resolved_log_paths],
        "source_ctx_aware_log_path": (
            str(resolved_log_paths[0]) if len(resolved_log_paths) == 1 else None
        ),
        "selected_ctx_aware_log_file_name": (
            resolved_log_paths[0].name if len(resolved_log_paths) == 1 else None
        ),
        "ctx_aware_log_candidate_count": len(resolved_log_paths),
        "ctx_aware_log_count": len(resolved_log_paths),
        "ctx_aware_log_candidates": [str(path) for path in resolved_log_paths[:20]],
        "multi_profile": multi_profile,
        "port_profile_ids": port_profile_ids,
        "started_at": combined_payload["started_at"],
        "ended_at": combined_payload["ended_at"],
        "sample_count": combined_payload["sample_count"],
        "duration_s": combined_payload["duration_s"],
        "avg_sample_interval_s": combined_payload["avg_sample_interval_s"],
        "metric_summaries": combined_payload["metric_summaries"],
        "samples": combined_payload["samples"],
        "ctx_aware_logs": series_payloads,
    }
    return result


def extract_run_dir(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    ctx_aware_log_path: Path | None = None,
) -> Path:
    resolved_output_path = (output_path or _default_output_path_for_run(run_dir)).expanduser().resolve()
    result = extract_ctx_aware_log_from_run_dir(
        run_dir,
        ctx_aware_log_path=ctx_aware_log_path,
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
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(
            f"--run-dir must point to a directory, got file: {run_dir}. "
            f"If you want to process many runs, use --root-dir {run_dir.parent}"
        )
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    ctx_aware_log_path = (
        Path(args.ctx_aware_log).expanduser().resolve()
        if args.ctx_aware_log
        else None
    )
    resolved_output_path = extract_run_dir(
        run_dir,
        output_path=output_path,
        ctx_aware_log_path=ctx_aware_log_path,
    )
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.ctx_aware_log:
        raise ValueError("--ctx-aware-log can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_ctx_aware_log(root_dir)
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
