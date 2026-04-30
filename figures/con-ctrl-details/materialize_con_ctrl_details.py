#!/usr/bin/env python3
"""Materialize the con-ctrl-details figure dataset."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from datetime import timezone
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "con-ctrl-details.json"
DEFAULT_MISSING_LOG_PATH = DEFAULT_OUTPUT_DIR / "con-ctrl-details.missing.log"
DEFAULT_CONTEXT_SMOOTH_WINDOW_S = 120.0

DEFAULT_NO_THRASH_PATH = Path(
    "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
    "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance/"
    "dabstep/mini-swe-agent/split/exclude-unranked/qps0_05/20260413T135808Z"
)
DEFAULT_CTX_AWARE_PATH = Path(
    "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
    "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware/"
    "dabstep/mini-swe-agent/split/exclude-unranked/qps0_05/20260413T235024Z"
)


def _utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the five-panel con-ctrl-details figure dataset."
    )
    parser.add_argument(
        "--no-thrash-run-dir",
        default=str(DEFAULT_NO_THRASH_PATH),
        help=(
            "Non-ctx-aware KAIROS replay run directory, or parent containing "
            f"timestamped runs. Default: {DEFAULT_NO_THRASH_PATH}"
        ),
    )
    parser.add_argument(
        "--ctx-aware-run-dir",
        default=str(DEFAULT_CTX_AWARE_PATH),
        help=(
            "Ctx-aware KAIROS replay run directory, or parent containing "
            f"timestamped runs. Default: {DEFAULT_CTX_AWARE_PATH}"
        ),
    )
    parser.add_argument(
        "--context-smooth-window-s",
        type=float,
        default=DEFAULT_CONTEXT_SMOOTH_WINDOW_S,
        help="Centered smoothing window for context usage in seconds (default: 120).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--missing-log",
        default=str(DEFAULT_MISSING_LOG_PATH),
        help="Optional missing-data log path.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _stable_round(value: float) -> float:
    return round(value, 6)


def _select_run_dir(base_path: Path, missing_log: list[str]) -> Path:
    if not base_path.exists():
        raise ValueError(f"Run path does not exist: {base_path}")

    if (base_path / "replay" / "summary.json").is_file():
        return base_path

    candidates = sorted(
        (
            child
            for child in base_path.iterdir()
            if child.is_dir() and (child / "replay" / "summary.json").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        raise ValueError(
            "Expected a replay run directory or a parent containing timestamped runs: "
            f"{base_path}"
        )
    selected = candidates[0]
    if selected != base_path:
        missing_log.append(f"[selected-latest-run] {base_path} -> {selected}")
    return selected


def _find_required_file(run_dir: Path, relative_path: str, missing_log: list[str]) -> Path:
    path = run_dir / relative_path
    if not path.is_file():
        missing_log.append(f"[missing-file] {path}")
        raise ValueError(f"Required file is missing: {path}")
    return path


def _load_optional_json_object(run_dir: Path, relative_path: str) -> dict[str, Any] | None:
    path = run_dir / relative_path
    if not path.is_file():
        return None
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else None


def _rolling_average(
    points: list[tuple[float, float]],
    *,
    window_s: float,
) -> list[tuple[float, float]]:
    if not points:
        return []
    if window_s <= 0.0:
        return points[:]

    half_window_s = window_s / 2.0
    queue: deque[tuple[float, float]] = deque()
    right_index = 0
    value_sum = 0.0
    smoothed: list[tuple[float, float]] = []

    for time_s, _ in points:
        while right_index < len(points) and points[right_index][0] <= time_s + half_window_s:
            right_time_s, right_value = points[right_index]
            queue.append((right_time_s, right_value))
            value_sum += right_value
            right_index += 1
        while queue and queue[0][0] < time_s - half_window_s:
            _, left_value = queue.popleft()
            value_sum -= left_value
        smoothed.append((time_s, value_sum / len(queue)))
    return smoothed


def _collapse_to_last_value_per_second(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    collapsed: list[tuple[float, float]] = []
    current_second: int | None = None
    current_item: tuple[float, float] | None = None

    for time_s, value in points:
        second = int(math.floor(time_s))
        if current_second is None:
            current_second = second
            current_item = (float(second), value)
            continue
        if second != current_second:
            if current_item is not None:
                collapsed.append(current_item)
            current_second = second
        current_item = (float(second), value)

    if current_item is not None:
        collapsed.append(current_item)
    return collapsed


def _materialize_job_throughput_series(
    payload: dict[str, Any],
) -> list[dict[str, float]]:
    raw_points = payload.get("throughput_points")
    if not isinstance(raw_points, list):
        raise ValueError("job-throughput-timeseries.json is missing throughput_points")

    points: list[dict[str, float]] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            continue
        time_s = _float_or_none(raw_point.get("time_s"))
        throughput = _float_or_none(raw_point.get("throughput_jobs_per_s"))
        if time_s is None or throughput is None:
            continue
        points.append(
            {
                "time_offset_s": _stable_round(time_s),
                "throughput_jobs_per_s": _stable_round(throughput),
            }
        )
    return points


def _materialize_context_series(
    payload: dict[str, Any],
    *,
    smooth_window_s: float,
) -> list[dict[str, float]]:
    raw_points = payload.get("points")
    if not isinstance(raw_points, list):
        raise ValueError("context-usage-stacked-histogram.json is missing points")

    points: list[tuple[float, float]] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            continue
        second = _float_or_none(raw_point.get("second"))
        value = _float_or_none(raw_point.get("accumulated_value"))
        if second is None or value is None:
            continue
        points.append((second, value))

    points.sort(key=lambda item: item[0])
    smoothed = _rolling_average(points, window_s=smooth_window_s)
    return [
        {
            "time_offset_s": _stable_round(time_s),
            "context_usage_tokens": _stable_round(value),
        }
        for time_s, value in smoothed
    ]


def _materialize_pending_agent_series(
    payload: dict[str, Any],
) -> list[dict[str, float]]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("ctx-aware-timeseries.json is missing samples")

    points: list[tuple[float, float]] = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            continue
        second = _float_or_none(raw_sample.get("second"))
        pending_count = _float_or_none(raw_sample.get("pending_agent_count"))
        if second is None or pending_count is None:
            continue
        points.append((second, pending_count))

    points.sort(key=lambda item: item[0])
    collapsed = _collapse_to_last_value_per_second(points)
    return [
        {
            "time_offset_s": _stable_round(time_s),
            "pending_agent_count": _stable_round(value),
        }
        for time_s, value in collapsed
    ]


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    missing_log_path = Path(args.missing_log).expanduser().resolve()

    missing_log: list[str] = []
    no_thrash_run_dir = _select_run_dir(
        Path(args.no_thrash_run_dir).expanduser().resolve(),
        missing_log,
    )
    ctx_aware_run_dir = _select_run_dir(
        Path(args.ctx_aware_run_dir).expanduser().resolve(),
        missing_log,
    )

    no_thrash_replay_summary_path = _find_required_file(
        no_thrash_run_dir,
        "replay/summary.json",
        missing_log,
    )
    ctx_aware_replay_summary_path = _find_required_file(
        ctx_aware_run_dir,
        "replay/summary.json",
        missing_log,
    )
    no_thrash_job_throughput_path = _find_required_file(
        no_thrash_run_dir,
        "post-processed/job-throughput/job-throughput-timeseries.json",
        missing_log,
    )
    no_thrash_context_histogram_path = _find_required_file(
        no_thrash_run_dir,
        "post-processed/gateway/stack-context/context-usage-stacked-histogram.json",
        missing_log,
    )
    ctx_aware_job_throughput_path = _find_required_file(
        ctx_aware_run_dir,
        "post-processed/job-throughput/job-throughput-timeseries.json",
        missing_log,
    )
    context_histogram_path = _find_required_file(
        ctx_aware_run_dir,
        "post-processed/gateway/stack-context/context-usage-stacked-histogram.json",
        missing_log,
    )
    ctx_aware_timeseries_path = _find_required_file(
        ctx_aware_run_dir,
        "post-processed/gateway/ctx-aware-log/ctx-aware-timeseries.json",
        missing_log,
    )

    no_thrash_replay_summary = _load_json(no_thrash_replay_summary_path)
    ctx_aware_replay_summary = _load_json(ctx_aware_replay_summary_path)
    no_thrash_job_throughput = _load_json(no_thrash_job_throughput_path)
    no_thrash_context_histogram = _load_json(no_thrash_context_histogram_path)
    ctx_aware_job_throughput = _load_json(ctx_aware_job_throughput_path)
    context_histogram = _load_json(context_histogram_path)
    ctx_aware_timeseries = _load_json(ctx_aware_timeseries_path)
    ctx_aware_freq_summary = _load_optional_json_object(
        ctx_aware_run_dir,
        "post-processed/freq-control-linespace-instance/freq-control-summary.json",
    )

    if not all(
        isinstance(item, dict)
        for item in (
            no_thrash_replay_summary,
            ctx_aware_replay_summary,
            no_thrash_job_throughput,
            no_thrash_context_histogram,
            ctx_aware_job_throughput,
            context_histogram,
            ctx_aware_timeseries,
        )
    ):
        raise ValueError("One or more required JSON payloads are not objects")

    payload = {
        "figure_key": "con-ctrl-details",
        "generated_at_utc": _utc_now_timestamp(),
        "source_run_dirs": {
            "kairos_no_thrashing_avoidance": str(no_thrash_run_dir),
            "kairos_with_thrashing_avoidance": str(ctx_aware_run_dir),
        },
        "context_smooth_window_s": _float_or_none(args.context_smooth_window_s),
        "series": {
            "job_throughput_no_thrashing_avoidance": _materialize_job_throughput_series(
                no_thrash_job_throughput
            ),
            "context_usage_no_thrashing_avoidance": _materialize_context_series(
                no_thrash_context_histogram,
                smooth_window_s=args.context_smooth_window_s,
            ),
            "job_throughput_with_thrashing_avoidance": _materialize_job_throughput_series(
                ctx_aware_job_throughput
            ),
            "context_usage_with_thrashing_avoidance": _materialize_context_series(
                context_histogram,
                smooth_window_s=args.context_smooth_window_s,
            ),
            "pending_agent_count_with_thrashing_avoidance": _materialize_pending_agent_series(
                ctx_aware_timeseries
            ),
        },
        "metadata": {
            "dataset": "dabstep",
            "agent": "mini-swe-agent",
            "qps": 0.05,
            "no_thrashing_avoidance_started_at": _string_or_none(
                no_thrash_replay_summary.get("started_at")
            ),
            "no_thrashing_avoidance_finished_at": _string_or_none(
                no_thrash_replay_summary.get("finished_at")
            ),
            "ctx_aware_started_at": _string_or_none(ctx_aware_replay_summary.get("started_at")),
            "ctx_aware_finished_at": _string_or_none(ctx_aware_replay_summary.get("finished_at")),
            "target_context_usage_threshold": _float_or_none(
                None
                if ctx_aware_freq_summary is None
                else ctx_aware_freq_summary.get("target_context_usage_threshold")
            ),
            "ctx_aware_duration_s": _float_or_none(ctx_aware_timeseries.get("duration_s")),
            "ctx_aware_pending_agent_count_max": _float_or_none(
                (
                    ctx_aware_timeseries.get("metric_summaries", {})
                    if isinstance(ctx_aware_timeseries.get("metric_summaries"), dict)
                    else {}
                )
                .get("pending_agent_count", {})
                .get("max")
                if isinstance(
                    (
                        ctx_aware_timeseries.get("metric_summaries", {})
                        if isinstance(ctx_aware_timeseries.get("metric_summaries"), dict)
                        else {}
                    ).get("pending_agent_count"),
                    dict,
                )
                else None
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    missing_log_path.parent.mkdir(parents=True, exist_ok=True)
    missing_log_path.write_text(
        ("\n".join(missing_log) + "\n") if missing_log else "",
        encoding="utf-8",
    )

    print(f"[written] {output_path}")
    print(f"[written] {missing_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
