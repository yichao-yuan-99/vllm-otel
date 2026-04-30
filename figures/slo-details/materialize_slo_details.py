#!/usr/bin/env python3
"""Materialize the slo-details figure dataset from one replay run."""

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
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "slo-details.json"
DEFAULT_MISSING_LOG_PATH = DEFAULT_OUTPUT_DIR / "slo-details.missing.log"
DEFAULT_RUN_PATH = Path(
    "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
    "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/"
    "dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/35/20260412T055419Z"
)
DEFAULT_POWER_SMOOTH_WINDOW_S = 120.0
DEFAULT_CONTEXT_SMOOTH_WINDOW_S = 120.0


def _utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the four-panel slo-details figure dataset."
    )
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_PATH),
        help=(
            "Replay run directory, or a parent directory containing timestamped runs. "
            f"Default: {DEFAULT_RUN_PATH}"
        ),
    )
    parser.add_argument(
        "--power-smooth-window-s",
        type=float,
        default=DEFAULT_POWER_SMOOTH_WINDOW_S,
        help="Centered smoothing window for power in seconds (default: 120).",
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


def _materialize_power_series(
    power_payload: dict[str, Any],
    *,
    smooth_window_s: float,
) -> list[dict[str, float]]:
    raw_points = power_payload.get("power_points")
    if not isinstance(raw_points, list):
        raise ValueError("power-summary.json is missing power_points")

    points: list[tuple[float, float]] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            continue
        time_s = _float_or_none(raw_point.get("time_offset_s"))
        power_w = _float_or_none(raw_point.get("power_w"))
        if time_s is None or power_w is None:
            continue
        points.append((time_s, power_w))

    points.sort(key=lambda item: item[0])
    smoothed = _rolling_average(points, window_s=smooth_window_s)
    collapsed = _collapse_to_last_value_per_second(smoothed)
    return [
        {
            "time_offset_s": _stable_round(time_s),
            "power_w": _stable_round(power_w),
        }
        for time_s, power_w in collapsed
    ]


def _materialize_context_series(
    histogram_payload: dict[str, Any],
    *,
    smooth_window_s: float,
) -> list[dict[str, float]]:
    raw_points = histogram_payload.get("points")
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


def _materialize_frequency_series(freq_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_points = freq_payload.get("decision_points")
    if not isinstance(raw_points, list):
        raise ValueError("freq-control-summary.json is missing decision_points")

    series: list[dict[str, Any]] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            continue
        time_s = _float_or_none(raw_point.get("time_offset_s"))
        frequency_mhz = _float_or_none(raw_point.get("target_frequency_mhz"))
        if time_s is None or frequency_mhz is None:
            continue
        series.append(
            {
                "time_offset_s": _stable_round(time_s),
                "target_frequency_mhz": int(round(frequency_mhz)),
                "changed": bool(raw_point.get("changed")),
                "action": _string_or_none(raw_point.get("action")) or "unknown",
                "slo_override_applied": bool(raw_point.get("slo_override_applied")),
            }
        )
    return series


def _materialize_slo_series(slo_payload: dict[str, Any]) -> tuple[float | None, list[dict[str, Any]]]:
    target_value = _float_or_none(slo_payload.get("target_output_throughput_tokens_per_s"))
    raw_points = slo_payload.get("decision_points")
    if not isinstance(raw_points, list):
        raise ValueError("slo-decision-summary.json is missing decision_points")

    series: list[dict[str, Any]] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            continue
        time_s = _float_or_none(raw_point.get("time_offset_s"))
        throughput = _float_or_none(raw_point.get("window_min_output_tokens_per_s"))
        if time_s is None or throughput is None:
            continue
        series.append(
            {
                "time_offset_s": _stable_round(time_s),
                "window_min_output_tokens_per_s": _stable_round(throughput),
                "changed": bool(raw_point.get("changed")),
                "action": _string_or_none(raw_point.get("action")) or "unknown",
                "target_frequency_mhz": int(round(_float_or_none(raw_point.get("target_frequency_mhz")) or 0.0)),
            }
        )
    return target_value, series


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    missing_log_path = Path(args.missing_log).expanduser().resolve()

    missing_log: list[str] = []
    run_dir = _select_run_dir(Path(args.run_dir).expanduser().resolve(), missing_log)

    replay_summary_path = _find_required_file(run_dir, "replay/summary.json", missing_log)
    power_summary_path = _find_required_file(
        run_dir,
        "post-processed/power/power-summary.json",
        missing_log,
    )
    freq_summary_path = _find_required_file(
        run_dir,
        "post-processed/freq-control-linespace-instance-slo/freq-control-summary.json",
        missing_log,
    )
    slo_summary_path = _find_required_file(
        run_dir,
        "post-processed/slo-decision/slo-decision-summary.json",
        missing_log,
    )
    context_histogram_path = _find_required_file(
        run_dir,
        "post-processed/gateway/stack-context/context-usage-stacked-histogram.json",
        missing_log,
    )

    replay_summary = _load_json(replay_summary_path)
    power_summary = _load_json(power_summary_path)
    freq_summary = _load_json(freq_summary_path)
    slo_summary = _load_json(slo_summary_path)
    context_histogram = _load_json(context_histogram_path)

    if not all(isinstance(item, dict) for item in (replay_summary, power_summary, freq_summary, slo_summary, context_histogram)):
        raise ValueError("One or more required JSON payloads are not objects")

    slo_target_tokens_per_s, slo_points = _materialize_slo_series(slo_summary)

    payload = {
        "figure_key": "slo-details",
        "generated_at_utc": _utc_now_timestamp(),
        "source_run_dir": str(run_dir),
        "experiment_started_at": _string_or_none(replay_summary.get("started_at")),
        "experiment_finished_at": _string_or_none(replay_summary.get("finished_at")),
        "time_constraint_s": _float_or_none(replay_summary.get("time_constraint_s")),
        "power_smooth_window_s": _float_or_none(args.power_smooth_window_s),
        "context_smooth_window_s": _float_or_none(args.context_smooth_window_s),
        "series": {
            "power": _materialize_power_series(
                power_summary,
                smooth_window_s=args.power_smooth_window_s,
            ),
            "frequency": _materialize_frequency_series(freq_summary),
            "slo": slo_points,
            "context_usage": _materialize_context_series(
                context_histogram,
                smooth_window_s=args.context_smooth_window_s,
            ),
        },
        "metadata": {
            "target_context_usage_threshold": _float_or_none(
                freq_summary.get("target_context_usage_threshold")
            ),
            "min_frequency_mhz": _float_or_none(freq_summary.get("min_frequency_mhz")),
            "max_frequency_mhz": _float_or_none(freq_summary.get("max_frequency_mhz")),
            "slo_target_tokens_per_s": slo_target_tokens_per_s,
            "max_context_usage": _float_or_none(freq_summary.get("max_context_usage")),
            "max_window_context_usage": _float_or_none(freq_summary.get("max_window_context_usage")),
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
