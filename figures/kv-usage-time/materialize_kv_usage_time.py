#!/usr/bin/env python3
"""Materialize selected KV-cache-usage time series into one figure dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "kv-usage-time"
DEFAULT_TIMESERIES_REL_PATH = Path("post-processed/vllm-log/gauge-counter-timeseries.json")
DEFAULT_RUN_SLUGS = ("core-345-1680", "core-345-1185", "core-345-660")
DEFAULT_METRIC_NAME = "vllm:kv_cache_usage_perc"
DEFAULT_ENGINE = "0"
DEFAULT_SMOOTH_WINDOW_S = 120.0
FREQUENCY_SLUG_RE = re.compile(r"^core-(?P<min>\d+)-(?P<max>\d+)(?:-mem-(?P<mem>\d+))?$")


@dataclass(frozen=True)
class RunSelection:
    run_slug: str
    run_dir: Path
    run_path: str
    frequency_mhz: int | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a multi-run KV-cache-usage dataset for the kv-usage-time figure."
        )
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory to recursively scan for the selected run slugs.",
    )
    parser.add_argument(
        "--run-slugs",
        nargs="+",
        default=list(DEFAULT_RUN_SLUGS),
        help=(
            "Run directory names to include, in plotting order. "
            f"Default: {' '.join(DEFAULT_RUN_SLUGS)}"
        ),
    )
    parser.add_argument(
        "--metric-name",
        default=DEFAULT_METRIC_NAME,
        help=f"Metric name to extract (default: {DEFAULT_METRIC_NAME}).",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help=f"Engine label to select from the vLLM metric series (default: {DEFAULT_ENGINE}).",
    )
    parser.add_argument(
        "--start-s",
        type=float,
        default=0.0,
        help="Window start in seconds from run start (default: 0).",
    )
    parser.add_argument(
        "--end-s",
        type=float,
        default=None,
        help=(
            "Optional shared window end in seconds from run start. "
            "Default: the common end available across all selected runs."
        ),
    )
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=DEFAULT_SMOOTH_WINDOW_S,
        help=(
            "Centered smoothing half-window in seconds. Each point is replaced "
            "with the average of samples in [t-w, t+w). Default: 120."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path. Default: "
            "figures/kv-usage-time/data/"
            "kv-usage-time.smooth-<window>.start-<start>.end-<end|full>.json"
        ),
    )
    return parser.parse_args()


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


def _format_window_label(value: float | None, *, default: str) -> str:
    if value is None:
        return default
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _default_output_path(
    *,
    start_s: float,
    end_s: float | None,
    smooth_window_s: float,
) -> Path:
    start_label = _format_window_label(start_s, default="0")
    end_label = _format_window_label(end_s, default="full")
    smooth_label = _format_window_label(smooth_window_s, default="120")
    return (
        DEFAULT_OUTPUT_DIR
        / f"{DEFAULT_OUTPUT_STEM}.smooth-{smooth_label}s.start-{start_label}.end-{end_label}.json"
    ).resolve()


def _parse_frequency_slug(slug: str) -> int | None:
    match = FREQUENCY_SLUG_RE.fullmatch(slug)
    if match is None:
        return None
    return int(match.group("max"))


def _discover_run_dir(root_dir: Path, run_slug: str) -> RunSelection:
    candidates: list[Path] = []
    for candidate in root_dir.rglob(run_slug):
        if not candidate.is_dir():
            continue
        if candidate.name != run_slug:
            continue
        if not (candidate / DEFAULT_TIMESERIES_REL_PATH).is_file():
            continue
        candidates.append(candidate.resolve())

    if not candidates:
        raise ValueError(f"No run directory named {run_slug!r} was found under {root_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"Expected exactly one run directory named {run_slug!r} under {root_dir}, "
            f"found {len(candidates)}: {', '.join(str(candidate) for candidate in candidates)}"
        )

    run_dir = candidates[0]
    return RunSelection(
        run_slug=run_slug,
        run_dir=run_dir,
        run_path=str(run_dir.relative_to(root_dir.resolve())),
        frequency_mhz=_parse_frequency_slug(run_slug),
    )


def _dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for time_s, value in sorted(points, key=lambda item: item[0]):
        if deduped and math.isclose(deduped[-1][0], time_s, rel_tol=0.0, abs_tol=1e-12):
            deduped[-1] = (time_s, value)
        else:
            deduped.append((time_s, value))
    return deduped


def _interpolate_value(
    left_time_s: float,
    left_value: float,
    right_time_s: float,
    right_value: float,
    target_time_s: float,
) -> float:
    if math.isclose(left_time_s, right_time_s, rel_tol=0.0, abs_tol=1e-12):
        return right_value
    ratio = (target_time_s - left_time_s) / (right_time_s - left_time_s)
    return left_value + (right_value - left_value) * ratio


def _extract_metric_points(
    *,
    timeseries_path: Path,
    metric_name: str,
    engine: str,
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    payload = _load_json(timeseries_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Timeseries JSON must be an object: {timeseries_path}")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Timeseries JSON is missing object field 'metrics': {timeseries_path}")

    matches: list[dict[str, Any]] = []
    for metric_payload in metrics.values():
        if not isinstance(metric_payload, dict):
            continue
        if metric_payload.get("name") != metric_name:
            continue
        labels = metric_payload.get("labels")
        if not isinstance(labels, dict):
            continue
        if str(labels.get("engine")) != engine:
            continue
        matches.append(metric_payload)

    if not matches:
        raise ValueError(
            f"No metric series matched name={metric_name!r}, engine={engine!r} in {timeseries_path}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Expected exactly one metric series for name={metric_name!r}, engine={engine!r} "
            f"in {timeseries_path}, found {len(matches)}"
        )

    metric_payload = matches[0]
    raw_times = metric_payload.get("time_from_start_s")
    raw_values = metric_payload.get("value")
    if not isinstance(raw_times, list) or not isinstance(raw_values, list):
        raise ValueError(
            f"Metric series is missing aligned time/value arrays in {timeseries_path}"
        )

    pairs: list[tuple[float, float]] = []
    for raw_time, raw_value in zip(raw_times, raw_values):
        time_s = _float_or_none(raw_time)
        value = _float_or_none(raw_value)
        if time_s is None or value is None:
            continue
        pairs.append((time_s, value))

    pairs = _dedupe_points(pairs)
    if not pairs:
        raise ValueError(f"No usable points were found in {timeseries_path}")
    return metric_payload, pairs


def _clip_points(
    points: list[tuple[float, float]],
    *,
    start_s: float,
    end_s: float,
) -> list[tuple[float, float]]:
    if end_s <= start_s:
        return []
    deduped_points = _dedupe_points(points)
    if not deduped_points:
        return []

    times = [point[0] for point in deduped_points]
    values = [point[1] for point in deduped_points]
    if start_s < times[0] or end_s > times[-1]:
        raise ValueError(
            f"Requested window [{start_s}, {end_s}] is outside available metric range "
            f"[{times[0]}, {times[-1]}]"
        )

    def value_at(time_s: float) -> float:
        if time_s <= times[0]:
            return values[0]
        if time_s >= times[-1]:
            return values[-1]
        for index in range(1, len(times)):
            left_time_s = times[index - 1]
            right_time_s = times[index]
            if left_time_s <= time_s <= right_time_s:
                if math.isclose(left_time_s, time_s, rel_tol=0.0, abs_tol=1e-12):
                    return values[index - 1]
                return _interpolate_value(
                    left_time_s,
                    values[index - 1],
                    right_time_s,
                    values[index],
                    time_s,
                )
        return values[-1]

    clipped: list[tuple[float, float]] = [(start_s, value_at(start_s))]
    clipped.extend(
        (time_s, value)
        for time_s, value in deduped_points
        if start_s < time_s < end_s
    )
    clipped.append((end_s, value_at(end_s)))
    return _dedupe_points(clipped)


def _smooth_values_with_centered_window(
    x_values: list[float],
    y_values: list[float],
    *,
    window_size_s: float,
) -> list[float]:
    if not x_values or not y_values:
        return []
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values length mismatch while smoothing")
    if window_size_s <= 0.0:
        return list(y_values)

    prefix_sums = [0.0]
    for value in y_values:
        prefix_sums.append(prefix_sums[-1] + value)

    smoothed_values: list[float] = []
    left = 0
    right = 0
    point_count = len(x_values)
    for index in range(point_count):
        center_time = x_values[index]
        left_bound = center_time - window_size_s
        right_bound = center_time + window_size_s

        while left < point_count and x_values[left] <= left_bound:
            left += 1
        while right < point_count and x_values[right] < right_bound:
            right += 1

        if right <= left:
            smoothed_values.append(y_values[index])
            continue

        window_sum = prefix_sums[right] - prefix_sums[left]
        smoothed_values.append(window_sum / (right - left))
    return smoothed_values


def _build_stats(x_values: list[float], y_values: list[float]) -> dict[str, float | int]:
    if not x_values or not y_values:
        raise ValueError("Cannot build stats for an empty point list")
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values length mismatch while building stats")
    peak_index = max(range(len(y_values)), key=lambda index: y_values[index])
    return {
        "sample_count": len(y_values),
        "avg": round(sum(y_values) / len(y_values), 6),
        "min": round(min(y_values), 6),
        "max": round(max(y_values), 6),
        "peak_time_s": round(x_values[peak_index], 6),
    }


def materialize_kv_usage_time(
    *,
    root_dir: Path,
    run_slugs: list[str],
    metric_name: str,
    engine: str,
    start_s: float,
    end_s: float | None,
    smooth_window_s: float,
) -> dict[str, Any]:
    resolved_root_dir = root_dir.expanduser().resolve()
    if not resolved_root_dir.is_dir():
        raise ValueError(f"Root directory not found: {resolved_root_dir}")
    if start_s < 0.0:
        raise ValueError(f"--start-s must be non-negative, got {start_s}")
    if smooth_window_s <= 0.0:
        raise ValueError(f"--smooth-window-s must be positive, got {smooth_window_s}")

    selections = [_discover_run_dir(resolved_root_dir, run_slug) for run_slug in run_slugs]
    extracted_series: list[dict[str, Any]] = []
    common_available_end_s: float | None = None
    for selection in selections:
        timeseries_path = (selection.run_dir / DEFAULT_TIMESERIES_REL_PATH).resolve()
        metric_payload, points = _extract_metric_points(
            timeseries_path=timeseries_path,
            metric_name=metric_name,
            engine=engine,
        )
        available_end_s = points[-1][0]
        if common_available_end_s is None or available_end_s < common_available_end_s:
            common_available_end_s = available_end_s
        extracted_series.append(
            {
                "selection": selection,
                "timeseries_path": timeseries_path,
                "metric_payload": metric_payload,
                "points": points,
                "available_end_s": available_end_s,
            }
        )

    assert common_available_end_s is not None
    resolved_end_s = common_available_end_s if end_s is None else end_s
    if resolved_end_s > common_available_end_s:
        raise ValueError(
            f"--end-s {resolved_end_s} exceeds the common available end {common_available_end_s}"
        )
    if resolved_end_s <= start_s:
        raise ValueError(
            f"Window end must be greater than start: start={start_s}, end={resolved_end_s}"
        )

    series_payload: list[dict[str, Any]] = []
    for index, extracted in enumerate(extracted_series):
        selection = extracted["selection"]
        clipped_points = _clip_points(
            extracted["points"],
            start_s=start_s,
            end_s=resolved_end_s,
        )
        x_values = [time_s for time_s, _ in clipped_points]
        raw_y_values = [value for _, value in clipped_points]
        smoothed_y_values = _smooth_values_with_centered_window(
            x_values,
            raw_y_values,
            window_size_s=smooth_window_s,
        )
        stats = _build_stats(x_values, smoothed_y_values)
        labels = extracted["metric_payload"].get("labels")
        series_payload.append(
            {
                "series_index": index,
                "run_slug": selection.run_slug,
                "series_label": selection.run_slug.removeprefix("core-"),
                "frequency_mhz": selection.frequency_mhz,
                "run_path": selection.run_path,
                "source_run_dir": str(selection.run_dir),
                "source_timeseries_path": str(extracted["timeseries_path"]),
                "metric_name": extracted["metric_payload"].get("name"),
                "metric_labels": labels if isinstance(labels, dict) else {},
                "available_end_s": round(float(extracted["available_end_s"]), 6),
                "stats": stats,
                "points": [
                    {
                        "time_from_start_s": round(time_s, 6),
                        "raw_value": round(raw_value, 6),
                        "value": round(smoothed_value, 6),
                        "smoothed_value": round(smoothed_value, 6),
                    }
                    for time_s, raw_value, smoothed_value in zip(
                        x_values,
                        raw_y_values,
                        smoothed_y_values,
                    )
                ],
            }
        )

    return {
        "source_root_dir": str(resolved_root_dir),
        "metric_name": metric_name,
        "engine": engine,
        "run_slugs": run_slugs,
        "analysis_window_start_s": round(start_s, 6),
        "analysis_window_end_s": round(resolved_end_s, 6),
        "analysis_window_duration_s": round(resolved_end_s - start_s, 6),
        "common_available_end_s": round(common_available_end_s, 6),
        "smooth_window_s": round(smooth_window_s, 6),
        "series_count": len(series_payload),
        "value_semantics": (
            "smoothed fraction where 1.0 means 100 percent KV-cache usage; "
            "smoothing uses a centered sample-average window [t-w, t+w)"
        ),
        "series": series_payload,
    }


def main() -> int:
    args = _parse_args()
    payload = materialize_kv_usage_time(
        root_dir=Path(args.root_dir),
        run_slugs=list(args.run_slugs),
        metric_name=args.metric_name,
        engine=args.engine,
        start_s=float(args.start_s),
        end_s=(float(args.end_s) if args.end_s is not None else None),
        smooth_window_s=float(args.smooth_window_s),
    )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(
            start_s=float(args.start_s),
            end_s=(float(args.end_s) if args.end_s is not None else None),
            smooth_window_s=float(args.smooth_window_s),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
