#!/usr/bin/env python3
"""Materialize frequency-sweep energy, throughput, and LLM-time metrics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import json
import math
import re
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "freq-vs-average-energy"
FREQUENCY_SLUG_RE = re.compile(r"^core-(?P<min>\d+)-(?P<max>\d+)(?:-mem-(?P<mem>\d+))?$")
CSV_FIELDNAMES = (
    "run_path",
    "source_run_dir",
    "frequency_mhz",
    "core_min_mhz",
    "core_max_mhz",
    "mem_freq_mhz",
    "analysis_window_available_end_s",
    "analysis_window_start_s",
    "analysis_window_end_s",
    "analysis_window_duration_s",
    "power_sample_count_in_window",
    "window_avg_power_w",
    "window_energy_estimate_j",
    "window_energy_integral_j",
    "finished_replay_count_in_window",
    "llm_request_count_in_window",
    "llm_request_with_model_inference_time_count_in_window",
    "average_request_time_in_llm_s",
    "average_throughput_jobs_per_s",
    "average_energy_per_finished_replay_j",
    "average_integral_energy_per_finished_replay_j",
)


@dataclass(frozen=True)
class RunMetadata:
    run_dir: Path
    run_path: str
    frequency_mhz: int
    core_min_mhz: int
    core_max_mhz: int
    mem_freq_mhz: int | None


def _utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a per-frequency CSV for the "
            "freq-vs-average-energy-duration-throughtpu figure. "
            "The CSV stores one row per frequency/run for the selected time window."
        )
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory that contains the frequency sweep results.",
    )
    parser.add_argument(
        "--start-s",
        type=float,
        default=0.0,
        help="Window start in seconds from experiment start (default: 0).",
    )
    parser.add_argument(
        "--end-s",
        type=float,
        default=None,
        help="Window end in seconds from experiment start. Default: each run's full analysis window.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV path. Default: "
            "figures/freq-vs-average-energy-duration-throughtpu/data/"
            "freq-vs-average-energy.start-<start>.end-<end|full>.csv"
        ),
    )
    return parser.parse_args()


def _parse_frequency_slug(slug: str) -> tuple[int, int, int | None] | None:
    match = FREQUENCY_SLUG_RE.fullmatch(slug)
    if match is None:
        return None
    core_min_mhz = int(match.group("min"))
    core_max_mhz = int(match.group("max"))
    mem_group = match.group("mem")
    mem_freq_mhz = int(mem_group) if mem_group is not None else None
    return (core_min_mhz, core_max_mhz, mem_freq_mhz)


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


def _parse_iso8601_to_utc(value: Any) -> datetime | None:
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


def _duration_s(start_utc: datetime | None, end_utc: datetime | None) -> float | None:
    if start_utc is None or end_utc is None:
        return None
    return round((end_utc - start_utc).total_seconds(), 6)


def _format_window_label(value: float | None, *, default: str) -> str:
    if value is None:
        return default
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _default_output_path(start_s: float, end_s: float | None) -> Path:
    start_label = _format_window_label(start_s, default="0")
    end_label = _format_window_label(end_s, default="full")
    return (DEFAULT_OUTPUT_DIR / f"{DEFAULT_OUTPUT_STEM}.start-{start_label}.end-{end_label}.csv").resolve()


def _discover_run_dirs(root_dir: Path) -> list[RunMetadata]:
    discovered: list[RunMetadata] = []
    for candidate in root_dir.rglob("*"):
        if not candidate.is_dir():
            continue
        parsed_slug = _parse_frequency_slug(candidate.name)
        if parsed_slug is None:
            continue
        if not (candidate / "replay" / "summary.json").is_file():
            continue
        if not (candidate / "post-processed" / "power" / "power-summary.json").is_file():
            continue
        if not (candidate / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json").is_file():
            continue
        core_min_mhz, core_max_mhz, mem_freq_mhz = parsed_slug
        discovered.append(
            RunMetadata(
                run_dir=candidate.resolve(),
                run_path=str(candidate.resolve().relative_to(root_dir.resolve())),
                frequency_mhz=core_max_mhz,
                core_min_mhz=core_min_mhz,
                core_max_mhz=core_max_mhz,
                mem_freq_mhz=mem_freq_mhz,
            )
        )

    discovered.sort(key=lambda item: (item.frequency_mhz, item.run_path))
    return discovered


def _extract_power_window_duration_s(power_summary_payload: dict[str, Any]) -> float:
    start_utc = _parse_iso8601_to_utc(power_summary_payload.get("analysis_window_start_utc"))
    end_utc = _parse_iso8601_to_utc(power_summary_payload.get("analysis_window_end_utc"))
    duration_s = _duration_s(start_utc, end_utc)
    if duration_s is not None and duration_s >= 0.0:
        return duration_s

    points = power_summary_payload.get("power_points")
    if not isinstance(points, list) or not points:
        raise ValueError("power-summary.json is missing a usable analysis window")

    last_offset_s = max(
        _float_or_none(point.get("time_offset_s")) or 0.0
        for point in points
        if isinstance(point, dict)
    )
    return round(last_offset_s, 6)


def _extract_power_points_in_window(
    power_summary_payload: dict[str, Any],
    *,
    start_s: float,
    end_s: float,
) -> list[tuple[float, float]]:
    raw_points = power_summary_payload.get("power_points")
    if not isinstance(raw_points, list):
        raise ValueError("power-summary.json is missing power_points")

    points: list[tuple[float, float]] = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        time_offset_s = _float_or_none(item.get("time_offset_s"))
        power_w = _float_or_none(item.get("power_w"))
        if time_offset_s is None or power_w is None:
            continue
        if start_s <= time_offset_s <= end_s:
            points.append((time_offset_s, power_w))

    points.sort(key=lambda pair: pair[0])
    return points


def _average_power_w(points: list[tuple[float, float]]) -> float | None:
    if not points:
        return None
    return round(sum(power_w for _, power_w in points) / len(points), 6)


def _dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for time_s, power_w in sorted(points, key=lambda pair: pair[0]):
        if deduped and math.isclose(deduped[-1][0], time_s, rel_tol=0.0, abs_tol=1e-9):
            deduped[-1] = (time_s, power_w)
        else:
            deduped.append((time_s, power_w))
    return deduped


def _interpolate_power_w(
    left_point: tuple[float, float],
    right_point: tuple[float, float],
    target_time_s: float,
) -> float:
    left_time_s, left_power_w = left_point
    right_time_s, right_power_w = right_point
    if math.isclose(left_time_s, right_time_s, rel_tol=0.0, abs_tol=1e-12):
        return right_power_w
    ratio = (target_time_s - left_time_s) / (right_time_s - left_time_s)
    return left_power_w + (right_power_w - left_power_w) * ratio


def _clip_power_points_for_integral(
    all_points: list[tuple[float, float]],
    *,
    start_s: float,
    end_s: float,
) -> list[tuple[float, float]]:
    if not all_points:
        return []

    sorted_points = _dedupe_points(all_points)
    in_window = [(time_s, power_w) for time_s, power_w in sorted_points if start_s <= time_s <= end_s]
    clipped = list(in_window)

    first_time_s, first_power_w = sorted_points[0]
    last_time_s, last_power_w = sorted_points[-1]

    if start_s < first_time_s or end_s > last_time_s:
        return clipped

    for index in range(1, len(sorted_points)):
        left_point = sorted_points[index - 1]
        right_point = sorted_points[index]
        left_time_s = left_point[0]
        right_time_s = right_point[0]
        if left_time_s <= start_s <= right_time_s:
            clipped.append((start_s, _interpolate_power_w(left_point, right_point, start_s)))
            break

    for index in range(1, len(sorted_points)):
        left_point = sorted_points[index - 1]
        right_point = sorted_points[index]
        left_time_s = left_point[0]
        right_time_s = right_point[0]
        if left_time_s <= end_s <= right_time_s:
            clipped.append((end_s, _interpolate_power_w(left_point, right_point, end_s)))
            break

    return _dedupe_points([(time_s, power_w) for time_s, power_w in clipped if start_s <= time_s <= end_s])


def _integrate_energy_j(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    energy_j = 0.0
    previous_time_s, previous_power_w = points[0]
    for time_s, power_w in points[1:]:
        delta_s = time_s - previous_time_s
        if delta_s > 0:
            energy_j += ((previous_power_w + power_w) / 2.0) * delta_s
        previous_time_s = time_s
        previous_power_w = power_w
    return round(energy_j, 6)


def _extract_finish_offsets_s(run_dir: Path, *, available_end_s: float) -> list[float]:
    replay_summary_path = run_dir / "replay" / "summary.json"
    payload = _load_json(replay_summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Replay summary must be a JSON object: {replay_summary_path}")

    experiment_start_utc = _parse_iso8601_to_utc(payload.get("started_at"))
    if experiment_start_utc is None:
        raise ValueError(f"Replay summary is missing started_at: {replay_summary_path}")

    worker_results = payload.get("worker_results")
    if not isinstance(worker_results, dict):
        raise ValueError(f"Replay summary is missing worker_results: {replay_summary_path}")

    finish_offsets_s: list[float] = []
    for worker_payload in worker_results.values():
        if not isinstance(worker_payload, dict):
            continue
        finish_utc = _parse_iso8601_to_utc(worker_payload.get("finished_at"))
        if finish_utc is None:
            continue
        offset_s = _duration_s(experiment_start_utc, finish_utc)
        if offset_s is None:
            continue
        if 0.0 <= offset_s <= available_end_s:
            finish_offsets_s.append(offset_s)

    finish_offsets_s.sort()
    return finish_offsets_s


def _count_finishes_in_window(
    finish_offsets_s: list[float],
    *,
    start_s: float,
    end_s: float,
) -> int:
    return sum(1 for offset_s in finish_offsets_s if start_s <= offset_s <= end_s)


def _extract_llm_time_metrics_in_window(
    run_dir: Path,
    *,
    available_end_s: float,
    start_s: float,
    end_s: float,
) -> tuple[int, int, float | None]:
    llm_requests_path = (
        run_dir / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json"
    )
    payload = _load_json(llm_requests_path)
    if not isinstance(payload, dict):
        raise ValueError(f"LLM requests file must be a JSON object: {llm_requests_path}")

    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        raise ValueError(f"LLM requests file is missing requests: {llm_requests_path}")

    llm_request_count_in_window = 0
    model_inference_times_s: list[float] = []
    for request in raw_requests:
        if not isinstance(request, dict):
            continue
        request_end_offset_s = _float_or_none(request.get("request_end_offset_s"))
        if request_end_offset_s is None:
            continue
        if request_end_offset_s < 0.0 or request_end_offset_s > available_end_s:
            continue
        if not (start_s <= request_end_offset_s <= end_s):
            continue

        llm_request_count_in_window += 1
        model_inference_time_s = _float_or_none(
            request.get("gen_ai.latency.time_in_model_inference")
        )
        if model_inference_time_s is None:
            continue
        model_inference_times_s.append(model_inference_time_s)

    average_request_time_in_llm_s: float | None = None
    if model_inference_times_s:
        average_request_time_in_llm_s = round(
            sum(model_inference_times_s) / len(model_inference_times_s),
            6,
        )
    return (
        llm_request_count_in_window,
        len(model_inference_times_s),
        average_request_time_in_llm_s,
    )


def _build_row(
    metadata: RunMetadata,
    *,
    start_s: float,
    requested_end_s: float | None,
) -> dict[str, Any]:
    power_summary_path = metadata.run_dir / "post-processed" / "power" / "power-summary.json"
    power_summary_payload = _load_json(power_summary_path)
    if not isinstance(power_summary_payload, dict):
        raise ValueError(f"Invalid power summary JSON: {power_summary_path}")

    available_end_s = _extract_power_window_duration_s(power_summary_payload)
    if start_s < 0.0:
        raise ValueError(f"--start-s must be non-negative, got {start_s}")
    if start_s >= available_end_s:
        raise ValueError(
            f"Window start {start_s} is outside the available range for {metadata.run_path} "
            f"(available end: {available_end_s})"
        )

    if requested_end_s is None:
        end_s = available_end_s
    else:
        end_s = requested_end_s
        if end_s > available_end_s:
            raise ValueError(
                f"Window end {end_s} is outside the available range for {metadata.run_path} "
                f"(available end: {available_end_s})"
            )

    if end_s <= start_s:
        raise ValueError(
            f"Window end must be greater than start for {metadata.run_path}: "
            f"start={start_s}, end={end_s}"
        )

    window_duration_s = round(end_s - start_s, 6)
    in_window_power_points = _extract_power_points_in_window(
        power_summary_payload,
        start_s=start_s,
        end_s=end_s,
    )
    if not in_window_power_points:
        raise ValueError(
            f"No power samples were found inside [{start_s}, {end_s}] for {metadata.run_path}"
        )

    window_avg_power_w = _average_power_w(in_window_power_points)
    assert window_avg_power_w is not None
    window_energy_estimate_j = round(window_avg_power_w * window_duration_s, 6)

    all_power_points = _extract_power_points_in_window(
        power_summary_payload,
        start_s=0.0,
        end_s=available_end_s,
    )
    clipped_integral_points = _clip_power_points_for_integral(
        all_power_points,
        start_s=start_s,
        end_s=end_s,
    )
    window_energy_integral_j = _integrate_energy_j(clipped_integral_points)

    finish_offsets_s = _extract_finish_offsets_s(metadata.run_dir, available_end_s=available_end_s)
    finished_replay_count_in_window = _count_finishes_in_window(
        finish_offsets_s,
        start_s=start_s,
        end_s=end_s,
    )
    (
        llm_request_count_in_window,
        llm_request_with_model_inference_time_count_in_window,
        average_request_time_in_llm_s,
    ) = _extract_llm_time_metrics_in_window(
        metadata.run_dir,
        available_end_s=available_end_s,
        start_s=start_s,
        end_s=end_s,
    )
    average_throughput_jobs_per_s = round(finished_replay_count_in_window / window_duration_s, 6)

    average_energy_per_finished_replay_j: float | None = None
    average_integral_energy_per_finished_replay_j: float | None = None
    if finished_replay_count_in_window > 0:
        average_energy_per_finished_replay_j = round(
            window_energy_estimate_j / finished_replay_count_in_window,
            6,
        )
        average_integral_energy_per_finished_replay_j = round(
            window_energy_integral_j / finished_replay_count_in_window,
            6,
        )

    return {
        "run_path": metadata.run_path,
        "source_run_dir": str(metadata.run_dir),
        "frequency_mhz": metadata.frequency_mhz,
        "core_min_mhz": metadata.core_min_mhz,
        "core_max_mhz": metadata.core_max_mhz,
        "mem_freq_mhz": metadata.mem_freq_mhz,
        "analysis_window_available_end_s": round(available_end_s, 6),
        "analysis_window_start_s": round(start_s, 6),
        "analysis_window_end_s": round(end_s, 6),
        "analysis_window_duration_s": window_duration_s,
        "power_sample_count_in_window": len(in_window_power_points),
        "window_avg_power_w": window_avg_power_w,
        "window_energy_estimate_j": window_energy_estimate_j,
        "window_energy_integral_j": window_energy_integral_j,
        "finished_replay_count_in_window": finished_replay_count_in_window,
        "llm_request_count_in_window": llm_request_count_in_window,
        "llm_request_with_model_inference_time_count_in_window": (
            llm_request_with_model_inference_time_count_in_window
        ),
        "average_request_time_in_llm_s": average_request_time_in_llm_s,
        "average_throughput_jobs_per_s": average_throughput_jobs_per_s,
        "average_energy_per_finished_replay_j": average_energy_per_finished_replay_j,
        "average_integral_energy_per_finished_replay_j": average_integral_energy_per_finished_replay_j,
    }


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: _csv_value(row.get(fieldname)) for fieldname in CSV_FIELDNAMES})


def main() -> int:
    args = _parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(args.start_s, args.end_s)
    )

    run_dirs = _discover_run_dirs(root_dir)
    if not run_dirs:
        raise SystemExit(f"No frequency-sweep run directories were found under {root_dir}")

    rows = [
        _build_row(
            metadata,
            start_s=float(args.start_s),
            requested_end_s=(float(args.end_s) if args.end_s is not None else None),
        )
        for metadata in run_dirs
    ]
    _write_csv(output_path, rows)

    print(f"[written] {output_path}")
    print(f"[rows] {len(rows)}")
    print(f"[generated_at_utc] {_utc_now_timestamp()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
