#!/usr/bin/env python3
"""Materialize frequency-sweep completed-agent request-energy metrics."""

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
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "freq-vs-average-energy-integral"
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
    "num_requests_running_metric_series_count",
    "num_requests_running_sample_count_in_window",
    "total_llm_request_count",
    "selected_finished_replay_count_in_window",
    "mapped_finished_replay_count_in_window",
    "unmapped_selected_finished_replay_count_in_window",
    "request_count_for_mapped_finished_replays",
    "request_count_overlapping_window_for_mapped_finished_replays",
    "traces_with_request_energy_in_window",
    "total_request_integral_energy_across_finished_replays_j",
    "average_request_integral_energy_per_finished_replay_j",
    "average_throughput_jobs_per_s",
    "window_total_power_energy_integral_j",
    "average_window_total_power_energy_per_finished_replay_j",
)


@dataclass(frozen=True)
class RunMetadata:
    run_dir: Path
    run_path: str
    frequency_mhz: int
    core_min_mhz: int
    core_max_mhz: int
    mem_freq_mhz: int | None


@dataclass(frozen=True)
class SampleSeries:
    times_s: tuple[float, ...]
    values: tuple[float, ...]


class PiecewiseLinearIntegral:
    def __init__(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 2:
            raise ValueError("at least two points are required to build an integral")
        deduped_points = _dedupe_points(points)
        if len(deduped_points) < 2:
            raise ValueError("at least two distinct timestamps are required to build an integral")
        self._times = [point[0] for point in deduped_points]
        self._values = [point[1] for point in deduped_points]
        self._cumulative_j = [0.0]
        for index in range(1, len(deduped_points)):
            previous_time_s = self._times[index - 1]
            previous_value = self._values[index - 1]
            current_time_s = self._times[index]
            current_value = self._values[index]
            delta_s = current_time_s - previous_time_s
            if delta_s < 0:
                raise ValueError("series timestamps must be sorted")
            area_j = ((previous_value + current_value) / 2.0) * delta_s
            self._cumulative_j.append(self._cumulative_j[-1] + area_j)

    @property
    def start_s(self) -> float:
        return self._times[0]

    @property
    def end_s(self) -> float:
        return self._times[-1]

    def integral_between(self, start_s: float, end_s: float) -> float:
        clipped_start_s = max(start_s, self.start_s)
        clipped_end_s = min(end_s, self.end_s)
        if clipped_end_s <= clipped_start_s:
            return 0.0
        return round(
            self._integral_upto(clipped_end_s) - self._integral_upto(clipped_start_s),
            6,
        )

    def _integral_upto(self, time_s: float) -> float:
        if time_s <= self._times[0]:
            return 0.0
        if time_s >= self._times[-1]:
            return self._cumulative_j[-1]

        left_index = _bisect_right(self._times, time_s) - 1
        left_time_s = self._times[left_index]
        left_value = self._values[left_index]
        if math.isclose(left_time_s, time_s, rel_tol=0.0, abs_tol=1e-12):
            return self._cumulative_j[left_index]

        right_index = left_index + 1
        right_time_s = self._times[right_index]
        right_value = self._values[right_index]
        interpolated_value = _interpolate_value(
            left_time_s,
            left_value,
            right_time_s,
            right_value,
            time_s,
        )
        delta_s = time_s - left_time_s
        area_j = ((left_value + interpolated_value) / 2.0) * delta_s
        return self._cumulative_j[left_index] + area_j


def _utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a per-frequency CSV for the integral freq-vs-average-energy figure. "
            "This version allocates power to requests using num_requests_running and "
            "averages completed-agent request energy."
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
        help="Window end in seconds from experiment start. Default: full available window.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV path. Default: "
            "figures/freq-vs-average-energy-integral/data/"
            "freq-vs-average-energy-integral.start-<start>.end-<end|full>.csv"
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
    return (
        DEFAULT_OUTPUT_DIR
        / f"{DEFAULT_OUTPUT_STEM}.start-{start_label}.end-{end_label}.csv"
    ).resolve()


def _discover_run_dirs(root_dir: Path) -> list[RunMetadata]:
    discovered: list[RunMetadata] = []
    for candidate in root_dir.rglob("*"):
        if not candidate.is_dir():
            continue
        parsed_slug = _parse_frequency_slug(candidate.name)
        if parsed_slug is None:
            continue
        required_paths = (
            candidate / "replay" / "summary.json",
            candidate / "post-processed" / "power" / "power-summary.json",
            candidate / "post-processed" / "vllm-log" / "gauge-counter-timeseries.json",
            candidate / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json",
            candidate / "post-processed" / "global" / "trial-timing-summary.json",
        )
        if not all(path.is_file() for path in required_paths):
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

    power_points = _extract_power_points(power_summary_payload)
    if not power_points:
        raise ValueError("power-summary.json is missing a usable analysis window")
    return round(power_points[-1][0], 6)


def _extract_power_points(power_summary_payload: dict[str, Any]) -> list[tuple[float, float]]:
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
        points.append((time_offset_s, power_w))

    return _dedupe_points(points)


def _extract_num_requests_running_series(vllm_timeseries_payload: dict[str, Any]) -> list[SampleSeries]:
    metrics = vllm_timeseries_payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("gauge-counter-timeseries.json is missing metrics")

    series_list: list[SampleSeries] = []
    for entry in metrics.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("name") != "vllm:num_requests_running":
            continue
        raw_times = entry.get("time_from_start_s")
        raw_values = entry.get("value")
        if not isinstance(raw_times, list) or not isinstance(raw_values, list):
            continue

        pairs: list[tuple[float, float]] = []
        for raw_time_s, raw_value in zip(raw_times, raw_values):
            time_s = _float_or_none(raw_time_s)
            value = _float_or_none(raw_value)
            if time_s is None or value is None:
                continue
            pairs.append((time_s, max(value, 0.0)))
        if not pairs:
            continue
        deduped_pairs = _dedupe_points(pairs)
        series_list.append(
            SampleSeries(
                times_s=tuple(pair[0] for pair in deduped_pairs),
                values=tuple(pair[1] for pair in deduped_pairs),
            )
        )

    if not series_list:
        raise ValueError("No vllm:num_requests_running series were found")
    return series_list


def _running_series_available_end_s(series_list: list[SampleSeries]) -> float:
    return min(series.times_s[-1] for series in series_list)


def _bisect_right(sorted_values: list[float] | tuple[float, ...], target: float) -> int:
    low = 0
    high = len(sorted_values)
    while low < high:
        mid = (low + high) // 2
        if target < sorted_values[mid]:
            high = mid
        else:
            low = mid + 1
    return low


def _evaluate_piecewise_constant(series: SampleSeries, time_s: float) -> float:
    if time_s <= series.times_s[0]:
        return series.values[0]
    if time_s >= series.times_s[-1]:
        return series.values[-1]
    index = _bisect_right(series.times_s, time_s) - 1
    return series.values[index]


def _evaluate_running_request_count(series_list: list[SampleSeries], time_s: float) -> float:
    return sum(_evaluate_piecewise_constant(series, time_s) for series in series_list)


def _dedupe_points(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for time_s, value in sorted(points, key=lambda pair: pair[0]):
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


def _clip_series_points(
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

    def value_at(time_s: float) -> float:
        if time_s <= times[0]:
            return values[0]
        if time_s >= times[-1]:
            return values[-1]
        right_index = _bisect_right(times, time_s)
        left_index = right_index - 1
        if math.isclose(times[left_index], time_s, rel_tol=0.0, abs_tol=1e-12):
            return values[left_index]
        return _interpolate_value(
            times[left_index],
            values[left_index],
            times[right_index],
            values[right_index],
            time_s,
        )

    clipped: list[tuple[float, float]] = [(start_s, value_at(start_s))]
    clipped.extend(
        (time_s, value)
        for time_s, value in deduped_points
        if start_s < time_s < end_s
    )
    clipped.append((end_s, value_at(end_s)))
    return _dedupe_points(clipped)


def _extract_request_records(llm_requests_payload: dict[str, Any]) -> list[dict[str, Any]]:
    requests = llm_requests_payload.get("requests")
    if not isinstance(requests, list):
        raise ValueError("llm-requests.json is missing requests")

    extracted: list[dict[str, Any]] = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        trace_id = request.get("trace_id")
        request_start_offset_s = _float_or_none(request.get("request_start_offset_s"))
        request_end_offset_s = _float_or_none(request.get("request_end_offset_s"))
        if not isinstance(trace_id, str) or not trace_id:
            continue
        if request_start_offset_s is None or request_end_offset_s is None:
            continue
        if request_end_offset_s < request_start_offset_s:
            continue
        extracted.append(
            {
                "trace_id": trace_id,
                "request_id": request.get("request_id"),
                "request_start_offset_s": request_start_offset_s,
                "request_end_offset_s": request_end_offset_s,
            }
        )

    return extracted


def _extract_trial_trace_mapping(trial_timing_payload: dict[str, Any]) -> dict[str, str]:
    agent_time_breakdown = trial_timing_payload.get("agent_time_breakdown_s")
    if not isinstance(agent_time_breakdown, dict):
        agent_time_breakdown = trial_timing_payload.get("agent_time_breakdown")
    if not isinstance(agent_time_breakdown, dict):
        raise ValueError("trial-timing-summary.json is missing agent_time_breakdown")

    agents = agent_time_breakdown.get("agents")
    if not isinstance(agents, list):
        raise ValueError("trial-timing-summary.json is missing agent_time_breakdown.agents")

    mapping: dict[str, str] = {}
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        trial_id = agent.get("trial_id")
        trace_id = agent.get("trace_id")
        if isinstance(trial_id, str) and trial_id and isinstance(trace_id, str) and trace_id:
            mapping[trial_id] = trace_id
    return mapping


def _extract_selected_finished_agents(
    replay_summary_payload: dict[str, Any],
    *,
    trial_id_to_trace_id: dict[str, str],
    start_s: float,
    end_s: float,
    available_end_s: float,
) -> tuple[int, dict[str, float]]:
    experiment_start_utc = _parse_iso8601_to_utc(replay_summary_payload.get("started_at"))
    if experiment_start_utc is None:
        raise ValueError("replay/summary.json is missing started_at")

    worker_results = replay_summary_payload.get("worker_results")
    if not isinstance(worker_results, dict):
        raise ValueError("replay/summary.json is missing worker_results")

    selected_finished_count = 0
    trace_finish_offsets_s: dict[str, float] = {}
    for trial_id, worker_payload in worker_results.items():
        if not isinstance(worker_payload, dict):
            continue
        finish_utc = _parse_iso8601_to_utc(worker_payload.get("finished_at"))
        if finish_utc is None:
            continue
        finish_offset_s = _duration_s(experiment_start_utc, finish_utc)
        if finish_offset_s is None:
            continue
        if not (start_s <= finish_offset_s <= end_s):
            continue
        if finish_offset_s > available_end_s:
            continue
        selected_finished_count += 1
        trace_id = trial_id_to_trace_id.get(str(trial_id))
        if trace_id is not None:
            trace_finish_offsets_s[trace_id] = finish_offset_s

    return (selected_finished_count, trace_finish_offsets_s)


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


def _build_row(
    metadata: RunMetadata,
    *,
    start_s: float,
    requested_end_s: float | None,
) -> dict[str, Any]:
    power_summary_payload = _load_json(
        metadata.run_dir / "post-processed" / "power" / "power-summary.json"
    )
    vllm_timeseries_payload = _load_json(
        metadata.run_dir / "post-processed" / "vllm-log" / "gauge-counter-timeseries.json"
    )
    llm_requests_payload = _load_json(
        metadata.run_dir / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json"
    )
    trial_timing_payload = _load_json(
        metadata.run_dir / "post-processed" / "global" / "trial-timing-summary.json"
    )
    replay_summary_payload = _load_json(metadata.run_dir / "replay" / "summary.json")

    if not isinstance(power_summary_payload, dict):
        raise ValueError("power-summary.json must be a JSON object")
    if not isinstance(vllm_timeseries_payload, dict):
        raise ValueError("gauge-counter-timeseries.json must be a JSON object")
    if not isinstance(llm_requests_payload, dict):
        raise ValueError("llm-requests.json must be a JSON object")
    if not isinstance(trial_timing_payload, dict):
        raise ValueError("trial-timing-summary.json must be a JSON object")
    if not isinstance(replay_summary_payload, dict):
        raise ValueError("replay/summary.json must be a JSON object")

    power_points = _extract_power_points(power_summary_payload)
    if len(power_points) < 2:
        raise ValueError(f"Not enough power samples for {metadata.run_path}")

    running_series_list = _extract_num_requests_running_series(vllm_timeseries_payload)
    power_available_end_s = _extract_power_window_duration_s(power_summary_payload)
    running_available_end_s = _running_series_available_end_s(running_series_list)
    available_end_s = round(min(power_available_end_s, running_available_end_s), 6)

    if start_s < 0.0:
        raise ValueError(f"--start-s must be non-negative, got {start_s}")
    if start_s >= available_end_s:
        raise ValueError(
            f"Window start {start_s} is outside the available range for {metadata.run_path} "
            f"(available end: {available_end_s})"
        )

    end_s = available_end_s if requested_end_s is None else requested_end_s
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
    power_points_in_window = _clip_series_points(power_points, start_s=start_s, end_s=end_s)
    power_sample_count_in_window = len(
        [1 for time_s, _ in power_points if start_s <= time_s <= end_s]
    )

    running_sample_count_in_window = sum(
        sum(1 for time_s in series.times_s if start_s <= time_s <= end_s)
        for series in running_series_list
    )

    per_request_power_points_full = [
        (
            time_s,
            (
                0.0
                if _evaluate_running_request_count(running_series_list, time_s) <= 0.0
                else power_w / _evaluate_running_request_count(running_series_list, time_s)
            ),
        )
        for time_s, power_w in power_points
        if time_s <= available_end_s
    ]
    per_request_power_points_window = _clip_series_points(
        per_request_power_points_full,
        start_s=start_s,
        end_s=end_s,
    )
    if len(per_request_power_points_window) < 2:
        raise ValueError(f"Not enough per-request-power samples for {metadata.run_path}")

    per_request_power_integral = PiecewiseLinearIntegral(per_request_power_points_window)
    total_power_integral = PiecewiseLinearIntegral(power_points_in_window)

    request_records = _extract_request_records(llm_requests_payload)
    requests_by_trace_id: dict[str, list[dict[str, Any]]] = {}
    for request in request_records:
        requests_by_trace_id.setdefault(str(request["trace_id"]), []).append(request)

    trial_id_to_trace_id = _extract_trial_trace_mapping(trial_timing_payload)
    selected_finished_count, trace_finish_offsets_s = _extract_selected_finished_agents(
        replay_summary_payload,
        trial_id_to_trace_id=trial_id_to_trace_id,
        start_s=start_s,
        end_s=end_s,
        available_end_s=available_end_s,
    )

    mapped_finished_count = len(trace_finish_offsets_s)
    unmapped_finished_count = selected_finished_count - mapped_finished_count

    total_request_count_for_mapped_finished_replays = 0
    overlapping_request_count_for_mapped_finished_replays = 0
    traces_with_request_energy_in_window = 0
    agent_energy_by_trace_id: dict[str, float] = {}

    for trace_id in sorted(trace_finish_offsets_s):
        trace_requests = requests_by_trace_id.get(trace_id, [])
        total_request_count_for_mapped_finished_replays += len(trace_requests)
        trace_energy_j = 0.0
        overlapping_request_seen = False
        for request in trace_requests:
            request_start_offset_s = float(request["request_start_offset_s"])
            request_end_offset_s = float(request["request_end_offset_s"])
            overlap_start_s = max(request_start_offset_s, start_s)
            overlap_end_s = min(request_end_offset_s, end_s)
            if overlap_end_s <= overlap_start_s:
                continue
            overlapping_request_count_for_mapped_finished_replays += 1
            overlapping_request_seen = True
            trace_energy_j += per_request_power_integral.integral_between(
                overlap_start_s,
                overlap_end_s,
            )

        if overlapping_request_seen and trace_energy_j > 0.0:
            traces_with_request_energy_in_window += 1
        agent_energy_by_trace_id[trace_id] = round(trace_energy_j, 6)

    total_request_integral_energy_j = round(sum(agent_energy_by_trace_id.values()), 6)
    average_request_integral_energy_j: float | None = None
    if mapped_finished_count > 0:
        average_request_integral_energy_j = round(
            total_request_integral_energy_j / mapped_finished_count,
            6,
        )

    average_throughput_jobs_per_s = round(selected_finished_count / window_duration_s, 6)
    window_total_power_energy_integral_j = total_power_integral.integral_between(start_s, end_s)

    average_window_total_power_energy_per_finished_replay_j: float | None = None
    if selected_finished_count > 0:
        average_window_total_power_energy_per_finished_replay_j = round(
            window_total_power_energy_integral_j / selected_finished_count,
            6,
        )

    return {
        "run_path": metadata.run_path,
        "source_run_dir": str(metadata.run_dir),
        "frequency_mhz": metadata.frequency_mhz,
        "core_min_mhz": metadata.core_min_mhz,
        "core_max_mhz": metadata.core_max_mhz,
        "mem_freq_mhz": metadata.mem_freq_mhz,
        "analysis_window_available_end_s": available_end_s,
        "analysis_window_start_s": round(start_s, 6),
        "analysis_window_end_s": round(end_s, 6),
        "analysis_window_duration_s": window_duration_s,
        "power_sample_count_in_window": power_sample_count_in_window,
        "num_requests_running_metric_series_count": len(running_series_list),
        "num_requests_running_sample_count_in_window": running_sample_count_in_window,
        "total_llm_request_count": len(request_records),
        "selected_finished_replay_count_in_window": selected_finished_count,
        "mapped_finished_replay_count_in_window": mapped_finished_count,
        "unmapped_selected_finished_replay_count_in_window": unmapped_finished_count,
        "request_count_for_mapped_finished_replays": total_request_count_for_mapped_finished_replays,
        "request_count_overlapping_window_for_mapped_finished_replays": (
            overlapping_request_count_for_mapped_finished_replays
        ),
        "traces_with_request_energy_in_window": traces_with_request_energy_in_window,
        "total_request_integral_energy_across_finished_replays_j": total_request_integral_energy_j,
        "average_request_integral_energy_per_finished_replay_j": average_request_integral_energy_j,
        "average_throughput_jobs_per_s": average_throughput_jobs_per_s,
        "window_total_power_energy_integral_j": window_total_power_energy_integral_j,
        "average_window_total_power_energy_per_finished_replay_j": (
            average_window_total_power_energy_per_finished_replay_j
        ),
    }


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
