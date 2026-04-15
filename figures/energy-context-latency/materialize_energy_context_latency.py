#!/usr/bin/env python3
"""Materialize clustered-bar comparison data for power, energy, context, and latency."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "energy-context-latency"
DEFAULT_CONTEXT_MAX_TOKENS = 527_664.0
DEFAULT_UNCONTROLLED_ROOT = Path(
    "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean"
)
DEFAULT_FIXED_FREQ_ROOT = Path(
    "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-fixed-freq"
)
DEFAULT_STEER_ROOT = Path(
    "/srv/scratch/yichaoy2/work/vllm-otel/"
    "results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance"
)
DEFAULT_STEER_CTX_AWARE_ROOT = Path(
    "/srv/scratch/yichaoy2/work/vllm-otel/"
    "results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware"
)
POWER_SUMMARY_REL_PATH = Path("post-processed/power/power-summary.json")
STACK_CONTEXT_REL_PATH = Path(
    "post-processed/gateway/stack-context/context-usage-stacked-histogram.json"
)
AGENT_OUTPUT_THROUGHPUT_REL_PATH = Path(
    "post-processed/agent-output-throughput/agent-output-throughput.json"
)
JOB_THROUGHPUT_REL_PATH = Path(
    "post-processed/job-throughput/job-throughput-timeseries.json"
)
REPLAY_SUMMARY_REL_PATH = Path("replay/summary.json")
RUN_DIR_NAME_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")
FIXED_FREQ_NESTED_RUN_DIR_NAME = "core-345-810"


@dataclass(frozen=True)
class ImplementationSpec:
    implementation_key: str
    implementation_label: str
    source_root: Path


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset_slug: str
    dataset_label: str
    agent_slug: str
    agent_label: str
    qps_slugs: tuple[str, ...]


@dataclass(frozen=True)
class MetricSpec:
    metric_key: str
    metric_label: str
    panel_title: str
    metric_unit: str
    y_axis_label: str
    formula: str


METRIC_SPECS = (
    MetricSpec(
        metric_key="average_energy_per_finished_agent_kj",
        metric_label="Average Energy per Finished Agent",
        panel_title="Average Energy per Finished Agent",
        metric_unit="kJ",
        y_axis_label="Energy per Finished Agent (kJ)",
        formula=(
            "(power_stats_w.avg * analysis_window_duration_s) / replay_summary.workers_completed / 1000"
        ),
    ),
    MetricSpec(
        metric_key="average_power_w",
        metric_label="Average Power",
        panel_title="Average Power",
        metric_unit="W",
        y_axis_label="Average Power (W)",
        formula="power_stats_w.avg; fallback: mean(power_points[].power_w)",
    ),
    MetricSpec(
        metric_key="average_context_usage_pct",
        metric_label="Average Context Usage",
        panel_title="Average Context Usage",
        metric_unit="%",
        y_axis_label="Average Context Usage (% of 527,664 tokens)",
        formula="mean(points[].accumulated_value) / 527664 * 100",
    ),
    MetricSpec(
        metric_key="p5_output_throughput_tokens_per_s",
        metric_label="P5 Output Throughput",
        panel_title="5th Percentile Output Throughput",
        metric_unit="tokens/s",
        y_axis_label="P5 Output Throughput (tokens/s)",
        formula=(
            "agent_output_throughput_tokens_per_s_summary.percentiles['5']; "
            "fallback: interpolated 5th percentile of agents[].output_throughput_tokens_per_s"
        ),
    ),
    MetricSpec(
        metric_key="average_job_throughput_jobs_per_s",
        metric_label="Average Job Throughput",
        panel_title="Average Job Throughput",
        metric_unit="jobs/s",
        y_axis_label="Average Job Throughput (jobs/s)",
        formula="mean(throughput_points[].throughput_jobs_per_s)",
    ),
)


def _default_implementation_specs() -> tuple[ImplementationSpec, ...]:
    return (
        ImplementationSpec(
            implementation_key="uncontrolled",
            implementation_label="Uncontrolled",
            source_root=DEFAULT_UNCONTROLLED_ROOT,
        ),
        ImplementationSpec(
            implementation_key="fixed_freq",
            implementation_label="Fixed Freq",
            source_root=DEFAULT_FIXED_FREQ_ROOT,
        ),
        ImplementationSpec(
            implementation_key="steer",
            implementation_label="STEER",
            source_root=DEFAULT_STEER_ROOT,
        ),
    )


EXPERIMENT_SPECS = (
    ExperimentSpec(
        experiment_id="A",
        dataset_slug="swebench-verified",
        dataset_label="swebench-verified",
        agent_slug="mini-swe-agent",
        agent_label="mini-swe-agent",
        qps_slugs=("qps0_04", "qps0_06", "qps0_08"),
    ),
    ExperimentSpec(
        experiment_id="B",
        dataset_slug="dabstep",
        dataset_label="dabstep",
        agent_slug="mini-swe-agent",
        agent_label="mini-swe-agent",
        qps_slugs=("qps0_03", "qps0_04", "qps0_05"),
    ),
    ExperimentSpec(
        experiment_id="C",
        dataset_slug="terminal-bench-2.0",
        dataset_label="terminal-bench-2.0",
        agent_slug="terminus-2",
        agent_label="terminus-2",
        qps_slugs=("qps0_015", "qps0_02", "qps0_025"),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the energy-context-latency figure dataset from the three "
            "requested implementation roots."
        )
    )
    parser.add_argument(
        "--uncontrolled-root",
        default=str(DEFAULT_UNCONTROLLED_ROOT),
        help=f"Uncontrolled result root (default: {DEFAULT_UNCONTROLLED_ROOT}).",
    )
    parser.add_argument(
        "--fixed-freq-root",
        default=str(DEFAULT_FIXED_FREQ_ROOT),
        help=f"Fixed-frequency result root (default: {DEFAULT_FIXED_FREQ_ROOT}).",
    )
    parser.add_argument(
        "--steer-root",
        default=str(DEFAULT_STEER_ROOT),
        help=f"STEER result root (default: {DEFAULT_STEER_ROOT}).",
    )
    parser.add_argument(
        "--steer-ctx-aware-root",
        default=str(DEFAULT_STEER_CTX_AWARE_ROOT),
        help=(
            "STEER result root override for the ctx-aware B/qps0_05 run "
            f"(default: {DEFAULT_STEER_CTX_AWARE_ROOT})."
        ),
    )
    parser.add_argument(
        "--context-max-tokens",
        type=float,
        default=DEFAULT_CONTEXT_MAX_TOKENS,
        help=(
            "Context normalization denominator. The plotted value is "
            "mean_context_usage / context_max_tokens * 100."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path. Default: "
            "figures/energy-context-latency/data/energy-context-latency.json"
        ),
    )
    parser.add_argument(
        "--missing-log",
        default=None,
        help=(
            "Optional missing-data log path. Default: "
            "figures/energy-context-latency/data/energy-context-latency.missing.log"
        ),
    )
    return parser.parse_args()


def _default_output_path() -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{DEFAULT_OUTPUT_STEM}.json").resolve()


def _default_missing_log_path(output_path: Path) -> Path:
    return output_path.with_suffix(".missing.log")


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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _parse_iso8601_to_utc(raw_value: Any) -> datetime | None:
    if not isinstance(raw_value, str) or not raw_value:
        return None
    normalized = raw_value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _parse_iso8601_to_epoch_seconds(raw_value: Any) -> float | None:
    parsed = _parse_iso8601_to_utc(raw_value)
    if parsed is None:
        return None
    return parsed.timestamp()


def _qps_dir_for(root_dir: Path, experiment: ExperimentSpec, qps_slug: str) -> Path:
    return (
        root_dir
        / experiment.dataset_slug
        / experiment.agent_slug
        / "split"
        / "exclude-unranked"
        / qps_slug
    )


def _qps_slug_to_value(qps_slug: str) -> float:
    if not qps_slug.startswith("qps"):
        raise ValueError(f"Expected qps slug starting with 'qps': {qps_slug}")
    return float(qps_slug[3:].replace("_", "."))


def _source_root_for(
    *,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    steer_ctx_aware_root: Path,
) -> Path:
    if (
        implementation.implementation_key == "steer"
        and experiment.experiment_id == "B"
        and qps_slug == "qps0_05"
    ):
        return steer_ctx_aware_root
    return implementation.source_root


def _record_missing(
    missing_entries: list[dict[str, Any]],
    *,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    scope: str,
    reason: str,
    path: Path | None = None,
    metric_key: str | None = None,
    detail: str | None = None,
) -> None:
    entry: dict[str, Any] = {
        "implementation_key": implementation.implementation_key,
        "implementation_label": implementation.implementation_label,
        "experiment_id": experiment.experiment_id,
        "dataset_slug": experiment.dataset_slug,
        "agent_slug": experiment.agent_slug,
        "qps_slug": qps_slug,
        "scope": scope,
        "reason": reason,
    }
    if path is not None:
        entry["path"] = str(path)
    if metric_key is not None:
        entry["metric_key"] = metric_key
    if detail is not None:
        entry["detail"] = detail
    missing_entries.append(entry)


def _record_missing_for_metric_keys(
    missing_entries: list[dict[str, Any]],
    *,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    scope: str,
    reason: str,
    path: Path | None = None,
    metric_keys: tuple[str, ...] | None = None,
    detail: str | None = None,
) -> None:
    if not metric_keys:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope=scope,
            reason=reason,
            path=path,
            detail=detail,
        )
        return
    for metric_key in metric_keys:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope=scope,
            reason=reason,
            path=path,
            metric_key=metric_key,
            detail=detail,
        )


def _discover_latest_run_dir(
    *,
    implementation: ImplementationSpec,
    source_root: Path,
    experiment: ExperimentSpec,
    qps_slug: str,
    missing_entries: list[dict[str, Any]],
) -> tuple[Path | None, int]:
    qps_dir = _qps_dir_for(source_root, experiment, qps_slug)
    if not qps_dir.is_dir():
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="qps-dir",
            reason="QPS directory does not exist",
            path=qps_dir,
        )
        return None, 0

    run_dirs = sorted(
        (
            child.resolve()
            for child in qps_dir.iterdir()
            if child.is_dir() and RUN_DIR_NAME_PATTERN.match(child.name)
        ),
        key=lambda path: (path.name, str(path)),
    )
    if not run_dirs:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="run-dir",
            reason="QPS directory has no timestamp run directories",
            path=qps_dir,
        )
        return None, 0

    if implementation.implementation_key != "fixed_freq":
        return run_dirs[-1], len(run_dirs)

    for selected_timestamp_dir in reversed(run_dirs):
        nested_run_dir = (selected_timestamp_dir / FIXED_FREQ_NESTED_RUN_DIR_NAME).resolve()
        if nested_run_dir.is_dir():
            return nested_run_dir, len(run_dirs)

    if run_dirs:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="run-dir",
            reason=(
                "QPS directory has no timestamp run directory containing the "
                "nested fixed-freq run directory"
            ),
            path=qps_dir,
        )
        return None, len(run_dirs)
    return None, len(run_dirs)


def _load_optional_json(
    *,
    run_dir: Path | None,
    rel_path: Path,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    missing_entries: list[dict[str, Any]],
    metric_keys: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    path = run_dir / rel_path
    if not path.is_file():
        _record_missing_for_metric_keys(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="file",
            reason="Required file does not exist",
            path=path,
            metric_keys=metric_keys,
        )
        return None
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        _record_missing_for_metric_keys(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="file",
            reason="Unable to load JSON payload",
            path=path,
            metric_keys=metric_keys,
            detail=str(exc),
        )
        return None
    if not isinstance(payload, dict):
        _record_missing_for_metric_keys(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="file",
            reason="JSON payload is not an object",
            path=path,
            metric_keys=metric_keys,
        )
        return None
    return payload


def _extract_workers_completed_count(
    *,
    replay_summary_payload: dict[str, Any] | None,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    missing_entries: list[dict[str, Any]],
) -> int:
    if replay_summary_payload is None:
        return 0
    workers_completed = _int_or_none(replay_summary_payload.get("workers_completed"))
    if workers_completed is None:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Replay summary is missing integer workers_completed",
            metric_key="average_energy_per_finished_agent_kj",
        )
        return 0
    if workers_completed <= 0:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Replay summary reported no completed workers",
            metric_key="average_energy_per_finished_agent_kj",
        )
        return 0
    return workers_completed


def _extract_power_duration_s(power_payload: dict[str, Any] | None) -> float | None:
    if power_payload is None:
        return None
    start_s = _parse_iso8601_to_epoch_seconds(power_payload.get("analysis_window_start_utc"))
    end_s = _parse_iso8601_to_epoch_seconds(power_payload.get("analysis_window_end_utc"))
    if start_s is not None and end_s is not None and end_s >= start_s:
        return round(end_s - start_s, 6)

    raw_points = power_payload.get("power_points")
    if not isinstance(raw_points, list):
        return None
    numeric_offsets = [
        offset
        for raw_point in raw_points
        if isinstance(raw_point, dict)
        for offset in [_float_or_none(raw_point.get("time_offset_s"))]
        if offset is not None
    ]
    if len(numeric_offsets) < 2:
        return None
    return round(max(numeric_offsets) - min(numeric_offsets), 6)


def _extract_average_power_w(power_payload: dict[str, Any] | None) -> float | None:
    if power_payload is None:
        return None
    power_stats = power_payload.get("power_stats_w")
    if isinstance(power_stats, dict):
        avg_value = _float_or_none(power_stats.get("avg"))
        if avg_value is not None:
            return avg_value

    raw_points = power_payload.get("power_points")
    if not isinstance(raw_points, list):
        return None
    power_values = [
        power_w
        for raw_point in raw_points
        if isinstance(raw_point, dict)
        for power_w in [_float_or_none(raw_point.get("power_w"))]
        if power_w is not None
    ]
    if not power_values:
        return None
    return round(sum(power_values) / len(power_values), 6)


def _average_power_source(power_payload: dict[str, Any] | None) -> str:
    if power_payload is None:
        return "missing"
    power_stats = power_payload.get("power_stats_w")
    if isinstance(power_stats, dict):
        avg_value = _float_or_none(power_stats.get("avg"))
        if avg_value is not None:
            return "power_stats_w.avg"

    raw_points = power_payload.get("power_points")
    if not isinstance(raw_points, list):
        return "missing"
    power_values = [
        power_w
        for raw_point in raw_points
        if isinstance(raw_point, dict)
        for power_w in [_float_or_none(raw_point.get("power_w"))]
        if power_w is not None
    ]
    if not power_values:
        return "missing"
    return "power_points[].power_w"


def _build_energy_metric(
    *,
    power_payload: dict[str, Any] | None,
    replay_summary_payload: dict[str, Any] | None,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    missing_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    avg_power_w = _extract_average_power_w(power_payload)
    if power_payload is not None and avg_power_w is None:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Average power is unavailable",
            metric_key="average_energy_per_finished_agent_kj",
        )
    duration_s = _extract_power_duration_s(power_payload)
    if power_payload is not None and duration_s is None:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Analysis window duration is unavailable",
            metric_key="average_energy_per_finished_agent_kj",
        )
    workers_completed = _extract_workers_completed_count(
        replay_summary_payload=replay_summary_payload,
        implementation=implementation,
        experiment=experiment,
        qps_slug=qps_slug,
        missing_entries=missing_entries,
    )
    total_energy_estimate_j = (
        round(avg_power_w * duration_s, 6)
        if avg_power_w is not None and duration_s is not None
        else 0.0
    )
    per_agent_energy_j = (
        round(total_energy_estimate_j / workers_completed, 6)
        if workers_completed > 0 and total_energy_estimate_j > 0.0
        else 0.0
    )
    return {
        "value": round(per_agent_energy_j / 1000.0, 6),
        "value_raw_j": per_agent_energy_j,
        "total_energy_estimate_j": total_energy_estimate_j,
        "average_power_w": 0.0 if avg_power_w is None else round(avg_power_w, 6),
        "duration_s": 0.0 if duration_s is None else round(duration_s, 6),
        "workers_completed": workers_completed,
    }


def _build_average_power_metric(
    *,
    power_payload: dict[str, Any] | None,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    missing_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    avg_power_w = _extract_average_power_w(power_payload)
    if power_payload is not None and avg_power_w is None:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Average power is unavailable",
            metric_key="average_power_w",
        )
    return {
        "value": 0.0 if avg_power_w is None else round(avg_power_w, 6),
        "source": _average_power_source(power_payload),
    }


def _build_context_metric(
    *,
    context_payload: dict[str, Any] | None,
    context_max_tokens: float,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    missing_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if context_payload is None:
        return {
            "value": 0.0,
            "average_context_tokens": 0.0,
            "point_count": 0,
            "context_max_tokens": context_max_tokens,
        }
    raw_points = context_payload.get("points")
    if not isinstance(raw_points, list):
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Context payload is missing points[]",
            metric_key="average_context_usage_pct",
        )
        return {
            "value": 0.0,
            "average_context_tokens": 0.0,
            "point_count": 0,
            "context_max_tokens": context_max_tokens,
        }

    values = [
        accumulated_value
        for raw_point in raw_points
        if isinstance(raw_point, dict)
        for accumulated_value in [_float_or_none(raw_point.get("accumulated_value"))]
        if accumulated_value is not None
    ]
    if not values:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Context payload has no numeric accumulated_value points",
            metric_key="average_context_usage_pct",
        )
        return {
            "value": 0.0,
            "average_context_tokens": 0.0,
            "point_count": 0,
            "context_max_tokens": context_max_tokens,
        }

    average_context_tokens = sum(values) / len(values)
    if context_max_tokens <= 0.0:
        raise ValueError(
            f"--context-max-tokens must be positive; got {context_max_tokens}"
        )
    return {
        "value": round((average_context_tokens / context_max_tokens) * 100.0, 6),
        "average_context_tokens": round(average_context_tokens, 6),
        "point_count": len(values),
        "context_max_tokens": round(context_max_tokens, 6),
    }


def _percentile_from_sorted(sorted_values: list[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if quantile <= 0.0:
        return sorted_values[0]
    if quantile >= 1.0:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    fraction = position - lower_index
    return lower_value + ((upper_value - lower_value) * fraction)


def _build_throughput_metric(
    *,
    throughput_payload: dict[str, Any] | None,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    missing_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if throughput_payload is None:
        return {
            "value": 0.0,
            "sample_count": 0,
            "source": "missing",
        }

    summary = throughput_payload.get("agent_output_throughput_tokens_per_s_summary")
    if isinstance(summary, dict):
        raw_percentiles = summary.get("percentiles")
        percentile_value = (
            _float_or_none(raw_percentiles.get("5"))
            if isinstance(raw_percentiles, dict)
            else None
        )
        if percentile_value is not None:
            return {
                "value": round(percentile_value, 6),
                "sample_count": int(summary.get("sample_count", 0))
                if isinstance(summary.get("sample_count"), int)
                else 0,
                "source": "summary.percentiles.5",
            }

    raw_agents = throughput_payload.get("agents")
    if not isinstance(raw_agents, list):
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Throughput payload is missing percentiles and agents[]",
            metric_key="p5_output_throughput_tokens_per_s",
        )
        return {
            "value": 0.0,
            "sample_count": 0,
            "source": "missing",
        }

    values = [
        throughput
        for raw_agent in raw_agents
        if isinstance(raw_agent, dict)
        for throughput in [_float_or_none(raw_agent.get("output_throughput_tokens_per_s"))]
        if throughput is not None
    ]
    if not values:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Throughput payload has no numeric output_throughput_tokens_per_s values",
            metric_key="p5_output_throughput_tokens_per_s",
        )
        return {
            "value": 0.0,
            "sample_count": 0,
            "source": "missing",
        }

    percentile_value = _percentile_from_sorted(sorted(values), 0.05)
    return {
        "value": 0.0 if percentile_value is None else round(percentile_value, 6),
        "sample_count": len(values),
        "source": "agents.output_throughput_tokens_per_s",
    }


def _build_job_throughput_metric(
    *,
    job_throughput_payload: dict[str, Any] | None,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    missing_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if job_throughput_payload is None:
        return {
            "value": 0.0,
            "sample_count": 0,
            "source": "missing",
        }

    raw_points = job_throughput_payload.get("throughput_points")
    if not isinstance(raw_points, list):
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Job-throughput payload is missing throughput_points[]",
            metric_key="average_job_throughput_jobs_per_s",
        )
        return {
            "value": 0.0,
            "sample_count": 0,
            "source": "missing",
        }

    values = [
        throughput_value
        for raw_point in raw_points
        if isinstance(raw_point, dict)
        for throughput_value in [_float_or_none(raw_point.get("throughput_jobs_per_s"))]
        if throughput_value is not None
    ]
    if not values:
        _record_missing(
            missing_entries,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            scope="metric",
            reason="Job-throughput payload has no numeric throughput_jobs_per_s values",
            metric_key="average_job_throughput_jobs_per_s",
        )
        return {
            "value": 0.0,
            "sample_count": 0,
            "source": "missing",
        }

    return {
        "value": round(sum(values) / len(values), 6),
        "sample_count": len(values),
        "source": "throughput_points[].throughput_jobs_per_s",
    }


def _build_implementation_entry(
    *,
    implementation: ImplementationSpec,
    experiment: ExperimentSpec,
    qps_slug: str,
    context_max_tokens: float,
    steer_ctx_aware_root: Path,
    missing_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    source_root = _source_root_for(
        implementation=implementation,
        experiment=experiment,
        qps_slug=qps_slug,
        steer_ctx_aware_root=steer_ctx_aware_root,
    )
    run_dir, candidate_run_count = _discover_latest_run_dir(
        implementation=implementation,
        source_root=source_root,
        experiment=experiment,
        qps_slug=qps_slug,
        missing_entries=missing_entries,
    )
    throughput_payload = _load_optional_json(
        run_dir=run_dir,
        rel_path=AGENT_OUTPUT_THROUGHPUT_REL_PATH,
        implementation=implementation,
        experiment=experiment,
        qps_slug=qps_slug,
        missing_entries=missing_entries,
        metric_keys=("p5_output_throughput_tokens_per_s",),
    )
    power_payload = _load_optional_json(
        run_dir=run_dir,
        rel_path=POWER_SUMMARY_REL_PATH,
        implementation=implementation,
        experiment=experiment,
        qps_slug=qps_slug,
        missing_entries=missing_entries,
        metric_keys=(
            "average_energy_per_finished_agent_kj",
            "average_power_w",
        ),
    )
    context_payload = _load_optional_json(
        run_dir=run_dir,
        rel_path=STACK_CONTEXT_REL_PATH,
        implementation=implementation,
        experiment=experiment,
        qps_slug=qps_slug,
        missing_entries=missing_entries,
        metric_keys=("average_context_usage_pct",),
    )
    job_throughput_payload = _load_optional_json(
        run_dir=run_dir,
        rel_path=JOB_THROUGHPUT_REL_PATH,
        implementation=implementation,
        experiment=experiment,
        qps_slug=qps_slug,
        missing_entries=missing_entries,
        metric_keys=("average_job_throughput_jobs_per_s",),
    )
    replay_summary_payload = _load_optional_json(
        run_dir=run_dir,
        rel_path=REPLAY_SUMMARY_REL_PATH,
        implementation=implementation,
        experiment=experiment,
        qps_slug=qps_slug,
        missing_entries=missing_entries,
        metric_keys=("average_energy_per_finished_agent_kj",),
    )

    metrics = {
        "average_energy_per_finished_agent_kj": _build_energy_metric(
            power_payload=power_payload,
            replay_summary_payload=replay_summary_payload,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            missing_entries=missing_entries,
        ),
        "average_power_w": _build_average_power_metric(
            power_payload=power_payload,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            missing_entries=missing_entries,
        ),
        "average_context_usage_pct": _build_context_metric(
            context_payload=context_payload,
            context_max_tokens=context_max_tokens,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            missing_entries=missing_entries,
        ),
        "p5_output_throughput_tokens_per_s": _build_throughput_metric(
            throughput_payload=throughput_payload,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            missing_entries=missing_entries,
        ),
        "average_job_throughput_jobs_per_s": _build_job_throughput_metric(
            job_throughput_payload=job_throughput_payload,
            implementation=implementation,
            experiment=experiment,
            qps_slug=qps_slug,
            missing_entries=missing_entries,
        ),
    }

    return {
        "implementation_key": implementation.implementation_key,
        "implementation_label": implementation.implementation_label,
        "source_root": str(source_root),
        "run_dir": None if run_dir is None else str(run_dir),
        "run_dir_name": None if run_dir is None else run_dir.name,
        "candidate_run_count": candidate_run_count,
        "metrics": metrics,
        "metric_values": {
            metric_key: metric_summary["value"]
            for metric_key, metric_summary in metrics.items()
        },
    }


def _build_payload(
    *,
    implementations: tuple[ImplementationSpec, ...],
    context_max_tokens: float,
    steer_ctx_aware_root: Path,
) -> dict[str, Any]:
    missing_entries: list[dict[str, Any]] = []
    experiments_payload: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_SPECS:
        qps_payload: list[dict[str, Any]] = []
        for qps_slug in experiment.qps_slugs:
            qps_payload.append(
                {
                    "qps_slug": qps_slug,
                    "qps_value": _qps_slug_to_value(qps_slug),
                    "qps_label": f"{_qps_slug_to_value(qps_slug):.3f}".rstrip("0").rstrip("."),
                    "implementations": [
                        _build_implementation_entry(
                            implementation=implementation,
                            experiment=experiment,
                            qps_slug=qps_slug,
                            context_max_tokens=context_max_tokens,
                            steer_ctx_aware_root=steer_ctx_aware_root,
                            missing_entries=missing_entries,
                        )
                        for implementation in implementations
                    ],
                }
            )
        experiments_payload.append(
            {
                "experiment_id": experiment.experiment_id,
                "dataset_slug": experiment.dataset_slug,
                "dataset_label": experiment.dataset_label,
                "agent_slug": experiment.agent_slug,
                "agent_label": experiment.agent_label,
                "subplot_title": (
                    f"{experiment.experiment_id}. "
                    f"{experiment.dataset_label} + {experiment.agent_label}"
                ),
                "qps": qps_payload,
            }
        )

    return {
        "figure_name": DEFAULT_OUTPUT_STEM,
        "context_max_tokens": context_max_tokens,
        "experiment_count": len(experiments_payload),
        "implementation_count": len(implementations),
        "metric_count": len(METRIC_SPECS),
        "implementations": [
            {
                "implementation_key": implementation.implementation_key,
                "implementation_label": implementation.implementation_label,
                "source_root": str(implementation.source_root),
            }
            for implementation in implementations
        ],
        "metrics": [
            {
                "metric_key": metric.metric_key,
                "metric_label": metric.metric_label,
                "panel_title": metric.panel_title,
                "metric_unit": metric.metric_unit,
                "y_axis_label": metric.y_axis_label,
                "formula": metric.formula,
            }
            for metric in METRIC_SPECS
        ],
        "experiments": experiments_payload,
        "missing_entries": missing_entries,
        "missing_entry_count": len(missing_entries),
        "notes": [
            (
                "Newest timestamp directory is selected for each "
                "(implementation, dataset, agent, qps) combination."
            ),
            "Missing directories, files, or fields are logged and plotted as 0.",
            (
                "The on-disk terminal benchmark path uses "
                "'terminal-bench-2.0/terminus-2'."
            ),
            (
                "For experiment B at qps0_05, the STEER run is read from the "
                "ctx-aware result root."
            ),
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_missing_log(path: Path, missing_entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not missing_entries:
        path.write_text("No missing data detected.\n", encoding="utf-8")
        return
    lines: list[str] = []
    for entry in missing_entries:
        parts = [
            f"implementation={entry['implementation_key']}",
            f"experiment={entry['experiment_id']}",
            f"dataset={entry['dataset_slug']}",
            f"agent={entry['agent_slug']}",
            f"qps={entry['qps_slug']}",
            f"scope={entry['scope']}",
            f"reason={entry['reason']}",
        ]
        metric_key = entry.get("metric_key")
        if isinstance(metric_key, str) and metric_key:
            parts.append(f"metric={metric_key}")
        path_value = entry.get("path")
        if isinstance(path_value, str) and path_value:
            parts.append(f"path={path_value}")
        detail = entry.get("detail")
        if isinstance(detail, str) and detail:
            parts.append(f"detail={detail}")
        lines.append(" | ".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    implementations = (
        ImplementationSpec(
            implementation_key="uncontrolled",
            implementation_label="Uncontrolled",
            source_root=Path(args.uncontrolled_root).expanduser().resolve(),
        ),
        ImplementationSpec(
            implementation_key="fixed_freq",
            implementation_label="Fixed Freq",
            source_root=Path(args.fixed_freq_root).expanduser().resolve(),
        ),
        ImplementationSpec(
            implementation_key="steer",
            implementation_label="STEER",
            source_root=Path(args.steer_root).expanduser().resolve(),
        ),
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path()
    )
    missing_log_path = (
        Path(args.missing_log).expanduser().resolve()
        if args.missing_log is not None
        else _default_missing_log_path(output_path)
    )

    payload = _build_payload(
        implementations=implementations,
        context_max_tokens=float(args.context_max_tokens),
        steer_ctx_aware_root=Path(args.steer_ctx_aware_root).expanduser().resolve(),
    )
    payload["output_path"] = str(output_path)
    payload["missing_log_path"] = str(missing_log_path)

    _write_json(output_path, payload)
    _write_missing_log(missing_log_path, payload["missing_entries"])
    print(f"[written] {output_path}")
    print(f"[written] {missing_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
