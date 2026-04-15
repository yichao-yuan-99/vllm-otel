#!/usr/bin/env python3
"""Materialize per-job agent-duration distributions for the turns-duration-violin figure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
POST_PROCESS_ROOT = REPO_ROOT / "post-process"
if str(POST_PROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(POST_PROCESS_ROOT))

from pp_common.service_failure import cutoff_datetime_utc_from_payload
from pp_common.service_failure import ensure_service_failure_payload
from pp_common.service_failure import parse_iso8601_to_utc


DEFAULT_ROOT_DIR = (REPO_ROOT / "results" / "qwen3-coder-30b").resolve()
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "turns-duration-violin"
DEFAULT_BENCHMARK_ORDER = ("dabstep", "swebench-verified", "terminal-bench-2.0")
DEFAULT_AGENT_ORDER = ("mini-swe-agent", "terminus-2")
SUMMARY_REL_PATH = Path("run-stats/run-stats-summary.json")
REQUESTS_REL_PATH = Path("requests/model_inference.jsonl")
LIFECYCLE_REL_PATH = Path("events/lifecycle.jsonl")
MANIFEST_REL_PATH = Path("manifest.json")
BENCHMARK_LABELS = {
    "dabstep": "DABStep",
    "swebench-verified": "SWE-bench Verified",
    "terminal-bench-2.0": "Terminal-Bench 2.0",
}
AGENT_LABELS = {
    "mini-swe-agent": "Mini-SWE-Agent",
    "terminus-2": "Terminus",
}
AGENT_ALIASES = {
    "mini-swe-agent": "mini-swe-agent",
    "minisweagent": "mini-swe-agent",
    "terminus": "terminus-2",
    "terminus-2": "terminus-2",
}


@dataclass(frozen=True)
class PanelSelection:
    benchmark: str
    benchmark_label: str
    agent_type: str
    agent_label: str
    run_dir: Path
    run_path: str
    summary_path: Path
    run_dir_name: str
    candidate_run_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize per-job agent-duration distributions for the "
            "turns-duration-violin figure from gateway lifecycle artifacts in the "
            "latest selected runs."
        )
    )
    parser.add_argument(
        "--root-dir",
        default=str(DEFAULT_ROOT_DIR),
        help=(
            "Root results directory that contains benchmark/agent/run directories. "
            f"Default: {DEFAULT_ROOT_DIR}"
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARK_ORDER),
        help=(
            "Benchmark directory names to compare, in plotting order. "
            f"Default: {' '.join(DEFAULT_BENCHMARK_ORDER)}"
        ),
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["mini-swe-agent", "terminus"],
        help=(
            "Agent directory names or aliases to compare within each benchmark, in "
            "plotting order. 'terminus' is treated as an alias for 'terminus-2'."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path. Default: "
            "figures/turns-duration-violin/data/"
            "turns-duration-violin.<root>.benchmarks-<...>.agents-<...>.json"
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl_dict_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        signed = stripped[1:] if stripped[0] in {"+", "-"} else stripped
        if signed.isdigit():
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        if math.isfinite(parsed):
            return parsed
    return None


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _slugify_filename_part(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "value"


def _default_output_path(
    *,
    root_dir: Path,
    benchmark_order: list[str],
    agent_order: list[str],
) -> Path:
    benchmark_label = "_".join(_slugify_filename_part(item) for item in benchmark_order)
    agent_label = "_".join(_slugify_filename_part(item) for item in agent_order)
    return (
        DEFAULT_OUTPUT_DIR
        / (
            f"{DEFAULT_OUTPUT_STEM}.{_slugify_filename_part(root_dir.name)}."
            f"benchmarks-{benchmark_label}.agents-{agent_label}.json"
        )
    ).resolve()


def _benchmark_label(benchmark: str) -> str:
    return BENCHMARK_LABELS.get(benchmark, benchmark)


def _normalize_agent(agent: str) -> str:
    normalized = agent.strip().lower()
    if not normalized:
        raise ValueError("Agent names must not be empty")
    return AGENT_ALIASES.get(normalized, normalized)


def _agent_label(agent_type: str) -> str:
    return AGENT_LABELS.get(agent_type, agent_type)


def _quantile_from_sorted_values(
    sorted_values: list[float],
    quantile: float,
) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a quantile from an empty sequence")
    if quantile <= 0.0:
        return float(sorted_values[0])
    if quantile >= 1.0:
        return float(sorted_values[-1])

    index = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(index)
    upper_index = math.ceil(index)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    upper_weight = index - lower_index
    lower_weight = 1.0 - upper_weight
    return (
        sorted_values[lower_index] * lower_weight
        + sorted_values[upper_index] * upper_weight
    )


def _summarize_durations(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("At least one duration value is required")
    sorted_values = sorted(values)
    q1 = _quantile_from_sorted_values(sorted_values, 0.25)
    median = _quantile_from_sorted_values(sorted_values, 0.50)
    q3 = _quantile_from_sorted_values(sorted_values, 0.75)
    mean = sum(sorted_values) / len(sorted_values)
    return {
        "sample_count": len(sorted_values),
        "min": min(sorted_values),
        "q1": q1,
        "median": median,
        "q3": q3,
        "p90": _quantile_from_sorted_values(sorted_values, 0.90),
        "p95": _quantile_from_sorted_values(sorted_values, 0.95),
        "mean": mean,
        "max": max(sorted_values),
        "iqr": q3 - q1,
    }


def _discover_selection(
    *,
    root_dir: Path,
    benchmark: str,
    agent_type: str,
) -> PanelSelection:
    benchmark_dir = (root_dir / benchmark).resolve()
    if not benchmark_dir.is_dir():
        raise ValueError(f"Benchmark directory does not exist: {benchmark_dir}")

    agent_dir = (benchmark_dir / agent_type).resolve()
    if not agent_dir.is_dir():
        raise ValueError(f"Agent directory does not exist: {agent_dir}")

    candidate_summaries: list[Path] = []
    for summary_path in agent_dir.rglob(SUMMARY_REL_PATH.name):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name != SUMMARY_REL_PATH.parent.name:
            continue
        candidate_summaries.append(summary_path.resolve())

    if not candidate_summaries:
        raise ValueError(
            f"No {SUMMARY_REL_PATH.as_posix()} files were found for "
            f"{benchmark!r} x {agent_type!r} under {agent_dir}"
        )

    candidate_summaries.sort(
        key=lambda path: (path.parent.parent.name, str(path.parent.parent)),
    )
    summary_path = candidate_summaries[-1]
    run_dir = summary_path.parent.parent
    run_path = run_dir.relative_to(root_dir.resolve()).as_posix()
    return PanelSelection(
        benchmark=benchmark,
        benchmark_label=_benchmark_label(benchmark),
        agent_type=agent_type,
        agent_label=_agent_label(agent_type),
        run_dir=run_dir,
        run_path=run_path,
        summary_path=summary_path,
        run_dir_name=run_dir.name,
        candidate_run_count=len(candidate_summaries),
    )


def _resolve_gateway_run_dir(
    run_dir: Path,
    *,
    gateway_run_id: str,
    gateway_profile_id: int | None,
) -> Path:
    gateway_output_dir = (run_dir / "gateway-output").resolve()
    if not gateway_output_dir.is_dir():
        raise ValueError(f"Missing gateway-output directory: {gateway_output_dir}")

    candidate_paths: list[Path] = []
    if gateway_profile_id is not None:
        candidate_paths.append(
            gateway_output_dir / f"profile-{gateway_profile_id}" / gateway_run_id
        )
    candidate_paths.append(gateway_output_dir / gateway_run_id)
    candidate_paths.extend(sorted(gateway_output_dir.glob(f"profile-*/{gateway_run_id}")))

    seen_candidates: set[Path] = set()
    for candidate in candidate_paths:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen_candidates:
            continue
        seen_candidates.add(resolved_candidate)
        if resolved_candidate.is_dir():
            return resolved_candidate

    raise ValueError(
        f"Could not locate gateway-output data for run {gateway_run_id!r} "
        f"under {gateway_output_dir}"
    )


def _duration_seconds(start_dt: Any, end_dt: Any) -> float | None:
    if start_dt is None or end_dt is None:
        return None
    duration_s = round((end_dt - start_dt).total_seconds(), 6)
    if duration_s < 0.0:
        return None
    return duration_s


def _clamp_window_to_cutoff(
    start_dt: Any,
    end_dt: Any,
    *,
    cutoff_time_utc: Any,
) -> tuple[Any, Any]:
    if cutoff_time_utc is None:
        return start_dt, end_dt
    if start_dt is not None and start_dt > cutoff_time_utc:
        return None, None
    if end_dt is None or end_dt > cutoff_time_utc:
        end_dt = cutoff_time_utc
    return start_dt, end_dt


def _load_lifecycle_windows(
    lifecycle_path: Path,
) -> tuple[Any, Any, Any, Any]:
    agent_start = None
    agent_end = None
    job_start = None
    job_end = None
    for record in _iter_jsonl_dict_records(lifecycle_path):
        event_type = record.get("event_type")
        timestamp = parse_iso8601_to_utc(record.get("timestamp"))
        if timestamp is None:
            continue
        if event_type == "agent_start" and agent_start is None:
            agent_start = timestamp
        elif event_type == "agent_end":
            agent_end = timestamp
        elif event_type == "job_start" and job_start is None:
            job_start = timestamp
        elif event_type == "job_end":
            job_end = timestamp
    return agent_start, agent_end, job_start, job_end


def _load_manifest_window(manifest_path: Path) -> tuple[Any, Any]:
    payload = _load_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {manifest_path}")
    run_start = parse_iso8601_to_utc(payload.get("run_start_time"))
    run_end = parse_iso8601_to_utc(payload.get("run_end_time"))
    return run_start, run_end


def _request_within_cutoff(
    record: dict[str, Any],
    *,
    cutoff_time_utc: Any,
) -> bool:
    if cutoff_time_utc is None:
        return True
    request_start = parse_iso8601_to_utc(record.get("request_start_time"))
    request_end = parse_iso8601_to_utc(record.get("request_end_time"))
    if request_start is not None and request_start > cutoff_time_utc:
        return False
    if request_end is not None and request_end > cutoff_time_utc:
        return False
    return True


def _load_request_window(
    requests_path: Path,
    *,
    cutoff_time_utc: Any,
) -> tuple[Any, Any]:
    first_start = None
    last_end = None
    for record in _iter_jsonl_dict_records(requests_path):
        if not _request_within_cutoff(record, cutoff_time_utc=cutoff_time_utc):
            continue
        request_start = parse_iso8601_to_utc(record.get("request_start_time"))
        request_end = parse_iso8601_to_utc(record.get("request_end_time"))
        if request_start is not None:
            if first_start is None or request_start < first_start:
                first_start = request_start
        if request_end is not None:
            if last_end is None or request_end > last_end:
                last_end = request_end
    return first_start, last_end


def _extract_duration_for_job(
    *,
    run_dir: Path,
    gateway_run_id: str,
    gateway_profile_id: int | None,
    cutoff_time_utc: Any,
) -> tuple[float | None, str | None, str]:
    gateway_run_dir = _resolve_gateway_run_dir(
        run_dir,
        gateway_run_id=gateway_run_id,
        gateway_profile_id=gateway_profile_id,
    )

    lifecycle_path = gateway_run_dir / LIFECYCLE_REL_PATH
    if lifecycle_path.is_file():
        agent_start, agent_end, job_start, job_end = _load_lifecycle_windows(
            lifecycle_path
        )
        agent_start, agent_end = _clamp_window_to_cutoff(
            agent_start,
            agent_end,
            cutoff_time_utc=cutoff_time_utc,
        )
        job_start, job_end = _clamp_window_to_cutoff(
            job_start,
            job_end,
            cutoff_time_utc=cutoff_time_utc,
        )

        duration_s = _duration_seconds(agent_start, agent_end)
        if duration_s is not None:
            return duration_s, "lifecycle:agent", str(gateway_run_dir)

        duration_s = _duration_seconds(job_start, job_end)
        if duration_s is not None:
            return duration_s, "lifecycle:job", str(gateway_run_dir)

    manifest_path = gateway_run_dir / MANIFEST_REL_PATH
    if manifest_path.is_file():
        manifest_start, manifest_end = _load_manifest_window(manifest_path)
        manifest_start, manifest_end = _clamp_window_to_cutoff(
            manifest_start,
            manifest_end,
            cutoff_time_utc=cutoff_time_utc,
        )
        duration_s = _duration_seconds(manifest_start, manifest_end)
        if duration_s is not None:
            return duration_s, "manifest:run", str(gateway_run_dir)

    requests_path = gateway_run_dir / REQUESTS_REL_PATH
    if requests_path.is_file():
        request_start, request_end = _load_request_window(
            requests_path,
            cutoff_time_utc=cutoff_time_utc,
        )
        duration_s = _duration_seconds(request_start, request_end)
        if duration_s is not None:
            return duration_s, "requests:window", str(gateway_run_dir)

    return None, None, str(gateway_run_dir)


def _materialize_panel(selection: PanelSelection) -> dict[str, Any]:
    payload = _load_json(selection.summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {selection.summary_path}")

    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError(f"Missing 'jobs' list in {selection.summary_path}")

    service_failure_payload = ensure_service_failure_payload(selection.run_dir)
    cutoff_time_utc = cutoff_datetime_utc_from_payload(service_failure_payload)

    durations_s: list[float] = []
    duration_source_counts: dict[str, int] = {}
    missing_duration_jobs: list[str] = []
    missing_duration_details: list[dict[str, Any]] = []

    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            continue
        gateway_run_id = _string_or_none(raw_job.get("gateway_run_id"))
        if gateway_run_id is None:
            continue
        gateway_profile_id = _int_or_none(raw_job.get("gateway_profile_id"))
        duration_s, duration_source, gateway_run_dir = _extract_duration_for_job(
            run_dir=selection.run_dir,
            gateway_run_id=gateway_run_id,
            gateway_profile_id=gateway_profile_id,
            cutoff_time_utc=cutoff_time_utc,
        )
        if duration_s is None or duration_source is None:
            missing_duration_jobs.append(gateway_run_id)
            missing_duration_details.append(
                {
                    "gateway_run_id": gateway_run_id,
                    "gateway_profile_id": gateway_profile_id,
                    "gateway_run_dir": gateway_run_dir,
                }
            )
            continue

        durations_s.append(duration_s)
        duration_source_counts[duration_source] = (
            duration_source_counts.get(duration_source, 0) + 1
        )

    if not durations_s:
        raise ValueError(
            f"No usable per-job durations were found for {selection.summary_path}"
        )

    dataset_value = payload.get("dataset")
    dataset_name = (
        dataset_value
        if isinstance(dataset_value, str) and dataset_value.strip()
        else selection.benchmark
    )
    agent_type_value = payload.get("agent_type")
    agent_type_name = (
        agent_type_value
        if isinstance(agent_type_value, str) and agent_type_value.strip()
        else selection.agent_type
    )
    stats = _summarize_durations(durations_s)
    return {
        "benchmark": selection.benchmark,
        "benchmark_label": selection.benchmark_label,
        "agent_type": selection.agent_type,
        "agent_label": selection.agent_label,
        "dataset": dataset_name,
        "reported_agent_type": agent_type_name,
        "panel_label": f"{selection.benchmark_label} | {selection.agent_label}",
        "run_dir": str(selection.run_dir),
        "run_path": selection.run_path,
        "run_dir_name": selection.run_dir_name,
        "summary_path": str(selection.summary_path),
        "candidate_run_count": selection.candidate_run_count,
        "score": _float_or_none(payload.get("score")),
        "job_count_reported": _int_or_none(payload.get("job_count")),
        "service_failure_detected": bool(
            service_failure_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": service_failure_payload.get(
            "cutoff_time_utc"
        ),
        "jobs_with_duration_count": len(durations_s),
        "jobs_missing_duration_count": len(missing_duration_jobs),
        "missing_duration_job_ids": missing_duration_jobs,
        "missing_duration_jobs_preview": missing_duration_details[:10],
        "duration_source_counts": duration_source_counts,
        "avg_agent_duration_s_computed": stats["mean"],
        "max_agent_duration_s_computed": stats["max"],
        "durations_s": durations_s,
        "stats": stats,
    }


def main() -> int:
    args = _parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory does not exist: {root_dir}")

    benchmark_order = [item.strip() for item in args.benchmarks if item.strip()]
    if not benchmark_order:
        raise ValueError("At least one benchmark must be provided")

    agent_order = [_normalize_agent(item) for item in args.agents]
    if not agent_order:
        raise ValueError("At least one agent must be provided")

    selections: list[PanelSelection] = []
    for benchmark in benchmark_order:
        for agent_type in agent_order:
            selections.append(
                _discover_selection(
                    root_dir=root_dir,
                    benchmark=benchmark,
                    agent_type=agent_type,
                )
            )

    panels = [_materialize_panel(selection) for selection in selections]
    for panel_index, panel in enumerate(panels):
        panel["panel_index"] = panel_index

    all_durations = [
        duration_s
        for panel in panels
        for duration_s in panel["durations_s"]
        if isinstance(duration_s, (int, float))
    ]

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(
            root_dir=root_dir,
            benchmark_order=benchmark_order,
            agent_order=agent_order,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    materialized_payload = {
        "figure_name": DEFAULT_OUTPUT_STEM,
        "figure_title": (
            "Agent Duration Distribution by Benchmark and Agent "
            "(Violin + Boxplot)"
        ),
        "metric_name": "agent_duration_s",
        "metric_label": "Agent duration per job (s)",
        "metric_definition": (
            "agent duration = agent_end - agent_start from "
            "gateway-output/*/events/lifecycle.jsonl; fallback to "
            "job_end - job_start from the same lifecycle file; fallback to "
            "run_end_time - run_start_time from gateway manifest.json; fallback to "
            "max(request_end_time) - min(request_start_time) from "
            "requests/model_inference.jsonl"
        ),
        "duration_source_priority": [
            "lifecycle:agent",
            "lifecycle:job",
            "manifest:run",
            "requests:window",
        ],
        "source_root_dir": str(root_dir),
        "summary_rel_path": str(SUMMARY_REL_PATH),
        "benchmark_order": benchmark_order,
        "agent_order": agent_order,
        "panel_count": len(panels),
        "total_job_count": len(all_durations),
        "global_min_duration_s": min(all_durations),
        "global_max_duration_s": max(all_durations),
        "panels": panels,
    }

    output_path.write_text(
        json.dumps(materialized_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
