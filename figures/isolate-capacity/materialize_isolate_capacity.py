#!/usr/bin/env python3
"""Materialize throughput, completed-agent LLM time, and completion-token metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "isolate-capacity"
DEFAULT_CASE_SLUGS = ("trail", "trail-usage75", "trail-usage75-lmcache100")
DEFAULT_CASE_LABELS = {
    "trail": "TRAIL",
    "trail-usage75": "TRAIL 75%",
    "trail-usage75-lmcache100": "TRAIL 75% + LMCache",
}
DEFAULT_QPS_SLUG = "qps0_5"


@dataclass(frozen=True)
class MetricSpec:
    metric_key: str
    metric_label: str
    panel_title: str
    metric_unit: str
    y_axis_label: str
    source_rel_path: Path
    extraction_mode: str
    points_field: str | None
    value_field: str | None
    nested_metric_key: str | None
    nested_stat_field: str | None
    formula: str


@dataclass(frozen=True)
class CaseSelection:
    case_slug: str
    case_label: str
    run_dir: Path
    run_path: str
    profile_dir_name: str | None
    qps_slug: str
    run_dir_name: str
    candidate_run_count: int


METRIC_SPECS = (
    MetricSpec(
        metric_key="average_throughput_jobs_per_s",
        metric_label="Average Throughput",
        panel_title="Average Throughput",
        metric_unit="jobs/s",
        y_axis_label="Throughput (jobs/s)",
        source_rel_path=Path("post-processed/job-throughput/job-throughput-timeseries.json"),
        extraction_mode="points-average",
        points_field="throughput_points",
        value_field="throughput_jobs_per_s",
        nested_metric_key=None,
        nested_stat_field=None,
        formula="mean(throughput_points[].throughput_jobs_per_s)",
    ),
    MetricSpec(
        metric_key="average_completed_agent_llm_time_s",
        metric_label="Average Completed-Agent LLM Time",
        panel_title="Completed-Agent LLM Time",
        metric_unit="s",
        y_axis_label="Average Completed-Agent LLM Time (s)",
        source_rel_path=Path("post-processed/agent-output-throughput/agent-output-throughput.json"),
        extraction_mode="completed-agent-duration-average",
        points_field="agents",
        value_field="llm_request_duration_s",
        nested_metric_key=None,
        nested_stat_field=None,
        formula=(
            "mean(agents[replay_completed == true].llm_request_duration_s); for "
            "identical completed agents this is proportional to the reciprocal of "
            "completed-replay-only output throughput"
        ),
    ),
    MetricSpec(
        metric_key="average_completion_tokens",
        metric_label="Average Completion Tokens",
        panel_title="Average Completion Tokens",
        metric_unit="tokens",
        y_axis_label="Average Completion Tokens",
        source_rel_path=Path("post-processed/gateway/stack/completion-tokens-stacked-histogram.json"),
        extraction_mode="points-average",
        points_field="points",
        value_field="accumulated_value",
        nested_metric_key=None,
        nested_stat_field=None,
        formula="mean(points[].accumulated_value)",
    ),
)
REQUIRED_REL_PATHS = tuple(metric.source_rel_path for metric in METRIC_SPECS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the isolate-capacity figure dataset by selecting one run for each "
            "case slug and materializing the requested comparison metrics."
        )
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory that contains the compared case directories.",
    )
    parser.add_argument(
        "--case-slugs",
        nargs="+",
        default=list(DEFAULT_CASE_SLUGS),
        help=(
            "Case directory names to compare, in plotting order. "
            f"Default: {' '.join(DEFAULT_CASE_SLUGS)}"
        ),
    )
    parser.add_argument(
        "--case-labels",
        nargs="+",
        default=None,
        help=(
            "Optional human-readable labels for --case-slugs, in the same order. "
            "Default: built-in labels for the known case slugs."
        ),
    )
    parser.add_argument(
        "--qps-slug",
        default=DEFAULT_QPS_SLUG,
        help=f"QPS directory name to compare (default: {DEFAULT_QPS_SLUG}).",
    )
    parser.add_argument(
        "--profile-dir-name",
        default=None,
        help=(
            "Optional exact profile directory name to require beneath each case root. "
            "Default: search all matching profiles and select the latest matching run."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path. Default: "
            "figures/isolate-capacity/data/isolate-capacity.<qps-slug>.json"
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


def _default_output_path(qps_slug: str) -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{DEFAULT_OUTPUT_STEM}.{qps_slug}.json").resolve()


def _default_case_label(case_slug: str) -> str:
    return DEFAULT_CASE_LABELS.get(case_slug, case_slug)


def _resolve_case_labels(args: argparse.Namespace) -> dict[str, str]:
    case_slugs = list(args.case_slugs)
    if args.case_labels is None:
        return {case_slug: _default_case_label(case_slug) for case_slug in case_slugs}
    if len(args.case_labels) != len(case_slugs):
        raise ValueError(
            "--case-labels must have the same number of items as --case-slugs "
            f"(got {len(args.case_labels)} labels for {len(case_slugs)} case slugs)"
        )
    return {
        case_slug: case_label
        for case_slug, case_label in zip(case_slugs, args.case_labels)
    }


def _discover_run_dir(
    *,
    root_dir: Path,
    case_slug: str,
    case_label: str,
    qps_slug: str,
    profile_dir_name: str | None,
) -> CaseSelection:
    case_root = (root_dir / case_slug).resolve()
    if not case_root.is_dir():
        raise ValueError(f"Case directory does not exist: {case_root}")

    candidates: list[Path] = []
    for qps_dir in case_root.rglob(qps_slug):
        if not qps_dir.is_dir() or qps_dir.name != qps_slug:
            continue
        if profile_dir_name is not None and qps_dir.parent.name != profile_dir_name:
            continue
        for run_dir in qps_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if all((run_dir / rel_path).is_file() for rel_path in REQUIRED_REL_PATHS):
                candidates.append(run_dir.resolve())

    if not candidates:
        extra_filter = (
            f" with profile filter {profile_dir_name!r}" if profile_dir_name is not None else ""
        )
        raise ValueError(
            f"No matching run directories were found for case {case_slug!r}, "
            f"qps {qps_slug!r}{extra_filter} under {case_root}"
        )

    selected_run_dir = max(candidates, key=lambda path: (path.name, str(path)))
    run_path = str(selected_run_dir.relative_to(root_dir.resolve()))
    selected_profile_dir_name = (
        selected_run_dir.parent.parent.name
        if selected_run_dir.parent.parent != selected_run_dir.parent
        else None
    )
    return CaseSelection(
        case_slug=case_slug,
        case_label=case_label,
        run_dir=selected_run_dir,
        run_path=run_path,
        profile_dir_name=selected_profile_dir_name,
        qps_slug=selected_run_dir.parent.name,
        run_dir_name=selected_run_dir.name,
        candidate_run_count=len(candidates),
    )


def _summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("At least one numeric value is required")
    return {
        "value": sum(values) / len(values),
        "sample_count": len(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _extract_metric_summary(
    *,
    run_dir: Path,
    metric_spec: MetricSpec,
) -> dict[str, Any]:
    input_path = (run_dir / metric_spec.source_rel_path).resolve()
    payload = _load_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {input_path}")

    if metric_spec.extraction_mode == "points-average":
        raw_points = payload.get(metric_spec.points_field)
        if not isinstance(raw_points, list):
            raise ValueError(
                f"Missing list field {metric_spec.points_field!r} in {input_path}"
            )

        values: list[float] = []
        for raw_point in raw_points:
            if not isinstance(raw_point, dict):
                continue
            value = _float_or_none(raw_point.get(metric_spec.value_field))
            if value is None:
                continue
            values.append(value)
        if not values:
            raise ValueError(
                f"No usable {metric_spec.value_field!r} values were found in {input_path}"
            )

        summary = _summarize_values(values)
        summary.update(
            {
                "source_file": str(input_path),
                "source_rel_path": str(metric_spec.source_rel_path),
                "points_field": metric_spec.points_field,
                "value_field": metric_spec.value_field,
            }
        )
    elif metric_spec.extraction_mode == "completed-agent-duration-average":
        raw_points = payload.get(metric_spec.points_field)
        if not isinstance(raw_points, list):
            raise ValueError(
                f"Missing list field {metric_spec.points_field!r} in {input_path}"
            )

        values: list[float] = []
        output_tokens_values: list[float] = []
        for raw_point in raw_points:
            if not isinstance(raw_point, dict):
                continue
            if raw_point.get("replay_completed") is not True:
                continue
            value = _float_or_none(raw_point.get(metric_spec.value_field))
            if value is None:
                continue
            values.append(value)
            output_tokens = _float_or_none(raw_point.get("output_tokens"))
            if output_tokens is not None:
                output_tokens_values.append(output_tokens)
        if not values:
            raise ValueError(
                f"No usable {metric_spec.value_field!r} values were found in {input_path}"
            )

        summary = _summarize_values(values)
        summary.update(
            {
                "source_file": str(input_path),
                "source_rel_path": str(metric_spec.source_rel_path),
                "points_field": metric_spec.points_field,
                "value_field": metric_spec.value_field,
                "completed_replay_only": True,
            }
        )
        if output_tokens_values:
            summary["completed_agent_output_tokens_summary"] = _summarize_values(
                output_tokens_values
            )
    elif metric_spec.extraction_mode == "nested-metric-stat":
        raw_metrics = payload.get("metrics")
        if not isinstance(raw_metrics, dict):
            raise ValueError(f"Missing 'metrics' object in {input_path}")
        metric_payload = raw_metrics.get(metric_spec.nested_metric_key)
        if not isinstance(metric_payload, dict):
            raise ValueError(
                f"Missing metric {metric_spec.nested_metric_key!r} in {input_path}"
            )
        value = _float_or_none(metric_payload.get(metric_spec.nested_stat_field))
        if value is None:
            raise ValueError(
                f"Metric {metric_spec.nested_metric_key!r} is missing numeric "
                f"field {metric_spec.nested_stat_field!r} in {input_path}"
            )
        count = metric_payload.get("count")
        sample_count = int(count) if isinstance(count, int) else 1
        summary = {
            "value": value,
            "sample_count": sample_count,
            "min": _float_or_none(metric_payload.get("min")),
            "max": _float_or_none(metric_payload.get("max")),
            "std": None,
            "source_file": str(input_path),
            "source_rel_path": str(metric_spec.source_rel_path),
            "nested_metric_key": metric_spec.nested_metric_key,
            "nested_stat_field": metric_spec.nested_stat_field,
        }
    else:
        raise ValueError(f"Unsupported extraction mode: {metric_spec.extraction_mode}")

    if "window_size_s" in payload:
        window_size_s = _float_or_none(payload.get("window_size_s"))
        if window_size_s is not None:
            summary["window_size_s"] = window_size_s
    if "bucket_width_s" in payload:
        bucket_width_s = _float_or_none(payload.get("bucket_width_s"))
        if bucket_width_s is not None:
            summary["bucket_width_s"] = bucket_width_s
    if isinstance(payload.get("metric"), str):
        summary["source_metric"] = payload["metric"]

    return summary


def main() -> int:
    args = _parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory does not exist: {root_dir}")

    case_labels = _resolve_case_labels(args)
    case_selections = [
        _discover_run_dir(
            root_dir=root_dir,
            case_slug=case_slug,
            case_label=case_labels[case_slug],
            qps_slug=args.qps_slug,
            profile_dir_name=args.profile_dir_name,
        )
        for case_slug in args.case_slugs
    ]

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(args.qps_slug)
    )

    payload: dict[str, Any] = {
        "figure_name": DEFAULT_OUTPUT_STEM,
        "source_root_dir": str(root_dir),
        "qps_slug": args.qps_slug,
        "profile_dir_name_filter": args.profile_dir_name,
        "selection_policy": (
            "latest matching run directory name within each case/qps search scope"
        ),
        "metric_count": len(METRIC_SPECS),
        "case_count": len(case_selections),
        "metrics": [
            {
                "metric_key": metric.metric_key,
                "metric_label": metric.metric_label,
                "panel_title": metric.panel_title,
                "metric_unit": metric.metric_unit,
                "y_axis_label": metric.y_axis_label,
                "source_rel_path": str(metric.source_rel_path),
                "extraction_mode": metric.extraction_mode,
                "points_field": metric.points_field,
                "value_field": metric.value_field,
                "nested_metric_key": metric.nested_metric_key,
                "nested_stat_field": metric.nested_stat_field,
                "formula": metric.formula,
            }
            for metric in METRIC_SPECS
        ],
        "cases": [],
    }

    for case_selection in case_selections:
        metric_values: dict[str, float] = {}
        metric_summaries: dict[str, dict[str, Any]] = {}
        for metric_spec in METRIC_SPECS:
            summary = _extract_metric_summary(
                run_dir=case_selection.run_dir,
                metric_spec=metric_spec,
            )
            metric_summaries[metric_spec.metric_key] = summary
            metric_values[metric_spec.metric_key] = float(summary["value"])

        payload["cases"].append(
            {
                "case_slug": case_selection.case_slug,
                "case_label": case_selection.case_label,
                "run_dir": str(case_selection.run_dir),
                "run_path": case_selection.run_path,
                "profile_dir_name": case_selection.profile_dir_name,
                "qps_slug": case_selection.qps_slug,
                "run_dir_name": case_selection.run_dir_name,
                "candidate_run_count": case_selection.candidate_run_count,
                "metric_values": metric_values,
                "metrics": metric_summaries,
            }
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
