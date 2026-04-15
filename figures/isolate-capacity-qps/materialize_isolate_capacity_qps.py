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
DEFAULT_OUTPUT_STEM = "isolate-capacity-qps"
DEFAULT_SOURCE_DIRS = (
    "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/"
    "swebench-verified/mini-swe-agent/trail/"
    "profile-3_run_20260306T072549Z_e5e91775e142_3ac3451764525449b91ff296d1ada643/"
    "qps0_5",
    "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/"
    "swebench-verified/mini-swe-agent/trail/"
    "profile-3_run_20260306T072549Z_e5e91775e142_3ac3451764525449b91ff296d1ada643/"
    "qps0_6/20260331T195832Z",
    "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/"
    "swebench-verified/mini-swe-agent/trail-lmcache/"
    "profile-3_run_20260306T072549Z_e5e91775e142_3ac3451764525449b91ff296d1ada643/"
    "qps0_6/20260331T212051Z",
)
DEFAULT_SOURCE_LABELS = (
    "TRAIL 0.5",
    "TRAIL 0.6",
    "TRAIL + LMCache 0.6",
)


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
class SourceSelection:
    source_input_path: Path
    source_input_kind: str
    case_slug: str | None
    case_label: str
    run_dir: Path
    profile_dir_name: str | None
    qps_slug: str | None
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
            "Materialize the isolate-capacity-qps figure dataset from explicit source "
            "directories or run directories."
        )
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=list(DEFAULT_SOURCE_DIRS),
        help=(
            "Ordered source directories to compare. Each path may be either an exact "
            "run directory or a QPS directory whose child run directories should be "
            "searched. Default: the three requested isolate-capacity-qps sources."
        ),
    )
    parser.add_argument(
        "--source-labels",
        nargs="+",
        default=None,
        help=(
            "Optional human-readable labels for --source-dirs, in the same order. "
            "Default: built-in labels for the three default sources, or a label "
            "derived from case slug and QPS slug."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path. Default: "
            "figures/isolate-capacity-qps/data/isolate-capacity-qps.json"
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


def _default_output_path() -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{DEFAULT_OUTPUT_STEM}.json").resolve()


def _ancestor_name(path: Path, levels_up: int) -> str | None:
    current = path
    for _ in range(levels_up):
        parent = current.parent
        if parent == current:
            return None
        current = parent
    return current.name or None


def _format_qps_slug(qps_slug: str | None) -> str | None:
    if qps_slug is None:
        return None
    if qps_slug.startswith("qps"):
        return qps_slug[3:].replace("_", ".")
    return qps_slug


def _format_case_slug(case_slug: str | None) -> str:
    if case_slug is None or not case_slug:
        return "Source"
    case_label_overrides = {
        "trail": "TRAIL",
        "trail-lmcache": "TRAIL + LMCache",
    }
    if case_slug in case_label_overrides:
        return case_label_overrides[case_slug]
    return case_slug.replace("-", " ")


def _default_source_label(*, case_slug: str | None, qps_slug: str | None) -> str:
    case_label = _format_case_slug(case_slug)
    formatted_qps = _format_qps_slug(qps_slug)
    if formatted_qps is None:
        return case_label
    return f"{case_label} {formatted_qps}"


def _resolve_source_labels(
    source_dirs: list[str],
    source_selections: list[SourceSelection],
    source_labels: list[str] | None,
) -> list[str]:
    if source_labels is not None:
        if len(source_labels) != len(source_dirs):
            raise ValueError(
                "--source-labels must have the same number of items as --source-dirs "
                f"(got {len(source_labels)} labels for {len(source_dirs)} source dirs)"
            )
        return list(source_labels)

    if source_dirs == list(DEFAULT_SOURCE_DIRS):
        return list(DEFAULT_SOURCE_LABELS)

    return [
        _default_source_label(
            case_slug=selection.case_slug,
            qps_slug=selection.qps_slug,
        )
        for selection in source_selections
    ]


def _is_run_dir(path: Path) -> bool:
    return all((path / rel_path).is_file() for rel_path in REQUIRED_REL_PATHS)


def _resolve_run_dir(source_input_path: Path) -> tuple[Path, str, int]:
    if _is_run_dir(source_input_path):
        return source_input_path, "run_dir", 1

    candidates = [
        child.resolve()
        for child in source_input_path.iterdir()
        if child.is_dir() and _is_run_dir(child)
    ]
    if not candidates:
        raise ValueError(
            "Source path is neither a usable run directory nor a QPS directory with "
            f"matching child runs: {source_input_path}"
        )
    selected_run_dir = max(candidates, key=lambda path: (path.name, str(path)))
    return selected_run_dir, "qps_dir", len(candidates)


def _discover_source(source_dir: str) -> SourceSelection:
    source_input_path = Path(source_dir).expanduser().resolve()
    if not source_input_path.is_dir():
        raise ValueError(f"Source directory does not exist: {source_input_path}")

    run_dir, source_input_kind, candidate_run_count = _resolve_run_dir(source_input_path)
    case_slug = _ancestor_name(run_dir, 3)
    return SourceSelection(
        source_input_path=source_input_path,
        source_input_kind=source_input_kind,
        case_slug=case_slug,
        case_label=_default_source_label(
            case_slug=case_slug,
            qps_slug=_ancestor_name(run_dir, 1),
        ),
        run_dir=run_dir,
        profile_dir_name=_ancestor_name(run_dir, 2),
        qps_slug=_ancestor_name(run_dir, 1),
        run_dir_name=run_dir.name,
        candidate_run_count=candidate_run_count,
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
    source_selections = [_discover_source(source_dir) for source_dir in args.source_dirs]
    source_labels = _resolve_source_labels(
        list(args.source_dirs),
        source_selections,
        args.source_labels,
    )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path()
    )

    source_qps_slugs = list(
        dict.fromkeys(
            selection.qps_slug
            for selection in source_selections
            if isinstance(selection.qps_slug, str) and selection.qps_slug
        )
    )
    payload: dict[str, Any] = {
        "figure_name": DEFAULT_OUTPUT_STEM,
        "selection_policy": (
            "for each explicit source path, use it directly if it is a run directory; "
            "otherwise, if it is a QPS directory, select the latest matching child run "
            "directory name"
        ),
        "source_count": len(source_selections),
        "source_qps_slugs": source_qps_slugs,
        "metric_count": len(METRIC_SPECS),
        "case_count": len(source_selections),
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

    for source_selection, case_label in zip(source_selections, source_labels):
        metric_values: dict[str, float] = {}
        metric_summaries: dict[str, dict[str, Any]] = {}
        for metric_spec in METRIC_SPECS:
            summary = _extract_metric_summary(
                run_dir=source_selection.run_dir,
                metric_spec=metric_spec,
            )
            metric_summaries[metric_spec.metric_key] = summary
            metric_values[metric_spec.metric_key] = float(summary["value"])

        payload["cases"].append(
            {
                "case_slug": source_selection.case_slug,
                "case_label": case_label,
                "source_input_path": str(source_selection.source_input_path),
                "source_input_kind": source_selection.source_input_kind,
                "run_dir": str(source_selection.run_dir),
                "profile_dir_name": source_selection.profile_dir_name,
                "qps_slug": source_selection.qps_slug,
                "run_dir_name": source_selection.run_dir_name,
                "candidate_run_count": source_selection.candidate_run_count,
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
