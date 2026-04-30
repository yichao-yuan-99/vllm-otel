#!/usr/bin/env python3
"""Materialize combined turn-count and duration distributions for a transposed figure."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_ROOT_DIR = (REPO_ROOT / "results" / "qwen3-coder-30b").resolve()
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "turns-scatter-duration-violin"
DEFAULT_BENCHMARK_ORDER = ("dabstep", "swebench-verified", "terminal-bench-2.0")
DEFAULT_AGENT_ORDER = ("mini-swe-agent", "terminus-2")
AGENT_ALIASES = {
    "mini-swe-agent": "mini-swe-agent",
    "minisweagent": "mini-swe-agent",
    "terminus": "terminus-2",
    "terminus-2": "terminus-2",
}
TURNS_MATERIALIZER_PATH = (
    REPO_ROOT
    / "figures"
    / "turns-scatter-volin"
    / "materialize_turns_scatter_volin.py"
)
DURATION_MATERIALIZER_PATH = (
    REPO_ROOT
    / "figures"
    / "turns-duration-violin"
    / "materialize_turns_duration_violin.py"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a combined turns-and-duration dataset for the "
            "turns-scatter-duration-violin figure."
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
            "figures/turns-scatter-duration-violin/data/"
            "turns-scatter-duration-violin.<root>.benchmarks-<...>.agents-<...>.json"
        ),
    )
    return parser.parse_args()


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


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


def _normalize_agent(agent: str) -> str:
    normalized = agent.strip().lower()
    if not normalized:
        raise ValueError("Agent names must not be empty")
    return AGENT_ALIASES.get(normalized, normalized)


def _ensure_matching_panel_metadata(
    turns_panel: dict[str, Any],
    duration_panel: dict[str, Any],
) -> None:
    for key in (
        "benchmark",
        "benchmark_label",
        "agent_type",
        "agent_label",
        "run_dir",
        "run_path",
        "run_dir_name",
        "summary_path",
    ):
        if turns_panel.get(key) != duration_panel.get(key):
            raise ValueError(
                "Turns and duration materializers selected different source runs "
                f"for key {key!r}: {turns_panel.get(key)!r} != {duration_panel.get(key)!r}"
            )


def _merge_panel_payload(
    *,
    panel_index: int,
    turns_panel: dict[str, Any],
    duration_panel: dict[str, Any],
) -> dict[str, Any]:
    _ensure_matching_panel_metadata(turns_panel, duration_panel)
    score = _float_or_none(turns_panel.get("score"))
    if score is None:
        score = _float_or_none(duration_panel.get("score"))

    return {
        "panel_index": panel_index,
        "benchmark": turns_panel["benchmark"],
        "benchmark_label": turns_panel["benchmark_label"],
        "agent_type": turns_panel["agent_type"],
        "agent_label": turns_panel["agent_label"],
        "panel_label": turns_panel.get("panel_label")
        or duration_panel.get("panel_label")
        or (
            f"{turns_panel['benchmark_label']} | {turns_panel['agent_label']}"
        ),
        "dataset": turns_panel.get("dataset") or duration_panel.get("dataset"),
        "reported_agent_type": (
            turns_panel.get("reported_agent_type")
            or duration_panel.get("reported_agent_type")
        ),
        "run_dir": turns_panel["run_dir"],
        "run_path": turns_panel["run_path"],
        "run_dir_name": turns_panel["run_dir_name"],
        "summary_path": turns_panel["summary_path"],
        "candidate_run_count": _int_or_none(turns_panel.get("candidate_run_count"))
        or _int_or_none(duration_panel.get("candidate_run_count")),
        "score": score,
        "job_count_reported": _int_or_none(turns_panel.get("job_count_reported"))
        or _int_or_none(duration_panel.get("job_count_reported")),
        "avg_turns_per_run_reported": _float_or_none(
            turns_panel.get("avg_turns_per_run_reported")
        ),
        "avg_turns_per_run_computed": _float_or_none(
            turns_panel.get("avg_turns_per_run_computed")
        ),
        "max_turns_per_run_computed": _float_or_none(
            turns_panel.get("max_turns_per_run_computed")
        ),
        "turns": list(turns_panel.get("turns", [])),
        "turn_stats": turns_panel.get("stats", {}),
        "service_failure_detected": bool(
            duration_panel.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": duration_panel.get(
            "service_failure_cutoff_time_utc"
        ),
        "jobs_with_duration_count": _int_or_none(
            duration_panel.get("jobs_with_duration_count")
        ),
        "jobs_missing_duration_count": _int_or_none(
            duration_panel.get("jobs_missing_duration_count")
        ),
        "missing_duration_job_ids": list(
            duration_panel.get("missing_duration_job_ids", [])
        ),
        "missing_duration_jobs_preview": list(
            duration_panel.get("missing_duration_jobs_preview", [])
        ),
        "duration_source_counts": dict(duration_panel.get("duration_source_counts", {})),
        "avg_agent_duration_s_computed": _float_or_none(
            duration_panel.get("avg_agent_duration_s_computed")
        ),
        "max_agent_duration_s_computed": _float_or_none(
            duration_panel.get("max_agent_duration_s_computed")
        ),
        "durations_s": list(duration_panel.get("durations_s", [])),
        "duration_stats": duration_panel.get("stats", {}),
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

    turns_module = _load_module(
        "turns_scatter_volin_materializer",
        TURNS_MATERIALIZER_PATH,
    )
    duration_module = _load_module(
        "turns_duration_violin_materializer",
        DURATION_MATERIALIZER_PATH,
    )

    panels: list[dict[str, Any]] = []
    for panel_index, benchmark in enumerate(benchmark_order):
        for agent_type in agent_order:
            turns_selection = turns_module._discover_selection(  # type: ignore[attr-defined]
                root_dir=root_dir,
                benchmark=benchmark,
                agent_type=agent_type,
            )
            duration_selection = duration_module._discover_selection(  # type: ignore[attr-defined]
                root_dir=root_dir,
                benchmark=benchmark,
                agent_type=agent_type,
            )
            turns_panel = turns_module._materialize_panel(  # type: ignore[attr-defined]
                turns_selection
            )
            duration_panel = duration_module._materialize_panel(  # type: ignore[attr-defined]
                duration_selection
            )
            panels.append(
                _merge_panel_payload(
                    panel_index=len(panels),
                    turns_panel=turns_panel,
                    duration_panel=duration_panel,
                )
            )

    all_turns = [
        turn
        for panel in panels
        for turn in panel["turns"]
        if isinstance(turn, int)
    ]
    if not all_turns:
        raise ValueError("No usable turn values were materialized")

    all_durations = [
        duration_s
        for panel in panels
        for duration_s in panel["durations_s"]
        if isinstance(duration_s, (int, float))
    ]
    if not all_durations:
        raise ValueError("No usable duration values were materialized")

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
            "Turn Count and Agent Duration Distributions by Benchmark and Agent "
            "(Horizontal Violin + Boxplot)"
        ),
        "figure_orientation": "horizontal",
        "source_root_dir": str(root_dir),
        "benchmark_order": benchmark_order,
        "agent_order": agent_order,
        "turn_metric_name": "turns",
        "turn_metric_label": "Turns per job",
        "turn_metric_definition": (
            "turns = jobs[].request_count from run-stats/run-stats-summary.json"
        ),
        "duration_metric_name": "agent_duration_s",
        "duration_metric_label": "Agent duration per job (s)",
        "duration_metric_definition": (
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
        "summary_rel_path": "run-stats/run-stats-summary.json",
        "panel_count": len(panels),
        "total_turn_job_count": len(all_turns),
        "total_duration_job_count": len(all_durations),
        "global_min_turns": min(all_turns),
        "global_max_turns": max(all_turns),
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
