#!/usr/bin/env python3
"""Materialize per-job turn-count distributions for the turns-scatter-volin figure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_ROOT_DIR = (REPO_ROOT / "results" / "qwen3-coder-30b").resolve()
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "turns-scatter-volin"
DEFAULT_BENCHMARK_ORDER = ("dabstep", "swebench-verified", "terminal-bench-2.0")
DEFAULT_AGENT_ORDER = ("mini-swe-agent", "terminus-2")
SUMMARY_REL_PATH = Path("run-stats/run-stats-summary.json")
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
            "Materialize per-job turn-count distributions for the "
            "turns-scatter-volin figure from run-stats/run-stats-summary.json files."
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
            "Agent directory names or aliases to compare within each benchmark, in plotting "
            "order. 'terminus' is treated as an alias for 'terminus-2'."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path. Default: "
            "figures/turns-scatter-volin/data/"
            "turns-scatter-volin.<root>.benchmarks-<...>.agents-<...>.json"
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _benchmark_label(benchmark: str) -> str:
    return BENCHMARK_LABELS.get(benchmark, benchmark)


def _normalize_agent(agent: str) -> str:
    normalized = agent.strip().lower()
    if not normalized:
        raise ValueError("Agent names must not be empty")
    return AGENT_ALIASES.get(normalized, normalized)


def _agent_label(agent_type: str) -> str:
    return AGENT_LABELS.get(agent_type, agent_type)


def _quantile_from_sorted_values(sorted_values: list[int], quantile: float) -> float:
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


def _summarize_turns(turns: list[int]) -> dict[str, float | int]:
    if not turns:
        raise ValueError("At least one turn count is required")
    sorted_turns = sorted(turns)
    q1 = _quantile_from_sorted_values(sorted_turns, 0.25)
    median = _quantile_from_sorted_values(sorted_turns, 0.50)
    q3 = _quantile_from_sorted_values(sorted_turns, 0.75)
    mean = sum(sorted_turns) / len(sorted_turns)
    return {
        "sample_count": len(sorted_turns),
        "min": min(sorted_turns),
        "q1": q1,
        "median": median,
        "q3": q3,
        "p90": _quantile_from_sorted_values(sorted_turns, 0.90),
        "p95": _quantile_from_sorted_values(sorted_turns, 0.95),
        "mean": mean,
        "max": max(sorted_turns),
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


def _materialize_panel(selection: PanelSelection) -> dict[str, Any]:
    payload = _load_json(selection.summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {selection.summary_path}")

    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError(f"Missing 'jobs' list in {selection.summary_path}")

    turns: list[int] = []
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            continue
        request_count = _int_or_none(raw_job.get("request_count"))
        if request_count is None:
            continue
        if request_count < 0:
            continue
        turns.append(request_count)

    if not turns:
        raise ValueError(
            f"No usable jobs[].request_count values were found in {selection.summary_path}"
        )

    dataset_value = payload.get("dataset")
    dataset_name = dataset_value if isinstance(dataset_value, str) and dataset_value.strip() else selection.benchmark
    agent_type_value = payload.get("agent_type")
    agent_type_name = (
        agent_type_value
        if isinstance(agent_type_value, str) and agent_type_value.strip()
        else selection.agent_type
    )
    stats = _summarize_turns(turns)
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
        "avg_turns_per_run_reported": _float_or_none(payload.get("avg_turns_per_run")),
        "avg_turns_per_run_computed": stats["mean"],
        "max_turns_per_run_computed": stats["max"],
        "turns": turns,
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

    all_turns = [
        turn
        for panel in panels
        for turn in panel["turns"]
        if isinstance(turn, int)
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
        "figure_title": "Turn Count Distribution by Benchmark and Agent (Violin + Boxplot)",
        "metric_name": "turns",
        "metric_label": "Turns per job",
        "metric_definition": "turns = jobs[].request_count from run-stats-summary.json",
        "turn_source_field": "jobs[].request_count",
        "source_root_dir": str(root_dir),
        "summary_rel_path": str(SUMMARY_REL_PATH),
        "benchmark_order": benchmark_order,
        "agent_order": agent_order,
        "panel_count": len(panels),
        "total_job_count": len(all_turns),
        "global_min_turns": min(all_turns),
        "global_max_turns": max(all_turns),
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
