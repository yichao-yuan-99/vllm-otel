#!/usr/bin/env python3
"""Materialize the control comparison figure dataset."""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "con-ctrl-compare.json"
DEFAULT_MISSING_LOG_PATH = DEFAULT_OUTPUT_DIR / "con-ctrl-compare.missing.log"

VARIANTS = (
    {
        "variant_key": "no_freq_control",
        "label": "No Freq Control",
        "base_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean/dabstep/mini-swe-agent/"
            "split/exclude-unranked/qps0_05"
        ),
    },
    {
        "variant_key": "kairos_no_thrashing_avoidance",
        "label": "KAIROS\nNo Thrash Avoid.",
        "base_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance/"
            "dabstep/mini-swe-agent/split/exclude-unranked/qps0_05/20260413T135808Z"
        ),
    },
    {
        "variant_key": "kairos_with_thrashing_avoidance",
        "label": "KAIROS\nThrash Avoid.",
        "base_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware/"
            "dabstep/mini-swe-agent/split/exclude-unranked/qps0_05/20260413T235024Z"
        ),
    },
)


def _utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the two-metric dataset for the con-ctrl-compare figure."
        )
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Optional output JSON path. Default: "
            "figures/con-ctrl-compare/data/con-ctrl-compare.json"
        ),
    )
    parser.add_argument(
        "--missing-log",
        default=str(DEFAULT_MISSING_LOG_PATH),
        help=(
            "Optional missing-data log path. Default: "
            "figures/con-ctrl-compare/data/con-ctrl-compare.missing.log"
        ),
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


def _stable_round(value: float | None) -> float:
    return 0.0 if value is None else round(value, 6)


def _percentile(sorted_values: list[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * quantile
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    fraction = rank - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def _select_run_dir(base_path: Path, missing_log: list[str]) -> Path | None:
    if not base_path.exists():
        missing_log.append(f"[missing-path] {base_path}")
        return None

    if (base_path / "replay" / "summary.json").is_file():
        return base_path

    candidates = sorted(
        (child for child in base_path.iterdir() if child.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "replay" / "summary.json").is_file():
            return candidate

    missing_log.append(f"[missing-run-dir] {base_path}")
    return None


def _extract_p5_output_throughput_tokens_per_s(
    throughput_payload: dict[str, Any] | None,
    *,
    run_dir: Path,
    missing_log: list[str],
) -> tuple[float | None, str | None]:
    if throughput_payload is None:
        missing_log.append(
            f"[missing-file] {run_dir / 'post-processed/agent-output-throughput/agent-output-throughput.json'}"
        )
        return (None, None)

    summary = throughput_payload.get("agent_output_throughput_tokens_per_s_summary")
    if isinstance(summary, dict):
        percentiles = summary.get("percentiles")
        if isinstance(percentiles, dict):
            p5_value = _float_or_none(percentiles.get("5"))
            if p5_value is not None:
                return (
                    p5_value,
                    "agent_output_throughput_tokens_per_s_summary.percentiles['5']",
                )

    agents = throughput_payload.get("agents")
    if isinstance(agents, list):
        values = sorted(
            numeric
            for numeric in (
                _float_or_none(agent.get("output_throughput_tokens_per_s"))
                for agent in agents
                if isinstance(agent, dict)
            )
            if numeric is not None
        )
        p5_value = _percentile(values, 0.05)
        if p5_value is not None:
            return (p5_value, "percentile_5(agents[].output_throughput_tokens_per_s)")

    missing_log.append(
        "[missing-metric] p5_output_throughput_tokens_per_s "
        f"{run_dir / 'post-processed/agent-output-throughput/agent-output-throughput.json'}"
    )
    return (None, None)


def _extract_average_job_throughput_jobs_per_s(
    job_throughput_payload: dict[str, Any] | None,
    *,
    run_dir: Path,
    missing_log: list[str],
) -> tuple[float | None, str | None]:
    if job_throughput_payload is None:
        missing_log.append(
            f"[missing-file] {run_dir / 'post-processed/job-throughput/job-throughput-timeseries.json'}"
        )
        return (None, None)

    throughput_points = job_throughput_payload.get("throughput_points")
    if isinstance(throughput_points, list):
        values = [
            numeric
            for numeric in (
                _float_or_none(point.get("throughput_jobs_per_s"))
                for point in throughput_points
                if isinstance(point, dict)
            )
            if numeric is not None
        ]
        if values:
            return (sum(values) / len(values), "mean(throughput_points[].throughput_jobs_per_s)")

    missing_log.append(
        "[missing-metric] average_job_throughput_jobs_per_s "
        f"{run_dir / 'post-processed/job-throughput/job-throughput-timeseries.json'}"
    )
    return (None, None)


def _build_variant_record(
    variant: dict[str, str],
    *,
    missing_log: list[str],
) -> dict[str, Any]:
    base_path = Path(variant["base_path"]).expanduser().resolve()
    run_dir = _select_run_dir(base_path, missing_log)

    throughput_payload: dict[str, Any] | None = None
    job_throughput_payload: dict[str, Any] | None = None
    if run_dir is not None:
        throughput_path = (
            run_dir
            / "post-processed"
            / "agent-output-throughput"
            / "agent-output-throughput.json"
        )
        job_throughput_path = (
            run_dir
            / "post-processed"
            / "job-throughput"
            / "job-throughput-timeseries.json"
        )
        if throughput_path.is_file():
            loaded_throughput = _load_json(throughput_path)
            if isinstance(loaded_throughput, dict):
                throughput_payload = loaded_throughput
        if job_throughput_path.is_file():
            loaded_job_throughput = _load_json(job_throughput_path)
            if isinstance(loaded_job_throughput, dict):
                job_throughput_payload = loaded_job_throughput

    p5_output_throughput_tokens_per_s, p5_source = (
        _extract_p5_output_throughput_tokens_per_s(
            throughput_payload,
            run_dir=run_dir if run_dir is not None else base_path,
            missing_log=missing_log,
        )
    )
    average_job_throughput_jobs_per_s, job_source = (
        _extract_average_job_throughput_jobs_per_s(
            job_throughput_payload,
            run_dir=run_dir if run_dir is not None else base_path,
            missing_log=missing_log,
        )
    )

    return {
        "variant_key": variant["variant_key"],
        "label": variant["label"],
        "base_path": str(base_path),
        "selected_run_dir": None if run_dir is None else str(run_dir),
        "metrics": {
            "p5_output_throughput_tokens_per_s": _stable_round(
                p5_output_throughput_tokens_per_s
            ),
            "average_job_throughput_jobs_per_s": _stable_round(
                average_job_throughput_jobs_per_s
            ),
        },
        "metric_sources": {
            "p5_output_throughput_tokens_per_s": p5_source,
            "average_job_throughput_jobs_per_s": job_source,
        },
    }


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    missing_log_path = Path(args.missing_log).expanduser().resolve()

    missing_log: list[str] = []
    variants = [
        _build_variant_record(variant, missing_log=missing_log)
        for variant in VARIANTS
    ]

    payload = {
        "figure_name": "con-ctrl-compare",
        "created_at_utc": _utc_now_timestamp(),
        "dataset": "dabstep",
        "agent": "mini-swe-agent",
        "qps": 0.05,
        "metrics": [
            {
                "metric_key": "p5_output_throughput_tokens_per_s",
                "label": "Per Agent P5 Throughput",
                "unit": "tokens/s",
            },
            {
                "metric_key": "average_job_throughput_jobs_per_s",
                "label": "System Job Throughput",
                "unit": "jobs/s",
            },
        ],
        "variant_count": len(variants),
        "variants": variants,
        "missing_log_path": str(missing_log_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
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
