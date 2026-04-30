#!/usr/bin/env python3
"""Materialize the SLO comparison figure dataset."""

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
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "slo-compare.json"
DEFAULT_MISSING_LOG_PATH = DEFAULT_OUTPUT_DIR / "slo-compare.missing.log"

VARIANTS = (
    {
        "variant_key": "no_freq_control",
        "label": "No Freq Control",
        "base_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean/dabstep/mini-swe-agent/"
            "split/exclude-unranked/qps0_03"
        ),
    },
    {
        "variant_key": "freq_control_no_slo",
        "label": "Freq Control\nSLO 20",
        "base_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance/"
            "dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/20260412T014805Z"
        ),
    },
    {
        "variant_key": "freq_control_slo_35",
        "label": "Freq Control\nSLO 35",
        "base_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/"
            "dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/35"
        ),
    },
    {
        "variant_key": "freq_control_slo_45",
        "label": "Freq Control\nSLO 45",
        "base_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/"
            "dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/45"
        ),
    },
)


def _utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the two-metric dataset for the slo-compare figure."
        )
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Optional output JSON path. Default: "
            "figures/slo-compare/data/slo-compare.json"
        ),
    )
    parser.add_argument(
        "--missing-log",
        default=str(DEFAULT_MISSING_LOG_PATH),
        help=(
            "Optional missing-data log path. Default: "
            "figures/slo-compare/data/slo-compare.missing.log"
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


def _extract_average_power_w(
    power_payload: dict[str, Any] | None,
    *,
    run_dir: Path,
    missing_log: list[str],
) -> tuple[float | None, str | None]:
    if power_payload is None:
        missing_log.append(f"[missing-file] {run_dir / 'post-processed/power/power-summary.json'}")
        return (None, None)

    power_stats = power_payload.get("power_stats_w")
    if isinstance(power_stats, dict):
        avg_power_w = _float_or_none(power_stats.get("avg"))
        if avg_power_w is not None:
            return (avg_power_w, "power_stats_w.avg")

    power_points = power_payload.get("power_points")
    if isinstance(power_points, list):
        values = [
            numeric
            for numeric in (
                _float_or_none(point.get("power_w"))
                for point in power_points
                if isinstance(point, dict)
            )
            if numeric is not None
        ]
        if values:
            return (sum(values) / len(values), "mean(power_points[].power_w)")

    missing_log.append(
        f"[missing-metric] average_power_w {run_dir / 'post-processed/power/power-summary.json'}"
    )
    return (None, None)


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


def _build_variant_record(
    variant: dict[str, str],
    *,
    missing_log: list[str],
) -> dict[str, Any]:
    base_path = Path(variant["base_path"]).expanduser().resolve()
    run_dir = _select_run_dir(base_path, missing_log)

    power_payload: dict[str, Any] | None = None
    throughput_payload: dict[str, Any] | None = None
    if run_dir is not None:
        power_path = run_dir / "post-processed" / "power" / "power-summary.json"
        throughput_path = (
            run_dir
            / "post-processed"
            / "agent-output-throughput"
            / "agent-output-throughput.json"
        )
        if power_path.is_file():
            loaded_power = _load_json(power_path)
            if isinstance(loaded_power, dict):
                power_payload = loaded_power
        if throughput_path.is_file():
            loaded_throughput = _load_json(throughput_path)
            if isinstance(loaded_throughput, dict):
                throughput_payload = loaded_throughput

    average_power_w, average_power_source = _extract_average_power_w(
        power_payload,
        run_dir=run_dir if run_dir is not None else base_path,
        missing_log=missing_log,
    )
    p5_output_throughput_tokens_per_s, throughput_source = (
        _extract_p5_output_throughput_tokens_per_s(
            throughput_payload,
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
            "average_power_w": _stable_round(average_power_w),
        },
        "metric_sources": {
            "p5_output_throughput_tokens_per_s": throughput_source,
            "average_power_w": average_power_source,
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
        "figure_name": "slo-compare",
        "created_at_utc": _utc_now_timestamp(),
        "dataset": "dabstep",
        "agent": "mini-swe-agent",
        "qps": 0.03,
        "metrics": [
            {
                "metric_key": "p5_output_throughput_tokens_per_s",
                "label": "p5 Output Throughput",
                "unit": "tokens/s",
            },
            {
                "metric_key": "average_power_w",
                "label": "Average Power",
                "unit": "W",
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
    if missing_log:
        missing_log_path.write_text("\n".join(missing_log) + "\n", encoding="utf-8")
    else:
        missing_log_path.write_text("", encoding="utf-8")

    print(f"[written] {output_path}")
    print(f"[written] {missing_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
