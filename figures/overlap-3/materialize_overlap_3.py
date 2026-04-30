#!/usr/bin/env python3
"""Materialize the overlap-3 figure dataset."""

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
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "overlap-3.json"
DEFAULT_MISSING_LOG_PATH = DEFAULT_OUTPUT_DIR / "overlap-3.missing.log"
DEFAULT_BASELINE_COLOR = "#111111"
DEFAULT_COMPARISON_COLORS = ("#1D4F91", "#0F766E", "#B45309")

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
        "label": "KAIROS SLO 20",
        "base_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance/"
            "dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/20260412T014805Z"
        ),
    },
    {
        "variant_key": "freq_control_slo_35",
        "label": "KAIROS SLO 35",
        "base_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/"
            "dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/35"
        ),
    },
    {
        "variant_key": "freq_control_slo_45",
        "label": "KAIROS SLO 45",
        "base_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/"
            "dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/45"
        ),
    },
)

COMPARISONS = (
    {
        "comparison_key": "baseline_vs_slo_20",
        "baseline_variant_key": "no_freq_control",
        "variant_key": "freq_control_no_slo",
        "color": DEFAULT_COMPARISON_COLORS[0],
    },
    {
        "comparison_key": "baseline_vs_slo_35",
        "baseline_variant_key": "no_freq_control",
        "variant_key": "freq_control_slo_35",
        "color": DEFAULT_COMPARISON_COLORS[1],
    },
    {
        "comparison_key": "baseline_vs_slo_45",
        "baseline_variant_key": "no_freq_control",
        "variant_key": "freq_control_slo_45",
        "color": DEFAULT_COMPARISON_COLORS[2],
    },
)


def _utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the three-row overlap histogram figure dataset."
    )
    parser.add_argument(
        "--baseline-path",
        default=VARIANTS[0]["base_path"],
        help="Optional override for the no-frequency-control source path.",
    )
    parser.add_argument(
        "--slo-20-path",
        default=VARIANTS[1]["base_path"],
        help="Optional override for the KAIROS SLO 20 source path.",
    )
    parser.add_argument(
        "--slo-35-path",
        default=VARIANTS[2]["base_path"],
        help="Optional override for the KAIROS SLO 35 source path.",
    )
    parser.add_argument(
        "--slo-45-path",
        default=VARIANTS[3]["base_path"],
        help="Optional override for the KAIROS SLO 45 source path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Optional output JSON path. Default: figures/overlap-3/data/overlap-3.json",
    )
    parser.add_argument(
        "--missing-log",
        default=str(DEFAULT_MISSING_LOG_PATH),
        help=(
            "Optional missing-data log path. Default: "
            "figures/overlap-3/data/overlap-3.missing.log"
        ),
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=None,
        help="Optional shared histogram bin size in tokens/s. Default: infer from input metadata.",
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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _stable_round(value: float | None) -> float:
    return 0.0 if value is None else round(value, 6)


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


def _extract_throughput_values(
    payload: dict[str, Any] | None,
    *,
    run_dir: Path,
    missing_log: list[str],
) -> tuple[list[float], float | None]:
    if payload is None:
        missing_log.append(
            f"[missing-file] {run_dir / 'post-processed/agent-output-throughput/agent-output-throughput.json'}"
        )
        return ([], None)

    histogram = payload.get("agent_output_throughput_tokens_per_s_histogram")
    bin_size = None
    if isinstance(histogram, dict):
        candidate_bin_size = _float_or_none(histogram.get("bin_size"))
        if candidate_bin_size is not None and candidate_bin_size > 0.0:
            bin_size = candidate_bin_size

    raw_agents = payload.get("agents")
    if not isinstance(raw_agents, list):
        missing_log.append(
            "[missing-metric] agents "
            f"{run_dir / 'post-processed/agent-output-throughput/agent-output-throughput.json'}"
        )
        return ([], bin_size)

    values: list[float] = []
    for raw_agent in raw_agents:
        if not isinstance(raw_agent, dict):
            continue
        throughput = _float_or_none(raw_agent.get("output_throughput_tokens_per_s"))
        if throughput is None:
            output_tokens = _int_or_none(raw_agent.get("output_tokens"))
            llm_request_duration_s = _float_or_none(raw_agent.get("llm_request_duration_s"))
            if (
                output_tokens is not None
                and llm_request_duration_s is not None
                and llm_request_duration_s > 0.0
            ):
                throughput = output_tokens / llm_request_duration_s
        if throughput is not None:
            values.append(throughput)

    if not values:
        missing_log.append(
            "[missing-metric] output_throughput_tokens_per_s "
            f"{run_dir / 'post-processed/agent-output-throughput/agent-output-throughput.json'}"
        )
    return (values, bin_size)


def _resolve_bin_size(
    *,
    override_bin_size: float | None,
    inferred_sizes: list[float],
) -> float:
    if override_bin_size is not None:
        if override_bin_size <= 0.0:
            raise ValueError(f"--bin-size must be positive: {override_bin_size}")
        return override_bin_size

    unique_sizes = sorted(set(round(value, 9) for value in inferred_sizes if value > 0.0))
    if len(unique_sizes) > 1:
        raise ValueError(
            "Input variants advertise different histogram bin sizes. "
            "Pass --bin-size explicitly to choose a shared bin size."
        )
    if unique_sizes:
        return unique_sizes[0]
    return 1.0


def _summarize_values(values: list[float]) -> dict[str, float | int]:
    average_value = sum(values) / len(values)
    variance = sum((value - average_value) ** 2 for value in values) / len(values)
    return {
        "sample_count": len(values),
        "avg": round(average_value, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "std": round(math.sqrt(variance), 6),
    }


def _build_bins(
    *,
    values: list[float],
    bin_size: float,
    min_bin_index: int,
    max_bin_index: int,
) -> list[dict[str, float | int]]:
    counts_by_index = {
        bin_index: 0 for bin_index in range(min_bin_index, max_bin_index + 1)
    }
    for value in values:
        counts_by_index[math.floor(value / bin_size)] += 1
    return [
        {
            "bin_start": round(bin_index * bin_size, 6),
            "bin_end": round((bin_index + 1) * bin_size, 6),
            "count": counts_by_index[bin_index],
        }
        for bin_index in range(min_bin_index, max_bin_index + 1)
    ]


def _build_variant_record(
    variant: dict[str, str],
    *,
    missing_log: list[str],
) -> tuple[dict[str, Any], float | None]:
    base_path = Path(variant["base_path"]).expanduser().resolve()
    run_dir = _select_run_dir(base_path, missing_log)

    payload: dict[str, Any] | None = None
    if run_dir is not None:
        throughput_path = (
            run_dir
            / "post-processed"
            / "agent-output-throughput"
            / "agent-output-throughput.json"
        )
        if throughput_path.is_file():
            loaded_payload = _load_json(throughput_path)
            if isinstance(loaded_payload, dict):
                payload = loaded_payload

    values, inferred_bin_size = _extract_throughput_values(
        payload,
        run_dir=run_dir if run_dir is not None else base_path,
        missing_log=missing_log,
    )
    summary = _summarize_values(values) if values else {
        "sample_count": 0,
        "avg": 0.0,
        "min": 0.0,
        "max": 0.0,
        "std": 0.0,
    }
    return (
        {
            "variant_key": variant["variant_key"],
            "label": variant["label"],
            "base_path": str(base_path),
            "selected_run_dir": None if run_dir is None else str(run_dir),
            "summary": summary,
            "throughput_values": [_stable_round(value) for value in values],
        },
        inferred_bin_size,
    )


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    missing_log_path = Path(args.missing_log).expanduser().resolve()
    configured_paths = {
        "no_freq_control": args.baseline_path,
        "freq_control_no_slo": args.slo_20_path,
        "freq_control_slo_35": args.slo_35_path,
        "freq_control_slo_45": args.slo_45_path,
    }

    missing_log: list[str] = []
    variant_records: list[dict[str, Any]] = []
    inferred_bin_sizes: list[float] = []
    for raw_variant in VARIANTS:
        variant = dict(raw_variant)
        variant["base_path"] = configured_paths[variant["variant_key"]]
        record, inferred_bin_size = _build_variant_record(variant, missing_log=missing_log)
        variant_records.append(record)
        if inferred_bin_size is not None:
            inferred_bin_sizes.append(inferred_bin_size)

    all_values = [
        float(value)
        for record in variant_records
        for value in record["throughput_values"]
        if isinstance(value, (int, float))
    ]
    if not all_values:
        raise ValueError("No valid throughput values were found for any overlap-3 variant.")

    bin_size = _resolve_bin_size(
        override_bin_size=args.bin_size,
        inferred_sizes=inferred_bin_sizes,
    )
    min_bin_index = math.floor(min(all_values) / bin_size)
    max_bin_index = math.floor(max(all_values) / bin_size)

    variants_by_key = {record["variant_key"]: record for record in variant_records}
    for record in variant_records:
        record["histogram"] = {
            "bin_size": round(bin_size, 6),
            "bins": _build_bins(
                values=[float(value) for value in record["throughput_values"]],
                bin_size=bin_size,
                min_bin_index=min_bin_index,
                max_bin_index=max_bin_index,
            ),
        }

    comparisons: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        baseline = variants_by_key[comparison["baseline_variant_key"]]
        candidate = variants_by_key[comparison["variant_key"]]
        comparisons.append(
            {
                "comparison_key": comparison["comparison_key"],
                "baseline_variant_key": baseline["variant_key"],
                "variant_key": candidate["variant_key"],
                "baseline_label": baseline["label"],
                "label": candidate["label"],
                "baseline_color": DEFAULT_BASELINE_COLOR,
                "color": comparison["color"],
                "baseline_histogram": baseline["histogram"],
                "histogram": candidate["histogram"],
                "baseline_summary": baseline["summary"],
                "summary": candidate["summary"],
            }
        )

    payload = {
        "figure_name": "overlap-3",
        "created_at_utc": _utc_now_timestamp(),
        "dataset": "dabstep",
        "agent": "mini-swe-agent",
        "qps": 0.03,
        "metric_definition": {
            "name": "output_throughput_tokens_per_s",
            "formula": "output_tokens / llm_request_duration_s",
            "fallback_behavior": (
                "Use output_throughput_tokens_per_s directly when present; otherwise "
                "recompute it from output_tokens and llm_request_duration_s."
            ),
        },
        "selection_policy": {
            "run_resolution": (
                "If a configured path is already a run directory, use it directly. "
                "Otherwise, scan child directories in reverse lexical order and pick "
                "the newest directory containing replay/summary.json."
            ),
            "histogram_grid": "One shared bin grid across all four variants.",
        },
        "bin_size": round(bin_size, 6),
        "variant_count": len(variant_records),
        "comparison_count": len(comparisons),
        "variants": variant_records,
        "comparisons": comparisons,
        "missing_log_path": str(missing_log_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    missing_log_path.parent.mkdir(parents=True, exist_ok=True)
    missing_log_path.write_text(
        "\n".join(missing_log) + ("\n" if missing_log else ""),
        encoding="utf-8",
    )

    print(f"[written] {output_path}")
    print(f"[written] {missing_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
