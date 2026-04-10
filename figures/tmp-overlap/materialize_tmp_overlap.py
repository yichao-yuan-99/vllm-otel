#!/usr/bin/env python3
"""Materialize a shared-bin overlay dataset for two throughput histograms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "tmp-overlap"
DEFAULT_SOURCE_A = (
    "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
    "sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/"
    "exclude-unranked/qps0_08/20260321T143624Z/post-processed/visualization/"
    "agent-output-throughput/agent-output-throughput-histogram.png"
)
DEFAULT_SOURCE_B = (
    "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
    "sweep-qps-docker-power-clean-freq-ctrl-linespace/swebench-verified/"
    "mini-swe-agent/split/exclude-unranked/qps0_08/20260403T004347Z/"
    "post-processed/visualization/agent-output-throughput/"
    "agent-output-throughput-histogram.png"
)
DEFAULT_COLORS = ("#1D4F91", "#D97706")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a figure-specific dataset for an overlaid comparison of "
            "two agent-output-throughput histograms."
        )
    )
    parser.add_argument(
        "--source-a",
        default=DEFAULT_SOURCE_A,
        help=(
            "First input source. May be a run directory, an "
            "agent-output-throughput.json path, or the rendered histogram PNG."
        ),
    )
    parser.add_argument(
        "--source-b",
        default=DEFAULT_SOURCE_B,
        help=(
            "Second input source. May be a run directory, an "
            "agent-output-throughput.json path, or the rendered histogram PNG."
        ),
    )
    parser.add_argument(
        "--label-a",
        default=None,
        help="Optional legend label for the first source.",
    )
    parser.add_argument(
        "--label-b",
        default=None,
        help="Optional legend label for the second source.",
    )
    parser.add_argument(
        "--color-a",
        default=DEFAULT_COLORS[0],
        help=f"Bar color for the first source (default: {DEFAULT_COLORS[0]}).",
    )
    parser.add_argument(
        "--color-b",
        default=DEFAULT_COLORS[1],
        help=f"Bar color for the second source (default: {DEFAULT_COLORS[1]}).",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=None,
        help=(
            "Optional shared histogram bin size in tokens/s. "
            "Default: infer from source histogram metadata."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path. Default: "
            "figures/tmp-overlap/data/tmp-overlap.<label-a>-vs-<label-b>.json"
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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower())
    normalized = normalized.strip("-")
    return normalized or DEFAULT_OUTPUT_STEM


def _default_output_path(label_a: str, label_b: str) -> Path:
    return (
        DEFAULT_OUTPUT_DIR
        / f"{DEFAULT_OUTPUT_STEM}.{_slugify(label_a)}-vs-{_slugify(label_b)}.json"
    ).resolve()


def _resolve_agent_output_json(source: str) -> Path:
    source_path = Path(source).expanduser().resolve()

    if source_path.is_dir():
        if (
            source_path.name == "agent-output-throughput"
            and source_path.parent.name == "visualization"
        ):
            candidate = (
                source_path.parent.parent
                / "agent-output-throughput"
                / "agent-output-throughput.json"
            ).resolve()
            if candidate.is_file():
                return candidate
        candidate = (
            source_path
            / "post-processed"
            / "agent-output-throughput"
            / "agent-output-throughput.json"
        ).resolve()
        if candidate.is_file():
            return candidate
        raise ValueError(
            "Could not resolve an agent-output-throughput.json file from directory: "
            f"{source_path}"
        )

    if source_path.is_file() and source_path.name == "agent-output-throughput.json":
        return source_path

    if (
        source_path.is_file()
        and source_path.parent.name == "agent-output-throughput"
        and source_path.parent.parent.name == "visualization"
    ):
        candidate = (
            source_path.parent.parent.parent
            / "agent-output-throughput"
            / "agent-output-throughput.json"
        ).resolve()
        if candidate.is_file():
            return candidate
        raise ValueError(
            "Could not find the sibling agent-output-throughput.json for figure path: "
            f"{source_path}"
        )

    raise ValueError(
        "Each source must be a run directory, an agent-output-throughput.json file, "
        f"or a visualization histogram image path: {source_path}"
    )


def _infer_label(source_spec: str, resolved_input_path: Path) -> str:
    for candidate in (Path(source_spec).expanduser().resolve(), resolved_input_path):
        parts = candidate.parts
        for index, part in enumerate(parts[:-1]):
            if part == "replay" and index + 1 < len(parts):
                replay_slug = parts[index + 1]
                if replay_slug:
                    return replay_slug.removeprefix("sweep-qps-")

    if (
        resolved_input_path.parent.name == "agent-output-throughput"
        and resolved_input_path.parent.parent.name == "post-processed"
    ):
        return resolved_input_path.parent.parent.parent.name
    return resolved_input_path.stem


def _source_run_dir(payload: dict[str, Any], input_path: Path) -> Path:
    source_run_dir = _string_or_none(payload.get("source_run_dir"))
    if source_run_dir is not None:
        return Path(source_run_dir).expanduser().resolve()
    if input_path.parent.name == "agent-output-throughput" and input_path.parent.parent.name == "post-processed":
        return input_path.parent.parent.parent.resolve()
    return input_path.parent.resolve()


def _extract_throughput_values(payload: dict[str, Any], input_path: Path) -> list[float]:
    raw_agents = payload.get("agents")
    if not isinstance(raw_agents, list):
        raise ValueError(f"Missing agents list in {input_path}")

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
        if throughput is None:
            continue
        values.append(throughput)

    if not values:
        raise ValueError(f"No valid per-agent throughput values found in {input_path}")
    return values


def _extract_histogram_bin_size(payload: dict[str, Any]) -> float | None:
    histogram = payload.get("agent_output_throughput_tokens_per_s_histogram")
    if not isinstance(histogram, dict):
        return None
    bin_size = _float_or_none(histogram.get("bin_size"))
    if bin_size is None or bin_size <= 0.0:
        return None
    return bin_size


def _resolve_bin_size(
    *,
    override_bin_size: float | None,
    payloads: list[dict[str, Any]],
) -> float:
    if override_bin_size is not None:
        if override_bin_size <= 0.0:
            raise ValueError(f"--bin-size must be positive: {override_bin_size}")
        return override_bin_size

    inferred_sizes = []
    for payload in payloads:
        inferred = _extract_histogram_bin_size(payload)
        if inferred is not None:
            inferred_sizes.append(round(inferred, 9))

    unique_sizes = sorted(set(inferred_sizes))
    if len(unique_sizes) > 1:
        raise ValueError(
            "Input sources advertise different histogram bin sizes. "
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


def _build_histogram(
    values: list[float],
    *,
    bin_size: float,
    domain_min: float,
    domain_max: float,
) -> dict[str, Any]:
    min_bin_index = math.floor(domain_min / bin_size)
    max_bin_index = math.floor(domain_max / bin_size)
    counts_by_index = {
        bin_index: 0 for bin_index in range(min_bin_index, max_bin_index + 1)
    }
    for value in values:
        counts_by_index[math.floor(value / bin_size)] += 1

    bins = [
        {
            "bin_start": round(bin_index * bin_size, 6),
            "bin_end": round((bin_index + 1) * bin_size, 6),
            "count": counts_by_index[bin_index],
        }
        for bin_index in range(min_bin_index, max_bin_index + 1)
    ]
    return {
        "metric": "output_throughput_tokens_per_s",
        "bin_size": round(bin_size, 6),
        "sample_count": len(values),
        "bin_count": len(bins),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "bins": bins,
    }


def _build_shared_histogram(
    *,
    bin_size: float,
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    if not datasets:
        raise ValueError("At least one dataset is required")
    first_bins = datasets[0]["histogram"]["bins"]
    bins = []
    for bin_index, first_bin in enumerate(first_bins):
        counts_by_dataset: dict[str, int] = {}
        for dataset in datasets:
            dataset_bin = dataset["histogram"]["bins"][bin_index]
            counts_by_dataset[dataset["dataset_id"]] = int(dataset_bin["count"])
        bins.append(
            {
                "bin_start": first_bin["bin_start"],
                "bin_end": first_bin["bin_end"],
                "counts_by_dataset": counts_by_dataset,
            }
        )

    max_count = 0
    for dataset in datasets:
        max_count = max(
            max_count,
            max(int(item["count"]) for item in dataset["histogram"]["bins"]),
        )
    return {
        "metric": "output_throughput_tokens_per_s",
        "bin_size": round(bin_size, 6),
        "bin_count": len(bins),
        "max_count": max_count,
        "bins": bins,
    }


def main() -> int:
    args = _parse_args()

    resolved_inputs = [
        _resolve_agent_output_json(args.source_a),
        _resolve_agent_output_json(args.source_b),
    ]
    payloads = [_load_json(path) for path in resolved_inputs]
    if not all(isinstance(payload, dict) for payload in payloads):
        raise ValueError("Each resolved input file must contain a JSON object")

    payload_dicts = [dict(payload) for payload in payloads]
    bin_size = _resolve_bin_size(
        override_bin_size=args.bin_size,
        payloads=payload_dicts,
    )

    throughput_values = [
        _extract_throughput_values(payload, input_path)
        for payload, input_path in zip(payload_dicts, resolved_inputs)
    ]
    domain_min = min(min(values) for values in throughput_values)
    domain_max = max(max(values) for values in throughput_values)

    labels = [
        args.label_a or _infer_label(args.source_a, resolved_inputs[0]),
        args.label_b or _infer_label(args.source_b, resolved_inputs[1]),
    ]
    colors = [args.color_a, args.color_b]

    datasets = []
    for index, (label, color, source_spec, input_path, payload, values) in enumerate(
        zip(
            labels,
            colors,
            (args.source_a, args.source_b),
            resolved_inputs,
            payload_dicts,
            throughput_values,
        ),
        start=1,
    ):
        datasets.append(
            {
                "dataset_id": f"dataset_{index}",
                "label": label,
                "color": color,
                "source_spec": source_spec,
                "resolved_input_path": str(input_path),
                "source_run_dir": str(_source_run_dir(payload, input_path)),
                "source_gateway_output_dir": _string_or_none(
                    payload.get("source_gateway_output_dir")
                ),
                "agent_count": _int_or_none(payload.get("agent_count")),
                "aggregate_run_output_throughput_tokens_per_s": _float_or_none(
                    payload.get("output_throughput_tokens_per_s")
                ),
                "summary": _summarize_values(values),
                "histogram": _build_histogram(
                    values,
                    bin_size=bin_size,
                    domain_min=domain_min,
                    domain_max=domain_max,
                ),
            }
        )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(labels[0], labels[1])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "figure_name": DEFAULT_OUTPUT_STEM,
        "figure_title": "Overlaid Agent Output Throughput Histogram",
        "dataset_count": len(datasets),
        "metric_definition": {
            "metric_key": "output_throughput_tokens_per_s",
            "per_agent_formula": "output_tokens / llm_request_duration_s",
            "fallback_rule": (
                "use agents[].output_throughput_tokens_per_s when present, "
                "otherwise recompute from output_tokens and llm_request_duration_s"
            ),
            "histogram_bin_formula": (
                "count(values where bin_start <= value < bin_end)"
            ),
            "y_axis_metric": "agent_count_per_bin",
        },
        "selection_policy": {
            "source_a": args.source_a,
            "source_b": args.source_b,
            "input_resolution": (
                "each source may be a run directory, an "
                "agent-output-throughput.json path, or a visualization histogram "
                "PNG path that resolves to the sibling agent-output-throughput.json"
            ),
        },
        "bin_size": round(bin_size, 6),
        "shared_histogram": _build_shared_histogram(
            bin_size=bin_size,
            datasets=datasets,
        ),
        "datasets": datasets,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
