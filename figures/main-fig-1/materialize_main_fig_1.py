#!/usr/bin/env python3
"""Materialize the combined main-fig-1 dataset from energy-context-latency JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_INPUT = (
    THIS_DIR.parent
    / "energy-context-latency"
    / "data"
    / "energy-context-latency.json"
).resolve()
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "main-fig-1"

PANEL_SPECS = (
    {
        "panel_key": "throughput_summary",
        "panel_title": "Throughput and SLO Attainment Rate (20tokens/s)",
        "panel_subtitle": "",
        "primary_metric_key": "p5_output_throughput_tokens_per_s",
        "secondary_metric_key": "pct_agents_above_20_output_throughput_tokens_per_s",
    },
    {
        "panel_key": "power_energy_summary",
        "panel_title": "Power and Energy",
        "panel_subtitle": "",
        "primary_metric_key": "average_power_w",
        "secondary_metric_key": "average_energy_per_finished_agent_kj",
    },
)
SELECTED_METRIC_KEYS = tuple(
    metric_key
    for panel in PANEL_SPECS
    for metric_key in (panel["primary_metric_key"], panel["secondary_metric_key"])
)
IMPLEMENTATION_COPY_KEYS = (
    "implementation_key",
    "implementation_label",
    "source_root",
    "run_dir",
    "run_dir_name",
    "candidate_run_count",
)
EXPERIMENT_COPY_KEYS = (
    "experiment_id",
    "dataset_slug",
    "dataset_label",
    "agent_slug",
    "agent_label",
    "subplot_title",
)
QPS_COPY_KEYS = ("qps_slug", "qps_value", "qps_label")


def _display_implementation_label(implementation_key: str, implementation_label: str) -> str:
    if implementation_key == "fixed_freq":
        return "Fixed Freq (810Mhz)"
    return implementation_label


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the main-fig-1 plotting payload from the upstream "
            "energy-context-latency JSON dataset."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_SOURCE_INPUT),
        help=(
            "Input JSON from figures/energy-context-latency/"
            f"(default: {DEFAULT_SOURCE_INPUT})."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path. Default: "
            "figures/main-fig-1/data/main-fig-1.json"
        ),
    )
    return parser.parse_args()


def _default_output_path() -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{DEFAULT_OUTPUT_STEM}.json").resolve()


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


def _metric_metadata(metric_payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "metric_key",
        "metric_label",
        "panel_title",
        "metric_unit",
        "y_axis_label",
        "formula",
    )
    return {
        key: metric_payload[key]
        for key in keys
        if key in metric_payload and isinstance(metric_payload[key], str)
    }


def _copy_filtered_metric_values(
    raw_metric_values: Any,
    *,
    experiment_id: str,
    qps_slug: str,
    implementation_key: str,
) -> dict[str, float]:
    if not isinstance(raw_metric_values, dict):
        raise ValueError(
            "Implementation entry is missing metric_values for "
            f"{experiment_id}/{qps_slug}/{implementation_key}"
        )

    filtered_values: dict[str, float] = {}
    for metric_key in SELECTED_METRIC_KEYS:
        metric_value = _float_or_none(raw_metric_values.get(metric_key))
        if metric_value is None:
            raise ValueError(
                "Implementation entry is missing numeric "
                f"{metric_key!r} for {experiment_id}/{qps_slug}/{implementation_key}"
            )
        filtered_values[metric_key] = metric_value
    return filtered_values


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path()
    )

    payload = _load_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {input_path}")

    raw_metrics = payload.get("metrics")
    raw_implementations = payload.get("implementations")
    raw_experiments = payload.get("experiments")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError(f"Input payload has no metrics: {input_path}")
    if not isinstance(raw_implementations, list) or not raw_implementations:
        raise ValueError(f"Input payload has no implementations: {input_path}")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError(f"Input payload has no experiments: {input_path}")

    metric_by_key = {
        str(metric["metric_key"]): metric
        for metric in raw_metrics
        if isinstance(metric, dict) and isinstance(metric.get("metric_key"), str)
    }
    missing_metric_keys = [
        metric_key for metric_key in SELECTED_METRIC_KEYS if metric_key not in metric_by_key
    ]
    if missing_metric_keys:
        raise ValueError(
            "Input payload is missing required metric definitions: "
            + ", ".join(missing_metric_keys)
        )

    implementations: list[dict[str, Any]] = []
    implementation_order: list[str] = []
    for implementation in raw_implementations:
        if not isinstance(implementation, dict):
            continue
        implementation_key = implementation.get("implementation_key")
        implementation_label = implementation.get("implementation_label")
        if not isinstance(implementation_key, str) or not isinstance(
            implementation_label, str
        ):
            continue
        display_label = _display_implementation_label(
            implementation_key, implementation_label
        )
        implementations.append(
            {
                key: (
                    display_label
                    if key == "implementation_label"
                    else implementation[key]
                )
                for key in ("implementation_key", "implementation_label", "source_root")
                if key in implementation
            }
        )
        implementation_order.append(implementation_key)
    if not implementations:
        raise ValueError(f"Input payload has no usable implementations: {input_path}")

    selected_metrics = [
        _metric_metadata(metric_by_key[metric_key]) for metric_key in SELECTED_METRIC_KEYS
    ]
    panels = []
    for panel_spec in PANEL_SPECS:
        primary_metric = metric_by_key[panel_spec["primary_metric_key"]]
        secondary_metric = metric_by_key[panel_spec["secondary_metric_key"]]
        panels.append(
            {
                "panel_key": panel_spec["panel_key"],
                "panel_title": panel_spec["panel_title"],
                "panel_subtitle": panel_spec["panel_subtitle"],
                "primary_metric_key": panel_spec["primary_metric_key"],
                "secondary_metric_key": panel_spec["secondary_metric_key"],
                "primary_metric_label": str(primary_metric["metric_label"]),
                "secondary_metric_label": str(secondary_metric["metric_label"]),
                "primary_metric_unit": str(primary_metric["metric_unit"]),
                "secondary_metric_unit": str(secondary_metric["metric_unit"]),
                "primary_y_axis_label": str(primary_metric["y_axis_label"]),
                "secondary_y_axis_label": str(secondary_metric["y_axis_label"]),
            }
        )

    experiments: list[dict[str, Any]] = []
    for experiment in raw_experiments:
        if not isinstance(experiment, dict):
            continue
        experiment_id = experiment.get("experiment_id")
        raw_qps_entries = experiment.get("qps")
        if not isinstance(experiment_id, str) or not isinstance(raw_qps_entries, list):
            continue

        filtered_qps_entries = []
        for qps_entry in raw_qps_entries:
            if not isinstance(qps_entry, dict):
                continue
            qps_slug = qps_entry.get("qps_slug")
            raw_implementation_entries = qps_entry.get("implementations")
            if not isinstance(qps_slug, str) or not isinstance(raw_implementation_entries, list):
                continue

            implementation_entries_by_key = {
                str(candidate.get("implementation_key")): candidate
                for candidate in raw_implementation_entries
                if isinstance(candidate, dict)
                and isinstance(candidate.get("implementation_key"), str)
            }

            filtered_implementation_entries = []
            for implementation_key in implementation_order:
                raw_implementation_entry = implementation_entries_by_key.get(implementation_key)
                if not isinstance(raw_implementation_entry, dict):
                    raise ValueError(
                        "QPS entry is missing implementation "
                        f"{implementation_key!r} for {experiment_id}/{qps_slug}"
                    )

                filtered_entry = {
                    key: (
                        _display_implementation_label(
                            implementation_key, str(raw_implementation_entry[key])
                        )
                        if key == "implementation_label"
                        else raw_implementation_entry[key]
                    )
                    for key in IMPLEMENTATION_COPY_KEYS
                    if key in raw_implementation_entry
                }
                filtered_entry["metric_values"] = _copy_filtered_metric_values(
                    raw_implementation_entry.get("metric_values"),
                    experiment_id=experiment_id,
                    qps_slug=qps_slug,
                    implementation_key=implementation_key,
                )
                raw_metric_details = raw_implementation_entry.get("metrics")
                if isinstance(raw_metric_details, dict):
                    filtered_entry["metrics"] = {
                        metric_key: raw_metric_details[metric_key]
                        for metric_key in SELECTED_METRIC_KEYS
                        if metric_key in raw_metric_details
                    }
                filtered_implementation_entries.append(filtered_entry)

            filtered_qps_entries.append(
                {
                    key: qps_entry[key] for key in QPS_COPY_KEYS if key in qps_entry
                }
                | {"implementations": filtered_implementation_entries}
            )

        if not filtered_qps_entries:
            continue

        experiments.append(
            {
                key: experiment[key] for key in EXPERIMENT_COPY_KEYS if key in experiment
            }
            | {"qps": filtered_qps_entries}
        )

    if not experiments:
        raise ValueError(f"Input payload has no usable experiments: {input_path}")

    source_missing_entry_count = (
        int(payload["missing_entry_count"])
        if isinstance(payload.get("missing_entry_count"), int)
        else 0
    )
    source_missing_log_path = input_path.with_suffix(".missing.log")

    output_payload = {
        "figure_name": DEFAULT_OUTPUT_STEM,
        "source_figure_name": str(payload.get("figure_name", "")),
        "source_input": str(input_path),
        "source_missing_entry_count": source_missing_entry_count,
        "source_missing_log": (
            str(source_missing_log_path) if source_missing_log_path.is_file() else None
        ),
        "experiment_count": len(experiments),
        "implementation_count": len(implementations),
        "metric_count": len(selected_metrics),
        "panel_count": len(panels),
        "implementations": implementations,
        "metrics": selected_metrics,
        "panels": panels,
        "experiments": experiments,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
