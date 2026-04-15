#!/usr/bin/env python3
"""Render the energy-context-latency figure set from materialized JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
DEFAULT_FORMAT = "pdf"
SUPPORTED_FORMATS = ("pdf", "png", "svg")
IMPLEMENTATION_COLORS = {
    "uncontrolled": "#1d4ed8",
    "fixed_freq": "#d97706",
    "steer": "#0f766e",
}


def _import_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import ticker
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to render this figure. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt, ticker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the requested energy-context-latency clustered bar charts "
            "from a materialized JSON payload."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_energy_context_latency.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. Default: "
            "figures/energy-context-latency/output/"
        ),
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        choices=SUPPORTED_FORMATS,
        help=f"Figure format. Default: {DEFAULT_FORMAT}",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=14.2,
        help="Figure width in inches (default: 14.2).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=4.8,
        help="Figure height in inches (default: 4.8).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for raster output (default: 220).",
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


def _default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR.resolve()


def _metric_output_name(input_path: Path, metric_key: str, figure_format: str) -> str:
    return f"{input_path.stem}.{metric_key}.{figure_format}"


def _metric_formatter(metric_unit: str, ticker: Any) -> Any:
    if metric_unit == "jobs/s":
        return ticker.FuncFormatter(lambda value, _: f"{value:.3f}")
    if metric_unit == "kJ":
        return ticker.FuncFormatter(lambda value, _: f"{value:.1f}")
    if metric_unit == "%":
        return ticker.FuncFormatter(lambda value, _: f"{value:.0f}%")
    return ticker.FuncFormatter(lambda value, _: f"{value:.1f}")


def _value_label(metric_unit: str, value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    if metric_unit == "jobs/s":
        return f"{value:.3f}"
    if metric_unit == "kJ":
        return f"{value:.1f}"
    if metric_unit == "%":
        return f"{value:.1f}%"
    return f"{value:.1f}"


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir()
    )

    payload = _load_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {input_path}")

    raw_metrics = payload.get("metrics")
    raw_experiments = payload.get("experiments")
    raw_implementations = payload.get("implementations")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError(f"Materialized payload has no metrics: {input_path}")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError(f"Materialized payload has no experiments: {input_path}")
    if not isinstance(raw_implementations, list) or not raw_implementations:
        raise ValueError(f"Materialized payload has no implementations: {input_path}")

    metrics: list[dict[str, Any]] = [
        metric
        for metric in raw_metrics
        if isinstance(metric, dict)
        and isinstance(metric.get("metric_key"), str)
        and isinstance(metric.get("panel_title"), str)
        and isinstance(metric.get("metric_unit"), str)
        and isinstance(metric.get("y_axis_label"), str)
    ]
    experiments: list[dict[str, Any]] = [
        experiment
        for experiment in raw_experiments
        if isinstance(experiment, dict) and isinstance(experiment.get("qps"), list)
    ]
    implementations: list[dict[str, Any]] = [
        implementation
        for implementation in raw_implementations
        if isinstance(implementation, dict)
        and isinstance(implementation.get("implementation_key"), str)
        and isinstance(implementation.get("implementation_label"), str)
    ]
    if not metrics or not experiments or not implementations:
        raise ValueError(f"Materialized payload is missing usable plotting metadata: {input_path}")

    plt, ticker = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.titlesize": 12.4,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    implementation_order = [
        str(implementation["implementation_key"]) for implementation in implementations
    ]
    implementation_labels = {
        str(implementation["implementation_key"]): str(implementation["implementation_label"])
        for implementation in implementations
    }
    missing_entry_count = (
        int(payload.get("missing_entry_count"))
        if isinstance(payload.get("missing_entry_count"), int)
        else 0
    )

    for metric in metrics:
        metric_key = str(metric["metric_key"])
        metric_title = str(metric["panel_title"])
        metric_unit = str(metric["metric_unit"])
        y_axis_label = str(metric["y_axis_label"])

        figure, axes = plt.subplots(
            1,
            len(experiments),
            figsize=(args.figure_width, args.figure_height),
            sharey=True,
            squeeze=False,
        )
        axes_list = list(axes[0])
        legend_handles: list[Any] = []

        for axis, experiment in zip(axes_list, experiments):
            raw_qps_entries = experiment.get("qps")
            if not isinstance(raw_qps_entries, list) or not raw_qps_entries:
                raise ValueError(f"Experiment is missing qps entries: {experiment}")

            qps_entries = [
                qps_entry for qps_entry in raw_qps_entries if isinstance(qps_entry, dict)
            ]
            x_positions = list(range(len(qps_entries)))
            total_bar_width = 0.72
            bar_width = total_bar_width / max(1, len(implementation_order))
            centered_offset = (len(implementation_order) - 1) / 2.0

            for implementation_index, implementation_key in enumerate(implementation_order):
                implementation_label = implementation_labels[implementation_key]
                color = IMPLEMENTATION_COLORS.get(implementation_key, "#475569")
                values: list[float] = []
                positions: list[float] = []
                for qps_index, qps_entry in enumerate(qps_entries):
                    raw_qps_implementations = qps_entry.get("implementations")
                    if not isinstance(raw_qps_implementations, list):
                        raise ValueError(f"QPS entry is missing implementations: {qps_entry}")
                    matched_entry = next(
                        (
                            candidate
                            for candidate in raw_qps_implementations
                            if isinstance(candidate, dict)
                            and candidate.get("implementation_key") == implementation_key
                        ),
                        None,
                    )
                    if not isinstance(matched_entry, dict):
                        raise ValueError(
                            f"QPS entry has no implementation {implementation_key!r}: {qps_entry}"
                        )
                    metric_values = matched_entry.get("metric_values")
                    if not isinstance(metric_values, dict):
                        raise ValueError(f"Implementation entry is missing metric_values: {matched_entry}")
                    value = _float_or_none(metric_values.get(metric_key))
                    if value is None:
                        raise ValueError(
                            f"Implementation entry has no numeric metric {metric_key!r}: {matched_entry}"
                        )
                    values.append(value)
                    positions.append(
                        qps_index + ((implementation_index - centered_offset) * bar_width)
                    )

                bars = axis.bar(
                    positions,
                    values,
                    width=bar_width * 0.92,
                    color=color,
                    linewidth=0.0,
                    label=implementation_label,
                )
                if len(legend_handles) < len(implementation_order) and len(bars) > 0:
                    legend_handles.append(bars[0])
                max_value = max(values) if values else 0.0
                label_offset = max_value * 0.025 if max_value > 0.0 else 0.04
                for bar, value in zip(bars, values):
                    axis.text(
                        bar.get_x() + (bar.get_width() / 2.0),
                        bar.get_height() + label_offset,
                        _value_label(metric_unit, value),
                        ha="center",
                        va="bottom",
                        fontsize=8.4,
                        color="#0f172a",
                    )

            qps_labels = [
                str(qps_entry.get("qps_label", qps_entry.get("qps_slug", "")))
                for qps_entry in qps_entries
            ]
            axis.set_title(str(experiment.get("subplot_title", "")), loc="left", fontweight="semibold")
            axis.set_xticks(x_positions, qps_labels)
            axis.set_xlabel("QPS")
            axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.55)
            axis.set_axisbelow(True)
            axis.yaxis.set_major_formatter(_metric_formatter(metric_unit, ticker))

        axes_list[0].set_ylabel(y_axis_label)
        all_values = [
            _float_or_none(implementation_entry.get("metric_values", {}).get(metric_key))
            for experiment in experiments
            for qps_entry in experiment.get("qps", [])
            if isinstance(qps_entry, dict)
            for implementation_entry in qps_entry.get("implementations", [])
            if isinstance(implementation_entry, dict)
        ]
        numeric_values = [value for value in all_values if value is not None]
        max_value = max(numeric_values) if numeric_values else 0.0
        y_limit = max_value * 1.22 if max_value > 0.0 else 1.0
        for axis in axes_list:
            axis.set_ylim(0.0, y_limit)

        figure.suptitle(metric_title, x=0.065, y=0.985, ha="left", fontweight="semibold")
        subtitle_parts = [
            "3 experiments x 3 QPS clusters x 3 implementations",
            "newest timestamp run per source",
        ]
        if missing_entry_count > 0:
            subtitle_parts.append(
                f"missing data plotted as 0 (see missing log, {missing_entry_count} entries)"
            )
        figure.text(
            0.065,
            0.94,
            " | ".join(subtitle_parts),
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        if legend_handles:
            figure.legend(
                legend_handles,
                [implementation_labels[key] for key in implementation_order],
                loc="upper right",
                bbox_to_anchor=(0.985, 0.985),
                frameon=False,
                ncol=len(implementation_order),
            )

        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
        output_path = output_dir / _metric_output_name(input_path, metric_key, args.format)
        figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"[written] {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
