#!/usr/bin/env python3
"""Render the combined main-fig-1 comparison figure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
DEFAULT_OUTPUT_STEM = "main-fig-1"
IMPLEMENTATION_COLORS = {
    "uncontrolled": "#7fc97f",
    "fixed_freq": "#beaed4",
    "steer": "#fdc086",
}


def _import_matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import ticker
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to render this figure. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt, ticker, Patch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the combined main-fig-1 summary figure from materialized JSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_main_fig_1.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: figures/main-fig-1/output/main-fig-1.pdf",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=24.6,
        help="Figure width in inches (default: 24.6).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=3.08,
        help="Figure height in inches (default: 3.08).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for raster outputs (default: 220).",
    )
    return parser.parse_args()


def _default_output_path() -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{DEFAULT_OUTPUT_STEM}.pdf").resolve()


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


def _metric_formatter(metric_unit: str, ticker: Any) -> Any:
    if metric_unit == "jobs/s":
        return ticker.FuncFormatter(lambda value, _: f"{value:.3f}")
    if metric_unit == "kJ":
        return ticker.FuncFormatter(lambda value, _: f"{value:.1f}")
    if metric_unit == "%":
        return ticker.FuncFormatter(lambda value, _: f"{value:.0f}%")
    return ticker.FuncFormatter(lambda value, _: f"{value:.1f}")


def _metric_upper_limit(metric_unit: str, values: list[float]) -> float:
    if not values:
        return 1.0
    max_value = max(values)
    if max_value <= 0.0:
        return 1.0
    if metric_unit == "%":
        return max_value * 1.06
    return max_value * 1.18


def _metric_values_for(payload: dict[str, Any], metric_key: str) -> list[float]:
    values: list[float] = []
    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list):
        return values
    for experiment in raw_experiments:
        if not isinstance(experiment, dict):
            continue
        raw_qps_entries = experiment.get("qps")
        if not isinstance(raw_qps_entries, list):
            continue
        for qps_entry in raw_qps_entries:
            if not isinstance(qps_entry, dict):
                continue
            raw_implementations = qps_entry.get("implementations")
            if not isinstance(raw_implementations, list):
                continue
            for implementation in raw_implementations:
                if not isinstance(implementation, dict):
                    continue
                raw_metric_values = implementation.get("metric_values")
                if not isinstance(raw_metric_values, dict):
                    continue
                value = _float_or_none(raw_metric_values.get(metric_key))
                if value is not None:
                    values.append(value)
    return values


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

    raw_panels = payload.get("panels")
    raw_experiments = payload.get("experiments")
    raw_implementations = payload.get("implementations")
    if not isinstance(raw_panels, list) or len(raw_panels) != 2:
        raise ValueError(f"Expected exactly two panels in {input_path}")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError(f"Materialized payload has no experiments: {input_path}")
    if not isinstance(raw_implementations, list) or not raw_implementations:
        raise ValueError(f"Materialized payload has no implementations: {input_path}")

    panels = [panel for panel in raw_panels if isinstance(panel, dict)]
    experiments = [experiment for experiment in raw_experiments if isinstance(experiment, dict)]
    implementations = [
        implementation
        for implementation in raw_implementations
        if isinstance(implementation, dict)
        and isinstance(implementation.get("implementation_key"), str)
        and isinstance(implementation.get("implementation_label"), str)
    ]
    if len(panels) != 2 or not experiments or not implementations:
        raise ValueError(f"Materialized payload is missing usable plotting metadata: {input_path}")

    implementation_order = [
        str(implementation["implementation_key"]) for implementation in implementations
    ]
    implementation_labels = {
        str(implementation["implementation_key"]): str(implementation["implementation_label"])
        for implementation in implementations
    }

    plt, ticker, Patch = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 13.9,
            "axes.titlesize": 15.6,
            "axes.labelsize": 13.7,
            "xtick.labelsize": 12.2,
            "ytick.labelsize": 12.2,
            "axes.spines.top": False,
        }
    )

    figure = plt.figure(figsize=(args.figure_width, args.figure_height))
    experiment_count = len(experiments)
    spacer_index = experiment_count
    width_ratios = ([1.0] * experiment_count) + [0.5] + ([1.0] * experiment_count)
    grid = figure.add_gridspec(
        1,
        (experiment_count * 2) + 1,
        width_ratios=width_ratios,
        wspace=0.16,
    )

    grouped_axes: list[list[Any]] = []
    for panel_index in range(len(panels)):
        start_col = 0 if panel_index == 0 else (spacer_index + 1)
        shared_axis = None
        group_axes = []
        for experiment_offset in range(experiment_count):
            subplot = figure.add_subplot(
                grid[0, start_col + experiment_offset],
                sharey=shared_axis,
            )
            if shared_axis is None:
                shared_axis = subplot
            group_axes.append(subplot)
        grouped_axes.append(group_axes)

    for panel, axes_group in zip(panels, grouped_axes):
        primary_metric_key = str(panel.get("primary_metric_key", ""))
        secondary_metric_key = str(panel.get("secondary_metric_key", ""))
        primary_metric_unit = str(panel.get("primary_metric_unit", ""))
        secondary_metric_unit = str(panel.get("secondary_metric_unit", ""))
        primary_y_axis_label = str(panel.get("primary_y_axis_label", ""))
        secondary_y_axis_label = str(panel.get("secondary_y_axis_label", ""))

        primary_y_limit = _metric_upper_limit(
            primary_metric_unit, _metric_values_for(payload, primary_metric_key)
        )
        secondary_y_limit = _metric_upper_limit(
            secondary_metric_unit, _metric_values_for(payload, secondary_metric_key)
        )

        for experiment_index, (axis, experiment) in enumerate(zip(axes_group, experiments)):
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
            secondary_axis = axis.twinx()
            secondary_axis.spines["right"].set_visible(True)
            secondary_groups: list[list[tuple[float, float, str]]] = [
                [] for _ in qps_entries
            ]

            for implementation_index, implementation_key in enumerate(implementation_order):
                color = IMPLEMENTATION_COLORS.get(implementation_key, "#475569")
                primary_values: list[float] = []
                secondary_values: list[float] = []
                shifted_positions: list[float] = []

                for qps_index, qps_entry in enumerate(qps_entries):
                    raw_implementation_entries = qps_entry.get("implementations")
                    if not isinstance(raw_implementation_entries, list):
                        raise ValueError(f"QPS entry is missing implementations: {qps_entry}")
                    matched_entry = next(
                        (
                            candidate
                            for candidate in raw_implementation_entries
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
                        raise ValueError(
                            f"Implementation entry is missing metric_values: {matched_entry}"
                        )
                    primary_value = _float_or_none(metric_values.get(primary_metric_key))
                    secondary_value = _float_or_none(metric_values.get(secondary_metric_key))
                    if primary_value is None or secondary_value is None:
                        raise ValueError(
                            "Implementation entry is missing required numeric values for "
                            f"{primary_metric_key!r} or {secondary_metric_key!r}: {matched_entry}"
                        )
                    shifted_position = (
                        qps_index + ((implementation_index - centered_offset) * bar_width)
                    )
                    shifted_positions.append(shifted_position)
                    primary_values.append(primary_value)
                    secondary_values.append(secondary_value)

                axis.bar(
                    shifted_positions,
                    primary_values,
                    width=bar_width * 0.92,
                    color=color,
                    alpha=0.86,
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=2,
                )
                for qps_index, (position, value) in enumerate(
                    zip(shifted_positions, secondary_values)
                ):
                    secondary_groups[qps_index].append((position, value, color))

            for secondary_group in secondary_groups:
                if not secondary_group:
                    continue
                secondary_group.sort(key=lambda item: item[0])
                secondary_positions = [item[0] for item in secondary_group]
                secondary_values = [item[1] for item in secondary_group]
                secondary_colors = [item[2] for item in secondary_group]
                secondary_axis.plot(
                    secondary_positions,
                    secondary_values,
                    color="#64748b",
                    linewidth=1.8,
                    linestyle=":",
                    zorder=3,
                )
                secondary_axis.scatter(
                    secondary_positions,
                    secondary_values,
                    s=28,
                    marker="o",
                    c=secondary_colors,
                    edgecolors="black",
                    linewidths=0.95,
                    zorder=4,
                )

            qps_labels = [
                str(qps_entry.get("qps_label", qps_entry.get("qps_slug", "")))
                for qps_entry in qps_entries
            ]
            axis.set_xticks(x_positions, qps_labels)
            axis.set_xlabel("Input Jobs/s")
            axis.set_ylim(0.0, primary_y_limit)
            secondary_axis.set_ylim(0.0, secondary_y_limit)
            axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
            axis.set_axisbelow(True)
            axis.yaxis.set_major_formatter(_metric_formatter(primary_metric_unit, ticker))
            secondary_axis.yaxis.set_major_formatter(
                _metric_formatter(secondary_metric_unit, ticker)
            )
            axis.tick_params(
                axis="y",
                labelleft=(experiment_index == 0),
                left=(experiment_index == 0),
            )
            secondary_axis.tick_params(
                axis="y",
                labelright=(experiment_index == (len(axes_group) - 1)),
                right=(experiment_index == (len(axes_group) - 1)),
            )

            if experiment_index == 0:
                axis.set_ylabel(primary_y_axis_label)
                if primary_y_axis_label == "Per Agent P5 Throughput (tokens/s)":
                    axis.yaxis.get_label().set_y(0.46)
            if experiment_index == (len(axes_group) - 1):
                secondary_axis.set_ylabel(secondary_y_axis_label)

    legend_handles = [
        Patch(
            facecolor=IMPLEMENTATION_COLORS.get(implementation_key, "#475569"),
            edgecolor="none",
            label=implementation_labels[implementation_key],
        )
        for implementation_key in implementation_order
    ]
    figure.legend(
        legend_handles,
        [implementation_labels[key] for key in implementation_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.085),
        frameon=False,
        ncol=len(implementation_order),
    )

    layout_rect = (0.0, 0.31, 1.0, 0.90)
    source_missing_entry_count = (
        int(payload.get("source_missing_entry_count"))
        if isinstance(payload.get("source_missing_entry_count"), int)
        else 0
    )
    if source_missing_entry_count > 0:
        layout_rect = (0.0, 0.34, 1.0, 0.90)

    figure.tight_layout(rect=layout_rect)

    for axes_group, experiment_group in zip(grouped_axes, [experiments] * len(grouped_axes)):
        for axis, experiment in zip(axes_group, experiment_group):
            dataset_label = str(
                experiment.get("dataset_label", experiment.get("dataset_slug", ""))
            )
            agent_label = str(experiment.get("agent_label", experiment.get("agent_slug", "")))
            axis_box = axis.get_position()
            center_x = (axis_box.x0 + axis_box.x1) / 2.0
            figure.text(
                center_x,
                axis_box.y0 - 0.19,
                f"{dataset_label}\n{agent_label}",
                ha="center",
                va="top",
                fontsize=13.0,
                fontweight="semibold",
            )

    if source_missing_entry_count > 0:
        source_missing_log = payload.get("source_missing_log")
        message = (
            f"Upstream missing data was zero-filled in energy-context-latency "
            f"({source_missing_entry_count} entries)"
        )
        if isinstance(source_missing_log, str) and source_missing_log:
            message += f"; see {Path(source_missing_log).name}"
        figure.text(
            0.5,
            0.018,
            message,
            ha="center",
            va="bottom",
            fontsize=11.0,
            color="#475569",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
