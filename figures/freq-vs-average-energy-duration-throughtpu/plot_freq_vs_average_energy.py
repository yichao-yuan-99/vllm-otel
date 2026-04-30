#!/usr/bin/env python3
"""Plot frequency vs energy, power, throughput, and average LLM time."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to render this figure. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the freq-vs-average-energy-duration-throughtpu figure "
            "from a materialized CSV."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV from materialize_windowed_metrics.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: "
            "figures/freq-vs-average-energy-duration-throughtpu/output/<input-stem>.pdf"
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=9.0,
        help="Figure width in inches (default: 9.0).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=3.9,
        help="Figure height in inches (default: 3.9).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for raster outputs (default: 220).",
    )
    return parser.parse_args()


def _float_or_none(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    return rows


def _default_output_path(input_path: Path) -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{input_path.stem}.pdf").resolve()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path)
    )

    rows = _load_rows(input_path)
    parsed_rows: list[tuple[float, float, float | None, float | None, float | None]] = []
    for row in rows:
        frequency_mhz = _float_or_none(row.get("frequency_mhz", ""))
        throughput_jobs_per_s = _float_or_none(row.get("average_throughput_jobs_per_s", ""))
        energy_per_finished_j = _float_or_none(
            row.get("average_energy_per_finished_replay_j", "")
        )
        average_power_w = _float_or_none(row.get("window_avg_power_w", ""))
        average_request_time_in_llm_s = _float_or_none(
            row.get("average_request_time_in_llm_s", "")
        )
        if frequency_mhz is None or throughput_jobs_per_s is None:
            continue
        energy_per_finished_kj = (
            None if energy_per_finished_j is None else (energy_per_finished_j / 1000.0)
        )
        parsed_rows.append(
            (
                frequency_mhz,
                throughput_jobs_per_s,
                energy_per_finished_kj,
                average_request_time_in_llm_s,
                average_power_w,
            )
        )

    if not parsed_rows:
        raise ValueError(f"No plottable rows were found in {input_path}")

    parsed_rows.sort(key=lambda item: item[0])
    x_values = [item[0] for item in parsed_rows]
    throughput_values = [item[1] for item in parsed_rows]
    energy_values = [
        (float("nan") if item[2] is None else item[2])
        for item in parsed_rows
    ]
    llm_time_values = [
        (float("nan") if item[3] is None else item[3])
        for item in parsed_rows
    ]
    power_values = [
        (float("nan") if item[4] is None else item[4])
        for item in parsed_rows
    ]

    plt = _import_matplotlib_pyplot()
    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(args.figure_width, args.figure_height),
        gridspec_kw={"height_ratios": (1.0, 1.0)},
    )
    energy_axis = axes[0]
    throughput_axis = axes[1]
    power_axis = energy_axis.twinx()
    llm_axis = throughput_axis.twinx()

    energy_color = "#d97706"
    energy_label_color = "#b45309"
    power_color = "#b91c1c"
    throughput_color = "#2563eb"
    llm_color = "#0f766e"

    energy_axis.plot(
        x_values,
        energy_values,
        color=energy_color,
        linewidth=2.2,
        marker="o",
        markersize=6.5,
        label="Avg energy / job",
    )
    power_axis.plot(
        x_values,
        power_values,
        color=power_color,
        linewidth=2.0,
        linestyle="--",
        marker="D",
        markersize=5.6,
        label="Avg power",
    )
    throughput_axis.plot(
        x_values,
        throughput_values,
        color=throughput_color,
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=5.8,
        label="Avg throughput",
    )
    llm_axis.plot(
        x_values,
        llm_time_values,
        color=llm_color,
        linewidth=2.0,
        linestyle="-.",
        marker="^",
        markersize=5.6,
        label="Avg time / LLM request",
    )

    energy_axis.set_ylabel("Average energy\nper job (kJ)", color=energy_label_color)
    power_axis.set_ylabel("Average power (W)", color=power_color)
    throughput_axis.set_xlabel("Frequency (MHz)")
    throughput_axis.set_ylabel("Average throughput\n(jobs/s)", color=throughput_color)
    llm_axis.set_ylabel("Average time spent\nper LLM request (s)", color=llm_color)
    energy_axis.tick_params(axis="y", colors=energy_color)
    power_axis.tick_params(axis="y", colors=power_color)
    throughput_axis.tick_params(axis="y", colors=throughput_color)
    llm_axis.tick_params(axis="y", colors=llm_color)
    energy_axis.grid(True, axis="both", alpha=0.25, linestyle=":")
    throughput_axis.grid(True, axis="both", alpha=0.25, linestyle=":")
    energy_axis.set_axisbelow(True)
    throughput_axis.set_axisbelow(True)
    throughput_axis.set_xlim(max(x_values), min(x_values))
    if args.title:
        fig.suptitle(args.title, y=0.995)

    energy_handles, energy_labels = energy_axis.get_legend_handles_labels()
    power_handles, power_labels = power_axis.get_legend_handles_labels()
    throughput_handles, throughput_labels = throughput_axis.get_legend_handles_labels()
    llm_handles, llm_labels = llm_axis.get_legend_handles_labels()
    fig.legend(
        energy_handles + power_handles + throughput_handles + llm_handles,
        energy_labels + power_labels + throughput_labels + llm_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98 if not args.title else 0.94),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.6,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86 if not args.title else 0.80))
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
