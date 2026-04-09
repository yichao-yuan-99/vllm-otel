#!/usr/bin/env python3
"""Plot frequency vs energy, throughput, and average LLM time."""

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
        default="Frequency vs Energy, Throughput, and LLM Time",
        help="Figure title.",
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
        default=7.8,
        help="Figure height in inches (default: 7.8).",
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


def _shared_window_label(rows: list[dict[str, Any]]) -> str | None:
    starts = {row.get("analysis_window_start_s", "").strip() for row in rows}
    ends = {row.get("analysis_window_end_s", "").strip() for row in rows}
    if len(starts) == 1 and len(ends) == 1:
        start = next(iter(starts))
        end = next(iter(ends))
        if start and end:
            return f"window: {start}s to {end}s"
    return None


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path)
    )

    rows = _load_rows(input_path)
    parsed_rows: list[tuple[float, float, float | None, float | None]] = []
    for row in rows:
        frequency_mhz = _float_or_none(row.get("frequency_mhz", ""))
        throughput_jobs_per_s = _float_or_none(row.get("average_throughput_jobs_per_s", ""))
        energy_per_finished_j = _float_or_none(
            row.get("average_energy_per_finished_replay_j", "")
        )
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
    llm_axis = throughput_axis.twinx()

    energy_color = "#d97706"
    throughput_color = "#2563eb"
    llm_color = "#0f766e"

    energy_axis.plot(
        x_values,
        energy_values,
        color=energy_color,
        linewidth=2.2,
        marker="o",
        markersize=6.5,
        label="Avg energy / finished replay",
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
        label="Avg time in LLM",
    )

    energy_axis.set_ylabel("Average energy per finished replay (kJ)", color=energy_color)
    throughput_axis.set_xlabel("Frequency (MHz)")
    throughput_axis.set_ylabel("Average throughput (jobs/s)", color=throughput_color)
    llm_axis.set_ylabel("Average time spent in LLM (s)", color=llm_color)
    energy_axis.tick_params(axis="y", colors=energy_color)
    throughput_axis.tick_params(axis="y", colors=throughput_color)
    llm_axis.tick_params(axis="y", colors=llm_color)
    energy_axis.grid(True, axis="both", alpha=0.25, linestyle=":")
    throughput_axis.grid(True, axis="both", alpha=0.25, linestyle=":")
    energy_axis.set_axisbelow(True)
    throughput_axis.set_axisbelow(True)
    energy_axis.set_title("Energy", loc="left", fontsize=11.5)
    throughput_axis.set_title("Throughput and Average LLM Time", loc="left", fontsize=11.5)

    title_lines = [args.title]
    shared_window_label = _shared_window_label(rows)
    if shared_window_label is not None:
        title_lines.append(shared_window_label)
    fig.suptitle("\n".join(title_lines))

    energy_handles, energy_labels = energy_axis.get_legend_handles_labels()
    throughput_handles, throughput_labels = throughput_axis.get_legend_handles_labels()
    llm_handles, llm_labels = llm_axis.get_legend_handles_labels()
    energy_axis.legend(
        energy_handles,
        energy_labels,
        loc="upper left",
        frameon=False,
    )
    throughput_axis.legend(
        throughput_handles + llm_handles,
        throughput_labels + llm_labels,
        loc="upper left",
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
