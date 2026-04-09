#!/usr/bin/env python3
"""Plot frequency vs completed-agent integral energy and throughput."""

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
        description="Plot the integral freq-vs-average-energy figure from a materialized CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV from materialize_integral_metrics.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: "
            "figures/freq-vs-average-energy-integral/output/<input-stem>.pdf"
        ),
    )
    parser.add_argument(
        "--title",
        default="Frequency vs Completed-Agent Integral Energy and Throughput",
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
        default=5.8,
        help="Figure height in inches (default: 5.8).",
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
        rows = list(csv.DictReader(handle))
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
    parsed_rows: list[tuple[float, float, float | None]] = []
    for row in rows:
        frequency_mhz = _float_or_none(row.get("frequency_mhz", ""))
        throughput_jobs_per_s = _float_or_none(row.get("average_throughput_jobs_per_s", ""))
        energy_per_finished_j = _float_or_none(
            row.get("average_request_integral_energy_per_finished_replay_j", "")
        )
        if frequency_mhz is None or throughput_jobs_per_s is None:
            continue
        energy_per_finished_kj = (
            None if energy_per_finished_j is None else (energy_per_finished_j / 1000.0)
        )
        parsed_rows.append((frequency_mhz, throughput_jobs_per_s, energy_per_finished_kj))

    if not parsed_rows:
        raise ValueError(f"No plottable rows were found in {input_path}")

    parsed_rows.sort(key=lambda item: item[0])
    x_values = [item[0] for item in parsed_rows]
    throughput_values = [item[1] for item in parsed_rows]
    energy_values = [
        (float("nan") if item[2] is None else item[2])
        for item in parsed_rows
    ]

    plt = _import_matplotlib_pyplot()
    fig, left_axis = plt.subplots(figsize=(args.figure_width, args.figure_height))
    right_axis = left_axis.twinx()

    left_color = "#b45309"
    right_color = "#1d4ed8"

    left_axis.plot(
        x_values,
        energy_values,
        color=left_color,
        linewidth=2.2,
        marker="o",
        markersize=6.5,
        label="Avg completed-agent request energy",
    )
    right_axis.plot(
        x_values,
        throughput_values,
        color=right_color,
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=5.8,
        label="Avg throughput",
    )

    left_axis.set_xlabel("Frequency (MHz)")
    left_axis.set_ylabel(
        "Average completed-agent request energy (kJ)",
        color=left_color,
    )
    right_axis.set_ylabel("Average throughput (jobs/s)", color=right_color)
    left_axis.tick_params(axis="y", colors=left_color)
    right_axis.tick_params(axis="y", colors=right_color)
    left_axis.grid(True, axis="both", alpha=0.25, linestyle=":")
    left_axis.set_axisbelow(True)

    title_lines = [args.title]
    shared_window_label = _shared_window_label(rows)
    if shared_window_label is not None:
        title_lines.append(shared_window_label)
    left_axis.set_title("\n".join(title_lines))

    left_handles, left_labels = left_axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="upper left",
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
