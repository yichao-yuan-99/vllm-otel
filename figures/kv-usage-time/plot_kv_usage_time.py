#!/usr/bin/env python3
"""Plot selected KV-cache-usage time series on one figure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
SERIES_COLORS = ("#1d4ed8", "#d97706", "#0f766e", "#be123c", "#7c3aed")


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
        description="Plot the kv-usage-time figure from a materialized JSON dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_kv_usage_time.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: figures/kv-usage-time/output/<input-stem>.pdf",
    )
    parser.add_argument(
        "--title",
        default="KV Cache Usage Over Time (Smoothed)",
        help="Figure title.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=10.8,
        help="Figure width in inches (default: 10.8).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=6.2,
        help="Figure height in inches (default: 6.2).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for raster outputs (default: 220).",
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

    payload = _load_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {input_path}")
    raw_series = payload.get("series")
    if not isinstance(raw_series, list) or not raw_series:
        raise ValueError(f"Materialized payload has no series: {input_path}")

    plt, ticker = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axis = plt.subplots(figsize=(args.figure_width, args.figure_height))

    stats_lines: list[str] = []
    plotted_series_count = 0
    for index, raw_item in enumerate(raw_series):
        if not isinstance(raw_item, dict):
            continue
        raw_points = raw_item.get("points")
        if not isinstance(raw_points, list):
            continue

        x_values_minutes: list[float] = []
        y_values_percent: list[float] = []
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            time_s = _float_or_none(point.get("time_from_start_s"))
            value = _float_or_none(point.get("value"))
            if time_s is None or value is None:
                continue
            x_values_minutes.append(time_s / 60.0)
            y_values_percent.append(value * 100.0)
        if not x_values_minutes:
            continue

        label = raw_item.get("series_label")
        if not isinstance(label, str) or not label:
            label = raw_item.get("run_slug") if isinstance(raw_item.get("run_slug"), str) else f"series-{index+1}"

        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        axis.plot(
            x_values_minutes,
            y_values_percent,
            color=color,
            linewidth=2.2,
            label=label,
        )
        plotted_series_count += 1

        stats = raw_item.get("stats")
        if isinstance(stats, dict):
            avg = _float_or_none(stats.get("avg"))
            max_value = _float_or_none(stats.get("max"))
            if avg is not None and max_value is not None:
                stats_lines.append(
                    f"{label}: avg {avg * 100.0:.1f}%, max {max_value * 100.0:.1f}%"
                )

    if plotted_series_count == 0:
        raise ValueError(f"No plottable series were found in {input_path}")

    axis.set_title(args.title, loc="left", fontweight="semibold")
    subtitle_parts: list[str] = []
    analysis_start_s = _float_or_none(payload.get("analysis_window_start_s"))
    analysis_end_s = _float_or_none(payload.get("analysis_window_end_s"))
    smooth_window_s = _float_or_none(payload.get("smooth_window_s"))
    metric_name = payload.get("metric_name")
    if analysis_start_s is not None and analysis_end_s is not None:
        subtitle_parts.append(
            f"window: {analysis_start_s:g}s to {analysis_end_s:g}s"
        )
    if isinstance(metric_name, str) and metric_name:
        subtitle_parts.append(f"metric: {metric_name}")
    if smooth_window_s is not None:
        subtitle_parts.append(f"smoothed window: +/- {smooth_window_s:g}s")
    subtitle_parts.append("y-axis shown as percent where 100% = full cache")
    axis.text(
        0.0,
        1.02,
        " | ".join(subtitle_parts),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#334155",
    )

    axis.set_xlabel("Time From Start (minutes)")
    axis.set_ylabel("KV Cache Usage (%)")
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
    axis.minorticks_on()
    axis.margins(x=0.0)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    axis.legend(loc="upper left", frameon=False)

    if stats_lines:
        axis.text(
            0.99,
            0.98,
            "\n".join(stats_lines),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "#F8FAFC",
                "edgecolor": "#CBD5E1",
                "alpha": 0.96,
            },
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
