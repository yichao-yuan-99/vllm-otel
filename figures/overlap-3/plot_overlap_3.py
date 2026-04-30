#!/usr/bin/env python3
"""Render the overlap-3 figure from a materialized JSON dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the overlap-3 figure from materialize_overlap_3.py output."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_overlap_3.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: figures/overlap-3/output/<input-stem>.pdf",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=8.0,
        help="Figure width in inches (default: 8.0).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=8.4,
        help="Figure height in inches (default: 8.4).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Bar alpha for overlay fills (default: 0.4).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for raster outputs (default: 220).",
    )
    return parser.parse_args()


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


def _default_output_path(input_path: Path) -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{input_path.stem}.pdf").resolve()


def _validated_bins(raw_bins: Any) -> list[dict[str, float | int]]:
    if not isinstance(raw_bins, list) or not raw_bins:
        raise ValueError("Expected a non-empty histogram bins list")

    bins: list[dict[str, float | int]] = []
    for raw_bin in raw_bins:
        if not isinstance(raw_bin, dict):
            raise ValueError("Histogram bins must be JSON objects")
        bin_start = _float_or_none(raw_bin.get("bin_start"))
        bin_end = _float_or_none(raw_bin.get("bin_end"))
        count = raw_bin.get("count")
        if bin_start is None or bin_end is None or not isinstance(count, int):
            raise ValueError("Histogram bins must contain numeric bin_start, bin_end, and int count")
        bins.append(
            {
                "bin_start": bin_start,
                "bin_end": bin_end,
                "count": count,
            }
        )
    return bins


def _add_panel_labels(axes: list[Any] | tuple[Any, ...]) -> None:
    for index, axis in enumerate(axes):
        axis.text(
            0.98,
            1.03,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=13,
        )


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError(f"--alpha must be in (0, 1]: {args.alpha}")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be positive: {args.dpi}")

    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path)
    )

    payload = _load_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {input_path}")
    raw_comparisons = payload.get("comparisons")
    if not isinstance(raw_comparisons, list) or not raw_comparisons:
        raise ValueError(f"Materialized payload has no comparisons: {input_path}")

    comparisons: list[dict[str, Any]] = []
    for raw_comparison in raw_comparisons:
        if not isinstance(raw_comparison, dict):
            continue
        baseline_histogram = raw_comparison.get("baseline_histogram")
        histogram = raw_comparison.get("histogram")
        if not isinstance(baseline_histogram, dict) or not isinstance(histogram, dict):
            continue
        comparisons.append(
            {
                "baseline_label": str(raw_comparison.get("baseline_label", "Baseline")),
                "label": str(raw_comparison.get("label", "Variant")),
                "baseline_color": str(raw_comparison.get("baseline_color", "#111111")),
                "color": str(raw_comparison.get("color", "#1D4F91")),
                "baseline_bins": _validated_bins(baseline_histogram.get("bins")),
                "bins": _validated_bins(histogram.get("bins")),
            }
        )

    if not comparisons:
        raise ValueError(f"Materialized payload has no usable comparisons: {input_path}")

    reference_bins = comparisons[0]["baseline_bins"]
    for comparison in comparisons:
        for bins in (comparison["baseline_bins"], comparison["bins"]):
            if len(bins) != len(reference_bins):
                raise ValueError("All comparisons must share the same histogram bin grid")
            for left, right in zip(reference_bins, bins):
                if left["bin_start"] != right["bin_start"] or left["bin_end"] != right["bin_end"]:
                    raise ValueError("All comparisons must share the same histogram bin edges")

    x_values = [float(bin_record["bin_start"]) for bin_record in reference_bins]
    widths = [float(bin_record["bin_end"]) - float(bin_record["bin_start"]) for bin_record in reference_bins]
    max_count = max(
        int(bin_record["count"])
        for comparison in comparisons
        for bins in (comparison["baseline_bins"], comparison["bins"])
        for bin_record in bins
    )

    plt, ticker = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axes = plt.subplots(
        len(comparisons),
        1,
        figsize=(args.figure_width, args.figure_height),
        sharex=True,
        sharey=True,
    )
    if hasattr(axes, "flatten"):
        axes = axes.flatten().tolist()
    elif not isinstance(axes, list):
        axes = [axes]

    for axis, comparison in zip(axes, comparisons):
        baseline_counts = [int(bin_record["count"]) for bin_record in comparison["baseline_bins"]]
        candidate_counts = [int(bin_record["count"]) for bin_record in comparison["bins"]]

        axis.bar(
            x_values,
            baseline_counts,
            width=widths,
            align="edge",
            color=comparison["baseline_color"],
            edgecolor=comparison["baseline_color"],
            linewidth=0.7,
            alpha=args.alpha,
            label=comparison["baseline_label"],
            zorder=2,
        )
        axis.bar(
            x_values,
            candidate_counts,
            width=widths,
            align="edge",
            color=comparison["color"],
            edgecolor=comparison["color"],
            linewidth=0.7,
            alpha=args.alpha,
            label=comparison["label"],
            zorder=3,
        )
        axis.set_title(f"{comparison['baseline_label']} vs {comparison['label']}", loc="left")
        axis.set_ylabel("Agent Frequency")
        axis.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.55, color="#9CA3AF")
        axis.set_axisbelow(True)
        axis.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.4, columnspacing=1.0)
        axis.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[-1].set_xlabel("Per-Agent Output Throughput (tokens/s)")
    axes[-1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    axes[-1].set_xlim(x_values[0], x_values[-1] + widths[-1])
    axes[-1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    for axis in axes:
        axis.set_ylim(0.0, max_count * 1.12 if max_count > 0 else 1.0)

    _add_panel_labels(axes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(h_pad=1.0)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
