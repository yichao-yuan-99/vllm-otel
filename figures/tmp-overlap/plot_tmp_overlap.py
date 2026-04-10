#!/usr/bin/env python3
"""Render a shared-axis overlay histogram from materialized tmp-overlap data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
DEFAULT_COLORS = (
    "#1D4F91",
    "#D97706",
    "#0F766E",
    "#9C4F96",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the overlaid throughput histogram from the JSON output of "
            "materialize_tmp_overlap.py."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_tmp_overlap.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: figures/tmp-overlap/output/<input-stem>.png",
    )
    parser.add_argument(
        "--title",
        default="Overlaid Agent Output Throughput Histogram",
        help="Figure title.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=11.0,
        help="Figure width in inches (default: 11.0).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=6.4,
        help="Figure height in inches (default: 6.4).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.48,
        help="Bar fill alpha for each overlaid histogram (default: 0.48).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for raster output (default: 220).",
    )
    return parser.parse_args()


def _import_matplotlib() -> Any:
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


def _default_output_path(input_path: Path) -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{input_path.stem}.png").resolve()


def _apply_plot_style(plt: Any) -> None:
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


def _validated_bins(raw_bins: Any) -> list[dict[str, float | int]]:
    if not isinstance(raw_bins, list) or not raw_bins:
        raise ValueError("Expected a non-empty histogram bins list")

    bins: list[dict[str, float | int]] = []
    for raw_bin in raw_bins:
        if not isinstance(raw_bin, dict):
            raise ValueError("Histogram bins must be JSON objects")
        bin_start = _float_or_none(raw_bin.get("bin_start"))
        bin_end = _float_or_none(raw_bin.get("bin_end"))
        count = _int_or_none(raw_bin.get("count"))
        if bin_start is None or bin_end is None or count is None:
            raise ValueError("Histogram bins must contain numeric bin_start, bin_end, and count")
        bins.append(
            {
                "bin_start": bin_start,
                "bin_end": bin_end,
                "count": count,
            }
        )
    return bins


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
        raise ValueError(f"Expected a JSON object in {input_path}")

    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError(f"Materialized payload has no datasets: {input_path}")

    datasets: list[dict[str, Any]] = []
    for index, raw_dataset in enumerate(raw_datasets):
        if not isinstance(raw_dataset, dict):
            continue
        histogram = raw_dataset.get("histogram")
        if not isinstance(histogram, dict):
            continue
        bins = _validated_bins(histogram.get("bins"))
        datasets.append(
            {
                "label": _string_or_none(raw_dataset.get("label")) or f"dataset-{index + 1}",
                "color": _string_or_none(raw_dataset.get("color")) or DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
                "summary": raw_dataset.get("summary") if isinstance(raw_dataset.get("summary"), dict) else {},
                "histogram_bins": bins,
            }
        )

    if not datasets:
        raise ValueError(f"Materialized payload has no usable datasets: {input_path}")

    reference_bins = datasets[0]["histogram_bins"]
    for dataset in datasets[1:]:
        if len(dataset["histogram_bins"]) != len(reference_bins):
            raise ValueError("All datasets must share the same histogram bin grid")
        for reference_bin, candidate_bin in zip(reference_bins, dataset["histogram_bins"]):
            if (
                reference_bin["bin_start"] != candidate_bin["bin_start"]
                or reference_bin["bin_end"] != candidate_bin["bin_end"]
            ):
                raise ValueError("All datasets must share the same histogram bin edges")

    x_values = [float(item["bin_start"]) for item in reference_bins]
    widths = [float(item["bin_end"]) - float(item["bin_start"]) for item in reference_bins]
    max_count = max(
        int(item["count"])
        for dataset in datasets
        for item in dataset["histogram_bins"]
    )

    plt = _import_matplotlib()
    _apply_plot_style(plt)

    figure, axis = plt.subplots(figsize=(args.figure_width, args.figure_height))
    for dataset_index, dataset in enumerate(datasets):
        y_values = [int(item["count"]) for item in dataset["histogram_bins"]]
        axis.bar(
            x_values,
            y_values,
            width=widths,
            align="edge",
            color=dataset["color"],
            edgecolor=dataset["color"],
            linewidth=0.9,
            alpha=args.alpha,
            label=dataset["label"],
            zorder=2 + dataset_index,
        )

        average_value = _float_or_none(dataset["summary"].get("avg"))
        if average_value is not None:
            axis.axvline(
                average_value,
                color=dataset["color"],
                linestyle=(0, (6, 4)),
                linewidth=1.25,
                alpha=0.95,
                zorder=5,
            )

    axis.set_title(args.title, loc="left", fontweight="semibold")
    axis.set_xlabel("Output Throughput (tokens/s)")
    axis.set_ylabel("Agent Count")
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.3)
    axis.minorticks_on()
    axis.set_xlim(x_values[0], x_values[-1] + widths[-1])
    axis.set_ylim(0, max_count * 1.12 if max_count > 0 else 1.0)

    bin_size = _float_or_none(payload.get("bin_size"))
    subtitle_parts = ["shared axes"]
    if bin_size is not None:
        subtitle_parts.append(f"bin size: {bin_size:.6g} tok/s")
    axis.text(
        0.0,
        1.02,
        " | ".join(subtitle_parts),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#2A3B47",
    )

    stats_lines = []
    for dataset in datasets:
        summary = dataset["summary"]
        average_value = _float_or_none(summary.get("avg"))
        min_value = _float_or_none(summary.get("min"))
        max_value = _float_or_none(summary.get("max"))
        sample_count = _int_or_none(summary.get("sample_count"))
        stats_lines.append(
            (
                f"{dataset['label']}: n={sample_count if sample_count is not None else 'n/a'}, "
                f"avg={average_value:.3g} tok/s, "
                f"min={min_value:.3g}, max={max_value:.3g}"
            )
            if average_value is not None and min_value is not None and max_value is not None
            else f"{dataset['label']}: n={sample_count if sample_count is not None else 'n/a'}"
        )
    axis.text(
        0.99,
        0.98,
        "\n".join(stats_lines),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#F7F9FC",
            "edgecolor": "#7A8B99",
            "alpha": 0.95,
        },
    )

    legend = axis.legend(frameon=True, loc="upper left")
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_edgecolor("#C7D2DA")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, format=output_path.suffix.lstrip(".") or "png", dpi=args.dpi)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
