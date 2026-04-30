#!/usr/bin/env python3
"""Plot the mutli figure from a materialized JSON payload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"

BAR_COLORS = ("#d95f02", "#7570b3", "#1b9e77")
PIE_COLORS = ("#bebada", "#fb8072", "#80b1d3", "#fdb462")
SINGLE_INSTANCE_BAR = {
    "label": "4x single instance",
    "value": 823.543664,
}
BASE_PIE_RADIUS = 1.22
PIE_AXIS_VERTICAL_OFFSET = 0.04
PANEL_LABEL_Y = 0.035
PIE_TITLE_PAD = 20.0
SUBPLOT_WSPACE = -0.08


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the mutli figure from materialize_mutli.py."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_mutli.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: figures/mutli/output/<input-stem>.pdf",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=16.0,
        help="Figure width in inches (default: 16.0).",
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_output_path(input_path: Path) -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{input_path.stem}.pdf").resolve()


def _float_or_zero(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return 0.0


def _autopct(values: list[float]) -> Any:
    total = sum(values)

    def formatter(pct: float) -> str:
        absolute = total * pct / 100.0
        if absolute <= 0.0:
            return ""
        return f"{pct:.1f}%"

    return formatter


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
    raw_variants = payload.get("variants")
    if not isinstance(raw_variants, list) or len(raw_variants) != 3:
        raise ValueError(f"Expected three variants in {input_path}")

    variants: list[dict[str, Any]] = [
        item for item in raw_variants if isinstance(item, dict)
    ]
    if len(variants) != 3:
        raise ValueError(f"Expected three variant objects in {input_path}")

    plt = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 16.0,
            "axes.titlesize": 19.0,
            "axes.labelsize": 17.0,
            "xtick.labelsize": 15.0,
            "ytick.labelsize": 15.0,
        }
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(args.figure_width, args.figure_height),
        subplot_kw={},
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]},
    )

    bar_axis = axes[0]
    bar_entries = [
        {
            "label": str(variant.get("label", f"Variant {index + 1}")),
            "value": _float_or_zero(variant.get("metrics", {}).get("average_power_w"))
            if isinstance(variant.get("metrics"), dict)
            else 0.0,
        }
        for index, variant in enumerate(variants)
    ]
    bar_entries.insert(1, SINGLE_INSTANCE_BAR)
    bar_labels = [entry["label"] for entry in bar_entries]
    bar_values = [
        entry["value"]
        for entry in bar_entries
    ]
    x_positions = list(range(len(bar_values)))
    bars = bar_axis.bar(
        x_positions,
        bar_values,
        color=("#111111", BAR_COLORS[0], BAR_COLORS[1], BAR_COLORS[2]),
        width=0.66,
        edgecolor="#111827",
        linewidth=1.1,
    )
    bar_axis.set_title("Average Total Power")
    bar_axis.set_ylabel("Average Power (W)")
    bar_axis.set_xticks(x_positions, bar_labels, rotation=12, ha="right")
    bar_axis.grid(True, axis="y", linestyle=":", linewidth=1.0, alpha=0.65)
    bar_axis.set_axisbelow(True)
    bar_axis.spines["top"].set_visible(False)
    bar_axis.spines["right"].set_visible(False)
    bar_axis.bar_label(
        bars,
        labels=[f"{value:.1f}" for value in bar_values],
        padding=3,
        fontsize=14.0,
    )
    max_bar = max(bar_values) if bar_values else 0.0
    bar_axis.set_ylim(0.0, max_bar * 1.2 if max_bar > 0.0 else 1.0)

    round_robin_metrics = variants[1].get("metrics")
    round_robin_total_power_w = (
        _float_or_zero(round_robin_metrics.get("average_power_w"))
        if isinstance(round_robin_metrics, dict)
        else 0.0
    )
    pie_specs = [
        (axes[1], variants[1], "Round Robin Routing"),
        (axes[2], variants[2], "Context-Aware Routing"),
    ]
    for axis, variant, title in pie_specs:
        per_gpu = variant.get("per_gpu_average_power")
        if not isinstance(per_gpu, list) or not per_gpu:
            raise ValueError(f"Variant is missing per_gpu_average_power: {variant}")
        metrics = variant.get("metrics")
        total_power_w = (
            _float_or_zero(metrics.get("average_power_w"))
            if isinstance(metrics, dict)
            else 0.0
        )
        pie_radius = BASE_PIE_RADIUS
        if (
            str(variant.get("label", "")).strip().upper() == "KAIROS"
            and round_robin_total_power_w > 0.0
        ):
            pie_radius = BASE_PIE_RADIUS * (total_power_w / round_robin_total_power_w)
        labels: list[str] = []
        values: list[float] = []
        for gpu in per_gpu:
            if not isinstance(gpu, dict):
                continue
            labels.append(f"INS. {len(labels) + 1}")
            values.append(_float_or_zero(gpu.get("average_power_w")))
        if not values:
            raise ValueError(f"Variant has no usable GPU slices: {variant}")
        _, _, autotexts = axis.pie(
            values,
            labels=labels,
            colors=PIE_COLORS[: len(values)],
            startangle=90,
            counterclock=False,
            autopct=_autopct(values),
            pctdistance=0.76,
            radius=pie_radius,
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
            textprops={"fontsize": 15.0},
        )
        for autotext in autotexts:
            autotext.set_color("black")
            autotext.set_fontsize(15.0)
        axis.set_title(
            f"{title}\nTotal Power = {total_power_w:.1f} W",
            pad=PIE_TITLE_PAD,
        )
        axis.text(
            0.5,
            -0.08,
            "Power Breakdown",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=16.0,
        )
        axis.axis("equal")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    figure.subplots_adjust(wspace=SUBPLOT_WSPACE)
    for axis in axes[1:]:
        position = axis.get_position()
        axis.set_position(
            [
                position.x0,
                position.y0 - PIE_AXIS_VERTICAL_OFFSET,
                position.width,
                position.height,
            ]
        )
    for index, axis in enumerate(axes):
        position = axis.get_position()
        figure.text(
            position.x0 + (position.width / 2.0),
            PANEL_LABEL_Y,
            f"({chr(ord('a') + index)})",
            ha="center",
            va="bottom",
            fontsize=20.0,
        )
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
