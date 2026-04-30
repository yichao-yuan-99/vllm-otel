#!/usr/bin/env python3
"""Plot the SLO comparison figure from a materialized JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
NO_FREQ_CONTROL_COLOR = "#000000"
THROUGHPUT_COLORS = ("#fde0dd", "#fcc5c0", "#fa9fb5")
POWER_COLORS = ("#e5f5e0", "#c7e9c0", "#a1d99b")
BASE_FONT_SIZE = 14
LABEL_FONT_SIZE = 16
VALUE_FONT_SIZE = 14


def _display_label(label: str) -> str:
    if label == "No Freq Control":
        return "No Freq.\nCtrl."

    display_label = (
        label.replace("Freq Control", "KAIROS")
        .replace("Freq\nControl", "KAIROS")
    )
    return display_label


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
        description="Plot the slo-compare figure from materialize_slo_compare.py."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_slo_compare.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: "
            "figures/slo-compare/output/<input-stem>.pdf"
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
        default=10.2,
        help="Figure width in inches (default: 10.2).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=3.44,
        help="Figure height in inches (default: 3.44).",
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


def _metric_values(
    variants: list[dict[str, Any]],
    metric_key: str,
) -> list[float]:
    values: list[float] = []
    for variant in variants:
        metrics = variant.get("metrics")
        if not isinstance(metrics, dict):
            values.append(0.0)
            continue
        raw_value = metrics.get(metric_key)
        values.append(float(raw_value) if isinstance(raw_value, (int, float)) else 0.0)
    return values


def _bar_colors(
    count: int,
    *,
    palette: tuple[str, ...],
) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return [NO_FREQ_CONTROL_COLOR]

    colors = [NO_FREQ_CONTROL_COLOR]
    for index in range(1, count):
        colors.append(palette[(index - 1) % len(palette)])
    return colors


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
        raise ValueError(f"Expected JSON object in {input_path}")
    raw_variants = payload.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError(f"Materialized payload has no variants: {input_path}")

    variants = [
        variant
        for variant in raw_variants
        if isinstance(variant, dict)
    ]
    labels = [
        _display_label(str(variant.get("label", f"Variant {index + 1}")))
        for index, variant in enumerate(variants)
    ]
    x_positions = list(range(len(variants)))
    throughput_values = _metric_values(variants, "p5_output_throughput_tokens_per_s")
    power_values = _metric_values(variants, "average_power_w")

    plt, ticker = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(args.figure_width, args.figure_height),
        sharex=True,
    )
    throughput_axis, power_axis = axes

    for axis, values, ylabel, colors in [
        (
            throughput_axis,
            throughput_values,
            "Per Agent P5 Throughput\n(tokens/s)",
            _bar_colors(
                len(variants),
                palette=THROUGHPUT_COLORS,
            ),
        ),
        (
            power_axis,
            power_values,
            "Average Power (W)",
            _bar_colors(
                len(variants),
                palette=POWER_COLORS,
            ),
        ),
    ]:
        bars = axis.bar(
            x_positions,
            values,
            width=0.68,
            color=colors,
            edgecolor="#202020",
            linewidth=1.0,
        )
        axis.set_ylabel(ylabel)
        axis.yaxis.set_label_coords(-0.12, 0.46)
        axis.set_xticks(x_positions, labels)
        axis.grid(
            True,
            axis="y",
            alpha=0.6,
            color="#9ca3af",
            linewidth=1.0,
            linestyle=":",
        )
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelrotation=0, pad=10)
        axis.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        axis.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in values],
            padding=3,
            fontsize=VALUE_FONT_SIZE,
        )

    throughput_axis.set_ylim(0.0, 60.0)
    power_axis.set_ylim(0.0, 400.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
