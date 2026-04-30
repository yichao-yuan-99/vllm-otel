#!/usr/bin/env python3
"""Plot the control comparison figure from a materialized JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
#bfd3e6
#9ebcda
#8c96c6
#fdd49e
#fdbb84
#fc8d59
P5_BAR_COLORS = ("#000000", "#bfd3e6", "#8c96c6")
SYSTEM_BAR_COLORS = ("#000000", "#fdd49e", "#fc8d59")
BASE_FONT_SIZE = 14
LABEL_FONT_SIZE = 16
VALUE_FONT_SIZE = 13


def _display_label(label: str) -> str:
    display_label = label.replace("Freq Control", "Freq. Ctrl.")
    display_label = display_label.replace("Thrash Avoid.", "Thrash\nAvoid.")
    if display_label == "No Freq. Ctrl.":
        return "No Freq.\nCtrl."
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
        description=(
            "Plot the con-ctrl-compare figure from materialize_con_ctrl_compare.py."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_con_ctrl_compare.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: "
            "figures/con-ctrl-compare/output/<input-stem>.pdf"
        ),
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=9.4,
        help="Figure width in inches (default: 9.4).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=3.28,
        help="Figure height in inches (default: 3.28).",
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
    p5_values = _metric_values(variants, "p5_output_throughput_tokens_per_s")
    system_values = _metric_values(variants, "average_job_throughput_jobs_per_s")

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
    p5_axis, system_axis = axes

    for axis, values, ylabel, formatter, bar_colors in [
        (
            p5_axis,
            p5_values,
            "Per Agent P5 Throughput\n(tokens/s)",
            ticker.StrMethodFormatter("{x:,.0f}"),
            P5_BAR_COLORS,
        ),
        (
            system_axis,
            system_values,
            "System Job Throughput\n(jobs/s)",
            ticker.StrMethodFormatter("{x:.3f}"),
            SYSTEM_BAR_COLORS,
        ),
    ]:
        bars = axis.bar(
            x_positions,
            values,
            width=0.68,
            color=[bar_colors[index % len(bar_colors)] for index in x_positions],
            edgecolor="#202020",
            linewidth=1.0,
        )
        axis.set_ylabel(ylabel)
        axis.yaxis.set_label_coords(-0.13, 0.40)
        axis.set_xticks(x_positions, labels)
        axis.grid(
            True,
            axis="y",
            alpha=0.5,
            color="#6b7280",
            linewidth=1.2,
            linestyle=":",
        )
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelrotation=0, pad=10)
        axis.yaxis.set_major_formatter(formatter)
        axis.bar_label(
            bars,
            labels=[
                f"{value:.1f}" if "tokens" in ylabel else f"{value:.3f}"
                for value in values
            ],
            padding=3,
            fontsize=VALUE_FONT_SIZE,
        )

    p5_axis.set_ylim(0.0, max(35.0, max(p5_values, default=0.0) * 1.18))
    system_axis.set_ylim(0.0, max(0.08, max(system_values, default=0.0) * 1.18))
    system_axis.yaxis.set_label_coords(-0.22, 0.40)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
