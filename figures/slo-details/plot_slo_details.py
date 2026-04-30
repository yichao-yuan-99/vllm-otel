#!/usr/bin/env python3
"""Render the slo-details timeline figure from a materialized JSON dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
POWER_COLOR = "#8c4b1f"
FREQUENCY_COLOR = "#1d4f91"
FREQUENCY_CHANGED_COLOR = "#c65f5f"
SLO_COLOR = "#b4235a"
SLO_TARGET_COLOR = "#555555"
CONTEXT_COLOR = "#0f766e"
CONTEXT_TARGET_COLOR = "#d97706"


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
        description="Render the slo-details figure from a materialized JSON dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_slo_details.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: figures/slo-details/output/<input-stem>.pdf",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=7.0,
        help="Figure width in inches (default: 7.0).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=4.2336,
        help="Figure height in inches (default: 4.2336).",
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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _series_xy(
    raw_points: Any,
    *,
    x_key: str,
    y_key: str,
) -> tuple[list[float], list[float]]:
    if not isinstance(raw_points, list):
        return ([], [])
    x_values: list[float] = []
    y_values: list[float] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            continue
        x_value = _float_or_none(raw_point.get(x_key))
        y_value = _float_or_none(raw_point.get(y_key))
        if x_value is None or y_value is None:
            continue
        x_values.append(x_value / 60.0)
        y_values.append(y_value)
    return (x_values, y_values)


def _format_context_tick(value: float, _position: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1_000_000.0:
        return f"{value / 1_000_000.0:.1f}M"
    if abs(value) >= 1_000.0:
        return f"{value / 1_000.0:.0f}k"
    return f"{value:g}"


def _style_axis(axis: Any) -> None:
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.45)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.28)
    axis.minorticks_on()
    axis.margins(x=0.0)


def _add_panel_labels(axes: list[Any] | tuple[Any, ...]) -> None:
    for index, axis in enumerate(axes):
        axis.text(
            0.95,
            1.02,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            clip_on=False,
        )


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
    if not isinstance(raw_series, dict):
        raise ValueError(f"Materialized payload has no series object: {input_path}")

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    power_points = raw_series.get("power")
    frequency_points = raw_series.get("frequency")
    context_points = raw_series.get("context_usage")

    power_x, power_y = _series_xy(
        power_points,
        x_key="time_offset_s",
        y_key="power_w",
    )
    frequency_x, frequency_y = _series_xy(
        frequency_points,
        x_key="time_offset_s",
        y_key="target_frequency_mhz",
    )
    context_x, context_y = _series_xy(
        context_points,
        x_key="time_offset_s",
        y_key="context_usage_tokens",
    )

    if not power_x or not frequency_x or not context_x:
        raise ValueError(f"Materialized payload is missing plottable series: {input_path}")

    plt, ticker = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(args.figure_width, args.figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.15]},
    )
    context_axis, frequency_axis, power_axis = axes

    context_axis.plot(context_x, context_y, color=CONTEXT_COLOR, linewidth=1.7)
    context_axis.fill_between(context_x, context_y, color=CONTEXT_COLOR, alpha=0.12)
    context_axis.set_ylim(0.0, 450000.0)
    context_axis.set_ylabel("Context\n(tokens)")
    context_axis.yaxis.set_major_formatter(ticker.FuncFormatter(_format_context_tick))
    _style_axis(context_axis)

    changed_x: list[float] = []
    changed_y: list[float] = []
    held_x: list[float] = []
    held_y: list[float] = []
    max_x: list[float] = []
    max_y: list[float] = []
    max_frequency_mhz = max(frequency_y) if frequency_y else None
    if isinstance(frequency_points, list):
        for raw_point in frequency_points:
            if not isinstance(raw_point, dict):
                continue
            time_s = _float_or_none(raw_point.get("time_offset_s"))
            frequency_mhz = _float_or_none(raw_point.get("target_frequency_mhz"))
            if time_s is None or frequency_mhz is None:
                continue
            if max_frequency_mhz is not None and math.isclose(
                frequency_mhz,
                max_frequency_mhz,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                max_x.append(time_s / 60.0)
                max_y.append(frequency_mhz)
                continue
            if bool(raw_point.get("changed")):
                changed_x.append(time_s / 60.0)
                changed_y.append(frequency_mhz)
            else:
                held_x.append(time_s / 60.0)
                held_y.append(frequency_mhz)
    if held_x:
        frequency_axis.scatter(
            held_x,
            held_y,
            s=14,
            color=FREQUENCY_COLOR,
            alpha=0.65,
            linewidths=0.0,
        )
    if changed_x:
        frequency_axis.scatter(
            changed_x,
            changed_y,
            s=18,
            color=FREQUENCY_CHANGED_COLOR,
            alpha=0.9,
            linewidths=0.0,
        )
    if max_x:
        frequency_axis.scatter(
            max_x,
            max_y,
            s=22,
            color=FREQUENCY_COLOR,
            alpha=0.95,
            linewidths=0.0,
        )
    frequency_axis.set_ylim(0.0, 2000.0)
    if max_frequency_mhz is not None:
        frequency_axis.annotate(
            "Max Freq.",
            xy=(0.98, max_frequency_mhz),
            xycoords=("axes fraction", "data"),
            xytext=(0, -6),
            textcoords="offset points",
            ha="right",
            va="top",
            color=FREQUENCY_COLOR,
            fontsize=11,
        )
    frequency_axis.set_ylabel("Freq (MHz)")
    _style_axis(frequency_axis)

    power_axis.plot(power_x, power_y, color=POWER_COLOR, linewidth=1.6)
    power_axis.set_ylabel("Power (W)")
    power_axis.set_xlabel("Time From Start (minutes)")
    _style_axis(power_axis)

    x_max_minutes = max(
        max(power_x),
        max(frequency_x),
        max(context_x),
    )
    power_axis.set_xlim(0.0, x_max_minutes)
    power_axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _add_panel_labels(axes)
    figure.align_ylabels(axes)
    figure.tight_layout()
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
