#!/usr/bin/env python3
"""Render the con-ctrl-details timeline figure from a materialized JSON dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
NO_THRASH_COLOR = "#8c4b1f"
NO_THRASH_CONTEXT_COLOR = "#5b6c1f"
CTX_AWARE_THROUGHPUT_COLOR = "#1d4f91"
CONTEXT_COLOR = "#0f766e"
PENDING_COLOR = "#b45309"
CONTEXT_REFERENCE_LINE_COLOR = "#dc2626"
CONTEXT_REFERENCE_TOKENS = 527664.0
CONTEXT_REFERENCE_LABEL_X = 3.0
CONTEXT_REFERENCE_LABEL_Y = 600000.0
THROUGHPUT_REFERENCE_JOBS_PER_S = 0.05


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
        description="Render the con-ctrl-details figure from a materialized JSON dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_con_ctrl_details.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: figures/con-ctrl-details/output/<input-stem>.pdf",
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
        default=6.7,
        help="Figure height in inches (default: 6.7).",
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


def _annotate_context_capacity(axis: Any) -> None:
    axis.text(
        CONTEXT_REFERENCE_LABEL_X,
        CONTEXT_REFERENCE_LABEL_Y,
        "Context Capacity",
        color=CONTEXT_REFERENCE_LINE_COLOR,
        fontsize=10.5,
        ha="left",
        va="bottom",
    )


def _add_panel_labels(axes: list[Any] | tuple[Any, ...]) -> None:
    for index, axis in enumerate(axes):
        axis.text(
            0.96,
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

    no_thrash_x, no_thrash_y = _series_xy(
        raw_series.get("job_throughput_no_thrashing_avoidance"),
        x_key="time_offset_s",
        y_key="throughput_jobs_per_s",
    )
    no_thrash_context_x, no_thrash_context_y = _series_xy(
        raw_series.get("context_usage_no_thrashing_avoidance"),
        x_key="time_offset_s",
        y_key="context_usage_tokens",
    )
    ctx_aware_throughput_x, ctx_aware_throughput_y = _series_xy(
        raw_series.get("job_throughput_with_thrashing_avoidance"),
        x_key="time_offset_s",
        y_key="throughput_jobs_per_s",
    )
    context_x, context_y = _series_xy(
        raw_series.get("context_usage_with_thrashing_avoidance"),
        x_key="time_offset_s",
        y_key="context_usage_tokens",
    )
    pending_x, pending_y = _series_xy(
        raw_series.get("pending_agent_count_with_thrashing_avoidance"),
        x_key="time_offset_s",
        y_key="pending_agent_count",
    )

    if (
        not no_thrash_x
        or not no_thrash_context_x
        or not ctx_aware_throughput_x
        or not context_x
        or not pending_x
    ):
        raise ValueError(f"Materialized payload is missing plottable series: {input_path}")

    plt, ticker = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    figure, axes = plt.subplots(
        5,
        1,
        figsize=(args.figure_width, args.figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15, 0.85, 1.0, 1.15]},
    )
    (
        no_thrash_context_axis,
        no_thrash_axis,
        pending_axis,
        context_axis,
        ctx_aware_throughput_axis,
    ) = axes

    no_thrash_axis.plot(no_thrash_x, no_thrash_y, color=NO_THRASH_COLOR, linewidth=1.6)
    no_thrash_axis.axhline(
        THROUGHPUT_REFERENCE_JOBS_PER_S,
        color=CTX_AWARE_THROUGHPUT_COLOR,
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
    )
    no_thrash_axis.set_ylabel("Jobs/s")
    no_thrash_axis.set_title("Job Throughput: KAIROS Without Thrashing Avoidance", loc="left")
    _style_axis(no_thrash_axis)

    no_thrash_context_axis.plot(
        no_thrash_context_x,
        no_thrash_context_y,
        color=NO_THRASH_CONTEXT_COLOR,
        linewidth=1.7,
    )
    no_thrash_context_axis.fill_between(
        no_thrash_context_x,
        no_thrash_context_y,
        color=NO_THRASH_CONTEXT_COLOR,
        alpha=0.12,
    )
    no_thrash_context_axis.set_ylabel("Context\n(tokens)")
    no_thrash_context_axis.set_title(
        "Context Usage: KAIROS Without Thrashing Avoidance",
        loc="left",
    )
    no_thrash_context_axis.axhline(
        CONTEXT_REFERENCE_TOKENS,
        color=CONTEXT_REFERENCE_LINE_COLOR,
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    _annotate_context_capacity(no_thrash_context_axis)
    no_thrash_context_axis.yaxis.set_major_formatter(ticker.FuncFormatter(_format_context_tick))
    _style_axis(no_thrash_context_axis)

    pending_axis.step(
        pending_x,
        pending_y,
        where="post",
        color=PENDING_COLOR,
        linewidth=1.7,
    )
    pending_axis.fill_between(
        pending_x,
        pending_y,
        step="post",
        color=PENDING_COLOR,
        alpha=0.14,
    )
    pending_axis.set_ylabel("Agents")
    pending_axis.set_title(
        "Pending Agent Count: KAIROS With Thrashing Avoidance",
        loc="left",
    )
    _style_axis(pending_axis)

    ctx_aware_throughput_axis.plot(
        ctx_aware_throughput_x,
        ctx_aware_throughput_y,
        color=CTX_AWARE_THROUGHPUT_COLOR,
        linewidth=1.6,
    )
    ctx_aware_throughput_axis.axhline(
        THROUGHPUT_REFERENCE_JOBS_PER_S,
        color=CTX_AWARE_THROUGHPUT_COLOR,
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
    )
    ctx_aware_throughput_axis.set_ylabel("Jobs/s")
    ctx_aware_throughput_axis.set_title(
        "Job Throughput: KAIROS With Thrashing Avoidance",
        loc="left",
    )
    ctx_aware_throughput_axis.set_xlabel("Time From Start (minutes)")
    _style_axis(ctx_aware_throughput_axis)

    context_axis.plot(context_x, context_y, color=CONTEXT_COLOR, linewidth=1.7)
    context_axis.fill_between(context_x, context_y, color=CONTEXT_COLOR, alpha=0.12)
    context_axis.set_ylabel("Context\n(tokens)")
    context_axis.set_title(
        "Context Usage: KAIROS With Thrashing Avoidance",
        loc="left",
    )
    context_axis.axhline(
        CONTEXT_REFERENCE_TOKENS,
        color=CONTEXT_REFERENCE_LINE_COLOR,
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    _annotate_context_capacity(context_axis)
    context_axis.yaxis.set_major_formatter(ticker.FuncFormatter(_format_context_tick))
    _style_axis(context_axis)

    throughput_max = max(
        max(no_thrash_y, default=0.0),
        max(ctx_aware_throughput_y, default=0.0),
    )
    context_max = max(
        max(no_thrash_context_y, default=0.0),
        max(context_y, default=0.0),
    )
    no_thrash_axis.set_ylim(0.0, max(0.055, throughput_max * 1.06))
    ctx_aware_throughput_axis.set_ylim(0.0, max(0.055, throughput_max * 1.06))
    no_thrash_context_axis.set_ylim(0.0, context_max * 1.06 if context_max > 0.0 else 1.0)
    context_axis.set_ylim(0.0, context_max * 1.06 if context_max > 0.0 else 1.0)
    pending_axis.set_ylim(0.0, max(1.0, math.ceil(max(pending_y, default=0.0) * 1.1)))

    x_max_minutes = max(
        max(no_thrash_x),
        max(no_thrash_context_x),
        max(ctx_aware_throughput_x),
        max(context_x),
        max(pending_x),
    )
    ctx_aware_throughput_axis.set_xlim(0.0, x_max_minutes)
    ctx_aware_throughput_axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    for axis in (no_thrash_axis, ctx_aware_throughput_axis):
        axis.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _add_panel_labels(axes)
    figure.align_ylabels(axes)
    figure.tight_layout(h_pad=0.45)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
