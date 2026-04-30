#!/usr/bin/env python3
"""Render the transposed turns-and-duration comparison figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
AGENT_COLORS = {
    "mini-swe-agent": "#1d4f91",
    "terminus-2": "#d2a22c",
}
MEAN_MARKER_COLOR = "#f72585"


def _import_matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import transforms
        from matplotlib import ticker
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to render this figure. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt, ticker, transforms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the turns-scatter-duration-violin figure from the JSON output "
            "of materialize_turns_scatter_duration_violin.py."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_turns_scatter_duration_violin.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: "
            "figures/turns-scatter-duration-violin/output/<input-stem>.pdf"
        ),
    )
    parser.add_argument(
        "--turn-scale",
        choices=("log", "linear"),
        default="log",
        help="X-axis scale for the turns panel. Default: log.",
    )
    parser.add_argument(
        "--duration-scale",
        choices=("log", "linear"),
        default="log",
        help="X-axis scale for the duration panel. Default: log.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. Default: title from the materialized payload.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=18.6,
        help="Figure width in inches (default: 18.6).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=7.4,
        help="Figure height in inches (default: 7.4).",
    )
    parser.add_argument(
        "--jitter-height",
        type=float,
        default=0.09,
        help="Half-height of vertical point jitter around each violin center.",
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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _jittered_positions(
    *,
    label: str,
    center: float,
    count: int,
    jitter_height: float,
) -> list[float]:
    generator = random.Random(_stable_seed(label))
    return [
        center + generator.uniform(-jitter_height, jitter_height)
        for _ in range(count)
    ]


def _panel_color(agent_type: str, agent_label: str, fallback_index: int) -> str:
    if agent_type in AGENT_COLORS:
        return AGENT_COLORS[agent_type]
    palette = (
        "#1d4f91",
        "#d2a22c",
        "#007c7c",
        "#9c4f96",
        "#c65f5f",
        "#4f7d39",
    )
    if agent_label in AGENT_COLORS:
        return AGENT_COLORS[agent_label]
    return palette[fallback_index % len(palette)]


def _format_turn_tick(value: float, _position: float) -> str:
    if value <= 0.0:
        return ""
    if value >= 1000.0:
        scaled = value / 1000.0
        if math.isclose(scaled, round(scaled)):
            return f"{int(round(scaled))}k"
        return f"{scaled:.1f}".rstrip("0").rstrip(".") + "k"
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _format_duration_tick(value: float, _position: float) -> str:
    if value <= 0.0:
        return ""
    if value >= 1000.0:
        scaled = value / 1000.0
        if math.isclose(scaled, round(scaled)):
            return f"{int(round(scaled))}k"
        return f"{scaled:.1f}".rstrip("0").rstrip(".") + "k"
    return f"{value:g}"


def _style_violin(violin: Any, *, color: str) -> None:
    for body in violin.get("bodies", []):
        body.set_facecolor(color)
        body.set_edgecolor("#2f2f2f")
        body.set_linewidth(1.1)
        body.set_alpha(0.74)
        body.set_zorder(1.5)


def _display_benchmark_label(label: str) -> str:
    if label == "SWE-bench Verified":
        return "SWE-bench\nVerified"
    if label == "Terminal-Bench 2.0":
        return "Terminal-\nBench 2.0"
    return label


def _display_metric_label(label: str) -> str:
    if label == "Turns per job":
        return "Turns per Job"
    if label == "Agent duration per job (s)":
        return "Agent Duration per Job (s)"
    return label


def _draw_metric_panel(
    *,
    axis: Any,
    panels: list[dict[str, Any]],
    y_positions: list[float],
    value_key: str,
    stats_key: str,
    x_scale: str,
    jitter_height: float,
    value_label: str,
    axis_title: str,
    tick_formatter: Any,
    ticker: Any,
) -> None:
    values_per_panel = [
        list(panel[value_key])
        for panel in panels
    ]
    min_value = min(min(values) for values in values_per_panel)
    max_value = max(max(values) for values in values_per_panel)
    if x_scale == "log" and min_value <= 0.0:
        raise ValueError(
            f"Log x-scale requires strictly positive values for {value_key!r}. "
            "Use the linear scale instead."
        )

    for y_position, panel in zip(y_positions, panels):
        values = list(panel[value_key])
        color = str(panel["color"])

        violin = axis.violinplot(
            [values],
            positions=[y_position],
            widths=0.95,
            vert=False,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        _style_violin(violin, color=color)

        axis.boxplot(
            [values],
            positions=[y_position],
            widths=0.38,
            vert=False,
            showfliers=False,
            patch_artist=True,
            boxprops={
                "facecolor": "white",
                "edgecolor": "#3e3e3e",
                "linewidth": 1.4,
                "alpha": 0.92,
                "zorder": 3.5,
            },
            whiskerprops={"color": "#3e3e3e", "linewidth": 1.2, "zorder": 3.5},
            capprops={"color": "#3e3e3e", "linewidth": 1.2, "zorder": 3.5},
            medianprops={"color": "#2f2f2f", "linewidth": 2.0, "zorder": 3.7},
        )

        axis.scatter(
            values,
            _jittered_positions(
                label=f"{value_key}::{panel['benchmark']}::{panel['agent_type']}",
                center=y_position,
                count=len(values),
                jitter_height=jitter_height,
            ),
            s=24,
            c="#666666",
            alpha=0.34,
            edgecolors="#4f4f4f",
            linewidths=0.45,
            zorder=3.1,
        )

        stats = panel[stats_key]
        mean = _float_or_none(stats.get("mean")) if isinstance(stats, dict) else None
        if mean is not None:
            axis.scatter(
                [mean],
                [y_position],
                s=100,
                marker="D",
                c=MEAN_MARKER_COLOR,
                edgecolors="white",
                linewidths=0.9,
                zorder=4.2,
            )

    if x_scale == "log":
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(
            ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0))
        )
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(tick_formatter))

    axis.grid(axis="x", which="major", color="#d7d7d7", linewidth=1.0)
    axis.grid(axis="x", which="minor", color="#ececec", linewidth=0.6, alpha=0.6)
    axis.set_xlim(
        min_value * (0.85 if x_scale == "linear" else 0.9),
        max_value * (1.18 if x_scale == "linear" else 1.30),
    )
    axis.set_xlabel(_display_metric_label(value_label), fontsize=24.0)
    axis.set_title("")


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

    raw_panels = payload.get("panels")
    if not isinstance(raw_panels, list) or not raw_panels:
        raise ValueError(f"Materialized payload has no panels: {input_path}")

    figure_title = (
        args.title
        or _string_or_none(payload.get("figure_title"))
        or (
            "Turn Count and Agent Duration Distributions by Benchmark and Agent "
            "(Horizontal Violin + Boxplot)"
        )
    )
    turn_metric_label = (
        _string_or_none(payload.get("turn_metric_label")) or "Turns per job"
    )
    duration_metric_label = (
        _string_or_none(payload.get("duration_metric_label"))
        or "Agent duration per job (s)"
    )

    panels: list[dict[str, Any]] = []
    for fallback_index, raw_panel in enumerate(raw_panels):
        if not isinstance(raw_panel, dict):
            continue
        raw_turns = raw_panel.get("turns")
        raw_durations = raw_panel.get("durations_s")
        if not isinstance(raw_turns, list) or not isinstance(raw_durations, list):
            continue

        turns = [
            parsed
            for parsed in (_float_or_none(value) for value in raw_turns)
            if parsed is not None
        ]
        durations = [
            parsed
            for parsed in (_float_or_none(value) for value in raw_durations)
            if parsed is not None
        ]
        if not turns or not durations:
            continue

        panel_index = _int_or_none(raw_panel.get("panel_index"))
        benchmark = (
            _string_or_none(raw_panel.get("benchmark"))
            or f"benchmark-{fallback_index + 1}"
        )
        benchmark_label = (
            _string_or_none(raw_panel.get("benchmark_label")) or benchmark
        )
        agent_type = (
            _string_or_none(raw_panel.get("agent_type"))
            or f"agent-{fallback_index + 1}"
        )
        agent_label = _string_or_none(raw_panel.get("agent_label")) or agent_type

        panels.append(
            {
                "panel_index": panel_index if panel_index is not None else fallback_index,
                "benchmark": benchmark,
                "benchmark_label": benchmark_label,
                "agent_type": agent_type,
                "agent_label": agent_label,
                "turns": turns,
                "turn_stats": (
                    raw_panel.get("turn_stats")
                    if isinstance(raw_panel.get("turn_stats"), dict)
                    else {}
                ),
                "durations_s": durations,
                "duration_stats": (
                    raw_panel.get("duration_stats")
                    if isinstance(raw_panel.get("duration_stats"), dict)
                    else {}
                ),
                "color": _panel_color(agent_type, agent_label, fallback_index),
            }
        )

    if not panels:
        raise ValueError(f"Materialized payload has no usable panels: {input_path}")

    panels.sort(key=lambda item: item["panel_index"])

    y_positions: list[float] = []
    benchmark_centers: dict[str, float] = {}
    benchmark_labels: dict[str, str] = {}
    group_members: dict[str, list[float]] = {}
    separators: list[float] = []
    current_position = 1.0
    previous_benchmark: str | None = None
    previous_position: float | None = None

    for panel in panels:
        benchmark = str(panel["benchmark"])
        if previous_benchmark is not None and benchmark != previous_benchmark:
            separators.append((previous_position + current_position) / 2.0)
            current_position += 1.0
        y_positions.append(current_position)
        benchmark_labels[benchmark] = str(panel["benchmark_label"])
        group_members.setdefault(benchmark, []).append(current_position)
        previous_benchmark = benchmark
        previous_position = current_position
        current_position += 1.0

    for benchmark, positions in group_members.items():
        benchmark_centers[benchmark] = sum(positions) / len(positions)

    plt, ticker, transforms = _import_matplotlib()
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 16.5,
            "axes.titlesize": 24.0,
            "axes.labelsize": 20.0,
            "xtick.labelsize": 22.0,
            "ytick.labelsize": 22.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(args.figure_width, args.figure_height),
        sharey=True,
    )
    figure.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.11, wspace=0.09)

    left_axis, right_axis = axes
    for axis in axes:
        axis.set_axisbelow(True)
        for separator in separators:
            axis.axhline(separator, color="#dddddd", linewidth=1.0, zorder=0)
        axis.set_ylim(min(y_positions) - 0.8, max(y_positions) + 0.8)
        axis.invert_yaxis()
        axis.tick_params(axis="y", length=0, pad=8)

    _draw_metric_panel(
        axis=left_axis,
        panels=panels,
        y_positions=y_positions,
        value_key="turns",
        stats_key="turn_stats",
        x_scale=args.turn_scale,
        jitter_height=args.jitter_height,
        value_label=turn_metric_label,
        axis_title="Turns per Job",
        tick_formatter=_format_turn_tick,
        ticker=ticker,
    )
    _draw_metric_panel(
        axis=right_axis,
        panels=panels,
        y_positions=y_positions,
        value_key="durations_s",
        stats_key="duration_stats",
        x_scale=args.duration_scale,
        jitter_height=args.jitter_height,
        value_label=duration_metric_label,
        axis_title="Agent Duration per Job",
        tick_formatter=_format_duration_tick,
        ticker=ticker,
    )

    left_axis.set_yticks(y_positions)
    left_axis.set_yticklabels([])
    left_axis.set_ylabel("")
    left_axis.tick_params(axis="y", labelleft=False)
    right_axis.tick_params(axis="y", left=False, labelleft=False)

    group_transform = transforms.blended_transform_factory(
        left_axis.transAxes,
        left_axis.transData,
    )
    for benchmark, center in benchmark_centers.items():
        left_axis.text(
            -0.085,
            center,
            _display_benchmark_label(benchmark_labels[benchmark]),
            transform=group_transform,
            ha="center",
            va="center",
            fontsize=21.0,
            fontweight="semibold",
            color="#333333",
            rotation=90,
        )

    unique_agents: list[tuple[str, str]] = []
    seen_agents: set[str] = set()
    for panel in panels:
        agent_type = str(panel["agent_type"])
        if agent_type in seen_agents:
            continue
        seen_agents.add(agent_type)
        unique_agents.append((agent_type, str(panel["agent_label"])))

    legend_handles = [
        Patch(
            facecolor=_panel_color(agent_type, agent_label, index),
            edgecolor="#2f2f2f",
            linewidth=1.0,
            alpha=0.74,
            label=agent_label,
        )
        for index, (agent_type, agent_label) in enumerate(unique_agents)
    ]
    if legend_handles:
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=max(1, len(legend_handles)),
            frameon=False,
            fontsize=20.0,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
