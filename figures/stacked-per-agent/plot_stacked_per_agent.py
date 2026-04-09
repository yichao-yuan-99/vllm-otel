#!/usr/bin/env python3
"""Render a stacked per-agent context-usage bar chart from materialized data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"


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
        description="Plot the stacked-per-agent figure from a materialized JSON dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_stacked_per_agent.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: figures/stacked-per-agent/output/<input-stem>.pdf",
    )
    parser.add_argument(
        "--value-mode",
        choices=("average", "integral"),
        default="average",
        help=(
            "Which materialized value to render. "
            "'average' matches the original smoothed line chart scale (default)."
        ),
    )
    parser.add_argument(
        "--legend",
        choices=("auto", "show", "hide"),
        default="auto",
        help="Legend mode. Default: auto.",
    )
    parser.add_argument(
        "--legend-max-agents",
        type=int,
        default=24,
        help="Auto-show the legend only when agent_count <= this threshold (default: 24).",
    )
    parser.add_argument(
        "--title",
        default="Per-Agent Stacked Context Usage",
        help="Figure title.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=13.0,
        help="Figure width in inches (default: 13.0).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=6.8,
        help="Figure height in inches (default: 6.8).",
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


def _default_output_path(input_path: Path) -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{input_path.stem}.pdf").resolve()


def _value_field_name(value_mode: str) -> str:
    if value_mode == "integral":
        return "integral_value"
    return "average_value"


def _y_axis_label(value_mode: str) -> str:
    if value_mode == "integral":
        return "Context Usage Integral in Window (token-s)"
    return "Average Context Usage in Window (tokens)"


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

    raw_agents = payload.get("agents")
    raw_windows = payload.get("windows")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ValueError(f"Materialized payload has no agents: {input_path}")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError(f"Materialized payload has no windows: {input_path}")

    agents: list[dict[str, Any]] = []
    for raw_agent in raw_agents:
        if not isinstance(raw_agent, dict):
            continue
        agent_index = _int_or_none(raw_agent.get("agent_index"))
        if agent_index is None:
            continue
        color_hex = raw_agent.get("color_hex")
        if not isinstance(color_hex, str) or not color_hex:
            continue
        agents.append(
            {
                "agent_index": agent_index,
                "agent_label": raw_agent.get("agent_label") or f"A{agent_index + 1}",
                "agent_key": raw_agent.get("agent_key"),
                "color_hex": color_hex,
            }
        )
    if not agents:
        raise ValueError(f"Materialized payload has no usable agents: {input_path}")

    agents.sort(key=lambda item: item["agent_index"])
    agent_count = len(agents)
    window_count = len(raw_windows)
    value_field_name = _value_field_name(args.value_mode)

    x_starts: list[float] = []
    widths: list[float] = []
    stacked_values_by_agent: list[list[float]] = [[0.0] * window_count for _ in range(agent_count)]

    for window_position, raw_window in enumerate(raw_windows):
        if not isinstance(raw_window, dict):
            raise ValueError(f"Window {window_position} is not a JSON object")
        window_start_s = _float_or_none(raw_window.get("window_start_s"))
        window_duration_s = _float_or_none(raw_window.get("window_duration_s"))
        if window_start_s is None or window_duration_s is None or window_duration_s <= 0.0:
            raise ValueError(f"Window {window_position} is missing valid timing fields")
        x_starts.append(window_start_s)
        widths.append(window_duration_s)

        contributions = raw_window.get("contributions")
        if not isinstance(contributions, list):
            continue
        for contribution in contributions:
            if not isinstance(contribution, dict):
                continue
            agent_index = _int_or_none(contribution.get("agent_index"))
            value = _float_or_none(contribution.get(value_field_name))
            if agent_index is None or value is None:
                continue
            if 0 <= agent_index < agent_count:
                stacked_values_by_agent[agent_index][window_position] = value

    plt, ticker = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axis = plt.subplots(figsize=(args.figure_width, args.figure_height))
    bottoms = [0.0] * window_count
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for agent in agents:
        agent_index = int(agent["agent_index"])
        heights = stacked_values_by_agent[agent_index]
        bar_container = axis.bar(
            x_starts,
            heights,
            width=widths,
            bottom=bottoms,
            align="edge",
            color=str(agent["color_hex"]),
            linewidth=0.0,
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
        legend_handles.append(bar_container[0])
        legend_labels.append(str(agent["agent_label"]))

    axis.set_title(args.title, loc="left", fontweight="semibold")
    subtitle_parts = [
        f"{agent_count} agents",
        f"{window_count} bars",
    ]
    window_size_s = _float_or_none(payload.get("window_size_s"))
    if window_size_s is not None:
        subtitle_parts.append(f"{window_size_s:g}s windows")
    if args.value_mode == "average":
        subtitle_parts.append("bar height = average context usage in each window")
    else:
        subtitle_parts.append("bar height = context-usage integral in each window")
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

    legend_should_show = False
    if args.legend == "show":
        legend_should_show = True
    elif args.legend == "auto":
        legend_should_show = agent_count <= args.legend_max_agents

    if legend_should_show:
        legend_columns = max(1, min(4, math.ceil(agent_count / 12)))
        axis.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            ncols=legend_columns,
            title="Agents",
            fontsize=8,
            title_fontsize=9,
        )
    elif args.legend == "auto" and agent_count > args.legend_max_agents:
        axis.text(
            0.99,
            0.99,
            f"Legend hidden automatically ({agent_count} agents)",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#475569",
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "#F8FAFC",
                "edgecolor": "#CBD5E1",
                "alpha": 0.95,
            },
        )

    axis.set_ylabel(_y_axis_label(args.value_mode))
    axis.set_xlabel("Time From Run Start (minutes)")
    axis.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
    axis.set_axisbelow(True)
    axis.margins(x=0.0)

    analysis_start_s = _float_or_none(payload.get("analysis_window_start_s"))
    analysis_end_s = _float_or_none(payload.get("analysis_window_end_s"))
    if analysis_start_s is not None and analysis_end_s is not None:
        axis.set_xlim(analysis_start_s, analysis_end_s)

    axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    axis.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value / 60.0:.0f}")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
