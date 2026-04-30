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


def _import_matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import patches as mpatches
        from matplotlib import ticker
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to render this figure. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt, ticker, mpatches


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
        default="",
        help="Figure title. Default: no title.",
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
        default=5.67,
        help="Figure height in inches (default: 5.67).",
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
        return "Context Usage Integral\nin Window (token-s)"
    return "Average Context Usage\nin Window (tokens)"


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
    agents_by_index: dict[int, dict[str, Any]] = {}
    for raw_agent in raw_agents:
        if not isinstance(raw_agent, dict):
            continue
        agent_index = _int_or_none(raw_agent.get("agent_index"))
        if agent_index is None:
            continue
        color_hex = raw_agent.get("color_hex")
        if not isinstance(color_hex, str) or not color_hex:
            continue
        stack_rank = _int_or_none(raw_agent.get("stack_rank"))
        agent = {
            "agent_index": agent_index,
            "agent_label": raw_agent.get("agent_label") or f"A{agent_index + 1}",
            "agent_key": raw_agent.get("agent_key"),
            "color_hex": color_hex,
            "stack_rank": agent_index if stack_rank is None else stack_rank,
        }
        agents.append(agent)
        agents_by_index[agent_index] = agent
    if not agents:
        raise ValueError(f"Materialized payload has no usable agents: {input_path}")

    agents.sort(key=lambda item: item["agent_index"])
    agent_count = len(agents)
    window_count = len(raw_windows)
    value_field_name = _value_field_name(args.value_mode)

    x_starts: list[float] = []
    widths: list[float] = []
    parsed_windows: list[dict[str, Any]] = []
    for window_position, raw_window in enumerate(raw_windows):
        if not isinstance(raw_window, dict):
            raise ValueError(f"Window {window_position} is not a JSON object")
        window_start_s = _float_or_none(raw_window.get("window_start_s"))
        window_duration_s = _float_or_none(raw_window.get("window_duration_s"))
        if window_start_s is None or window_duration_s is None or window_duration_s <= 0.0:
            raise ValueError(f"Window {window_position} is missing valid timing fields")
        x_starts.append(window_start_s)
        widths.append(window_duration_s)

        parsed_contributions: list[dict[str, Any]] = []
        contributions = raw_window.get("contributions")
        if isinstance(contributions, list):
            for contribution in contributions:
                if not isinstance(contribution, dict):
                    continue
                agent_index = _int_or_none(contribution.get("agent_index"))
                value = _float_or_none(contribution.get(value_field_name))
                if agent_index is None or value is None or value <= 0.0:
                    continue
                agent = agents_by_index.get(agent_index)
                if agent is None:
                    continue
                parsed_contributions.append(
                    {
                        "agent_index": agent_index,
                        "value": value,
                        "stack_rank": int(agent["stack_rank"]),
                    }
                )
        parsed_contributions.sort(key=lambda item: (item["stack_rank"], item["agent_index"]))
        parsed_windows.append(
            {
                "window_start_s": window_start_s,
                "window_duration_s": window_duration_s,
                "contributions": parsed_contributions,
            }
        )

    plt, ticker, mpatches = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 20.0,
            "axes.titlesize": 24.0,
            "axes.labelsize": 21.0,
            "xtick.labelsize": 19.0,
            "ytick.labelsize": 19.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    figure, axis = plt.subplots(figsize=(args.figure_width, args.figure_height))
    left_spine = axis.spines["left"]
    for spine_name in ("bottom", "top", "right"):
        axis.spines[spine_name].set_visible(True)
        axis.spines[spine_name].set_linewidth(left_spine.get_linewidth())
        axis.spines[spine_name].set_color(left_spine.get_edgecolor())
    for window in parsed_windows:
        bottom = 0.0
        for contribution in window["contributions"]:
            agent = agents_by_index[int(contribution["agent_index"])]
            axis.bar(
                [window["window_start_s"]],
                [contribution["value"]],
                width=[window["window_duration_s"]],
                bottom=[bottom],
                align="edge",
                color=str(agent["color_hex"]),
                linewidth=0.0,
            )
            bottom += float(contribution["value"])

    if args.title:
        axis.set_title(args.title, loc="left", fontweight="semibold")

    legend_should_show = False
    if args.legend == "show":
        legend_should_show = True
    elif args.legend == "auto":
        legend_should_show = agent_count <= args.legend_max_agents

    if legend_should_show:
        legend_columns = max(1, min(4, math.ceil(agent_count / 12)))
        legend_agents = sorted(agents, key=lambda item: (int(item["stack_rank"]), item["agent_index"]))
        axis.legend(
            [
                mpatches.Patch(facecolor=str(agent["color_hex"]), edgecolor="none")
                for agent in legend_agents
            ],
            [str(agent["agent_label"]) for agent in legend_agents],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            ncols=legend_columns,
            title="Agents",
            fontsize=13.5,
            title_fontsize=14.5,
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
    axis.yaxis.set_major_locator(ticker.MultipleLocator(base=50000.0))
    axis.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda value, _: "0" if abs(value) < 1e-9 else f"{value / 1000.0:.0f}k"
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
