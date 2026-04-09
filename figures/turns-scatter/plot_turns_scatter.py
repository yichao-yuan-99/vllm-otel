#!/usr/bin/env python3
"""Render the turns-scatter figure from materialized turn-count distributions."""

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
            "Render the turns-scatter figure from the JSON output of "
            "materialize_turns_scatter.py."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_turns_scatter.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: figures/turns-scatter/output/<input-stem>.pdf",
    )
    parser.add_argument(
        "--y-scale",
        choices=("log", "linear"),
        default="log",
        help="Y-axis scale. Default: log.",
    )
    parser.add_argument(
        "--title",
        default="Turn Count Distribution by Benchmark and Agent",
        help="Figure title.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=14.0,
        help="Figure width in inches (default: 14.0).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=7.4,
        help="Figure height in inches (default: 7.4).",
    )
    parser.add_argument(
        "--jitter-width",
        type=float,
        default=0.20,
        help="Half-width of horizontal point jitter around each boxplot center.",
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
    jitter_width: float,
) -> list[float]:
    generator = random.Random(_stable_seed(label))
    return [
        center + generator.uniform(-jitter_width, jitter_width)
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
        return f"{int(value):,}"
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


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

    panels: list[dict[str, Any]] = []
    for fallback_index, raw_panel in enumerate(raw_panels):
        if not isinstance(raw_panel, dict):
            continue
        raw_turns = raw_panel.get("turns")
        if not isinstance(raw_turns, list):
            continue
        turns = [
            parsed
            for parsed in (_float_or_none(value) for value in raw_turns)
            if parsed is not None
        ]
        if not turns:
            continue

        panel_index = _int_or_none(raw_panel.get("panel_index"))
        benchmark = _string_or_none(raw_panel.get("benchmark")) or f"benchmark-{fallback_index + 1}"
        benchmark_label = (
            _string_or_none(raw_panel.get("benchmark_label"))
            or benchmark
        )
        agent_type = _string_or_none(raw_panel.get("agent_type")) or f"agent-{fallback_index + 1}"
        agent_label = _string_or_none(raw_panel.get("agent_label")) or agent_type
        stats = raw_panel.get("stats") if isinstance(raw_panel.get("stats"), dict) else {}
        panels.append(
            {
                "panel_index": panel_index if panel_index is not None else fallback_index,
                "benchmark": benchmark,
                "benchmark_label": benchmark_label,
                "agent_type": agent_type,
                "agent_label": agent_label,
                "score": _float_or_none(raw_panel.get("score")),
                "turns": turns,
                "stats": stats,
                "color": _panel_color(agent_type, agent_label, fallback_index),
            }
        )

    if not panels:
        raise ValueError(f"Materialized payload has no usable panels: {input_path}")

    panels.sort(key=lambda item: item["panel_index"])
    min_turn = min(min(panel["turns"]) for panel in panels)
    max_turn = max(max(panel["turns"]) for panel in panels)
    if args.y_scale == "log" and min_turn <= 0.0:
        raise ValueError(
            "Log y-scale requires strictly positive turn counts. "
            "Use --y-scale linear if your input contains zeros."
        )

    x_positions: list[float] = []
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
        x_positions.append(current_position)
        benchmark_labels[benchmark] = str(panel["benchmark_label"])
        group_members.setdefault(benchmark, []).append(current_position)
        previous_benchmark = benchmark
        previous_position = current_position
        current_position += 1.0

    for benchmark, positions in group_members.items():
        benchmark_centers[benchmark] = sum(positions) / len(positions)

    plt, ticker, transforms = _import_matplotlib()
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axis = plt.subplots(figsize=(args.figure_width, args.figure_height))
    figure.subplots_adjust(left=0.08, right=0.96, top=0.83, bottom=0.25)
    axis.set_axisbelow(True)

    for separator in separators:
        axis.axvline(separator, color="#dddddd", linewidth=1.0, zorder=0)

    for x_position, panel in zip(x_positions, panels):
        turns = list(panel["turns"])
        color = str(panel["color"])
        axis.boxplot(
            [turns],
            positions=[x_position],
            widths=0.62,
            showfliers=False,
            patch_artist=True,
            boxprops={
                "facecolor": "white",
                "edgecolor": "#454545",
                "linewidth": 1.5,
            },
            whiskerprops={"color": "#454545", "linewidth": 1.4},
            capprops={"color": "#454545", "linewidth": 1.4},
            medianprops={"color": "#333333", "linewidth": 2.0},
        )
        axis.scatter(
            _jittered_positions(
                label=f"{panel['benchmark']}::{panel['agent_type']}",
                center=x_position,
                count=len(turns),
                jitter_width=args.jitter_width,
            ),
            turns,
            s=24,
            c=color,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

        stats = panel["stats"]
        median = _float_or_none(stats.get("median")) if isinstance(stats, dict) else None
        sample_count = _int_or_none(stats.get("sample_count")) if isinstance(stats, dict) else None
        annotation_parts: list[str] = []
        if sample_count is not None:
            annotation_parts.append(f"n={sample_count}")
        if median is not None:
            annotation_parts.append(f"med={median:g}")
        if annotation_parts:
            blended = transforms.blended_transform_factory(axis.transData, axis.transAxes)
            axis.text(
                x_position,
                1.015,
                " | ".join(annotation_parts),
                transform=blended,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#5a5a5a",
            )

    if args.y_scale == "log":
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(_format_turn_tick))
        axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

    axis.grid(axis="y", which="major", color="#d7d7d7", linewidth=1.0)
    axis.grid(axis="y", which="minor", color="#ececec", linewidth=0.6, alpha=0.6)
    axis.set_xlim(min(x_positions) - 0.8, max(x_positions) + 0.8)
    axis.set_ylim(
        min_turn * (0.85 if args.y_scale == "linear" else 0.9),
        max_turn * (1.18 if args.y_scale == "linear" else 1.30),
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([str(panel["agent_label"]) for panel in panels])
    axis.tick_params(axis="x", length=0, pad=8)
    axis.set_ylabel("Turns per job")

    group_transform = transforms.blended_transform_factory(axis.transData, axis.transAxes)
    for benchmark, center in benchmark_centers.items():
        axis.text(
            center,
            -0.16,
            benchmark_labels[benchmark],
            transform=group_transform,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="semibold",
            color="#333333",
        )

    axis.set_title(args.title, loc="left", fontweight="semibold", pad=26.0)
    total_job_count = _int_or_none(payload.get("total_job_count")) or sum(
        len(panel["turns"]) for panel in panels
    )
    subtitle_parts = [
        f"{len(panels)} columns",
        f"{total_job_count} jobs",
        "box = median and IQR",
        "points = per-job turns",
        f"y-scale = {args.y_scale}",
    ]
    axis.text(
        0.0,
        1.085,
        " | ".join(subtitle_parts),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#5a5a5a",
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
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=_panel_color(agent_type, agent_label, index),
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=agent_label,
        )
        for index, (agent_type, agent_label) in enumerate(unique_agents)
    ]
    if legend_handles:
        axis.legend(
            handles=legend_handles,
            title="Agent",
            loc="upper right",
            frameon=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
