#!/usr/bin/env python3
"""Materialize fixed-width per-agent stacked context windows for plotting."""

from __future__ import annotations

import argparse
import colorsys
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_STEM = "stacked-per-agent"
DEFAULT_INPUT_REL_PATH = Path("post-processed/gateway/stack-context/context-usage-ranges.json")


@dataclass(frozen=True)
class AgentStats:
    agent_key: str
    gateway_run_id: str | None
    gateway_profile_id: int | None
    first_active_s: float
    last_active_end_s: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize fixed-width context-usage windows for the stacked-per-agent figure. "
            "Each output window stores sparse per-agent contributions for one contiguous bar."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Run directory that contains "
            "post-processed/gateway/stack-context/context-usage-ranges.json."
        ),
    )
    target_group.add_argument(
        "--input",
        default=None,
        help="Path to context-usage-ranges.json.",
    )
    parser.add_argument(
        "--window-size-s",
        type=float,
        default=120.0,
        help="Bar width in seconds (default: 120).",
    )
    parser.add_argument(
        "--start-s",
        type=float,
        default=0.0,
        help="Optional analysis window start in seconds from run start (default: 0).",
    )
    parser.add_argument(
        "--end-s",
        type=float,
        default=None,
        help=(
            "Optional analysis window end in seconds from run start. "
            "Default: the latest range end in the input."
        ),
    )
    parser.add_argument(
        "--agent-order",
        choices=("first-active", "agent-key"),
        default="first-active",
        help="Stable ordering used for stacked layers and colors (default: first-active).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: "
            "figures/stacked-per-agent/data/"
            "stacked-per-agent.window-<window>.start-<start>.end-<end|full>.json"
        ),
    )
    return parser.parse_args()


def _default_input_path_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_INPUT_REL_PATH).resolve()


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


def _format_label(value: float | None, *, default: str) -> str:
    if value is None:
        return default
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _default_output_path(
    *,
    window_size_s: float,
    start_s: float,
    end_s: float | None,
) -> Path:
    window_label = _format_label(window_size_s, default="120")
    start_label = _format_label(start_s, default="0")
    end_label = _format_label(end_s, default="full")
    return (
        DEFAULT_OUTPUT_DIR
        / (
            f"{DEFAULT_OUTPUT_STEM}.window-{window_label}s."
            f"start-{start_label}.end-{end_label}.json"
        )
    ).resolve()


def _materialize_agent_color(agent_index: int) -> str:
    golden_ratio_conjugate = 0.618033988749895
    hue = (0.08 + (agent_index * golden_ratio_conjugate)) % 1.0
    saturation_values = (0.72, 0.58, 0.82)
    value_values = (0.88, 0.76)
    saturation = saturation_values[agent_index % len(saturation_values)]
    value = value_values[(agent_index // len(saturation_values)) % len(value_values)]
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(red * 255.0)),
        int(round(green * 255.0)),
        int(round(blue * 255.0)),
    )


def _load_range_entries(
    input_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {input_path}")

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Missing 'entries' list in {input_path}")

    entries = [entry for entry in raw_entries if isinstance(entry, dict)]
    if not entries:
        raise ValueError(f"No usable entries were found in {input_path}")
    return payload, entries


def _active_segments(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("segment_type") != "active":
            continue
        agent_key = entry.get("agent_key")
        if not isinstance(agent_key, str) or not agent_key:
            continue

        range_start_s = _float_or_none(entry.get("range_start_s"))
        range_end_s = _float_or_none(entry.get("range_end_s"))
        avg_value_per_s = _float_or_none(entry.get("avg_value_per_s"))
        if range_start_s is None or range_end_s is None or avg_value_per_s is None:
            continue
        if range_end_s <= range_start_s:
            continue
        if avg_value_per_s <= 0.0:
            continue

        segments.append(
            {
                "agent_key": agent_key,
                "gateway_run_id": (
                    entry.get("gateway_run_id")
                    if isinstance(entry.get("gateway_run_id"), str)
                    else None
                ),
                "gateway_profile_id": _int_or_none(entry.get("gateway_profile_id")),
                "range_start_s": range_start_s,
                "range_end_s": range_end_s,
                "avg_value_per_s": avg_value_per_s,
            }
        )

    if not segments:
        raise ValueError("No active context segments were found in the input ranges JSON")
    return segments


def _discover_analysis_end_s(
    *,
    entries: list[dict[str, Any]],
    active_segments: list[dict[str, Any]],
) -> float:
    max_end_s = 0.0
    for entry in entries:
        range_end_s = _float_or_none(entry.get("range_end_s"))
        if range_end_s is not None:
            max_end_s = max(max_end_s, range_end_s)
    for segment in active_segments:
        max_end_s = max(max_end_s, float(segment["range_end_s"]))
    return max_end_s


def _clip_range(
    *,
    range_start_s: float,
    range_end_s: float,
    analysis_start_s: float,
    analysis_end_s: float,
) -> tuple[float, float] | None:
    clipped_start_s = max(range_start_s, analysis_start_s)
    clipped_end_s = min(range_end_s, analysis_end_s)
    if clipped_end_s <= clipped_start_s:
        return None
    return (clipped_start_s, clipped_end_s)


def _build_full_agent_stats(active_segments: list[dict[str, Any]]) -> dict[str, AgentStats]:
    first_start_by_agent: dict[str, float] = {}
    last_end_by_agent: dict[str, float] = {}
    gateway_run_id_by_agent: dict[str, str | None] = {}
    gateway_profile_id_by_agent: dict[str, int | None] = {}

    for segment in active_segments:
        agent_key = str(segment["agent_key"])
        range_start_s = float(segment["range_start_s"])
        range_end_s = float(segment["range_end_s"])

        if agent_key not in first_start_by_agent or range_start_s < first_start_by_agent[agent_key]:
            first_start_by_agent[agent_key] = range_start_s
        if agent_key not in last_end_by_agent or range_end_s > last_end_by_agent[agent_key]:
            last_end_by_agent[agent_key] = range_end_s

        gateway_run_id_by_agent.setdefault(agent_key, segment.get("gateway_run_id"))
        gateway_profile_id_by_agent.setdefault(agent_key, segment.get("gateway_profile_id"))

    return {
        agent_key: AgentStats(
            agent_key=agent_key,
            gateway_run_id=gateway_run_id_by_agent.get(agent_key),
            gateway_profile_id=gateway_profile_id_by_agent.get(agent_key),
            first_active_s=first_start_by_agent[agent_key],
            last_active_end_s=last_end_by_agent[agent_key],
        )
        for agent_key in first_start_by_agent
    }


def _selected_integral_by_agent(
    *,
    active_segments: list[dict[str, Any]],
    analysis_start_s: float,
    analysis_end_s: float,
) -> dict[str, float]:
    selected_integral_by_agent_key: dict[str, float] = {}
    for segment in active_segments:
        clipped = _clip_range(
            range_start_s=float(segment["range_start_s"]),
            range_end_s=float(segment["range_end_s"]),
            analysis_start_s=analysis_start_s,
            analysis_end_s=analysis_end_s,
        )
        if clipped is None:
            continue
        clipped_start_s, clipped_end_s = clipped
        contribution = float(segment["avg_value_per_s"]) * (clipped_end_s - clipped_start_s)
        if contribution <= 0.0:
            continue
        agent_key = str(segment["agent_key"])
        selected_integral_by_agent_key[agent_key] = (
            selected_integral_by_agent_key.get(agent_key, 0.0) + contribution
        )
    return selected_integral_by_agent_key


def _sort_agent_stats(
    *,
    agent_stats_by_key: dict[str, AgentStats],
    visible_agent_keys: set[str],
    agent_order: str,
) -> list[AgentStats]:
    visible_agents = [agent_stats_by_key[agent_key] for agent_key in visible_agent_keys]
    if agent_order == "agent-key":
        visible_agents.sort(key=lambda item: item.agent_key)
        return visible_agents
    visible_agents.sort(key=lambda item: (item.first_active_s, item.agent_key))
    return visible_agents


def _build_windows(
    *,
    analysis_start_s: float,
    analysis_end_s: float,
    window_size_s: float,
) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    window_index = 0
    current_start_s = analysis_start_s
    while current_start_s < analysis_end_s:
        current_end_s = min(current_start_s + window_size_s, analysis_end_s)
        windows.append(
            {
                "window_index": float(window_index),
                "window_start_s": current_start_s,
                "window_end_s": current_end_s,
                "window_duration_s": current_end_s - current_start_s,
            }
        )
        current_start_s = current_end_s
        window_index += 1
    return windows


def materialize_windows(
    *,
    input_path: Path,
    window_size_s: float,
    analysis_start_s: float,
    analysis_end_s: float | None,
    agent_order: str,
) -> dict[str, Any]:
    payload, entries = _load_range_entries(input_path)
    active_segments = _active_segments(entries)

    resolved_analysis_end_s = (
        _discover_analysis_end_s(entries=entries, active_segments=active_segments)
        if analysis_end_s is None
        else analysis_end_s
    )
    if window_size_s <= 0.0:
        raise ValueError(f"window_size_s must be positive: {window_size_s}")
    if analysis_start_s < 0.0:
        raise ValueError(f"analysis_start_s must be non-negative: {analysis_start_s}")
    if resolved_analysis_end_s <= analysis_start_s:
        raise ValueError(
            "analysis window end must be greater than start: "
            f"start={analysis_start_s}, end={resolved_analysis_end_s}"
        )

    agent_stats_by_key = _build_full_agent_stats(active_segments)
    selected_integral_by_agent_key = _selected_integral_by_agent(
        active_segments=active_segments,
        analysis_start_s=analysis_start_s,
        analysis_end_s=resolved_analysis_end_s,
    )
    visible_agent_keys = {
        agent_key
        for agent_key, total_integral in selected_integral_by_agent_key.items()
        if total_integral > 0.0
    }
    if not visible_agent_keys:
        raise ValueError("No active per-agent context usage overlaps the selected analysis window")

    ordered_agents = _sort_agent_stats(
        agent_stats_by_key=agent_stats_by_key,
        visible_agent_keys=visible_agent_keys,
        agent_order=agent_order,
    )
    windows = _build_windows(
        analysis_start_s=analysis_start_s,
        analysis_end_s=resolved_analysis_end_s,
        window_size_s=window_size_s,
    )
    if not windows:
        raise ValueError("The selected analysis window produced no bars")

    agent_count = len(ordered_agents)
    label_width = max(3, len(str(agent_count)))
    agent_index_by_key: dict[str, int] = {}
    agents_payload: list[dict[str, Any]] = []
    for agent_index, agent in enumerate(ordered_agents):
        agent_index_by_key[agent.agent_key] = agent_index
        agents_payload.append(
            {
                "agent_index": agent_index,
                "agent_label": f"A{agent_index + 1:0{label_width}d}",
                "agent_key": agent.agent_key,
                "gateway_run_id": agent.gateway_run_id,
                "gateway_profile_id": agent.gateway_profile_id,
                "first_active_s": round(agent.first_active_s, 6),
                "last_active_end_s": round(agent.last_active_end_s, 6),
                "analysis_total_integral_value": round(
                    selected_integral_by_agent_key.get(agent.agent_key, 0.0),
                    6,
                ),
                "analysis_average_value": round(
                    selected_integral_by_agent_key.get(agent.agent_key, 0.0)
                    / (resolved_analysis_end_s - analysis_start_s),
                    6,
                ),
                "color_hex": _materialize_agent_color(agent_index),
            }
        )

    sparse_integrals_by_window: list[dict[int, float]] = [dict() for _ in windows]
    for segment in active_segments:
        agent_key = str(segment["agent_key"])
        agent_index = agent_index_by_key.get(agent_key)
        if agent_index is None:
            continue

        clipped = _clip_range(
            range_start_s=float(segment["range_start_s"]),
            range_end_s=float(segment["range_end_s"]),
            analysis_start_s=analysis_start_s,
            analysis_end_s=resolved_analysis_end_s,
        )
        if clipped is None:
            continue

        segment_start_s, segment_end_s = clipped
        avg_value_per_s = float(segment["avg_value_per_s"])
        cursor_s = segment_start_s
        while cursor_s < segment_end_s:
            window_index = min(
                len(windows) - 1,
                int(math.floor((cursor_s - analysis_start_s) / window_size_s)),
            )
            window = windows[window_index]
            window_end_s = float(window["window_end_s"])
            overlap_end_s = min(segment_end_s, window_end_s)
            overlap_duration_s = overlap_end_s - cursor_s
            if overlap_duration_s <= 0.0:
                break

            sparse_integrals_by_window[window_index][agent_index] = (
                sparse_integrals_by_window[window_index].get(agent_index, 0.0)
                + (avg_value_per_s * overlap_duration_s)
            )
            cursor_s = overlap_end_s

    windows_payload: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        window_start_s = float(window["window_start_s"])
        window_end_s = float(window["window_end_s"])
        window_duration_s = float(window["window_duration_s"])
        sparse_entries = sparse_integrals_by_window[window_index]

        contributions_payload: list[dict[str, Any]] = []
        total_integral_value = 0.0
        for agent_index in sorted(sparse_entries):
            integral_value = sparse_entries[agent_index]
            if integral_value <= 0.0:
                continue
            total_integral_value += integral_value
            contributions_payload.append(
                {
                    "agent_index": agent_index,
                    "agent_label": agents_payload[agent_index]["agent_label"],
                    "agent_key": agents_payload[agent_index]["agent_key"],
                    "integral_value": round(integral_value, 6),
                    "average_value": round(integral_value / window_duration_s, 6),
                }
            )

        windows_payload.append(
            {
                "window_index": window_index,
                "window_start_s": round(window_start_s, 6),
                "window_end_s": round(window_end_s, 6),
                "window_duration_s": round(window_duration_s, 6),
                "active_agent_count": len(contributions_payload),
                "total_integral_value": round(total_integral_value, 6),
                "total_average_value": round(total_integral_value / window_duration_s, 6),
                "contributions": contributions_payload,
            }
        )

    return {
        "source_run_dir": payload.get("source_run_dir"),
        "source_gateway_output_dir": payload.get("source_gateway_output_dir"),
        "source_ranges_path": str(input_path),
        "input_entry_count": len(entries),
        "active_entry_count": len(active_segments),
        "metric": payload.get("metric", "context_usage_tokens"),
        "phase": payload.get("phase", "context"),
        "agent_order": agent_order,
        "window_size_s": round(window_size_s, 6),
        "analysis_window_start_s": round(analysis_start_s, 6),
        "analysis_window_end_s": round(resolved_analysis_end_s, 6),
        "analysis_window_duration_s": round(resolved_analysis_end_s - analysis_start_s, 6),
        "agent_count": len(agents_payload),
        "window_count": len(windows_payload),
        "agents": agents_payload,
        "windows": windows_payload,
    }


def main() -> int:
    args = _parse_args()
    input_path = (
        _default_input_path_for_run(Path(args.run_dir).expanduser().resolve())
        if args.run_dir is not None
        else Path(args.input).expanduser().resolve()
    )
    if not input_path.is_file():
        raise ValueError(f"Input ranges file not found: {input_path}")

    materialized_payload = materialize_windows(
        input_path=input_path,
        window_size_s=args.window_size_s,
        analysis_start_s=args.start_s,
        analysis_end_s=args.end_s,
        agent_order=args.agent_order,
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(
            window_size_s=args.window_size_s,
            start_s=args.start_s,
            end_s=args.end_s,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(materialized_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
