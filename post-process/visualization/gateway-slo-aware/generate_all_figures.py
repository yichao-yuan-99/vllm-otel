from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_INPUT_NAME = "slo-aware-events.json"
DEFAULT_INPUT_REL_PATH = Path("post-processed/gateway/slo-aware-log")
DEFAULT_OUTPUT_REL_PATH = Path("post-processed/visualization/gateway-slo-aware")
DEFAULT_MANIFEST_NAME = "figures-manifest.json"
DEFAULT_FIGURE_STEM = "slo-aware-events-timeline"
DEFAULT_STORED_THROUGHPUT_FIGURE_STEM = "slo-aware-stored-throughput"
DEFAULT_FORMAT = "png"
SUPPORTED_FORMATS = ("png", "pdf", "svg")
ENTER_EVENT_TYPE = "agent_entered_ralexation"
LEAVE_EVENT_TYPE = "agent_left_ralexation"


def _default_max_procs() -> int:
    max_procs_env = os.getenv("MAX_PROCS")
    if max_procs_env:
        try:
            parsed = int(max_procs_env)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        return 1
    return cpu_count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate gateway SLO-aware decision figures from extracted "
            "event summaries."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing post-processed/gateway/slo-aware-log/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/gateway/slo-aware-log/slo-aware-events.json will "
            "be processed."
        ),
    )
    parser.add_argument(
        "--slo-aware-input",
        default=None,
        help=(
            "Optional extracted gateway SLO-aware input path. Default: "
            "<run-dir>/post-processed/gateway/slo-aware-log/"
            f"{DEFAULT_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional figure output directory. Default: "
            "<run-dir>/post-processed/visualization/gateway-slo-aware/"
        ),
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        choices=SUPPORTED_FORMATS,
        help=f"Figure format. Default: {DEFAULT_FORMAT}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for raster output (default: 220).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered run directories and exit (only for --root-dir).",
    )
    parser.add_argument(
        "--max-procs",
        type=int,
        default=_default_max_procs(),
        help=(
            "Number of worker processes for --root-dir mode. "
            "Default: MAX_PROCS env var, else CPU count."
        ),
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_slo_aware_input_path_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_INPUT_REL_PATH / DEFAULT_INPUT_NAME).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_OUTPUT_REL_PATH).resolve()


def discover_run_dirs_with_gateway_slo_aware(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for input_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not input_path.is_file():
            continue
        if input_path.parent.name != "slo-aware-log":
            continue
        gateway_dir = input_path.parent.parent
        if gateway_dir.name != "gateway":
            continue
        post_processed_dir = gateway_dir.parent
        if post_processed_dir.name != "post-processed":
            continue
        run_dirs.add(post_processed_dir.parent.resolve())
    return sorted(run_dirs)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _non_empty_str_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _extract_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return []
    events: list[dict[str, Any]] = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        time_offset_s = _float_or_none(event.get("time_offset_s"))
        if time_offset_s is None:
            continue
        normalized = dict(event)
        normalized["time_offset_s"] = time_offset_s
        events.append(normalized)
    events.sort(
        key=lambda event: (
            _float_or_none(event.get("time_offset_s")) or 0.0,
            _non_empty_str_or_none(event.get("timestamp_utc")) or "",
            _non_empty_str_or_none(event.get("event_type")) or "",
            _non_empty_str_or_none(event.get("api_token_hash")) or "",
        )
    )
    return events


def _format_stat_value(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}{suffix}"


def _counts_summary(counts: Any) -> str:
    if not isinstance(counts, dict):
        return "n/a"
    pairs: list[str] = []
    for key in sorted(counts):
        value = counts.get(key)
        if not isinstance(key, str) or not key:
            continue
        if not isinstance(value, int):
            continue
        pairs.append(f"{key}={value}")
    return ", ".join(pairs) if pairs else "n/a"


def _build_stats_annotation(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"events: {_format_stat_value(_int_or_none(payload.get('slo_aware_event_count')))}",
            f"agents: {_format_stat_value(_int_or_none(payload.get('unique_agent_count')))}",
            f"target: {_format_stat_value(_float_or_none(payload.get('target_output_throughput_tokens_per_s')), ' tok/s')}",
            f"event throughput: {_format_stat_value(_float_or_none(payload.get('min_output_tokens_per_s_at_events')), ' tok/s')} to {_format_stat_value(_float_or_none(payload.get('max_output_tokens_per_s_at_events')), ' tok/s')}",
            f"slack max: {_format_stat_value(_float_or_none(payload.get('max_slo_slack_s')), ' s')}",
            f"ralexation max: {_format_stat_value(_float_or_none(payload.get('max_ralexation_duration_s')), ' s')}",
            f"wake reasons: {_counts_summary(payload.get('wake_reason_counts'))}",
        ]
    )


def _format_value_range(values: list[float], suffix: str = "") -> str:
    if not values:
        return "n/a"
    return f"{min(values):.6g}{suffix} to {max(values):.6g}{suffix}"


def _build_stored_throughput_annotation(
    *,
    payload: dict[str, Any],
    min_y: list[float],
    avg_y: list[float],
) -> str:
    return "\n".join(
        [
            f"events: {_format_stat_value(_int_or_none(payload.get('slo_aware_event_count')))}",
            f"target: {_format_stat_value(_float_or_none(payload.get('target_output_throughput_tokens_per_s')), ' tok/s')}",
            f"stored min range: {_format_value_range(min_y, ' tok/s')}",
            f"stored avg range: {_format_value_range(avg_y, ' tok/s')}",
        ]
    )


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate gateway SLO-aware figures. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt


def _apply_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _scatter_if_any(axis: Any, x_values: list[float], y_values: list[float], **kwargs: Any) -> None:
    if x_values and y_values and len(x_values) == len(y_values):
        axis.scatter(x_values, y_values, **kwargs)


def _extract_plot_series(events: list[dict[str, Any]]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {
        "all_x_values": [],
        "min_x": [],
        "min_y": [],
        "avg_x": [],
        "avg_y": [],
        "enter_x": [],
        "enter_throughput_y": [],
        "enter_slack_x": [],
        "enter_slack_y": [],
        "enter_duration_x": [],
        "enter_duration_y": [],
        "leave_ongoing_x": [],
        "leave_ongoing_throughput_y": [],
        "leave_ongoing_slack_x": [],
        "leave_ongoing_slack_y": [],
        "leave_pending_x": [],
        "leave_pending_throughput_y": [],
        "leave_pending_slack_x": [],
        "leave_pending_slack_y": [],
        "leave_other_x": [],
        "leave_other_throughput_y": [],
        "leave_other_slack_x": [],
        "leave_other_slack_y": [],
    }

    for event in events:
        time_offset_s = _float_or_none(event.get("time_offset_s"))
        if time_offset_s is None:
            continue
        series["all_x_values"].append(time_offset_s)

        min_output = _float_or_none(event.get("min_output_tokens_per_s"))
        if min_output is not None:
            series["min_x"].append(time_offset_s)
            series["min_y"].append(min_output)

        avg_output = _float_or_none(event.get("avg_output_tokens_per_s"))
        if avg_output is not None:
            series["avg_x"].append(time_offset_s)
            series["avg_y"].append(avg_output)

        event_type = _non_empty_str_or_none(event.get("event_type"))
        throughput = _float_or_none(event.get("output_tokens_per_s"))
        slack_s = _float_or_none(event.get("slo_slack_s"))
        duration_s = _float_or_none(event.get("ralexation_duration_s"))

        if event_type == ENTER_EVENT_TYPE:
            if throughput is not None:
                series["enter_x"].append(time_offset_s)
                series["enter_throughput_y"].append(throughput)
            if slack_s is not None:
                series["enter_slack_x"].append(time_offset_s)
                series["enter_slack_y"].append(slack_s)
            if duration_s is not None:
                series["enter_duration_x"].append(time_offset_s)
                series["enter_duration_y"].append(duration_s)
            continue

        if event_type == LEAVE_EVENT_TYPE:
            destination = _non_empty_str_or_none(event.get("to_schedule_state"))
            if destination == "ongoing":
                if throughput is not None:
                    series["leave_ongoing_x"].append(time_offset_s)
                    series["leave_ongoing_throughput_y"].append(throughput)
                if slack_s is not None:
                    series["leave_ongoing_slack_x"].append(time_offset_s)
                    series["leave_ongoing_slack_y"].append(slack_s)
                continue
            if destination == "pending":
                if throughput is not None:
                    series["leave_pending_x"].append(time_offset_s)
                    series["leave_pending_throughput_y"].append(throughput)
                if slack_s is not None:
                    series["leave_pending_slack_x"].append(time_offset_s)
                    series["leave_pending_slack_y"].append(slack_s)
                continue
            if throughput is not None:
                series["leave_other_x"].append(time_offset_s)
                series["leave_other_throughput_y"].append(throughput)
            if slack_s is not None:
                series["leave_other_slack_x"].append(time_offset_s)
                series["leave_other_slack_y"].append(slack_s)
    return series


def _set_x_limits(axis: Any, all_x_values: list[float]) -> None:
    if not all_x_values:
        return
    x_min = min(all_x_values)
    x_max = max(all_x_values)
    if x_min == x_max:
        padding = 1.0
        axis.set_xlim(x_min - padding, x_max + padding)
    else:
        axis.set_xlim(x_min, x_max)


def _build_subtitle_text(payload: dict[str, Any]) -> str | None:
    subtitle_parts: list[str] = []
    source_type = payload.get("source_type")
    if isinstance(source_type, str) and source_type:
        subtitle_parts.append(f"source: {source_type}")
    analysis_start = payload.get("analysis_window_start_utc")
    if isinstance(analysis_start, str) and analysis_start:
        subtitle_parts.append(f"start: {analysis_start}")
    log_paths = payload.get("source_slo_aware_log_paths")
    if isinstance(log_paths, list):
        subtitle_parts.append(f"logs: {len(log_paths)}")
    if not subtitle_parts:
        return None
    return " | ".join(subtitle_parts)


def _render_gateway_slo_aware_figure(
    *,
    slo_aware_payload: dict[str, Any],
    output_path: Path,
    image_format: str,
    dpi: int,
) -> bool:
    events = _extract_events(slo_aware_payload)
    if not events:
        return False

    plot_series = _extract_plot_series(events)
    all_x_values = plot_series["all_x_values"]
    if not all_x_values:
        return False

    target_output_throughput_tokens_per_s = _float_or_none(
        slo_aware_payload.get("target_output_throughput_tokens_per_s")
    )

    plt = _import_matplotlib_pyplot()
    _apply_plot_style(plt)

    figure, (throughput_axis, slack_axis) = plt.subplots(
        2,
        1,
        figsize=(11.6, 8.1),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.9], "hspace": 0.14},
    )

    if target_output_throughput_tokens_per_s is not None:
        throughput_axis.axhline(
            target_output_throughput_tokens_per_s,
            color="#0F766E",
            linestyle=(0, (6, 4)),
            linewidth=1.2,
            alpha=0.95,
            label="throughput target",
        )
    if plot_series["min_x"]:
        throughput_axis.plot(
            plot_series["min_x"],
            plot_series["min_y"],
            color="#334155",
            linestyle=(0, (3, 2)),
            linewidth=1.5,
            alpha=0.92,
            label="stored min throughput",
        )
    if plot_series["avg_x"]:
        throughput_axis.plot(
            plot_series["avg_x"],
            plot_series["avg_y"],
            color="#2563EB",
            linewidth=1.8,
            alpha=0.9,
            label="stored avg throughput",
        )
    _scatter_if_any(
        throughput_axis,
        plot_series["enter_x"],
        plot_series["enter_throughput_y"],
        color="#B91C1C",
        s=28,
        zorder=3,
        label="entered ralexation",
    )
    _scatter_if_any(
        throughput_axis,
        plot_series["leave_ongoing_x"],
        plot_series["leave_ongoing_throughput_y"],
        color="#059669",
        marker="^",
        s=32,
        zorder=3,
        label="left to ongoing",
    )
    _scatter_if_any(
        throughput_axis,
        plot_series["leave_pending_x"],
        plot_series["leave_pending_throughput_y"],
        color="#B45309",
        marker="s",
        s=28,
        zorder=3,
        label="left to pending",
    )
    _scatter_if_any(
        throughput_axis,
        plot_series["leave_other_x"],
        plot_series["leave_other_throughput_y"],
        facecolors="none",
        edgecolors="#475569",
        marker="D",
        s=28,
        linewidths=1.0,
        zorder=3,
        label="other exits",
    )
    throughput_axis.axvline(0.0, color="#334155", linestyle=":", linewidth=1.0, alpha=0.75)
    throughput_axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    throughput_axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    throughput_axis.minorticks_on()
    throughput_axis.set_title(
        "Gateway SLO-Aware Decisions",
        loc="left",
        fontweight="semibold",
    )
    throughput_axis.set_ylabel("Throughput (tok/s)")
    throughput_axis.legend(loc="upper left", frameon=False, ncols=2)

    subtitle_text = _build_subtitle_text(slo_aware_payload)
    if subtitle_text is not None:
        throughput_axis.text(
            0.0,
            1.02,
            subtitle_text,
            transform=throughput_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#2A3B47",
        )

    throughput_axis.text(
        0.99,
        0.98,
        _build_stats_annotation(slo_aware_payload),
        transform=throughput_axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#F7F9FC",
            "edgecolor": "#7A8B99",
            "alpha": 0.96,
        },
    )

    _scatter_if_any(
        slack_axis,
        plot_series["enter_slack_x"],
        plot_series["enter_slack_y"],
        color="#B91C1C",
        s=28,
        zorder=3,
        label="slack at enter",
    )
    _scatter_if_any(
        slack_axis,
        plot_series["leave_ongoing_slack_x"],
        plot_series["leave_ongoing_slack_y"],
        color="#059669",
        marker="^",
        s=32,
        zorder=3,
        label="slack at ongoing resume",
    )
    _scatter_if_any(
        slack_axis,
        plot_series["leave_pending_slack_x"],
        plot_series["leave_pending_slack_y"],
        color="#B45309",
        marker="s",
        s=28,
        zorder=3,
        label="slack at pending resume",
    )
    _scatter_if_any(
        slack_axis,
        plot_series["leave_other_slack_x"],
        plot_series["leave_other_slack_y"],
        facecolors="none",
        edgecolors="#475569",
        marker="D",
        s=28,
        linewidths=1.0,
        zorder=3,
        label="slack at other exit",
    )
    _scatter_if_any(
        slack_axis,
        plot_series["enter_duration_x"],
        plot_series["enter_duration_y"],
        facecolors="none",
        edgecolors="#1D4ED8",
        marker="o",
        s=28,
        linewidths=1.1,
        zorder=3,
        label="ralexation duration",
    )
    slack_axis.axvline(0.0, color="#334155", linestyle=":", linewidth=1.0, alpha=0.75)
    slack_axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    slack_axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    slack_axis.minorticks_on()
    slack_axis.set_xlabel("Time From Replay Start (s)")
    slack_axis.set_ylabel("Slack / Duration (s)")
    slack_axis.legend(loc="upper left", frameon=False, ncols=2)

    _set_x_limits(throughput_axis, all_x_values)
    figure.savefig(output_path, format=image_format, dpi=dpi)
    plt.close(figure)
    return True


def _render_gateway_stored_throughput_figure(
    *,
    slo_aware_payload: dict[str, Any],
    output_path: Path,
    image_format: str,
    dpi: int,
) -> bool:
    events = _extract_events(slo_aware_payload)
    if not events:
        return False

    plot_series = _extract_plot_series(events)
    all_x_values = plot_series["all_x_values"]
    if not all_x_values:
        return False
    if not plot_series["min_x"] and not plot_series["avg_x"]:
        return False

    plt = _import_matplotlib_pyplot()
    _apply_plot_style(plt)

    figure, axis = plt.subplots(figsize=(11.2, 5.2))

    if plot_series["min_x"]:
        axis.plot(
            plot_series["min_x"],
            plot_series["min_y"],
            color="#334155",
            linestyle=(0, (3, 2)),
            linewidth=1.6,
            alpha=0.94,
            label="stored min throughput",
        )
    if plot_series["avg_x"]:
        axis.plot(
            plot_series["avg_x"],
            plot_series["avg_y"],
            color="#2563EB",
            linewidth=1.9,
            alpha=0.92,
            label="stored avg throughput",
        )

    axis.axvline(0.0, color="#334155", linestyle=":", linewidth=1.0, alpha=0.75)
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    axis.minorticks_on()
    axis.set_title(
        "Gateway Stored Throughput",
        loc="left",
        fontweight="semibold",
    )
    axis.set_xlabel("Time From Replay Start (s)")
    axis.set_ylabel("Throughput (tok/s)")
    axis.legend(loc="upper left", frameon=False)

    subtitle_text = _build_subtitle_text(slo_aware_payload)
    if subtitle_text is not None:
        axis.text(
            0.0,
            1.02,
            subtitle_text,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#2A3B47",
        )

    axis.text(
        0.99,
        0.98,
        _build_stored_throughput_annotation(
            payload=slo_aware_payload,
            min_y=plot_series["min_y"],
            avg_y=plot_series["avg_y"],
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#F7F9FC",
            "edgecolor": "#7A8B99",
            "alpha": 0.96,
        },
    )

    _set_x_limits(axis, all_x_values)
    figure.tight_layout()
    figure.savefig(output_path, format=image_format, dpi=dpi)
    plt.close(figure)
    return True


def _build_manifest_figure_entry(
    *,
    figure_kind: str,
    figure_file_name: str,
    figure_path: Path,
    rendered: bool,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "figure_kind": figure_kind,
        "figure_generated": rendered,
        "figure_file_name": figure_file_name if rendered else None,
        "figure_path": str(figure_path.resolve()) if rendered else None,
        "skip_reason": None if rendered else skip_reason,
    }


def generate_figure_for_run_dir(
    run_dir: Path,
    *,
    slo_aware_input_path: Path | None = None,
    output_dir: Path | None = None,
    image_format: str = DEFAULT_FORMAT,
    dpi: int = 220,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_slo_aware_input_path = (
        slo_aware_input_path or _default_slo_aware_input_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    resolved_output_dir = (
        output_dir or _default_output_dir_for_run(resolved_run_dir)
    ).expanduser().resolve()

    if not resolved_slo_aware_input_path.is_file():
        raise ValueError(
            f"Missing gateway SLO-aware summary file: {resolved_slo_aware_input_path}"
        )
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer: {dpi}")

    slo_aware_payload = _load_json(resolved_slo_aware_input_path)
    if not isinstance(slo_aware_payload, dict):
        raise ValueError(
            "Gateway SLO-aware summary JSON must be an object: "
            f"{resolved_slo_aware_input_path}"
        )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    figure_specs = [
        {
            "figure_kind": "timeline",
            "figure_file_name": f"{DEFAULT_FIGURE_STEM}.{image_format}",
            "renderer": _render_gateway_slo_aware_figure,
            "skip_reason": "No valid SLO-aware events",
        },
        {
            "figure_kind": "stored_throughput",
            "figure_file_name": f"{DEFAULT_STORED_THROUGHPUT_FIGURE_STEM}.{image_format}",
            "renderer": _render_gateway_stored_throughput_figure,
            "skip_reason": "No stored throughput points",
        },
    ]

    figure_entries: list[dict[str, Any]] = []
    for figure_spec in figure_specs:
        figure_path = resolved_output_dir / figure_spec["figure_file_name"]
        rendered = figure_spec["renderer"](
            slo_aware_payload=slo_aware_payload,
            output_path=figure_path,
            image_format=image_format,
            dpi=dpi,
        )
        figure_entries.append(
            _build_manifest_figure_entry(
                figure_kind=str(figure_spec["figure_kind"]),
                figure_file_name=str(figure_spec["figure_file_name"]),
                figure_path=figure_path,
                rendered=rendered,
                skip_reason=str(figure_spec["skip_reason"]),
            )
        )

    primary_entry = figure_entries[0] if figure_entries else {}
    rendered_figure_count = sum(1 for entry in figure_entries if entry["figure_generated"])

    manifest = {
        "source_run_dir": str(resolved_run_dir),
        "source_slo_aware_summary_path": str(resolved_slo_aware_input_path),
        "output_dir": str(resolved_output_dir),
        "image_format": image_format,
        "dpi": dpi,
        "figure_count": rendered_figure_count,
        "figure_generated": bool(primary_entry.get("figure_generated", False)),
        "figure_file_name": primary_entry.get("figure_file_name"),
        "figure_path": primary_entry.get("figure_path"),
        "slo_aware_log_found": bool(slo_aware_payload.get("slo_aware_log_found", False)),
        "slo_aware_event_count": _int_or_none(slo_aware_payload.get("slo_aware_event_count")),
        "unique_agent_count": _int_or_none(slo_aware_payload.get("unique_agent_count")),
        "target_output_throughput_tokens_per_s": _float_or_none(
            slo_aware_payload.get("target_output_throughput_tokens_per_s")
        ),
        "event_type_counts": slo_aware_payload.get("event_type_counts"),
        "wake_reason_counts": slo_aware_payload.get("wake_reason_counts"),
        "resume_disposition_counts": slo_aware_payload.get("resume_disposition_counts"),
        "min_output_tokens_per_s_at_events": _float_or_none(
            slo_aware_payload.get("min_output_tokens_per_s_at_events")
        ),
        "max_output_tokens_per_s_at_events": _float_or_none(
            slo_aware_payload.get("max_output_tokens_per_s_at_events")
        ),
        "min_slo_slack_s": _float_or_none(slo_aware_payload.get("min_slo_slack_s")),
        "max_slo_slack_s": _float_or_none(slo_aware_payload.get("max_slo_slack_s")),
        "min_ralexation_duration_s": _float_or_none(
            slo_aware_payload.get("min_ralexation_duration_s")
        ),
        "max_ralexation_duration_s": _float_or_none(
            slo_aware_payload.get("max_ralexation_duration_s")
        ),
        "source_type": slo_aware_payload.get("source_type"),
        "service_failure_detected": bool(
            slo_aware_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": slo_aware_payload.get(
            "service_failure_cutoff_time_utc"
        ),
        "skip_reason": primary_entry.get("skip_reason"),
        "figures": figure_entries,
    }
    manifest_path = resolved_output_dir / DEFAULT_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _generate_run_dir_worker(task: tuple[str, str, int]) -> tuple[str, str | None, str | None]:
    run_dir_text, image_format, dpi = task
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = generate_figure_for_run_dir(
            run_dir,
            image_format=image_format,
            dpi=dpi,
        )
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(run_dirs: list[Path], *, image_format: str, dpi: int) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = generate_figure_for_run_dir(
                run_dir,
                image_format=image_format,
                dpi=dpi,
            )
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(
    run_dirs: list[Path],
    *,
    max_procs: int,
    image_format: str,
    dpi: int,
) -> int:
    failure_count = 0
    tasks = [(str(run_dir), image_format, dpi) for run_dir in run_dirs]
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_path_text, error_text in executor.map(
            _generate_run_dir_worker,
            tasks,
        ):
            if error_text is None:
                print(f"[done] {run_dir_text} -> {output_path_text}")
            else:
                failure_count += 1
                print(f"[error] {run_dir_text}: {error_text}", file=sys.stderr)
    return failure_count


def _main_run_dir(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")
    run_dir = Path(args.run_dir).expanduser().resolve()
    slo_aware_input_path = (
        Path(args.slo_aware_input).expanduser().resolve()
        if args.slo_aware_input
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_path = generate_figure_for_run_dir(
        run_dir,
        slo_aware_input_path=slo_aware_input_path,
        output_dir=output_dir,
        image_format=args.format,
        dpi=args.dpi,
    )
    print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.slo_aware_input:
        raise ValueError("--slo-aware-input can only be used with --run-dir")
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be a positive integer: {args.dpi}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_gateway_slo_aware(root_dir)
    print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
    if not run_dirs:
        return 0
    if args.dry_run:
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    worker_count = min(args.max_procs, len(run_dirs))
    print(f"Running visualization with {worker_count} worker process(es)")
    if worker_count <= 1:
        failure_count = _run_root_dir_sequential(
            run_dirs,
            image_format=args.format,
            dpi=args.dpi,
        )
    else:
        try:
            failure_count = _run_root_dir_parallel(
                run_dirs,
                max_procs=worker_count,
                image_format=args.format,
                dpi=args.dpi,
            )
        except (OSError, PermissionError) as exc:
            print(
                "[warn] Falling back to sequential visualization because the "
                f"process pool could not be created: {exc}",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(
                run_dirs,
                image_format=args.format,
                dpi=args.dpi,
            )

    return 0 if failure_count == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_dir is not None:
        return _main_run_dir(args)
    return _main_root_dir(args)


if __name__ == "__main__":
    raise SystemExit(main())
