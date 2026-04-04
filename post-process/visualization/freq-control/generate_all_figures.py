from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_INPUT_NAME = "freq-control-summary.json"
SEGMENTED_INPUT_DIR_NAME = "freq-control-seg"
LINESPACE_INPUT_DIR_NAME = "freq-control-linespace"
BASELINE_INPUT_DIR_NAME = "freq-control"
DEFAULT_MANIFEST_NAME = "figures-manifest.json"
DEFAULT_FIGURE_STEM = "freq-control-timeline"
DEFAULT_FORMAT = "png"
SUPPORTED_FORMATS = ("png", "pdf", "svg")


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
            "Generate freq-control timeline figures from extracted "
            "freq-controller query and decision summaries."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Run result root directory containing post-processed/freq-control/ "
            "or post-processed/freq-control-seg/ or post-processed/freq-control-linespace/."
        ),
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/freq-control/freq-control-summary.json or "
            "post-processed/freq-control-seg/freq-control-summary.json or "
            "post-processed/freq-control-linespace/freq-control-summary.json will be processed."
        ),
    )
    parser.add_argument(
        "--freq-control-input",
        default=None,
        help=(
            "Optional freq-control summary input path. Default: "
            f"<run-dir>/post-processed/freq-control-seg/{DEFAULT_INPUT_NAME} "
            f"if present, else <run-dir>/post-processed/freq-control-linespace/{DEFAULT_INPUT_NAME} "
            f"if present, else <run-dir>/post-processed/freq-control/{DEFAULT_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional figure output directory. Default: "
            "matches the detected summary family under "
            "<run-dir>/post-processed/visualization/."
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


def _candidate_freq_control_input_paths_for_run(run_dir: Path) -> list[Path]:
    return [
        (run_dir / "post-processed" / SEGMENTED_INPUT_DIR_NAME / DEFAULT_INPUT_NAME).resolve(),
        (run_dir / "post-processed" / LINESPACE_INPUT_DIR_NAME / DEFAULT_INPUT_NAME).resolve(),
        (run_dir / "post-processed" / BASELINE_INPUT_DIR_NAME / DEFAULT_INPUT_NAME).resolve(),
    ]


def _default_freq_control_input_path_for_run(run_dir: Path) -> Path:
    for candidate in _candidate_freq_control_input_paths_for_run(run_dir):
        if candidate.is_file():
            return candidate
    return _candidate_freq_control_input_paths_for_run(run_dir)[0]


def _summary_dir_name_for_input_path(freq_control_input_path: Path) -> str:
    parent_name = freq_control_input_path.resolve().parent.name
    if parent_name == SEGMENTED_INPUT_DIR_NAME:
        return SEGMENTED_INPUT_DIR_NAME
    if parent_name == LINESPACE_INPUT_DIR_NAME:
        return LINESPACE_INPUT_DIR_NAME
    return BASELINE_INPUT_DIR_NAME


def _default_output_dir_for_run(run_dir: Path, *, freq_control_input_path: Path) -> Path:
    summary_dir_name = _summary_dir_name_for_input_path(freq_control_input_path)
    return (run_dir / "post-processed" / "visualization" / summary_dir_name).resolve()


def discover_run_dirs_with_freq_control_summary(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for summary_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name not in {
            BASELINE_INPUT_DIR_NAME,
            SEGMENTED_INPUT_DIR_NAME,
            LINESPACE_INPUT_DIR_NAME,
        }:
            continue
        if summary_path.parent.parent.name != "post-processed":
            continue
        run_dirs.add(summary_path.parent.parent.parent.resolve())
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


def _format_stat_value(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}{suffix}"


def _extract_query_series(
    freq_control_payload: dict[str, Any],
) -> tuple[list[float], list[float], list[float], list[float]]:
    pending_pairs: list[tuple[float, float]] = []
    active_pairs: list[tuple[float, float]] = []
    raw_points = freq_control_payload.get("query_points")
    if not isinstance(raw_points, list):
        return [], [], [], []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        context_usage = _float_or_none(point.get("context_usage"))
        phase = point.get("phase")
        if time_offset_s is None or context_usage is None:
            continue
        if phase == "pending":
            pending_pairs.append((time_offset_s, context_usage))
        else:
            active_pairs.append((time_offset_s, context_usage))
    pending_pairs.sort()
    active_pairs.sort()
    return (
        [pair[0] for pair in pending_pairs],
        [pair[1] for pair in pending_pairs],
        [pair[0] for pair in active_pairs],
        [pair[1] for pair in active_pairs],
    )


def _extract_decision_series(
    freq_control_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    raw_points = freq_control_payload.get("decision_points")
    if not isinstance(raw_points, list):
        return [], [], []
    decisions: list[dict[str, Any]] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        if time_offset_s is None:
            continue
        normalized = dict(point)
        normalized["time_offset_s"] = time_offset_s
        decisions.append(normalized)
    decisions.sort(
        key=lambda point: (
            _float_or_none(point.get("time_offset_s")) or 0.0,
            point.get("timestamp_utc") or "",
        )
    )

    context_times: list[float] = []
    window_values: list[float] = []
    for point in decisions:
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        window_context_usage = _float_or_none(point.get("window_context_usage"))
        if time_offset_s is None or window_context_usage is None:
            continue
        context_times.append(time_offset_s)
        window_values.append(window_context_usage)
    return decisions, context_times, window_values


def _shorten_error_label(message: str, *, max_length: int = 56) -> str:
    compact = " ".join(message.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def _extract_query_error_series(
    freq_control_payload: dict[str, Any],
) -> tuple[list[str], list[float], list[int], list[float], list[int]]:
    raw_points = freq_control_payload.get("query_points")
    if not isinstance(raw_points, list):
        return [], [], [], [], []

    ordered_labels: list[str] = []
    counts_by_label: dict[str, int] = {}
    events: list[tuple[float, str, str]] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        error = _non_empty_str_or_none(point.get("error"))
        if time_offset_s is None or error is None:
            continue
        label = _shorten_error_label(error)
        phase = point.get("phase")
        normalized_phase = phase if phase == "pending" else "active"
        if label not in counts_by_label:
            ordered_labels.append(label)
            counts_by_label[label] = 0
        counts_by_label[label] += 1
        events.append((time_offset_s, label, normalized_phase))

    if not events:
        return [], [], [], [], []

    y_by_label = {label: index for index, label in enumerate(ordered_labels)}
    display_labels = [
        f"{label} ({counts_by_label[label]})" if counts_by_label[label] > 1 else label
        for label in ordered_labels
    ]
    pending_x: list[float] = []
    pending_y: list[int] = []
    active_x: list[float] = []
    active_y: list[int] = []
    for time_offset_s, label, phase in sorted(events):
        if phase == "pending":
            pending_x.append(time_offset_s)
            pending_y.append(y_by_label[label])
        else:
            active_x.append(time_offset_s)
            active_y.append(y_by_label[label])
    return display_labels, pending_x, pending_y, active_x, active_y


def _extract_control_error_series(
    freq_control_payload: dict[str, Any],
) -> tuple[list[str], list[float], list[int]]:
    raw_points = freq_control_payload.get("control_error_points")
    if not isinstance(raw_points, list):
        return [], [], []

    ordered_labels: list[str] = []
    counts_by_label: dict[str, int] = {}
    events: list[tuple[float, str]] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        error = _non_empty_str_or_none(point.get("error"))
        if time_offset_s is None or error is None:
            continue
        reason = _non_empty_str_or_none(point.get("reason"))
        action = _non_empty_str_or_none(point.get("action"))
        label_prefix_parts = [part for part in (reason, action) if part and part != "error"]
        label_prefix = " ".join(label_prefix_parts) if label_prefix_parts else "control"
        label = _shorten_error_label(f"{label_prefix}: {error}")
        if label not in counts_by_label:
            ordered_labels.append(label)
            counts_by_label[label] = 0
        counts_by_label[label] += 1
        events.append((time_offset_s, label))

    if not events:
        return [], [], []

    y_by_label = {label: index for index, label in enumerate(ordered_labels)}
    display_labels = [
        f"{label} ({counts_by_label[label]})" if counts_by_label[label] > 1 else label
        for label in ordered_labels
    ]
    control_x = [time_offset_s for time_offset_s, _ in sorted(events)]
    control_y = [y_by_label[label] for _, label in sorted(events)]
    return display_labels, control_x, control_y


def _extract_combined_failure_series(
    freq_control_payload: dict[str, Any],
) -> tuple[
    list[str],
    list[float],
    list[int],
    list[float],
    list[int],
    list[float],
    list[int],
]:
    raw_query_points = freq_control_payload.get("query_points")
    raw_control_points = freq_control_payload.get("control_error_points")
    query_points = raw_query_points if isinstance(raw_query_points, list) else []
    control_points = raw_control_points if isinstance(raw_control_points, list) else []

    ordered_labels: list[str] = []
    counts_by_label: dict[str, int] = {}
    pending_events: list[tuple[float, str]] = []
    active_events: list[tuple[float, str]] = []
    control_events: list[tuple[float, str]] = []

    def register_event(label: str) -> None:
        if label not in counts_by_label:
            ordered_labels.append(label)
            counts_by_label[label] = 0
        counts_by_label[label] += 1

    for point in query_points:
        if not isinstance(point, dict):
            continue
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        error = _non_empty_str_or_none(point.get("error"))
        if time_offset_s is None or error is None:
            continue
        label = _shorten_error_label(error)
        register_event(label)
        phase = point.get("phase")
        if phase == "pending":
            pending_events.append((time_offset_s, label))
        else:
            active_events.append((time_offset_s, label))

    for point in control_points:
        if not isinstance(point, dict):
            continue
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        error = _non_empty_str_or_none(point.get("error"))
        if time_offset_s is None or error is None:
            continue
        reason = _non_empty_str_or_none(point.get("reason"))
        action = _non_empty_str_or_none(point.get("action"))
        label_prefix_parts = [part for part in (reason, action) if part and part != "error"]
        label_prefix = " ".join(label_prefix_parts) if label_prefix_parts else "control"
        label = _shorten_error_label(f"{label_prefix}: {error}")
        register_event(label)
        control_events.append((time_offset_s, label))

    if not ordered_labels:
        return [], [], [], [], [], [], []

    y_by_label = {label: index for index, label in enumerate(ordered_labels)}
    display_labels = [
        f"{label} ({counts_by_label[label]})" if counts_by_label[label] > 1 else label
        for label in ordered_labels
    ]

    pending_sorted = sorted(pending_events)
    active_sorted = sorted(active_events)
    control_sorted = sorted(control_events)
    return (
        display_labels,
        [time_offset_s for time_offset_s, _ in pending_sorted],
        [y_by_label[label] for _, label in pending_sorted],
        [time_offset_s for time_offset_s, _ in active_sorted],
        [y_by_label[label] for _, label in active_sorted],
        [time_offset_s for time_offset_s, _ in control_sorted],
        [y_by_label[label] for _, label in control_sorted],
    )


def _extract_frequency_step_series(
    decisions: list[dict[str, Any]],
) -> tuple[list[float], list[int], list[float], list[int], list[float], list[int]]:
    if not decisions:
        return [], [], [], [], [], []

    step_x: list[float] = []
    step_y: list[int] = []
    changed_x: list[float] = []
    changed_y: list[int] = []
    hold_x: list[float] = []
    hold_y: list[int] = []

    first_time = _float_or_none(decisions[0].get("time_offset_s"))
    first_current = _int_or_none(decisions[0].get("current_frequency_mhz"))
    if first_time is not None and first_current is not None:
        step_x.append(min(0.0, first_time))
        step_y.append(first_current)

    for point in decisions:
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        target_frequency_mhz = _int_or_none(point.get("target_frequency_mhz"))
        changed = point.get("changed") is True
        if time_offset_s is None or target_frequency_mhz is None:
            continue
        step_x.append(time_offset_s)
        step_y.append(target_frequency_mhz)
        if changed:
            changed_x.append(time_offset_s)
            changed_y.append(target_frequency_mhz)
        else:
            hold_x.append(time_offset_s)
            hold_y.append(target_frequency_mhz)
    return step_x, step_y, changed_x, changed_y, hold_x, hold_y


def _build_stats_annotation(freq_control_payload: dict[str, Any]) -> str:
    stats_lines = [
        f"query samples: {_format_stat_value(_int_or_none(freq_control_payload.get('query_point_count')))}",
        f"pending query: {_format_stat_value(_int_or_none(freq_control_payload.get('pending_query_point_count')))}",
        f"query errors: {_format_stat_value(_int_or_none(freq_control_payload.get('query_error_count')))}",
        f"control errors: {_format_stat_value(_int_or_none(freq_control_payload.get('control_error_point_count')))}",
        f"decisions: {_format_stat_value(_int_or_none(freq_control_payload.get('decision_point_count')))}",
        f"freq changes: {_format_stat_value(_int_or_none(freq_control_payload.get('decision_change_count')))}",
        f"query max: {_format_stat_value(_float_or_none(freq_control_payload.get('max_context_usage')))}",
        f"window max: {_format_stat_value(_float_or_none(freq_control_payload.get('max_window_context_usage')))}",
        f"freq min: {_format_stat_value(_int_or_none(freq_control_payload.get('min_frequency_mhz')), ' MHz')}",
        f"freq max: {_format_stat_value(_int_or_none(freq_control_payload.get('max_frequency_mhz')), ' MHz')}",
    ]
    low_freq_threshold, low_freq_cap_mhz, segmented_policy_detected = (
        _extract_segmented_policy_markers(freq_control_payload)
    )
    if segmented_policy_detected:
        stats_lines.append(
            f"lfth: {_format_stat_value(low_freq_threshold)}"
        )
        stats_lines.append(
            f"low-freq cap: {_format_stat_value(low_freq_cap_mhz, ' MHz')}"
        )
    return "\n".join(stats_lines)


def _extract_segmented_policy_markers(
    freq_control_payload: dict[str, Any],
) -> tuple[float | None, int | None, bool]:
    low_freq_threshold = _float_or_none(freq_control_payload.get("low_freq_threshold"))
    low_freq_cap_mhz = _int_or_none(freq_control_payload.get("low_freq_cap_mhz"))
    segmented_policy_detected = bool(
        freq_control_payload.get("segmented_policy_detected", False)
    )
    raw_decisions = freq_control_payload.get("decision_points")
    if not isinstance(raw_decisions, list):
        raw_decisions = []
    for point in raw_decisions:
        if not isinstance(point, dict):
            continue
        if low_freq_threshold is None:
            low_freq_threshold = _float_or_none(point.get("low_freq_threshold"))
        if low_freq_cap_mhz is None:
            low_freq_cap_mhz = _int_or_none(point.get("low_freq_cap_mhz"))
    if low_freq_threshold is not None or low_freq_cap_mhz is not None:
        segmented_policy_detected = True
    return low_freq_threshold, low_freq_cap_mhz, segmented_policy_detected


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate freq-control figures. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt


def _render_freq_control_figure(
    *,
    freq_control_payload: dict[str, Any],
    output_path: Path,
    image_format: str,
    dpi: int,
) -> bool:
    pending_x, pending_y, active_x, active_y = _extract_query_series(freq_control_payload)
    decisions, decision_x, decision_window_y = _extract_decision_series(freq_control_payload)
    (
        error_labels,
        pending_error_x,
        pending_error_y,
        active_error_x,
        active_error_y,
        control_error_x,
        control_error_y,
    ) = _extract_combined_failure_series(freq_control_payload)
    step_x, step_y, changed_x, changed_y, hold_x, hold_y = _extract_frequency_step_series(
        decisions
    )

    all_x_values = (
        pending_x
        + active_x
        + decision_x
        + step_x
        + pending_error_x
        + active_error_x
        + control_error_x
    )
    if not all_x_values:
        return False

    lower_bound = _float_or_none(freq_control_payload.get("lower_bound"))
    upper_bound = _float_or_none(freq_control_payload.get("upper_bound"))
    (
        low_freq_threshold,
        low_freq_cap_mhz,
        segmented_policy_detected,
    ) = _extract_segmented_policy_markers(freq_control_payload)

    plt = _import_matplotlib_pyplot()
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

    figure, (context_axis, frequency_axis, failure_axis) = plt.subplots(
        3,
        1,
        figsize=(11.6, 9.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 2.0, 1.35], "hspace": 0.14},
    )

    x_min = min(all_x_values)
    x_max = max(all_x_values)

    if lower_bound is not None and upper_bound is not None and x_min < x_max:
        context_axis.fill_between(
            [x_min, x_max],
            [lower_bound, lower_bound],
            [upper_bound, upper_bound],
            color="#FEF3C7",
            alpha=0.28,
            zorder=0,
        )
    if lower_bound is not None:
        context_axis.axhline(
            lower_bound,
            color="#047857",
            linestyle=(0, (6, 4)),
            linewidth=1.1,
            alpha=0.9,
            label="lower bound",
        )
    if low_freq_threshold is not None:
        context_axis.axhline(
            low_freq_threshold,
            color="#BE123C",
            linestyle=(0, (2, 3)),
            linewidth=1.2,
            alpha=0.95,
            label="lfth",
        )
    if upper_bound is not None:
        context_axis.axhline(
            upper_bound,
            color="#B45309",
            linestyle=(0, (6, 4)),
            linewidth=1.1,
            alpha=0.9,
            label="upper bound",
        )
    if pending_x:
        context_axis.plot(
            pending_x,
            pending_y,
            color="#94A3B8",
            linewidth=1.5,
            linestyle=(0, (3, 3)),
            alpha=0.92,
            label="context query (pending)",
        )
    if active_x:
        context_axis.plot(
            active_x,
            active_y,
            color="#2563EB",
            linewidth=1.9,
            alpha=0.95,
            label="context query",
        )
    if decision_x:
        context_axis.plot(
            decision_x,
            decision_window_y,
            color="#D97706",
            linewidth=2.1,
            alpha=0.96,
            label="decision window",
        )
    context_axis.axvline(
        0.0,
        color="#334155",
        linestyle=":",
        linewidth=1.0,
        alpha=0.75,
    )
    context_axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    context_axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    context_axis.minorticks_on()
    context_axis.set_title(
        "Segmented Frequency Control Timeline"
        if segmented_policy_detected
        else "Frequency Control Timeline",
        loc="left",
        fontweight="semibold",
    )
    context_axis.set_ylabel("Context Tokens")
    context_axis.legend(loc="upper left", frameon=False, ncols=2)

    subtitle_parts: list[str] = []
    source_type = freq_control_payload.get("source_type")
    if isinstance(source_type, str) and source_type:
        subtitle_parts.append(f"source: {source_type}")
    analysis_start = freq_control_payload.get("analysis_window_start_utc")
    if isinstance(analysis_start, str) and analysis_start:
        subtitle_parts.append(f"start: {analysis_start}")
    if subtitle_parts:
        context_axis.text(
            0.0,
            1.02,
            " | ".join(subtitle_parts),
            transform=context_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#2A3B47",
        )

    context_axis.text(
        0.99,
        0.98,
        _build_stats_annotation(freq_control_payload),
        transform=context_axis.transAxes,
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

    if step_x and step_y:
        frequency_axis.step(
            step_x,
            step_y,
            where="post",
            color="#0F172A",
            linewidth=2.0,
            alpha=0.95,
            label="applied freq",
        )
    if low_freq_cap_mhz is not None:
        frequency_axis.axhline(
            low_freq_cap_mhz,
            color="#BE123C",
            linestyle=(0, (2, 3)),
            linewidth=1.2,
            alpha=0.95,
            label="low-freq cap",
        )
    if changed_x:
        frequency_axis.scatter(
            changed_x,
            changed_y,
            color="#059669",
            s=24,
            zorder=3,
            label="freq change",
        )
    if hold_x:
        frequency_axis.scatter(
            hold_x,
            hold_y,
            facecolors="none",
            edgecolors="#64748B",
            s=22,
            linewidths=1.0,
            zorder=3,
            label="hold",
        )
    frequency_axis.axvline(
        0.0,
        color="#334155",
        linestyle=":",
        linewidth=1.0,
        alpha=0.75,
    )
    frequency_axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    frequency_axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    frequency_axis.minorticks_on()
    frequency_axis.set_ylabel("GPU Core Frequency (MHz)")
    if step_x:
        frequency_axis.legend(loc="upper left", frameon=False, ncols=3)

    if pending_error_x:
        failure_axis.scatter(
            pending_error_x,
            pending_error_y,
            color="#B91C1C",
            marker="x",
            s=38,
            linewidths=1.4,
            zorder=3,
            label="pending read failure",
        )
    if active_error_x:
        failure_axis.scatter(
            active_error_x,
            active_error_y,
            color="#DC2626",
            marker="o",
            s=30,
            zorder=3,
            label="active read failure",
        )
    if control_error_x:
        failure_axis.scatter(
            control_error_x,
            control_error_y,
            color="#B45309",
            marker="D",
            s=34,
            zorder=3,
            label="control write failure",
        )
    failure_axis.axvline(
        0.0,
        color="#334155",
        linestyle=":",
        linewidth=1.0,
        alpha=0.75,
    )
    failure_axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    failure_axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    failure_axis.minorticks_on()
    failure_axis.set_title(
        "Gateway Read and Control Failures",
        loc="left",
        fontweight="semibold",
    )
    failure_axis.set_xlabel("Time From Replay Start (s)")
    failure_axis.set_ylabel("Failure Event")
    if error_labels:
        failure_axis.set_yticks(list(range(len(error_labels))))
        failure_axis.set_yticklabels(error_labels)
        failure_axis.set_ylim(-0.75, len(error_labels) - 0.25)
        if pending_error_x or active_error_x or control_error_x:
            failure_axis.legend(loc="upper left", frameon=False, ncols=3)
    else:
        failure_axis.set_yticks([])
        failure_axis.text(
            0.5,
            0.5,
            "No query read failures or control write failures",
            transform=failure_axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#475569",
        )

    figure.savefig(output_path, format=image_format, dpi=dpi)
    plt.close(figure)
    return True


def generate_figure_for_run_dir(
    run_dir: Path,
    *,
    freq_control_input_path: Path | None = None,
    output_dir: Path | None = None,
    image_format: str = DEFAULT_FORMAT,
    dpi: int = 220,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_freq_control_input_path = (
        freq_control_input_path or _default_freq_control_input_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    resolved_output_dir = (
        output_dir
        or _default_output_dir_for_run(
            resolved_run_dir,
            freq_control_input_path=resolved_freq_control_input_path,
        )
    ).expanduser().resolve()

    if not resolved_freq_control_input_path.is_file():
        raise ValueError(
            f"Missing freq-control summary file: {resolved_freq_control_input_path}"
        )
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer: {dpi}")

    freq_control_payload = _load_json(resolved_freq_control_input_path)
    if not isinstance(freq_control_payload, dict):
        raise ValueError(
            f"Freq-control summary JSON must be an object: {resolved_freq_control_input_path}"
        )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    figure_file_name = f"{DEFAULT_FIGURE_STEM}.{image_format}"
    figure_path = resolved_output_dir / figure_file_name
    rendered = _render_freq_control_figure(
        freq_control_payload=freq_control_payload,
        output_path=figure_path,
        image_format=image_format,
        dpi=dpi,
    )
    (
        low_freq_threshold,
        low_freq_cap_mhz,
        segmented_policy_detected,
    ) = _extract_segmented_policy_markers(freq_control_payload)

    manifest = {
        "source_run_dir": str(resolved_run_dir),
        "source_freq_control_summary_path": str(resolved_freq_control_input_path),
        "source_freq_control_summary_dir_name": _summary_dir_name_for_input_path(
            resolved_freq_control_input_path
        ),
        "output_dir": str(resolved_output_dir),
        "image_format": image_format,
        "dpi": dpi,
        "figure_count": 1 if rendered else 0,
        "figure_generated": rendered,
        "figure_file_name": figure_file_name if rendered else None,
        "figure_path": str(figure_path.resolve()) if rendered else None,
        "freq_control_log_found": bool(freq_control_payload.get("freq_control_log_found", False)),
        "query_log_found": bool(freq_control_payload.get("query_log_found", False)),
        "decision_log_found": bool(freq_control_payload.get("decision_log_found", False)),
        "control_error_log_found": bool(
            freq_control_payload.get("control_error_log_found", False)
        ),
        "query_point_count": _int_or_none(freq_control_payload.get("query_point_count")),
        "pending_query_point_count": _int_or_none(
            freq_control_payload.get("pending_query_point_count")
        ),
        "active_query_point_count": _int_or_none(
            freq_control_payload.get("active_query_point_count")
        ),
        "query_error_count": _int_or_none(freq_control_payload.get("query_error_count")),
        "control_error_point_count": _int_or_none(
            freq_control_payload.get("control_error_point_count")
        ),
        "decision_point_count": _int_or_none(freq_control_payload.get("decision_point_count")),
        "decision_change_count": _int_or_none(
            freq_control_payload.get("decision_change_count")
        ),
        "segmented_policy_detected": segmented_policy_detected,
        "low_freq_threshold": low_freq_threshold,
        "low_freq_cap_mhz": low_freq_cap_mhz,
        "max_context_usage": _float_or_none(freq_control_payload.get("max_context_usage")),
        "max_window_context_usage": _float_or_none(
            freq_control_payload.get("max_window_context_usage")
        ),
        "lower_bound": _float_or_none(freq_control_payload.get("lower_bound")),
        "upper_bound": _float_or_none(freq_control_payload.get("upper_bound")),
        "min_frequency_mhz": _int_or_none(freq_control_payload.get("min_frequency_mhz")),
        "max_frequency_mhz": _int_or_none(freq_control_payload.get("max_frequency_mhz")),
        "source_type": freq_control_payload.get("source_type"),
        "service_failure_detected": bool(
            freq_control_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": freq_control_payload.get(
            "service_failure_cutoff_time_utc"
        ),
        "skip_reason": None if rendered else "No valid freq-control points",
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
    freq_control_input_path = (
        Path(args.freq_control_input).expanduser().resolve()
        if args.freq_control_input
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_path = generate_figure_for_run_dir(
        run_dir,
        freq_control_input_path=freq_control_input_path,
        output_dir=output_dir,
        image_format=args.format,
        dpi=args.dpi,
    )
    print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.freq_control_input:
        raise ValueError("--freq-control-input can only be used with --run-dir")
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be a positive integer: {args.dpi}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_freq_control_summary(root_dir)
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
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(
                run_dirs,
                image_format=args.format,
                dpi=args.dpi,
            )

    if failure_count:
        print(
            f"Completed with {failure_count} failure(s) out of {len(run_dirs)} run directories.",
            file=sys.stderr,
        )
        return 1
    print(f"Completed visualization for {len(run_dirs)} run directories.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_dir:
        return _main_run_dir(args)
    return _main_root_dir(args)


if __name__ == "__main__":
    raise SystemExit(main())
