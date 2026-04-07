from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_INPUT_NAME = "agent-output-throughput.json"
DEFAULT_OUTPUT_REL_PATH = Path("post-processed/visualization/agent-output-throughput")
DEFAULT_MANIFEST_NAME = "figures-manifest.json"
DEFAULT_HISTOGRAM_FIGURE_STEM = "agent-output-throughput-histogram"
DEFAULT_SCATTER_FIGURE_STEM = "agent-output-throughput-vs-output-tokens"
DEFAULT_COMPLETED_REPLAY_ONLY_HISTOGRAM_FIGURE_STEM = (
    "agent-output-throughput-histogram-completed-replay-only"
)
DEFAULT_COMPLETED_REPLAY_ONLY_SCATTER_FIGURE_STEM = (
    "agent-output-throughput-vs-output-tokens-completed-replay-only"
)
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
            "Generate histogram and scatter figures from extracted "
            "agent output-throughput summaries."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing post-processed/agent-output-throughput/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/agent-output-throughput/agent-output-throughput.json "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--agent-output-input",
        default=None,
        help=(
            "Optional agent-output-throughput input path. Default: "
            "<run-dir>/post-processed/agent-output-throughput/"
            f"{DEFAULT_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional figure output directory. Default: "
            "<run-dir>/post-processed/visualization/agent-output-throughput/"
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


def _default_agent_output_input_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "agent-output-throughput" / DEFAULT_INPUT_NAME).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_OUTPUT_REL_PATH).resolve()


def discover_run_dirs_with_agent_output_throughput(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for input_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not input_path.is_file():
            continue
        if input_path.parent.name != "agent-output-throughput":
            continue
        if input_path.parent.parent.name != "post-processed":
            continue
        run_dirs.add(input_path.parent.parent.parent.resolve())
    return sorted(run_dirs)


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
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _format_stat_value(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}{suffix}"


def _variant_label(payload: dict[str, Any]) -> str | None:
    label = payload.get("_figure_variant_label")
    if not isinstance(label, str):
        return None
    stripped = label.strip()
    return stripped or None


def _throughput_from_totals(
    *,
    output_tokens: int,
    llm_request_duration_s: float,
) -> float | None:
    if llm_request_duration_s <= 0.0:
        return None
    return output_tokens / llm_request_duration_s


def _summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "avg": None,
            "min": None,
            "max": None,
            "std": None,
        }

    avg_value = sum(values) / len(values)
    variance = sum((value - avg_value) ** 2 for value in values) / len(values)
    return {
        "sample_count": len(values),
        "avg": round(avg_value, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "std": round(math.sqrt(variance), 6),
    }


def _build_histogram_from_values(
    values: list[float],
    *,
    bin_size: float,
) -> dict[str, Any]:
    if bin_size <= 0.0:
        raise ValueError(f"bin_size must be positive: {bin_size}")

    finite_values = [float(value) for value in values if math.isfinite(value)]
    if not finite_values:
        return {
            "metric": "output_throughput_tokens_per_s",
            "bin_size": round(bin_size, 6),
            "sample_count": 0,
            "bin_count": 0,
            "min": None,
            "max": None,
            "bins": [],
        }

    min_value = min(finite_values)
    max_value = max(finite_values)
    min_bin_index = math.floor(min_value / bin_size)
    max_bin_index = math.floor(max_value / bin_size)
    counts_by_index = {
        bin_index: 0 for bin_index in range(min_bin_index, max_bin_index + 1)
    }
    for value in finite_values:
        counts_by_index[math.floor(value / bin_size)] += 1

    bins = [
        {
            "bin_start": round(bin_index * bin_size, 6),
            "bin_end": round((bin_index + 1) * bin_size, 6),
            "count": counts_by_index[bin_index],
        }
        for bin_index in range(min_bin_index, max_bin_index + 1)
    ]
    return {
        "metric": "output_throughput_tokens_per_s",
        "bin_size": round(bin_size, 6),
        "sample_count": len(finite_values),
        "bin_count": len(bins),
        "min": round(min_value, 6),
        "max": round(max_value, 6),
        "bins": bins,
    }


def _histogram_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    histogram = payload.get("agent_output_throughput_tokens_per_s_histogram")
    if not isinstance(histogram, dict):
        return None
    return histogram


def _extract_histogram_bins(
    payload: dict[str, Any],
) -> tuple[list[dict[str, float | int]], float | None]:
    histogram = _histogram_payload(payload)
    if histogram is None:
        return [], None

    bin_size = _float_or_none(histogram.get("bin_size"))
    raw_bins = histogram.get("bins")
    if not isinstance(raw_bins, list):
        return [], bin_size

    bins: list[dict[str, float | int]] = []
    for item in raw_bins:
        if not isinstance(item, dict):
            continue
        bin_start = _float_or_none(item.get("bin_start"))
        bin_end = _float_or_none(item.get("bin_end"))
        count = _int_or_none(item.get("count"))
        if bin_start is None or bin_end is None or count is None:
            continue
        bins.append(
            {
                "bin_start": bin_start,
                "bin_end": bin_end,
                "count": count,
            }
        )
    return bins, bin_size


def _extract_scatter_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_agents = payload.get("agents")
    if not isinstance(raw_agents, list):
        return []

    points: list[dict[str, Any]] = []
    for agent in raw_agents:
        if not isinstance(agent, dict):
            continue
        output_tokens = _int_or_none(agent.get("output_tokens"))
        throughput = _float_or_none(agent.get("output_throughput_tokens_per_s"))
        if output_tokens is None or throughput is None:
            continue
        points.append(
            {
                "gateway_run_id": agent.get("gateway_run_id"),
                "gateway_profile_id": agent.get("gateway_profile_id"),
                "output_tokens": output_tokens,
                "output_throughput_tokens_per_s": throughput,
            }
        )
    return points


def _build_payload_for_agents(
    source_payload: dict[str, Any],
    agents: list[dict[str, Any]],
    *,
    figure_variant_label: str,
) -> dict[str, Any]:
    request_count = 0
    requests_with_output_tokens = 0
    requests_with_llm_request_duration = 0
    requests_with_output_tokens_and_llm_request_duration = 0
    output_tokens = 0
    llm_request_duration_s = 0.0
    throughput_values: list[float] = []

    for agent in agents:
        request_count += _int_or_none(agent.get("request_count")) or 0
        requests_with_output_tokens += (
            _int_or_none(agent.get("requests_with_output_tokens")) or 0
        )
        requests_with_llm_request_duration += (
            _int_or_none(agent.get("requests_with_llm_request_duration")) or 0
        )
        requests_with_output_tokens_and_llm_request_duration += (
            _int_or_none(agent.get("requests_with_output_tokens_and_llm_request_duration")) or 0
        )
        output_tokens += _int_or_none(agent.get("output_tokens")) or 0
        llm_request_duration_s += _float_or_none(agent.get("llm_request_duration_s")) or 0.0

        throughput_value = _float_or_none(agent.get("output_throughput_tokens_per_s"))
        if throughput_value is not None:
            throughput_values.append(throughput_value)

    llm_request_duration_s = round(llm_request_duration_s, 6)
    output_throughput = _throughput_from_totals(
        output_tokens=output_tokens,
        llm_request_duration_s=llm_request_duration_s,
    )

    histogram = _histogram_payload(source_payload)
    histogram_bin_size = None
    if histogram is not None:
        histogram_bin_size = _float_or_none(histogram.get("bin_size"))
    resolved_histogram_bin_size = histogram_bin_size or 1.0

    variant_payload = dict(source_payload)
    variant_payload.update(
        {
            "agent_count": len(agents),
            "request_count": request_count,
            "requests_with_output_tokens": requests_with_output_tokens,
            "requests_with_llm_request_duration": requests_with_llm_request_duration,
            "requests_with_output_tokens_and_llm_request_duration": (
                requests_with_output_tokens_and_llm_request_duration
            ),
            "output_tokens": output_tokens,
            "completion_tokens": output_tokens,
            "llm_request_duration_s": llm_request_duration_s,
            "output_throughput_tokens_per_s": (
                round(output_throughput, 6) if output_throughput is not None else None
            ),
            "agent_output_throughput_tokens_per_s_summary": _summarize_values(
                throughput_values
            ),
            "agent_output_throughput_tokens_per_s_histogram": _build_histogram_from_values(
                throughput_values,
                bin_size=resolved_histogram_bin_size,
            ),
            "agents": agents,
            "_figure_variant_label": figure_variant_label,
        }
    )
    return variant_payload


def _completed_replay_variant_payload(
    source_payload: dict[str, Any],
) -> dict[str, Any] | None:
    raw_agents = source_payload.get("agents")
    if not isinstance(raw_agents, list):
        return None

    completed_agents: list[dict[str, Any]] = []
    for agent in raw_agents:
        if not isinstance(agent, dict):
            continue

        replay_completed = agent.get("replay_completed")
        if isinstance(replay_completed, bool):
            if replay_completed:
                completed_agents.append(dict(agent))
            continue

        replay_worker_status = agent.get("replay_worker_status")
        if isinstance(replay_worker_status, str):
            if replay_worker_status == "completed":
                completed_agents.append(dict(agent))
            continue

        return None

    return _build_payload_for_agents(
        source_payload,
        completed_agents,
        figure_variant_label="completed replay only",
    )


def _variant_payload_with_scope_label(
    payload: dict[str, Any],
    *,
    scope_label: str | None,
) -> dict[str, Any]:
    if scope_label is None:
        return payload

    scoped_payload = dict(payload)
    base_label = _variant_label(payload)
    scoped_payload["_figure_variant_label"] = (
        scope_label if base_label is None else f"{base_label} | {scope_label}"
    )
    return scoped_payload


def _variant_specs_for_payload(
    agent_output_payload: dict[str, Any],
    *,
    scope_label: str | None = None,
) -> list[dict[str, Any]]:
    all_agents_payload = dict(agent_output_payload)
    all_agents_payload["_figure_variant_label"] = "all agents"

    variants = [
        {
            "variant_id": "all-agents",
            "payload": _variant_payload_with_scope_label(
                all_agents_payload,
                scope_label=scope_label,
            ),
            "histogram_figure_stem": DEFAULT_HISTOGRAM_FIGURE_STEM,
            "scatter_figure_stem": DEFAULT_SCATTER_FIGURE_STEM,
        }
    ]

    completed_replay_payload = _completed_replay_variant_payload(agent_output_payload)
    if completed_replay_payload is not None:
        variants.append(
            {
                "variant_id": "completed-replay-only",
                "payload": _variant_payload_with_scope_label(
                    completed_replay_payload,
                    scope_label=scope_label,
                ),
                "histogram_figure_stem": DEFAULT_COMPLETED_REPLAY_ONLY_HISTOGRAM_FIGURE_STEM,
                "scatter_figure_stem": DEFAULT_COMPLETED_REPLAY_ONLY_SCATTER_FIGURE_STEM,
            }
        )

    return variants


def _series_specs_for_payload(agent_output_payload: dict[str, Any]) -> list[dict[str, Any]]:
    series_specs = [
        {
            "series_id": "aggregate",
            "scope_label": None,
            "relative_output_subdir": Path(),
            "series_payload": agent_output_payload,
        }
    ]

    raw_series_by_profile = agent_output_payload.get("series_by_profile")
    if not isinstance(raw_series_by_profile, dict):
        return series_specs

    for series_id in agent_output_payload.get("series_keys", []):
        if not isinstance(series_id, str) or not series_id:
            continue
        series_payload = raw_series_by_profile.get(series_id)
        if not isinstance(series_payload, dict):
            continue
        series_specs.append(
            {
                "series_id": series_id,
                "scope_label": series_id,
                "relative_output_subdir": Path(series_id),
                "series_payload": series_payload,
            }
        )

    for series_id, series_payload in sorted(raw_series_by_profile.items()):
        if not isinstance(series_id, str) or not series_id:
            continue
        if not isinstance(series_payload, dict):
            continue
        if any(item["series_id"] == series_id for item in series_specs):
            continue
        series_specs.append(
            {
                "series_id": series_id,
                "scope_label": series_id,
                "relative_output_subdir": Path(series_id),
                "series_payload": series_payload,
            }
        )

    return series_specs


def _build_common_stats_annotation(payload: dict[str, Any]) -> str:
    summary = payload.get("agent_output_throughput_tokens_per_s_summary")
    if not isinstance(summary, dict):
        summary = {}

    return (
        f"agents: {_format_stat_value(_int_or_none(payload.get('agent_count')))}\n"
        f"run throughput: {_format_stat_value(_float_or_none(payload.get('output_throughput_tokens_per_s')), ' tok/s')}\n"
        f"avg agent: {_format_stat_value(_float_or_none(summary.get('avg')), ' tok/s')}\n"
        f"min agent: {_format_stat_value(_float_or_none(summary.get('min')), ' tok/s')}\n"
        f"max agent: {_format_stat_value(_float_or_none(summary.get('max')), ' tok/s')}"
    )


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate agent-output-throughput figures. "
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


def _render_histogram_figure(
    *,
    agent_output_payload: dict[str, Any],
    output_path: Path,
    image_format: str,
    dpi: int,
) -> bool:
    histogram_bins, bin_size = _extract_histogram_bins(agent_output_payload)
    if not histogram_bins:
        return False

    x_values = [float(item["bin_start"]) for item in histogram_bins]
    widths = [float(item["bin_end"]) - float(item["bin_start"]) for item in histogram_bins]
    y_values = [int(item["count"]) for item in histogram_bins]

    summary = agent_output_payload.get("agent_output_throughput_tokens_per_s_summary")
    avg_value = None
    if isinstance(summary, dict):
        avg_value = _float_or_none(summary.get("avg"))

    plt = _import_matplotlib_pyplot()
    _apply_plot_style(plt)

    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    axis.bar(
        x_values,
        y_values,
        width=widths,
        align="edge",
        color="#0F766E",
        edgecolor="#134E4A",
        linewidth=0.9,
        alpha=0.9,
    )
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    axis.minorticks_on()

    if avg_value is not None:
        axis.axvline(
            avg_value,
            color="#7C2D12",
            linestyle=(0, (6, 4)),
            linewidth=1.1,
            alpha=0.85,
        )
        axis.annotate(
            f"Avg {avg_value:.3g} tok/s",
            xy=(avg_value, max(y_values)),
            xytext=(10, -14),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
            color="#7C2D12",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#FFF7ED",
                "edgecolor": "#FDBA74",
                "alpha": 0.95,
            },
        )

    axis.set_title("Agent Output Throughput Histogram", loc="left", fontweight="semibold")
    axis.set_xlabel("Output Throughput (tokens/s)")
    axis.set_ylabel("Agent Count")

    subtitle_parts = ["source: gateway request logs"]
    figure_variant_label = _variant_label(agent_output_payload)
    if figure_variant_label is not None:
        subtitle_parts.append(f"subset: {figure_variant_label}")
    if bin_size is not None:
        subtitle_parts.append(f"bin size: {bin_size:.6g} tok/s")
    axis.text(
        0.0,
        1.02,
        " | ".join(subtitle_parts),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#2A3B47",
    )

    axis.text(
        0.99,
        0.98,
        (
            f"{_build_common_stats_annotation(agent_output_payload)}\n"
            f"samples: {sum(y_values)}\n"
            f"bins: {len(histogram_bins)}"
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

    figure.tight_layout()
    figure.savefig(output_path, format=image_format, dpi=dpi)
    plt.close(figure)
    return True


def _render_scatter_figure(
    *,
    agent_output_payload: dict[str, Any],
    output_path: Path,
    image_format: str,
    dpi: int,
) -> bool:
    scatter_points = _extract_scatter_points(agent_output_payload)
    if not scatter_points:
        return False

    x_values = [int(point["output_tokens"]) for point in scatter_points]
    y_values = [float(point["output_throughput_tokens_per_s"]) for point in scatter_points]
    peak_index = max(range(len(scatter_points)), key=lambda index: y_values[index])

    plt = _import_matplotlib_pyplot()
    _apply_plot_style(plt)

    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    axis.scatter(
        x_values,
        y_values,
        color="#0F766E",
        edgecolors="#CCFBF1",
        linewidths=0.9,
        s=54,
        alpha=0.9,
        zorder=3,
    )
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    axis.minorticks_on()

    peak_x = x_values[peak_index]
    peak_y = y_values[peak_index]
    axis.scatter([peak_x], [peak_y], color="#7C2D12", s=28, zorder=4)
    axis.annotate(
        f"Peak {peak_y:.3g} tok/s",
        xy=(peak_x, peak_y),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color="#7C2D12",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#FFF7ED",
            "edgecolor": "#FDBA74",
            "alpha": 0.95,
        },
    )

    axis.set_title(
        "Agent Output Throughput vs Output Tokens",
        loc="left",
        fontweight="semibold",
    )
    axis.set_xlabel("Output Tokens")
    axis.set_ylabel("Output Throughput (tokens/s)")

    figure_variant_label = _variant_label(agent_output_payload)
    axis.text(
        0.0,
        1.02,
        " | ".join(
            part
            for part in (
                "source: per-agent gateway request aggregates",
                (
                    f"subset: {figure_variant_label}"
                    if figure_variant_label is not None
                    else None
                ),
            )
            if part is not None
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#2A3B47",
    )

    axis.text(
        0.99,
        0.98,
        (
            f"{_build_common_stats_annotation(agent_output_payload)}\n"
            f"points: {len(scatter_points)}\n"
            f"max tokens: {max(x_values)}"
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

    figure.tight_layout()
    figure.savefig(output_path, format=image_format, dpi=dpi)
    plt.close(figure)
    return True


def generate_figures_for_run_dir(
    run_dir: Path,
    *,
    agent_output_input_path: Path | None = None,
    output_dir: Path | None = None,
    image_format: str = DEFAULT_FORMAT,
    dpi: int = 220,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_input_path = (
        agent_output_input_path or _default_agent_output_input_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    resolved_output_dir = (
        output_dir or _default_output_dir_for_run(resolved_run_dir)
    ).expanduser().resolve()

    if not resolved_input_path.is_file():
        raise ValueError(f"Missing agent-output-throughput file: {resolved_input_path}")
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer: {dpi}")

    agent_output_payload = _load_json(resolved_input_path)
    if not isinstance(agent_output_payload, dict):
        raise ValueError(
            f"Agent-output-throughput JSON must be an object: {resolved_input_path}"
        )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    figure_entries: list[dict[str, Any]] = []
    series_specs = _series_specs_for_payload(agent_output_payload)
    skipped_variant_ids: set[str] = set()

    for series_spec in series_specs:
        base_series_payload = series_spec["series_payload"]
        for variant_spec in _variant_specs_for_payload(
            base_series_payload,
            scope_label=series_spec["scope_label"],
        ):
            variant_id = str(variant_spec["variant_id"])
            variant_payload = variant_spec["payload"]
            if not isinstance(variant_payload, dict):
                continue

            histogram_bins, bin_size = _extract_histogram_bins(variant_payload)
            scatter_points = _extract_scatter_points(variant_payload)

            output_subdir = resolved_output_dir / series_spec["relative_output_subdir"]
            output_subdir.mkdir(parents=True, exist_ok=True)

            histogram_file_name = f"{variant_spec['histogram_figure_stem']}.{image_format}"
            histogram_path = output_subdir / histogram_file_name
            histogram_rendered = _render_histogram_figure(
                agent_output_payload=variant_payload,
                output_path=histogram_path,
                image_format=image_format,
                dpi=dpi,
            )
            figure_entries.append(
                {
                    "series_id": series_spec["series_id"],
                    "gateway_profile_id": _int_or_none(
                        base_series_payload.get("gateway_profile_id")
                    ),
                    "variant_id": variant_id,
                    "variant_label": _variant_label(variant_payload),
                    "figure_id": "histogram",
                    "relative_output_subdir": (
                        series_spec["relative_output_subdir"].as_posix()
                        if series_spec["relative_output_subdir"].parts
                        else ""
                    ),
                    "figure_generated": histogram_rendered,
                    "figure_file_name": histogram_file_name if histogram_rendered else None,
                    "figure_path": (
                        str(histogram_path.resolve()) if histogram_rendered else None
                    ),
                    "agent_count": _int_or_none(variant_payload.get("agent_count")),
                    "run_output_throughput_tokens_per_s": _float_or_none(
                        variant_payload.get("output_throughput_tokens_per_s")
                    ),
                    "sample_count": sum(int(item["count"]) for item in histogram_bins),
                    "bin_count": len(histogram_bins),
                    "bin_size": bin_size,
                    "skip_reason": None if histogram_rendered else "No valid histogram bins",
                }
            )

            scatter_file_name = f"{variant_spec['scatter_figure_stem']}.{image_format}"
            scatter_path = output_subdir / scatter_file_name
            scatter_rendered = _render_scatter_figure(
                agent_output_payload=variant_payload,
                output_path=scatter_path,
                image_format=image_format,
                dpi=dpi,
            )
            series_variant_id = (
                variant_id
                if series_spec["series_id"] == "aggregate"
                else f"{series_spec['series_id']}:{variant_id}"
            )
            if not histogram_rendered and not scatter_rendered:
                skipped_variant_ids.add(series_variant_id)
            figure_entries.append(
                {
                    "series_id": series_spec["series_id"],
                    "gateway_profile_id": _int_or_none(
                        base_series_payload.get("gateway_profile_id")
                    ),
                    "variant_id": variant_id,
                    "variant_label": _variant_label(variant_payload),
                    "figure_id": "scatter",
                    "relative_output_subdir": (
                        series_spec["relative_output_subdir"].as_posix()
                        if series_spec["relative_output_subdir"].parts
                        else ""
                    ),
                    "figure_generated": scatter_rendered,
                    "figure_file_name": scatter_file_name if scatter_rendered else None,
                    "figure_path": (
                        str(scatter_path.resolve()) if scatter_rendered else None
                    ),
                    "agent_count": _int_or_none(variant_payload.get("agent_count")),
                    "run_output_throughput_tokens_per_s": _float_or_none(
                        variant_payload.get("output_throughput_tokens_per_s")
                    ),
                    "sample_count": len(scatter_points),
                    "max_output_tokens": max(
                        (point["output_tokens"] for point in scatter_points),
                        default=None,
                    ),
                    "max_output_throughput_tokens_per_s": max(
                        (point["output_throughput_tokens_per_s"] for point in scatter_points),
                        default=None,
                    ),
                    "skip_reason": None if scatter_rendered else "No valid scatter points",
                }
            )

    primary_figure = next(
        (item for item in figure_entries if item["figure_generated"]),
        figure_entries[0] if figure_entries else None,
    )
    manifest = {
        "source_run_dir": str(resolved_run_dir),
        "source_agent_output_input_path": str(resolved_input_path),
        "output_dir": str(resolved_output_dir),
        "image_format": image_format,
        "dpi": dpi,
        "multi_profile": bool(agent_output_payload.get("multi_profile", False)),
        "port_profile_ids": agent_output_payload.get("port_profile_ids"),
        "series_count": len(series_specs),
        "figure_count": len([item for item in figure_entries if item["figure_generated"]]),
        "requested_figure_count": len(figure_entries),
        "variant_count": len(_variant_specs_for_payload(agent_output_payload)),
        "skipped_variant_count": len(skipped_variant_ids),
        "skipped_variant_ids": sorted(skipped_variant_ids),
        "figures": figure_entries,
        "figure_generated": primary_figure["figure_generated"] if primary_figure else False,
        "figure_file_name": primary_figure["figure_file_name"] if primary_figure else None,
        "figure_path": primary_figure["figure_path"] if primary_figure else None,
        "agent_count": _int_or_none(agent_output_payload.get("agent_count")),
        "run_output_throughput_tokens_per_s": _float_or_none(
            agent_output_payload.get("output_throughput_tokens_per_s")
        ),
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
        output_path = generate_figures_for_run_dir(
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
            output_path = generate_figures_for_run_dir(
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
    agent_output_input_path = (
        Path(args.agent_output_input).expanduser().resolve()
        if args.agent_output_input
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_path = generate_figures_for_run_dir(
        run_dir,
        agent_output_input_path=agent_output_input_path,
        output_dir=output_dir,
        image_format=args.format,
        dpi=args.dpi,
    )
    print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.agent_output_input:
        raise ValueError("--agent-output-input can only be used with --run-dir")
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be a positive integer: {args.dpi}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_agent_output_throughput(root_dir)
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
