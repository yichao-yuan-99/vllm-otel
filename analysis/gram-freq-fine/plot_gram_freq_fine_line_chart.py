#!/usr/bin/env python3
"""Plot fine-grained gram-freq output as a line chart."""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


WINDOW_MODE_CENTERED = "centered"
WINDOW_MODE_TRAILING = "trailing"
WINDOW_MODE_CUMULATIVE = "cumulative"
WINDOW_MODE_CHOICES = (WINDOW_MODE_CENTERED, WINDOW_MODE_TRAILING, WINDOW_MODE_CUMULATIVE)
STAT_CHOICES = ("avg", "p25", "p75", "std")


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to render the gram-freq-fine figure. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot_gram_freq_fine_line_chart.py",
        description=(
            "Plot gram-freq-fine output JSON as a line chart: "
            "x=normalized step, y=<stats>, one series group per trace."
        ),
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        help="Path to one gram-freq-fine output JSON file.",
    )
    source_group.add_argument(
        "--root",
        help="Root directory to recursively find gram-freq-fine output JSON files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PDF path for --input mode only "
            "(default per-stat: <input>.w<window-size>.<window-mode>.n<n>.<stat>.line.pdf)."
        ),
    )
    parser.add_argument(
        "--glob",
        default="*.gram-freq-fine.json",
        help=(
            "Glob pattern for --root recursive discovery "
            "(default: *.gram-freq-fine.json)."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="N-gram key to plot from n_gram_stats (default: 1).",
    )
    parser.add_argument(
        "--stats",
        default="avg,p25,p75,std",
        help=(
            "Comma-separated statistic lines to plot from each step's n_gram_stats "
            "(choices: avg,p25,p75,std; default: avg,p25,p75,std)."
        ),
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=18.0,
        help="Figure width in inches (default: 18.0).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=10.5,
        help="Figure height in inches (default: 10.5).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel worker process count for --root mode (default: CPU count).",
    )
    return parser


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON root must be an object.")
    traces = payload.get("traces")
    if not isinstance(traces, list):
        raise ValueError("Input JSON missing valid 'traces' list.")
    return payload


def _to_float_nan(value: Any) -> float:
    if isinstance(value, bool):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def _parse_stats(raw: str) -> list[str]:
    values: list[str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token not in STAT_CHOICES:
            raise ValueError(
                f"invalid stat '{token}', expected one of: {','.join(STAT_CHOICES)}"
            )
        if token not in values:
            values.append(token)
    if not values:
        raise ValueError("at least one stat must be selected")
    return values


def _extract_series(
    payload: dict[str, Any],
    n_value: int,
    stats_to_plot: list[str],
) -> list[dict[str, Any]]:
    traces = payload.get("traces")
    assert isinstance(traces, list)

    n_key = str(n_value)
    series_rows: list[dict[str, Any]] = []

    for trace_index, trace in enumerate(traces):
        if not isinstance(trace, dict):
            continue
        steps = trace.get("steps")
        if not isinstance(steps, list) or not steps:
            continue

        worker_id = trace.get("worker_id")
        if isinstance(worker_id, str) and worker_id.strip():
            label = worker_id.strip()
        else:
            label = f"trace-{trace_index}"

        step_count = len(steps)
        x_values: list[float] = []
        stat_values: dict[str, list[float]] = {key: [] for key in stats_to_plot}

        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            n_gram_stats = step.get("n_gram_stats")
            if not isinstance(n_gram_stats, dict):
                continue
            stats = n_gram_stats.get(n_key)
            if not isinstance(stats, dict):
                continue

            # User-requested normalization: for 20 steps, step 1 -> 0.05.
            norm_step = float(step_index) / float(step_count)

            x_values.append(norm_step)
            for stat_key in stats_to_plot:
                stat_values[stat_key].append(_to_float_nan(stats.get(stat_key)))

        if not x_values:
            continue

        series_rows.append(
            {
                "label": label,
                "x": x_values,
                "stats": stat_values,
            }
        )

    return series_rows


def _payload_window_size(payload: dict[str, Any]) -> int | None:
    value = payload.get("window_size")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def _payload_window_mode(payload: dict[str, Any]) -> str | None:
    value = payload.get("window_mode")
    if isinstance(value, str) and value in WINDOW_MODE_CHOICES:
        return value
    return None


def _default_output_path(
    input_path: Path,
    n_value: int,
    stat_key: str,
    window_size: int | None,
    window_mode: str | None,
) -> Path:
    stem = input_path.stem
    marker_parts: list[str] = []
    if window_size is not None and window_mode != WINDOW_MODE_CUMULATIVE:
        marker_parts.append(f"w{window_size}")
    if window_mode:
        marker_parts.append(window_mode)
    marker = "." + ".".join(marker_parts) if marker_parts else ""
    if marker and marker not in stem:
        stem = f"{stem}{marker}"
    return input_path.with_name(f"{stem}.n{n_value}.{stat_key}.line.pdf")


def _append_stat_to_output_path(path: Path, stat_key: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}.{stat_key}{path.suffix}")
    return path.with_name(f"{path.name}.{stat_key}")


def _iter_input_paths(root: Path, pattern: str) -> list[Path]:
    paths = [path.resolve() for path in root.rglob(pattern) if path.is_file()]
    paths.sort()
    return paths


def _render_single_file(
    *,
    input_path: Path,
    output_path: Path | None,
    n_value: int,
    stats_to_plot: list[str],
    figure_width: float,
    figure_height: float,
) -> dict[str, Any]:
    payload = _load_payload(input_path)
    window_size = _payload_window_size(payload)
    window_mode = _payload_window_mode(payload)
    if output_path is None:
        per_stat_output_paths = {
            stat_key: _default_output_path(
                input_path=input_path,
                n_value=n_value,
                stat_key=stat_key,
                window_size=window_size,
                window_mode=window_mode,
            )
            for stat_key in stats_to_plot
        }
    elif len(stats_to_plot) == 1:
        per_stat_output_paths = {stats_to_plot[0]: output_path}
    else:
        per_stat_output_paths = {
            stat_key: _append_stat_to_output_path(output_path, stat_key)
            for stat_key in stats_to_plot
        }

    series_rows = _extract_series(
        payload=payload,
        n_value=n_value,
        stats_to_plot=stats_to_plot,
    )
    if not series_rows:
        raise ValueError(f"no trace series with n_gram_stats['{n_value}'] in payload")

    rendered_outputs: list[str] = []
    for stat_key in stats_to_plot:
        output_for_stat = per_stat_output_paths[stat_key]
        _plot_series(
            series_rows=series_rows,
            output_path=output_for_stat,
            payload=payload,
            n_value=n_value,
            stat_key=stat_key,
            figure_width=figure_width,
            figure_height=figure_height,
        )
        rendered_outputs.append(str(output_for_stat))

    return {
        "input": str(input_path),
        "output_pdfs": rendered_outputs,
        "window_size": window_size,
        "window_mode": window_mode,
        "stats": stats_to_plot,
        "trace_series_count": len(series_rows),
    }


def _render_single_file_worker(
    *,
    input_path: str,
    n_value: int,
    stats_to_plot: list[str],
    figure_width: float,
    figure_height: float,
) -> dict[str, Any]:
    return _render_single_file(
        input_path=Path(input_path),
        output_path=None,
        n_value=n_value,
        stats_to_plot=stats_to_plot,
        figure_width=figure_width,
        figure_height=figure_height,
    )


def _plot_series(
    *,
    series_rows: list[dict[str, Any]],
    output_path: Path,
    payload: dict[str, Any],
    n_value: int,
    stat_key: str,
    figure_width: float,
    figure_height: float,
) -> None:
    plt = _import_matplotlib_pyplot()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )

    stat_style_map: dict[str, dict[str, Any]] = {
        "avg": {"linestyle": "-", "linewidth": 1.9, "marker": "o", "markersize": 2.5, "alpha": 0.9},
        "p25": {"linestyle": "--", "linewidth": 1.35, "marker": None, "markersize": 0.0, "alpha": 0.8},
        "p75": {"linestyle": "-.", "linewidth": 1.35, "marker": None, "markersize": 0.0, "alpha": 0.8},
        "std": {"linestyle": ":", "linewidth": 1.35, "marker": None, "markersize": 0.0, "alpha": 0.8},
    }
    style = stat_style_map.get(
        stat_key,
        {"linestyle": "-", "linewidth": 1.7, "marker": None, "markersize": 0.0, "alpha": 0.85},
    )

    color_map = plt.get_cmap("tab20")
    for index, row in enumerate(series_rows):
        color = color_map(index % 20)
        y_values = row.get("stats", {}).get(stat_key)
        if not isinstance(y_values, list):
            continue
        if not any(not math.isnan(value) for value in y_values):
            continue
        axis.plot(
            row["x"],
            y_values,
            label=row["label"],
            color=color,
            linewidth=float(style["linewidth"]),
            linestyle=str(style["linestyle"]),
            marker=style["marker"],
            markersize=float(style["markersize"]),
            alpha=float(style["alpha"]),
        )

    title_suffix = payload.get("relative_plan_path")
    if not isinstance(title_suffix, str) or not title_suffix:
        title_suffix = payload.get("source_plan_path")
    if not isinstance(title_suffix, str) or not title_suffix:
        title_suffix = "unknown plan"

    axis.set_title(
        f"N={n_value} Windowed Gram Frequency ({stat_key})\n{title_suffix}",
        loc="left",
        fontweight="semibold",
    )
    axis.set_xlabel("Normalized Step")
    axis.set_ylabel("Average Frequency")
    axis.set_xlim(0.0, 1.0)
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    axis.minorticks_on()

    legend_columns = 2 if len(series_rows) > 14 else 1
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        ncol=legend_columns,
        fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.n <= 0:
        raise SystemExit("error: --n must be positive")
    if args.figure_width <= 0.0 or args.figure_height <= 0.0:
        raise SystemExit("error: --figure-width and --figure-height must be positive")
    if args.jobs <= 0:
        raise SystemExit("error: --jobs must be positive")
    try:
        stats_to_plot = _parse_stats(str(args.stats))
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.is_file():
            raise SystemExit(f"error: --input is not a file: {input_path}")

        if args.output:
            output_path = Path(args.output).expanduser().resolve()
        else:
            output_path = None

        row = _render_single_file(
            input_path=input_path,
            output_path=output_path,
            n_value=int(args.n),
            stats_to_plot=stats_to_plot,
            figure_width=float(args.figure_width),
            figure_height=float(args.figure_height),
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "single",
                    **row,
                    "n": int(args.n),
                    "stats": stats_to_plot,
                    "figure_width": float(args.figure_width),
                    "figure_height": float(args.figure_height),
                },
                ensure_ascii=True,
            )
        )
        return 0

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"error: --root is not a directory: {root}")
    if args.output:
        raise SystemExit("error: --output is only supported with --input")

    input_paths = _iter_input_paths(root, pattern=str(args.glob))
    failures: list[dict[str, str]] = []
    outputs: list[str] = []
    with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
        future_to_input = {
            executor.submit(
                _render_single_file_worker,
                input_path=str(input_path),
                n_value=int(args.n),
                stats_to_plot=stats_to_plot,
                figure_width=float(args.figure_width),
                figure_height=float(args.figure_height),
            ): input_path
            for input_path in input_paths
        }
        for future in as_completed(future_to_input):
            input_path = future_to_input[future]
            try:
                row = future.result()
            except Exception as exc:
                failures.append({"input": str(input_path), "error": str(exc)})
                continue
            output_paths = row.get("output_pdfs")
            if isinstance(output_paths, list):
                outputs.extend(str(item) for item in output_paths)

    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "root",
                "root": str(root),
                "glob": str(args.glob),
                "input_count": len(input_paths),
                "rendered_count": len(outputs),
                "failed_count": len(failures),
                "failures": failures,
                "n": int(args.n),
                "stats": stats_to_plot,
                "jobs": int(args.jobs),
                "figure_width": float(args.figure_width),
                "figure_height": float(args.figure_height),
            },
            ensure_ascii=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
