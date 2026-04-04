from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_INPUT_NAME = "slo-decision-summary.json"
DEFAULT_INPUT_DIR_NAME = "slo-decision"
DEFAULT_OUTPUT_REL_PATH = Path("post-processed/visualization/slo-decision")
DEFAULT_MANIFEST_NAME = "figures-manifest.json"
DEFAULT_FIGURE_STEM = "slo-decision-timeline"
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
            "Generate SLO-decision timeline figures from extracted "
            "SLO-driven controller decision summaries."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing post-processed/slo-decision/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/slo-decision/slo-decision-summary.json will be processed."
        ),
    )
    parser.add_argument(
        "--slo-decision-input",
        default=None,
        help=(
            "Optional SLO-decision input path. Default: "
            "<run-dir>/post-processed/slo-decision/"
            f"{DEFAULT_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional figure output directory. Default: "
            "<run-dir>/post-processed/visualization/slo-decision/"
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


def _default_slo_decision_input_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / DEFAULT_INPUT_DIR_NAME / DEFAULT_INPUT_NAME).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_OUTPUT_REL_PATH).resolve()


def discover_run_dirs_with_slo_decision_summary(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for input_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not input_path.is_file():
            continue
        if input_path.parent.name != DEFAULT_INPUT_DIR_NAME:
            continue
        if input_path.parent.parent.name != "post-processed":
            continue
        run_dirs.add(input_path.parent.parent.parent.resolve())
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


def _format_stat_value(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}{suffix}"


def _extract_decision_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_points = payload.get("decision_points")
    if not isinstance(raw_points, list):
        return []
    points: list[dict[str, Any]] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        if time_offset_s is None:
            continue
        normalized = dict(point)
        normalized["time_offset_s"] = time_offset_s
        points.append(normalized)
    points.sort(
        key=lambda point: (
            _float_or_none(point.get("time_offset_s")) or 0.0,
            point.get("timestamp_utc") or "",
        )
    )
    return points


def _extract_throughput_series(
    decisions: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    line_x: list[float] = []
    line_y: list[float] = []
    changed_x: list[float] = []
    changed_y: list[float] = []
    hold_x: list[float] = []
    hold_y: list[float] = []
    for point in decisions:
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        throughput = _float_or_none(point.get("window_min_output_tokens_per_s"))
        changed = point.get("changed") is True
        if time_offset_s is None or throughput is None:
            continue
        line_x.append(time_offset_s)
        line_y.append(throughput)
        if changed:
            changed_x.append(time_offset_s)
            changed_y.append(throughput)
        else:
            hold_x.append(time_offset_s)
            hold_y.append(throughput)
    return line_x, line_y, changed_x, changed_y, hold_x, hold_y


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


def _build_stats_annotation(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"SLO decisions: {_format_stat_value(_int_or_none(payload.get('slo_decision_point_count')))}",
            f"freq changes: {_format_stat_value(_int_or_none(payload.get('slo_decision_change_count')))}",
            f"target throughput: {_format_stat_value(_float_or_none(payload.get('target_output_throughput_tokens_per_s')), ' tok/s')}",
            f"window min: {_format_stat_value(_float_or_none(payload.get('min_window_min_output_tokens_per_s')), ' tok/s')}",
            f"window max: {_format_stat_value(_float_or_none(payload.get('max_window_min_output_tokens_per_s')), ' tok/s')}",
            f"freq min: {_format_stat_value(_int_or_none(payload.get('min_frequency_mhz')), ' MHz')}",
            f"freq max: {_format_stat_value(_int_or_none(payload.get('max_frequency_mhz')), ' MHz')}",
        ]
    )


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate SLO-decision figures. "
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


def _render_slo_decision_figure(
    *,
    slo_decision_payload: dict[str, Any],
    output_path: Path,
    image_format: str,
    dpi: int,
) -> bool:
    decisions = _extract_decision_points(slo_decision_payload)
    if not decisions:
        return False

    (
        throughput_x,
        throughput_y,
        throughput_changed_x,
        throughput_changed_y,
        throughput_hold_x,
        throughput_hold_y,
    ) = _extract_throughput_series(decisions)
    step_x, step_y, changed_x, changed_y, hold_x, hold_y = _extract_frequency_step_series(
        decisions
    )

    all_x_values = throughput_x + step_x
    if not all_x_values:
        return False

    target_output_throughput_tokens_per_s = _float_or_none(
        slo_decision_payload.get("target_output_throughput_tokens_per_s")
    )

    plt = _import_matplotlib_pyplot()
    _apply_plot_style(plt)

    figure, (throughput_axis, frequency_axis) = plt.subplots(
        2,
        1,
        figsize=(11.4, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 2.0], "hspace": 0.14},
    )

    x_min = min(all_x_values)
    x_max = max(all_x_values)

    if target_output_throughput_tokens_per_s is not None:
        throughput_axis.axhline(
            target_output_throughput_tokens_per_s,
            color="#0F766E",
            linestyle=(0, (6, 4)),
            linewidth=1.2,
            alpha=0.95,
            label="throughput target",
        )
    if throughput_x:
        throughput_axis.plot(
            throughput_x,
            throughput_y,
            color="#DC2626",
            linewidth=2.0,
            alpha=0.95,
            label="window min throughput",
        )
    if throughput_changed_x:
        throughput_axis.scatter(
            throughput_changed_x,
            throughput_changed_y,
            color="#B91C1C",
            s=28,
            zorder=3,
            label="freq change",
        )
    if throughput_hold_x:
        throughput_axis.scatter(
            throughput_hold_x,
            throughput_hold_y,
            facecolors="none",
            edgecolors="#7F1D1D",
            s=24,
            linewidths=1.0,
            zorder=3,
            label="hold at cap",
        )
    throughput_axis.axvline(
        0.0,
        color="#334155",
        linestyle=":",
        linewidth=1.0,
        alpha=0.75,
    )
    throughput_axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    throughput_axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    throughput_axis.minorticks_on()
    throughput_axis.set_title(
        "SLO-Driven Frequency Decisions",
        loc="left",
        fontweight="semibold",
    )
    throughput_axis.set_ylabel("Min Output Throughput (tok/s)")
    throughput_axis.legend(loc="upper left", frameon=False, ncols=2)

    subtitle_parts: list[str] = []
    source_type = slo_decision_payload.get("source_type")
    if isinstance(source_type, str) and source_type:
        subtitle_parts.append(f"source: {source_type}")
    analysis_start = slo_decision_payload.get("analysis_window_start_utc")
    if isinstance(analysis_start, str) and analysis_start:
        subtitle_parts.append(f"start: {analysis_start}")
    if subtitle_parts:
        throughput_axis.text(
            0.0,
            1.02,
            " | ".join(subtitle_parts),
            transform=throughput_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#2A3B47",
        )

    throughput_axis.text(
        0.99,
        0.98,
        _build_stats_annotation(slo_decision_payload),
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

    if step_x and step_y:
        frequency_axis.step(
            step_x,
            step_y,
            where="post",
            color="#0F172A",
            linewidth=2.0,
            alpha=0.95,
            label="target freq",
        )
    if changed_x:
        frequency_axis.scatter(
            changed_x,
            changed_y,
            color="#059669",
            s=26,
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
    frequency_axis.set_xlabel("Time From Replay Start (s)")
    frequency_axis.set_ylabel("GPU Core Frequency (MHz)")
    if step_x:
        frequency_axis.legend(loc="upper left", frameon=False, ncols=3)

    if x_min == x_max:
        padding = 1.0
        throughput_axis.set_xlim(x_min - padding, x_max + padding)
    else:
        throughput_axis.set_xlim(x_min, x_max)
    figure.savefig(output_path, format=image_format, dpi=dpi)
    plt.close(figure)
    return True


def generate_figure_for_run_dir(
    run_dir: Path,
    *,
    slo_decision_input_path: Path | None = None,
    output_dir: Path | None = None,
    image_format: str = DEFAULT_FORMAT,
    dpi: int = 220,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_slo_decision_input_path = (
        slo_decision_input_path or _default_slo_decision_input_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    resolved_output_dir = (
        output_dir or _default_output_dir_for_run(resolved_run_dir)
    ).expanduser().resolve()

    if not resolved_slo_decision_input_path.is_file():
        raise ValueError(
            f"Missing SLO-decision summary file: {resolved_slo_decision_input_path}"
        )
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer: {dpi}")

    slo_decision_payload = _load_json(resolved_slo_decision_input_path)
    if not isinstance(slo_decision_payload, dict):
        raise ValueError(
            "SLO-decision summary JSON must be an object: "
            f"{resolved_slo_decision_input_path}"
        )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    figure_file_name = f"{DEFAULT_FIGURE_STEM}.{image_format}"
    figure_path = resolved_output_dir / figure_file_name
    rendered = _render_slo_decision_figure(
        slo_decision_payload=slo_decision_payload,
        output_path=figure_path,
        image_format=image_format,
        dpi=dpi,
    )

    manifest = {
        "source_run_dir": str(resolved_run_dir),
        "source_slo_decision_summary_path": str(resolved_slo_decision_input_path),
        "output_dir": str(resolved_output_dir),
        "image_format": image_format,
        "dpi": dpi,
        "figure_count": 1 if rendered else 0,
        "figure_generated": rendered,
        "figure_file_name": figure_file_name if rendered else None,
        "figure_path": str(figure_path.resolve()) if rendered else None,
        "slo_decision_log_found": bool(
            slo_decision_payload.get("slo_decision_log_found", False)
        ),
        "slo_decision_point_count": _int_or_none(
            slo_decision_payload.get("slo_decision_point_count")
        ),
        "slo_decision_change_count": _int_or_none(
            slo_decision_payload.get("slo_decision_change_count")
        ),
        "target_output_throughput_tokens_per_s": _float_or_none(
            slo_decision_payload.get("target_output_throughput_tokens_per_s")
        ),
        "min_window_min_output_tokens_per_s": _float_or_none(
            slo_decision_payload.get("min_window_min_output_tokens_per_s")
        ),
        "max_window_min_output_tokens_per_s": _float_or_none(
            slo_decision_payload.get("max_window_min_output_tokens_per_s")
        ),
        "min_frequency_mhz": _int_or_none(slo_decision_payload.get("min_frequency_mhz")),
        "max_frequency_mhz": _int_or_none(slo_decision_payload.get("max_frequency_mhz")),
        "source_type": slo_decision_payload.get("source_type"),
        "service_failure_detected": bool(
            slo_decision_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": slo_decision_payload.get(
            "service_failure_cutoff_time_utc"
        ),
        "skip_reason": None if rendered else "No valid SLO decision points",
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
    slo_decision_input_path = (
        Path(args.slo_decision_input).expanduser().resolve()
        if args.slo_decision_input
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_path = generate_figure_for_run_dir(
        run_dir,
        slo_decision_input_path=slo_decision_input_path,
        output_dir=output_dir,
        image_format=args.format,
        dpi=args.dpi,
    )
    print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.slo_decision_input:
        raise ValueError("--slo-decision-input can only be used with --run-dir")
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be a positive integer: {args.dpi}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_slo_decision_summary(root_dir)
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
