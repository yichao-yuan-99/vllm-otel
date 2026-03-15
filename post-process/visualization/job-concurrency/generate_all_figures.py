from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_INPUT_NAME = "job-concurrency-timeseries.json"
DEFAULT_OUTPUT_REL_PATH = Path("post-processed/visualization/job-concurrency")
DEFAULT_MANIFEST_NAME = "figures-manifest.json"
DEFAULT_FIGURE_STEM = "job-concurrency"
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
        description="Generate job-concurrency line-chart figures from extracted concurrency timeseries."
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing post-processed/job-concurrency/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/job-concurrency/job-concurrency-timeseries.json "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--timeseries-input",
        default=None,
        help=(
            "Optional concurrency timeseries input path. Default: "
            "<run-dir>/post-processed/job-concurrency/"
            f"{DEFAULT_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional figure output directory. Default: "
            "<run-dir>/post-processed/visualization/job-concurrency/"
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


def _default_timeseries_input_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "job-concurrency" / DEFAULT_INPUT_NAME).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_OUTPUT_REL_PATH).resolve()


def discover_run_dirs_with_job_concurrency(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for timeseries_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not timeseries_path.is_file():
            continue
        if timeseries_path.parent.name != "job-concurrency":
            continue
        if timeseries_path.parent.parent.name != "post-processed":
            continue
        run_dirs.add(timeseries_path.parent.parent.parent.resolve())
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
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    return None


def _extract_xy_from_points(raw_points: Any) -> tuple[list[float], list[float]]:
    if not isinstance(raw_points, list):
        return [], []
    pairs: list[tuple[float, float]] = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        second = _float_or_none(item.get("second"))
        concurrency = _float_or_none(item.get("concurrency"))
        if second is None or concurrency is None:
            continue
        pairs.append((second, concurrency))
    if not pairs:
        return [], []
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _extract_xy(timeseries_payload: dict[str, Any]) -> tuple[list[float], list[float]]:
    return _extract_xy_from_points(timeseries_payload.get("concurrency_points"))


def _format_stat_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _build_summary_stats(x_values: list[float], y_values: list[float]) -> dict[str, float | int | None]:
    if not x_values or not y_values:
        return {
            "sample_count": 0,
            "avg": None,
            "min": None,
            "max": None,
            "peak_time_s": None,
        }
    max_index = max(range(len(y_values)), key=lambda index: y_values[index])
    return {
        "sample_count": len(y_values),
        "avg": sum(y_values) / len(y_values),
        "min": min(y_values),
        "max": max(y_values),
        "peak_time_s": x_values[max_index],
    }


def _build_stats_annotation(
    *,
    timeseries_payload: dict[str, Any],
    summary_stats: dict[str, float | int | None],
) -> str:
    jobs_with_valid_range_count = _int_or_none(timeseries_payload.get("jobs_with_valid_range_count"))
    replay_count = _int_or_none(timeseries_payload.get("replay_count"))
    total_duration_s = _float_or_none(timeseries_payload.get("total_duration_s"))

    jobs_line = "jobs: n/a"
    if replay_count is not None:
        if jobs_with_valid_range_count is None:
            jobs_line = f"jobs: ? / {replay_count}"
        else:
            jobs_line = f"jobs: {jobs_with_valid_range_count} / {replay_count}"

    return (
        f"{jobs_line}\n"
        f"samples: {summary_stats['sample_count']}\n"
        f"avg: {_format_stat_value(_float_or_none(summary_stats.get('avg')))}\n"
        f"min: {_format_stat_value(_float_or_none(summary_stats.get('min')))}\n"
        f"max: {_format_stat_value(_float_or_none(summary_stats.get('max')))}\n"
        f"peak t: {_format_stat_value(_float_or_none(summary_stats.get('peak_time_s')))} s\n"
        f"duration: {_format_stat_value(total_duration_s)} s"
    )


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate job-concurrency figures. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt


def _render_concurrency_figure(
    *,
    timeseries_payload: dict[str, Any],
    output_path: Path,
    image_format: str,
    dpi: int,
) -> bool:
    x_values, y_values = _extract_xy(timeseries_payload)
    if not x_values:
        return False

    summary_stats = _build_summary_stats(x_values, y_values)
    avg_value = _float_or_none(summary_stats.get("avg"))
    peak_time_s = _float_or_none(summary_stats.get("peak_time_s"))
    max_value = _float_or_none(summary_stats.get("max"))

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

    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    axis.plot(x_values, y_values, color="#1D4ED8", linewidth=2.2, alpha=0.97)
    axis.fill_between(x_values, y_values, color="#BFDBFE", alpha=0.25)
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    axis.minorticks_on()

    if avg_value is not None:
        axis.axhline(
            avg_value,
            color="#991B1B",
            linestyle=(0, (6, 4)),
            linewidth=1.1,
            alpha=0.8,
        )

    if peak_time_s is not None and max_value is not None:
        axis.scatter(
            [peak_time_s],
            [max_value],
            color="#991B1B",
            s=28,
            zorder=3,
        )
        axis.annotate(
            f"Peak {max_value:.3g} @ {peak_time_s:.3g}s",
            xy=(peak_time_s, max_value),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            color="#991B1B",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#FEF2F2",
                "edgecolor": "#FCA5A5",
                "alpha": 0.95,
            },
        )

    axis.set_title("Job Concurrency Over Time", loc="left", fontweight="semibold")
    axis.set_xlabel("Time From Start (s)")
    axis.set_ylabel("Active Jobs")

    source_type = timeseries_payload.get("source_type")
    if isinstance(source_type, str) and source_type:
        axis.text(
            0.0,
            1.02,
            f"source: {source_type}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#2A3B47",
        )

    axis.text(
        0.99,
        0.98,
        _build_stats_annotation(
            timeseries_payload=timeseries_payload,
            summary_stats=summary_stats,
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


def generate_figure_for_run_dir(
    run_dir: Path,
    *,
    timeseries_input_path: Path | None = None,
    output_dir: Path | None = None,
    image_format: str = DEFAULT_FORMAT,
    dpi: int = 220,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_timeseries_path = (
        timeseries_input_path or _default_timeseries_input_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    resolved_output_dir = (
        output_dir or _default_output_dir_for_run(resolved_run_dir)
    ).expanduser().resolve()

    if not resolved_timeseries_path.is_file():
        raise ValueError(f"Missing timeseries file: {resolved_timeseries_path}")
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer: {dpi}")

    timeseries_payload = _load_json(resolved_timeseries_path)
    if not isinstance(timeseries_payload, dict):
        raise ValueError(f"Timeseries JSON must be an object: {resolved_timeseries_path}")

    x_values, y_values = _extract_xy(timeseries_payload)
    summary_stats = _build_summary_stats(x_values, y_values)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    figure_file_name = f"{DEFAULT_FIGURE_STEM}.{image_format}"
    figure_path = resolved_output_dir / figure_file_name
    rendered = _render_concurrency_figure(
        timeseries_payload=timeseries_payload,
        output_path=figure_path,
        image_format=image_format,
        dpi=dpi,
    )

    manifest = {
        "source_run_dir": str(resolved_run_dir),
        "source_timeseries_path": str(resolved_timeseries_path),
        "output_dir": str(resolved_output_dir),
        "image_format": image_format,
        "dpi": dpi,
        "figure_count": 1 if rendered else 0,
        "figure_generated": rendered,
        "figure_file_name": figure_file_name if rendered else None,
        "figure_path": str(figure_path.resolve()) if rendered else None,
        "sample_count": summary_stats["sample_count"],
        "avg_concurrency": _float_or_none(summary_stats.get("avg")),
        "min_concurrency": _float_or_none(summary_stats.get("min")),
        "max_concurrency": _float_or_none(summary_stats.get("max")),
        "peak_time_s": _float_or_none(summary_stats.get("peak_time_s")),
        "replay_count": _int_or_none(timeseries_payload.get("replay_count")),
        "jobs_with_valid_range_count": _int_or_none(
            timeseries_payload.get("jobs_with_valid_range_count")
        ),
        "total_duration_s": _float_or_none(timeseries_payload.get("total_duration_s")),
        "time_constraint_s": _float_or_none(timeseries_payload.get("time_constraint_s")),
        "source_type": timeseries_payload.get("source_type"),
        "skip_reason": None if rendered else "No valid concurrency points",
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
    timeseries_input_path = (
        Path(args.timeseries_input).expanduser().resolve()
        if args.timeseries_input
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_path = generate_figure_for_run_dir(
        run_dir,
        timeseries_input_path=timeseries_input_path,
        output_dir=output_dir,
        image_format=args.format,
        dpi=args.dpi,
    )
    print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.timeseries_input:
        raise ValueError("--timeseries-input can only be used with --run-dir")
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be a positive integer: {args.dpi}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_job_concurrency(root_dir)
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
