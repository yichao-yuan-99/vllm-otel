from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_INPUT_NAME = "context-usage-stacked-histogram.json"
DEFAULT_STACK_CONTEXT_INPUT_REL_PATH = Path("post-processed/gateway/stack-context")
DEFAULT_OUTPUT_REL_PATH = Path("post-processed/visualization/gateway-stack-context")
DEFAULT_MANIFEST_NAME = "figures-manifest.json"
DEFAULT_FIGURE_STEM = "context-usage-stacked-histogram"
DEFAULT_FORMAT = "png"
SUPPORTED_FORMATS = ("png", "pdf", "svg")
SMOOTH_WINDOW_SECONDS = (10, 30, 60, 120)


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
            "Generate figures from gateway stacked context-usage histograms "
            "for raw and smoothed variants."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing post-processed/gateway/stack-context/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/gateway/stack-context/context-usage-stacked-histogram.json "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--stack-context-input-dir",
        default=None,
        help=(
            "Optional gateway stack-context input directory. Default: "
            "<run-dir>/post-processed/gateway/stack-context/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional figure output directory. Default: "
            "<run-dir>/post-processed/visualization/gateway-stack-context/"
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


def _default_stack_context_input_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_STACK_CONTEXT_INPUT_REL_PATH).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_OUTPUT_REL_PATH).resolve()


def _run_dir_from_stack_context_input_dir(stack_context_input_dir: Path) -> Path | None:
    if stack_context_input_dir.name != "stack-context":
        return None
    gateway_dir = stack_context_input_dir.parent
    if gateway_dir.name != "gateway":
        return None
    post_processed_dir = gateway_dir.parent
    if post_processed_dir.name != "post-processed":
        return None
    return post_processed_dir.parent.resolve()


def discover_run_dirs_with_gateway_stack_context(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for input_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not input_path.is_file():
            continue
        stack_context_input_dir = input_path.parent
        run_dir = _run_dir_from_stack_context_input_dir(stack_context_input_dir)
        if run_dir is None:
            continue
        run_dirs.add(run_dir)
    return sorted(run_dirs)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_xy(histogram_payload: dict[str, Any]) -> tuple[list[float], list[float]]:
    raw_points = histogram_payload.get("points")
    if not isinstance(raw_points, list):
        return [], []

    pairs: list[tuple[float, float]] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        second = _float_or_none(point.get("second"))
        accumulated_value = _float_or_none(point.get("accumulated_value"))
        if second is None or accumulated_value is None:
            continue
        pairs.append((second, accumulated_value))

    if not pairs:
        return [], []

    pairs.sort(key=lambda pair: pair[0])
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _variant_specs() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {
            "variant_id": "raw",
            "window_size_s": None,
            "file_suffix": "",
            "title_suffix": "",
        }
    ]
    for window_size_s in SMOOTH_WINDOW_SECONDS:
        variants.append(
            {
                "variant_id": f"smoothed-{window_size_s}s",
                "window_size_s": window_size_s,
                "file_suffix": f"-smoothed-{window_size_s}s",
                "title_suffix": f" (Smoothed, w={window_size_s}s)",
            }
        )
    return variants


def _smooth_values_with_centered_window(
    x_values: list[float],
    y_values: list[float],
    *,
    window_size_s: float,
) -> list[float]:
    if not x_values or not y_values:
        return []
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values length mismatch while smoothing")
    if window_size_s <= 0:
        return list(y_values)

    prefix_sums = [0.0]
    for value in y_values:
        prefix_sums.append(prefix_sums[-1] + value)

    smoothed_values: list[float] = []
    left = 0
    right = 0
    point_count = len(x_values)
    for index in range(point_count):
        center_time = x_values[index]
        left_bound = center_time - window_size_s
        right_bound = center_time + window_size_s

        while left < point_count and x_values[left] <= left_bound:
            left += 1
        while right < point_count and x_values[right] < right_bound:
            right += 1

        if right <= left:
            smoothed_values.append(y_values[index])
            continue

        window_sum = prefix_sums[right] - prefix_sums[left]
        smoothed_values.append(window_sum / (right - left))
    return smoothed_values


def _format_stat(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.6g}"


def _build_summary_stats(x_values: list[float], y_values: list[float]) -> dict[str, float | int | None]:
    if not x_values or not y_values:
        return {
            "sample_count": 0,
            "avg": None,
            "min": None,
            "max": None,
            "peak_second": None,
            "total_accumulated_value": None,
        }

    max_index = max(range(len(y_values)), key=lambda index: y_values[index])
    return {
        "sample_count": len(y_values),
        "avg": sum(y_values) / len(y_values),
        "min": min(y_values),
        "max": max(y_values),
        "peak_second": x_values[max_index],
        "total_accumulated_value": sum(y_values),
    }


def _build_stats_annotation(summary_stats: dict[str, Any]) -> str:
    return (
        f"samples: {summary_stats['sample_count']}\n"
        f"avg: {_format_stat(summary_stats.get('avg'))}\n"
        f"min: {_format_stat(summary_stats.get('min'))}\n"
        f"max: {_format_stat(summary_stats.get('max'))}\n"
        f"peak t: {_format_stat(summary_stats.get('peak_second'))} s\n"
        f"sum: {_format_stat(summary_stats.get('total_accumulated_value'))}"
    )


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate gateway stack-context figures. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt


def _render_variant_figure(
    *,
    histogram_payload: dict[str, Any],
    window_size_s: int | None,
    title_suffix: str,
    output_path: Path,
    image_format: str,
    dpi: int,
) -> tuple[bool, dict[str, Any]]:
    x_values, y_values = _extract_xy(histogram_payload)
    if window_size_s is not None:
        y_values = _smooth_values_with_centered_window(
            x_values,
            y_values,
            window_size_s=float(window_size_s),
        )
    summary_stats = _build_summary_stats(x_values, y_values)
    if not x_values:
        return False, summary_stats

    avg_value = _float_or_none(summary_stats.get("avg"))
    peak_second = _float_or_none(summary_stats.get("peak_second"))
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

    line_color = "#155E75"
    fill_color = "#A5F3FC"
    accent_color = "#0F172A"

    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    axis.plot(
        x_values,
        y_values,
        color=line_color,
        linewidth=2.1,
        alpha=0.97,
    )
    axis.fill_between(
        x_values,
        y_values,
        color=fill_color,
        alpha=0.3,
    )
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    axis.minorticks_on()

    if avg_value is not None:
        axis.axhline(
            avg_value,
            color=accent_color,
            linestyle=(0, (6, 4)),
            linewidth=1.1,
            alpha=0.82,
        )

    if peak_second is not None and max_value is not None:
        axis.scatter(
            [peak_second],
            [max_value],
            color=accent_color,
            s=30,
            zorder=3,
        )
        axis.annotate(
            f"Peak {max_value:.3g} @ {peak_second:.3g}s",
            xy=(peak_second, max_value),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            color=accent_color,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#F0FDFA",
                "edgecolor": "#67E8F9",
                "alpha": 0.95,
            },
        )

    axis.set_title(
        "Agent Context Usage (Stacked)" + title_suffix,
        loc="left",
        fontweight="semibold",
    )
    axis.set_xlabel("Second From Run Start (s)")
    axis.set_ylabel("Accumulated Context Usage in 1s Bucket")

    subtitle_parts: list[str] = []
    metric_id = histogram_payload.get("metric")
    if isinstance(metric_id, str) and metric_id:
        subtitle_parts.append(f"metric: {metric_id}")
    phase = histogram_payload.get("phase")
    if isinstance(phase, str) and phase:
        subtitle_parts.append(f"phase: {phase}")
    bucket_width_s = _float_or_none(histogram_payload.get("bucket_width_s"))
    if bucket_width_s is not None:
        subtitle_parts.append(f"bucket: {bucket_width_s:g}s")
    if window_size_s is not None:
        subtitle_parts.append(f"smoothed window: +/- {window_size_s:g}s")
    if subtitle_parts:
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
        _build_stats_annotation(summary_stats),
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
    return True, summary_stats


def generate_figures_for_run_dir(
    run_dir: Path,
    *,
    stack_context_input_dir: Path | None = None,
    output_dir: Path | None = None,
    image_format: str = DEFAULT_FORMAT,
    dpi: int = 220,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_stack_context_input_dir = (
        stack_context_input_dir or _default_stack_context_input_dir_for_run(resolved_run_dir)
    ).expanduser().resolve()
    resolved_output_dir = (
        output_dir or _default_output_dir_for_run(resolved_run_dir)
    ).expanduser().resolve()

    if not resolved_stack_context_input_dir.is_dir():
        raise ValueError(
            f"Missing stack-context input directory: {resolved_stack_context_input_dir}"
        )
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer: {dpi}")

    input_path = resolved_stack_context_input_dir / DEFAULT_INPUT_NAME
    if not input_path.is_file():
        raise ValueError(f"Missing stack-context histogram file: {input_path}")

    histogram_payload = _load_json(input_path)
    if not isinstance(histogram_payload, dict):
        raise ValueError(f"Histogram JSON must be an object: {input_path}")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    figure_entries: list[dict[str, Any]] = []
    skipped_variants: list[str] = []
    variants = _variant_specs()

    for variant_spec in variants:
        variant_id = variant_spec["variant_id"]
        window_size_s = variant_spec["window_size_s"]

        figure_file_name = f"{DEFAULT_FIGURE_STEM}{variant_spec['file_suffix']}.{image_format}"
        figure_path = resolved_output_dir / figure_file_name
        rendered, summary_stats = _render_variant_figure(
            histogram_payload=histogram_payload,
            window_size_s=window_size_s,
            title_suffix=variant_spec["title_suffix"],
            output_path=figure_path,
            image_format=image_format,
            dpi=dpi,
        )
        if not rendered:
            skipped_variants.append(variant_id)

        figure_entries.append(
            {
                "variant_id": variant_id,
                "window_size_s": window_size_s,
                "figure_generated": rendered,
                "figure_file_name": figure_file_name if rendered else None,
                "figure_path": str(figure_path.resolve()) if rendered else None,
                "source_histogram_path": str(input_path.resolve()),
                "sample_count": summary_stats["sample_count"],
                "avg_accumulated_value": _float_or_none(summary_stats.get("avg")),
                "min_accumulated_value": _float_or_none(summary_stats.get("min")),
                "max_accumulated_value": _float_or_none(summary_stats.get("max")),
                "peak_second": _float_or_none(summary_stats.get("peak_second")),
                "total_accumulated_value": _float_or_none(
                    summary_stats.get("total_accumulated_value")
                ),
                "skip_reason": None if rendered else "No valid histogram points",
            }
        )

    manifest = {
        "source_run_dir": str(resolved_run_dir),
        "source_stack_context_input_dir": str(resolved_stack_context_input_dir),
        "output_dir": str(resolved_output_dir),
        "image_format": image_format,
        "dpi": dpi,
        "variant_count": len(variants),
        "figure_count": len([entry for entry in figure_entries if entry["figure_generated"]]),
        "skipped_variant_count": len(skipped_variants),
        "skipped_variants": skipped_variants,
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
    stack_context_input_dir = (
        Path(args.stack_context_input_dir).expanduser().resolve()
        if args.stack_context_input_dir
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_path = generate_figures_for_run_dir(
        run_dir,
        stack_context_input_dir=stack_context_input_dir,
        output_dir=output_dir,
        image_format=args.format,
        dpi=args.dpi,
    )
    print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.stack_context_input_dir:
        raise ValueError("--stack-context-input-dir can only be used with --run-dir")
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be a positive integer: {args.dpi}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_gateway_stack_context(root_dir)
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
