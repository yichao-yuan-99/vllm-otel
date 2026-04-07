from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_INPUT_NAME = "request-throughput-timeseries.json"
DEFAULT_OUTPUT_REL_PATH = Path("post-processed/visualization/request-throughput")
DEFAULT_MANIFEST_NAME = "figures-manifest.json"
DEFAULT_FIGURE_STEM = "request-throughput"
DEFAULT_STATUS_200_FIGURE_STEM = "request-throughput-status-200"
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
            "Generate request-throughput line-chart figures from extracted throughput timeseries."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing post-processed/request-throughput/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/request-throughput/request-throughput-timeseries.json "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--timeseries-input",
        default=None,
        help=(
            "Optional throughput timeseries input path. Default: "
            "<run-dir>/post-processed/request-throughput/"
            f"{DEFAULT_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional figure output directory. Default: "
            "<run-dir>/post-processed/visualization/request-throughput/"
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
    return (run_dir / "post-processed" / "request-throughput" / DEFAULT_INPUT_NAME).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_OUTPUT_REL_PATH).resolve()


def discover_run_dirs_with_request_throughput(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for timeseries_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not timeseries_path.is_file():
            continue
        if timeseries_path.parent.name != "request-throughput":
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
        time_s = _float_or_none(item.get("time_s"))
        throughput_requests_per_s = _float_or_none(item.get("throughput_requests_per_s"))
        if time_s is None or throughput_requests_per_s is None:
            continue
        pairs.append((time_s, throughput_requests_per_s))
    if not pairs:
        return [], []
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _extract_xy(
    timeseries_payload: dict[str, Any],
    *,
    points_key: str = "throughput_points",
) -> tuple[list[float], list[float]]:
    return _extract_xy_from_points(timeseries_payload.get(points_key))


def _format_stat_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _build_summary_stats(
    x_values: list[float],
    y_values: list[float],
) -> dict[str, float | int | None]:
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
    series_payload: dict[str, Any],
    summary_stats: dict[str, float | int | None],
) -> str:
    finished_request_count = _int_or_none(series_payload.get("finished_request_count"))
    request_count = _int_or_none(series_payload.get("request_count"))
    total_duration_s = _float_or_none(series_payload.get("total_duration_s"))
    window_size_s = _float_or_none(series_payload.get("window_size_s"))
    timepoint_frequency_hz = _float_or_none(series_payload.get("timepoint_frequency_hz"))

    requests_line = "requests: n/a"
    if request_count is not None:
        if finished_request_count is None:
            requests_line = f"requests: ? / {request_count}"
        else:
            requests_line = f"requests: {finished_request_count} / {request_count}"

    return (
        f"{requests_line}\n"
        f"samples: {summary_stats['sample_count']}\n"
        f"avg: {_format_stat_value(_float_or_none(summary_stats.get('avg')))}\n"
        f"min: {_format_stat_value(_float_or_none(summary_stats.get('min')))}\n"
        f"max: {_format_stat_value(_float_or_none(summary_stats.get('max')))}\n"
        f"peak t: {_format_stat_value(_float_or_none(summary_stats.get('peak_time_s')))} s\n"
        f"duration: {_format_stat_value(total_duration_s)} s\n"
        f"window: +/- {_format_stat_value(window_size_s)} s\n"
        f"freq: {_format_stat_value(timepoint_frequency_hz)} Hz"
    )


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate request-throughput figures. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt


def _render_throughput_figure(
    *,
    series_payload: dict[str, Any],
    output_path: Path,
    image_format: str,
    dpi: int,
) -> bool:
    x_values, y_values = _extract_xy(series_payload)
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
            color="#9A3412",
            linestyle=(0, (6, 4)),
            linewidth=1.1,
            alpha=0.8,
        )

    if peak_time_s is not None and max_value is not None:
        axis.scatter(
            [peak_time_s],
            [max_value],
            color="#9A3412",
            s=28,
            zorder=3,
        )
        axis.annotate(
            f"Peak {max_value:.3g} @ {peak_time_s:.3g}s",
            xy=(peak_time_s, max_value),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            color="#9A3412",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#FFF7ED",
                "edgecolor": "#FDBA74",
                "alpha": 0.95,
            },
        )

    title = series_payload.get("_figure_title")
    if not isinstance(title, str) or not title:
        title = "Request Throughput Over Time"
    axis.set_title(title, loc="left", fontweight="semibold")
    axis.set_xlabel("Time From Start (s)")
    axis.set_ylabel("Throughput (requests/s)")

    subtitle_parts: list[str] = []
    variant_label = series_payload.get("_figure_variant_label")
    if isinstance(variant_label, str) and variant_label:
        subtitle_parts.append(variant_label)
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
        _build_stats_annotation(
            series_payload=series_payload,
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


def _variant_specs_for_payload(timeseries_payload: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [
        {
            "variant_id": "all-finished",
            "figure_stem": DEFAULT_FIGURE_STEM,
            "points_key": "throughput_points",
            "finished_count_key": "finished_request_count",
            "figure_title": "Request Throughput Over Time",
            "figure_variant_label": "all finished requests",
        }
    ]
    if isinstance(timeseries_payload.get("throughput_points_status_200"), list):
        variants.append(
            {
                "variant_id": "status-200-only",
                "figure_stem": DEFAULT_STATUS_200_FIGURE_STEM,
                "points_key": "throughput_points_status_200",
                "finished_count_key": "finished_request_count_status_200",
                "figure_title": "Request Throughput Over Time",
                "figure_variant_label": "HTTP 200 only",
            }
        )
    return variants


def _build_series_payload(
    timeseries_payload: dict[str, Any],
    *,
    points_key: str,
    finished_count_key: str,
    figure_title: str,
    figure_variant_label: str,
    scope_label: str | None = None,
) -> dict[str, Any]:
    series_payload = dict(timeseries_payload)
    series_payload["throughput_points"] = timeseries_payload.get(points_key)
    series_payload["finished_request_count"] = timeseries_payload.get(finished_count_key)
    series_payload["_figure_title"] = figure_title
    variant_parts = [figure_variant_label] if figure_variant_label else []
    if scope_label:
        variant_parts.append(scope_label)
    series_payload["_figure_variant_label"] = " | ".join(variant_parts)
    return series_payload


def _series_specs_for_payload(timeseries_payload: dict[str, Any]) -> list[dict[str, Any]]:
    series_specs = [
        {
            "series_id": "aggregate",
            "scope_label": None,
            "relative_output_subdir": Path(),
            "series_payload": timeseries_payload,
        }
    ]
    raw_series_by_profile = timeseries_payload.get("series_by_profile")
    if not isinstance(raw_series_by_profile, dict):
        return series_specs
    for series_id in timeseries_payload.get("series_keys", []):
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

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    figure_entries: list[dict[str, Any]] = []
    skipped_variant_ids: list[str] = []
    series_specs = _series_specs_for_payload(timeseries_payload)

    for series_spec in series_specs:
        base_series_payload = series_spec["series_payload"]
        for variant_spec in _variant_specs_for_payload(base_series_payload):
            series_payload = _build_series_payload(
                base_series_payload,
                points_key=variant_spec["points_key"],
                finished_count_key=variant_spec["finished_count_key"],
                figure_title=variant_spec["figure_title"],
                figure_variant_label=variant_spec["figure_variant_label"],
                scope_label=series_spec["scope_label"],
            )
            x_values, y_values = _extract_xy(series_payload)
            summary_stats = _build_summary_stats(x_values, y_values)
            figure_file_name = f"{variant_spec['figure_stem']}.{image_format}"
            figure_path = (
                resolved_output_dir / series_spec["relative_output_subdir"] / figure_file_name
            )
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            rendered = _render_throughput_figure(
                series_payload=series_payload,
                output_path=figure_path,
                image_format=image_format,
                dpi=dpi,
            )
            figure_id = (
                variant_spec["variant_id"]
                if series_spec["series_id"] == "aggregate"
                else f"{series_spec['series_id']}:{variant_spec['variant_id']}"
            )
            if not rendered:
                skipped_variant_ids.append(figure_id)
            figure_entries.append(
                {
                    "series_id": series_spec["series_id"],
                    "gateway_profile_id": _int_or_none(base_series_payload.get("gateway_profile_id")),
                    "variant_id": variant_spec["variant_id"],
                    "figure_id": figure_id,
                    "figure_generated": rendered,
                    "relative_output_subdir": (
                        series_spec["relative_output_subdir"].as_posix()
                        if series_spec["relative_output_subdir"].parts
                        else ""
                    ),
                    "figure_file_name": figure_file_name if rendered else None,
                    "figure_path": str(figure_path.resolve()) if rendered else None,
                    "sample_count": summary_stats["sample_count"],
                    "avg_throughput_requests_per_s": _float_or_none(summary_stats.get("avg")),
                    "min_throughput_requests_per_s": _float_or_none(summary_stats.get("min")),
                    "max_throughput_requests_per_s": _float_or_none(summary_stats.get("max")),
                    "peak_time_s": _float_or_none(summary_stats.get("peak_time_s")),
                    "request_count": _int_or_none(base_series_payload.get("request_count")),
                    "finished_request_count": _int_or_none(series_payload.get("finished_request_count")),
                    "total_duration_s": _float_or_none(base_series_payload.get("total_duration_s")),
                    "window_size_s": _float_or_none(base_series_payload.get("window_size_s")),
                    "timepoint_frequency_hz": _float_or_none(
                        base_series_payload.get("timepoint_frequency_hz")
                    ),
                    "skip_reason": None if rendered else "No valid throughput points",
                }
            )

    primary_figure = figure_entries[0] if figure_entries else None
    manifest = {
        "source_run_dir": str(resolved_run_dir),
        "source_timeseries_path": str(resolved_timeseries_path),
        "output_dir": str(resolved_output_dir),
        "image_format": image_format,
        "dpi": dpi,
        "multi_profile": bool(timeseries_payload.get("multi_profile", False)),
        "port_profile_ids": timeseries_payload.get("port_profile_ids"),
        "series_count": len(series_specs),
        "figure_count": len([item for item in figure_entries if item["figure_generated"]]),
        "variant_count": len(_variant_specs_for_payload(timeseries_payload)),
        "skipped_variant_count": len(skipped_variant_ids),
        "skipped_variant_ids": skipped_variant_ids,
        "figures": figure_entries,
        "figure_generated": primary_figure["figure_generated"] if primary_figure else False,
        "figure_file_name": primary_figure["figure_file_name"] if primary_figure else None,
        "figure_path": primary_figure["figure_path"] if primary_figure else None,
        "sample_count": primary_figure["sample_count"] if primary_figure else 0,
        "avg_throughput_requests_per_s": (
            primary_figure["avg_throughput_requests_per_s"] if primary_figure else None
        ),
        "min_throughput_requests_per_s": (
            primary_figure["min_throughput_requests_per_s"] if primary_figure else None
        ),
        "max_throughput_requests_per_s": (
            primary_figure["max_throughput_requests_per_s"] if primary_figure else None
        ),
        "peak_time_s": primary_figure["peak_time_s"] if primary_figure else None,
        "request_count": _int_or_none(timeseries_payload.get("request_count")),
        "finished_request_count": _int_or_none(timeseries_payload.get("finished_request_count")),
        "finished_request_count_status_200": _int_or_none(
            timeseries_payload.get("finished_request_count_status_200")
        ),
        "non_200_finished_request_count": _int_or_none(
            timeseries_payload.get("non_200_finished_request_count")
        ),
        "total_duration_s": _float_or_none(timeseries_payload.get("total_duration_s")),
        "window_size_s": _float_or_none(timeseries_payload.get("window_size_s")),
        "timepoint_frequency_hz": _float_or_none(timeseries_payload.get("timepoint_frequency_hz")),
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

    run_dirs = discover_run_dirs_with_request_throughput(root_dir)
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
