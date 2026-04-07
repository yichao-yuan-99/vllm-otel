from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


DEFAULT_TIMESERIES_INPUT_NAME = "gauge-counter-timeseries.json"
DEFAULT_STATS_INPUT_NAME = "gauge-counter-timeseries.stats.json"
DEFAULT_OUTPUT_REL_PATH = Path("post-processed/visualization/vllm-metrics")
DEFAULT_MANIFEST_NAME = "figures-manifest.json"
DEFAULT_FORMAT = "png"
SUPPORTED_FORMATS = ("png", "pdf", "svg")

_FILENAME_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


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
            "Generate line-chart figures for extracted vLLM timeseries metrics."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing post-processed/vllm-log/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/vllm-log/gauge-counter-timeseries.stats.json "
            "and gauge-counter-timeseries.json will be processed."
        ),
    )
    parser.add_argument(
        "--timeseries-input",
        default=None,
        help=(
            "Optional timeseries input path. Default: "
            "<run-dir>/post-processed/vllm-log/"
            f"{DEFAULT_TIMESERIES_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--stats-input",
        default=None,
        help=(
            "Optional stats input path. Default: "
            "<run-dir>/post-processed/vllm-log/"
            f"{DEFAULT_STATS_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional figure output directory. Default: "
            "<run-dir>/post-processed/visualization/vllm-metrics/"
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
    return (
        run_dir / "post-processed" / "vllm-log" / DEFAULT_TIMESERIES_INPUT_NAME
    ).resolve()


def _default_stats_input_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "vllm-log" / DEFAULT_STATS_INPUT_NAME).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_OUTPUT_REL_PATH).resolve()


def discover_run_dirs_with_metric_stats(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for stats_path in root_dir.rglob(DEFAULT_STATS_INPUT_NAME):
        if not stats_path.is_file():
            continue
        if stats_path.parent.name != "vllm-log":
            continue
        if stats_path.parent.parent.name != "post-processed":
            continue
        timeseries_path = stats_path.parent / DEFAULT_TIMESERIES_INPUT_NAME
        if not timeseries_path.is_file():
            continue
        run_dirs.add(stats_path.parent.parent.parent.resolve())
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


def _extract_series_xy(metric_payload: dict[str, Any]) -> tuple[list[float], list[float]]:
    raw_times = metric_payload.get("time_from_start_s")
    raw_values = metric_payload.get("value")
    if not isinstance(raw_times, list) or not isinstance(raw_values, list):
        return [], []

    pairs: list[tuple[float, float]] = []
    for raw_time, raw_value in zip(raw_times, raw_values):
        x_value = _float_or_none(raw_time)
        y_value = _float_or_none(raw_value)
        if x_value is None or y_value is None:
            continue
        pairs.append((x_value, y_value))
    if not pairs:
        return [], []
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _format_stat_value(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.6g}"


def _build_stats_annotation(stats_payload: dict[str, Any] | None) -> str:
    if not isinstance(stats_payload, dict):
        return "n: n/a\navg: n/a\nmin: n/a\nmax: n/a"
    sample_count = _int_or_none(stats_payload.get("sample_count"))
    return (
        f"n: {sample_count if sample_count is not None else 'n/a'}\n"
        f"avg: {_format_stat_value(stats_payload.get('avg'))}\n"
        f"min: {_format_stat_value(stats_payload.get('min'))}\n"
        f"max: {_format_stat_value(stats_payload.get('max'))}"
    )


def _label_text(metric_payload: dict[str, Any]) -> str:
    labels = metric_payload.get("labels")
    if not isinstance(labels, dict) or not labels:
        return ""
    parts = [f"{key}={labels[key]}" for key in sorted(labels)]
    label_line = ", ".join(parts)
    if len(label_line) > 160:
        return label_line[:157] + "..."
    return label_line


def _sanitize_filename(name: str, *, max_length: int = 120) -> str:
    cleaned = _FILENAME_SAFE_PATTERN.sub("_", name).strip("._-")
    if not cleaned:
        cleaned = "metric"
    return cleaned[:max_length]


def _build_figure_file_name(
    series_key: str,
    *,
    index: int,
    extension: str,
    used_names: set[str],
) -> str:
    base = f"{index:04d}-{_sanitize_filename(series_key)}"
    candidate = f"{base}.{extension}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    suffix = hashlib.sha1(series_key.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}-{suffix}.{extension}"
    used_names.add(candidate)
    return candidate


def _metric_output_subdir(metric_payload: dict[str, Any], *, cluster_mode: bool) -> Path:
    if not cluster_mode:
        return Path()
    labels = metric_payload.get("labels")
    if isinstance(labels, dict):
        port_profile_id = labels.get("port_profile_id")
        if isinstance(port_profile_id, str) and port_profile_id.strip():
            return Path(f"profile-{port_profile_id.strip()}")
    return Path("shared")


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate vLLM metric figures. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt


def _render_metric_figure(
    *,
    series_key: str,
    metric_payload: dict[str, Any],
    stats_payload: dict[str, Any] | None,
    output_path: Path,
    image_format: str,
    dpi: int,
) -> bool:
    x_values, y_values = _extract_series_xy(metric_payload)
    if not x_values:
        return False

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
    axis.plot(x_values, y_values, color="#1F4E79", linewidth=2.0, alpha=0.95)
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
    axis.minorticks_on()

    metric_name = metric_payload.get("name")
    if not isinstance(metric_name, str) or not metric_name:
        metric_name = series_key
    axis.set_title(metric_name, loc="left", fontweight="semibold")

    labels_line = _label_text(metric_payload)
    if labels_line:
        axis.text(
            0.0,
            1.02,
            labels_line,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#2A3B47",
        )

    axis.set_xlabel("Time From Start (s)")
    axis.set_ylabel("Value")

    axis.text(
        0.99,
        0.98,
        _build_stats_annotation(stats_payload),
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
    timeseries_input_path: Path | None = None,
    stats_input_path: Path | None = None,
    output_dir: Path | None = None,
    image_format: str = DEFAULT_FORMAT,
    dpi: int = 220,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_timeseries_path = (
        timeseries_input_path or _default_timeseries_input_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    resolved_stats_path = (
        stats_input_path or _default_stats_input_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    resolved_output_dir = (
        output_dir or _default_output_dir_for_run(resolved_run_dir)
    ).expanduser().resolve()

    if not resolved_timeseries_path.is_file():
        raise ValueError(f"Missing timeseries file: {resolved_timeseries_path}")
    if not resolved_stats_path.is_file():
        raise ValueError(f"Missing stats file: {resolved_stats_path}")
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer: {dpi}")

    timeseries_payload = _load_json(resolved_timeseries_path)
    if not isinstance(timeseries_payload, dict):
        raise ValueError(f"Timeseries JSON must be an object: {resolved_timeseries_path}")
    timeseries_metrics = timeseries_payload.get("metrics")
    if not isinstance(timeseries_metrics, dict):
        raise ValueError(
            "Timeseries payload is missing object field metrics: "
            f"{resolved_timeseries_path}"
        )

    stats_payload = _load_json(resolved_stats_path)
    if not isinstance(stats_payload, dict):
        raise ValueError(f"Stats JSON must be an object: {resolved_stats_path}")
    stats_metrics = stats_payload.get("metrics")
    if not isinstance(stats_metrics, dict):
        raise ValueError(
            "Stats payload is missing object field metrics: "
            f"{resolved_stats_path}"
        )
    cluster_mode = bool(timeseries_payload.get("cluster_mode", False))
    port_profile_ids = timeseries_payload.get("port_profile_ids", [])

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    used_file_names_by_subdir: dict[Path, set[str]] = {}
    figure_entries: list[dict[str, Any]] = []
    skipped_series_keys: list[str] = []

    for index, (series_key, metric_payload) in enumerate(sorted(timeseries_metrics.items()), start=1):
        if not isinstance(metric_payload, dict):
            skipped_series_keys.append(series_key)
            continue
        relative_subdir = _metric_output_subdir(metric_payload, cluster_mode=cluster_mode)
        used_file_names = used_file_names_by_subdir.setdefault(relative_subdir, set())
        file_name = _build_figure_file_name(
            series_key,
            index=index,
            extension=image_format,
            used_names=used_file_names,
        )
        figure_dir = (resolved_output_dir / relative_subdir).resolve()
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_path = figure_dir / file_name
        rendered = _render_metric_figure(
            series_key=series_key,
            metric_payload=metric_payload,
            stats_payload=stats_metrics.get(series_key)
            if isinstance(stats_metrics.get(series_key), dict)
            else None,
            output_path=figure_path,
            image_format=image_format,
            dpi=dpi,
        )
        if not rendered:
            skipped_series_keys.append(series_key)
            continue
        figure_entries.append(
            {
                "series_key": series_key,
                "metric_name": metric_payload.get("name"),
                "relative_output_subdir": relative_subdir.as_posix() if relative_subdir.parts else "",
                "figure_file_name": file_name,
                "figure_path": str(figure_path.resolve()),
                "sample_count": len(_extract_series_xy(metric_payload)[0]),
            }
        )

    manifest = {
        "source_run_dir": str(resolved_run_dir),
        "source_timeseries_path": str(resolved_timeseries_path),
        "source_stats_path": str(resolved_stats_path),
        "output_dir": str(resolved_output_dir),
        "image_format": image_format,
        "dpi": dpi,
        "cluster_mode": cluster_mode,
        "port_profile_ids": port_profile_ids,
        "metric_count": len(timeseries_metrics),
        "figure_count": len(figure_entries),
        "skipped_metric_count": len(skipped_series_keys),
        "skipped_metric_series_keys": skipped_series_keys,
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
    timeseries_input_path = (
        Path(args.timeseries_input).expanduser().resolve()
        if args.timeseries_input
        else None
    )
    stats_input_path = (
        Path(args.stats_input).expanduser().resolve()
        if args.stats_input
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_path = generate_figures_for_run_dir(
        run_dir,
        timeseries_input_path=timeseries_input_path,
        stats_input_path=stats_input_path,
        output_dir=output_dir,
        image_format=args.format,
        dpi=args.dpi,
    )
    print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.timeseries_input:
        raise ValueError("--timeseries-input can only be used with --run-dir")
    if args.stats_input:
        raise ValueError("--stats-input can only be used with --run-dir")
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be a positive integer: {args.dpi}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_metric_stats(root_dir)
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
