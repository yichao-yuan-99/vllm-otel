from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any


DEFAULT_INPUT_NAME = "ctx-aware-timeseries.json"
DEFAULT_INPUT_REL_PATH = Path("post-processed/gateway/ctx-aware-log")
DEFAULT_OUTPUT_REL_PATH = Path("post-processed/visualization/gateway-ctx-aware")
DEFAULT_MANIFEST_NAME = "figures-manifest.json"
DEFAULT_FIGURE_STEM = "ctx-aware-over-time"
DEFAULT_FORMAT = "png"
SUPPORTED_FORMATS = ("png", "pdf", "svg")
SERIES_SPECS = (
    {
        "field": "ongoing_agent_count",
        "title": "Ongoing Agent Count",
        "ylabel": "Agents",
        "color": "#0F766E",
        "kind": "line",
    },
    {
        "field": "pending_agent_count",
        "title": "Pending Agent Count",
        "ylabel": "Agents",
        "color": "#B45309",
        "kind": "line",
    },
    {
        "field": "ongoing_effective_context_tokens",
        "title": "Ongoing Effective Context",
        "ylabel": "Tokens",
        "color": "#1D4ED8",
        "kind": "line",
    },
    {
        "field": "pending_effective_context_tokens",
        "title": "Pending Effective Context",
        "ylabel": "Tokens",
        "color": "#BE123C",
        "kind": "line",
    },
    {
        "field": "agents_turned_pending_due_to_context_threshold",
        "title": "Active to Pending",
        "ylabel": "Agents / Sample",
        "color": "#DC2626",
        "kind": "bar",
    },
    {
        "field": "agents_turned_ongoing",
        "title": "Pending to Active",
        "ylabel": "Agents / Sample",
        "color": "#059669",
        "kind": "bar",
    },
    {
        "field": "new_agents_added_as_pending",
        "title": "New Agents Added as Pending",
        "ylabel": "Agents / Sample",
        "color": "#D97706",
        "kind": "bar",
    },
    {
        "field": "new_agents_added_as_ongoing",
        "title": "New Agents Added as Active",
        "ylabel": "Agents / Sample",
        "color": "#2563EB",
        "kind": "bar",
    },
)


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
            "Generate aggregate and per-profile figures from extracted gateway_multi "
            "ctx-aware timeseries."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing post-processed/gateway/ctx-aware-log/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/gateway/ctx-aware-log/ctx-aware-timeseries.json "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--timeseries-input",
        default=None,
        help=(
            "Optional ctx-aware timeseries input path. Default: "
            "<run-dir>/post-processed/gateway/ctx-aware-log/"
            f"{DEFAULT_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional figure output directory. Default: "
            "<run-dir>/post-processed/visualization/gateway-ctx-aware/"
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
    return (run_dir / DEFAULT_INPUT_REL_PATH / DEFAULT_INPUT_NAME).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_OUTPUT_REL_PATH).resolve()


def discover_run_dirs_with_gateway_ctx_aware(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for timeseries_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not timeseries_path.is_file():
            continue
        if timeseries_path.parent.name != "ctx-aware-log":
            continue
        gateway_dir = timeseries_path.parent.parent
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


def _extract_x_values(timeseries_payload: dict[str, Any]) -> list[float]:
    raw_samples = timeseries_payload.get("samples")
    if not isinstance(raw_samples, list):
        return []
    x_values: list[float] = []
    for sample in raw_samples:
        if not isinstance(sample, dict):
            continue
        second = _float_or_none(sample.get("second"))
        if second is None:
            continue
        x_values.append(second)
    return x_values


def _extract_series_values(timeseries_payload: dict[str, Any], field_name: str) -> list[float]:
    raw_samples = timeseries_payload.get("samples")
    if not isinstance(raw_samples, list):
        return []
    values: list[float] = []
    for sample in raw_samples:
        if not isinstance(sample, dict):
            continue
        value = _float_or_none(sample.get(field_name))
        second = _float_or_none(sample.get("second"))
        if second is None or value is None:
            continue
        values.append(value)
    return values


def _build_summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "sample_count": 0,
            "avg": None,
            "min": None,
            "max": None,
            "total": None,
        }
    return {
        "sample_count": len(values),
        "avg": round(sum(values) / len(values), 6),
        "min": min(values),
        "max": max(values),
        "total": round(sum(values), 6),
    }


def _bar_width(x_values: list[float]) -> float:
    if len(x_values) < 2:
        return 0.16
    deltas = [
        x_values[index] - x_values[index - 1]
        for index in range(1, len(x_values))
        if x_values[index] > x_values[index - 1]
    ]
    if not deltas:
        return 0.16
    return max(0.02, round(statistics.median(deltas) * 0.82, 6))


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate gateway ctx-aware figures. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt


def _render_ctx_aware_figure(
    *,
    timeseries_payload: dict[str, Any],
    output_path: Path,
    image_format: str,
    dpi: int,
    figure_title: str,
    subtitle_prefix: str | None = None,
) -> bool:
    x_values = _extract_x_values(timeseries_payload)
    if not x_values:
        return False

    series_data = {
        spec["field"]: _extract_series_values(timeseries_payload, spec["field"])
        for spec in SERIES_SPECS
    }
    if any(len(series_data[spec["field"]]) != len(x_values) for spec in SERIES_SPECS):
        raise ValueError("Ctx-aware timeseries samples are missing one or more required metrics")

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

    figure, axes = plt.subplots(
        nrows=len(SERIES_SPECS),
        ncols=1,
        sharex=True,
        figsize=(17.0, 29.0),
    )
    bar_width = _bar_width(x_values)

    for axis, spec in zip(axes, SERIES_SPECS):
        y_values = series_data[spec["field"]]
        color = spec["color"]
        axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
        axis.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.3)
        axis.minorticks_on()

        if spec["kind"] == "bar":
            axis.bar(
                x_values,
                y_values,
                width=bar_width,
                color=color,
                alpha=0.85,
                align="center",
            )
        else:
            axis.step(
                x_values,
                y_values,
                where="post",
                color=color,
                linewidth=2.0,
                alpha=0.97,
            )
            axis.fill_between(
                x_values,
                y_values,
                step="post",
                color=color,
                alpha=0.16,
            )

        axis.set_ylabel(spec["ylabel"])
        axis.set_title(spec["title"], loc="left", fontweight="semibold")

    duration_s = _float_or_none(timeseries_payload.get("duration_s"))
    sample_count = timeseries_payload.get("sample_count")
    started_at = timeseries_payload.get("started_at")
    subtitle_parts: list[str] = []
    if subtitle_prefix:
        subtitle_parts.append(subtitle_prefix)
    if sample_count is not None:
        subtitle_parts.append(f"samples: {sample_count}")
    if duration_s is not None:
        subtitle_parts.append(f"duration: {duration_s:.3f}s")
    if isinstance(started_at, str) and started_at:
        subtitle_parts.append(f"job start: {started_at}")

    figure.suptitle(
        figure_title,
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="semibold",
    )
    if subtitle_parts:
        figure.text(
            0.06,
            0.983,
            " | ".join(subtitle_parts),
            ha="left",
            va="top",
            fontsize=10,
            color="#334155",
        )

    axes[-1].set_xlabel("Seconds From Job Start (s)")
    figure.tight_layout(rect=(0.03, 0.03, 0.995, 0.975))
    figure.savefig(output_path, format=image_format, dpi=dpi)
    plt.close(figure)
    return True


def _series_summaries(timeseries_payload: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    return {
        spec["field"]: _build_summary_stats(
            _extract_series_values(timeseries_payload, spec["field"])
        )
        for spec in SERIES_SPECS
    }


def _figure_payloads(timeseries_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_ctx_aware_logs = timeseries_payload.get("ctx_aware_logs")
    if not isinstance(raw_ctx_aware_logs, list) or len(raw_ctx_aware_logs) <= 1:
        return [
            {
                "series_key": "aggregate",
                "display_label": "Aggregate",
                "port_profile_id": None,
                "relative_output_subdir": Path(),
                "figure_file_stem": DEFAULT_FIGURE_STEM,
                "timeseries_payload": timeseries_payload,
                "figure_title": "Gateway Ctx-Aware State Over Time",
                "subtitle_prefix": None,
                "source_ctx_aware_log_path": timeseries_payload.get("source_ctx_aware_log_path"),
            }
        ]

    figures: list[dict[str, Any]] = [
        {
            "series_key": "aggregate",
            "display_label": "Aggregate",
            "port_profile_id": None,
            "relative_output_subdir": Path(),
            "figure_file_stem": DEFAULT_FIGURE_STEM,
            "timeseries_payload": timeseries_payload,
            "figure_title": "Gateway Ctx-Aware State Over Time",
            "subtitle_prefix": "aggregate across all gateway profiles",
            "source_ctx_aware_log_path": None,
        }
    ]

    for raw_payload in raw_ctx_aware_logs:
        if not isinstance(raw_payload, dict):
            continue
        series_key = raw_payload.get("series_key")
        if not isinstance(series_key, str) or not series_key:
            port_profile_id = _int_or_none(raw_payload.get("port_profile_id"))
            if port_profile_id is not None:
                series_key = f"profile-{port_profile_id}"
            else:
                series_key = f"series-{len(figures)}"
        display_label = raw_payload.get("display_label")
        if not isinstance(display_label, str) or not display_label:
            display_label = series_key
        figures.append(
            {
                "series_key": series_key,
                "display_label": display_label,
                "port_profile_id": _int_or_none(raw_payload.get("port_profile_id")),
                "relative_output_subdir": Path(series_key),
                "figure_file_stem": DEFAULT_FIGURE_STEM,
                "timeseries_payload": raw_payload,
                "figure_title": f"Gateway Ctx-Aware State Over Time ({display_label})",
                "subtitle_prefix": f"source: {display_label}",
                "source_ctx_aware_log_path": raw_payload.get("source_ctx_aware_log_path"),
            }
        )
    return figures


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

    figure_manifests: list[dict[str, Any]] = []
    for figure_payload in _figure_payloads(timeseries_payload):
        figure_file_name = f"{figure_payload['figure_file_stem']}.{image_format}"
        figure_path = (
            resolved_output_dir / figure_payload["relative_output_subdir"] / figure_file_name
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        series_payload = figure_payload["timeseries_payload"]
        rendered = _render_ctx_aware_figure(
            timeseries_payload=series_payload,
            output_path=figure_path,
            image_format=image_format,
            dpi=dpi,
            figure_title=figure_payload["figure_title"],
            subtitle_prefix=figure_payload["subtitle_prefix"],
        )
        figure_manifests.append(
            {
                "series_key": figure_payload["series_key"],
                "display_label": figure_payload["display_label"],
                "port_profile_id": figure_payload["port_profile_id"],
                "figure_generated": rendered,
                "relative_output_subdir": (
                    figure_payload["relative_output_subdir"].as_posix()
                    if figure_payload["relative_output_subdir"].parts
                    else ""
                ),
                "figure_file_name": figure_file_name if rendered else None,
                "figure_path": str(figure_path.resolve()) if rendered else None,
                "sample_count": len(_extract_x_values(series_payload)),
                "duration_s": _float_or_none(series_payload.get("duration_s")),
                "started_at": series_payload.get("started_at"),
                "ended_at": series_payload.get("ended_at"),
                "avg_sample_interval_s": _float_or_none(series_payload.get("avg_sample_interval_s")),
                "series_summaries": _series_summaries(series_payload),
                "skip_reason": None if rendered else "No valid ctx-aware samples",
                "source_ctx_aware_log_path": figure_payload["source_ctx_aware_log_path"],
            }
        )

    generated_figures = [item for item in figure_manifests if item["figure_generated"]]
    primary_figure = generated_figures[0] if generated_figures else None
    manifest = {
        "source_run_dir": str(resolved_run_dir),
        "source_timeseries_path": str(resolved_timeseries_path),
        "output_dir": str(resolved_output_dir),
        "image_format": image_format,
        "dpi": dpi,
        "multi_profile": bool(timeseries_payload.get("multi_profile", False)),
        "port_profile_ids": timeseries_payload.get("port_profile_ids", []),
        "figure_count": len(generated_figures),
        "figure_generated": bool(generated_figures),
        "figure_file_name": primary_figure["figure_file_name"] if primary_figure else None,
        "figure_path": primary_figure["figure_path"] if primary_figure else None,
        "sample_count": len(_extract_x_values(timeseries_payload)),
        "duration_s": _float_or_none(timeseries_payload.get("duration_s")),
        "started_at": timeseries_payload.get("started_at"),
        "ended_at": timeseries_payload.get("ended_at"),
        "avg_sample_interval_s": _float_or_none(timeseries_payload.get("avg_sample_interval_s")),
        "series_summaries": _series_summaries(timeseries_payload),
        "skip_reason": None if generated_figures else "No valid ctx-aware samples",
        "figures": figure_manifests,
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

    run_dirs = discover_run_dirs_with_gateway_ctx_aware(root_dir)
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
