from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_INPUT_NAME = "run-stats-summary.json"
DEFAULT_OUTPUT_STEM = "job-max-request-length-cdf"
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
            "Generate per-run CDF figures for job context usage. "
            "x-axis is per-job max request length and y-axis is cumulative fraction "
            "of jobs in [0, 1]."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="One run directory.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for runs that already have "
            "run-stats/run-stats-summary.json."
        ),
    )
    parser.add_argument(
        "--summary-input",
        default=None,
        help=(
            "Optional run summary path. Default: "
            "<run-dir>/run-stats/run-stats-summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output figure path. Default: "
            "<run-dir>/run-stats/job-max-request-length-cdf.<format>"
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


def _default_summary_input_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "run-stats" / DEFAULT_INPUT_NAME).resolve()


def _default_output_path_for_run(run_dir: Path, *, image_format: str) -> Path:
    return (
        run_dir / "run-stats" / f"{DEFAULT_OUTPUT_STEM}.{image_format}"
    ).resolve()


def discover_run_dirs_with_run_stats(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for summary_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name != "run-stats":
            continue
        run_dirs.add(summary_path.parent.parent.resolve())
    return sorted(run_dirs)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        signed = stripped[1:] if stripped[0] in {"+", "-"} else stripped
        if signed.isdigit():
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _extract_job_max_request_lengths(summary_payload: dict[str, Any]) -> list[int]:
    lengths: list[int] = []
    raw_lengths = summary_payload.get("job_max_request_lengths")
    if isinstance(raw_lengths, list):
        for raw in raw_lengths:
            parsed = _int_or_none(raw)
            if parsed is None:
                continue
            lengths.append(parsed)
    if lengths:
        return lengths

    jobs_payload = summary_payload.get("jobs")
    if not isinstance(jobs_payload, list):
        return []
    for job in jobs_payload:
        if not isinstance(job, dict):
            continue
        parsed = _int_or_none(job.get("max_request_length"))
        if parsed is None:
            continue
        lengths.append(parsed)
    return lengths


def _build_cdf_points(lengths: list[int]) -> tuple[list[int], list[float]]:
    sorted_lengths = sorted(lengths)
    if not sorted_lengths:
        return [], []
    total = len(sorted_lengths)
    y_values = [index / total for index in range(1, total + 1)]
    return sorted_lengths, y_values


def _quantile_from_sorted_values(sorted_values: list[int], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile from empty values.")
    if quantile <= 0:
        return float(sorted_values[0])
    if quantile >= 1:
        return float(sorted_values[-1])

    index = quantile * (len(sorted_values) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(sorted_values[low])
    weight_high = index - low
    return (
        sorted_values[low] * (1.0 - weight_high)
        + sorted_values[high] * weight_high
    )


def _import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate CDF figures. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt


def _render_cdf_figure(
    *,
    x_values: list[int],
    y_values: list[float],
    title: str,
    output_path: Path,
    image_format: str,
    dpi: int,
) -> None:
    plt = _import_matplotlib_pyplot()
    figure, axis = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)

    axis.step(x_values, y_values, where="post", color="#006d77", linewidth=2.0)
    axis.scatter(x_values, y_values, color="#005f73", s=14, alpha=0.9, zorder=3)

    axis.set_xlabel("Per-job max request length (tokens)")
    axis.set_ylabel("Cumulative fraction of jobs")
    axis.set_ylim(0.0, 1.0)
    if x_values:
        x_min = x_values[0]
        x_max = x_values[-1]
        if x_min == x_max:
            axis.set_xlim(x_min - 1, x_max + 1)
        else:
            axis.set_xlim(x_min, x_max)

    stats_text = (
        f"jobs: {len(x_values)}\n"
        f"min: {x_values[0]}\n"
        f"p50: {_quantile_from_sorted_values(x_values, 0.5):.1f}\n"
        f"p80: {_quantile_from_sorted_values(x_values, 0.8):.1f}\n"
        f"p90: {_quantile_from_sorted_values(x_values, 0.9):.1f}\n"
        f"p95: {_quantile_from_sorted_values(x_values, 0.95):.1f}\n"
        f"max: {x_values[-1]}"
    )
    axis.text(
        0.985,
        0.02,
        stats_text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#edf6f9",
            "edgecolor": "#83c5be",
            "alpha": 0.95,
        },
    )

    axis.set_title(title)
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format=image_format, dpi=dpi)
    plt.close(figure)


def generate_cdf_for_run_dir(
    run_dir: Path,
    *,
    summary_input_path: Path | None = None,
    output_path: Path | None = None,
    image_format: str = DEFAULT_FORMAT,
    dpi: int = 220,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_summary_input_path = (
        summary_input_path.expanduser().resolve()
        if summary_input_path
        else _default_summary_input_path_for_run(resolved_run_dir)
    )
    if not resolved_summary_input_path.is_file():
        raise ValueError(f"Missing run stats summary: {resolved_summary_input_path}")

    resolved_output_path = (
        output_path.expanduser().resolve()
        if output_path
        else _default_output_path_for_run(resolved_run_dir, image_format=image_format)
    )

    summary_payload = _load_json(resolved_summary_input_path)
    if not isinstance(summary_payload, dict):
        raise ValueError(
            f"Run stats summary payload must be a JSON object: {resolved_summary_input_path}"
        )

    job_max_request_lengths = _extract_job_max_request_lengths(summary_payload)
    if not job_max_request_lengths:
        raise ValueError(
            "No job max request lengths found in run summary. "
            "Expected job_max_request_lengths or jobs[*].max_request_length."
        )

    x_values, y_values = _build_cdf_points(job_max_request_lengths)
    dataset = _string_or_none(summary_payload.get("dataset")) or "unknown-dataset"
    agent_type = _string_or_none(summary_payload.get("agent_type")) or "unknown-agent"
    title = f"Job Context Usage CDF ({dataset}, {agent_type})"
    _render_cdf_figure(
        x_values=x_values,
        y_values=y_values,
        title=title,
        output_path=resolved_output_path,
        image_format=image_format,
        dpi=dpi,
    )
    return resolved_output_path


def _generate_cdf_for_run_dir_worker(
    run_dir_text: str,
    image_format: str,
    dpi: int,
) -> tuple[str, str | None, str | None]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = generate_cdf_for_run_dir(
            run_dir,
            image_format=image_format,
            dpi=dpi,
        )
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(
    run_dirs: list[Path],
    *,
    image_format: str,
    dpi: int,
) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = generate_cdf_for_run_dir(
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
    image_format: str,
    dpi: int,
    max_procs: int,
) -> int:
    failure_count = 0
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_path_text, error_text in executor.map(
            _generate_cdf_for_run_dir_worker,
            [str(run_dir) for run_dir in run_dirs],
            [image_format] * len(run_dirs),
            [dpi] * len(run_dirs),
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
    summary_input_path = (
        Path(args.summary_input).expanduser().resolve()
        if args.summary_input
        else None
    )
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    resolved_output_path = generate_cdf_for_run_dir(
        run_dir,
        summary_input_path=summary_input_path,
        output_path=output_path,
        image_format=args.format,
        dpi=args.dpi,
    )
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.summary_input:
        raise ValueError("--summary-input can only be used with --run-dir")
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_run_stats(root_dir)
    print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
    if not run_dirs:
        return 0
    if args.dry_run:
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    worker_count = min(args.max_procs, len(run_dirs))
    print(f"Generating CDF figures with {worker_count} worker process(es)")

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
                image_format=args.format,
                dpi=args.dpi,
                max_procs=worker_count,
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
    print(f"Completed CDF figure generation for {len(run_dirs)} run directories.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be a positive integer: {args.dpi}")
    if args.run_dir:
        return _main_run_dir(args)
    return _main_root_dir(args)


if __name__ == "__main__":
    raise SystemExit(main())
