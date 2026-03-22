from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_POWER_SUMMARY_INPUT_NAME = "power-summary.json"
DEFAULT_PREFILL_TIMESERIES_INPUT_NAME = "prefill-concurrency-timeseries.json"
DEFAULT_OUTPUT_NAME = "power-sampling-summary.json"


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
            "Sample GPU power at prefill-concurrency ticks and summarize power stats "
            "for each prefill-concurrency value."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/power/power-summary.json and "
            "post-processed/prefill-concurrency/prefill-concurrency-timeseries.json "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/post-processed/power-sampling/"
            f"{DEFAULT_OUTPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--power-summary",
        default=None,
        help=(
            "Optional input path to power-summary.json (only for --run-dir). "
            "Default: <run-dir>/post-processed/power/"
            f"{DEFAULT_POWER_SUMMARY_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--prefill-timeseries",
        default=None,
        help=(
            "Optional input path to prefill-concurrency-timeseries.json "
            "(only for --run-dir). Default: <run-dir>/post-processed/"
            "prefill-concurrency/"
            f"{DEFAULT_PREFILL_TIMESERIES_INPUT_NAME}"
        ),
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


def _default_power_summary_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "power" / DEFAULT_POWER_SUMMARY_INPUT_NAME).resolve()


def _default_prefill_timeseries_path_for_run(run_dir: Path) -> Path:
    return (
        run_dir
        / "post-processed"
        / "prefill-concurrency"
        / DEFAULT_PREFILL_TIMESERIES_INPUT_NAME
    ).resolve()


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "power-sampling" / DEFAULT_OUTPUT_NAME).resolve()


def discover_run_dirs_with_power_sampling_inputs(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for prefill_timeseries_path in root_dir.rglob(DEFAULT_PREFILL_TIMESERIES_INPUT_NAME):
        if not prefill_timeseries_path.is_file():
            continue
        if prefill_timeseries_path.parent.name != "prefill-concurrency":
            continue
        if prefill_timeseries_path.parent.parent.name != "post-processed":
            continue

        run_dir = prefill_timeseries_path.parent.parent.parent.resolve()
        if _default_power_summary_path_for_run(run_dir).is_file():
            run_dirs.add(run_dir)
    return sorted(run_dirs)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing required file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return payload


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


def _extract_power_points(power_summary_payload: dict[str, Any]) -> list[tuple[float, float]]:
    raw_points = power_summary_payload.get("power_points")
    if not isinstance(raw_points, list):
        return []

    points: list[tuple[float, float]] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        power_w = _float_or_none(point.get("power_w"))
        if time_offset_s is None or power_w is None:
            continue
        points.append((time_offset_s, power_w))

    if not points:
        return []

    points.sort(key=lambda pair: pair[0])
    deduped_points: list[tuple[float, float]] = []
    for time_offset_s, power_w in points:
        if deduped_points and deduped_points[-1][0] == time_offset_s:
            deduped_points[-1] = (time_offset_s, power_w)
        else:
            deduped_points.append((time_offset_s, power_w))
    return deduped_points


def _extract_prefill_points(prefill_timeseries_payload: dict[str, Any]) -> list[tuple[float, int]]:
    raw_points = prefill_timeseries_payload.get("concurrency_points")
    if not isinstance(raw_points, list):
        return []

    points: list[tuple[float, int]] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        time_offset_s = _float_or_none(point.get("time_offset_s"))
        concurrency = _int_or_none(point.get("concurrency"))
        if time_offset_s is None or concurrency is None:
            continue
        points.append((time_offset_s, concurrency))

    points.sort(key=lambda pair: pair[0])
    return points


def _sample_power_for_times(
    power_points: list[tuple[float, float]],
    sample_times_s: list[float],
) -> list[float]:
    if not power_points:
        return []

    if len(power_points) == 1:
        return [power_points[0][1] for _ in sample_times_s]

    first_time_s, first_power_w = power_points[0]
    last_time_s, last_power_w = power_points[-1]

    sampled_power_w: list[float] = []
    left_index = 0
    for sample_time_s in sample_times_s:
        if sample_time_s <= first_time_s:
            sampled_power_w.append(first_power_w)
            continue
        if sample_time_s >= last_time_s:
            sampled_power_w.append(last_power_w)
            continue

        while (
            left_index + 1 < len(power_points)
            and power_points[left_index + 1][0] < sample_time_s
        ):
            left_index += 1

        left_time_s, left_power_w = power_points[left_index]
        right_time_s, right_power_w = power_points[left_index + 1]

        if right_time_s <= left_time_s:
            sampled_power_w.append(right_power_w)
            continue

        ratio = (sample_time_s - left_time_s) / (right_time_s - left_time_s)
        sampled_power_w.append(left_power_w + (right_power_w - left_power_w) * ratio)

    return sampled_power_w


def _build_power_stats(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "sample_count": 0,
            "avg_power_w": None,
            "min_power_w": None,
            "max_power_w": None,
            "std_power_w": None,
        }

    sample_count = len(values)
    avg_power_w = sum(values) / sample_count
    variance = sum((value - avg_power_w) ** 2 for value in values) / sample_count
    std_power_w = math.sqrt(variance)

    return {
        "sample_count": sample_count,
        "avg_power_w": round(avg_power_w, 6),
        "min_power_w": round(min(values), 6),
        "max_power_w": round(max(values), 6),
        "std_power_w": round(std_power_w, 6),
    }


def extract_power_sampling_summary_from_run_dir(
    run_dir: Path,
    *,
    power_summary_path: Path | None = None,
    prefill_timeseries_path: Path | None = None,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_power_summary_path = (
        power_summary_path or _default_power_summary_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    resolved_prefill_timeseries_path = (
        prefill_timeseries_path or _default_prefill_timeseries_path_for_run(resolved_run_dir)
    ).expanduser().resolve()

    power_summary_payload = _load_json_object(resolved_power_summary_path)
    prefill_timeseries_payload = _load_json_object(resolved_prefill_timeseries_path)

    power_points = _extract_power_points(power_summary_payload)
    prefill_points = _extract_prefill_points(prefill_timeseries_payload)

    sample_times_s = [time_offset_s for time_offset_s, _ in prefill_points]
    sampled_power_w = _sample_power_for_times(power_points, sample_times_s)
    sampled_tick_count = len(sampled_power_w)

    power_by_concurrency: dict[int, list[float]] = {}
    for index, power_w in enumerate(sampled_power_w):
        concurrency = prefill_points[index][1]
        power_by_concurrency.setdefault(concurrency, []).append(power_w)

    concurrency_power_stats_w: dict[str, dict[str, int | float | None]] = {}
    for concurrency in sorted(power_by_concurrency):
        concurrency_power_stats_w[str(concurrency)] = {
            "concurrency": concurrency,
            **_build_power_stats(power_by_concurrency[concurrency]),
        }

    non_zero_values = [
        power_w
        for concurrency, values in power_by_concurrency.items()
        if concurrency != 0
        for power_w in values
    ]

    return {
        "source_run_dir": str(resolved_run_dir),
        "source_power_summary_path": str(resolved_power_summary_path),
        "source_prefill_concurrency_timeseries_path": str(resolved_prefill_timeseries_path),
        "source_type": power_summary_payload.get("source_type"),
        "service_failure_detected": bool(
            power_summary_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": power_summary_payload.get(
            "service_failure_cutoff_time_utc"
        ),
        "power_log_found": bool(power_summary_payload.get("power_log_found", False)),
        "request_count": _int_or_none(prefill_timeseries_payload.get("request_count")),
        "prefill_activity_count": _int_or_none(
            prefill_timeseries_payload.get("prefill_activity_count")
        ),
        "total_duration_s": _float_or_none(prefill_timeseries_payload.get("total_duration_s")),
        "tick_ms": _int_or_none(prefill_timeseries_payload.get("tick_ms")),
        "tick_s": _float_or_none(prefill_timeseries_payload.get("tick_s")),
        "prefill_tick_count": len(prefill_points),
        "power_point_count": len(power_points),
        "sampled_tick_count": sampled_tick_count,
        "all_power_stats_w": _build_power_stats(sampled_power_w),
        "non_zero_power_stats_w": _build_power_stats(non_zero_values),
        "concurrency_power_stats_w": concurrency_power_stats_w,
        "sampling_method": {
            "interpolation": "linear",
            "outside_power_range": "clamp_to_nearest_endpoint",
        },
    }


def extract_run_dir(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    power_summary_path: Path | None = None,
    prefill_timeseries_path: Path | None = None,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_path = (output_path or _default_output_path_for_run(resolved_run_dir)).expanduser().resolve()
    result = extract_power_sampling_summary_from_run_dir(
        resolved_run_dir,
        power_summary_path=power_summary_path,
        prefill_timeseries_path=prefill_timeseries_path,
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def _extract_run_dir_worker(run_dir_text: str) -> tuple[str, str | None, str | None]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = extract_run_dir(run_dir)
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(run_dirs: list[Path]) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = extract_run_dir(run_dir)
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int) -> int:
    failure_count = 0
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_path_text, error_text in executor.map(
            _extract_run_dir_worker,
            [str(run_dir) for run_dir in run_dirs],
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
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    power_summary_path = (
        Path(args.power_summary).expanduser().resolve() if args.power_summary else None
    )
    prefill_timeseries_path = (
        Path(args.prefill_timeseries).expanduser().resolve()
        if args.prefill_timeseries
        else None
    )
    resolved_output_path = extract_run_dir(
        run_dir,
        output_path=output_path,
        power_summary_path=power_summary_path,
        prefill_timeseries_path=prefill_timeseries_path,
    )
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.power_summary:
        raise ValueError("--power-summary can only be used with --run-dir")
    if args.prefill_timeseries:
        raise ValueError("--prefill-timeseries can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_power_sampling_inputs(root_dir)
    print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
    if not run_dirs:
        return 0
    if args.dry_run:
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    worker_count = min(args.max_procs, len(run_dirs))
    print(f"Running extraction with {worker_count} worker process(es)")

    if worker_count <= 1:
        failure_count = _run_root_dir_sequential(run_dirs)
    else:
        try:
            failure_count = _run_root_dir_parallel(run_dirs, max_procs=worker_count)
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(run_dirs)

    if failure_count:
        print(
            f"Completed with {failure_count} failure(s) out of {len(run_dirs)} run directories.",
            file=sys.stderr,
        )
        return 1
    print(f"Completed extraction for {len(run_dirs)} run directories.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_dir:
        return _main_run_dir(args)
    return _main_root_dir(args)


if __name__ == "__main__":
    raise SystemExit(main())
