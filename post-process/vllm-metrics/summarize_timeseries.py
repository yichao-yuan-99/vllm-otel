from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_INPUT_NAME = "gauge-counter-timeseries.json"
DEFAULT_OUTPUT_NAME = "gauge-counter-timeseries.stats.json"


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
            "Summarize extracted vLLM gauge/counter timeseries metrics with min/max/avg."
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
            "with post-processed/vllm-log/gauge-counter-timeseries.json "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Optional input path. Default: <run-dir>/post-processed/vllm-log/"
            f"{DEFAULT_INPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/post-processed/vllm-log/"
            f"{DEFAULT_OUTPUT_NAME}"
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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_timeseries_payload(
    payload: dict[str, Any],
    *,
    source_timeseries_path: Path | None = None,
) -> dict[str, Any]:
    metrics_payload = payload.get("metrics")
    if not isinstance(metrics_payload, dict):
        raise ValueError("Input payload is missing object field: metrics")

    summary_metrics: dict[str, Any] = {}
    for series_key, metric_payload in sorted(metrics_payload.items()):
        if not isinstance(metric_payload, dict):
            continue
        raw_values = metric_payload.get("value")
        if not isinstance(raw_values, list):
            continue

        numeric_values = []
        for raw in raw_values:
            numeric = _float_or_none(raw)
            if numeric is not None:
                numeric_values.append(numeric)

        if numeric_values:
            min_value = min(numeric_values)
            max_value = max(numeric_values)
            avg_value = sum(numeric_values) / len(numeric_values)
        else:
            min_value = None
            max_value = None
            avg_value = None

        labels = metric_payload.get("labels")
        if not isinstance(labels, dict):
            labels = {}

        summary_metrics[series_key] = {
            "name": metric_payload.get("name"),
            "type": metric_payload.get("type"),
            "labels": labels,
            "sample_count": len(numeric_values),
            "min": min_value,
            "max": max_value,
            "avg": avg_value,
        }

    result = {
        "source_run_dir": payload.get("source_run_dir"),
        "source_vllm_log_dir": payload.get("source_vllm_log_dir"),
        "source_timeseries_path": (
            str(source_timeseries_path.resolve()) if source_timeseries_path is not None else None
        ),
        "cluster_mode": bool(payload.get("cluster_mode", False)),
        "port_profile_ids": payload.get("port_profile_ids", []),
        "metric_count": len(summary_metrics),
        "metrics": summary_metrics,
    }
    return result


def _default_input_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "vllm-log" / DEFAULT_INPUT_NAME).resolve()


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "vllm-log" / DEFAULT_OUTPUT_NAME).resolve()


def discover_run_dirs_with_extracted_timeseries(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for input_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not input_path.is_file():
            continue
        if input_path.parent.name != "vllm-log":
            continue
        if input_path.parent.parent.name != "post-processed":
            continue
        run_dirs.add(input_path.parent.parent.parent.resolve())
    return sorted(run_dirs)


def summarize_run_dir(
    run_dir: Path,
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    resolved_input_path = (input_path or _default_input_path_for_run(run_dir)).expanduser().resolve()
    resolved_output_path = (output_path or _default_output_path_for_run(run_dir)).expanduser().resolve()

    if not resolved_input_path.is_file():
        raise ValueError(f"Missing input timeseries file: {resolved_input_path}")

    payload = json.loads(resolved_input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Input JSON must be an object: {resolved_input_path}")

    result = summarize_timeseries_payload(
        payload,
        source_timeseries_path=resolved_input_path,
    )

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def _summarize_run_dir_worker(run_dir_text: str) -> tuple[str, str | None, str | None]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = summarize_run_dir(run_dir)
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(run_dirs: list[Path]) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = summarize_run_dir(run_dir)
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int) -> int:
    failure_count = 0
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_path_text, error_text in executor.map(
            _summarize_run_dir_worker,
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
    input_path = Path(args.input).expanduser().resolve() if args.input else None
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    resolved_output_path = summarize_run_dir(
        run_dir,
        input_path=input_path,
        output_path=output_path,
    )
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.input:
        raise ValueError("--input can only be used with --run-dir")
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_extracted_timeseries(root_dir)
    print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
    if not run_dirs:
        return 0
    if args.dry_run:
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    worker_count = min(args.max_procs, len(run_dirs))
    print(f"Running summarize with {worker_count} worker process(es)")

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
    print(f"Completed summary for {len(run_dirs)} run directories.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_dir:
        return _main_run_dir(args)
    return _main_root_dir(args)


if __name__ == "__main__":
    raise SystemExit(main())
