from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from contextlib import AbstractContextManager
import os
from pathlib import Path
import sys
from typing import Any

try:
    from rich.progress import (
        BarColumn,
        Progress as RichProgress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
except ImportError:  # pragma: no cover
    RichProgress = None

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import count_metric_blocks_in_run_dir, extract_gauge_counter_timeseries_from_run_dir


DEFAULT_OUTPUT_NAME = "gauge-counter-timeseries.json"


class _NullProgress(AbstractContextManager["_NullProgress"]):
    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def add_task(self, description: str, *, total: int, **fields: Any) -> int:
        return 0

    def update(self, task_id: int, *, advance: int = 0, **fields: Any) -> None:
        return None


def create_extract_progress() -> Any:
    if RichProgress is None:
        return _NullProgress()
    return RichProgress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} blocks"),
        TimeElapsedColumn(),
        transient=False,
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
            "Extract vLLM gauge/counter timeseries from a run's vllm-log directory."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing vllm-log/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with a direct vllm-log/ child will be processed."
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


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "vllm-log" / DEFAULT_OUTPUT_NAME).resolve()


def discover_run_dirs_with_vllm_log(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for vllm_log_dir in root_dir.rglob("vllm-log"):
        if not vllm_log_dir.is_dir():
            continue
        # Ignore outputs produced by post-process itself:
        # <run-dir>/post-processed/vllm-log
        if vllm_log_dir.parent.name == "post-processed":
            continue
        run_dirs.add(vllm_log_dir.parent.resolve())
    return sorted(run_dirs)


def extract_run_dir(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    show_progress: bool = True,
) -> Path:
    if show_progress:
        total_blocks = count_metric_blocks_in_run_dir(run_dir)
        with create_extract_progress() as progress:
            task_id = progress.add_task(
                "extracting vllm metrics",
                total=max(total_blocks, 1),
            )

            def _on_block_loaded(completed: int, total: int) -> None:
                progress.update(task_id, completed=completed, total=max(total, 1))

            result = extract_gauge_counter_timeseries_from_run_dir(
                run_dir,
                on_block_loaded=_on_block_loaded,
            )
            if total_blocks == 0:
                progress.update(task_id, completed=1, total=1)
    else:
        result = extract_gauge_counter_timeseries_from_run_dir(run_dir)

    resolved_output_path = output_path or _default_output_path_for_run(run_dir)
    resolved_output_path = resolved_output_path.expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def _main_run_dir(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = None
    resolved_output_path = extract_run_dir(run_dir, output_path=output_path)
    print(str(resolved_output_path))
    return 0


def _extract_run_dir_worker(run_dir_text: str) -> tuple[str, str | None, str | None]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = extract_run_dir(run_dir, show_progress=False)
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


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_vllm_log(root_dir)
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
