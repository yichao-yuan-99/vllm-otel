from __future__ import annotations

import argparse
import json
from contextlib import AbstractContextManager
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract vLLM gauge/counter timeseries from a run's vllm-log directory."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run result root directory containing vllm-log/.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/post-processed/vllm-log/"
            f"{DEFAULT_OUTPUT_NAME}"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
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

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = (
            run_dir / "post-processed" / "vllm-log" / DEFAULT_OUTPUT_NAME
        ).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
