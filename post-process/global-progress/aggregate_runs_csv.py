from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any

import extract_run


DEFAULT_INPUT_NAME = extract_run.DEFAULT_OUTPUT_NAME
DEFAULT_OUTPUT_NAME = "replay-progress-summary.csv"


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
            "Extract and aggregate per-run global-progress milestone summaries into one CSV."
        )
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory containing many run result directories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Optional output CSV path. Default: <root-dir>/{DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--milestone-step",
        type=int,
        default=extract_run.DEFAULT_MILESTONE_STEP,
        help=f"Milestone step used during extraction (default: {extract_run.DEFAULT_MILESTONE_STEP}).",
    )
    parser.add_argument(
        "--max-procs",
        type=int,
        default=_default_max_procs(),
        help=(
            "Number of worker processes for extraction in root-dir mode. "
            "Default: MAX_PROCS env var, else CPU count."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered run directories and exit.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction and only aggregate existing summary files.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _path_sort_key(relative_path: str) -> tuple[tuple[Any, ...], ...]:
    key_parts: list[tuple[Any, ...]] = []
    for part in relative_path.split("/"):
        if part.isdigit():
            key_parts.append((0, int(part), part))
        else:
            key_parts.append((1, part))
    return tuple(key_parts)


def discover_progress_summary_paths(root_dir: Path) -> list[Path]:
    summary_paths: set[Path] = set()
    for summary_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name != "global-progress":
            continue
        if summary_path.parent.parent.name != "post-processed":
            continue
        summary_paths.add(summary_path.resolve())
    return sorted(summary_paths)


def _run_dir_from_summary_path(summary_path: Path) -> Path:
    # <run-dir>/post-processed/global-progress/replay-progress-summary.json
    return summary_path.parent.parent.parent


def _milestone_count_sort_key(column_name: str) -> int:
    prefix = "finish_time_s_at_"
    suffix = column_name[len(prefix) :] if column_name.startswith(prefix) else ""
    try:
        return int(suffix)
    except ValueError:
        return 0


def _build_row_from_summary_path(
    summary_path: Path,
    *,
    root_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Summary payload must be a JSON object: {summary_path}")

    run_dir = _run_dir_from_summary_path(summary_path)
    relative_path = run_dir.relative_to(root_dir).as_posix()

    row: dict[str, Any] = {
        "run_path": relative_path,
        "source_type": payload.get("source_type"),
        "replay_count": _int_or_none(payload.get("replay_count")),
        "finished_replay_count": _int_or_none(payload.get("finished_replay_count")),
        "milestone_step": _int_or_none(payload.get("milestone_step")),
    }
    milestone_columns: list[str] = []

    milestones = payload.get("milestones")
    if isinstance(milestones, list):
        for item in milestones:
            if not isinstance(item, dict):
                continue
            replay_count = _int_or_none(item.get("replay_count"))
            if replay_count is None:
                continue
            column_name = f"finish_time_s_at_{replay_count}"
            row[column_name] = _float_or_none(item.get("finish_time_s"))
            milestone_columns.append(column_name)
    return row, milestone_columns


def build_rows(root_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    milestone_columns: set[str] = set()

    for summary_path in discover_progress_summary_paths(root_dir):
        row, row_milestone_columns = _build_row_from_summary_path(
            summary_path,
            root_dir=root_dir,
        )
        rows.append(row)
        milestone_columns.update(row_milestone_columns)

    rows.sort(key=lambda item: _path_sort_key(item["run_path"]))
    ordered_milestone_columns = sorted(
        milestone_columns,
        key=_milestone_count_sort_key,
    )
    return rows, ordered_milestone_columns


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(value)
    return str(value)


def _extract_run_dir_worker(job: tuple[str, int]) -> tuple[str, str | None, str | None]:
    run_dir_text, milestone_step = job
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = extract_run.extract_run_dir(
            run_dir,
            milestone_step=milestone_step,
        )
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(run_dirs: list[Path], *, milestone_step: int) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = extract_run.extract_run_dir(
                run_dir,
                milestone_step=milestone_step,
            )
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int, milestone_step: int) -> int:
    failure_count = 0
    jobs = [(str(run_dir), milestone_step) for run_dir in run_dirs]
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_path_text, error_text in executor.map(
            _extract_run_dir_worker,
            jobs,
        ):
            if error_text is None:
                print(f"[done] {run_dir_text} -> {output_path_text}")
            else:
                failure_count += 1
                print(f"[error] {run_dir_text}: {error_text}", file=sys.stderr)
    return failure_count


def _extract_all_runs(root_dir: Path, *, max_procs: int, milestone_step: int, dry_run: bool) -> int:
    run_dirs = extract_run.discover_run_dirs_with_global_sources(root_dir)
    print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
    if not run_dirs:
        return 0
    if dry_run:
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    worker_count = min(max_procs, len(run_dirs))
    print(f"Running extraction with {worker_count} worker process(es)")

    if worker_count <= 1:
        failure_count = _run_root_dir_sequential(run_dirs, milestone_step=milestone_step)
    else:
        try:
            failure_count = _run_root_dir_parallel(
                run_dirs,
                max_procs=worker_count,
                milestone_step=milestone_step,
            )
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(run_dirs, milestone_step=milestone_step)

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
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.milestone_step <= 0:
        raise ValueError(f"--milestone-step must be a positive integer: {args.milestone_step}")

    if args.dry_run:
        run_dirs = extract_run.discover_run_dirs_with_global_sources(root_dir)
        print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    if not args.skip_extract:
        extract_exit_code = _extract_all_runs(
            root_dir,
            max_procs=args.max_procs,
            milestone_step=args.milestone_step,
            dry_run=False,
        )
        if extract_exit_code != 0:
            return extract_exit_code

    rows, milestone_columns = build_rows(root_dir)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (root_dir / DEFAULT_OUTPUT_NAME).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_path",
        "source_type",
        "replay_count",
        "finished_replay_count",
        "milestone_step",
        *milestone_columns,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
