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
DEFAULT_OUTPUT_DIR_NAME = "split-duration-tables"


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
            "Extract and aggregate split-duration summaries into one CSV table per metric."
        )
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory containing many run result directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory for aggregated CSV tables. "
            f"Default: <root-dir>/{DEFAULT_OUTPUT_DIR_NAME}"
        ),
    )
    parser.add_argument(
        "--split-count",
        type=int,
        default=extract_run.DEFAULT_SPLIT_COUNT,
        help=f"Split count used during extraction (default: {extract_run.DEFAULT_SPLIT_COUNT}).",
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


def _path_sort_key(relative_path: str) -> tuple[tuple[Any, ...], ...]:
    key_parts: list[tuple[Any, ...]] = []
    for part in relative_path.split("/"):
        if part.isdigit():
            key_parts.append((0, int(part), part))
        else:
            key_parts.append((1, part))
    return tuple(key_parts)


def _bin_sort_key(bin_label: str) -> tuple[int, int, str]:
    base = bin_label.rstrip("%")
    parts = base.split("-", 1)
    if len(parts) != 2:
        return (0, 0, bin_label)
    try:
        return (int(parts[0]), int(parts[1]), bin_label)
    except ValueError:
        return (0, 0, bin_label)


def discover_split_duration_summary_paths(root_dir: Path) -> list[Path]:
    summary_paths: set[Path] = set()
    for summary_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name != "duration":
            continue
        if summary_path.parent.parent.name != "split":
            continue
        if summary_path.parent.parent.parent.name != "post-processed":
            continue
        summary_paths.add(summary_path.resolve())
    return sorted(summary_paths)


def _run_dir_from_summary_path(summary_path: Path) -> Path:
    # <run-dir>/post-processed/split/duration/duration-split-summary.json
    return summary_path.parent.parent.parent.parent


def build_tables(
    root_dir: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[str],
]:
    rows_by_metric: dict[str, list[dict[str, Any]]] = {
        metric_name: []
        for metric_name in extract_run.METRIC_NAMES
    }
    discovered_bins: set[str] = set()

    for summary_path in discover_split_duration_summary_paths(root_dir):
        payload = _load_json(summary_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Summary payload must be a JSON object: {summary_path}")

        run_dir = _run_dir_from_summary_path(summary_path)
        relative_path = run_dir.relative_to(root_dir).as_posix()

        bin_labels_payload = payload.get("bin_labels")
        if isinstance(bin_labels_payload, list):
            for label in bin_labels_payload:
                if isinstance(label, str) and label:
                    discovered_bins.add(label)

        tables_payload = payload.get("tables")
        tables = tables_payload if isinstance(tables_payload, dict) else {}
        for metric_name in extract_run.METRIC_NAMES:
            row: dict[str, Any] = {"run_path": relative_path}
            metric_table_payload = tables.get(metric_name)
            metric_table = metric_table_payload if isinstance(metric_table_payload, dict) else {}
            for bin_label, cell_payload in metric_table.items():
                if not isinstance(bin_label, str) or not bin_label:
                    continue
                discovered_bins.add(bin_label)
                if not isinstance(cell_payload, dict):
                    continue
                row[bin_label] = _float_or_none(cell_payload.get("avg"))
            rows_by_metric[metric_name].append(row)

    ordered_bins = sorted(discovered_bins, key=_bin_sort_key)
    for metric_name in rows_by_metric:
        rows_by_metric[metric_name].sort(key=lambda item: _path_sort_key(item["run_path"]))
    return rows_by_metric, ordered_bins


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(value)
    return str(value)


def _extract_run_dir_worker(job: tuple[str, int]) -> tuple[str, str | None, str | None]:
    run_dir_text, split_count = job
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = extract_run.extract_run_dir(
            run_dir,
            split_count=split_count,
        )
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(run_dirs: list[Path], *, split_count: int) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = extract_run.extract_run_dir(
                run_dir,
                split_count=split_count,
            )
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int, split_count: int) -> int:
    failure_count = 0
    jobs = [(str(run_dir), split_count) for run_dir in run_dirs]
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


def _extract_all_runs(root_dir: Path, *, max_procs: int, split_count: int, dry_run: bool) -> int:
    run_dirs = extract_run.discover_run_dirs_with_gateway_output(root_dir)
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
        failure_count = _run_root_dir_sequential(run_dirs, split_count=split_count)
    else:
        try:
            failure_count = _run_root_dir_parallel(
                run_dirs,
                max_procs=worker_count,
                split_count=split_count,
            )
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(run_dirs, split_count=split_count)

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
    if args.split_count <= 0:
        raise ValueError(f"--split-count must be a positive integer: {args.split_count}")

    if args.dry_run:
        run_dirs = extract_run.discover_run_dirs_with_gateway_output(root_dir)
        print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    if not args.skip_extract:
        extract_exit_code = _extract_all_runs(
            root_dir,
            max_procs=args.max_procs,
            split_count=args.split_count,
            dry_run=False,
        )
        if extract_exit_code != 0:
            return extract_exit_code

    rows_by_metric, ordered_bins = build_tables(root_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (root_dir / DEFAULT_OUTPUT_DIR_NAME).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_csv_paths: dict[str, str] = {}
    for metric_name, rows in rows_by_metric.items():
        output_path = output_dir / f"{metric_name}.csv"
        fieldnames = ["run_path", *ordered_bins]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
        metric_csv_paths[metric_name] = str(output_path)
        print(str(output_path))

    manifest_path = output_dir / "split-duration-tables-manifest.json"
    manifest_payload = {
        "source_root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "split_count": args.split_count,
        "bin_labels": ordered_bins,
        "metric_csv_paths": metric_csv_paths,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
