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
DEFAULT_OUTPUT_DIR_NAME = "split-top-p-ratio-tables"
TOKEN_RATIO_NAME = "top_p_token_usage_ratio"
CONTEXT_RATIO_NAME = "top_p_context_usage_ratio"


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
            "Extract and aggregate top-p split ratios into one CSV table per series."
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
    return None


def _path_sort_key(relative_path: str) -> tuple[tuple[Any, ...], ...]:
    key_parts: list[tuple[Any, ...]] = []
    for part in relative_path.split("/"):
        if part.isdigit():
            key_parts.append((0, int(part), part))
        else:
            key_parts.append((1, part))
    return tuple(key_parts)


def discover_split_summary_paths(root_dir: Path) -> list[Path]:
    summary_paths: set[Path] = set()
    for summary_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name != "split":
            continue
        if summary_path.parent.parent.name != "original-analysis":
            continue
        summary_paths.add(summary_path.resolve())
    return sorted(summary_paths)


def _run_dir_from_summary_path(summary_path: Path) -> Path:
    # <run-dir>/original-analysis/split/top-p-usage-ratio-summary.json
    return summary_path.parent.parent.parent


def _column_name_for_percentile(p: int) -> str:
    return f"p{p}"


def _series_from_summary(payload: dict[str, Any], series_name: str) -> list[float | None]:
    ratio_definition = payload.get("ratio_definition")
    has_top_rest_definition = ratio_definition == "top/rest"

    by_percentile_payload = payload.get("by_percentile")
    if isinstance(by_percentile_payload, list):
        field_name: str
        top_total_field: str
        rest_total_field: str
        if series_name == TOKEN_RATIO_NAME:
            field_name = "top_token_usage_ratio"
            top_total_field = "top_token_usage_total"
            rest_total_field = "rest_token_usage_total"
        else:
            field_name = "top_context_usage_ratio"
            top_total_field = "top_context_length_total"
            rest_total_field = "rest_context_length_total"

        rows = []
        for item in by_percentile_payload:
            if not isinstance(item, dict):
                continue
            p = _int_or_none(item.get("p"))
            if p is None:
                continue

            if has_top_rest_definition:
                rows.append((p, _float_or_none(item.get(field_name))))
                continue

            # Backward compatibility:
            # Older split summaries stored top/total in top_*_ratio.
            # Recompute top/rest from raw totals when available.
            top_total = _float_or_none(item.get(top_total_field))
            rest_total = _float_or_none(item.get(rest_total_field))
            if top_total is not None and rest_total is not None:
                if rest_total > 0:
                    rows.append((p, top_total / rest_total))
                else:
                    rows.append((p, None))
            else:
                rows.append((p, _float_or_none(item.get(field_name))))

        if rows:
            rows.sort(key=lambda item: item[0])
            return [item[1] for item in rows]

    table_payload = payload.get("table_2x99")
    if isinstance(table_payload, dict):
        series_payload = table_payload.get(series_name)
        if isinstance(series_payload, list):
            values: list[float | None] = []
            for item in series_payload:
                values.append(_float_or_none(item))
            if values:
                return values
    return []


def build_rows(
    root_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    token_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    max_percentile_count = 0

    for summary_path in discover_split_summary_paths(root_dir):
        payload = _load_json(summary_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Summary payload must be a JSON object: {summary_path}")

        run_dir = _run_dir_from_summary_path(summary_path)
        relative_path = run_dir.relative_to(root_dir).as_posix()

        token_series = _series_from_summary(payload, TOKEN_RATIO_NAME)
        context_series = _series_from_summary(payload, CONTEXT_RATIO_NAME)
        max_percentile_count = max(max_percentile_count, len(token_series), len(context_series))

        token_row: dict[str, Any] = {"run_path": relative_path}
        context_row: dict[str, Any] = {"run_path": relative_path}
        for index, value in enumerate(token_series, start=1):
            token_row[_column_name_for_percentile(index)] = value
        for index, value in enumerate(context_series, start=1):
            context_row[_column_name_for_percentile(index)] = value

        token_rows.append(token_row)
        context_rows.append(context_row)

    token_rows.sort(key=lambda item: _path_sort_key(item["run_path"]))
    context_rows.sort(key=lambda item: _path_sort_key(item["run_path"]))
    percentiles = list(range(1, max_percentile_count + 1))
    return token_rows, context_rows, percentiles


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(value)
    return str(value)


def _extract_run_dir_worker(run_dir_text: str) -> tuple[str, str | None, str | None]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = extract_run.extract_run_dir(run_dir)
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(run_dirs: list[Path]) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = extract_run.extract_run_dir(run_dir)
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


def _extract_all_runs(root_dir: Path, *, max_procs: int, dry_run: bool) -> int:
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


def _write_series_csv(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    percentiles: list[int],
) -> None:
    fieldnames = ["run_path", *[_column_name_for_percentile(p) for p in percentiles]]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")

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
            dry_run=False,
        )
        if extract_exit_code != 0:
            return extract_exit_code

    token_rows, context_rows, percentiles = build_rows(root_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (root_dir / DEFAULT_OUTPUT_DIR_NAME).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    token_path = output_dir / f"{TOKEN_RATIO_NAME}.csv"
    context_path = output_dir / f"{CONTEXT_RATIO_NAME}.csv"
    _write_series_csv(output_path=token_path, rows=token_rows, percentiles=percentiles)
    _write_series_csv(output_path=context_path, rows=context_rows, percentiles=percentiles)

    manifest_path = output_dir / "split-top-p-ratio-manifest.json"
    manifest_payload = {
        "source_root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "percentiles": percentiles,
        "tables": {
            TOKEN_RATIO_NAME: str(token_path),
            CONTEXT_RATIO_NAME: str(context_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(str(token_path))
    print(str(context_path))
    print(str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
