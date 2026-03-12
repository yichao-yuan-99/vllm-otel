from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
import json
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable

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

DEFAULT_REQUESTS_OUTPUT_NAME = "llm-requests.json"
DEFAULT_STATS_OUTPUT_NAME = "llm-request-stats.json"
DEFAULT_LONGEST_OUTPUT_NAME = "llm-requests-longest-10.json"
DEFAULT_SHORTEST_OUTPUT_NAME = "llm-requests-shortest-10.json"
EXTREME_REQUEST_LIMIT = 10


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
        TextColumn("{task.completed}/{task.total} requests"),
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


def _parse_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _profile_id_from_name(name: str) -> int | None:
    prefix = "profile-"
    if not name.startswith(prefix):
        return None
    raw = name[len(prefix) :]
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def discover_gateway_run_dirs(gateway_output_dir: Path) -> list[tuple[Path, int | None]]:
    run_dirs: list[tuple[Path, int | None]] = []

    for run_dir in sorted(gateway_output_dir.glob("run_*")):
        if run_dir.is_dir():
            run_dirs.append((run_dir, None))

    for child in sorted(gateway_output_dir.iterdir()):
        if not child.is_dir():
            continue
        profile_id = _profile_id_from_name(child.name)
        if profile_id is None:
            continue
        for run_dir in sorted(child.glob("run_*")):
            if run_dir.is_dir():
                run_dirs.append((run_dir, profile_id))

    return run_dirs


def _load_lifecycle_window(lifecycle_path: Path) -> tuple[datetime, datetime | None]:
    records = _load_jsonl(lifecycle_path)
    job_start: datetime | None = None
    job_end: datetime | None = None
    for record in records:
        event_type = record.get("event_type")
        timestamp = _parse_iso8601(record.get("timestamp"))
        if timestamp is None:
            continue
        if event_type == "job_start" and job_start is None:
            job_start = timestamp
        if event_type == "job_end":
            job_end = timestamp

    if job_start is None:
        raise ValueError(f"Missing job_start timestamp in lifecycle file: {lifecycle_path}")
    return job_start, job_end


def _index_llm_tags_by_model_span(jaeger_trace_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data = jaeger_trace_payload.get("data")
    if not isinstance(data, list):
        return {}

    tags_by_model_span: dict[str, dict[str, Any]] = {}
    for trace_obj in data:
        if not isinstance(trace_obj, dict):
            continue
        spans = trace_obj.get("spans")
        if not isinstance(spans, list):
            continue
        for span in spans:
            if not isinstance(span, dict):
                continue
            if span.get("operationName") != "llm_request":
                continue
            parent_span_id: str | None = None
            references = span.get("references")
            if isinstance(references, list):
                for ref in references:
                    if not isinstance(ref, dict):
                        continue
                    if ref.get("refType") != "CHILD_OF":
                        continue
                    candidate = ref.get("spanID")
                    if isinstance(candidate, str) and candidate:
                        parent_span_id = candidate
                        break
            if parent_span_id is None:
                continue
            tags = span.get("tags")
            if not isinstance(tags, list):
                continue
            flattened = tags_by_model_span.setdefault(parent_span_id, {})
            for tag in tags:
                if not isinstance(tag, dict):
                    continue
                key = tag.get("key")
                if not isinstance(key, str) or not key:
                    continue
                flattened[key] = tag.get("value")

    return tags_by_model_span


def _extract_usage_tokens(record: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    response = record.get("response")
    if not isinstance(response, dict):
        return None, None, None, None

    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None, None, None, None

    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens_details = usage.get("prompt_tokens_details")
    cached_tokens = None
    if isinstance(prompt_tokens_details, dict):
        cached_tokens = prompt_tokens_details.get("cached_tokens")
    return prompt_tokens, total_tokens, completion_tokens, cached_tokens


def _build_request_record(
    record: dict[str, Any],
    *,
    run_id: str,
    profile_id: int | None,
    job_start_time: datetime,
    job_end_time: datetime | None,
    llm_tags: dict[str, Any],
) -> dict[str, Any]:
    request_start_time_raw = record.get("request_start_time")
    request_end_time_raw = record.get("request_end_time")
    request_start_time = _parse_iso8601(request_start_time_raw)
    request_end_time = _parse_iso8601(request_end_time_raw)

    request_start_offset_s = None
    request_end_offset_s = None
    request_end_to_run_end_s = None
    if request_start_time is not None:
        request_start_offset_s = (request_start_time - job_start_time).total_seconds()
    if request_end_time is not None:
        request_end_offset_s = (request_end_time - job_start_time).total_seconds()
        if job_end_time is not None:
            request_end_to_run_end_s = (job_end_time - request_end_time).total_seconds()

    prompt_tokens, total_tokens, completion_tokens, cached_tokens = _extract_usage_tokens(record)

    flattened = {
        "gateway_run_id": run_id,
        "gateway_profile_id": profile_id,
        "trace_id": record.get("trace_id"),
        "request_id": record.get("request_id"),
        "model_inference_span_id": record.get("model_inference_span_id"),
        "model_inference_parent_span_id": record.get("model_inference_parent_span_id"),
        "request_start_time": request_start_time_raw,
        "request_end_time": request_end_time_raw,
        "request_start_offset_s": request_start_offset_s,
        "request_end_offset_s": request_end_offset_s,
        "request_end_to_run_end_s": request_end_to_run_end_s,
        "request_duration_ms": record.get("request_duration_ms"),
        "duration_ms": record.get("duration_ms"),
        "status_code": record.get("status_code"),
        "http_method": record.get("http_method"),
        "http_path": record.get("http_path"),
        "model": record.get("model"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }
    flattened.update(llm_tags)
    return flattened


def collect_llm_request_records(
    run_dir: Path,
    *,
    on_request_loaded: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    gateway_output_dir = run_dir / "gateway-output"
    if not gateway_output_dir.is_dir():
        raise ValueError(f"Missing gateway-output directory: {gateway_output_dir}")

    discovered_run_dirs = discover_gateway_run_dirs(gateway_output_dir)
    if not discovered_run_dirs:
        raise ValueError(
            "No run_* artifacts found under gateway-output. "
            "Expected either gateway-output/run_* or gateway-output/profile-*/run_*."
        )

    total_requests = 0
    for gateway_run_dir, _profile_id in discovered_run_dirs:
        requests_path = gateway_run_dir / "requests" / "model_inference.jsonl"
        if not requests_path.is_file():
            raise ValueError(f"Missing required file: {requests_path}")
        total_requests += _count_jsonl_records(requests_path)

    loaded_requests = 0
    all_requests: list[dict[str, Any]] = []
    for gateway_run_dir, profile_id in discovered_run_dirs:
        lifecycle_path = gateway_run_dir / "events" / "lifecycle.jsonl"
        requests_path = gateway_run_dir / "requests" / "model_inference.jsonl"
        trace_path = gateway_run_dir / "trace" / "jaeger_trace.json"
        for path in [lifecycle_path, requests_path, trace_path]:
            if not path.is_file():
                raise ValueError(f"Missing required file: {path}")

        job_start_time, job_end_time = _load_lifecycle_window(lifecycle_path)
        model_requests = _load_jsonl(requests_path)
        trace_payload = _load_json(trace_path)
        if not isinstance(trace_payload, dict):
            raise ValueError(f"Invalid JSON object in trace file: {trace_path}")
        llm_tags_by_span = _index_llm_tags_by_model_span(trace_payload)

        for record in model_requests:
            model_span_id = record.get("model_inference_span_id")
            llm_tags = {}
            if isinstance(model_span_id, str):
                llm_tags = llm_tags_by_span.get(model_span_id, {})
            all_requests.append(
                _build_request_record(
                    record,
                    run_id=gateway_run_dir.name,
                    profile_id=profile_id,
                    job_start_time=job_start_time,
                    job_end_time=job_end_time,
                    llm_tags=llm_tags,
                )
            )
            loaded_requests += 1
            if on_request_loaded is not None:
                on_request_loaded(loaded_requests, total_requests)

    all_requests.sort(
        key=lambda item: (
            _parse_iso8601(item.get("request_start_time")) or datetime.max,
            item.get("request_id") or "",
        )
    )
    return all_requests, total_requests


def build_numeric_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_metric: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                values_by_metric.setdefault(key, []).append(float(value))

    metrics: dict[str, Any] = {}
    for key, values in sorted(values_by_metric.items()):
        if not values:
            continue
        metrics[key] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
        }

    return {
        "metric_count": len(metrics),
        "metrics": metrics,
    }


def _status_code_key(record: dict[str, Any]) -> str | None:
    value = record.get("status_code")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return stripped
    return None


def build_stats_by_status_code(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        status_key = _status_code_key(record)
        if status_key is None:
            continue
        grouped_records.setdefault(status_key, []).append(record)

    grouped_payloads: dict[str, dict[str, Any]] = {}
    for status_key, grouped in sorted(grouped_records.items(), key=lambda item: int(item[0])):
        payload = {
            "status_code": int(status_key),
            "request_count": len(grouped),
        }
        payload.update(build_numeric_stats(grouped))
        grouped_payloads[status_key] = payload
    return grouped_payloads


def _duration_value_ms(record: dict[str, Any]) -> float | None:
    value = record.get("request_duration_ms")
    if isinstance(value, bool):
        value = None
    if isinstance(value, (int, float)):
        return float(value)

    fallback = record.get("duration_ms")
    if isinstance(fallback, bool):
        return None
    if isinstance(fallback, (int, float)):
        return float(fallback)
    return None


def _request_tiebreak_key(record: dict[str, Any]) -> tuple[datetime, str]:
    return (
        _parse_iso8601(record.get("request_start_time")) or datetime.max,
        str(record.get("request_id") or ""),
    )


def select_extreme_duration_requests(
    records: list[dict[str, Any]],
    *,
    limit: int = EXTREME_REQUEST_LIMIT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with_duration = [record for record in records if _duration_value_ms(record) is not None]

    longest = sorted(
        with_duration,
        key=lambda record: (
            -(_duration_value_ms(record) or 0.0),
            _request_tiebreak_key(record),
        ),
    )[:limit]
    shortest = sorted(
        with_duration,
        key=lambda record: (
            (_duration_value_ms(record) or 0.0),
            _request_tiebreak_key(record),
        ),
    )[:limit]
    return longest, shortest


def discover_run_dirs_with_gateway_output(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for gateway_output_dir in root_dir.rglob("gateway-output"):
        if not gateway_output_dir.is_dir():
            continue
        run_dirs.add(gateway_output_dir.parent.resolve())
    return sorted(run_dirs)


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "gateway" / "llm-requests").resolve()


def extract_run_dir(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    show_progress: bool = True,
) -> list[Path]:
    if show_progress:
        with create_extract_progress() as progress:
            task_id = progress.add_task(
                "extracting gateway llm requests",
                total=1,
            )

            def _on_request_loaded(completed: int, total: int) -> None:
                progress.update(task_id, completed=completed, total=max(total, 1))

            request_records, total_requests = collect_llm_request_records(
                run_dir,
                on_request_loaded=_on_request_loaded,
            )
            if total_requests == 0:
                progress.update(task_id, completed=1, total=1)
    else:
        request_records, _total_requests = collect_llm_request_records(run_dir)

    resolved_output_dir = (output_dir or _default_output_dir_for_run(run_dir)).expanduser().resolve()
    source_gateway_output_dir = str((run_dir / "gateway-output").resolve())

    requests_payload = {
        "source_run_dir": str(run_dir),
        "source_gateway_output_dir": source_gateway_output_dir,
        "request_count": len(request_records),
        "requests": request_records,
    }
    stats_payload = {
        "source_run_dir": str(run_dir),
        "source_gateway_output_dir": source_gateway_output_dir,
        "request_count": len(request_records),
    }
    stats_payload.update(build_numeric_stats(request_records))
    status_code_stats_payloads = build_stats_by_status_code(request_records)
    longest_requests, shortest_requests = select_extreme_duration_requests(
        request_records,
        limit=EXTREME_REQUEST_LIMIT,
    )
    longest_payload = {
        "source_run_dir": str(run_dir),
        "source_gateway_output_dir": source_gateway_output_dir,
        "request_count": len(request_records),
        "selected_count": len(longest_requests),
        "selection": "longest",
        "limit": EXTREME_REQUEST_LIMIT,
        "duration_field": "request_duration_ms (fallback: duration_ms)",
        "requests": longest_requests,
    }
    shortest_payload = {
        "source_run_dir": str(run_dir),
        "source_gateway_output_dir": source_gateway_output_dir,
        "request_count": len(request_records),
        "selected_count": len(shortest_requests),
        "selection": "shortest",
        "limit": EXTREME_REQUEST_LIMIT,
        "duration_field": "request_duration_ms (fallback: duration_ms)",
        "requests": shortest_requests,
    }

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    requests_output_path = resolved_output_dir / DEFAULT_REQUESTS_OUTPUT_NAME
    stats_output_path = resolved_output_dir / DEFAULT_STATS_OUTPUT_NAME
    longest_output_path = resolved_output_dir / DEFAULT_LONGEST_OUTPUT_NAME
    shortest_output_path = resolved_output_dir / DEFAULT_SHORTEST_OUTPUT_NAME
    status_output_paths: list[Path] = []
    requests_output_path.write_text(
        json.dumps(requests_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    stats_output_path.write_text(
        json.dumps(stats_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    longest_output_path.write_text(
        json.dumps(longest_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    shortest_output_path.write_text(
        json.dumps(shortest_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    for status_key, payload in status_code_stats_payloads.items():
        status_output_path = resolved_output_dir / f"llm-requests-stats.{status_key}.json"
        status_output_path.write_text(
            json.dumps(
                {
                    "source_run_dir": str(run_dir),
                    "source_gateway_output_dir": source_gateway_output_dir,
                    **payload,
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        status_output_paths.append(status_output_path)

    return [
        requests_output_path,
        stats_output_path,
        longest_output_path,
        shortest_output_path,
        *status_output_paths,
    ]


def _extract_run_dir_worker(run_dir_text: str) -> tuple[str, list[str] | None, str | None]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_paths = extract_run_dir(run_dir, show_progress=False)
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), [str(path) for path in output_paths], None)


def _run_root_dir_sequential(run_dirs: list[Path]) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_paths = extract_run_dir(run_dir)
            output_dir = output_paths[0].parent if output_paths else _default_output_dir_for_run(run_dir)
            print(f"[done] {run_dir} -> {output_dir}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int) -> int:
    failure_count = 0
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_paths_text, error_text in executor.map(
            _extract_run_dir_worker,
            [str(run_dir) for run_dir in run_dirs],
        ):
            if error_text is None:
                output_dir_text = (
                    str(Path(output_paths_text[0]).parent) if output_paths_text else "<unknown-output-dir>"
                )
                print(f"[done] {run_dir_text} -> {output_dir_text}")
            else:
                failure_count += 1
                print(f"[error] {run_dir_text}: {error_text}", file=sys.stderr)
    return failure_count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract flattened gateway LLM request records and numeric stats from a run."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing gateway-output/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with a direct gateway-output/ child will be processed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. Default: "
            "<run-dir>/post-processed/gateway/llm-requests/"
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


def _main_run_dir(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_paths = extract_run_dir(run_dir, output_dir=output_dir)
    for output_path in output_paths:
        print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_gateway_output(root_dir)
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
