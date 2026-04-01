from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_OUTPUT_NAME = "key-stats.json"
DEFAULT_POST_PROCESSED_DIRNAME = "post-processed"
DEFAULT_REQUIRED_DISCOVERY_RELATIVE_PATH = Path("global/trial-timing-summary.json")

VLLM_TIMESERIES_INPUT_CANDIDATES = (
    Path("vllm-log/gauge-counter-timeseries.json"),
    Path("vllm-metrics/gauge-counter-timeseries.json"),
)
JOB_CONCURRENCY_INPUT_REL_PATH = Path("job-concurrency/job-concurrency-timeseries.json")
JOB_THROUGHPUT_INPUT_REL_PATH = Path("job-throughput/job-throughput-timeseries.json")
LLM_REQUEST_STATS_INPUT_REL_PATH = Path("gateway/llm-requests/llm-request-stats.json")
STACK_KV_INPUT_REL_PATH = Path("gateway/stack-kv/kv-usage-stacked-histogram.json")
STACK_CONTEXT_INPUT_REL_PATH = Path("gateway/stack-context/context-usage-stacked-histogram.json")
STACK_INPUT_REL_PATHS = {
    "prompt_tokens": Path("gateway/stack/prompt-tokens-stacked-histogram.json"),
    "cached_tokens": Path("gateway/stack/cached-tokens-stacked-histogram.json"),
    "compute_prompt_tokens": Path("gateway/stack/compute-prompt-tokens-stacked-histogram.json"),
    "completion_tokens": Path("gateway/stack/completion-tokens-stacked-histogram.json"),
    "compute_prompt_plus_completion_tokens": Path(
        "gateway/stack/compute-prompt-plus-completion-tokens-stacked-histogram.json"
    ),
}

VLLM_KV_CACHE_USAGE_METRIC_NAME = "vllm:kv_cache_usage_perc"


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
            "Summarize key post-processed metrics for one run, one post-processed "
            "directory, or every run under a root directory."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing post-processed/.",
    )
    target_group.add_argument(
        "--post-processed-dir",
        default=None,
        help="Explicit post-processed directory to summarize.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/global/trial-timing-summary.json will be processed."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path for single-target mode. Default: "
            "<post-processed-dir>/key-stats/"
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _default_post_processed_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_POST_PROCESSED_DIRNAME).resolve()


def _default_output_path_for_post_processed_dir(post_processed_dir: Path) -> Path:
    return (post_processed_dir / "key-stats" / DEFAULT_OUTPUT_NAME).resolve()


def discover_run_dirs_with_post_processed(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for summary_path in root_dir.rglob(DEFAULT_REQUIRED_DISCOVERY_RELATIVE_PATH.name):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name != DEFAULT_REQUIRED_DISCOVERY_RELATIVE_PATH.parent.name:
            continue
        if summary_path.parent.parent.name != DEFAULT_POST_PROCESSED_DIRNAME:
            continue
        run_dirs.add(summary_path.parent.parent.parent.resolve())
    return sorted(run_dirs)


def _require_object(payload: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _require_existing_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing required file: {path}")
    return _require_object(_load_json(path), path=path)


def _resolve_vllm_timeseries_path(post_processed_dir: Path) -> Path:
    for relative_path in VLLM_TIMESERIES_INPUT_CANDIDATES:
        candidate = (post_processed_dir / relative_path).resolve()
        if candidate.is_file():
            return candidate
    rendered = ", ".join(
        str((post_processed_dir / item).resolve())
        for item in VLLM_TIMESERIES_INPUT_CANDIDATES
    )
    raise ValueError(f"Missing required vLLM timeseries file. Checked: {rendered}")


def _summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "avg": None,
            "min": None,
            "max": None,
            "std": None,
        }

    avg_value = sum(values) / len(values)
    variance = sum((value - avg_value) ** 2 for value in values) / len(values)
    return {
        "sample_count": len(values),
        "avg": avg_value,
        "min": min(values),
        "max": max(values),
        "std": math.sqrt(variance),
    }


def _summarize_point_series(
    payload: dict[str, Any],
    *,
    points_key: str,
    value_key: str,
    metric_name: str,
) -> dict[str, Any]:
    raw_points = payload.get(points_key)
    if not isinstance(raw_points, list):
        raise ValueError(f"Input payload is missing list field: {points_key}")

    values: list[float] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        numeric = _float_or_none(point.get(value_key))
        if numeric is not None:
            values.append(numeric)

    summary = _summarize_values(values)
    summary["metric_name"] = metric_name
    return summary


def _summarize_vllm_metric(
    payload: dict[str, Any],
    *,
    metric_name: str,
) -> dict[str, Any]:
    metrics_payload = payload.get("metrics")
    if not isinstance(metrics_payload, dict):
        raise ValueError("Input payload is missing object field: metrics")

    values: list[float] = []
    series_count = 0
    for metric_payload in metrics_payload.values():
        if not isinstance(metric_payload, dict):
            continue
        if metric_payload.get("name") != metric_name:
            continue
        series_count += 1
        raw_values = metric_payload.get("value")
        if not isinstance(raw_values, list):
            continue
        for raw_value in raw_values:
            numeric = _float_or_none(raw_value)
            if numeric is not None:
                values.append(numeric)

    if series_count == 0:
        raise ValueError(f"Unable to find vLLM metric series: {metric_name}")

    summary = _summarize_values(values)
    summary["metric_name"] = metric_name
    summary["series_count"] = series_count
    return summary


def _copy_llm_request_metric_summary(metric_payload: Any) -> dict[str, Any]:
    if not isinstance(metric_payload, dict):
        raise ValueError("Each llm-request metric summary must be a JSON object")
    return {
        "count": metric_payload.get("count"),
        "avg": _float_or_none(metric_payload.get("avg")),
        "min": _float_or_none(metric_payload.get("min")),
        "max": _float_or_none(metric_payload.get("max")),
    }


def _copy_stage_speed_summary(stage_payload: Any) -> dict[str, Any]:
    if not isinstance(stage_payload, dict):
        raise ValueError("Stage speed summary must be a JSON object")
    return {
        "eligible_request_count": stage_payload.get("eligible_request_count"),
        "excluded_request_count": stage_payload.get("excluded_request_count"),
        "avg": _float_or_none(stage_payload.get("avg_tokens_per_s")),
        "min": _float_or_none(stage_payload.get("min_tokens_per_s")),
        "max": _float_or_none(stage_payload.get("max_tokens_per_s")),
    }


def _service_failure_fields(payloads: list[dict[str, Any]]) -> tuple[bool, Any]:
    for payload in payloads:
        if "service_failure_detected" in payload or "service_failure_cutoff_time_utc" in payload:
            return (
                bool(payload.get("service_failure_detected", False)),
                payload.get("service_failure_cutoff_time_utc"),
            )
    return False, None


def build_key_stats_payload(post_processed_dir: Path) -> dict[str, Any]:
    resolved_post_processed_dir = post_processed_dir.expanduser().resolve()
    if not resolved_post_processed_dir.is_dir():
        raise ValueError(f"Post-processed directory not found: {resolved_post_processed_dir}")

    vllm_timeseries_path = _resolve_vllm_timeseries_path(resolved_post_processed_dir)
    job_concurrency_path = (resolved_post_processed_dir / JOB_CONCURRENCY_INPUT_REL_PATH).resolve()
    job_throughput_path = (resolved_post_processed_dir / JOB_THROUGHPUT_INPUT_REL_PATH).resolve()
    llm_request_stats_path = (resolved_post_processed_dir / LLM_REQUEST_STATS_INPUT_REL_PATH).resolve()
    stack_kv_path = (resolved_post_processed_dir / STACK_KV_INPUT_REL_PATH).resolve()
    stack_context_path = (resolved_post_processed_dir / STACK_CONTEXT_INPUT_REL_PATH).resolve()
    stack_paths = {
        metric: (resolved_post_processed_dir / relative_path).resolve()
        for metric, relative_path in STACK_INPUT_REL_PATHS.items()
    }

    vllm_timeseries_payload = _require_existing_json(vllm_timeseries_path)
    job_concurrency_payload = _require_existing_json(job_concurrency_path)
    job_throughput_payload = _require_existing_json(job_throughput_path)
    llm_request_stats_payload = _require_existing_json(llm_request_stats_path)
    stack_kv_payload = _require_existing_json(stack_kv_path)
    stack_context_payload = _require_existing_json(stack_context_path)
    stack_payloads = {
        metric: _require_existing_json(path)
        for metric, path in stack_paths.items()
    }

    service_failure_detected, service_failure_cutoff_time_utc = _service_failure_fields(
        [
            llm_request_stats_payload,
            job_concurrency_payload,
            job_throughput_payload,
            stack_kv_payload,
            stack_context_payload,
            vllm_timeseries_payload,
        ]
    )

    llm_metrics_payload = llm_request_stats_payload.get("metrics")
    if not isinstance(llm_metrics_payload, dict):
        raise ValueError(
            f"Input payload is missing object field: metrics ({llm_request_stats_path})"
        )
    stage_speed_payload = llm_request_stats_payload.get("average_stage_speed_tokens_per_s")
    if not isinstance(stage_speed_payload, dict):
        raise ValueError(
            "Input payload is missing object field: average_stage_speed_tokens_per_s "
            f"({llm_request_stats_path})"
        )

    gateway_stack_summary = {
        metric: _summarize_point_series(
            payload=stack_payloads[metric],
            points_key="points",
            value_key="accumulated_value",
            metric_name=metric,
        )
        for metric in sorted(stack_payloads)
    }

    request_metrics_summary = {
        metric_name: _copy_llm_request_metric_summary(metric_payload)
        for metric_name, metric_payload in sorted(llm_metrics_payload.items())
    }

    return {
        "source_run_dir": str(resolved_post_processed_dir.parent),
        "source_post_processed_dir": str(resolved_post_processed_dir),
        "service_failure_detected": service_failure_detected,
        "service_failure_cutoff_time_utc": service_failure_cutoff_time_utc,
        "input_paths": {
            "vllm_metrics_timeseries": str(vllm_timeseries_path),
            "job_concurrency_timeseries": str(job_concurrency_path),
            "job_throughput_timeseries": str(job_throughput_path),
            "gateway_llm_request_stats": str(llm_request_stats_path),
            "gateway_stack_kv_histogram": str(stack_kv_path),
            "gateway_stack_context_histogram": str(stack_context_path),
            "gateway_stack_histograms": {
                metric: str(path) for metric, path in sorted(stack_paths.items())
            },
        },
        "vllm_metrics": {
            "kv_cache_usage_perc": _summarize_vllm_metric(
                vllm_timeseries_payload,
                metric_name=VLLM_KV_CACHE_USAGE_METRIC_NAME,
            )
        },
        "job_concurrency": _summarize_point_series(
            job_concurrency_payload,
            points_key="concurrency_points",
            value_key="concurrency",
            metric_name="concurrency",
        ),
        "job_throughput": _summarize_point_series(
            job_throughput_payload,
            points_key="throughput_points",
            value_key="throughput_jobs_per_s",
            metric_name="throughput_jobs_per_s",
        ),
        "gateway": {
            "stack_kv": {
                "kv_usage_tokens": _summarize_point_series(
                    stack_kv_payload,
                    points_key="points",
                    value_key="accumulated_value",
                    metric_name="kv_usage_tokens",
                )
            },
            "stack_context": {
                "context_usage_tokens": _summarize_point_series(
                    stack_context_payload,
                    points_key="points",
                    value_key="accumulated_value",
                    metric_name="context_usage_tokens",
                )
            },
            "stack": gateway_stack_summary,
            "llm_requests": {
                "request_count": llm_request_stats_payload.get("request_count"),
                "metric_count": llm_request_stats_payload.get("metric_count"),
                "average_stage_speed_tokens_per_s": {
                    "request_status_code": stage_speed_payload.get("request_status_code"),
                    "request_count_200": stage_speed_payload.get("request_count_200"),
                    "prefill": _copy_stage_speed_summary(stage_speed_payload.get("prefill")),
                    "decode": _copy_stage_speed_summary(stage_speed_payload.get("decode")),
                },
                "metrics": request_metrics_summary,
            },
        },
    }


def summarize_post_processed_dir(
    post_processed_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    resolved_post_processed_dir = post_processed_dir.expanduser().resolve()
    resolved_output_path = (
        output_path or _default_output_path_for_post_processed_dir(resolved_post_processed_dir)
    ).expanduser().resolve()

    payload = build_key_stats_payload(resolved_post_processed_dir)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def extract_run_dir(
    run_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    return summarize_post_processed_dir(
        _default_post_processed_dir_for_run(run_dir.expanduser().resolve()),
        output_path=output_path,
    )


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


def _main_single_target(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")

    output_path = Path(args.output).expanduser().resolve() if args.output else None
    if args.post_processed_dir:
        resolved_output_path = summarize_post_processed_dir(
            Path(args.post_processed_dir).expanduser().resolve(),
            output_path=output_path,
        )
    else:
        run_dir = Path(args.run_dir).expanduser().resolve()
        resolved_output_path = extract_run_dir(
            run_dir,
            output_path=output_path,
        )
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir or --post-processed-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_post_processed(root_dir)
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
    if args.root_dir:
        return _main_root_dir(args)
    return _main_single_target(args)


if __name__ == "__main__":
    raise SystemExit(main())
