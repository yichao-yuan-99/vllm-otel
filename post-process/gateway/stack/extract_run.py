from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pp_common.profile_id import int_or_none
from pp_common.profile_id import profile_ids_from_payload
from pp_common.profile_id import profile_label


DEFAULT_LLM_REQUESTS_OUTPUT_NAME = "llm-requests.json"

RANGE_OUTPUT_NAMES = {
    "prompt_tokens": "prompt-tokens-ranges.json",
    "cached_tokens": "cached-tokens-ranges.json",
    "compute_prompt_tokens": "compute-prompt-tokens-ranges.json",
    "completion_tokens": "completion-tokens-ranges.json",
    "compute_prompt_plus_completion_tokens": "compute-prompt-plus-completion-tokens-ranges.json",
}

HISTOGRAM_OUTPUT_NAMES = {
    "prompt_tokens": "prompt-tokens-stacked-histogram.json",
    "cached_tokens": "cached-tokens-stacked-histogram.json",
    "compute_prompt_tokens": "compute-prompt-tokens-stacked-histogram.json",
    "completion_tokens": "completion-tokens-stacked-histogram.json",
    "compute_prompt_plus_completion_tokens": (
        "compute-prompt-plus-completion-tokens-stacked-histogram.json"
    ),
}

METRICS_IN_ORDER = [
    "prompt_tokens",
    "cached_tokens",
    "compute_prompt_tokens",
    "completion_tokens",
    "compute_prompt_plus_completion_tokens",
]

METRIC_PHASE = {
    "prompt_tokens": "prefill",
    "cached_tokens": "prefill",
    "compute_prompt_tokens": "prefill",
    "completion_tokens": "decode",
    "compute_prompt_plus_completion_tokens": "mixed",
}

_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


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
            "Recover gateway token throughput over time by stacking per-request "
            "prefill/decode token ranges from llm-requests.json."
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
            "with a direct gateway-output/ child will be processed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. Default: "
            "<run-dir>/post-processed/gateway/stack/"
        ),
    )
    parser.add_argument(
        "--llm-requests",
        default=None,
        help=(
            "Optional input path to llm-requests.json (only for --run-dir). "
            "Default: <run-dir>/post-processed/gateway/llm-requests/"
            f"{DEFAULT_LLM_REQUESTS_OUTPUT_NAME}"
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


def discover_run_dirs_with_gateway_output(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for gateway_output_dir in root_dir.rglob("gateway-output"):
        if not gateway_output_dir.is_dir():
            continue
        run_dirs.add(gateway_output_dir.parent.resolve())
    return sorted(run_dirs)


def _default_llm_requests_path_for_run(run_dir: Path) -> Path:
    return (
        run_dir
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / DEFAULT_LLM_REQUESTS_OUTPUT_NAME
    ).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "gateway" / "stack").resolve()


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if _FLOAT_PATTERN.fullmatch(stripped) is None:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        if math.isfinite(parsed):
            return parsed
    return None


def _load_llm_request_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing required file: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object in llm request file: {path}")

    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        raise ValueError(f"Missing 'requests' list in llm request file: {path}")

    request_records: list[dict[str, Any]] = []
    for item in raw_requests:
        if isinstance(item, dict):
            request_records.append(item)
    return request_records, payload


def _gateway_profile_id_or_none(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    return int_or_none(payload.get("gateway_profile_id"))


def _port_profile_ids_from_payload_and_requests(
    llm_requests_payload: dict[str, Any],
    request_records: list[dict[str, Any]],
) -> list[int]:
    return sorted(
        set(profile_ids_from_payload(llm_requests_payload))
        | {
            profile_id
            for profile_id in (_gateway_profile_id_or_none(record) for record in request_records)
            if profile_id is not None
        }
    )


def _build_range_entry(
    request_record: dict[str, Any],
    *,
    metric: str,
    phase: str,
    range_start_s: float,
    range_duration_s: float,
    total_value: float,
) -> dict[str, Any]:
    range_end_s = range_start_s + range_duration_s
    return {
        "gateway_run_id": request_record.get("gateway_run_id"),
        "gateway_profile_id": request_record.get("gateway_profile_id"),
        "request_id": request_record.get("request_id"),
        "trace_id": request_record.get("trace_id"),
        "metric": metric,
        "phase": phase,
        "range_start_s": range_start_s,
        "range_end_s": range_end_s,
        "range_duration_s": range_duration_s,
        "total_value": total_value,
        "avg_value_per_s": total_value / range_duration_s,
    }


def build_token_ranges_by_metric(
    request_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    ranges_by_metric = {metric: [] for metric in METRICS_IN_ORDER}

    for request_record in request_records:
        request_start_offset_s = _float_or_none(request_record.get("request_start_offset_s"))
        if request_start_offset_s is None:
            continue

        time_in_queue_s = _float_or_none(request_record.get("gen_ai.latency.time_in_queue"))
        time_in_prefill_s = _float_or_none(request_record.get("gen_ai.latency.time_in_model_prefill"))
        prompt_tokens = _float_or_none(request_record.get("prompt_tokens"))

        cached_tokens_raw = _float_or_none(request_record.get("cached_tokens"))
        cached_tokens = 0.0 if cached_tokens_raw is None else cached_tokens_raw

        if (
            time_in_queue_s is not None
            and time_in_prefill_s is not None
            and time_in_prefill_s > 0.0
        ):
            prefill_start_s = request_start_offset_s + time_in_queue_s
            ranges_by_metric["cached_tokens"].append(
                _build_range_entry(
                    request_record,
                    metric="cached_tokens",
                    phase="prefill",
                    range_start_s=prefill_start_s,
                    range_duration_s=time_in_prefill_s,
                    total_value=cached_tokens,
                )
            )

            if prompt_tokens is not None:
                compute_prompt_tokens = prompt_tokens - cached_tokens
                ranges_by_metric["prompt_tokens"].append(
                    _build_range_entry(
                        request_record,
                        metric="prompt_tokens",
                        phase="prefill",
                        range_start_s=prefill_start_s,
                        range_duration_s=time_in_prefill_s,
                        total_value=prompt_tokens,
                    )
                )
                ranges_by_metric["compute_prompt_tokens"].append(
                    _build_range_entry(
                        request_record,
                        metric="compute_prompt_tokens",
                        phase="prefill",
                        range_start_s=prefill_start_s,
                        range_duration_s=time_in_prefill_s,
                        total_value=compute_prompt_tokens,
                    )
                )
                ranges_by_metric["compute_prompt_plus_completion_tokens"].append(
                    _build_range_entry(
                        request_record,
                        metric="compute_prompt_plus_completion_tokens",
                        phase="prefill",
                        range_start_s=prefill_start_s,
                        range_duration_s=time_in_prefill_s,
                        total_value=compute_prompt_tokens,
                    )
                )

        time_to_first_token_s = _float_or_none(
            request_record.get("gen_ai.latency.time_to_first_token")
        )
        time_in_decode_s = _float_or_none(request_record.get("gen_ai.latency.time_in_model_decode"))
        completion_tokens = _float_or_none(request_record.get("completion_tokens"))
        if (
            time_to_first_token_s is not None
            and time_in_decode_s is not None
            and time_in_decode_s > 0.0
            and completion_tokens is not None
        ):
            decode_start_s = request_start_offset_s + time_to_first_token_s
            ranges_by_metric["completion_tokens"].append(
                _build_range_entry(
                    request_record,
                    metric="completion_tokens",
                    phase="decode",
                    range_start_s=decode_start_s,
                    range_duration_s=time_in_decode_s,
                    total_value=completion_tokens,
                )
            )
            ranges_by_metric["compute_prompt_plus_completion_tokens"].append(
                _build_range_entry(
                    request_record,
                    metric="compute_prompt_plus_completion_tokens",
                    phase="decode",
                    range_start_s=decode_start_s,
                    range_duration_s=time_in_decode_s,
                    total_value=completion_tokens,
                )
            )

    def _range_sort_key(entry: dict[str, Any]) -> tuple[float, str]:
        range_start_s = _float_or_none(entry.get("range_start_s"))
        if range_start_s is None:
            range_start_s = float("inf")
        return range_start_s, str(entry.get("request_id") or "")

    for metric in METRICS_IN_ORDER:
        ranges_by_metric[metric].sort(key=_range_sort_key)
    return ranges_by_metric


def build_stacked_histogram(range_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals: list[tuple[float, float, float]] = []
    max_second = 0

    for entry in range_entries:
        range_start_s = _float_or_none(entry.get("range_start_s"))
        range_end_s = _float_or_none(entry.get("range_end_s"))
        avg_value_per_s = _float_or_none(entry.get("avg_value_per_s"))
        if range_start_s is None or range_end_s is None or avg_value_per_s is None:
            continue
        if range_end_s <= range_start_s:
            continue

        clipped_start_s = max(0.0, range_start_s)
        clipped_end_s = max(0.0, range_end_s)
        if clipped_end_s <= clipped_start_s:
            continue

        intervals.append((clipped_start_s, clipped_end_s, avg_value_per_s))
        max_second = max(max_second, int(math.ceil(clipped_end_s)))

    if max_second <= 0:
        return []

    full_bin_rate_deltas = [0.0] * (max_second + 1)
    partial_values: dict[int, float] = {}

    for interval_start_s, interval_end_s, avg_value_per_s in intervals:
        start_bin = int(math.floor(interval_start_s))
        end_floor = int(math.floor(interval_end_s))

        if start_bin == end_floor:
            partial_values[start_bin] = partial_values.get(start_bin, 0.0) + (
                avg_value_per_s * (interval_end_s - interval_start_s)
            )
            continue

        partial_values[start_bin] = partial_values.get(start_bin, 0.0) + (
            avg_value_per_s * ((start_bin + 1) - interval_start_s)
        )

        end_partial_width = interval_end_s - end_floor
        if end_partial_width > 0.0:
            partial_values[end_floor] = partial_values.get(end_floor, 0.0) + (
                avg_value_per_s * end_partial_width
            )

        full_start_bin = start_bin + 1
        full_end_bin = end_floor - 1
        if full_start_bin <= full_end_bin:
            full_bin_rate_deltas[full_start_bin] += avg_value_per_s
            full_bin_rate_deltas[full_end_bin + 1] -= avg_value_per_s

    points: list[dict[str, Any]] = []
    running_full_bin_rate = 0.0
    for second in range(max_second):
        running_full_bin_rate += full_bin_rate_deltas[second]
        value = running_full_bin_rate + partial_values.get(second, 0.0)
        points.append(
            {
                "second": second,
                "accumulated_value": value,
            }
        )

    return points


def extract_gateway_stack_from_run_dir(
    run_dir: Path,
    *,
    llm_requests_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_llm_requests_path = (
        llm_requests_path or _default_llm_requests_path_for_run(resolved_run_dir)
    ).expanduser().resolve()

    request_records, llm_requests_payload = _load_llm_request_records(resolved_llm_requests_path)
    ranges_by_metric = build_token_ranges_by_metric(request_records)
    histograms_by_metric = {
        metric: build_stacked_histogram(ranges_by_metric[metric])
        for metric in METRICS_IN_ORDER
    }

    source_gateway_output_dir = str((resolved_run_dir / "gateway-output").resolve())
    source_run_dir = str(resolved_run_dir)
    input_request_count = llm_requests_payload.get("request_count")
    if not isinstance(input_request_count, int):
        input_request_count = len(request_records)

    service_failure_detected = bool(
        llm_requests_payload.get("service_failure_detected", False)
    )
    service_failure_cutoff_time_utc = llm_requests_payload.get(
        "service_failure_cutoff_time_utc"
    )
    port_profile_ids = _port_profile_ids_from_payload_and_requests(
        llm_requests_payload,
        request_records,
    )
    multi_profile = len(port_profile_ids) > 1

    range_payloads: dict[str, dict[str, Any]] = {}
    histogram_payloads: dict[str, dict[str, Any]] = {}

    for metric in METRICS_IN_ORDER:
        range_entries = ranges_by_metric[metric]
        histogram_points = histograms_by_metric[metric]
        entries_by_profile: dict[str, list[dict[str, Any]]] = {}
        series_by_profile: dict[str, dict[str, Any]] = {}
        for gateway_profile_id in port_profile_ids:
            series_key = profile_label(gateway_profile_id)
            profile_entries = [
                entry
                for entry in range_entries
                if _gateway_profile_id_or_none(entry) == gateway_profile_id
            ]
            profile_histogram_points = build_stacked_histogram(profile_entries)
            entries_by_profile[series_key] = profile_entries
            series_by_profile[series_key] = {
                "source_run_dir": source_run_dir,
                "source_gateway_output_dir": source_gateway_output_dir,
                "source_llm_requests_path": str(resolved_llm_requests_path),
                "service_failure_detected": service_failure_detected,
                "service_failure_cutoff_time_utc": service_failure_cutoff_time_utc,
                "input_request_count": input_request_count,
                "metric": metric,
                "phase": METRIC_PHASE[metric],
                "gateway_profile_id": gateway_profile_id,
                "entry_count": len(profile_entries),
                "bucket_width_s": 1,
                "point_count": len(profile_histogram_points),
                "points": profile_histogram_points,
            }

        range_payloads[metric] = {
            "source_run_dir": source_run_dir,
            "source_gateway_output_dir": source_gateway_output_dir,
            "source_llm_requests_path": str(resolved_llm_requests_path),
            "service_failure_detected": service_failure_detected,
            "service_failure_cutoff_time_utc": service_failure_cutoff_time_utc,
            "input_request_count": input_request_count,
            "metric": metric,
            "phase": METRIC_PHASE[metric],
            "multi_profile": multi_profile,
            "port_profile_ids": port_profile_ids,
            "series_keys": list(series_by_profile.keys()),
            "entry_count": len(range_entries),
            "entries": range_entries,
            "entries_by_profile": entries_by_profile,
        }

        histogram_payloads[metric] = {
            "source_run_dir": source_run_dir,
            "source_gateway_output_dir": source_gateway_output_dir,
            "source_llm_requests_path": str(resolved_llm_requests_path),
            "service_failure_detected": service_failure_detected,
            "service_failure_cutoff_time_utc": service_failure_cutoff_time_utc,
            "input_request_count": input_request_count,
            "metric": metric,
            "phase": METRIC_PHASE[metric],
            "multi_profile": multi_profile,
            "port_profile_ids": port_profile_ids,
            "series_keys": list(series_by_profile.keys()),
            "bucket_width_s": 1,
            "point_count": len(histogram_points),
            "points": histograms_by_metric[metric],
            "series_by_profile": series_by_profile,
        }

    return range_payloads, histogram_payloads


def extract_run_dir(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    llm_requests_path: Path | None = None,
) -> list[Path]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_dir = (output_dir or _default_output_dir_for_run(resolved_run_dir)).expanduser().resolve()
    range_payloads, histogram_payloads = extract_gateway_stack_from_run_dir(
        resolved_run_dir,
        llm_requests_path=llm_requests_path,
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for metric in METRICS_IN_ORDER:
        range_output_path = resolved_output_dir / RANGE_OUTPUT_NAMES[metric]
        range_output_path.write_text(
            json.dumps(range_payloads[metric], ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        output_paths.append(range_output_path)

    for metric in METRICS_IN_ORDER:
        histogram_output_path = resolved_output_dir / HISTOGRAM_OUTPUT_NAMES[metric]
        histogram_output_path.write_text(
            json.dumps(histogram_payloads[metric], ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        output_paths.append(histogram_output_path)

    return output_paths


def _extract_run_dir_worker(run_dir_text: str) -> tuple[str, list[str] | None, str | None]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_paths = extract_run_dir(run_dir)
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


def _main_run_dir(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    llm_requests_path = (
        Path(args.llm_requests).expanduser().resolve() if args.llm_requests else None
    )
    output_paths = extract_run_dir(
        run_dir,
        output_dir=output_dir,
        llm_requests_path=llm_requests_path,
    )
    for output_path in output_paths:
        print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.llm_requests:
        raise ValueError("--llm-requests can only be used with --run-dir")
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
