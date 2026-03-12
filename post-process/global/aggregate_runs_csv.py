from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_NAME = "trial-timing-summary.json"
DEFAULT_OUTPUT_NAME = "trial-timing-summary.csv"
VLLM_STATS_INPUT_REL_PATH = Path("post-processed/vllm-log/gauge-counter-timeseries.stats.json")
GATEWAY_USAGE_INPUT_REL_PATH = Path("post-processed/gateway/usage/usage-summary.json")
GATEWAY_USAGE_FIELDNAMES = (
    "prompt_tokens",
    "generation_tokens",
    "cached_prompt_tokens",
    "prefill_prompt_tokens",
    "avg_worker_max_request_length",
)
VLLM_METRIC_NAMES = (
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)
VLLM_METRIC_STAT_SUFFIXES = ("avg", "min", "max")
VLLM_METRIC_FIELDNAMES = tuple(
    f"{metric}:{suffix}"
    for metric in VLLM_METRIC_NAMES
    for suffix in VLLM_METRIC_STAT_SUFFIXES
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-run global trial timing summaries into one CSV for a root directory."
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


def discover_global_summary_paths(root_dir: Path) -> list[Path]:
    summary_paths: set[Path] = set()
    for summary_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name != "global":
            continue
        if summary_path.parent.parent.name != "post-processed":
            continue
        summary_paths.add(summary_path.resolve())
    return sorted(summary_paths)


def _run_dir_from_summary_path(summary_path: Path) -> Path:
    # <run-dir>/post-processed/global/trial-timing-summary.json
    return summary_path.parent.parent.parent


def _path_sort_key(relative_path: str) -> tuple[tuple[Any, ...], ...]:
    key_parts: list[tuple[Any, ...]] = []
    for part in relative_path.split("/"):
        if part.isdigit():
            # Numeric path levels sort by integer value (1, 2, 10).
            key_parts.append((0, int(part), part))
        else:
            key_parts.append((1, part))
    return tuple(key_parts)


def _weighted_avg_or_none(values: list[tuple[float, int | None]]) -> float | None:
    if not values:
        return None

    weighted_sum = 0.0
    total_weight = 0
    for avg, sample_count in values:
        if sample_count is None or sample_count <= 0:
            return sum(item[0] for item in values) / len(values)
        weighted_sum += avg * sample_count
        total_weight += sample_count
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _metric_avg_from_vllm_stats(stats_payload: Any, metric_name: str) -> float | None:
    if not isinstance(stats_payload, dict):
        return None
    metrics_payload = stats_payload.get("metrics")
    if not isinstance(metrics_payload, dict):
        return None

    values: list[tuple[float, int | None]] = []
    for metric_payload in metrics_payload.values():
        if not isinstance(metric_payload, dict):
            continue
        if metric_payload.get("name") != metric_name:
            continue
        avg = _float_or_none(metric_payload.get("avg"))
        if avg is None:
            continue
        sample_count_raw = metric_payload.get("sample_count")
        sample_count = sample_count_raw if isinstance(sample_count_raw, int) else None
        values.append((avg, sample_count))
    return _weighted_avg_or_none(values)


def _metric_extreme_from_vllm_stats(
    stats_payload: Any,
    metric_name: str,
    *,
    kind: str,
) -> float | None:
    if not isinstance(stats_payload, dict):
        return None
    metrics_payload = stats_payload.get("metrics")
    if not isinstance(metrics_payload, dict):
        return None

    values: list[float] = []
    for metric_payload in metrics_payload.values():
        if not isinstance(metric_payload, dict):
            continue
        if metric_payload.get("name") != metric_name:
            continue
        value = _float_or_none(metric_payload.get(kind))
        if value is None:
            continue
        values.append(value)

    if not values:
        return None
    return min(values) if kind == "min" else max(values)


def _load_vllm_metric_stats(run_dir: Path) -> dict[str, float | None]:
    field_values: dict[str, float | None] = {
        fieldname: None for fieldname in VLLM_METRIC_FIELDNAMES
    }
    stats_path = run_dir / VLLM_STATS_INPUT_REL_PATH
    if not stats_path.is_file():
        return field_values

    stats_payload = _load_json(stats_path)
    for metric_name in VLLM_METRIC_NAMES:
        field_values[f"{metric_name}:avg"] = _metric_avg_from_vllm_stats(
            stats_payload, metric_name
        )
        field_values[f"{metric_name}:min"] = _metric_extreme_from_vllm_stats(
            stats_payload, metric_name, kind="min"
        )
        field_values[f"{metric_name}:max"] = _metric_extreme_from_vllm_stats(
            stats_payload, metric_name, kind="max"
        )
    return field_values


def _load_gateway_usage_totals(run_dir: Path) -> dict[str, int | float | None]:
    field_values: dict[str, int | float | None] = {
        fieldname: None for fieldname in GATEWAY_USAGE_FIELDNAMES
    }
    usage_path = run_dir / GATEWAY_USAGE_INPUT_REL_PATH
    if not usage_path.is_file():
        return field_values

    payload = _load_json(usage_path)
    if not isinstance(payload, dict):
        return field_values
    usage_payload = payload.get("usage")
    if not isinstance(usage_payload, dict):
        return field_values

    prompt_tokens = _int_or_none(usage_payload.get("prompt_tokens"))
    generation_tokens = _int_or_none(usage_payload.get("generation_tokens"))
    completion_tokens = _int_or_none(usage_payload.get("completion_tokens"))
    cached_prompt_tokens = _int_or_none(usage_payload.get("cached_prompt_tokens"))
    prefill_prompt_tokens = _int_or_none(usage_payload.get("prefill_prompt_tokens"))
    avg_worker_max_request_length = _float_or_none(
        usage_payload.get("avg_worker_max_request_length")
    )

    if generation_tokens is None:
        generation_tokens = completion_tokens
    if prefill_prompt_tokens is None and prompt_tokens is not None and cached_prompt_tokens is not None:
        prefill_prompt_tokens = prompt_tokens - cached_prompt_tokens

    field_values["prompt_tokens"] = prompt_tokens
    field_values["generation_tokens"] = generation_tokens
    field_values["cached_prompt_tokens"] = cached_prompt_tokens
    field_values["prefill_prompt_tokens"] = prefill_prompt_tokens
    field_values["avg_worker_max_request_length"] = avg_worker_max_request_length
    return field_values


def build_rows(root_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in discover_global_summary_paths(root_dir):
        payload = _load_json(summary_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Summary payload must be a JSON object: {summary_path}")

        run_dir = _run_dir_from_summary_path(summary_path)
        relative_path = run_dir.relative_to(root_dir).as_posix()
        stats_payload = payload.get("trial_duration_stats_s")
        stats = stats_payload if isinstance(stats_payload, dict) else {}
        gateway_usage_totals = _load_gateway_usage_totals(run_dir)
        vllm_metric_stats = _load_vllm_metric_stats(run_dir)
        rows.append(
            {
                "run_path": relative_path,
                "total_duration_s": _float_or_none(payload.get("total_duration_s")),
                "trial_duration_avg_s": _float_or_none(stats.get("avg")),
                "trial_duration_min_s": _float_or_none(stats.get("min")),
                "trial_duration_max_s": _float_or_none(stats.get("max")),
                **gateway_usage_totals,
                **vllm_metric_stats,
            }
        )

    rows.sort(key=lambda item: _path_sort_key(item["run_path"]))
    return rows


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    rows = build_rows(root_dir)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (root_dir / DEFAULT_OUTPUT_NAME).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_path",
        "total_duration_s",
        "trial_duration_avg_s",
        "trial_duration_min_s",
        "trial_duration_max_s",
        *GATEWAY_USAGE_FIELDNAMES,
        *VLLM_METRIC_FIELDNAMES,
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
