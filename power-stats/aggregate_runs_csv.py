from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from pathlib import Path
import sys
from typing import Any


DEFAULT_INPUT_SUBDIR = "post-processed"
DEFAULT_OUTPUT_SUBDIR = "power-stats"
DEFAULT_FREQ_OUTPUT_NAME = "frequency-sweep-summary.csv"
DEFAULT_NON_FREQ_OUTPUT_NAME = "run-summary.csv"
AVG_ONLY_SUFFIX = "-avg-only"

REPO_ROOT = Path(__file__).resolve().parents[1]
_HELPER_MODULES: dict[str, Any] = {}

POWER_SUMMARY_SUBPATH = Path("power/power-summary.json")
JOB_THROUGHPUT_SUBPATH = Path("job-throughput/job-throughput-timeseries.json")
GLOBAL_SUMMARY_SUBPATH = Path("global/trial-timing-summary.json")
LLM_REQUEST_STATS_SUBPATH = Path("gateway/llm-requests/llm-request-stats.json")
KEY_STATS_SUBPATH = Path("key-stats/key-stats.json")

POWER_SUMMARY_REL_PATH = Path(DEFAULT_INPUT_SUBDIR) / POWER_SUMMARY_SUBPATH
JOB_THROUGHPUT_REL_PATH = Path(DEFAULT_INPUT_SUBDIR) / JOB_THROUGHPUT_SUBPATH
GLOBAL_SUMMARY_REL_PATH = Path(DEFAULT_INPUT_SUBDIR) / GLOBAL_SUMMARY_SUBPATH
LLM_REQUEST_STATS_REL_PATH = Path(DEFAULT_INPUT_SUBDIR) / LLM_REQUEST_STATS_SUBPATH

FREQUENCY_SLUG_RE = re.compile(r"^core-(?P<min>\d+)-(?P<max>\d+)(?:-mem-(?P<mem>\d+))?$")
FIELDNAME_PART_RE = re.compile(r"[^A-Za-z0-9]+")

BASE_CSV_FIELDNAMES = (
    "run_path",
    "frequency_mhz",
    "core_min_mhz",
    "core_max_mhz",
    "mem_freq_mhz",
    "avg_power_w",
    "std_power_w",
    "avg_job_throughput_jobs_per_s",
    "std_job_throughput_jobs_per_s",
    "agent_total_time_avg_s",
    "agent_total_time_std_s",
    "llm_time_avg_s",
    "llm_time_std_s",
    "non_llm_time_avg_s",
    "non_llm_time_std_s",
    "prefill_avg_tokens_per_s",
    "prefill_std_tokens_per_s",
    "decode_avg_tokens_per_s",
    "decode_std_tokens_per_s",
    "completion_tokens_avg",
    "completion_tokens_std",
    "cached_tokens_avg",
    "cached_tokens_std",
    "duration_ms_avg",
    "duration_ms_std",
    "prompt_tokens_avg",
    "prompt_tokens_std",
)

FIXED_METADATA_FIELDNAMES = (
    "run_path",
    "frequency_mhz",
    "core_min_mhz",
    "core_max_mhz",
    "mem_freq_mhz",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate run summaries into one CSV table. "
            "In freq mode, run directories are expected to be named like "
            "core-<min>-<max>(-mem-<mhz>)."
        )
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory that contains frequency-sweep run results.",
    )
    parser.add_argument(
        "--mode",
        choices=("freq", "non-freq"),
        default="freq",
        help=(
            "Aggregation mode. "
            "'freq' expects core-<min>-<max> run directory names. "
            "'non-freq' discovers run directories by post-processed layout."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV path. "
            "Default depends on --mode: "
            "<root-dir>/power-stats/frequency-sweep-summary.csv (freq), "
            "<root-dir>/power-stats/run-summary.csv (non-freq). "
            "With --percent X, the default output subdir becomes "
            "<root-dir>/power-stats-<X>."
        ),
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=None,
        help=(
            "Optional selected-window percent label from post-process-select. "
            "For example, --percent 50 reads from post-processed-50/ and writes "
            "to power-stats-50/. Default: use post-processed/ and power-stats/."
        ),
    )
    parser.add_argument(
        "--expected-core-min-mhz",
        type=int,
        default=345,
        help=(
            "Expected lower bound in run dir slug (default: 345). "
            "This enforces the README assumption that min core clock is fixed."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-frequency",
        action="store_true",
        help=(
            "Allow repeated frequency_mhz values. By default duplicates are rejected "
            "to keep frequency_mhz as the primary key."
        ),
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_helper_module(cache_key: str, relative_path: str) -> Any:
    cached = _HELPER_MODULES.get(cache_key)
    if cached is not None:
        return cached

    module_path = (REPO_ROOT / relative_path).resolve()
    spec = importlib.util.spec_from_file_location(
        f"power_stats_{cache_key}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _HELPER_MODULES[cache_key] = module
    return module


def _normalize_percent(percent: float | None) -> float | None:
    if percent is None:
        return None
    if not math.isfinite(percent):
        raise ValueError(f"--percent must be finite: {percent!r}")
    if percent <= 0 or percent > 100:
        raise ValueError(f"--percent must be in the interval (0, 100]: {percent}")
    return float(percent)


def _percent_label(percent: float) -> str:
    return f"{percent:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _processed_subdir_name(*, percent: float | None = None) -> str:
    normalized_percent = _normalize_percent(percent)
    if normalized_percent is None:
        return DEFAULT_INPUT_SUBDIR
    return f"{DEFAULT_INPUT_SUBDIR}-{_percent_label(normalized_percent)}"


def _output_subdir_name(*, percent: float | None = None) -> str:
    normalized_percent = _normalize_percent(percent)
    if normalized_percent is None:
        return DEFAULT_OUTPUT_SUBDIR
    return f"{DEFAULT_OUTPUT_SUBDIR}-{_percent_label(normalized_percent)}"


def _summary_rel_path(subpath: Path, *, percent: float | None = None) -> Path:
    return Path(_processed_subdir_name(percent=percent)) / subpath


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


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(value)
    return str(value)


def _population_std(values: list[float]) -> float | None:
    if not values:
        return None
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _sanitize_fieldname_part(part: str) -> str:
    sanitized = FIELDNAME_PART_RE.sub("_", part).strip("_")
    if not sanitized:
        return "value"
    return sanitized


def parse_frequency_slug(slug: str) -> tuple[int, int, int | None] | None:
    match = FREQUENCY_SLUG_RE.fullmatch(slug)
    if match is None:
        return None
    min_mhz = int(match.group("min"))
    max_mhz = int(match.group("max"))
    mem_group = match.group("mem")
    mem_mhz = int(mem_group) if mem_group is not None else None
    return (min_mhz, max_mhz, mem_mhz)


def discover_frequency_run_dirs(root_dir: Path, *, percent: float | None = None) -> list[Path]:
    discovered: set[Path] = set()
    processed_subdir_name = _processed_subdir_name(percent=percent)
    for candidate in root_dir.rglob("*"):
        if not candidate.is_dir():
            continue
        if parse_frequency_slug(candidate.name) is None:
            continue
        if not (candidate / processed_subdir_name).is_dir():
            continue
        discovered.add(candidate.resolve())
    return sorted(discovered)


def _has_any_summary_input(run_dir: Path, *, percent: float | None = None) -> bool:
    return any(
        (run_dir / _summary_rel_path(rel_path, percent=percent)).is_file()
        for rel_path in (
            POWER_SUMMARY_SUBPATH,
            JOB_THROUGHPUT_SUBPATH,
            GLOBAL_SUMMARY_SUBPATH,
            LLM_REQUEST_STATS_SUBPATH,
            KEY_STATS_SUBPATH,
        )
    )


def discover_non_freq_run_dirs(root_dir: Path, *, percent: float | None = None) -> list[Path]:
    discovered: set[Path] = set()
    processed_subdir_name = _processed_subdir_name(percent=percent)
    for processed_dir in root_dir.rglob(processed_subdir_name):
        if not processed_dir.is_dir():
            continue
        run_dir = processed_dir.parent
        if not run_dir.is_dir():
            continue
        if not _has_any_summary_input(run_dir, percent=percent):
            continue
        discovered.add(run_dir.resolve())
    return sorted(discovered)


def discover_run_dirs(
    root_dir: Path,
    *,
    mode: str,
    percent: float | None = None,
) -> list[Path]:
    if mode == "freq":
        return discover_frequency_run_dirs(root_dir, percent=percent)
    return discover_non_freq_run_dirs(root_dir, percent=percent)


def _nested_get(payload: Any, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _average_job_throughput_jobs_per_s(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None

    values: list[float] = []
    throughput_points = payload.get("throughput_points")
    if isinstance(throughput_points, list):
        for item in throughput_points:
            if not isinstance(item, dict):
                continue
            value = _float_or_none(item.get("throughput_jobs_per_s"))
            if value is None:
                continue
            values.append(value)
    if values:
        return sum(values) / len(values)

    finished_count = _float_or_none(payload.get("finished_replay_count_excluding_cancelled"))
    if finished_count is None:
        finished_count = _float_or_none(payload.get("finished_replay_count"))
    total_duration_s = _float_or_none(payload.get("total_duration_s"))
    if finished_count is None or total_duration_s is None or total_duration_s <= 0:
        return None
    return finished_count / total_duration_s


def _job_throughput_avg_and_std(payload: Any) -> tuple[float | None, float | None]:
    if not isinstance(payload, dict):
        return (None, None)

    values: list[float] = []
    throughput_points = payload.get("throughput_points")
    if isinstance(throughput_points, list):
        for item in throughput_points:
            if not isinstance(item, dict):
                continue
            value = _float_or_none(item.get("throughput_jobs_per_s"))
            if value is None:
                continue
            values.append(value)
    if values:
        avg = sum(values) / len(values)
        return (avg, _population_std(values))

    return (_average_job_throughput_jobs_per_s(payload), None)


def _run_path_relative_to_root(run_dir: Path, root_dir: Path) -> str:
    try:
        return run_dir.relative_to(root_dir).as_posix()
    except ValueError:
        return run_dir.as_posix()


def _load_key_stats_payload(
    run_dir: Path,
    *,
    percent: float | None = None,
) -> dict[str, Any] | None:
    key_stats_path = run_dir / _summary_rel_path(KEY_STATS_SUBPATH, percent=percent)
    if key_stats_path.is_file():
        payload = _load_json(key_stats_path)
        if isinstance(payload, dict):
            return payload
        return None

    processed_dir = run_dir / _processed_subdir_name(percent=percent)
    if not processed_dir.is_dir():
        return None

    try:
        key_stats_helper = _load_helper_module(
            "key_stats_extract",
            "post-process/key-stats/extract_run.py",
        )
        payload = key_stats_helper.build_key_stats_payload(processed_dir)
    except Exception:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def _flatten_numeric_key_stats(
    payload: Any,
    *,
    prefix: tuple[str, ...] = (),
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in sorted(payload.items()):
            flattened.update(
                _flatten_numeric_key_stats(
                    value,
                    prefix=(*prefix, _sanitize_fieldname_part(str(key))),
                )
            )
        return flattened

    numeric = _float_or_none(payload)
    if numeric is None or not prefix:
        return flattened
    flattened["_".join(("key_stats", *prefix))] = numeric
    return flattened


def _full_csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    extra_fieldnames: set[str] = set()
    for row in rows:
        extra_fieldnames.update(
            key for key in row.keys() if key not in BASE_CSV_FIELDNAMES
        )
    return list(BASE_CSV_FIELDNAMES) + sorted(extra_fieldnames)


def _is_min_max_std_fieldname(fieldname: str) -> bool:
    if fieldname in FIXED_METADATA_FIELDNAMES:
        return False
    if fieldname.startswith("key_stats_"):
        return fieldname.endswith(("_min", "_max", "_std"))
    return (
        fieldname.startswith("std_")
        or "_std_" in fieldname
        or fieldname.endswith("_std")
        or fieldname.startswith("min_")
        or "_min_" in fieldname
        or fieldname.endswith("_min")
        or fieldname.startswith("max_")
        or "_max_" in fieldname
        or fieldname.endswith("_max")
    )


def _avg_only_csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return [
        fieldname
        for fieldname in _full_csv_fieldnames(rows)
        if not _is_min_max_std_fieldname(fieldname)
    ]


def _avg_only_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(
            f"{output_path.stem}{AVG_ONLY_SUFFIX}{output_path.suffix}"
        )
    return output_path.with_name(f"{output_path.name}{AVG_ONLY_SUFFIX}")


def _write_csv(
    output_path: Path,
    *,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _build_row_from_run_dir(
    run_dir: Path,
    *,
    root_dir: Path,
    percent: float | None = None,
) -> dict[str, Any]:
    slug_fields = parse_frequency_slug(run_dir.name)
    core_min_mhz: int | None = None
    core_max_mhz: int | None = None
    mem_freq_mhz: int | None = None
    if slug_fields is not None:
        core_min_mhz, core_max_mhz, mem_freq_mhz = slug_fields

    row: dict[str, Any] = {
        "run_path": _run_path_relative_to_root(run_dir, root_dir),
        "frequency_mhz": core_max_mhz,
        "core_min_mhz": core_min_mhz,
        "core_max_mhz": core_max_mhz,
        "mem_freq_mhz": mem_freq_mhz,
        "avg_power_w": None,
        "std_power_w": None,
        "avg_job_throughput_jobs_per_s": None,
        "std_job_throughput_jobs_per_s": None,
        "agent_total_time_avg_s": None,
        "agent_total_time_std_s": None,
        "llm_time_avg_s": None,
        "llm_time_std_s": None,
        "non_llm_time_avg_s": None,
        "non_llm_time_std_s": None,
        "prefill_avg_tokens_per_s": None,
        "prefill_std_tokens_per_s": None,
        "decode_avg_tokens_per_s": None,
        "decode_std_tokens_per_s": None,
        "completion_tokens_avg": None,
        "completion_tokens_std": None,
        "cached_tokens_avg": None,
        "cached_tokens_std": None,
        "duration_ms_avg": None,
        "duration_ms_std": None,
        "prompt_tokens_avg": None,
        "prompt_tokens_std": None,
    }

    power_summary_path = run_dir / _summary_rel_path(POWER_SUMMARY_SUBPATH, percent=percent)
    if power_summary_path.is_file():
        power_payload = _load_json(power_summary_path)
        row["avg_power_w"] = _float_or_none(
            _nested_get(power_payload, "power_stats_w", "avg")
        )
        row["std_power_w"] = _float_or_none(
            _nested_get(power_payload, "power_stats_w", "std")
        )
        if row["std_power_w"] is None:
            power_values: list[float] = []
            power_points = _nested_get(power_payload, "power_points")
            if isinstance(power_points, list):
                for item in power_points:
                    if not isinstance(item, dict):
                        continue
                    value = _float_or_none(item.get("power_w"))
                    if value is None:
                        continue
                    power_values.append(value)
            row["std_power_w"] = _population_std(power_values)

    throughput_path = run_dir / _summary_rel_path(JOB_THROUGHPUT_SUBPATH, percent=percent)
    if throughput_path.is_file():
        throughput_payload = _load_json(throughput_path)
        avg_job_throughput, std_job_throughput = _job_throughput_avg_and_std(
            throughput_payload
        )
        row["avg_job_throughput_jobs_per_s"] = avg_job_throughput
        row["std_job_throughput_jobs_per_s"] = std_job_throughput

    global_summary_path = run_dir / _summary_rel_path(GLOBAL_SUMMARY_SUBPATH, percent=percent)
    if global_summary_path.is_file():
        global_payload = _load_json(global_summary_path)
        row["agent_total_time_avg_s"] = _float_or_none(
            _nested_get(
                global_payload,
                "agent_time_breakdown_s",
                "agent_total_time_stats_s",
                "avg",
            )
        )
        row["agent_total_time_std_s"] = _float_or_none(
            _nested_get(
                global_payload,
                "agent_time_breakdown_s",
                "agent_total_time_stats_s",
                "std",
            )
        )
        row["llm_time_avg_s"] = _float_or_none(
            _nested_get(global_payload, "agent_time_breakdown_s", "llm_time_stats_s", "avg")
        )
        row["llm_time_std_s"] = _float_or_none(
            _nested_get(global_payload, "agent_time_breakdown_s", "llm_time_stats_s", "std")
        )
        row["non_llm_time_avg_s"] = _float_or_none(
            _nested_get(
                global_payload,
                "agent_time_breakdown_s",
                "non_llm_time_stats_s",
                "avg",
            )
        )
        row["non_llm_time_std_s"] = _float_or_none(
            _nested_get(
                global_payload,
                "agent_time_breakdown_s",
                "non_llm_time_stats_s",
                "std",
            )
        )

    llm_request_stats_path = run_dir / _summary_rel_path(
        LLM_REQUEST_STATS_SUBPATH,
        percent=percent,
    )
    if llm_request_stats_path.is_file():
        llm_payload = _load_json(llm_request_stats_path)
        row["prefill_avg_tokens_per_s"] = _float_or_none(
            _nested_get(
                llm_payload,
                "average_stage_speed_tokens_per_s",
                "prefill",
                "avg_tokens_per_s",
            )
        )
        row["prefill_std_tokens_per_s"] = _float_or_none(
            _nested_get(
                llm_payload,
                "average_stage_speed_tokens_per_s",
                "prefill",
                "std_tokens_per_s",
            )
        )
        if row["prefill_std_tokens_per_s"] is None:
            row["prefill_std_tokens_per_s"] = _float_or_none(
                _nested_get(
                    llm_payload,
                    "average_stage_speed_tokens_per_s",
                    "prefill",
                    "std",
                )
            )
        row["decode_avg_tokens_per_s"] = _float_or_none(
            _nested_get(
                llm_payload,
                "average_stage_speed_tokens_per_s",
                "decode",
                "avg_tokens_per_s",
            )
        )
        row["decode_std_tokens_per_s"] = _float_or_none(
            _nested_get(
                llm_payload,
                "average_stage_speed_tokens_per_s",
                "decode",
                "std_tokens_per_s",
            )
        )
        if row["decode_std_tokens_per_s"] is None:
            row["decode_std_tokens_per_s"] = _float_or_none(
                _nested_get(
                    llm_payload,
                    "average_stage_speed_tokens_per_s",
                    "decode",
                    "std",
                )
            )
        row["completion_tokens_avg"] = _float_or_none(
            _nested_get(llm_payload, "metrics", "completion_tokens", "avg")
        )
        row["completion_tokens_std"] = _float_or_none(
            _nested_get(llm_payload, "metrics", "completion_tokens", "std")
        )
        row["cached_tokens_avg"] = _float_or_none(
            _nested_get(llm_payload, "metrics", "cached_tokens", "avg")
        )
        row["cached_tokens_std"] = _float_or_none(
            _nested_get(llm_payload, "metrics", "cached_tokens", "std")
        )
        row["duration_ms_avg"] = _float_or_none(
            _nested_get(llm_payload, "metrics", "duration_ms", "avg")
        )
        row["duration_ms_std"] = _float_or_none(
            _nested_get(llm_payload, "metrics", "duration_ms", "std")
        )
        prompt_avg = _float_or_none(_nested_get(llm_payload, "metrics", "prompt_tokens", "avg"))
        prompt_std = _float_or_none(_nested_get(llm_payload, "metrics", "prompt_tokens", "std"))
        if prompt_avg is None:
            prompt_avg = _float_or_none(_nested_get(llm_payload, "metrics", "prompt_token", "avg"))
        if prompt_std is None:
            prompt_std = _float_or_none(_nested_get(llm_payload, "metrics", "prompt_token", "std"))
        row["prompt_tokens_avg"] = prompt_avg
        row["prompt_tokens_std"] = prompt_std

    key_stats_payload = _load_key_stats_payload(run_dir, percent=percent)
    if key_stats_payload is not None:
        row.update(_flatten_numeric_key_stats(key_stats_payload))

    return row


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    expected_core_min_mhz: int,
    allow_duplicate_frequency: bool,
) -> None:
    if mode != "freq":
        return

    mismatched_min: list[str] = []
    for row in rows:
        core_min_mhz = row.get("core_min_mhz")
        if core_min_mhz is None:
            continue
        if core_min_mhz != expected_core_min_mhz:
            mismatched_min.append(
                f"{row.get('run_path')} (core_min_mhz={core_min_mhz})"
            )
    if mismatched_min:
        mismatch_text = ", ".join(mismatched_min)
        raise ValueError(
            "Found runs that violate --expected-core-min-mhz "
            f"({expected_core_min_mhz}): {mismatch_text}"
        )

    if allow_duplicate_frequency:
        return

    seen_frequency: dict[int, str] = {}
    duplicate_messages: list[str] = []
    for row in rows:
        frequency_mhz = row.get("frequency_mhz")
        if not isinstance(frequency_mhz, int):
            continue
        run_path = str(row.get("run_path"))
        previous = seen_frequency.get(frequency_mhz)
        if previous is None:
            seen_frequency[frequency_mhz] = run_path
            continue
        duplicate_messages.append(
            f"{frequency_mhz} MHz appears in both {previous} and {run_path}"
        )

    if duplicate_messages:
        joined = "; ".join(duplicate_messages)
        raise ValueError(
            "Duplicate frequency_mhz values found; pass --allow-duplicate-frequency "
            f"to keep all rows. Details: {joined}"
        )


def build_rows(
    root_dir: Path,
    *,
    mode: str = "freq",
    percent: float | None = None,
    expected_core_min_mhz: int,
    allow_duplicate_frequency: bool,
) -> list[dict[str, Any]]:
    normalized_percent = _normalize_percent(percent)
    rows = [
        _build_row_from_run_dir(run_dir, root_dir=root_dir, percent=normalized_percent)
        for run_dir in discover_run_dirs(root_dir, mode=mode, percent=normalized_percent)
    ]

    _validate_rows(
        rows,
        mode=mode,
        expected_core_min_mhz=expected_core_min_mhz,
        allow_duplicate_frequency=allow_duplicate_frequency,
    )

    if mode == "freq":
        rows.sort(
            key=lambda row: (
                row["frequency_mhz"] if isinstance(row.get("frequency_mhz"), int) else 2**31,
                _path_sort_key(str(row.get("run_path", ""))),
            )
        )
    else:
        rows.sort(
            key=lambda row: _path_sort_key(str(row.get("run_path", ""))),
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        root_dir = Path(args.root_dir).expanduser().resolve()
        if not root_dir.is_dir():
            raise ValueError(f"Root directory not found: {root_dir}")
        normalized_percent = _normalize_percent(args.percent)

        rows = build_rows(
            root_dir,
            mode=str(args.mode),
            percent=normalized_percent,
            expected_core_min_mhz=int(args.expected_core_min_mhz),
            allow_duplicate_frequency=bool(args.allow_duplicate_frequency),
        )

        default_output_name = (
            DEFAULT_FREQ_OUTPUT_NAME if args.mode == "freq" else DEFAULT_NON_FREQ_OUTPUT_NAME
        )
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output is not None
            else (root_dir / _output_subdir_name(percent=normalized_percent) / default_output_name).resolve()
        )
        avg_only_output_path = _avg_only_output_path(output_path)

        _write_csv(
            output_path,
            rows=rows,
            fieldnames=_full_csv_fieldnames(rows),
        )
        _write_csv(
            avg_only_output_path,
            rows=rows,
            fieldnames=_avg_only_csv_fieldnames(rows),
        )

        print(f"Discovered {len(rows)} run(s) under {root_dir} (mode={args.mode})")
        print(f"Wrote summary CSV: {output_path}")
        print(f"Wrote avg-only summary CSV: {avg_only_output_path}")
        return 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
