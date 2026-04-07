from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pp_common.service_failure import cutoff_datetime_utc_from_payload
from pp_common.service_failure import ensure_service_failure_payload
from pp_common.service_failure import parse_iso8601_to_utc
from pp_common.profile_id import gateway_run_profile_id_from_manifest
from pp_common.profile_id import profile_label


DEFAULT_OUTPUT_NAME = "agent-output-throughput.json"
DEFAULT_THROUGHPUT_HISTOGRAM_BIN_SIZE = 1.0

_INTEGER_PATTERN = re.compile(r"[-+]?\d+")


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
            "Extract per-agent output-token throughput from gateway request logs "
            "for one run or for all runs under a root directory."
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
        "--output",
        default=None,
        help=(
            "Optional output path. Default: "
            "<run-dir>/post-processed/agent-output-throughput/"
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
            run_dirs.append((run_dir, gateway_run_profile_id_from_manifest(run_dir)))

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


def discover_run_dirs_with_gateway_output(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for gateway_output_dir in root_dir.rglob("gateway-output"):
        if not gateway_output_dir.is_dir():
            continue
        run_dirs.add(gateway_output_dir.parent.resolve())
    return sorted(run_dirs)


def _iter_jsonl_dict_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        if _INTEGER_PATTERN.fullmatch(stripped):
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        if math.isfinite(numeric):
            return numeric
    return None


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


def _request_within_cutoff(
    record: dict[str, Any],
    *,
    cutoff_time_utc: datetime | None = None,
) -> bool:
    if cutoff_time_utc is None:
        return True
    request_start_time = parse_iso8601_to_utc(record.get("request_start_time"))
    request_end_time = parse_iso8601_to_utc(record.get("request_end_time"))
    if request_start_time is not None and request_start_time > cutoff_time_utc:
        return False
    if request_end_time is not None and request_end_time > cutoff_time_utc:
        return False
    return True


def _extract_completion_tokens(record: dict[str, Any]) -> int | None:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    completion_tokens = _int_or_none(usage.get("completion_tokens"))
    if completion_tokens is None or completion_tokens < 0:
        return None
    return completion_tokens


def _extract_request_duration_s(record: dict[str, Any]) -> float | None:
    for key in ("request_duration_ms", "duration_ms"):
        duration_ms = _float_or_none(record.get(key))
        if duration_ms is None or duration_ms < 0.0:
            continue
        return duration_ms / 1000.0

    start_time = _parse_iso8601(record.get("request_start_time"))
    end_time = _parse_iso8601(record.get("request_end_time"))
    if start_time is None or end_time is None:
        return None
    duration_s = (end_time - start_time).total_seconds()
    if duration_s < 0.0:
        return None
    return duration_s


def _new_throughput_accumulator() -> dict[str, int | float]:
    return {
        "request_count": 0,
        "requests_with_output_tokens": 0,
        "requests_with_llm_request_duration": 0,
        "requests_with_output_tokens_and_llm_request_duration": 0,
        "output_tokens": 0,
        "llm_request_duration_s": 0.0,
    }


def _add_request_to_accumulator(
    accumulator: dict[str, int | float],
    *,
    completion_tokens: int | None,
    request_duration_s: float | None,
) -> None:
    accumulator["request_count"] += 1

    has_output_tokens = completion_tokens is not None
    has_duration = request_duration_s is not None

    if has_output_tokens:
        accumulator["requests_with_output_tokens"] += 1
        accumulator["output_tokens"] += completion_tokens
    if has_duration:
        accumulator["requests_with_llm_request_duration"] += 1
        accumulator["llm_request_duration_s"] += request_duration_s
    if has_output_tokens and has_duration:
        accumulator["requests_with_output_tokens_and_llm_request_duration"] += 1


def _merge_accumulators(
    target: dict[str, int | float],
    incoming: dict[str, int | float],
) -> None:
    for key in (
        "request_count",
        "requests_with_output_tokens",
        "requests_with_llm_request_duration",
        "requests_with_output_tokens_and_llm_request_duration",
        "output_tokens",
    ):
        target[key] += int(incoming[key])
    target["llm_request_duration_s"] += float(incoming["llm_request_duration_s"])


def _throughput_from_totals(
    *,
    output_tokens: int,
    llm_request_duration_s: float,
) -> float | None:
    if llm_request_duration_s <= 0.0:
        return None
    return output_tokens / llm_request_duration_s


def _payload_from_accumulator(accumulator: dict[str, int | float]) -> dict[str, Any]:
    output_tokens = int(accumulator["output_tokens"])
    llm_request_duration_s = round(float(accumulator["llm_request_duration_s"]), 6)
    throughput = _throughput_from_totals(
        output_tokens=output_tokens,
        llm_request_duration_s=llm_request_duration_s,
    )
    return {
        "request_count": int(accumulator["request_count"]),
        "requests_with_output_tokens": int(accumulator["requests_with_output_tokens"]),
        "requests_with_llm_request_duration": int(
            accumulator["requests_with_llm_request_duration"]
        ),
        "requests_with_output_tokens_and_llm_request_duration": int(
            accumulator["requests_with_output_tokens_and_llm_request_duration"]
        ),
        "output_tokens": output_tokens,
        "completion_tokens": output_tokens,
        "llm_request_duration_s": llm_request_duration_s,
        "output_throughput_tokens_per_s": (
            round(throughput, 6) if throughput is not None else None
        ),
    }


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
        "avg": round(avg_value, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "std": round(math.sqrt(variance), 6),
    }


def build_agent_output_throughput_histogram(
    values: list[float],
    *,
    bin_size: float = DEFAULT_THROUGHPUT_HISTOGRAM_BIN_SIZE,
) -> dict[str, Any]:
    if bin_size <= 0.0:
        raise ValueError(f"bin_size must be positive: {bin_size}")

    finite_values = [float(value) for value in values if math.isfinite(value)]
    if not finite_values:
        return {
            "metric": "output_throughput_tokens_per_s",
            "bin_size": round(bin_size, 6),
            "sample_count": 0,
            "bin_count": 0,
            "min": None,
            "max": None,
            "bins": [],
        }

    min_value = min(finite_values)
    max_value = max(finite_values)
    min_bin_index = math.floor(min_value / bin_size)
    max_bin_index = math.floor(max_value / bin_size)
    counts_by_index = {
        bin_index: 0 for bin_index in range(min_bin_index, max_bin_index + 1)
    }
    for value in finite_values:
        counts_by_index[math.floor(value / bin_size)] += 1

    bins = [
        {
            "bin_start": round(bin_index * bin_size, 6),
            "bin_end": round((bin_index + 1) * bin_size, 6),
            "count": counts_by_index[bin_index],
        }
        for bin_index in range(min_bin_index, max_bin_index + 1)
    ]
    return {
        "metric": "output_throughput_tokens_per_s",
        "bin_size": round(bin_size, 6),
        "sample_count": len(finite_values),
        "bin_count": len(bins),
        "min": round(min_value, 6),
        "max": round(max_value, 6),
        "bins": bins,
    }


def _extract_api_token_hash(gateway_run_dir: Path) -> str | None:
    lifecycle_path = gateway_run_dir / "events" / "lifecycle.jsonl"
    if not lifecycle_path.is_file():
        return None
    try:
        for record in _iter_jsonl_dict_records(lifecycle_path):
            candidate = record.get("api_token_hash")
            if isinstance(candidate, str) and candidate:
                return candidate
    except (json.JSONDecodeError, OSError):
        return None
    return None


def _hash_api_token(api_token: Any) -> str | None:
    if not isinstance(api_token, str) or not api_token:
        return None
    return hashlib.sha256(api_token.encode("utf-8")).hexdigest()


def _replay_status_mapping_for_run(run_dir: Path) -> dict[str, str]:
    summary_path = run_dir / "replay" / "summary.json"
    if not summary_path.is_file():
        return {}

    try:
        payload = _load_json(summary_path)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}

    worker_results = payload.get("worker_results")
    if not isinstance(worker_results, dict):
        return {}

    status_by_api_token_hash: dict[str, str] = {}
    for worker_payload in worker_results.values():
        if not isinstance(worker_payload, dict):
            continue
        api_token_hash = _hash_api_token(worker_payload.get("api_token"))
        status = worker_payload.get("status")
        if api_token_hash is None or not isinstance(status, str) or not status:
            continue
        status_by_api_token_hash[api_token_hash] = status
    return status_by_api_token_hash


def _collect_agent_output_throughput(
    gateway_run_dir: Path,
    *,
    profile_id: int | None,
    replay_status_by_api_token_hash: dict[str, str] | None = None,
    cutoff_time_utc: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, int | float]]:
    requests_path = gateway_run_dir / "requests" / "model_inference.jsonl"
    if not requests_path.is_file():
        raise ValueError(f"Missing required file: {requests_path}")

    accumulator = _new_throughput_accumulator()
    for record in _iter_jsonl_dict_records(requests_path):
        if not _request_within_cutoff(record, cutoff_time_utc=cutoff_time_utc):
            continue
        _add_request_to_accumulator(
            accumulator,
            completion_tokens=_extract_completion_tokens(record),
            request_duration_s=_extract_request_duration_s(record),
        )

    api_token_hash = _extract_api_token_hash(gateway_run_dir)
    replay_worker_status = None
    if replay_status_by_api_token_hash and api_token_hash is not None:
        replay_worker_status = replay_status_by_api_token_hash.get(api_token_hash)

    payload = _payload_from_accumulator(accumulator)
    payload.update(
        {
            "gateway_run_id": gateway_run_dir.name,
            "gateway_profile_id": profile_id,
            "api_token_hash": api_token_hash,
            "replay_worker_status": replay_worker_status,
            "replay_completed": (
                replay_worker_status == "completed"
                if replay_worker_status is not None
                else None
            ),
        }
    )
    return payload, accumulator


def _build_result_payload(
    *,
    resolved_run_dir: Path,
    gateway_output_dir: Path,
    service_failure_payload: dict[str, Any],
    agents: list[dict[str, Any]],
    accumulator: dict[str, int | float],
    gateway_profile_id: int | None = None,
) -> dict[str, Any]:
    throughput_values = [
        float(agent["output_throughput_tokens_per_s"])
        for agent in agents
        if isinstance(agent.get("output_throughput_tokens_per_s"), (int, float))
    ]

    result = _payload_from_accumulator(accumulator)
    result.update(
        {
            "source_run_dir": str(resolved_run_dir),
            "source_gateway_output_dir": str(gateway_output_dir.resolve()),
            "service_failure_detected": bool(
                service_failure_payload.get("service_failure_detected", False)
            ),
            "service_failure_cutoff_time_utc": service_failure_payload.get(
                "cutoff_time_utc"
            ),
            "agent_count": len(agents),
            "agent_output_throughput_tokens_per_s_summary": _summarize_values(
                throughput_values
            ),
            "agent_output_throughput_tokens_per_s_histogram": (
                build_agent_output_throughput_histogram(throughput_values)
            ),
            "agents": agents,
        }
    )
    if gateway_profile_id is not None:
        result["gateway_profile_id"] = gateway_profile_id
    return result


def extract_agent_output_throughput_from_run_dir(run_dir: Path) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    service_failure_payload = ensure_service_failure_payload(resolved_run_dir)
    cutoff_time_utc = cutoff_datetime_utc_from_payload(service_failure_payload)

    gateway_output_dir = resolved_run_dir / "gateway-output"
    if not gateway_output_dir.is_dir():
        raise ValueError(f"Missing gateway-output directory: {gateway_output_dir}")

    discovered_run_dirs = discover_gateway_run_dirs(gateway_output_dir)
    if not discovered_run_dirs:
        raise ValueError(
            "No run_* artifacts found under gateway-output. "
            "Expected either gateway-output/run_* or gateway-output/profile-*/run_*."
        )

    replay_status_by_api_token_hash = _replay_status_mapping_for_run(resolved_run_dir)
    run_accumulator = _new_throughput_accumulator()
    accumulators_by_profile: dict[int, dict[str, int | float]] = {}
    agents: list[dict[str, Any]] = []
    agents_by_profile: dict[int, list[dict[str, Any]]] = {}
    for gateway_run_dir, profile_id in discovered_run_dirs:
        agent_payload, agent_accumulator = _collect_agent_output_throughput(
            gateway_run_dir,
            profile_id=profile_id,
            replay_status_by_api_token_hash=replay_status_by_api_token_hash,
            cutoff_time_utc=cutoff_time_utc,
        )
        agents.append(agent_payload)
        _merge_accumulators(run_accumulator, agent_accumulator)
        if profile_id is None:
            continue
        agents_by_profile.setdefault(profile_id, []).append(agent_payload)
        profile_accumulator = accumulators_by_profile.setdefault(
            profile_id,
            _new_throughput_accumulator(),
        )
        _merge_accumulators(profile_accumulator, agent_accumulator)

    port_profile_ids = sorted(agents_by_profile)
    series_by_profile = {
        profile_label(gateway_profile_id): _build_result_payload(
            resolved_run_dir=resolved_run_dir,
            gateway_output_dir=gateway_output_dir,
            service_failure_payload=service_failure_payload,
            agents=agents_by_profile[gateway_profile_id],
            accumulator=accumulators_by_profile[gateway_profile_id],
            gateway_profile_id=gateway_profile_id,
        )
        for gateway_profile_id in port_profile_ids
    }

    result = _build_result_payload(
        resolved_run_dir=resolved_run_dir,
        gateway_output_dir=gateway_output_dir,
        service_failure_payload=service_failure_payload,
        agents=agents,
        accumulator=run_accumulator,
    )
    result.update(
        {
            "multi_profile": len(port_profile_ids) > 1,
            "port_profile_ids": port_profile_ids,
            "series_keys": list(series_by_profile.keys()),
            "series_by_profile": series_by_profile,
        }
    )
    return result


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "agent-output-throughput" / DEFAULT_OUTPUT_NAME).resolve()


def _series_output_paths(
    aggregate_output_path: Path,
    result: dict[str, Any],
) -> dict[str, Path]:
    raw_series_by_profile = result.get("series_by_profile")
    if not isinstance(raw_series_by_profile, dict):
        return {}

    output_paths: dict[str, Path] = {}
    for series_key, series_payload in raw_series_by_profile.items():
        if not isinstance(series_key, str) or not series_key:
            continue
        if not isinstance(series_payload, dict):
            continue
        output_paths[series_key] = (
            aggregate_output_path.parent / series_key / aggregate_output_path.name
        ).resolve()
    return output_paths


def extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_path = (
        output_path or _default_output_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    result = extract_agent_output_throughput_from_run_dir(resolved_run_dir)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    for series_key, series_output_path in _series_output_paths(
        resolved_output_path,
        result,
    ).items():
        series_payload = result["series_by_profile"][series_key]
        series_output_path.parent.mkdir(parents=True, exist_ok=True)
        series_output_path.write_text(
            json.dumps(series_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return resolved_output_path


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


def _main_run_dir(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")
    run_dir = Path(args.run_dir).expanduser().resolve()
    if run_dir.is_file():
        raise ValueError(
            f"--run-dir must point to a directory, not a file under {run_dir.parent}"
        )
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    resolved_output_path = extract_run_dir(run_dir, output_path=output_path)
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
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
