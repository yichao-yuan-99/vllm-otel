from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
from typing import Iterable


DEFAULT_OUTPUT_NAME = "context-usage-percentiles.json"
DEFAULT_PERCENTILES = tuple(range(5, 100, 5))
UNRANKED_MODE_STRICT_499 = "strict_499"

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
            "Select one recorded trail at each 5,10,...,95 context-usage percentile "
            "for replay compile --single-trail."
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
            "Optional output path. Default: <run-dir>/original-analysis/percentiles/"
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


def _iter_jsonl_dict_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


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


def discover_run_dirs_with_gateway_output(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for gateway_output_dir in root_dir.rglob("gateway-output"):
        if not gateway_output_dir.is_dir():
            continue
        run_dirs.add(gateway_output_dir.parent.resolve())
    return sorted(run_dirs)


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


def _extract_usage_tokens(record: dict[str, Any]) -> tuple[int | None, int | None]:
    response = record.get("response")
    if not isinstance(response, dict):
        return None, None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
    completion_tokens = _int_or_none(usage.get("completion_tokens"))
    return prompt_tokens, completion_tokens


def _extract_status_code(record: dict[str, Any]) -> int | None:
    status_code = _int_or_none(record.get("status_code"))
    if status_code is not None:
        return status_code
    response_summary = record.get("response_summary")
    if isinstance(response_summary, dict):
        return _int_or_none(response_summary.get("status_code"))
    return None


def _trail_name(trail: dict[str, Any]) -> str:
    gateway_run_id = str(trail.get("gateway_run_id") or "")
    profile_id = trail.get("gateway_profile_id")
    if isinstance(profile_id, int):
        return f"profile-{profile_id}/{gateway_run_id}"
    return gateway_run_id


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _parse_iso8601_or_none(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_seconds(start_time: datetime | None, end_time: datetime | None) -> float | None:
    if start_time is None or end_time is None:
        return None
    try:
        return round(max(0.0, (end_time - start_time).total_seconds()), 6)
    except TypeError:
        return None


def _trail_duration_from_manifest(gateway_run_dir: Path) -> float | None:
    manifest_path = gateway_run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest_payload = _load_json_object(manifest_path)
    if manifest_payload is None:
        return None
    return _duration_seconds(
        _parse_iso8601_or_none(manifest_payload.get("run_start_time")),
        _parse_iso8601_or_none(manifest_payload.get("run_end_time")),
    )


def _trail_duration_from_lifecycle(gateway_run_dir: Path) -> float | None:
    lifecycle_path = gateway_run_dir / "events" / "lifecycle.jsonl"
    if not lifecycle_path.is_file():
        return None

    agent_start_time: datetime | None = None
    agent_end_time: datetime | None = None
    for record in _iter_jsonl_dict_records(lifecycle_path):
        event_type = record.get("event_type")
        timestamp = _parse_iso8601_or_none(record.get("timestamp"))
        if event_type == "agent_start" and agent_start_time is None:
            agent_start_time = timestamp
        elif event_type == "agent_end" and timestamp is not None:
            agent_end_time = timestamp
    return _duration_seconds(agent_start_time, agent_end_time)


def _extract_trail_duration_seconds(gateway_run_dir: Path) -> float | None:
    manifest_duration_s = _trail_duration_from_manifest(gateway_run_dir)
    if manifest_duration_s is not None:
        return manifest_duration_s
    return _trail_duration_from_lifecycle(gateway_run_dir)


def _extract_trail_stats(
    gateway_run_dir: Path,
    *,
    profile_id: int | None,
) -> dict[str, Any]:
    requests_path = gateway_run_dir / "requests" / "model_inference.jsonl"
    if not requests_path.is_file():
        raise ValueError(f"Missing required file: {requests_path}")

    request_count = 0
    requests_with_length = 0
    context_length: int | None = None
    has_status_499 = False
    requests_with_status_499 = 0

    for record in _iter_jsonl_dict_records(requests_path):
        request_count += 1
        status_code = _extract_status_code(record)
        if status_code == 499:
            has_status_499 = True
            requests_with_status_499 += 1
        prompt_tokens, completion_tokens = _extract_usage_tokens(record)
        if prompt_tokens is None or completion_tokens is None:
            continue
        request_length = prompt_tokens + completion_tokens
        if context_length is None:
            context_length = request_length
        else:
            context_length = max(context_length, request_length)
        requests_with_length += 1

    trail = {
        "gateway_run_id": gateway_run_dir.name,
        "gateway_profile_id": profile_id,
        "request_count": request_count,
        "requests_with_length": requests_with_length,
        "context_length": context_length,
        "trail_duration_s": _extract_trail_duration_seconds(gateway_run_dir),
        "has_status_499": has_status_499,
        "requests_with_status_499": requests_with_status_499,
    }
    trail["source_trail_name"] = _trail_name(trail)
    return trail


def _load_trails_from_run_dir(run_dir: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    resolved_run_dir = run_dir.expanduser().resolve()
    gateway_output_dir = resolved_run_dir / "gateway-output"
    if not gateway_output_dir.is_dir():
        raise ValueError(f"Missing required directory: {gateway_output_dir}")

    discovered_run_dirs = discover_gateway_run_dirs(gateway_output_dir)
    if not discovered_run_dirs:
        raise ValueError(
            "No run_* artifacts found under gateway-output. "
            "Expected either gateway-output/run_* or gateway-output/profile-*/run_*."
        )

    trails: list[dict[str, Any]] = []
    for gateway_run_dir, profile_id in discovered_run_dirs:
        trails.append(_extract_trail_stats(gateway_run_dir, profile_id=profile_id))
    return resolved_run_dir, gateway_output_dir.resolve(), trails


def _split_ranked_and_unranked_trails(
    trails: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked_trails: list[dict[str, Any]] = []
    unranked_trails: list[dict[str, Any]] = []
    for trail in trails:
        trail_copy = dict(trail)
        if (
            not isinstance(trail_copy.get("context_length"), int)
            or bool(trail_copy.get("has_status_499"))
        ):
            unranked_trails.append(trail_copy)
        else:
            ranked_trails.append(trail_copy)
    return ranked_trails, unranked_trails


def _rank_trails_ascending_by_context(ranked_trails: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trails_sorted = sorted(
        ranked_trails,
        key=lambda trail: (
            int(trail["context_length"]),
            str(trail.get("source_trail_name") or ""),
        ),
    )
    total = len(trails_sorted)
    ranked_rows: list[dict[str, Any]] = []
    for rank_ascending, trail in enumerate(trails_sorted, start=1):
        ranked_row = dict(trail)
        ranked_row["context_rank_ascending"] = rank_ascending
        ranked_row["context_rank_descending"] = total - rank_ascending + 1
        ranked_rows.append(ranked_row)
    return ranked_rows


def _nearest_rank_index(trail_count: int, percentile: int) -> int:
    if trail_count <= 0:
        raise ValueError("nearest-rank index requires trail_count > 0")
    raw_index = math.ceil((trail_count * percentile) / 100.0) - 1
    return max(0, min(trail_count - 1, raw_index))


def extract_percentiles_from_run_dir(run_dir: Path) -> dict[str, Any]:
    resolved_run_dir, gateway_output_dir, trails = _load_trails_from_run_dir(run_dir)
    ranked_trails, unranked_trails = _split_ranked_and_unranked_trails(trails)
    ranked_trails = _rank_trails_ascending_by_context(ranked_trails)
    trail_count_with_status_499 = sum(1 for trail in trails if bool(trail.get("has_status_499")))

    selected_trails: list[dict[str, Any]] = []
    source_trail_names_by_percentile: dict[str, str] = {}
    for percentile in DEFAULT_PERCENTILES:
        if not ranked_trails:
            break
        selected_index = _nearest_rank_index(len(ranked_trails), percentile)
        selected_trail = dict(ranked_trails[selected_index])
        selected_trails.append(
            {
                "percentile": percentile,
                "selected_index_zero_based": selected_index,
                "source_trail_name": selected_trail["source_trail_name"],
                "gateway_run_id": selected_trail["gateway_run_id"],
                "gateway_profile_id": selected_trail["gateway_profile_id"],
                "context_length": selected_trail["context_length"],
                "trail_duration_s": selected_trail.get("trail_duration_s"),
                "request_count": selected_trail["request_count"],
                "requests_with_length": selected_trail["requests_with_length"],
                "context_rank_ascending": selected_trail["context_rank_ascending"],
                "context_rank_descending": selected_trail["context_rank_descending"],
            }
        )
        source_trail_names_by_percentile[str(percentile)] = selected_trail[
            "source_trail_name"
        ]

    return {
        "source_run_dir": str(resolved_run_dir),
        "source_gateway_output_dir": str(gateway_output_dir),
        "metric": "context_usage",
        "context_usage_definition": (
            "trail context usage = max request length in trail, where request length = "
            "prompt_tokens + completion_tokens"
        ),
        "selection_method": "nearest_rank_on_ascending_context_length",
        "percentiles": list(DEFAULT_PERCENTILES),
        "trail_count_total": len(trails),
        "trail_count_ranked": len(ranked_trails),
        "trail_count_unranked": len(unranked_trails),
        "trail_count_with_status_499": trail_count_with_status_499,
        "unranked_mode": UNRANKED_MODE_STRICT_499,
        "unranked_criteria": (
            "trails without valid request length OR trails containing any request "
            "with status_code=499"
        ),
        "ranked_trails": ranked_trails,
        "unranked_trails": unranked_trails,
        "selected_trails": selected_trails,
        "source_trail_names_by_percentile": source_trail_names_by_percentile,
    }


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "original-analysis" / "percentiles" / DEFAULT_OUTPUT_NAME).resolve()


def extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
    resolved_output_path = (output_path or _default_output_path_for_run(run_dir)).expanduser().resolve()
    payload = extract_percentiles_from_run_dir(run_dir)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
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
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            return 130
    return 0 if failure_count == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.run_dir:
            return _main_run_dir(args)
        return _main_root_dir(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
