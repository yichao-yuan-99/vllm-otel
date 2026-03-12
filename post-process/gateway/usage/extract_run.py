from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from typing import Iterable


DEFAULT_OUTPUT_NAME = "usage-summary.json"


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
            "Extract run-level and per-agent gateway token usage from "
            "gateway-output request logs."
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
            "Optional output path. Default: <run-dir>/post-processed/gateway/usage/"
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


def _iter_jsonl_dict_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


_INTEGER_PATTERN = re.compile(r"[-+]?\d+")


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


def _extract_usage_tokens(record: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    response = record.get("response")
    if not isinstance(response, dict):
        return None, None, None

    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None, None, None

    prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
    completion_tokens = _int_or_none(usage.get("completion_tokens"))

    cached_tokens = None
    prompt_tokens_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_tokens_details, dict):
        cached_tokens = _int_or_none(prompt_tokens_details.get("cached_tokens"))
    return prompt_tokens, completion_tokens, cached_tokens


def _new_usage_accumulator() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "requests_with_prompt_tokens": 0,
        "requests_with_completion_tokens": 0,
        "requests_with_cached_prompt_tokens": 0,
        "max_request_length": 0,
        "requests_with_request_length": 0,
    }


def _add_request_usage(
    accumulator: dict[str, int],
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_prompt_tokens: int | None,
) -> None:
    if prompt_tokens is not None:
        accumulator["prompt_tokens"] += prompt_tokens
        accumulator["requests_with_prompt_tokens"] += 1
    if completion_tokens is not None:
        accumulator["completion_tokens"] += completion_tokens
        accumulator["requests_with_completion_tokens"] += 1
    if cached_prompt_tokens is not None:
        accumulator["cached_prompt_tokens"] += cached_prompt_tokens
        accumulator["requests_with_cached_prompt_tokens"] += 1
    if prompt_tokens is not None and completion_tokens is not None:
        request_length = prompt_tokens + completion_tokens
        accumulator["max_request_length"] = max(
            accumulator["max_request_length"], request_length
        )
        accumulator["requests_with_request_length"] += 1


def _merge_usage_accumulators(target: dict[str, int], incoming: dict[str, int]) -> None:
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "cached_prompt_tokens",
        "requests_with_prompt_tokens",
        "requests_with_completion_tokens",
        "requests_with_cached_prompt_tokens",
        "requests_with_request_length",
    ):
        target[key] += incoming[key]
    target["max_request_length"] = max(
        target["max_request_length"], incoming["max_request_length"]
    )


def _usage_payload_from_accumulator(accumulator: dict[str, int]) -> dict[str, int | None]:
    prompt_tokens = accumulator["prompt_tokens"]
    completion_tokens = accumulator["completion_tokens"]
    cached_prompt_tokens = accumulator["cached_prompt_tokens"]
    max_request_length: int | None = None
    if accumulator["requests_with_request_length"] > 0:
        max_request_length = accumulator["max_request_length"]
    return {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": completion_tokens,
        "completion_tokens": completion_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "prefill_prompt_tokens": prompt_tokens - cached_prompt_tokens,
        "max_request_length": max_request_length,
        "requests_with_prompt_tokens": accumulator["requests_with_prompt_tokens"],
        "requests_with_generation_tokens": accumulator["requests_with_completion_tokens"],
        "requests_with_completion_tokens": accumulator["requests_with_completion_tokens"],
        "requests_with_cached_prompt_tokens": accumulator["requests_with_cached_prompt_tokens"],
        "requests_with_request_length": accumulator["requests_with_request_length"],
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


def _collect_agent_usage(
    gateway_run_dir: Path,
    *,
    profile_id: int | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    requests_path = gateway_run_dir / "requests" / "model_inference.jsonl"
    if not requests_path.is_file():
        raise ValueError(f"Missing required file: {requests_path}")

    request_count = 0
    usage_accumulator = _new_usage_accumulator()
    for record in _iter_jsonl_dict_records(requests_path):
        request_count += 1
        prompt_tokens, completion_tokens, cached_prompt_tokens = _extract_usage_tokens(record)
        _add_request_usage(
            usage_accumulator,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )

    usage_payload = _usage_payload_from_accumulator(usage_accumulator)
    payload = {
        "gateway_run_id": gateway_run_dir.name,
        "gateway_profile_id": profile_id,
        "api_token_hash": _extract_api_token_hash(gateway_run_dir),
        "request_count": request_count,
        "usage": usage_payload,
    }
    return payload, usage_accumulator


def extract_gateway_usage_from_run_dir(run_dir: Path) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    gateway_output_dir = resolved_run_dir / "gateway-output"
    if not gateway_output_dir.is_dir():
        raise ValueError(f"Missing gateway-output directory: {gateway_output_dir}")

    discovered_run_dirs = discover_gateway_run_dirs(gateway_output_dir)
    if not discovered_run_dirs:
        raise ValueError(
            "No run_* artifacts found under gateway-output. "
            "Expected either gateway-output/run_* or gateway-output/profile-*/run_*."
        )

    total_request_count = 0
    run_usage_accumulator = _new_usage_accumulator()
    agents: list[dict[str, Any]] = []
    for gateway_run_dir, profile_id in discovered_run_dirs:
        agent_payload, agent_usage_accumulator = _collect_agent_usage(
            gateway_run_dir,
            profile_id=profile_id,
        )
        agents.append(agent_payload)
        total_request_count += agent_payload["request_count"]
        _merge_usage_accumulators(run_usage_accumulator, agent_usage_accumulator)

    run_usage = _usage_payload_from_accumulator(run_usage_accumulator)
    worker_max_lengths: list[int] = []
    for agent in agents:
        usage_payload = agent.get("usage")
        if not isinstance(usage_payload, dict):
            continue
        max_request_length = usage_payload.get("max_request_length")
        if isinstance(max_request_length, int):
            worker_max_lengths.append(max_request_length)

    run_usage["avg_worker_max_request_length"] = (
        (sum(worker_max_lengths) / len(worker_max_lengths))
        if worker_max_lengths
        else None
    )

    return {
        "source_run_dir": str(resolved_run_dir),
        "source_gateway_output_dir": str(gateway_output_dir.resolve()),
        "agent_count": len(agents),
        "request_count": total_request_count,
        "usage": run_usage,
        "agents": agents,
    }


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "gateway" / "usage" / DEFAULT_OUTPUT_NAME).resolve()


def extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_path = (output_path or _default_output_path_for_run(resolved_run_dir)).expanduser().resolve()
    result = extract_gateway_usage_from_run_dir(resolved_run_dir)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
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
