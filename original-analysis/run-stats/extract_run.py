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


DEFAULT_OUTPUT_NAME = "run-stats-summary.json"

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
            "Extract run-level stats for con-driver runs (dataset, score, job counts, "
            "and request-token length stats)."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="One con-driver run directory.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with meta/results.json + meta/run_manifest.json will be processed."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/run-stats/"
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
        return float(value)
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


def _extract_job_stats(
    gateway_run_dir: Path,
    *,
    profile_id: int | None,
) -> tuple[dict[str, Any], int, int, int, int]:
    requests_path = gateway_run_dir / "requests" / "model_inference.jsonl"
    if not requests_path.is_file():
        raise ValueError(f"Missing required file: {requests_path}")

    request_count = 0
    prompt_token_sum = 0
    generation_token_sum = 0
    prompt_token_count = 0
    generation_token_count = 0
    max_request_length: int | None = None

    for record in _iter_jsonl_dict_records(requests_path):
        request_count += 1
        prompt_tokens, generation_tokens = _extract_usage_tokens(record)
        if prompt_tokens is not None:
            prompt_token_sum += prompt_tokens
            prompt_token_count += 1
        if generation_tokens is not None:
            generation_token_sum += generation_tokens
            generation_token_count += 1
        if prompt_tokens is not None and generation_tokens is not None:
            request_length = prompt_tokens + generation_tokens
            if max_request_length is None:
                max_request_length = request_length
            else:
                max_request_length = max(max_request_length, request_length)

    avg_prompt_tokens_per_request = (
        (prompt_token_sum / prompt_token_count) if prompt_token_count > 0 else None
    )
    avg_generation_tokens_per_request = (
        (generation_token_sum / generation_token_count)
        if generation_token_count > 0
        else None
    )

    job_payload = {
        "gateway_run_id": gateway_run_dir.name,
        "gateway_profile_id": profile_id,
        "request_count": request_count,
        "max_request_length": max_request_length,
        "avg_prompt_tokens_per_request": avg_prompt_tokens_per_request,
        "avg_generation_tokens_per_request": avg_generation_tokens_per_request,
    }
    return (
        job_payload,
        prompt_token_sum,
        prompt_token_count,
        generation_token_sum,
        generation_token_count,
    )


def _extract_dataset(run_manifest_payload: dict[str, Any], results_payload: list[dict[str, Any]]) -> str | None:
    dataset = run_manifest_payload.get("dataset_mode_dataset")
    if isinstance(dataset, str) and dataset:
        return dataset
    for entry in results_payload:
        candidate = entry.get("dataset")
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _extract_agent_type(results_payload: list[dict[str, Any]], run_dir: Path) -> str | None:
    for entry in results_payload:
        command = entry.get("command")
        if not isinstance(command, list):
            continue
        command_parts = [part for part in command if isinstance(part, str)]
        for flag_name in ("--agent-name", "--agent"):
            try:
                idx = command_parts.index(flag_name)
            except ValueError:
                continue
            value_idx = idx + 1
            if value_idx >= len(command_parts):
                continue
            candidate = command_parts[value_idx].strip()
            if candidate:
                return candidate

    fallback = run_dir.parent.name.strip()
    return fallback if fallback else None


def extract_run_stats_from_run_dir(run_dir: Path) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    manifest_path = resolved_run_dir / "meta" / "run_manifest.json"
    results_path = resolved_run_dir / "meta" / "results.json"
    gateway_output_dir = resolved_run_dir / "gateway-output"

    if not manifest_path.is_file():
        raise ValueError(f"Missing required file: {manifest_path}")
    if not results_path.is_file():
        raise ValueError(f"Missing required file: {results_path}")
    if not gateway_output_dir.is_dir():
        raise ValueError(f"Missing required directory: {gateway_output_dir}")

    run_manifest_payload = _load_json(manifest_path)
    if not isinstance(run_manifest_payload, dict):
        raise ValueError(f"run_manifest must be JSON object: {manifest_path}")
    results_raw = _load_json(results_path)
    if not isinstance(results_raw, list):
        raise ValueError(f"results must be JSON array: {results_path}")

    results_payload = [entry for entry in results_raw if isinstance(entry, dict)]
    discovered_run_dirs = discover_gateway_run_dirs(gateway_output_dir)
    if not discovered_run_dirs:
        raise ValueError(
            "No run_* artifacts found under gateway-output. "
            "Expected either gateway-output/run_* or gateway-output/profile-*/run_*."
        )

    jobs: list[dict[str, Any]] = []
    total_prompt_token_sum = 0
    total_prompt_token_count = 0
    total_generation_token_sum = 0
    total_generation_token_count = 0
    for gateway_run_dir, profile_id in discovered_run_dirs:
        (
            job_payload,
            prompt_token_sum,
            prompt_token_count,
            generation_token_sum,
            generation_token_count,
        ) = _extract_job_stats(gateway_run_dir, profile_id=profile_id)
        jobs.append(job_payload)
        total_prompt_token_sum += prompt_token_sum
        total_prompt_token_count += prompt_token_count
        total_generation_token_sum += generation_token_sum
        total_generation_token_count += generation_token_count

    job_max_request_lengths = [
        max_request_length
        for max_request_length in (job.get("max_request_length") for job in jobs)
        if isinstance(max_request_length, int)
    ]
    avg_job_max_request_length = (
        (sum(job_max_request_lengths) / len(job_max_request_lengths))
        if job_max_request_lengths
        else None
    )
    max_job_max_request_length = (
        max(job_max_request_lengths)
        if job_max_request_lengths
        else None
    )
    job_avg_prompt_tokens_per_request = [
        avg_prompt_tokens
        for avg_prompt_tokens in (
            _float_or_none(job.get("avg_prompt_tokens_per_request")) for job in jobs
        )
        if avg_prompt_tokens is not None
    ]
    job_avg_generation_tokens_per_request = [
        avg_generation_tokens
        for avg_generation_tokens in (
            _float_or_none(job.get("avg_generation_tokens_per_request")) for job in jobs
        )
        if avg_generation_tokens is not None
    ]

    avg_run_prompt_tokens_per_request = (
        (total_prompt_token_sum / total_prompt_token_count)
        if total_prompt_token_count > 0
        else None
    )
    avg_run_generation_tokens_per_request = (
        (total_generation_token_sum / total_generation_token_count)
        if total_generation_token_count > 0
        else None
    )
    avg_turns_per_run = (
        (sum(_int_or_none(job.get("request_count")) or 0 for job in jobs) / len(jobs))
        if jobs
        else None
    )
    turn_counts = [
        request_count
        for request_count in (_int_or_none(job.get("request_count")) for job in jobs)
        if request_count is not None
    ]
    max_turns_per_run = max(turn_counts) if turn_counts else None

    return {
        "source_run_dir": str(resolved_run_dir),
        "source_manifest_path": str(manifest_path.resolve()),
        "source_results_path": str(results_path.resolve()),
        "agent_type": _extract_agent_type(results_payload, resolved_run_dir),
        "dataset": _extract_dataset(run_manifest_payload, results_payload),
        "score": _float_or_none(run_manifest_payload.get("reward_avg")),
        "job_count": len(jobs),
        "results_count": len(results_payload),
        "jobs": jobs,
        "job_max_request_lengths": job_max_request_lengths,
        "avg_job_max_request_length": avg_job_max_request_length,
        "max_job_max_request_length": max_job_max_request_length,
        "job_avg_prompt_tokens_per_request": job_avg_prompt_tokens_per_request,
        "job_avg_generation_tokens_per_request": job_avg_generation_tokens_per_request,
        "avg_run_prompt_tokens_per_request": avg_run_prompt_tokens_per_request,
        "avg_run_generation_tokens_per_request": avg_run_generation_tokens_per_request,
        "avg_turns_per_run": avg_turns_per_run,
        "max_turns_per_run": max_turns_per_run,
    }


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "run-stats" / DEFAULT_OUTPUT_NAME).resolve()


def discover_con_driver_run_dirs(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for results_path in root_dir.rglob("results.json"):
        if not results_path.is_file():
            continue
        if results_path.parent.name != "meta":
            continue
        manifest_path = results_path.parent / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        run_dirs.add(results_path.parent.parent.resolve())
    return sorted(run_dirs)


def extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_path = (output_path or _default_output_path_for_run(resolved_run_dir)).expanduser().resolve()
    result = extract_run_stats_from_run_dir(resolved_run_dir)
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

    run_dirs = discover_con_driver_run_dirs(root_dir)
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
