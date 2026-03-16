#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a list of shell-script jobs over grouped start-group port slots."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


DEFAULT_JOB_LIST_ENV = "SBATCH_ORCHESTRATOR_JOB_LIST"
DEFAULT_PROFILE_LIST_ENV = "SBATCH_ORCHESTRATOR_PROFILE_IDS_CSV"
DEFAULT_PORT_LIST_ENV = "SBATCH_ORCHESTRATOR_PORTS_CSV"
DEFAULT_SUMMARY_PATH_ENV = "SBATCH_ORCHESTRATOR_SUMMARY_PATH"


@dataclass(frozen=True)
class PortSlot:
    slot_index: int
    profile_id: int
    port: int


@dataclass(frozen=True)
class JobSpec:
    job_index: int
    source_line_number: int
    script_path: str
    script_args: list[str]


@dataclass
class RunningJob:
    job: JobSpec
    slot: PortSlot
    command: list[str]
    process: subprocess.Popen
    started_at: str
    started_monotonic: float


@dataclass(frozen=True)
class JobResult:
    job_index: int
    source_line_number: int
    script_path: str
    profile_id: int
    port: int
    command: list[str]
    started_at: str
    finished_at: str
    duration_s: float
    exit_code: int | None
    status: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_csv_int_list(*, raw: str, label: str) -> list[int]:
    values: list[int] = []
    tokens = [item.strip() for item in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{label} must be a non-empty comma-separated list")
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"{label} contains non-integer value {token!r}") from exc
        if value < 0:
            raise ValueError(f"{label} values must be >= 0")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} cannot contain duplicate values")
    return values


def _parse_job_list_file(path: Path) -> list[JobSpec]:
    if not path.exists():
        raise ValueError(f"job list file not found: {path}")
    if not path.is_file():
        raise ValueError(f"job list path must be a file: {path}")

    jobs: list[JobSpec] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: failed to parse line as shell tokens: {exc}"
            ) from exc
        if not tokens:
            continue

        script_path_raw = tokens[0]
        script_path = Path(script_path_raw).expanduser()
        if not script_path.is_absolute():
            raise ValueError(
                f"{path}:{line_number}: script path must be absolute (got {script_path_raw!r})"
            )
        script_path = script_path.resolve()
        if not script_path.exists():
            raise ValueError(f"{path}:{line_number}: script not found: {script_path}")
        if not script_path.is_file():
            raise ValueError(f"{path}:{line_number}: script path is not a file: {script_path}")

        jobs.append(
            JobSpec(
                job_index=len(jobs) + 1,
                source_line_number=line_number,
                script_path=str(script_path),
                script_args=tokens[1:],
            )
        )

    if not jobs:
        raise ValueError(f"job list file has no runnable jobs: {path}")
    return jobs


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbatch-orchestrator",
        description=(
            "Assign shell-script jobs to grouped start-group port slots. "
            "Each line in --job-list-file is: /absolute/path/to/script.sh [extra args...]"
        ),
    )
    parser.add_argument(
        "--job-list-file",
        default=os.environ.get(DEFAULT_JOB_LIST_ENV, ""),
        help=f"Path to job-list file (default: env {DEFAULT_JOB_LIST_ENV})",
    )
    parser.add_argument(
        "--profile-list",
        default=os.environ.get(DEFAULT_PROFILE_LIST_ENV, ""),
        help=(
            "Comma-separated profile IDs aligned with --port-list "
            f"(default: env {DEFAULT_PROFILE_LIST_ENV})"
        ),
    )
    parser.add_argument(
        "--port-list",
        default=os.environ.get(DEFAULT_PORT_LIST_ENV, ""),
        help=(
            "Comma-separated vLLM ports aligned with --profile-list "
            f"(default: env {DEFAULT_PORT_LIST_ENV})"
        ),
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=float(os.environ.get("SBATCH_ORCHESTRATOR_POLL_INTERVAL_S", "1.0")),
        help="Polling interval in seconds while jobs are running.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=_env_bool("SBATCH_ORCHESTRATOR_FAIL_FAST", default=False),
        help="Stop launching new jobs after the first failure.",
    )
    parser.add_argument(
        "--summary-path",
        default=os.environ.get(DEFAULT_SUMMARY_PATH_ENV, ""),
        help=f"Optional JSON summary output path (default: env {DEFAULT_SUMMARY_PATH_ENV})",
    )
    return parser


def _run_jobs(
    *,
    jobs: list[JobSpec],
    slots: list[PortSlot],
    poll_interval_s: float,
    fail_fast: bool,
) -> dict[str, object]:
    if poll_interval_s <= 0:
        raise ValueError("--poll-interval-s must be > 0")

    pending = deque(jobs)
    available_slots = deque(slots)
    running_by_slot_index: dict[int, RunningJob] = {}
    results: list[JobResult] = []
    interrupted = False
    stop_launching = False
    started_at = _utc_now_iso()

    try:
        while pending or running_by_slot_index:
            while pending and available_slots and not stop_launching:
                job = pending.popleft()
                slot = available_slots.popleft()
                command = ["bash", job.script_path, str(slot.port), *job.script_args]
                child_env = os.environ.copy()
                child_env["PORT_PROFILE_ID"] = str(slot.profile_id)
                child_env["VLLM_PORT"] = str(slot.port)
                child_env["VLLM_SERVICE_PORT"] = str(slot.port)
                child_env["SBATCH_ORCHESTRATOR_SLOT_INDEX"] = str(slot.slot_index)
                child_env["SBATCH_ORCHESTRATOR_JOB_INDEX"] = str(job.job_index)

                print(
                    (
                        f"[launch] job={job.job_index} line={job.source_line_number} "
                        f"profile={slot.profile_id} port={slot.port} "
                        f"script={job.script_path}"
                    ),
                    flush=True,
                )
                process = subprocess.Popen(command, env=child_env)
                running_by_slot_index[slot.slot_index] = RunningJob(
                    job=job,
                    slot=slot,
                    command=command,
                    process=process,
                    started_at=_utc_now_iso(),
                    started_monotonic=time.monotonic(),
                )

            if not running_by_slot_index:
                if stop_launching:
                    break
                continue

            time.sleep(poll_interval_s)
            for slot_index, running in list(running_by_slot_index.items()):
                exit_code = running.process.poll()
                if exit_code is None:
                    continue

                duration_s = round(max(0.0, time.monotonic() - running.started_monotonic), 3)
                status = "succeeded" if exit_code == 0 else "failed"
                if exit_code != 0 and fail_fast:
                    stop_launching = True

                result = JobResult(
                    job_index=running.job.job_index,
                    source_line_number=running.job.source_line_number,
                    script_path=running.job.script_path,
                    profile_id=running.slot.profile_id,
                    port=running.slot.port,
                    command=running.command,
                    started_at=running.started_at,
                    finished_at=_utc_now_iso(),
                    duration_s=duration_s,
                    exit_code=exit_code,
                    status=status,
                )
                results.append(result)
                print(
                    (
                        f"[done] job={running.job.job_index} profile={running.slot.profile_id} "
                        f"port={running.slot.port} status={status} exit_code={exit_code}"
                    ),
                    flush=True,
                )
                running_by_slot_index.pop(slot_index, None)
                available_slots.append(running.slot)
    except KeyboardInterrupt:
        interrupted = True
        print("[interrupt] received KeyboardInterrupt; terminating running jobs", flush=True)
        for running in running_by_slot_index.values():
            try:
                running.process.terminate()
            except Exception:
                pass
        for running in running_by_slot_index.values():
            try:
                running.process.wait(timeout=10.0)
            except Exception:
                pass

    failed_count = sum(1 for result in results if result.status == "failed")
    succeeded_count = sum(1 for result in results if result.status == "succeeded")
    if interrupted:
        status = "interrupted"
    elif failed_count == 0 and not pending:
        status = "ok"
    else:
        status = "failed"

    return {
        "status": status,
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "slot_count": len(slots),
        "total_jobs": len(jobs),
        "launched_jobs": len(results),
        "pending_jobs": len(pending),
        "succeeded": succeeded_count,
        "failed": failed_count,
        "interrupted": interrupted,
        "fail_fast": fail_fast,
        "results": [asdict(result) for result in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    job_list_file_raw = str(args.job_list_file).strip()
    profile_list_raw = str(args.profile_list).strip()
    port_list_raw = str(args.port_list).strip()
    summary_path_raw = str(args.summary_path).strip()
    poll_interval_s = float(args.poll_interval_s)
    fail_fast = bool(args.fail_fast)

    if not job_list_file_raw:
        parser.error(
            f"--job-list-file is required (or set env {DEFAULT_JOB_LIST_ENV})"
        )
    if not profile_list_raw:
        parser.error(
            f"--profile-list is required (or set env {DEFAULT_PROFILE_LIST_ENV})"
        )
    if not port_list_raw:
        parser.error(
            f"--port-list is required (or set env {DEFAULT_PORT_LIST_ENV})"
        )

    try:
        jobs = _parse_job_list_file(Path(job_list_file_raw).expanduser().resolve())
        profile_ids = _parse_csv_int_list(raw=profile_list_raw, label="--profile-list")
        ports = _parse_csv_int_list(raw=port_list_raw, label="--port-list")
        if len(profile_ids) != len(ports):
            raise ValueError(
                "--profile-list and --port-list must have the same number of items "
                f"(got {len(profile_ids)} vs {len(ports)})"
            )
        slots = [
            PortSlot(slot_index=index, profile_id=profile_id, port=port)
            for index, (profile_id, port) in enumerate(zip(profile_ids, ports))
        ]
        summary = _run_jobs(
            jobs=jobs,
            slots=slots,
            poll_interval_s=poll_interval_s,
            fail_fast=fail_fast,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if summary_path_raw:
        summary_path = Path(summary_path_raw).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[summary] wrote {summary_path}", flush=True)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
