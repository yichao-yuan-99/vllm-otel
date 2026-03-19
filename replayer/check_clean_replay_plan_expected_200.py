#!/usr/bin/env python3
"""Check that clean replay plans only contain expected_status_code == 200 requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _iter_candidate_clean_plan_paths(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(run_dir.rglob("replay-plan.clean*.json")):
        if not path.is_file():
            continue
        name = path.name
        if ".removal-stats" in name or ".removal-details" in name:
            continue
        candidates.append(path.resolve())
    return candidates


def _load_plan_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != "replay-plan.v1":
        return None
    workers = payload.get("workers")
    if not isinstance(workers, list):
        return None
    return payload


def _find_plan_violations(plan_path: Path, plan_payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    workers = plan_payload.get("workers")
    assert isinstance(workers, list)

    total_requests = 0
    violations: list[dict[str, Any]] = []
    for worker_index, worker in enumerate(workers):
        if not isinstance(worker, dict):
            violations.append(
                {
                    "plan": str(plan_path),
                    "worker_index": worker_index,
                    "error": "worker entry is not an object",
                }
            )
            continue
        requests = worker.get("requests")
        if not isinstance(requests, list):
            violations.append(
                {
                    "plan": str(plan_path),
                    "worker_index": worker_index,
                    "worker_id": worker.get("worker_id"),
                    "error": "worker.requests is not a list",
                }
            )
            continue
        for request_index, request in enumerate(requests):
            total_requests += 1
            if not isinstance(request, dict):
                violations.append(
                    {
                        "plan": str(plan_path),
                        "worker_index": worker_index,
                        "worker_id": worker.get("worker_id"),
                        "request_index": request_index,
                        "error": "request entry is not an object",
                    }
                )
                continue
            expected_status_code = request.get("expected_status_code")
            replay_mode = request.get("replay_mode")
            if expected_status_code != 200 or replay_mode == "client_disconnect_after_duration":
                violations.append(
                    {
                        "plan": str(plan_path),
                        "worker_index": worker_index,
                        "worker_id": worker.get("worker_id"),
                        "request_index": request_index,
                        "request_id": request.get("request_id"),
                        "expected_status_code": expected_status_code,
                        "replay_mode": replay_mode,
                        "error": "request is not expected-status-200 deterministic replay",
                    }
                )
    return total_requests, violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_clean_replay_plan_expected_200.py",
        description=(
            "Recursively scan replay-plan.clean*.json under --run-dir and verify "
            "all planned requests have expected_status_code == 200."
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory to recursively scan for clean replay plans.",
    )
    parser.add_argument(
        "--allow-no-clean-plans",
        action="store_true",
        help="Exit 0 when no clean replay plans are found (default: fail).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"error: invalid --run-dir: {run_dir}", file=sys.stderr)
        return 2

    candidate_paths = _iter_candidate_clean_plan_paths(run_dir)
    if not candidate_paths:
        summary = {
            "status": "no_clean_plans_found",
            "run_dir": str(run_dir),
            "checked_plan_count": 0,
            "checked_request_count": 0,
            "violation_count": 0,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0 if args.allow_no_clean_plans else 2

    checked_plans = 0
    checked_requests = 0
    violations: list[dict[str, Any]] = []
    unreadable_candidates: list[str] = []

    for candidate in candidate_paths:
        payload = _load_plan_payload(candidate)
        if payload is None:
            unreadable_candidates.append(str(candidate))
            continue
        checked_plans += 1
        plan_request_count, plan_violations = _find_plan_violations(candidate, payload)
        checked_requests += plan_request_count
        violations.extend(plan_violations)

    summary = {
        "status": "ok" if not violations else "violations_found",
        "run_dir": str(run_dir),
        "checked_plan_count": checked_plans,
        "checked_request_count": checked_requests,
        "candidate_path_count": len(candidate_paths),
        "unreadable_candidate_count": len(unreadable_candidates),
        "unreadable_candidates": unreadable_candidates,
        "violation_count": len(violations),
        "violations": violations,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
