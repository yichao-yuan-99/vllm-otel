from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pp_common.service_failure import cutoff_datetime_utc_from_payload
from pp_common.service_failure import ensure_service_failure_payload
from pp_common.service_failure import parse_iso8601_to_utc


DEFAULT_OUTPUT_NAME = "trial-timing-summary.json"


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
            "Extract high-level trial timing summary from a run directory "
            "(con-driver or replay)."
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
            "with replay/summary.json or meta/results.json + meta/run_manifest.json "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/post-processed/global/"
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


def _duration_seconds(start_dt: datetime | None, end_dt: datetime | None) -> float | None:
    if start_dt is None or end_dt is None:
        return None
    return round((end_dt - start_dt).total_seconds(), 6)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _build_duration_stats(trials: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [
        duration
        for duration in (trial.get("duration_s") for trial in trials)
        if isinstance(duration, (int, float))
    ]
    if not values:
        return {"avg": None, "min": None, "max": None}
    return {
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _build_replay_summary(
    run_dir: Path,
    *,
    cutoff_time_utc: datetime | None = None,
) -> dict[str, Any]:
    summary_path = run_dir / "replay" / "summary.json"
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Replay summary must be a JSON object: {summary_path}")

    experiment_started_at = payload.get("started_at")
    experiment_finished_at = payload.get("finished_at")
    experiment_start_dt = _parse_iso8601(experiment_started_at)
    experiment_start_dt_utc = parse_iso8601_to_utc(experiment_started_at)
    experiment_finish_dt = _parse_iso8601(experiment_finished_at)

    worker_results = payload.get("worker_results")
    if not isinstance(worker_results, dict):
        raise ValueError(f"Replay summary is missing object field worker_results: {summary_path}")

    trials: list[dict[str, Any]] = []
    for trial_id, worker_payload in sorted(worker_results.items()):
        if not isinstance(worker_payload, dict):
            continue
        worker_id = worker_payload.get("worker_id")
        resolved_trial_id = worker_id if isinstance(worker_id, str) and worker_id else str(trial_id)
        started_at = worker_payload.get("started_at")
        finished_at = worker_payload.get("finished_at")
        start_dt = _parse_iso8601(started_at)
        finish_dt = _parse_iso8601(finished_at)
        start_dt_utc = parse_iso8601_to_utc(started_at)
        finish_dt_utc = parse_iso8601_to_utc(finished_at)

        if cutoff_time_utc is not None:
            if start_dt_utc is not None and start_dt_utc > cutoff_time_utc:
                continue
            if finish_dt_utc is not None and finish_dt_utc > cutoff_time_utc:
                continue

        duration_s = _duration_seconds(start_dt, finish_dt)
        start_offset_s = _duration_seconds(experiment_start_dt, start_dt)
        end_offset_s = _duration_seconds(experiment_start_dt, finish_dt)

        trials.append(
            {
                "trial_id": resolved_trial_id,
                "status": worker_payload.get("status"),
                "started_at": started_at,
                "finished_at": finished_at,
                "start_offset_s": start_offset_s,
                "end_offset_s": end_offset_s,
                "duration_s": duration_s,
            }
        )

    trials.sort(
        key=lambda trial: (
            trial["start_offset_s"] if isinstance(trial.get("start_offset_s"), (int, float)) else float("inf"),
            trial.get("trial_id") or "",
        )
    )

    cutoff_offset_s = _duration_seconds(experiment_start_dt_utc, cutoff_time_utc)
    if cutoff_offset_s is not None:
        cutoff_offset_s = max(cutoff_offset_s, 0.0)

    total_duration_s = _float_or_none(payload.get("run_duration_s"))
    if total_duration_s is None:
        total_duration_s = _duration_seconds(experiment_start_dt, experiment_finish_dt)
    if cutoff_offset_s is not None:
        if total_duration_s is None:
            total_duration_s = cutoff_offset_s
        else:
            total_duration_s = min(total_duration_s, cutoff_offset_s)

    return {
        "source_type": "replay",
        "source_path": str(summary_path.resolve()),
        "experiment_started_at": experiment_started_at,
        "experiment_finished_at": experiment_finished_at,
        "total_duration_s": total_duration_s,
        "trials": trials,
    }


def _build_con_driver_summary(
    run_dir: Path,
    *,
    cutoff_time_utc: datetime | None = None,
) -> dict[str, Any]:
    results_path = run_dir / "meta" / "results.json"
    manifest_path = run_dir / "meta" / "run_manifest.json"

    results_payload = _load_json(results_path)
    if not isinstance(results_payload, list):
        raise ValueError(f"Con-driver results must be a JSON array: {results_path}")

    manifest_payload = _load_json(manifest_path)
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"Con-driver run_manifest must be a JSON object: {manifest_path}")

    experiment_started_at = manifest_payload.get("started_at")
    experiment_finished_at = manifest_payload.get("finished_at")
    experiment_start_dt = _parse_iso8601(experiment_started_at)
    experiment_start_dt_utc = parse_iso8601_to_utc(experiment_started_at)
    experiment_finish_dt = _parse_iso8601(experiment_finished_at)

    trials: list[dict[str, Any]] = []
    for entry in results_payload:
        if not isinstance(entry, dict):
            continue
        started_at = entry.get("started_at")
        finished_at = entry.get("finished_at")
        start_dt = _parse_iso8601(started_at)
        finish_dt = _parse_iso8601(finished_at)
        start_dt_utc = parse_iso8601_to_utc(started_at)
        finish_dt_utc = parse_iso8601_to_utc(finished_at)

        if cutoff_time_utc is not None:
            if start_dt_utc is not None and start_dt_utc > cutoff_time_utc:
                continue
            if finish_dt_utc is not None and finish_dt_utc > cutoff_time_utc:
                continue

        duration_s = _float_or_none(entry.get("duration_s"))
        if duration_s is None:
            duration_s = _duration_seconds(start_dt, finish_dt)

        start_offset_s = _duration_seconds(experiment_start_dt, start_dt)
        end_offset_s = _duration_seconds(experiment_start_dt, finish_dt)

        trials.append(
            {
                "trial_id": entry.get("trial_id"),
                "status": entry.get("status"),
                "started_at": started_at,
                "finished_at": finished_at,
                "start_offset_s": start_offset_s,
                "end_offset_s": end_offset_s,
                "duration_s": duration_s,
            }
        )

    trials.sort(
        key=lambda trial: (
            trial["start_offset_s"] if isinstance(trial.get("start_offset_s"), (int, float)) else float("inf"),
            trial.get("trial_id") or "",
        )
    )

    cutoff_offset_s = _duration_seconds(experiment_start_dt_utc, cutoff_time_utc)
    if cutoff_offset_s is not None:
        cutoff_offset_s = max(cutoff_offset_s, 0.0)

    total_duration_s = _float_or_none(manifest_payload.get("run_duration_s"))
    if total_duration_s is None:
        total_duration_s = _duration_seconds(experiment_start_dt, experiment_finish_dt)
    if cutoff_offset_s is not None:
        if total_duration_s is None:
            total_duration_s = cutoff_offset_s
        else:
            total_duration_s = min(total_duration_s, cutoff_offset_s)

    return {
        "source_type": "con-driver",
        "source_path": str(results_path.resolve()),
        "experiment_started_at": experiment_started_at,
        "experiment_finished_at": experiment_finished_at,
        "total_duration_s": total_duration_s,
        "trials": trials,
    }


def extract_global_trial_summary_from_run_dir(run_dir: Path) -> dict[str, Any]:
    service_failure_payload = ensure_service_failure_payload(run_dir)
    cutoff_time_utc = cutoff_datetime_utc_from_payload(service_failure_payload)
    replay_summary_path = run_dir / "replay" / "summary.json"
    con_driver_results_path = run_dir / "meta" / "results.json"
    con_driver_manifest_path = run_dir / "meta" / "run_manifest.json"

    if replay_summary_path.is_file():
        base = _build_replay_summary(run_dir, cutoff_time_utc=cutoff_time_utc)
    elif con_driver_results_path.is_file() and con_driver_manifest_path.is_file():
        base = _build_con_driver_summary(run_dir, cutoff_time_utc=cutoff_time_utc)
    else:
        raise ValueError(
            "Unrecognized run layout. Expected either replay/summary.json "
            "or meta/results.json + meta/run_manifest.json."
        )

    trials = base["trials"]
    trial_count = len(trials)
    return {
        "source_run_dir": str(run_dir.resolve()),
        "source_type": base["source_type"],
        "source_path": base["source_path"],
        "experiment_started_at": base["experiment_started_at"],
        "experiment_finished_at": base["experiment_finished_at"],
        "total_duration_s": base["total_duration_s"],
        "service_failure_detected": bool(
            service_failure_payload.get("service_failure_detected", False)
        ),
        "service_failure_cutoff_time_utc": service_failure_payload.get("cutoff_time_utc"),
        "trial_count": trial_count,
        "trail_count": trial_count,
        "trial_duration_stats_s": _build_duration_stats(trials),
        "trials": trials,
        "trails": trials,
    }


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "global" / DEFAULT_OUTPUT_NAME).resolve()


def discover_run_dirs_with_global_sources(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()

    for summary_path in root_dir.rglob("summary.json"):
        if not summary_path.is_file():
            continue
        if summary_path.parent.name != "replay":
            continue
        run_dirs.add(summary_path.parent.parent.resolve())

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
    resolved_output_path = (output_path or _default_output_path_for_run(run_dir)).expanduser().resolve()
    result = extract_global_trial_summary_from_run_dir(run_dir)
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

    run_dirs = discover_run_dirs_with_global_sources(root_dir)
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
