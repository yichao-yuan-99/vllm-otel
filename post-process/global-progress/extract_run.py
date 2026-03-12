from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_OUTPUT_NAME = "replay-progress-summary.json"
DEFAULT_MILESTONE_STEP = 50


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
            "Extract milestone finish times since run start for replay/trial completion "
            "(first 50, 100, ... until all replays)."
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
            "Optional output path. Default: <run-dir>/post-processed/global-progress/"
            f"{DEFAULT_OUTPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--milestone-step",
        type=int,
        default=DEFAULT_MILESTONE_STEP,
        help=f"Milestone step size (default: {DEFAULT_MILESTONE_STEP}).",
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


def _milestone_counts(total_count: int, *, step: int) -> list[int]:
    if total_count <= 0:
        return []
    counts = list(range(step, total_count + 1, step))
    if not counts or counts[-1] != total_count:
        counts.append(total_count)
    return counts


def _build_milestones(
    *,
    end_offsets_s: list[float],
    total_replay_count: int,
    step: int,
) -> list[dict[str, Any]]:
    sorted_offsets = sorted(end_offsets_s)
    milestones: list[dict[str, Any]] = []
    for replay_count in _milestone_counts(total_replay_count, step=step):
        finish_time_s = None
        if replay_count <= len(sorted_offsets):
            finish_time_s = sorted_offsets[replay_count - 1]
        milestones.append(
            {
                "replay_count": replay_count,
                "finish_time_s": finish_time_s,
            }
        )
    return milestones


def _extract_replay_offsets_from_replay_run(run_dir: Path) -> tuple[str, Any, int, list[float]]:
    summary_path = run_dir / "replay" / "summary.json"
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Replay summary must be a JSON object: {summary_path}")

    experiment_started_at = payload.get("started_at")
    experiment_start_dt = _parse_iso8601(experiment_started_at)
    if experiment_start_dt is None:
        raise ValueError(f"Invalid or missing started_at in replay summary: {summary_path}")

    worker_results = payload.get("worker_results")
    if not isinstance(worker_results, dict):
        raise ValueError(f"Replay summary is missing object field worker_results: {summary_path}")

    replay_count = 0
    end_offsets_s: list[float] = []
    for worker_payload in worker_results.values():
        if not isinstance(worker_payload, dict):
            continue
        replay_count += 1
        finish_dt = _parse_iso8601(worker_payload.get("finished_at"))
        end_offset_s = _duration_seconds(experiment_start_dt, finish_dt)
        if end_offset_s is not None:
            end_offsets_s.append(end_offset_s)
    return ("replay", experiment_started_at, replay_count, end_offsets_s)


def _extract_replay_offsets_from_con_driver_run(run_dir: Path) -> tuple[str, Any, int, list[float]]:
    manifest_path = run_dir / "meta" / "run_manifest.json"
    results_path = run_dir / "meta" / "results.json"

    manifest_payload = _load_json(manifest_path)
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"Con-driver run_manifest must be a JSON object: {manifest_path}")
    results_payload = _load_json(results_path)
    if not isinstance(results_payload, list):
        raise ValueError(f"Con-driver results must be a JSON array: {results_path}")

    experiment_started_at = manifest_payload.get("started_at")
    experiment_start_dt = _parse_iso8601(experiment_started_at)
    if experiment_start_dt is None:
        raise ValueError(f"Invalid or missing started_at in run_manifest: {manifest_path}")

    replay_count = 0
    end_offsets_s: list[float] = []
    for entry in results_payload:
        if not isinstance(entry, dict):
            continue
        replay_count += 1
        finish_dt = _parse_iso8601(entry.get("finished_at"))
        end_offset_s = _duration_seconds(experiment_start_dt, finish_dt)
        if end_offset_s is not None:
            end_offsets_s.append(end_offset_s)
    return ("con-driver", experiment_started_at, replay_count, end_offsets_s)


def extract_global_progress_from_run_dir(
    run_dir: Path,
    *,
    milestone_step: int = DEFAULT_MILESTONE_STEP,
) -> dict[str, Any]:
    if milestone_step <= 0:
        raise ValueError(f"milestone_step must be a positive integer: {milestone_step}")

    replay_summary_path = run_dir / "replay" / "summary.json"
    con_driver_results_path = run_dir / "meta" / "results.json"
    con_driver_manifest_path = run_dir / "meta" / "run_manifest.json"

    if replay_summary_path.is_file():
        source_type, experiment_started_at, replay_count, end_offsets_s = _extract_replay_offsets_from_replay_run(
            run_dir
        )
    elif con_driver_results_path.is_file() and con_driver_manifest_path.is_file():
        source_type, experiment_started_at, replay_count, end_offsets_s = _extract_replay_offsets_from_con_driver_run(
            run_dir
        )
    else:
        raise ValueError(
            "Unrecognized run layout. Expected either replay/summary.json "
            "or meta/results.json + meta/run_manifest.json."
        )

    milestones = _build_milestones(
        end_offsets_s=end_offsets_s,
        total_replay_count=replay_count,
        step=milestone_step,
    )
    return {
        "source_run_dir": str(run_dir.resolve()),
        "source_type": source_type,
        "experiment_started_at": experiment_started_at,
        "replay_count": replay_count,
        "finished_replay_count": len(end_offsets_s),
        "milestone_step": milestone_step,
        "milestones": milestones,
    }


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "global-progress" / DEFAULT_OUTPUT_NAME).resolve()


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


def extract_run_dir(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    milestone_step: int = DEFAULT_MILESTONE_STEP,
) -> Path:
    resolved_output_path = (output_path or _default_output_path_for_run(run_dir)).expanduser().resolve()
    result = extract_global_progress_from_run_dir(run_dir, milestone_step=milestone_step)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def _extract_run_dir_worker(job: tuple[str, int]) -> tuple[str, str | None, str | None]:
    run_dir_text, milestone_step = job
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = extract_run_dir(run_dir, milestone_step=milestone_step)
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(run_dirs: list[Path], *, milestone_step: int) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = extract_run_dir(run_dir, milestone_step=milestone_step)
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int, milestone_step: int) -> int:
    failure_count = 0
    jobs = [(str(run_dir), milestone_step) for run_dir in run_dirs]
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_path_text, error_text in executor.map(
            _extract_run_dir_worker,
            jobs,
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
    if args.milestone_step <= 0:
        raise ValueError(f"--milestone-step must be a positive integer: {args.milestone_step}")
    run_path = Path(args.run_dir).expanduser().resolve()
    if not run_path.exists():
        raise ValueError(f"Run directory not found: {run_path}")
    if not run_path.is_dir():
        raise ValueError(
            f"--run-dir must point to a directory, got file: {run_path}. "
            f"If you want to process many runs, use --root-dir {run_path.parent}"
        )
    run_dir = run_path
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    resolved_output_path = extract_run_dir(
        run_dir,
        output_path=output_path,
        milestone_step=args.milestone_step,
    )
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.milestone_step <= 0:
        raise ValueError(f"--milestone-step must be a positive integer: {args.milestone_step}")
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
        failure_count = _run_root_dir_sequential(run_dirs, milestone_step=args.milestone_step)
    else:
        try:
            failure_count = _run_root_dir_parallel(
                run_dirs,
                max_procs=worker_count,
                milestone_step=args.milestone_step,
            )
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(run_dirs, milestone_step=args.milestone_step)

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
