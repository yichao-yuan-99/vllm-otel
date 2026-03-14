from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


THIS_DIR = Path(__file__).resolve().parent


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
            "Run all post-process scripts in dependency order for one run or for all "
            "runs under a root directory."
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
        help="Root directory containing many run result directories.",
    )
    parser.add_argument(
        "--max-procs",
        type=int,
        default=_default_max_procs(),
        help=(
            "Number of worker processes for root-dir steps. "
            "Default: MAX_PROCS env var, else CPU count."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered run directories for root-dir steps and exit.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization/vllm-metrics/generate_all_figures.py.",
    )
    parser.add_argument(
        "--skip-aggregate-csv",
        action="store_true",
        help="Skip global/aggregate_runs_csv.py in root-dir mode.",
    )
    parser.add_argument(
        "--aggregate-output",
        default=None,
        help=(
            "Optional output path for global/aggregate_runs_csv.py (root-dir mode only)."
        ),
    )
    return parser.parse_args(argv)


def _script_path(relative_path: str) -> Path:
    return (THIS_DIR / relative_path).resolve()


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _run_command(command: list[str]) -> int:
    print(f"[run] {_shell_join(command)}")
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _is_single_run_layout(run_dir: Path) -> bool:
    replay_summary = run_dir / "replay" / "summary.json"
    con_driver_results = run_dir / "meta" / "results.json"
    con_driver_manifest = run_dir / "meta" / "run_manifest.json"
    return replay_summary.is_file() or (
        con_driver_results.is_file() and con_driver_manifest.is_file()
    )


def _has_nested_run_layout(root_dir: Path) -> bool:
    for summary_path in root_dir.rglob("summary.json"):
        if summary_path.parent.name == "replay":
            return True
    for results_path in root_dir.rglob("results.json"):
        if results_path.parent.name != "meta":
            continue
        manifest_path = results_path.parent / "run_manifest.json"
        if manifest_path.is_file():
            return True
    return False


def _build_step_commands_for_run_dir(
    run_dir: Path,
    *,
    skip_visualization: bool,
) -> list[list[str]]:
    steps = [
        "global/extract_run.py",
        "global-progress/extract_run.py",
        "job-throughput/extract_run.py",
        "gateway/llm-requests/extract_run.py",
        "gateway/usage/extract_run.py",
        "split/duration/extract_run.py",
        "vllm-metrics/extract_run.py",
        "vllm-metrics/summarize_timeseries.py",
    ]
    if not skip_visualization:
        steps.extend(
            [
                "visualization/job-throughput/generate_all_figures.py",
                "visualization/vllm-metrics/generate_all_figures.py",
            ]
        )

    commands: list[list[str]] = []
    for script_rel_path in steps:
        commands.append(
            [
                sys.executable,
                str(_script_path(script_rel_path)),
                "--run-dir",
                str(run_dir),
            ]
        )
    return commands


def _build_step_commands_for_root_dir(
    root_dir: Path,
    *,
    max_procs: int,
    dry_run: bool,
    skip_visualization: bool,
    skip_aggregate_csv: bool,
    aggregate_output: Path | None,
) -> list[list[str]]:
    steps = [
        "global/extract_run.py",
        "global-progress/extract_run.py",
        "job-throughput/extract_run.py",
        "gateway/llm-requests/extract_run.py",
        "gateway/usage/extract_run.py",
        "split/duration/extract_run.py",
        "vllm-metrics/extract_run.py",
        "vllm-metrics/summarize_timeseries.py",
    ]
    if not skip_visualization:
        steps.extend(
            [
                "visualization/job-throughput/generate_all_figures.py",
                "visualization/vllm-metrics/generate_all_figures.py",
            ]
        )

    commands: list[list[str]] = []
    for script_rel_path in steps:
        command = [
            sys.executable,
            str(_script_path(script_rel_path)),
            "--root-dir",
            str(root_dir),
            "--max-procs",
            str(max_procs),
        ]
        if dry_run:
            command.append("--dry-run")
        commands.append(command)

    if not dry_run and not skip_aggregate_csv:
        aggregate_command = [
            sys.executable,
            str(_script_path("global/aggregate_runs_csv.py")),
            "--root-dir",
            str(root_dir),
        ]
        if aggregate_output is not None:
            aggregate_command.extend(["--output", str(aggregate_output)])
        commands.append(aggregate_command)
    return commands


def _main_run_dir(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory not found: {run_dir}")

    if _is_single_run_layout(run_dir):
        if args.aggregate_output:
            raise ValueError("--aggregate-output can only be used with --root-dir")
        commands = _build_step_commands_for_run_dir(
            run_dir,
            skip_visualization=bool(args.skip_visualization),
        )
    elif _has_nested_run_layout(run_dir):
        aggregate_output = (
            Path(args.aggregate_output).expanduser().resolve()
            if args.aggregate_output
            else None
        )
        print(
            "[warn] --run-dir does not match a direct run layout; "
            "detected nested runs. Running root-dir pipeline instead."
        )
        commands = _build_step_commands_for_root_dir(
            run_dir,
            max_procs=args.max_procs,
            dry_run=False,
            skip_visualization=bool(args.skip_visualization),
            skip_aggregate_csv=bool(args.skip_aggregate_csv),
            aggregate_output=aggregate_output,
        )
    else:
        if args.aggregate_output:
            raise ValueError("--aggregate-output can only be used with --root-dir")
        commands = _build_step_commands_for_run_dir(
            run_dir,
            skip_visualization=bool(args.skip_visualization),
        )

    for command in commands:
        return_code = _run_command(command)
        if return_code != 0:
            return return_code

    print("[done] post-process pipeline completed")
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    aggregate_output = (
        Path(args.aggregate_output).expanduser().resolve()
        if args.aggregate_output
        else None
    )
    commands = _build_step_commands_for_root_dir(
        root_dir,
        max_procs=args.max_procs,
        dry_run=bool(args.dry_run),
        skip_visualization=bool(args.skip_visualization),
        skip_aggregate_csv=bool(args.skip_aggregate_csv),
        aggregate_output=aggregate_output,
    )
    for command in commands:
        return_code = _run_command(command)
        if return_code != 0:
            return return_code

    print("[done] post-process pipeline completed")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_dir:
        return _main_run_dir(args)
    return _main_root_dir(args)


if __name__ == "__main__":
    raise SystemExit(main())
