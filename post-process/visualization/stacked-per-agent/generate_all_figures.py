from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
MATERIALIZE_SCRIPT = (
    REPO_ROOT / "figures" / "stacked-per-agent" / "materialize_stacked_per_agent.py"
)
PLOT_SCRIPT = REPO_ROOT / "figures" / "stacked-per-agent" / "plot_stacked_per_agent.py"

DEFAULT_INPUT_NAME = "context-usage-ranges.json"
DEFAULT_STACK_CONTEXT_INPUT_REL_PATH = Path("post-processed/gateway/stack-context")
DEFAULT_OUTPUT_REL_PATH = Path("post-processed/visualization/stacked-per-agent")
DEFAULT_MANIFEST_NAME = "figures-manifest.json"
DEFAULT_FIGURE_STEM = "stacked-per-agent"
DEFAULT_FORMAT = "png"
SUPPORTED_FORMATS = ("png", "pdf", "svg")
DEFAULT_WINDOW_SIZE_S = 120.0
DEFAULT_AGENT_ORDER = "first-active"
DEFAULT_VALUE_MODE = "average"
DEFAULT_LEGEND = "auto"
DEFAULT_LEGEND_MAX_AGENTS = 24
DEFAULT_TITLE = "Per-Agent Stacked Context Usage"


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
            "Materialize and render the stacked-per-agent figure from "
            "post-processed gateway stack-context ranges."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing post-processed/gateway/stack-context/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/gateway/stack-context/context-usage-ranges.json "
            "will be processed."
        ),
    )
    parser.add_argument(
        "--ranges-input",
        default=None,
        help=(
            "Optional context-usage-ranges input path. Default: "
            "<run-dir>/post-processed/gateway/stack-context/context-usage-ranges.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional figure output directory. Default: "
            "<run-dir>/post-processed/visualization/stacked-per-agent/"
        ),
    )
    parser.add_argument(
        "--window-size-s",
        type=float,
        default=DEFAULT_WINDOW_SIZE_S,
        help=f"Bar width in seconds (default: {DEFAULT_WINDOW_SIZE_S:g}).",
    )
    parser.add_argument(
        "--start-s",
        type=float,
        default=0.0,
        help="Optional analysis window start in seconds from run start (default: 0).",
    )
    parser.add_argument(
        "--end-s",
        type=float,
        default=None,
        help=(
            "Optional analysis window end in seconds from run start. "
            "Default: the latest active range end in the input."
        ),
    )
    parser.add_argument(
        "--agent-order",
        choices=("first-active", "agent-key"),
        default=DEFAULT_AGENT_ORDER,
        help=f"Stable ordering for stacked layers (default: {DEFAULT_AGENT_ORDER}).",
    )
    parser.add_argument(
        "--value-mode",
        choices=("average", "integral"),
        default=DEFAULT_VALUE_MODE,
        help=(
            "Which materialized value to render. "
            f"Default: {DEFAULT_VALUE_MODE}."
        ),
    )
    parser.add_argument(
        "--legend",
        choices=("auto", "show", "hide"),
        default=DEFAULT_LEGEND,
        help=f"Legend mode. Default: {DEFAULT_LEGEND}.",
    )
    parser.add_argument(
        "--legend-max-agents",
        type=int,
        default=DEFAULT_LEGEND_MAX_AGENTS,
        help=(
            "Auto-show the legend only when agent_count <= this threshold "
            f"(default: {DEFAULT_LEGEND_MAX_AGENTS})."
        ),
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Figure title.",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        choices=SUPPORTED_FORMATS,
        help=f"Figure format. Default: {DEFAULT_FORMAT}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for raster output (default: 220).",
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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    return None


def _default_ranges_input_path_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_STACK_CONTEXT_INPUT_REL_PATH / DEFAULT_INPUT_NAME).resolve()


def _default_output_dir_for_run(run_dir: Path) -> Path:
    return (run_dir / DEFAULT_OUTPUT_REL_PATH).resolve()


def _run_dir_from_ranges_path(ranges_path: Path) -> Path | None:
    if ranges_path.name != DEFAULT_INPUT_NAME:
        return None
    stack_context_dir = ranges_path.parent
    if stack_context_dir.name != "stack-context":
        return None
    gateway_dir = stack_context_dir.parent
    if gateway_dir.name != "gateway":
        return None
    post_processed_dir = gateway_dir.parent
    if post_processed_dir.name != "post-processed":
        return None
    return post_processed_dir.parent.resolve()


def discover_run_dirs_with_stacked_per_agent_input(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for ranges_path in root_dir.rglob(DEFAULT_INPUT_NAME):
        if not ranges_path.is_file():
            continue
        run_dir = _run_dir_from_ranges_path(ranges_path)
        if run_dir is None:
            continue
        run_dirs.add(run_dir)
    return sorted(run_dirs)


def _format_label(value: float | None, *, default: str) -> str:
    if value is None:
        return default
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _default_materialized_output_path(
    output_dir: Path,
    *,
    window_size_s: float,
    analysis_start_s: float,
    analysis_end_s: float | None,
) -> Path:
    window_label = _format_label(window_size_s, default="120")
    start_label = _format_label(analysis_start_s, default="0")
    end_label = _format_label(analysis_end_s, default="full")
    return (
        output_dir
        / (
            f"{DEFAULT_FIGURE_STEM}.window-{window_label}s."
            f"start-{start_label}.end-{end_label}.json"
        )
    ).resolve()


def _default_figure_output_path(
    materialized_output_path: Path,
    *,
    image_format: str,
    value_mode: str,
) -> Path:
    value_suffix = "" if value_mode == "average" else f".{value_mode}"
    return (
        materialized_output_path.parent
        / f"{materialized_output_path.stem}{value_suffix}.{image_format}"
    ).resolve()


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _run_command(command: list[str]) -> int:
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _build_materialize_command(
    *,
    ranges_input_path: Path,
    materialized_output_path: Path,
    window_size_s: float,
    analysis_start_s: float,
    analysis_end_s: float | None,
    agent_order: str,
) -> list[str]:
    command = [
        sys.executable,
        str(MATERIALIZE_SCRIPT),
        "--input",
        str(ranges_input_path),
        "--window-size-s",
        str(window_size_s),
        "--start-s",
        str(analysis_start_s),
        "--agent-order",
        agent_order,
        "--output",
        str(materialized_output_path),
    ]
    if analysis_end_s is not None:
        command.extend(["--end-s", str(analysis_end_s)])
    return command


def _build_plot_command(
    *,
    materialized_output_path: Path,
    figure_output_path: Path,
    value_mode: str,
    legend: str,
    legend_max_agents: int,
    title: str,
    dpi: int,
) -> list[str]:
    return [
        sys.executable,
        str(PLOT_SCRIPT),
        "--input",
        str(materialized_output_path),
        "--output",
        str(figure_output_path),
        "--value-mode",
        value_mode,
        "--legend",
        legend,
        "--legend-max-agents",
        str(legend_max_agents),
        "--title",
        title,
        "--dpi",
        str(dpi),
    ]


def generate_figures_for_run_dir(
    run_dir: Path,
    *,
    ranges_input_path: Path | None = None,
    output_dir: Path | None = None,
    window_size_s: float = DEFAULT_WINDOW_SIZE_S,
    analysis_start_s: float = 0.0,
    analysis_end_s: float | None = None,
    agent_order: str = DEFAULT_AGENT_ORDER,
    value_mode: str = DEFAULT_VALUE_MODE,
    legend: str = DEFAULT_LEGEND,
    legend_max_agents: int = DEFAULT_LEGEND_MAX_AGENTS,
    title: str = DEFAULT_TITLE,
    image_format: str = DEFAULT_FORMAT,
    dpi: int = 220,
) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_ranges_input_path = (
        ranges_input_path or _default_ranges_input_path_for_run(resolved_run_dir)
    ).expanduser().resolve()
    resolved_output_dir = (
        output_dir or _default_output_dir_for_run(resolved_run_dir)
    ).expanduser().resolve()

    if not resolved_ranges_input_path.is_file():
        raise ValueError(f"Missing context-usage ranges file: {resolved_ranges_input_path}")
    if window_size_s <= 0.0:
        raise ValueError(f"window_size_s must be positive: {window_size_s}")
    if analysis_start_s < 0.0:
        raise ValueError(f"analysis_start_s must be non-negative: {analysis_start_s}")
    if analysis_end_s is not None and analysis_end_s <= analysis_start_s:
        raise ValueError(
            "analysis_end_s must be greater than analysis_start_s: "
            f"start={analysis_start_s}, end={analysis_end_s}"
        )
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer: {dpi}")
    if legend_max_agents <= 0:
        raise ValueError(
            f"legend_max_agents must be a positive integer: {legend_max_agents}"
        )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    materialized_output_path = _default_materialized_output_path(
        resolved_output_dir,
        window_size_s=window_size_s,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
    )
    figure_output_path = _default_figure_output_path(
        materialized_output_path,
        image_format=image_format,
        value_mode=value_mode,
    )

    materialize_command = _build_materialize_command(
        ranges_input_path=resolved_ranges_input_path,
        materialized_output_path=materialized_output_path,
        window_size_s=window_size_s,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
        agent_order=agent_order,
    )
    materialize_return_code = _run_command(materialize_command)
    if materialize_return_code != 0:
        raise RuntimeError(
            "stacked-per-agent materialization failed "
            f"(rc={materialize_return_code}): {_shell_join(materialize_command)}"
        )
    if not materialized_output_path.is_file():
        raise RuntimeError(
            "stacked-per-agent materialization did not produce the expected JSON: "
            f"{materialized_output_path}"
        )

    plot_command = _build_plot_command(
        materialized_output_path=materialized_output_path,
        figure_output_path=figure_output_path,
        value_mode=value_mode,
        legend=legend,
        legend_max_agents=legend_max_agents,
        title=title,
        dpi=dpi,
    )
    plot_return_code = _run_command(plot_command)
    if plot_return_code != 0:
        raise RuntimeError(
            "stacked-per-agent plotting failed "
            f"(rc={plot_return_code}): {_shell_join(plot_command)}"
        )
    if not figure_output_path.is_file():
        raise RuntimeError(
            "stacked-per-agent plotting did not produce the expected figure: "
            f"{figure_output_path}"
        )

    materialized_payload = _load_json(materialized_output_path)
    if not isinstance(materialized_payload, dict):
        raise ValueError(
            f"Materialized stacked-per-agent JSON must be an object: {materialized_output_path}"
        )

    manifest = {
        "source_run_dir": str(resolved_run_dir),
        "source_ranges_path": str(resolved_ranges_input_path),
        "output_dir": str(resolved_output_dir),
        "materialized_file_name": materialized_output_path.name,
        "materialized_data_path": str(materialized_output_path),
        "figure_generated": True,
        "figure_file_name": figure_output_path.name,
        "figure_path": str(figure_output_path),
        "image_format": image_format,
        "dpi": dpi,
        "window_size_s": _float_or_none(materialized_payload.get("window_size_s")),
        "analysis_window_start_s": _float_or_none(
            materialized_payload.get("analysis_window_start_s")
        ),
        "analysis_window_end_s": _float_or_none(
            materialized_payload.get("analysis_window_end_s")
        ),
        "analysis_window_duration_s": _float_or_none(
            materialized_payload.get("analysis_window_duration_s")
        ),
        "agent_order": materialized_payload.get("agent_order"),
        "agent_count": _int_or_none(materialized_payload.get("agent_count")),
        "window_count": _int_or_none(materialized_payload.get("window_count")),
        "metric": materialized_payload.get("metric"),
        "phase": materialized_payload.get("phase"),
        "value_mode": value_mode,
        "legend": legend,
        "legend_max_agents": legend_max_agents,
        "title": title,
    }
    manifest_path = resolved_output_dir / DEFAULT_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _generate_run_dir_worker(
    task: dict[str, Any],
) -> tuple[str, str | None, str | None]:
    run_dir = Path(str(task["run_dir"])).expanduser().resolve()
    try:
        output_path = generate_figures_for_run_dir(
            run_dir,
            window_size_s=float(task["window_size_s"]),
            analysis_start_s=float(task["analysis_start_s"]),
            analysis_end_s=(
                float(task["analysis_end_s"])
                if task["analysis_end_s"] is not None
                else None
            ),
            agent_order=str(task["agent_order"]),
            value_mode=str(task["value_mode"]),
            legend=str(task["legend"]),
            legend_max_agents=int(task["legend_max_agents"]),
            title=str(task["title"]),
            image_format=str(task["image_format"]),
            dpi=int(task["dpi"]),
        )
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(
    run_dirs: list[Path],
    *,
    window_size_s: float,
    analysis_start_s: float,
    analysis_end_s: float | None,
    agent_order: str,
    value_mode: str,
    legend: str,
    legend_max_agents: int,
    title: str,
    image_format: str,
    dpi: int,
) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = generate_figures_for_run_dir(
                run_dir,
                window_size_s=window_size_s,
                analysis_start_s=analysis_start_s,
                analysis_end_s=analysis_end_s,
                agent_order=agent_order,
                value_mode=value_mode,
                legend=legend,
                legend_max_agents=legend_max_agents,
                title=title,
                image_format=image_format,
                dpi=dpi,
            )
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(
    run_dirs: list[Path],
    *,
    max_procs: int,
    window_size_s: float,
    analysis_start_s: float,
    analysis_end_s: float | None,
    agent_order: str,
    value_mode: str,
    legend: str,
    legend_max_agents: int,
    title: str,
    image_format: str,
    dpi: int,
) -> int:
    failure_count = 0
    tasks = [
        {
            "run_dir": str(run_dir),
            "window_size_s": window_size_s,
            "analysis_start_s": analysis_start_s,
            "analysis_end_s": analysis_end_s,
            "agent_order": agent_order,
            "value_mode": value_mode,
            "legend": legend,
            "legend_max_agents": legend_max_agents,
            "title": title,
            "image_format": image_format,
            "dpi": dpi,
        }
        for run_dir in run_dirs
    ]
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_path_text, error_text in executor.map(
            _generate_run_dir_worker,
            tasks,
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
    ranges_input_path = (
        Path(args.ranges_input).expanduser().resolve()
        if args.ranges_input
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_path = generate_figures_for_run_dir(
        run_dir,
        ranges_input_path=ranges_input_path,
        output_dir=output_dir,
        window_size_s=args.window_size_s,
        analysis_start_s=args.start_s,
        analysis_end_s=args.end_s,
        agent_order=args.agent_order,
        value_mode=args.value_mode,
        legend=args.legend,
        legend_max_agents=args.legend_max_agents,
        title=args.title,
        image_format=args.format,
        dpi=args.dpi,
    )
    print(str(output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.ranges_input:
        raise ValueError("--ranges-input can only be used with --run-dir")
    if args.output_dir:
        raise ValueError("--output-dir can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")
    if args.window_size_s <= 0.0:
        raise ValueError(f"--window-size-s must be positive: {args.window_size_s}")
    if args.start_s < 0.0:
        raise ValueError(f"--start-s must be non-negative: {args.start_s}")
    if args.end_s is not None and args.end_s <= args.start_s:
        raise ValueError(
            f"--end-s must be greater than --start-s: start={args.start_s}, end={args.end_s}"
        )
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be a positive integer: {args.dpi}")
    if args.legend_max_agents <= 0:
        raise ValueError(
            f"--legend-max-agents must be a positive integer: {args.legend_max_agents}"
        )
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_stacked_per_agent_input(root_dir)
    print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
    if not run_dirs:
        return 0
    if args.dry_run:
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    worker_count = min(args.max_procs, len(run_dirs))
    print(f"Running visualization with {worker_count} worker process(es)")

    if worker_count <= 1:
        failure_count = _run_root_dir_sequential(
            run_dirs,
            window_size_s=args.window_size_s,
            analysis_start_s=args.start_s,
            analysis_end_s=args.end_s,
            agent_order=args.agent_order,
            value_mode=args.value_mode,
            legend=args.legend,
            legend_max_agents=args.legend_max_agents,
            title=args.title,
            image_format=args.format,
            dpi=args.dpi,
        )
    else:
        try:
            failure_count = _run_root_dir_parallel(
                run_dirs,
                max_procs=worker_count,
                window_size_s=args.window_size_s,
                analysis_start_s=args.start_s,
                analysis_end_s=args.end_s,
                agent_order=args.agent_order,
                value_mode=args.value_mode,
                legend=args.legend,
                legend_max_agents=args.legend_max_agents,
                title=args.title,
                image_format=args.format,
                dpi=args.dpi,
            )
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(
                run_dirs,
                window_size_s=args.window_size_s,
                analysis_start_s=args.start_s,
                analysis_end_s=args.end_s,
                agent_order=args.agent_order,
                value_mode=args.value_mode,
                legend=args.legend,
                legend_max_agents=args.legend_max_agents,
                title=args.title,
                image_format=args.format,
                dpi=args.dpi,
            )

    if failure_count:
        print(
            f"Completed with {failure_count} failure(s) out of {len(run_dirs)} run directories.",
            file=sys.stderr,
        )
        return 1
    print(f"Completed visualization for {len(run_dirs)} run directories.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_dir:
        return _main_run_dir(args)
    return _main_root_dir(args)


if __name__ == "__main__":
    raise SystemExit(main())
