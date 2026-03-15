#!/usr/bin/env python3
"""Generate local-mode single-profile exclude-unranked sweep-qps bundles.

This wrapper reuses the single-profile local generator while enforcing
exclude-unranked defaults for plan selection and output lineage.
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
EXCLUDE_UNRANKED_PLAN_FILE_NAME = "replay-plan.exclude-unranked.json"


def default_output_config_dir() -> Path:
    return (
        REPO_ROOT
        / "experiments"
        / "sweep-qps-local"
        / "exclude-unranked"
        / "generated"
    )


def _load_single_generator_module() -> Any:
    module_path = (
        Path(__file__).resolve().parents[1] / "single" / "generate_replay_configs.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "generate_replay_qps_local_single_configs_shared",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_replay_qps_local_single_configs_shared"] = module
    spec.loader.exec_module(module)
    return module


_SINGLE = _load_single_generator_module()
_ORIGINAL_SINGLE_BUILD_PARSER = _SINGLE.build_parser

# Expose hooks for tests/mocks, then sync into _SINGLE before main().
build_control_plane = _SINGLE.build_control_plane
render_local_mode_sbatch = _SINGLE.render_local_mode_sbatch


def derive_replay_output_base(
    source_run_dir: Path,
    *,
    replay_root_dir: Path | None = None,
    batch_timestamp: str,
) -> Path:
    source_resolved = source_run_dir.expanduser().resolve()
    try:
        source_relative = source_resolved.relative_to(RESULTS_ROOT)
    except ValueError as exc:
        raise ValueError(
            "source_run_dir must be under top-level results/, for example: "
            "results/qwen3-coder-30b/livecodebench/mini-swe-agent/<run-dir>"
        ) from exc

    if len(source_relative.parts) < 2:
        raise ValueError(
            "source_run_dir under results/ must include at least one parent plus run dir"
        )
    lineage_without_run_dir = source_relative.parts[:-1]
    if replay_root_dir is None:
        replay_root = Path("results") / "replay"
    else:
        replay_root = replay_root_dir.expanduser().resolve()
    return (
        replay_root
        / batch_timestamp
        / Path(*lineage_without_run_dir)
        / "sweep-qps-local"
        / "exclude-unranked"
    )


def build_generation_command_raw(argv: list[str] | None) -> str:
    """Return a shell-safe raw command string for this generator invocation."""
    if argv is None:
        original = getattr(sys, "orig_argv", None)
        if isinstance(original, list) and original:
            return shlex.join([str(token) for token in original])
        return shlex.join([str(sys.executable), *[str(token) for token in sys.argv]])

    return shlex.join(
        [
            str(sys.executable),
            str(Path(__file__).resolve()),
            *[str(token) for token in argv],
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _ORIGINAL_SINGLE_BUILD_PARSER()
    parser.description = (
        "Generate local-mode single-profile exclude-unranked sweep-qps "
        "experiment bundles (replay config + local script + rendered sbatch)."
    )

    option_actions = parser._option_string_actions
    default_output_dir = str(default_output_config_dir())

    output_action = option_actions.get("--output-config-dir")
    if output_action is not None:
        output_action.default = default_output_dir
        output_action.help = (
            "Root directory for generated experiment bundles "
            f"(default: {default_output_dir})"
        )

    source_action = option_actions.get("--source-run-dir")
    if source_action is not None:
        source_action.help = (
            "Source run result directory that contains "
            "replay-plan.exclude-unranked.json by default"
        )

    plan_action = option_actions.get("--plan-path")
    if plan_action is not None:
        plan_action.help = (
            "Optional explicit replay plan path "
            "(default: <source-run-dir>/replay-plan.exclude-unranked.json)"
        )

    replay_root_action = option_actions.get("--replay-root-dir")
    if replay_root_action is not None:
        replay_root_action.help = (
            "Optional root directory for replay outputs. "
            "<utc-timestamp>/source-lineage/sweep-qps-local/exclude-unranked/<qps> "
            "is appended under this root."
        )

    exclude_action = option_actions.get("--exclude-unranked-plan")
    if exclude_action is not None:
        exclude_action.help = (
            "Enabled by default in this variant. You normally do not need to set it."
        )

    return parser


def _has_option(arguments: list[str], option: str) -> bool:
    option_prefix = f"{option}="
    return any(arg == option or arg.startswith(option_prefix) for arg in arguments)


def _sync_single_module_globals() -> None:
    _SINGLE.REPO_ROOT = REPO_ROOT
    _SINGLE.RESULTS_ROOT = RESULTS_ROOT
    _SINGLE.DEFAULT_OUTPUT_CONFIG_DIR = default_output_config_dir()
    _SINGLE.DEFAULT_PLAN_FILE_NAME = EXCLUDE_UNRANKED_PLAN_FILE_NAME
    _SINGLE.build_control_plane = build_control_plane
    _SINGLE.render_local_mode_sbatch = render_local_mode_sbatch
    _SINGLE.derive_replay_output_base = derive_replay_output_base
    _SINGLE.build_generation_command_raw = build_generation_command_raw
    _SINGLE.build_parser = build_parser


def path_for_config(path: Path) -> str:
    _sync_single_module_globals()
    return _SINGLE.path_for_config(path)


def main(argv: list[str] | None = None) -> int:
    forwarded_argv = [str(token) for token in (argv if argv is not None else sys.argv[1:])]
    if not _has_option(forwarded_argv, "--exclude-unranked-plan"):
        forwarded_argv.append("--exclude-unranked-plan")
    if not _has_option(forwarded_argv, "--output-config-dir"):
        forwarded_argv.extend(
            [
                "--output-config-dir",
                str(default_output_config_dir()),
            ]
        )

    _sync_single_module_globals()
    return int(_SINGLE.main(forwarded_argv))


if __name__ == "__main__":
    raise SystemExit(main())
