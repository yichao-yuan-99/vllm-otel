#!/usr/bin/env python3
"""Set AMD GPU core clock frequency by bridging to the system shell script."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

from deamon.set_amd_gpu_clockfreq import (
    DEFAULT_SCRIPT_PATH,
    DEFAULT_SUDO_PATH,
    build_command,
    format_missing_executable_error,
)


def build_set_commands(
    *,
    gpu_index: int,
    min_mhz: int | None,
    max_mhz: int | None,
    script_path: str,
    use_sudo: bool = True,
    sudo_path: str = DEFAULT_SUDO_PATH,
) -> list[list[str]]:
    """Build the shell commands needed to update sclk limits."""
    commands: list[list[str]] = []
    if max_mhz is not None:
        commands.append(
            build_command(
                clock="sclk",
                limit="max",
                value=max_mhz,
                gpu_index=gpu_index,
                script_path=script_path,
                use_sudo=use_sudo,
                sudo_path=sudo_path,
            )
        )
    if min_mhz is not None:
        commands.append(
            build_command(
                clock="sclk",
                limit="min",
                value=min_mhz,
                gpu_index=gpu_index,
                script_path=script_path,
                use_sudo=use_sudo,
                sudo_path=sudo_path,
            )
        )
    return commands


def main(argv: Sequence[str] | None = None) -> int:
    """Set AMD GPU core frequency."""
    parser = argparse.ArgumentParser(
        prog="amd-set-gpu-core-freq",
        description="Set AMD GPU core clock frequency (sclk).",
    )
    parser.add_argument(
        "--gpu-index",
        "-g",
        type=int,
        required=True,
        help="GPU index to configure.",
    )
    parser.add_argument(
        "--min-mhz",
        "-n",
        type=int,
        help="Minimum core clock in MHz.",
    )
    parser.add_argument(
        "--max-mhz",
        "-x",
        type=int,
        help="Maximum core clock in MHz.",
    )
    parser.add_argument(
        "--script-path",
        default=DEFAULT_SCRIPT_PATH,
        help=f"Path to the bridge script (default: {DEFAULT_SCRIPT_PATH}).",
    )
    parser.add_argument(
        "--sudo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the bridge script through sudo (default: enabled).",
    )
    parser.add_argument(
        "--sudo-path",
        default=DEFAULT_SUDO_PATH,
        help=f"Executable used when --sudo is enabled (default: {DEFAULT_SUDO_PATH}).",
    )
    args = parser.parse_args(argv)

    if args.min_mhz is None and args.max_mhz is None:
        print("Error: at least one of --min-mhz or --max-mhz is required", file=sys.stderr)
        return 1
    if (
        args.min_mhz is not None
        and args.max_mhz is not None
        and args.min_mhz > args.max_mhz
    ):
        print("Error: --min-mhz must be less than or equal to --max-mhz", file=sys.stderr)
        return 1

    try:
        for command in build_set_commands(
            gpu_index=args.gpu_index,
            min_mhz=args.min_mhz,
            max_mhz=args.max_mhz,
            script_path=args.script_path,
            use_sudo=args.sudo,
            sudo_path=args.sudo_path,
        ):
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                return completed.returncode
    except FileNotFoundError as exc:
        print(
            format_missing_executable_error(exc, script_path=args.script_path),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    changes: list[str] = []
    if args.min_mhz is not None:
        changes.append(f"min={args.min_mhz} MHz")
    if args.max_mhz is not None:
        changes.append(f"max={args.max_mhz} MHz")
    print(f"GPU {args.gpu_index}: Set core clock {' '.join(changes)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
