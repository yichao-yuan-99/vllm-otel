#!/usr/bin/env python3
"""Thin bridge to /usr/local/bin/set_gpu_clockfreq.sh."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence


DEFAULT_SCRIPT_PATH = "/usr/local/bin/set_gpu_clockfreq.sh"
DEFAULT_SUDO_PATH = "sudo"


def format_missing_executable_error(
    exc: FileNotFoundError,
    *,
    script_path: str,
) -> str:
    """Render a useful error when the bridge command cannot be launched."""
    missing = exc.filename
    if missing == script_path:
        return f"Error: script not found: {script_path}"
    if missing:
        return f"Error: executable not found: {missing}"
    return f"Error: failed to launch command for script: {script_path}"


def build_command(
    *,
    clock: str,
    limit: str,
    value: int,
    gpu_index: int | None,
    script_path: str,
    use_sudo: bool = True,
    sudo_path: str = DEFAULT_SUDO_PATH,
) -> list[str]:
    """Build the shell-script invocation."""
    command: list[str] = []
    if use_sudo:
        command.append(sudo_path)
    command.extend(
        [
        script_path,
        "-clock",
        clock,
        "-limit",
        limit,
        "-value",
        str(value),
        ]
    )
    if gpu_index is not None:
        command.extend(["-gpu", str(gpu_index)])
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Run the external GPU clock script with repo-friendly flags."""
    parser = argparse.ArgumentParser(
        prog="set-amd-gpu-clockfreq",
        description="Bridge to /usr/local/bin/set_gpu_clockfreq.sh.",
    )
    parser.add_argument(
        "--clock",
        choices=("sclk", "mclk"),
        required=True,
        help="Clock type to modify.",
    )
    parser.add_argument(
        "--limit",
        choices=("min", "max"),
        required=True,
        help="Which limit to modify.",
    )
    parser.add_argument(
        "--value",
        type=int,
        required=True,
        help="Desired frequency in MHz.",
    )
    parser.add_argument(
        "--gpu-index",
        "-g",
        type=int,
        help="Optional GPU index. If omitted, the script targets all GPUs.",
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

    command = build_command(
        clock=args.clock,
        limit=args.limit,
        value=args.value,
        gpu_index=args.gpu_index,
        script_path=args.script_path,
        use_sudo=args.sudo,
        sudo_path=args.sudo_path,
    )

    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError as exc:
        print(
            format_missing_executable_error(exc, script_path=args.script_path),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
