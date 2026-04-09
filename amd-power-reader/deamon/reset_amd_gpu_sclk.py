#!/usr/bin/env python3
"""Reset AMD GPU sclk by setting both min and max to 1700 MHz."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

try:
    from .set_amd_gpu_clockfreq import (
        DEFAULT_SCRIPT_PATH,
        DEFAULT_SUDO_PATH,
        build_command,
        format_missing_executable_error,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from set_amd_gpu_clockfreq import (
        DEFAULT_SCRIPT_PATH,
        DEFAULT_SUDO_PATH,
        build_command,
        format_missing_executable_error,
    )


RESET_SCLK_MHZ = 1700


def build_reset_commands(
    *,
    gpu_index: int | None,
    script_path: str,
    use_sudo: bool = True,
    sudo_path: str = DEFAULT_SUDO_PATH,
) -> list[list[str]]:
    """Build the shell commands needed to lock sclk at 1700 MHz."""
    return [
        build_command(
            clock="sclk",
            limit="max",
            value=RESET_SCLK_MHZ,
            gpu_index=gpu_index,
            script_path=script_path,
            use_sudo=use_sudo,
            sudo_path=sudo_path,
        ),
        build_command(
            clock="sclk",
            limit="min",
            value=RESET_SCLK_MHZ,
            gpu_index=gpu_index,
            script_path=script_path,
            use_sudo=use_sudo,
            sudo_path=sudo_path,
        ),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for resetting sclk to 1700 MHz."""
    parser = argparse.ArgumentParser(
        prog="reset-amd-gpu-sclk",
        description="Reset AMD GPU sclk by locking both min and max to 1700 MHz.",
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

    try:
        for command in build_reset_commands(
            gpu_index=args.gpu_index,
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
