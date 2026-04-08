#!/usr/bin/env python3
"""Reset AMD GPU core clock frequency by setting sclk max to 1700 MHz."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

from deamon.reset_amd_gpu_sclk import RESET_SCLK_MHZ
from deamon.set_amd_gpu_clockfreq import (
    DEFAULT_SCRIPT_PATH,
    DEFAULT_SUDO_PATH,
    build_command,
    format_missing_executable_error,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Reset AMD GPU core max frequency to the project default."""
    parser = argparse.ArgumentParser(
        prog="amd-reset-gpu-core-freq",
        description=(
            "Reset AMD GPU core clock frequency by setting sclk max to 1700 MHz."
        ),
    )
    parser.add_argument(
        "--gpu-index",
        "-g",
        type=int,
        required=True,
        help="GPU index to reset.",
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
        command = build_command(
            clock="sclk",
            limit="max",
            value=RESET_SCLK_MHZ,
            gpu_index=args.gpu_index,
            script_path=args.script_path,
            use_sudo=args.sudo,
            sudo_path=args.sudo_path,
        )
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

    print(
        f"GPU {args.gpu_index}: Reset max core clock to {RESET_SCLK_MHZ} MHz",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
