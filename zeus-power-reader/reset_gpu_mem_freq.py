#!/usr/bin/env python3
"""Reset GPU memory clock frequency to defaults.

Usage:
    reset-gpu-mem-freq --gpu-index 0
    reset-gpu-mem-freq -g 0
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

# Set environment variable BEFORE importing zeus
os.environ.setdefault("ZEUSD_SOCK_PATH", "/var/run/zeusd.sock")

from zeus.device import get_gpus


def main(argv: Sequence[str] | None = None) -> int:
    """Reset GPU memory clock frequency to defaults."""
    parser = argparse.ArgumentParser(
        prog="reset-gpu-mem-freq",
        description="Reset GPU memory clock frequency to defaults.",
    )
    parser.add_argument(
        "--gpu-index",
        "-g",
        type=int,
        required=True,
        help="GPU index to reset.",
    )
    
    args = parser.parse_args(argv)
    
    try:
        gpus = get_gpus()
        gpus.reset_memory_locked_clocks(gpu_index=args.gpu_index)
        print(f"GPU {args.gpu_index}: Reset memory clock to defaults", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
