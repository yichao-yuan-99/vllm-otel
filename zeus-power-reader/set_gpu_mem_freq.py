#!/usr/bin/env python3
"""Set GPU memory clock frequency (locked clocks).

Usage:
    set-gpu-mem-freq --gpu-index 0 --min-mhz 5000 --max-mhz 6000
    set-gpu-mem-freq -g 0 -n 5000 -x 6000
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
    """Set GPU memory clock frequency."""
    parser = argparse.ArgumentParser(
        prog="set-gpu-mem-freq",
        description="Set GPU memory clock frequency (locked clocks).",
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
        required=True,
        help="Minimum memory clock in MHz.",
    )
    parser.add_argument(
        "--max-mhz",
        "-x",
        type=int,
        required=True,
        help="Maximum memory clock in MHz.",
    )
    
    args = parser.parse_args(argv)
    
    try:
        gpus = get_gpus()
        gpus.set_memory_locked_clocks(
            gpu_index=args.gpu_index,
            min_clock_mhz=args.min_mhz,
            max_clock_mhz=args.max_mhz,
        )
        print(
            f"GPU {args.gpu_index}: Set memory clock to {args.min_mhz}-{args.max_mhz} MHz",
            file=sys.stderr,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
