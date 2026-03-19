#!/usr/bin/env python3
"""Set GPU core clock frequency (locked clocks).

Usage:
    set-gpu-core-freq --gpu-index 0 --min-mhz 1200 --max-mhz 1500
    set-gpu-core-freq -g 0 -n 1200 -x 1500
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
    """Set GPU core clock frequency."""
    parser = argparse.ArgumentParser(
        prog="set-gpu-core-freq",
        description="Set GPU core clock frequency (locked clocks).",
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
        help="Minimum core clock in MHz.",
    )
    parser.add_argument(
        "--max-mhz",
        "-x",
        type=int,
        required=True,
        help="Maximum core clock in MHz.",
    )
    
    args = parser.parse_args(argv)
    
    try:
        gpus = get_gpus()
        gpus.set_gpu_locked_clocks(
            gpu_index=args.gpu_index,
            min_clock_mhz=args.min_mhz,
            max_clock_mhz=args.max_mhz,
        )
        print(
            f"GPU {args.gpu_index}: Set core clock to {args.min_mhz}-{args.max_mhz} MHz",
            file=sys.stderr,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
