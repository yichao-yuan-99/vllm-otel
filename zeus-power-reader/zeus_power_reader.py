#!/usr/bin/env python3
"""Zeus power reader - stream power readings from zeusd to JSONL file.

This module provides a power reader that connects to zeusd via Unix domain socket
and writes power readings to a JSONL file with ISO timestamps.

Example usage:
    python -m zeus_power_reader --output-dir ./logs --gpu-indices 0 1 2 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# Set environment variable BEFORE importing zeus
os.environ.setdefault("ZEUSD_SOCK_PATH", "/var/run/zeusd.sock")

from zeus.utils.zeusd import ZeusdConfig
from zeus.monitor.power_streaming import PowerStreamingClient


def get_iso_timestamp_microseconds() -> str:
    """Get current UTC timestamp in ISO format with microsecond resolution.
    
    Returns:
        Timestamp string like "2026-03-17T04:50:16.123456Z"
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def power_reading_to_dict(power_reading: dict) -> dict:
    """Convert PowerReadings dict to a serializable dictionary.
    
    Args:
        power_reading: Dict mapping server endpoint to PowerReadings objects.
        
    Returns:
        Dictionary with serialized power readings.
    """
    result = {}
    for server, readings in power_reading.items():
        result[server] = {
            "timestamp_s": readings.timestamp_s,
            "gpu_power_w": readings.gpu_power_w,
            "cpu_power_w": {
                cpu_idx: {
                    "cpu_w": reading.cpu_w,
                    "dram_w": reading.dram_w,
                }
                for cpu_idx, reading in readings.cpu_power_w.items()
            } if readings.cpu_power_w else {},
        }
    return result


def run_power_reader(
    output_dir: Path,
    gpu_indices: list[int],
    socket_path: str = "/var/run/zeusd.sock",
) -> None:
    """Run the power reader and write readings to JSONL file.
    
    Args:
        output_dir: Directory to write the power-log.jsonl file.
        gpu_indices: List of GPU indices to monitor.
        socket_path: Path to the zeusd Unix domain socket.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "power-log.jsonl"
    
    print(f"Starting power reader...", file=sys.stderr)
    print(f"Socket path: {socket_path}", file=sys.stderr)
    print(f"GPU indices: {gpu_indices}", file=sys.stderr)
    print(f"Output file: {output_file}", file=sys.stderr)
    
    # Create client with Unix domain socket
    client = PowerStreamingClient(
        servers=[
            ZeusdConfig.uds(socket_path=socket_path, gpu_indices=gpu_indices),
        ],
    )
    
    print("Connected to zeusd. Streaming power readings...", file=sys.stderr)
    print("Press Ctrl+C to stop.", file=sys.stderr)
    
    try:
        # Continuously stream power readings
        for power_reading in client:
            timestamp = get_iso_timestamp_microseconds()
            payload = power_reading_to_dict(power_reading)
            
            record = {
                "timestamp": timestamp,
                "payload": payload,
            }
            
            # Append to output file
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                
    except KeyboardInterrupt:
        print("\nStopping power reader...", file=sys.stderr)
    finally:
        client.stop()
        print(f"Power readings saved to: {output_file}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the power reader CLI.
    
    Args:
        argv: Command line arguments.
        
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = argparse.ArgumentParser(
        prog="zeus-power-reader",
        description="Stream power readings from zeusd to JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to write the power-log.jsonl file.",
    )
    parser.add_argument(
        "--gpu-indices",
        "-g",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="GPU indices to monitor (default: 0 1 2 3).",
    )
    parser.add_argument(
        "--socket-path",
        "-s",
        type=str,
        default="/var/run/zeusd.sock",
        help="Path to the zeusd Unix domain socket (default: /var/run/zeusd.sock).",
    )
    
    args = parser.parse_args(argv)
    
    try:
        run_power_reader(
            output_dir=args.output_dir,
            gpu_indices=args.gpu_indices,
            socket_path=args.socket_path,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
