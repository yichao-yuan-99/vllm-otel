#!/usr/bin/env python3
"""Stream AMD GPU power readings from the local daemon to a JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from deamon.amdsmi_power_service import DEFAULT_SOCKET_PATH, request_gpu_list, request_power


def get_iso_timestamp_microseconds() -> str:
    """Get current UTC timestamp in ISO format with microsecond resolution."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def resolve_gpu_indices(
    *,
    socket_path: str,
    gpu_indices: list[int] | None,
    timeout: float,
) -> list[int]:
    """Resolve the target GPU indices, defaulting to all GPUs known to the daemon."""
    if gpu_indices:
        return gpu_indices

    response = request_gpu_list(socket_path=socket_path, timeout=timeout)
    if not response.get("ok"):
        error = response.get("error", "unknown error")
        raise RuntimeError(f"Failed to list GPUs from daemon: {error}")

    gpus = response.get("gpus", [])
    indices = [gpu["index"] for gpu in gpus if isinstance(gpu, dict) and "index" in gpu]
    if not indices:
        raise RuntimeError("Daemon reported no GPUs")
    return indices


def collect_power_sample(
    *,
    socket_path: str,
    gpu_indices: list[int],
    timeout: float,
) -> dict[str, Any]:
    """Collect one power sample across the selected GPUs."""
    gpu_power_w: dict[str, Any] = {}
    gpu_payload: dict[str, Any] = {}

    for gpu_index in gpu_indices:
        response = request_power(
            gpu_index=gpu_index,
            socket_path=socket_path,
            timeout=timeout,
        )
        if not response.get("ok"):
            error = response.get("error", "unknown error")
            raise RuntimeError(f"Failed to read GPU {gpu_index} power: {error}")

        gpu_power_w[str(gpu_index)] = response.get("power_w")
        gpu_payload[str(gpu_index)] = {
            "gpu": response.get("gpu"),
            "power_info": response.get("power_info"),
        }

    return {
        "timestamp": get_iso_timestamp_microseconds(),
        "payload": {
            socket_path: {
                "timestamp_s": time.time(),
                "gpu_power_w": gpu_power_w,
                "gpu_payload": gpu_payload,
            }
        },
    }


def run_power_reader(
    *,
    output_dir: Path,
    gpu_indices: list[int] | None,
    socket_path: str = DEFAULT_SOCKET_PATH,
    interval_s: float = 0.1,
    timeout: float = 5.0,
    max_samples: int | None = None,
) -> None:
    """Poll the daemon and append AMD GPU power samples to JSONL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_gpu_indices = resolve_gpu_indices(
        socket_path=socket_path,
        gpu_indices=gpu_indices,
        timeout=timeout,
    )
    output_file = output_dir / "power-log.jsonl"

    print("Starting AMD power reader...", file=sys.stderr)
    print(f"Socket path: {socket_path}", file=sys.stderr)
    print(f"GPU indices: {target_gpu_indices}", file=sys.stderr)
    print(f"Output file: {output_file}", file=sys.stderr)
    print(f"Polling interval: {interval_s:.3f} s", file=sys.stderr)

    with output_file.open("a", encoding="utf-8") as handle:
        sample_count = 0
        try:
            while max_samples is None or sample_count < max_samples:
                record = collect_power_sample(
                    socket_path=socket_path,
                    gpu_indices=target_gpu_indices,
                    timeout=timeout,
                )
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                sample_count += 1

                if max_samples is None or sample_count < max_samples:
                    time.sleep(interval_s)
        except KeyboardInterrupt:
            print("\nStopping AMD power reader...", file=sys.stderr)
        finally:
            print(f"Power readings saved to: {output_file}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the AMD power reader."""
    parser = argparse.ArgumentParser(
        prog="amd-power-reader",
        description="Stream AMD GPU power readings from the local daemon to JSONL.",
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
        help="GPU indices to monitor. Defaults to all GPUs reported by the daemon.",
    )
    parser.add_argument(
        "--socket-path",
        "-s",
        default=DEFAULT_SOCKET_PATH,
        help=f"Path to the daemon Unix socket (default: {DEFAULT_SOCKET_PATH}).",
    )
    parser.add_argument(
        "--interval-ms",
        "-i",
        type=float,
        default=100.0,
        help="Polling interval in milliseconds (default: 100).",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=5.0,
        help="Per-request IPC timeout in seconds (default: 5.0).",
    )
    args = parser.parse_args(argv)

    if args.interval_ms <= 0:
        print("Error: --interval-ms must be positive", file=sys.stderr)
        return 1
    if args.timeout <= 0:
        print("Error: --timeout must be positive", file=sys.stderr)
        return 1

    try:
        run_power_reader(
            output_dir=args.output_dir,
            gpu_indices=args.gpu_indices,
            socket_path=args.socket_path,
            interval_s=args.interval_ms / 1000.0,
            timeout=args.timeout,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
