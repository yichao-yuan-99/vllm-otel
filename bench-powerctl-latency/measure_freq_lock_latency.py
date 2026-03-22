#!/usr/bin/env python3
"""Measure NVML locked-clock control latency on an idle GPU.

The benchmark sets GPU locked clocks via pynvml and polls clock telemetry at a
high rate until the first observed frequency change or timeout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

try:
    import pynvml
except Exception as exc:  # pragma: no cover - import failure path
    pynvml = None
    _PYNVML_IMPORT_ERROR = exc
else:
    _PYNVML_IMPORT_ERROR = None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set locked GPU clocks using NVML and measure time to first observed "
            "clock change."
        )
    )
    parser.add_argument(
        "--gpu-index",
        "-g",
        type=int,
        required=True,
        help="GPU index to benchmark (must be idle).",
    )
    parser.add_argument(
        "--target-min-mhz",
        type=int,
        default=1305,
        help="Target min locked graphics clock in MHz (default: 1305).",
    )
    parser.add_argument(
        "--target-max-mhz",
        type=int,
        default=None,
        help=(
            "Target max locked graphics clock in MHz. "
            "If omitted, uses the same value as --target-min-mhz."
        ),
    )
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=1000.0,
        help="Telemetry polling frequency in Hz (default: 1000).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=5.0,
        help="Timeout in seconds for observing a change (default: 5.0).",
    )
    parser.add_argument(
        "--min-delta-mhz",
        type=int,
        default=1,
        help=(
            "Minimum absolute MHz delta from baseline to count as changed "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary.",
    )
    args = parser.parse_args(argv)

    if args.gpu_index < 0:
        parser.error("--gpu-index must be >= 0")
    if args.target_min_mhz <= 0:
        parser.error("--target-min-mhz must be > 0")
    if args.target_max_mhz is not None and args.target_max_mhz <= 0:
        parser.error("--target-max-mhz must be > 0")
    if args.poll_hz <= 0:
        parser.error("--poll-hz must be > 0")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be > 0")
    if args.min_delta_mhz <= 0:
        parser.error("--min-delta-mhz must be > 0")
    return args


def _clock_mhz(handle: Any) -> int:
    return int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))


def _ns_to_ms(delta_ns: int) -> float:
    return delta_ns / 1_000_000.0


def _device_name_str(handle: Any) -> str:
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    return str(name)


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if pynvml is None:
        raise RuntimeError(f"pynvml is required but could not be imported: {_PYNVML_IMPORT_ERROR}")

    target_min_mhz = int(args.target_min_mhz)
    target_max_mhz = int(args.target_max_mhz or args.target_min_mhz)
    if target_max_mhz < target_min_mhz:
        raise ValueError("--target-max-mhz must be >= --target-min-mhz")

    result: dict[str, Any] = {
        "gpu_index": int(args.gpu_index),
        "target_min_mhz": target_min_mhz,
        "target_max_mhz": target_max_mhz,
        "poll_hz": float(args.poll_hz),
        "timeout_s": float(args.timeout_s),
        "min_delta_mhz": int(args.min_delta_mhz),
        "clock_changed": False,
        "timed_out": False,
    }

    nvml_initialized = False
    set_called = False
    reset_attempted = False
    reset_ok = False
    handle: Any | None = None
    try:
        pynvml.nvmlInit()
        nvml_initialized = True

        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
        result["gpu_name"] = _device_name_str(handle)

        baseline_clock = _clock_mhz(handle)
        result["baseline_clock_mhz"] = baseline_clock

        set_start_ns = time.perf_counter_ns()
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_min_mhz, target_max_mhz)
        set_end_ns = time.perf_counter_ns()
        set_called = True

        result["set_call_latency_ms"] = _ns_to_ms(set_end_ns - set_start_ns)

        poll_period_ns = max(1, int(round(1_000_000_000 / args.poll_hz)))
        timeout_ns = int(round(args.timeout_s * 1_000_000_000))
        deadline_ns = set_end_ns + timeout_ns
        next_sample_ns = set_end_ns

        sample_count = 0
        observed_clock_mhz: int | None = None
        observed_ns: int | None = None

        while True:
            now_ns = time.perf_counter_ns()
            if now_ns < next_sample_ns:
                time.sleep((next_sample_ns - now_ns) / 1_000_000_000.0)
                continue

            current_clock = _clock_mhz(handle)
            sample_count += 1

            if abs(current_clock - baseline_clock) >= args.min_delta_mhz:
                observed_clock_mhz = current_clock
                observed_ns = now_ns
                break

            if now_ns >= deadline_ns:
                break

            next_sample_ns += poll_period_ns
            if next_sample_ns <= now_ns:
                skipped = ((now_ns - next_sample_ns) // poll_period_ns) + 1
                next_sample_ns += skipped * poll_period_ns

        result["sample_count"] = sample_count
        result["observed_clock_mhz"] = observed_clock_mhz
        result["effective_poll_hz"] = (
            sample_count / args.timeout_s if observed_ns is None else None
        )

        if observed_ns is None:
            result["timed_out"] = True
            return result, 2

        result["clock_changed"] = True
        result["latency_from_set_start_ms"] = _ns_to_ms(observed_ns - set_start_ns)
        result["latency_from_set_return_ms"] = _ns_to_ms(observed_ns - set_end_ns)
        return result, 0

    finally:
        if nvml_initialized:
            reset_err: Exception | None = None
            try:
                if set_called and handle is not None:
                    reset_attempted = True
                    pynvml.nvmlDeviceResetGpuLockedClocks(handle)
                    reset_ok = True
            except Exception as exc:
                reset_err = exc
            finally:
                pynvml.nvmlShutdown()
                result["reset_attempted"] = reset_attempted
                result["reset_ok"] = reset_ok
            if reset_err is not None:
                raise RuntimeError(f"Failed to reset locked clocks: {reset_err}") from reset_err


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result, exit_code = _run(args)
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    if exit_code == 2:
        print("No clock change observed before timeout.", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
