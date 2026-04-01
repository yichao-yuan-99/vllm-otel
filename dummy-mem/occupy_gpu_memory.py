#!/usr/bin/env python3
"""Occupy a configurable amount of GPU memory with PyTorch until interrupted."""

from __future__ import annotations

import argparse
import gc
import signal
import sys
import time
from types import FrameType
from typing import Any


BYTES_PER_GIB = 1024**3


def parse_positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def gib_to_bytes(size_gib: float) -> int:
    return int(round(size_gib * BYTES_PER_GIB))


def format_bytes(num_bytes: int) -> str:
    gib = num_bytes / BYTES_PER_GIB
    return f"{num_bytes} bytes ({gib:.3f} GiB)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Allocate dummy CUDA memory with PyTorch and keep it occupied until "
            "the process is interrupted."
        )
    )
    parser.add_argument(
        "--size",
        required=True,
        type=parse_positive_float,
        help="Dummy GPU memory size in GiB. For example: --size 10",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to allocate on (default: cuda:0).",
    )
    return parser


def import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is not installed. Install torch in this environment first."
        ) from exc
    return torch


def resolve_cuda_device(torch: Any, device_text: str) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment.")

    device = torch.device(device_text)
    if device.type != "cuda":
        raise ValueError(f"--device must be a CUDA device, got: {device_text!r}")

    device_count = int(torch.cuda.device_count())
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif device.index < 0 or device.index >= device_count:
        raise ValueError(
            f"--device index out of range: {device.index}; available devices: 0..{device_count - 1}"
        )

    torch.cuda.set_device(device)
    return device


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        torch = import_torch()
        device = resolve_cuda_device(torch, str(args.device))
        target_bytes = gib_to_bytes(float(args.size))

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except Exception:  # noqa: BLE001
            free_bytes, total_bytes = None, None

        device_name = torch.cuda.get_device_name(device)
        print(
            f"[dummy-mem] allocating {format_bytes(target_bytes)} on {device} ({device_name})",
            flush=True,
        )
        if free_bytes is not None and total_bytes is not None:
            print(
                f"[dummy-mem] current free={format_bytes(int(free_bytes))} "
                f"total={format_bytes(int(total_bytes))}",
                flush=True,
            )
            if target_bytes > int(free_bytes):
                raise RuntimeError(
                    "requested size exceeds currently free CUDA memory on the selected device"
                )

        stop_requested = False

        def _request_stop(signum: int, _frame: FrameType | None) -> None:
            nonlocal stop_requested
            stop_requested = True
            signal_name = signal.Signals(signum).name
            print(f"\n[dummy-mem] received {signal_name}, releasing memory...", flush=True)

        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

        buffer = torch.empty(target_bytes, dtype=torch.uint8, device=device)
        buffer.zero_()
        torch.cuda.synchronize(device)

        allocated_bytes = int(buffer.numel() * buffer.element_size())
        print(
            f"[dummy-mem] holding {format_bytes(allocated_bytes)} on {device}. "
            "Press Ctrl+C to release it.",
            flush=True,
        )

        try:
            while not stop_requested:
                time.sleep(1.0)
        finally:
            del buffer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            print("[dummy-mem] released dummy allocation", flush=True)

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
