#!/usr/bin/env python3
"""Read current AMD GPU clock frequency through the AMD SMI Python API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


try:
    import amdsmi as _amdsmi
except Exception as exc:  # pragma: no cover - depends on local ROCm install
    _amdsmi = None
    _AMDSMI_IMPORT_ERROR = exc
else:
    _AMDSMI_IMPORT_ERROR = None


CLOCK_TYPE_NAMES = {
    "sclk": "GFX",
    "mclk": "MEM",
}


def _safe_call(amdsmi_module: Any, name: str, *args: Any) -> Any:
    func = getattr(amdsmi_module, name)
    try:
        return func(*args)
    except Exception:
        return None


def _build_gpu_clock_sample(
    *,
    amdsmi_module: Any,
    handle: Any,
    gpu_index: int,
    clock: str,
) -> dict[str, Any]:
    clock_type = getattr(amdsmi_module.AmdSmiClkType, CLOCK_TYPE_NAMES[clock])
    clock_info = amdsmi_module.amdsmi_get_clock_info(handle, clock_type)
    freq_info = amdsmi_module.amdsmi_get_clk_freq(handle, clock_type)
    frequencies_hz = freq_info.get("frequency", []) if isinstance(freq_info, dict) else []
    supported_mhz = [
        round(freq_hz / 1_000_000)
        for freq_hz in frequencies_hz
        if isinstance(freq_hz, (int, float))
    ]

    return {
        "gpu_index": gpu_index,
        "clock": clock,
        "bdf": _safe_call(amdsmi_module, "amdsmi_get_gpu_device_bdf", handle),
        "uuid": _safe_call(amdsmi_module, "amdsmi_get_gpu_device_uuid", handle),
        "current_mhz": clock_info.get("clk"),
        "min_mhz": clock_info.get("min_clk"),
        "max_mhz": clock_info.get("max_clk"),
        "clk_locked": clock_info.get("clk_locked"),
        "clk_deep_sleep": clock_info.get("clk_deep_sleep"),
        "current_level": freq_info.get("current") if isinstance(freq_info, dict) else None,
        "supported_mhz": supported_mhz,
        "raw_clock_info": clock_info,
        "raw_freq_info": freq_info,
    }


def read_clock_samples(
    *,
    clock: str,
    gpu_index: int | None,
    amdsmi_module: Any | None = None,
) -> list[dict[str, Any]]:
    """Read one or more GPU clock samples from AMD SMI."""
    if amdsmi_module is None:
        if _amdsmi is None:
            raise RuntimeError(
                "amdsmi import failed; install ROCm AMD SMI Python bindings first"
            ) from _AMDSMI_IMPORT_ERROR
        amdsmi_module = _amdsmi

    amdsmi_module.amdsmi_init()
    try:
        handles = list(amdsmi_module.amdsmi_get_processor_handles())
        if gpu_index is not None:
            if gpu_index < 0 or gpu_index >= len(handles):
                raise IndexError(
                    f"gpu_index {gpu_index} out of range; found {len(handles)} GPU(s)"
                )
            target_indices = [gpu_index]
        else:
            target_indices = list(range(len(handles)))

        return [
            _build_gpu_clock_sample(
                amdsmi_module=amdsmi_module,
                handle=handles[index],
                gpu_index=index,
                clock=clock,
            )
            for index in target_indices
        ]
    finally:
        amdsmi_module.amdsmi_shut_down()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for reading current GPU clocks."""
    parser = argparse.ArgumentParser(
        prog="get-amd-gpu-clockfreq",
        description="Read current AMD GPU clock frequency through AMD SMI.",
    )
    parser.add_argument(
        "--clock",
        choices=("sclk", "mclk"),
        required=True,
        help="Clock type to read.",
    )
    parser.add_argument(
        "--gpu-index",
        "-g",
        type=int,
        help="Optional GPU index. If omitted, all GPUs are queried.",
    )
    parser.add_argument(
        "--mhz-only",
        action="store_true",
        help="Print only the current frequency in MHz. Requires --gpu-index.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args(argv)

    if args.mhz_only and args.gpu_index is None:
        print("Error: --mhz-only requires --gpu-index", file=sys.stderr)
        return 1

    try:
        samples = read_clock_samples(clock=args.clock, gpu_index=args.gpu_index)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.mhz_only:
        print(samples[0]["current_mhz"])
        return 0

    payload: dict[str, Any]
    if args.gpu_index is None:
        payload = {"clock": args.clock, "gpus": samples}
    else:
        payload = samples[0]

    print(json.dumps(payload, ensure_ascii=True, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
