from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import get_amd_gpu_clockfreq as reader


class FakeClkType:
    GFX = "gfx"
    MEM = "mem"


class FakeAmdSmi:
    AmdSmiClkType = FakeClkType

    def __init__(self) -> None:
        self.init_calls = 0
        self.shutdown_calls = 0
        self.handles = ["gpu0", "gpu1"]

    def amdsmi_init(self) -> None:
        self.init_calls += 1

    def amdsmi_shut_down(self) -> None:
        self.shutdown_calls += 1

    def amdsmi_get_processor_handles(self) -> list[str]:
        return list(self.handles)

    def amdsmi_get_gpu_device_bdf(self, handle: str) -> str:
        return {
            "gpu0": "0000:01:00.0",
            "gpu1": "0000:02:00.0",
        }[handle]

    def amdsmi_get_gpu_device_uuid(self, handle: str) -> str:
        return {
            "gpu0": "uuid-0",
            "gpu1": "uuid-1",
        }[handle]

    def amdsmi_get_clock_info(self, handle: str, clock_type: str) -> dict[str, int | bool | str]:
        return {
            ("gpu0", "gfx"): {
                "clk": 800,
                "min_clk": 500,
                "max_clk": 1700,
                "clk_locked": False,
                "clk_deep_sleep": "N/A",
            },
            ("gpu1", "gfx"): {
                "clk": 1200,
                "min_clk": 500,
                "max_clk": 1700,
                "clk_locked": True,
                "clk_deep_sleep": "N/A",
            },
            ("gpu0", "mem"): {
                "clk": 1600,
                "min_clk": 400,
                "max_clk": 1600,
                "clk_locked": False,
                "clk_deep_sleep": "N/A",
            },
            ("gpu1", "mem"): {
                "clk": 1200,
                "min_clk": 400,
                "max_clk": 1600,
                "clk_locked": False,
                "clk_deep_sleep": "N/A",
            },
        }[(handle, clock_type)]

    def amdsmi_get_clk_freq(self, handle: str, clock_type: str) -> dict[str, int | list[int]]:
        return {
            ("gpu0", "gfx"): {
                "num_supported": 3,
                "current": 1,
                "frequency": [500_000_000, 800_000_000, 1_700_000_000],
            },
            ("gpu1", "gfx"): {
                "num_supported": 3,
                "current": 2,
                "frequency": [500_000_000, 800_000_000, 1_700_000_000],
            },
            ("gpu0", "mem"): {
                "num_supported": 4,
                "current": 3,
                "frequency": [400_000_000, 700_000_000, 1_200_000_000, 1_600_000_000],
            },
            ("gpu1", "mem"): {
                "num_supported": 4,
                "current": 2,
                "frequency": [400_000_000, 700_000_000, 1_200_000_000, 1_600_000_000],
            },
        }[(handle, clock_type)]


def test_read_single_gpu_clock_sample() -> None:
    backend = FakeAmdSmi()

    samples = reader.read_clock_samples(
        clock="sclk",
        gpu_index=1,
        amdsmi_module=backend,
    )

    assert backend.init_calls == 1
    assert backend.shutdown_calls == 1
    assert len(samples) == 1
    assert samples[0]["gpu_index"] == 1
    assert samples[0]["current_mhz"] == 1200
    assert samples[0]["supported_mhz"] == [500, 800, 1700]
    assert samples[0]["current_level"] == 2


def test_read_all_gpu_clock_samples() -> None:
    backend = FakeAmdSmi()

    samples = reader.read_clock_samples(
        clock="mclk",
        gpu_index=None,
        amdsmi_module=backend,
    )

    assert [sample["gpu_index"] for sample in samples] == [0, 1]
    assert [sample["current_mhz"] for sample in samples] == [1600, 1200]


def test_main_rejects_mhz_only_without_gpu_index(capsys) -> None:
    exit_code = reader.main(["--clock", "sclk", "--mhz-only"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--mhz-only requires --gpu-index" in captured.err
