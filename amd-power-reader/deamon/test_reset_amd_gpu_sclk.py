from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reset_amd_gpu_sclk as reset_sclk


def test_build_reset_commands_for_single_gpu() -> None:
    commands = reset_sclk.build_reset_commands(
        gpu_index=3,
        script_path="/usr/local/bin/set_gpu_clockfreq.sh",
    )

    assert commands == [
        [
            "sudo",
            "/usr/local/bin/set_gpu_clockfreq.sh",
            "-clock",
            "sclk",
            "-limit",
            "max",
            "-value",
            "1700",
            "-gpu",
            "3",
        ],
        [
            "sudo",
            "/usr/local/bin/set_gpu_clockfreq.sh",
            "-clock",
            "sclk",
            "-limit",
            "min",
            "-value",
            "1700",
            "-gpu",
            "3",
        ],
    ]


def test_build_reset_commands_for_all_gpus() -> None:
    commands = reset_sclk.build_reset_commands(
        gpu_index=None,
        script_path="/usr/local/bin/set_gpu_clockfreq.sh",
    )

    assert commands == [
        [
            "sudo",
            "/usr/local/bin/set_gpu_clockfreq.sh",
            "-clock",
            "sclk",
            "-limit",
            "max",
            "-value",
            "1700",
        ],
        [
            "sudo",
            "/usr/local/bin/set_gpu_clockfreq.sh",
            "-clock",
            "sclk",
            "-limit",
            "min",
            "-value",
            "1700",
        ],
    ]


def test_main_runs_both_commands_in_order(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(reset_sclk.subprocess, "run", fake_run)

    exit_code = reset_sclk.main(["--gpu-index", "1"])

    assert exit_code == 0
    assert calls == [
        [
            "sudo",
            "/usr/local/bin/set_gpu_clockfreq.sh",
            "-clock",
            "sclk",
            "-limit",
            "max",
            "-value",
            "1700",
            "-gpu",
            "1",
        ],
        [
            "sudo",
            "/usr/local/bin/set_gpu_clockfreq.sh",
            "-clock",
            "sclk",
            "-limit",
            "min",
            "-value",
            "1700",
            "-gpu",
            "1",
        ],
    ]
