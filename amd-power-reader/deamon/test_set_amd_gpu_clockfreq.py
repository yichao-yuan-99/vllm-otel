from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import set_amd_gpu_clockfreq as bridge


def test_build_command_for_single_gpu() -> None:
    command = bridge.build_command(
        clock="sclk",
        limit="max",
        value=1500,
        gpu_index=2,
        script_path="/usr/local/bin/set_gpu_clockfreq.sh",
    )

    assert command == [
        "sudo",
        "/usr/local/bin/set_gpu_clockfreq.sh",
        "-clock",
        "sclk",
        "-limit",
        "max",
        "-value",
        "1500",
        "-gpu",
        "2",
    ]


def test_build_command_for_all_gpus() -> None:
    command = bridge.build_command(
        clock="mclk",
        limit="min",
        value=900,
        gpu_index=None,
        script_path="/usr/local/bin/set_gpu_clockfreq.sh",
    )

    assert command == [
        "sudo",
        "/usr/local/bin/set_gpu_clockfreq.sh",
        "-clock",
        "mclk",
        "-limit",
        "min",
        "-value",
        "900",
    ]


def test_main_runs_script(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)

    exit_code = bridge.main(
        [
            "--clock",
            "sclk",
            "--limit",
            "max",
            "--value",
            "1200",
            "--gpu-index",
            "0",
        ]
    )

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
            "1200",
            "-gpu",
            "0",
        ]
    ]


def test_build_command_without_sudo() -> None:
    command = bridge.build_command(
        clock="sclk",
        limit="min",
        value=1100,
        gpu_index=1,
        script_path="/usr/local/bin/set_gpu_clockfreq.sh",
        use_sudo=False,
    )

    assert command == [
        "/usr/local/bin/set_gpu_clockfreq.sh",
        "-clock",
        "sclk",
        "-limit",
        "min",
        "-value",
        "1100",
        "-gpu",
        "1",
    ]
