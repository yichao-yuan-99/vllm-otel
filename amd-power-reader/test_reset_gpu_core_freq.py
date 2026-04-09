from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reset_gpu_core_freq as reset_core_freq


def test_main_runs_reset_commands(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(reset_core_freq.subprocess, "run", fake_run)

    exit_code = reset_core_freq.main(["--gpu-index", "2"])

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
            "2",
        ]
    ]


def test_main_runs_reset_commands_without_sudo_when_requested(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(reset_core_freq.subprocess, "run", fake_run)

    exit_code = reset_core_freq.main(["--gpu-index", "2", "--no-sudo"])

    assert exit_code == 0
    assert calls == [
        [
            "/usr/local/bin/set_gpu_clockfreq.sh",
            "-clock",
            "sclk",
            "-limit",
            "max",
            "-value",
            "1700",
            "-gpu",
            "2",
        ]
    ]
