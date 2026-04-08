from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import set_gpu_core_freq as set_core_freq


def test_build_set_commands() -> None:
    commands = set_core_freq.build_set_commands(
        gpu_index=1,
        min_mhz=1200,
        max_mhz=1500,
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
            "1500",
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
            "1200",
            "-gpu",
            "1",
        ],
    ]


def test_main_runs_commands(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(set_core_freq.subprocess, "run", fake_run)

    exit_code = set_core_freq.main(
        ["--gpu-index", "0", "--min-mhz", "1200", "--max-mhz", "1500"]
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
            "1500",
            "-gpu",
            "0",
        ],
        [
            "sudo",
            "/usr/local/bin/set_gpu_clockfreq.sh",
            "-clock",
            "sclk",
            "-limit",
            "min",
            "-value",
            "1200",
            "-gpu",
            "0",
        ],
    ]


def test_build_set_commands_max_only() -> None:
    commands = set_core_freq.build_set_commands(
        gpu_index=1,
        min_mhz=None,
        max_mhz=1500,
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
            "1500",
            "-gpu",
            "1",
        ],
    ]


def test_main_runs_max_only_command(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(set_core_freq.subprocess, "run", fake_run)

    exit_code = set_core_freq.main(
        ["--gpu-index", "0", "--max-mhz", "1500"]
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
            "1500",
            "-gpu",
            "0",
        ],
    ]


def test_main_runs_commands_without_sudo_when_requested(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(set_core_freq.subprocess, "run", fake_run)

    exit_code = set_core_freq.main(
        ["--gpu-index", "0", "--max-mhz", "1500", "--no-sudo"]
    )

    assert exit_code == 0
    assert calls == [
        [
            "/usr/local/bin/set_gpu_clockfreq.sh",
            "-clock",
            "sclk",
            "-limit",
            "max",
            "-value",
            "1500",
            "-gpu",
            "0",
        ],
    ]


def test_main_requires_at_least_one_limit(capsys) -> None:
    exit_code = set_core_freq.main(["--gpu-index", "0"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "at least one of --min-mhz or --max-mhz is required" in captured.err


def test_main_rejects_invalid_range(capsys) -> None:
    exit_code = set_core_freq.main(
        ["--gpu-index", "0", "--min-mhz", "1600", "--max-mhz", "1500"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--min-mhz must be less than or equal to --max-mhz" in captured.err
