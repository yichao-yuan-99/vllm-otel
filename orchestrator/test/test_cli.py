from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.cli import JobType
from orchestrator.cli import build_job_command
from orchestrator.cli import discover_job_configs
from orchestrator.cli import parse_port_profile_id_list
from orchestrator.cli import run_orchestrator


def test_parse_port_profile_id_list_valid() -> None:
    assert parse_port_profile_id_list("0,1,2") == [0, 1, 2]


def test_parse_port_profile_id_list_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_port_profile_id_list("0,1,1")


def test_discover_job_configs_sorts_files(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "b.toml").write_text("x=1\n", encoding="utf-8")
    (jobs_dir / "a.toml").write_text("x=2\n", encoding="utf-8")
    (jobs_dir / "ignore.txt").write_text("x\n", encoding="utf-8")

    discovered = discover_job_configs(jobs_dir=jobs_dir, config_glob="*.toml")
    assert [path.name for path in discovered] == ["a.toml", "b.toml"]


def test_build_job_command_for_replay() -> None:
    cmd = build_job_command(
        job_type=JobType.REPLAY,
        config_path=Path("/tmp/job.toml"),
        profile_id=3,
    )
    assert cmd == [
        sys.executable,
        "-m",
        "replayer",
        "replay",
        "--config",
        "/tmp/job.toml",
        "--port-profile-id",
        "3",
    ]


def test_run_orchestrator_serial_scheduling_single_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "00.toml").write_text("x=1\n", encoding="utf-8")
    (jobs_dir / "01.toml").write_text("x=1\n", encoding="utf-8")
    (jobs_dir / "02.toml").write_text("x=1\n", encoding="utf-8")

    monkeypatch.setattr("orchestrator.cli.check_gateway_health", lambda **_: None)
    monkeypatch.setattr("orchestrator.cli.time.sleep", lambda _: None)

    class FakePopen:
        commands: list[list[str]] = []

        def __init__(
            self,
            command: list[str],
            *,
            stdout,
            stderr,
            text: bool,
            cwd: Path,
            env: dict[str, str],
        ) -> None:
            del stderr, text, cwd, env
            self.command = command
            self.returncode: int | None = None
            self._poll_count = 0
            self._stdout = stdout
            FakePopen.commands.append(command)

        def poll(self) -> int | None:
            self._poll_count += 1
            if self.returncode is None and self._poll_count >= 2:
                self.returncode = 0
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 1

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr("orchestrator.cli.subprocess.Popen", FakePopen)

    summary = run_orchestrator(
        job_type=JobType.REPLAY,
        jobs_dir=jobs_dir,
        config_glob="*.toml",
        profile_ids=[0],
        output_dir=tmp_path / "out",
        poll_interval_s=0.01,
        health_timeout_s=5.0,
        fail_fast=False,
    )

    assert summary["status"] == "ok"
    assert summary["total_jobs"] == 3
    assert summary["launched_jobs"] == 3
    assert summary["failed"] == 0
    assert all("--port-profile-id" in command for command in FakePopen.commands)
    assert all(command[-1] == "0" for command in FakePopen.commands)


def test_run_orchestrator_fail_fast_stops_new_launches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "00-fail.toml").write_text("x=1\n", encoding="utf-8")
    (jobs_dir / "01-ok.toml").write_text("x=1\n", encoding="utf-8")

    monkeypatch.setattr("orchestrator.cli.check_gateway_health", lambda **_: None)
    monkeypatch.setattr("orchestrator.cli.time.sleep", lambda _: None)

    class FakePopen:
        def __init__(
            self,
            command: list[str],
            *,
            stdout,
            stderr,
            text: bool,
            cwd: Path,
            env: dict[str, str],
        ) -> None:
            del stderr, text, cwd, env
            self.command = command
            self.returncode: int | None = None
            self._poll_count = 0
            self._stdout = stdout
            config_path = Path(command[command.index("--config") + 1])
            self._target_exit_code = 1 if "fail" in config_path.name else 0

        def poll(self) -> int | None:
            self._poll_count += 1
            if self.returncode is None and self._poll_count >= 2:
                self.returncode = self._target_exit_code
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 1

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                self.returncode = self._target_exit_code
            return self.returncode

    monkeypatch.setattr("orchestrator.cli.subprocess.Popen", FakePopen)

    summary = run_orchestrator(
        job_type=JobType.CON_DRIVER,
        jobs_dir=jobs_dir,
        config_glob="*.toml",
        profile_ids=[0],
        output_dir=tmp_path / "out",
        poll_interval_s=0.01,
        health_timeout_s=5.0,
        fail_fast=True,
    )

    assert summary["status"] == "failed"
    assert summary["launched_jobs"] == 1
    assert summary["pending_jobs"] == 1
    assert summary["failed"] == 1


def test_run_orchestrator_con_driver_skips_gateway_health_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "00.toml").write_text("x=1\n", encoding="utf-8")

    def fail_if_called(**_: object) -> None:
        raise AssertionError("check_gateway_health should not be called for con-driver jobs")

    monkeypatch.setattr("orchestrator.cli.check_gateway_health", fail_if_called)
    monkeypatch.setattr("orchestrator.cli.time.sleep", lambda _: None)

    class FakePopen:
        def __init__(
            self,
            command: list[str],
            *,
            stdout,
            stderr,
            text: bool,
            cwd: Path,
            env: dict[str, str],
        ) -> None:
            del command, stderr, text, cwd, env
            self.returncode: int | None = None
            self._poll_count = 0
            self._stdout = stdout

        def poll(self) -> int | None:
            self._poll_count += 1
            if self.returncode is None and self._poll_count >= 2:
                self.returncode = 0
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 1

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr("orchestrator.cli.subprocess.Popen", FakePopen)

    summary = run_orchestrator(
        job_type=JobType.CON_DRIVER,
        jobs_dir=jobs_dir,
        config_glob="*.toml",
        profile_ids=[0],
        output_dir=tmp_path / "out",
        poll_interval_s=0.01,
        health_timeout_s=5.0,
        fail_fast=False,
    )

    assert summary["status"] == "ok"
    assert summary["failed"] == 0
