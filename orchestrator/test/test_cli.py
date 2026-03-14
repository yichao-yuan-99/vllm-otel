from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import orchestrator.cli as orchestrator_cli
from orchestrator.cli import JobType
from orchestrator.cli import build_job_command
from orchestrator.cli import derive_default_output_dir
from orchestrator.cli import derive_timestamped_output_dir
from orchestrator.cli import discover_job_configs
from orchestrator.cli import move_jobs_dir_to_output_dir
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


def test_derive_timestamped_output_dir_creates_child_under_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakeNow:
        @staticmethod
        def strftime(fmt: str) -> str:
            del fmt
            return "20260312T220000Z"

    class _FakeDateTime:
        @staticmethod
        def now(tz: object) -> _FakeNow:
            del tz
            return _FakeNow()

    monkeypatch.setattr("orchestrator.cli.datetime", _FakeDateTime)

    resolved = derive_timestamped_output_dir(
        job_type=JobType.REPLAY,
        output_dir_root=tmp_path / "out-root",
    )
    assert resolved == (tmp_path / "out-root" / "orchestrator-replay-20260312T220000Z").resolve()


def test_derive_default_output_dir_replay_uses_common_declared_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    jobs_dir = repo_root / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "a.toml").write_text(
        "\n".join(
            [
                "[replay]",
                'output_dir = "results/replay/batch-1/qps0_1"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (jobs_dir / "b.toml").write_text(
        "\n".join(
            [
                "[replay]",
                'output_dir = "results/replay/batch-1/qps0_2"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    resolved = derive_default_output_dir(
        job_type=JobType.REPLAY,
        jobs_dir=jobs_dir,
        config_glob="*.toml",
        repo_root=repo_root,
    )
    assert resolved == (repo_root / "results" / "replay" / "batch-1").resolve()


def test_derive_default_output_dir_con_driver_uses_common_declared_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    jobs_dir = repo_root / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "a.toml").write_text(
        "\n".join(
            [
                "[driver]",
                'results_dir = "results/record/dabstep/run-a"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (jobs_dir / "b.toml").write_text(
        "\n".join(
            [
                "[driver]",
                'results_dir = "results/record/dabstep/run-b"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    resolved = derive_default_output_dir(
        job_type=JobType.CON_DRIVER,
        jobs_dir=jobs_dir,
        config_glob="*.toml",
        repo_root=repo_root,
    )
    assert resolved == (repo_root / "results" / "record" / "dabstep").resolve()


def test_derive_default_output_dir_falls_back_to_jobs_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    jobs_dir = repo_root / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "a.toml").write_text("[replay]\nplan = \"x\"\n", encoding="utf-8")

    resolved = derive_default_output_dir(
        job_type=JobType.REPLAY,
        jobs_dir=jobs_dir,
        config_glob="*.toml",
        repo_root=repo_root,
    )
    assert resolved == jobs_dir.resolve()


def test_run_callback_uses_timestamped_output_subdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    output_root = tmp_path / "out-root"

    class _FakeNow:
        @staticmethod
        def strftime(fmt: str) -> str:
            del fmt
            return "20260312T220000Z"

    class _FakeDateTime:
        @staticmethod
        def now(tz: object) -> _FakeNow:
            del tz
            return _FakeNow()

    class _FakeExit(Exception):
        def __init__(self, code: int = 0) -> None:
            super().__init__(code)
            self.code = code

    captured: dict[str, Path] = {}

    def fake_run_orchestrator(**kwargs: object) -> dict[str, object]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        captured["output_dir"] = output_dir
        return {
            "status": "ok",
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("orchestrator.cli.datetime", _FakeDateTime)
    monkeypatch.setattr(orchestrator_cli, "run_orchestrator", fake_run_orchestrator)
    monkeypatch.setattr(orchestrator_cli.typer, "Exit", _FakeExit)
    monkeypatch.setattr(orchestrator_cli.typer, "echo", lambda *args, **kwargs: None)

    with pytest.raises(_FakeExit) as excinfo:
        orchestrator_cli.run(
            job_type=JobType.REPLAY,
            jobs_dir=jobs_dir,
            port_profile_id_list="0",
            config_glob="*.toml",
            output_dir=output_root,
            poll_interval_s=1.0,
            health_timeout_s=5.0,
            fail_fast=False,
            move_jobs_dir=False,
        )

    assert excinfo.value.code == 0
    expected_output_dir = (
        output_root / "orchestrator-replay-20260312T220000Z"
    ).resolve()
    assert captured["output_dir"] == expected_output_dir

    summary_path = expected_output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["output_dir"] == str(expected_output_dir)
    assert summary["output_dir_root"] == str(output_root.resolve())
    assert summary["summary_path"] == str(summary_path)


def test_run_callback_infers_output_root_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    declared_root = (tmp_path / "declared-root").resolve()
    (jobs_dir / "a.toml").write_text(
        "\n".join(
            [
                "[replay]",
                f'output_dir = "{declared_root / "run-a"}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (jobs_dir / "b.toml").write_text(
        "\n".join(
            [
                "[replay]",
                f'output_dir = "{declared_root / "run-b"}"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeNow:
        @staticmethod
        def strftime(fmt: str) -> str:
            del fmt
            return "20260312T220000Z"

    class _FakeDateTime:
        @staticmethod
        def now(tz: object) -> _FakeNow:
            del tz
            return _FakeNow()

    class _FakeExit(Exception):
        def __init__(self, code: int = 0) -> None:
            super().__init__(code)
            self.code = code

    captured: dict[str, Path] = {}

    def fake_run_orchestrator(**kwargs: object) -> dict[str, object]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        captured["output_dir"] = output_dir
        return {
            "status": "ok",
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("orchestrator.cli.datetime", _FakeDateTime)
    monkeypatch.setattr(orchestrator_cli, "run_orchestrator", fake_run_orchestrator)
    monkeypatch.setattr(orchestrator_cli.typer, "Exit", _FakeExit)
    monkeypatch.setattr(orchestrator_cli.typer, "echo", lambda *args, **kwargs: None)

    with pytest.raises(_FakeExit) as excinfo:
        orchestrator_cli.run(
            job_type=JobType.REPLAY,
            jobs_dir=jobs_dir,
            port_profile_id_list="0",
            config_glob="*.toml",
            output_dir=None,
            poll_interval_s=1.0,
            health_timeout_s=5.0,
            fail_fast=False,
            move_jobs_dir=False,
        )

    assert excinfo.value.code == 0
    expected_output_dir = (
        declared_root / "orchestrator-replay-20260312T220000Z"
    ).resolve()
    assert captured["output_dir"] == expected_output_dir

    summary_path = expected_output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["output_dir"] == str(expected_output_dir)
    assert summary["output_dir_root"] == str(declared_root)
    assert summary["summary_path"] == str(summary_path)


def test_move_jobs_dir_to_output_dir_moves_directory(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "a.toml").write_text("x=1\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)

    moved_to = move_jobs_dir_to_output_dir(jobs_dir=jobs_dir, output_dir=output_dir)
    assert moved_to == (output_dir / "jobs").resolve()
    assert moved_to.exists()
    assert not jobs_dir.exists()
    assert (moved_to / "a.toml").exists()


def test_move_jobs_dir_to_output_dir_rejects_output_inside_jobs_dir(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    output_dir = jobs_dir / "orchestrator-out"
    output_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="inside --jobs-dir"):
        move_jobs_dir_to_output_dir(jobs_dir=jobs_dir, output_dir=output_dir)


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
