#!/usr/bin/env python3
"""Background control CLI for the interactive AMD HPC embedded TP1 stack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
START_SCRIPT = SCRIPT_DIR / "start-services.sh"
LOG_DIR = SCRIPT_DIR / "logs"
PID_FILE = LOG_DIR / "background-service.pid.json"
READY_MARKER = "Services are ready."
ProgressFn = Callable[[str], None]


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_run_id() -> str:
    return f"interactive-bg-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    status_path = Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("State:"):
                return "\tZ" not in line and "(zombie)" not in line.lower()
    except OSError:
        pass
    return True


def _read_cmdline(pid: int) -> str | None:
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline_path.read_bytes()
    except OSError:
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _tail_text(path: Path, *, lines: int = 20) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    tail = text.splitlines()[-lines:]
    return "\n".join(tail)


def _emit_progress(progress: ProgressFn | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _reap_pid_if_child(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return


def _detach_background_process(proc: subprocess.Popen[object]) -> None:
    # The launcher is intentionally orphaned so the CLI can exit immediately.
    proc._child_created = False  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ServiceRecord:
    pid: int
    run_id: str
    started_at: str
    pid_file: str
    launcher_log: str
    launcher_log_offset: int
    start_script: str
    repo_root: str

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "pid_file": self.pid_file,
            "launcher_log": self.launcher_log,
            "launcher_log_offset": self.launcher_log_offset,
            "start_script": self.start_script,
            "repo_root": self.repo_root,
        }

    @classmethod
    def from_dict(cls, raw: object) -> ServiceRecord | None:
        if not isinstance(raw, dict):
            return None
        try:
            pid = int(raw["pid"])
            run_id = str(raw["run_id"])
            started_at = str(raw["started_at"])
            pid_file = str(raw["pid_file"])
            launcher_log = str(raw["launcher_log"])
            launcher_log_offset = int(raw.get("launcher_log_offset", 0))
            start_script = str(raw["start_script"])
            repo_root = str(raw["repo_root"])
        except (KeyError, TypeError, ValueError):
            return None
        return cls(
            pid=pid,
            run_id=run_id,
            started_at=started_at,
            pid_file=pid_file,
            launcher_log=launcher_log,
            launcher_log_offset=launcher_log_offset,
            start_script=start_script,
            repo_root=repo_root,
        )


@dataclass
class LogProgressState:
    offset: int
    partial_line: str = ""


class BackgroundServiceController:
    def __init__(
        self,
        *,
        start_script: Path = START_SCRIPT,
        repo_root: Path = REPO_ROOT,
        log_dir: Path = LOG_DIR,
        pid_file: Path = PID_FILE,
    ) -> None:
        self.start_script = Path(start_script)
        self.repo_root = Path(repo_root)
        self.log_dir = Path(log_dir)
        self.pid_file = Path(pid_file)

    def _wait_for_ready(
        self,
        *,
        pid: int,
        launcher_log: Path,
        offset: int,
        timeout_seconds: float,
        poll_interval_seconds: float,
        progress: ProgressFn | None = None,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        start_time = time.monotonic()
        next_heartbeat_at = start_time
        log_state = LogProgressState(offset=offset)
        while time.monotonic() < deadline:
            if not _pid_is_running(pid):
                _reap_pid_if_child(pid)
                self._emit_new_log_progress(launcher_log=launcher_log, state=log_state, progress=progress)
                tail = _tail_text(launcher_log)
                raise RuntimeError(
                    f"background service exited before readiness. See {launcher_log}\n{tail}".rstrip()
                )
            emitted_lines = self._emit_new_log_progress(
                launcher_log=launcher_log,
                state=log_state,
                progress=progress,
            )
            if any(READY_MARKER in line for line in emitted_lines):
                _emit_progress(progress, "Readiness confirmed.")
                return
            now = time.monotonic()
            if now >= next_heartbeat_at:
                elapsed_seconds = int(now - start_time)
                _emit_progress(progress, f"Waiting for readiness... {elapsed_seconds}s elapsed")
                next_heartbeat_at = now + max(5.0, poll_interval_seconds)
            time.sleep(poll_interval_seconds)

        self._emit_new_log_progress(launcher_log=launcher_log, state=log_state, progress=progress)
        tail = _tail_text(launcher_log)
        raise RuntimeError(
            f"timed out waiting for background service readiness. See {launcher_log}\n{tail}".rstrip()
        )

    def _emit_new_log_progress(
        self,
        *,
        launcher_log: Path,
        state: LogProgressState,
        progress: ProgressFn | None,
    ) -> list[str]:
        if not launcher_log.exists():
            return []
        with launcher_log.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(state.offset)
            chunk = handle.read()
            state.offset = handle.tell()
        if not chunk:
            return []

        text = f"{state.partial_line}{chunk}"
        state.partial_line = ""
        lines: list[str] = []
        for piece in text.splitlines(keepends=True):
            if piece.endswith("\n") or piece.endswith("\r"):
                line = piece.rstrip("\r\n")
                lines.append(line)
            else:
                state.partial_line = piece
        for line in lines:
            _emit_progress(progress, f"launcher: {line}")
        return lines

    def _resolve_wait_settings(
        self,
        *,
        env: Mapping[str, str] | None,
        timeout_seconds: float | None,
        poll_interval_seconds: float | None,
    ) -> tuple[float, float]:
        env_map = dict(os.environ)
        if env is not None:
            env_map.update(dict(env))

        service_ready_timeout_seconds = float(env_map.get("SERVICE_READY_TIMEOUT_SECONDS", "900"))
        effective_timeout = timeout_seconds
        if effective_timeout is None:
            effective_timeout = (service_ready_timeout_seconds * 2.0) + 30.0

        effective_poll_interval = poll_interval_seconds
        if effective_poll_interval is None:
            effective_poll_interval = float(env_map.get("SERVICE_READY_POLL_INTERVAL_SECONDS", "2.0"))
        return effective_timeout, effective_poll_interval

    def load_record(self) -> ServiceRecord | None:
        try:
            raw = json.loads(self.pid_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return ServiceRecord.from_dict(raw)

    def running_record(self) -> ServiceRecord | None:
        record = self.load_record()
        if record is None or not _pid_is_running(record.pid):
            return None
        cmdline = _read_cmdline(record.pid)
        if cmdline and record.start_script not in cmdline:
            return None
        return record

    def start(
        self,
        *,
        env: Mapping[str, str] | None = None,
        wait_for_ready: bool = True,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
        progress: ProgressFn | None = None,
    ) -> tuple[str, ServiceRecord]:
        running = self.running_record()
        if running is not None:
            if wait_for_ready:
                _emit_progress(
                    progress,
                    f"Service already running (pid={running.pid}); waiting for readiness via {running.launcher_log}",
                )
                effective_timeout, effective_poll_interval = self._resolve_wait_settings(
                    env=env,
                    timeout_seconds=timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                )
                self._wait_for_ready(
                    pid=running.pid,
                    launcher_log=Path(running.launcher_log),
                    offset=running.launcher_log_offset,
                    timeout_seconds=effective_timeout,
                    poll_interval_seconds=effective_poll_interval,
                    progress=progress,
                )
            return "already_running", running

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file.unlink(missing_ok=True)

        if not self.start_script.is_file():
            raise RuntimeError(f"start script not found: {self.start_script}")

        launcher_env = dict(os.environ)
        if env is not None:
            launcher_env.update(dict(env))
        run_id = launcher_env.get("RUN_ID") or _default_run_id()
        launcher_env["RUN_ID"] = run_id

        launcher_log = self.log_dir / f"launcher.{run_id}.log"
        cmd = ["bash", str(self.start_script)]
        _emit_progress(progress, f"Starting background service with run_id={run_id}")
        _emit_progress(progress, f"Launcher log: {launcher_log}")
        with launcher_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{_utc_now_iso()}] starting background service: {' '.join(cmd)}\n")
            handle.flush()
            launcher_log_offset = handle.tell()
            proc = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                env=launcher_env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )

        time.sleep(0.25)
        if proc.poll() is not None:
            _reap_pid_if_child(proc.pid)
            tail = _tail_text(launcher_log)
            raise RuntimeError(
                f"failed to start background service (exit={proc.returncode})\n{tail}".rstrip()
            )
        _detach_background_process(proc)

        record = ServiceRecord(
            pid=proc.pid,
            run_id=run_id,
            started_at=_utc_now_iso(),
            pid_file=str(self.pid_file),
            launcher_log=str(launcher_log),
            launcher_log_offset=launcher_log_offset,
            start_script=str(self.start_script),
            repo_root=str(self.repo_root),
        )
        self.pid_file.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        if wait_for_ready:
            effective_timeout, effective_poll_interval = self._resolve_wait_settings(
                env=launcher_env,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            _emit_progress(progress, "Waiting for service startup to complete...")
            self._wait_for_ready(
                pid=record.pid,
                launcher_log=launcher_log,
                offset=launcher_log_offset,
                timeout_seconds=effective_timeout,
                poll_interval_seconds=effective_poll_interval,
                progress=progress,
            )
        return "started", record

    def stop(
        self,
        *,
        timeout_seconds: float = 20.0,
        progress: ProgressFn | None = None,
    ) -> tuple[str, ServiceRecord | None]:
        record = self.running_record()
        if record is None:
            stale = self.load_record()
            self.pid_file.unlink(missing_ok=True)
            return "not_running", stale

        launcher_log = Path(record.launcher_log)
        log_state = LogProgressState(
            offset=launcher_log.stat().st_size if launcher_log.exists() else 0
        )
        _emit_progress(progress, f"Stopping background service pid={record.pid}")
        try:
            os.kill(record.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.pid_file.unlink(missing_ok=True)
            return "stopped", record

        deadline = time.monotonic() + timeout_seconds
        start_time = time.monotonic()
        next_heartbeat_at = start_time
        while time.monotonic() < deadline:
            self._emit_new_log_progress(launcher_log=launcher_log, state=log_state, progress=progress)
            if not _pid_is_running(record.pid):
                _reap_pid_if_child(record.pid)
                self._emit_new_log_progress(launcher_log=launcher_log, state=log_state, progress=progress)
                self.pid_file.unlink(missing_ok=True)
                return "stopped", record
            now = time.monotonic()
            if now >= next_heartbeat_at:
                elapsed_seconds = int(now - start_time)
                _emit_progress(progress, f"Waiting for shutdown... {elapsed_seconds}s elapsed")
                next_heartbeat_at = now + 1.0
            time.sleep(0.2)

        try:
            os.killpg(record.pid, signal.SIGKILL)
            _emit_progress(progress, f"Graceful shutdown timed out; force killing process group {record.pid}")
        except ProcessLookupError:
            self.pid_file.unlink(missing_ok=True)
            return "stopped", record

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self._emit_new_log_progress(launcher_log=launcher_log, state=log_state, progress=progress)
            if not _pid_is_running(record.pid):
                _reap_pid_if_child(record.pid)
                self._emit_new_log_progress(launcher_log=launcher_log, state=log_state, progress=progress)
                self.pid_file.unlink(missing_ok=True)
                return "stopped", record
            time.sleep(0.2)

        raise RuntimeError(f"failed to stop background service pid={record.pid}")

    def status(self, *, clean_stale: bool = False) -> tuple[str, ServiceRecord | None]:
        running = self.running_record()
        if running is not None:
            return "running", running
        stale = self.load_record()
        if clean_stale:
            self.pid_file.unlink(missing_ok=True)
        return "not_running", stale


def _print_record(record: ServiceRecord | None) -> None:
    if record is None:
        return
    print(f"pid: {record.pid}")
    print(f"run_id: {record.run_id}")
    print(f"started_at: {record.started_at}")
    print(f"launcher_log: {record.launcher_log}")
    print(f"pid_file: {record.pid_file}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Background start/stop/status interface for the interactive embedded TP1 stack."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("start", help="Start the stack in the background.")
    stop_parser = subparsers.add_parser("stop", help="Stop the background stack.")
    stop_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="Seconds to wait for graceful shutdown before forcing termination.",
    )
    subparsers.add_parser("status", help="Show background stack status.")

    args = parser.parse_args(argv)
    controller = BackgroundServiceController()

    try:
        if args.command == "start":
            state, record = controller.start(progress=lambda message: print(message, flush=True))
            if state == "already_running":
                print("Service is already running in the background.")
            else:
                print("Service started in the background.")
            _print_record(record)
            return 0
        if args.command == "stop":
            state, record = controller.stop(
                timeout_seconds=args.timeout_seconds,
                progress=lambda message: print(message, flush=True),
            )
            if state == "not_running":
                print("Service is not running.")
            else:
                print("Service stopped.")
            _print_record(record)
            return 0

        state, record = controller.status(clean_stale=True)
        if state == "running":
            print("Service is running.")
            _print_record(record)
            return 0
        print("Service is not running.")
        if record is not None:
            print(f"last_launcher_log: {record.launcher_log}")
        print(f"pid_file: {controller.pid_file}")
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
