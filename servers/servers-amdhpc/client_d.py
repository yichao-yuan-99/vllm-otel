#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SSH tunnel daemon for running client/server on different machines."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from typing import Any
import tomllib

import typer


REPO_ROOT = Path(__file__).resolve().parents[2]
PORT_PROFILES_PATH = REPO_ROOT / "configs" / "port_profiles.toml"
DEFAULT_RUNTIME_DIR = Path.home() / ".cache" / "vllm-otel-apptainer" / "client-d"
DEFAULT_PID_FILE_NAME = "client_d.pid"
DEFAULT_LOG_FILE_NAME = "client_d.log"
DEFAULT_SERVER_PORT = 23971
DEFAULT_VLLM_PORT = 11451
DEFAULT_JAEGER_OTLP_PORT = 4317
DEFAULT_JAEGER_UI_PORT = 16686
REQUIRED_PORT_PROFILE_KEYS = (
    "vllm_port",
    "jaeger_api_port",
    "jaeger_otlp_port",
)
DEFAULT_SSH_OPTIONS = [
    "-o",
    "ExitOnForwardFailure=yes",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=3",
]

app = typer.Typer(add_completion=False, no_args_is_help=True, help="SSH tunnel daemon for vLLM control client")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_payload(*, ok: bool, code: int, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": ok,
        "code": code,
        "message": message,
        "data": data or {},
    }


def _expand_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _runtime_files(runtime_dir: Path | str) -> tuple[Path, Path]:
    runtime_path = _expand_path(runtime_dir)
    return (runtime_path / DEFAULT_PID_FILE_NAME, runtime_path / DEFAULT_LOG_FILE_NAME)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_cmdline(pid: int) -> str:
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return ""
    try:
        raw = cmdline_path.read_bytes()
    except OSError:
        return ""
    parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part]
    return " ".join(parts)


def _load_pid_record(pid_file: Path) -> dict[str, Any] | None:
    if not pid_file.exists():
        return None
    try:
        data = json.loads(pid_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_pid_record(pid_file: Path, payload: dict[str, Any]) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _tail_file(path: Path, *, lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    split_lines = data.splitlines()
    return "\n".join(split_lines[-lines:])


def _parse_port(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    if value <= 0 or value > 65535:
        raise ValueError(f"{key} must be in range 1..65535")
    return value


def _load_port_profile(profile_id: int) -> dict[str, Any]:
    if not PORT_PROFILES_PATH.exists():
        raise FileNotFoundError(f"port profiles file not found: {PORT_PROFILES_PATH}")

    payload = tomllib.loads(PORT_PROFILES_PATH.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"port profiles file is missing [profiles]: {PORT_PROFILES_PATH}")

    profile_key = str(profile_id)
    profile = profiles.get(profile_key)
    if not isinstance(profile, dict):
        raise KeyError(f"unknown port profile id: {profile_id}")

    resolved_ports: dict[str, int] = {}
    for required_key in REQUIRED_PORT_PROFILE_KEYS:
        if required_key not in profile:
            raise ValueError(f"profile {profile_id} missing required key: {required_key}")
        resolved_ports[required_key] = _parse_port(profile[required_key], f"profiles.{profile_key}.{required_key}")

    label = profile.get("label")
    if label is not None and not isinstance(label, str):
        raise ValueError(f"profiles.{profile_key}.label must be a string")

    return {
        "id": profile_id,
        "label": label or profile_key,
        "ports": {
            "vllm_port": resolved_ports["vllm_port"],
            "jaeger_otlp_port": resolved_ports["jaeger_otlp_port"],
            "jaeger_ui_port": resolved_ports["jaeger_api_port"],
        },
    }


def _build_ssh_command(
    *,
    ssh_target: str,
    server_port: int,
    local_vllm_port: int,
    local_jaeger_otlp_port: int,
    local_jaeger_ui_port: int,
    remote_vllm_port: int,
    remote_jaeger_otlp_port: int,
    remote_jaeger_ui_port: int,
    ssh_options: list[str],
) -> list[str]:
    cmd: list[str] = ["ssh", "-N", "-T", *ssh_options]
    cmd.extend(["-L", f"{server_port}:127.0.0.1:{server_port}"])
    cmd.extend(["-L", f"{local_vllm_port}:127.0.0.1:{remote_vllm_port}"])
    cmd.extend(["-L", f"{local_jaeger_otlp_port}:127.0.0.1:{remote_jaeger_otlp_port}"])
    cmd.extend(["-L", f"{local_jaeger_ui_port}:127.0.0.1:{remote_jaeger_ui_port}"])
    cmd.append(ssh_target)
    return cmd


def _record_matches_running_process(record: dict[str, Any]) -> bool:
    raw_pid = record.get("pid")
    if isinstance(raw_pid, bool):
        return False
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return False
    if not _pid_is_running(pid):
        return False
    cmdline = _process_cmdline(pid)
    if not cmdline:
        return True
    ssh_target = record.get("ssh_target")
    if isinstance(ssh_target, str) and ssh_target:
        return "ssh" in cmdline and ssh_target in cmdline
    return "ssh" in cmdline


def start_client_d(
    *,
    ssh_target: str,
    port_profile_id: int,
    runtime_dir: Path | str = DEFAULT_RUNTIME_DIR,
    server_port: int = DEFAULT_SERVER_PORT,
    ssh_options: list[str] | None = None,
) -> dict[str, Any]:
    if not ssh_target:
        return _json_payload(ok=False, code=301, message="ssh_target must be non-empty")
    try:
        server_port = _parse_port(server_port, "server_port")
    except ValueError as exc:
        return _json_payload(ok=False, code=302, message=str(exc))

    try:
        port_profile = _load_port_profile(port_profile_id)
    except Exception as exc:  # noqa: BLE001
        return _json_payload(
            ok=False,
            code=305,
            message=f"failed to load port profile {port_profile_id}: {exc}",
            data={
                "port_profile_id": port_profile_id,
                "port_profiles_path": str(PORT_PROFILES_PATH),
            },
        )

    local_ports = dict(port_profile["ports"])

    pid_file, log_file = _runtime_files(runtime_dir)
    runtime_path = pid_file.parent
    runtime_path.mkdir(parents=True, exist_ok=True)
    active_record = _load_pid_record(pid_file)
    if isinstance(active_record, dict):
        if _record_matches_running_process(active_record):
            return _json_payload(
                ok=False,
                code=303,
                message="client-d is already running",
                data={"pid_file": str(pid_file), "record": active_record},
            )
        pid_file.unlink(missing_ok=True)

    merged_ssh_options = list(ssh_options) if ssh_options else list(DEFAULT_SSH_OPTIONS)
    ssh_cmd = _build_ssh_command(
        ssh_target=ssh_target,
        server_port=server_port,
        local_vllm_port=local_ports["vllm_port"],
        local_jaeger_otlp_port=local_ports["jaeger_otlp_port"],
        local_jaeger_ui_port=local_ports["jaeger_ui_port"],
        remote_vllm_port=DEFAULT_VLLM_PORT,
        remote_jaeger_otlp_port=DEFAULT_JAEGER_OTLP_PORT,
        remote_jaeger_ui_port=DEFAULT_JAEGER_UI_PORT,
        ssh_options=merged_ssh_options,
    )

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now_iso()}] starting client-d: {shlex.join(ssh_cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            tail = _tail_file(log_file)
            return _json_payload(
                ok=False,
                code=304,
                message=f"failed to start client-d tunnel (ssh exited with code {rc})",
                data={
                    "pid_file": str(pid_file),
                    "log_file": str(log_file),
                    "ssh_command": ssh_cmd,
                    "log_tail": tail,
                },
            )
        time.sleep(0.1)

    record = {
        "pid": proc.pid,
        "ssh_target": ssh_target,
        "started_at": _utc_now_iso(),
        "port_profile_id": port_profile_id,
        "port_profile": {
            "id": port_profile["id"],
            "label": port_profile["label"],
            "config_path": str(PORT_PROFILES_PATH),
        },
        "ssh_command": ssh_cmd,
        "ports": {
            "server_port": server_port,
            "vllm_port": local_ports["vllm_port"],
            "jaeger_otlp_port": local_ports["jaeger_otlp_port"],
            "jaeger_ui_port": local_ports["jaeger_ui_port"],
        },
        "remote_ports": {
            "server_port": server_port,
            "vllm_port": DEFAULT_VLLM_PORT,
            "jaeger_otlp_port": DEFAULT_JAEGER_OTLP_PORT,
            "jaeger_ui_port": DEFAULT_JAEGER_UI_PORT,
        },
        "log_file": str(log_file),
        "pid_file": str(pid_file),
    }
    _write_pid_record(pid_file, record)
    return _json_payload(
        ok=True,
        code=0,
        message="client-d started",
        data={
            **record,
            "server_url": f"http://127.0.0.1:{server_port}",
            "vllm_url": f"http://127.0.0.1:{local_ports['vllm_port']}",
            "jaeger_ui_url": f"http://127.0.0.1:{local_ports['jaeger_ui_port']}",
            "jaeger_otlp_endpoint": f"grpc://127.0.0.1:{local_ports['jaeger_otlp_port']}",
        },
    )


def stop_client_d(
    *,
    runtime_dir: Path | str = DEFAULT_RUNTIME_DIR,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    pid_file, log_file = _runtime_files(runtime_dir)
    record = _load_pid_record(pid_file)
    if not isinstance(record, dict):
        pid_file.unlink(missing_ok=True)
        return _json_payload(
            ok=True,
            code=0,
            message="client-d is not running",
            data={"pid_file": str(pid_file), "log_file": str(log_file)},
        )

    raw_pid = record.get("pid")
    if isinstance(raw_pid, bool):
        raw_pid = None
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        pid_file.unlink(missing_ok=True)
        return _json_payload(
            ok=True,
            code=0,
            message="removed invalid client-d pid record",
            data={"pid_file": str(pid_file), "record": record},
        )

    if not _pid_is_running(pid):
        pid_file.unlink(missing_ok=True)
        return _json_payload(
            ok=True,
            code=0,
            message="client-d is not running (stale pid file removed)",
            data={"pid": pid, "pid_file": str(pid_file)},
        )

    forced = False
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            break
        time.sleep(0.1)

    if _pid_is_running(pid):
        forced = True
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    pid_file.unlink(missing_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now_iso()}] stopped client-d pid={pid} forced={forced}\n")

    return _json_payload(
        ok=True,
        code=0,
        message="client-d stopped",
        data={
            "pid": pid,
            "forced": forced,
            "pid_file": str(pid_file),
            "log_file": str(log_file),
        },
    )


def client_d_status(*, runtime_dir: Path | str = DEFAULT_RUNTIME_DIR) -> dict[str, Any]:
    pid_file, log_file = _runtime_files(runtime_dir)
    record = _load_pid_record(pid_file)
    if not isinstance(record, dict):
        return _json_payload(
            ok=True,
            code=0,
            message="client-d is not running",
            data={"running": False, "pid_file": str(pid_file), "log_file": str(log_file)},
        )

    raw_pid = record.get("pid")
    if isinstance(raw_pid, bool):
        raw_pid = None
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return _json_payload(
            ok=True,
            code=0,
            message="client-d pid file is invalid",
            data={"running": False, "pid_file": str(pid_file), "record": record},
        )

    running = _record_matches_running_process(record)
    return _json_payload(
        ok=True,
        code=0,
        message="client-d status",
        data={
            "running": running,
            "pid": pid,
            "pid_file": str(pid_file),
            "log_file": str(log_file),
            "record": record,
            "process_cmdline": _process_cmdline(pid) if running else "",
        },
    )


def _print_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _require_ok(payload: dict[str, Any]) -> None:
    if payload.get("ok"):
        return
    _print_json(payload)
    raise typer.Exit(code=1)


@app.command()
def start(
    ssh_target: str = typer.Option(..., "--ssh-target", "-t", help="SSH target, same as in `ssh <target>`."),
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-p",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    runtime_dir: Path = typer.Option(
        DEFAULT_RUNTIME_DIR,
        "--runtime-dir",
        help="Directory for client-d pid and log files.",
    ),
    server_port: int = typer.Option(DEFAULT_SERVER_PORT, "--server-port", help="Control server port."),
    ssh_option: list[str] = typer.Option(
        [],
        "--ssh-option",
        help="Additional raw ssh option token. Repeat to pass multiple tokens.",
    ),
) -> None:
    """Start local SSH tunnels to the remote HPC login node."""
    payload = start_client_d(
        ssh_target=ssh_target,
        port_profile_id=port_profile,
        runtime_dir=runtime_dir,
        server_port=server_port,
        ssh_options=ssh_option or None,
    )
    _require_ok(payload)
    _print_json(payload)


@app.command()
def stop(
    runtime_dir: Path = typer.Option(
        DEFAULT_RUNTIME_DIR,
        "--runtime-dir",
        help="Directory for client-d pid and log files.",
    ),
    timeout_seconds: float = typer.Option(
        10.0,
        "--timeout-seconds",
        help="Seconds to wait before forcing kill.",
    ),
) -> None:
    """Stop the local SSH tunnel daemon."""
    payload = stop_client_d(runtime_dir=runtime_dir, timeout_seconds=timeout_seconds)
    _require_ok(payload)
    _print_json(payload)


@app.command(name="status")
def status_cmd(
    runtime_dir: Path = typer.Option(
        DEFAULT_RUNTIME_DIR,
        "--runtime-dir",
        help="Directory for client-d pid and log files.",
    ),
) -> None:
    """Show client-d status."""
    payload = client_d_status(runtime_dir=runtime_dir)
    _require_ok(payload)
    _print_json(payload)


if __name__ == "__main__":
    app()
