#!/usr/bin/env python3
"""Minimal SSH port-forward daemon for hardcoded AMD HPC vLLM jobs."""

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
from urllib import error as urlerror
from urllib import request as urlrequest

import typer

try:
    from .resolve_vllm_port import resolve_vllm_port
except ImportError:  # pragma: no cover
    from resolve_vllm_port import resolve_vllm_port  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT_PROFILES_PATH = REPO_ROOT / "configs" / "port_profiles.toml"
DEFAULT_RUNTIME_DIR = Path.home() / ".cache" / "vllm-otel-apptainer" / "hardcoded-port-forward-d"
DEFAULT_PID_FILE_PREFIX = "hardcoded_port_forward_d"
DEFAULT_LOG_FILE_PREFIX = "hardcoded_port_forward_d"
DEFAULT_SSH_OPTIONS = [
    "-o",
    "ExitOnForwardFailure=yes",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=3",
]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Minimal SSH tunnel daemon for hardcoded AMD HPC vLLM jobs.",
)


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


def _runtime_files(runtime_dir: Path | str, *, port_profile_id: int) -> tuple[Path, Path]:
    runtime_path = _expand_path(runtime_dir)
    suffix = f".{port_profile_id}"
    return (
        runtime_path / f"{DEFAULT_PID_FILE_PREFIX}{suffix}.pid",
        runtime_path / f"{DEFAULT_LOG_FILE_PREFIX}{suffix}.log",
    )


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


def _http_json_get(url: str, *, timeout_seconds: float) -> tuple[bool, dict[str, Any]]:
    request = urlrequest.Request(url, method="GET")
    started = time.monotonic()
    try:
        with urlrequest.urlopen(request, timeout=max(timeout_seconds, 0.1)) as response:
            text = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(text) if text else {}
            return True, {
                "status_code": int(response.status),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "body": parsed,
            }
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        parsed_error: Any = text
        if text:
            try:
                parsed_error = json.loads(text)
            except json.JSONDecodeError:
                parsed_error = text
        return False, {
            "status_code": int(exc.code),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": parsed_error,
        }
    except Exception as exc:  # pragma: no cover - network boundary
        return False, {
            "status_code": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": str(exc),
        }


def _extract_model_name(models_response: dict[str, Any]) -> str:
    body = models_response.get("body")
    if not isinstance(body, dict):
        raise ValueError("vLLM /v1/models response body is not an object")
    data = body.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("vLLM /v1/models returned no models")
    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("vLLM /v1/models item is not an object")
    model_name = first.get("id")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("vLLM /v1/models first item has invalid id")
    return model_name


def _build_ssh_command(
    *,
    ssh_target: str,
    local_vllm_port: int,
    remote_vllm_port: int,
    ssh_options: list[str],
) -> list[str]:
    cmd: list[str] = ["ssh", "-N", "-T", *ssh_options]
    cmd.extend(["-L", f"0.0.0.0:{local_vllm_port}:127.0.0.1:{remote_vllm_port}"])
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


def _resolve_profile_vllm_port(*, port_profiles_path: Path, port_profile_id: int) -> int:
    return resolve_vllm_port(config_path=port_profiles_path, profile_id=str(port_profile_id))


def start_port_forward_daemon(
    *,
    ssh_target: str,
    port_profile_id: int,
    runtime_dir: Path | str = DEFAULT_RUNTIME_DIR,
    port_profiles_path: Path = DEFAULT_PORT_PROFILES_PATH,
    ssh_options: list[str] | None = None,
) -> dict[str, Any]:
    if not ssh_target:
        return _json_payload(ok=False, code=301, message="ssh_target must be non-empty")

    try:
        vllm_port = _resolve_profile_vllm_port(
            port_profiles_path=_expand_path(port_profiles_path),
            port_profile_id=port_profile_id,
        )
    except Exception as exc:  # noqa: BLE001
        return _json_payload(
            ok=False,
            code=302,
            message=f"failed to resolve port profile {port_profile_id}: {exc}",
            data={
                "port_profile_id": port_profile_id,
                "port_profiles_path": str(_expand_path(port_profiles_path)),
            },
        )

    pid_file, log_file = _runtime_files(runtime_dir, port_profile_id=port_profile_id)
    runtime_path = pid_file.parent
    runtime_path.mkdir(parents=True, exist_ok=True)
    active_record = _load_pid_record(pid_file)
    if isinstance(active_record, dict):
        if _record_matches_running_process(active_record):
            return _json_payload(
                ok=False,
                code=303,
                message="hardcoded port-forward daemon is already running",
                data={"pid_file": str(pid_file), "record": active_record},
            )
        pid_file.unlink(missing_ok=True)

    merged_ssh_options = list(ssh_options) if ssh_options else list(DEFAULT_SSH_OPTIONS)
    ssh_cmd = _build_ssh_command(
        ssh_target=ssh_target,
        local_vllm_port=vllm_port,
        remote_vllm_port=vllm_port,
        ssh_options=merged_ssh_options,
    )

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now_iso()}] starting hardcoded port-forward-d: {shlex.join(ssh_cmd)}\n")
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
                message=f"failed to start hardcoded port-forward daemon (ssh exited with code {rc})",
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
        "ssh_command": ssh_cmd,
        "ports": {
            "vllm_port": vllm_port,
        },
        "remote_ports": {
            "vllm_port": vllm_port,
        },
        "log_file": str(log_file),
        "pid_file": str(pid_file),
        "port_profiles_path": str(_expand_path(port_profiles_path)),
    }
    _write_pid_record(pid_file, record)
    return _json_payload(
        ok=True,
        code=0,
        message="hardcoded port-forward daemon started",
        data={
            **record,
            "vllm_url": f"http://127.0.0.1:{vllm_port}",
        },
    )


def stop_port_forward_daemon(
    *,
    port_profile_id: int,
    runtime_dir: Path | str = DEFAULT_RUNTIME_DIR,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    pid_file, log_file = _runtime_files(runtime_dir, port_profile_id=port_profile_id)
    record = _load_pid_record(pid_file)
    if not isinstance(record, dict):
        pid_file.unlink(missing_ok=True)
        return _json_payload(
            ok=True,
            code=0,
            message=f"hardcoded port-forward daemon is not running for port profile {port_profile_id}",
            data={"port_profile_id": port_profile_id, "pid_file": str(pid_file), "log_file": str(log_file)},
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
            message="removed invalid hardcoded port-forward daemon pid record",
            data={"port_profile_id": port_profile_id, "pid_file": str(pid_file), "record": record},
        )

    if not _pid_is_running(pid):
        pid_file.unlink(missing_ok=True)
        return _json_payload(
            ok=True,
            code=0,
            message=(
                "hardcoded port-forward daemon is not running for "
                f"port profile {port_profile_id} (stale pid file removed)"
            ),
            data={"port_profile_id": port_profile_id, "pid": pid, "pid_file": str(pid_file)},
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
        handle.write(f"[{_utc_now_iso()}] stopped hardcoded port-forward-d pid={pid} forced={forced}\n")

    return _json_payload(
        ok=True,
        code=0,
        message="hardcoded port-forward daemon stopped",
        data={
            "port_profile_id": port_profile_id,
            "pid": pid,
            "forced": forced,
            "pid_file": str(pid_file),
            "log_file": str(log_file),
        },
    )


def port_forward_daemon_status(
    *,
    port_profile_id: int,
    runtime_dir: Path | str = DEFAULT_RUNTIME_DIR,
) -> dict[str, Any]:
    pid_file, log_file = _runtime_files(runtime_dir, port_profile_id=port_profile_id)
    record = _load_pid_record(pid_file)
    if not isinstance(record, dict):
        return _json_payload(
            ok=True,
            code=0,
            message=f"hardcoded port-forward daemon is not running for port profile {port_profile_id}",
            data={
                "running": False,
                "port_profile_id": port_profile_id,
                "pid_file": str(pid_file),
                "log_file": str(log_file),
            },
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
            message="hardcoded port-forward daemon pid file is invalid",
            data={
                "running": False,
                "port_profile_id": port_profile_id,
                "pid_file": str(pid_file),
                "record": record,
            },
        )

    running = _record_matches_running_process(record)
    return _json_payload(
        ok=True,
        code=0,
        message="hardcoded port-forward daemon status",
        data={
            "running": running,
            "port_profile_id": port_profile_id,
            "pid": pid,
            "pid_file": str(pid_file),
            "log_file": str(log_file),
            "record": record,
            "process_cmdline": _process_cmdline(pid) if running else "",
        },
    )


def check_vllm_health(
    *,
    port_profile_id: int,
    runtime_dir: Path | str = DEFAULT_RUNTIME_DIR,
    port_profiles_path: Path = DEFAULT_PORT_PROFILES_PATH,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    try:
        vllm_port = _resolve_profile_vllm_port(
            port_profiles_path=_expand_path(port_profiles_path),
            port_profile_id=port_profile_id,
        )
    except Exception as exc:  # noqa: BLE001
        return _json_payload(
            ok=False,
            code=305,
            message=f"failed to resolve port profile {port_profile_id}: {exc}",
            data={
                "healthy": False,
                "port_profile_id": port_profile_id,
                "port_profiles_path": str(_expand_path(port_profiles_path)),
            },
        )

    status_payload = port_forward_daemon_status(
        port_profile_id=port_profile_id,
        runtime_dir=runtime_dir,
    )
    status_data = status_payload.get("data")
    daemon_running = None
    daemon_pid = None
    if isinstance(status_data, dict):
        daemon_running = status_data.get("running")
        daemon_pid = status_data.get("pid")

    vllm_url = f"http://127.0.0.1:{vllm_port}/v1/models"
    ok, probe = _http_json_get(vllm_url, timeout_seconds=timeout_seconds)
    if not ok:
        return _json_payload(
            ok=False,
            code=306,
            message="vLLM endpoint health check failed",
            data={
                "healthy": False,
                "port_profile_id": port_profile_id,
                "vllm_port": vllm_port,
                "vllm_url": vllm_url,
                "daemon_running": daemon_running,
                "daemon_pid": daemon_pid,
                "probe": probe,
            },
        )

    try:
        model_name = _extract_model_name(probe)
    except ValueError as exc:
        return _json_payload(
            ok=False,
            code=307,
            message=f"vLLM endpoint returned an unexpected /v1/models payload: {exc}",
            data={
                "healthy": False,
                "port_profile_id": port_profile_id,
                "vllm_port": vllm_port,
                "vllm_url": vllm_url,
                "daemon_running": daemon_running,
                "daemon_pid": daemon_pid,
                "probe": probe,
            },
        )

    return _json_payload(
        ok=True,
        code=0,
        message="vLLM endpoint is healthy",
        data={
            "healthy": True,
            "port_profile_id": port_profile_id,
            "vllm_port": vllm_port,
            "vllm_url": vllm_url,
            "daemon_running": daemon_running,
            "daemon_pid": daemon_pid,
            "model_name": model_name,
            "probe": probe,
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
    ssh_target: str = typer.Option(
        "amd-hpc",
        "--ssh-target",
        "--remote-host",
        "-r",
        help="SSH target, same as in `ssh <target>`.",
    ),
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    runtime_dir: Path = typer.Option(
        DEFAULT_RUNTIME_DIR,
        "--runtime-dir",
        help="Directory for pid and log files.",
    ),
    port_profiles: Path = typer.Option(
        DEFAULT_PORT_PROFILES_PATH,
        "--port-profiles",
        help="Path to configs/port_profiles.toml.",
    ),
    ssh_option: list[str] = typer.Option(
        [],
        "--ssh-option",
        help="Additional raw ssh option token. Repeat to pass multiple tokens.",
    ),
) -> None:
    """Start a local SSH tunnel for one hardcoded vLLM port profile."""
    payload = start_port_forward_daemon(
        ssh_target=ssh_target,
        port_profile_id=port_profile,
        runtime_dir=runtime_dir,
        port_profiles_path=port_profiles,
        ssh_options=ssh_option or None,
    )
    _require_ok(payload)
    _print_json(payload)


@app.command()
def stop(
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    runtime_dir: Path = typer.Option(
        DEFAULT_RUNTIME_DIR,
        "--runtime-dir",
        help="Directory for pid and log files.",
    ),
    timeout_seconds: float = typer.Option(
        10.0,
        "--timeout-seconds",
        help="Seconds to wait before forcing kill.",
    ),
) -> None:
    """Stop the local hardcoded SSH tunnel daemon."""
    payload = stop_port_forward_daemon(
        port_profile_id=port_profile,
        runtime_dir=runtime_dir,
        timeout_seconds=timeout_seconds,
    )
    _require_ok(payload)
    _print_json(payload)


@app.command(name="status")
def status_cmd(
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    runtime_dir: Path = typer.Option(
        DEFAULT_RUNTIME_DIR,
        "--runtime-dir",
        help="Directory for pid and log files.",
    ),
) -> None:
    """Show hardcoded port-forward daemon status."""
    payload = port_forward_daemon_status(
        port_profile_id=port_profile,
        runtime_dir=runtime_dir,
    )
    _require_ok(payload)
    _print_json(payload)


@app.command(name="check-health")
def check_health_cmd(
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    runtime_dir: Path = typer.Option(
        DEFAULT_RUNTIME_DIR,
        "--runtime-dir",
        help="Directory for pid and log files.",
    ),
    port_profiles: Path = typer.Option(
        DEFAULT_PORT_PROFILES_PATH,
        "--port-profiles",
        help="Path to configs/port_profiles.toml.",
    ),
    timeout_seconds: float = typer.Option(
        5.0,
        "--timeout-seconds",
        help="HTTP timeout for the local forwarded /v1/models probe.",
    ),
) -> None:
    """Check whether the local forwarded vLLM endpoint is healthy."""
    payload = check_vllm_health(
        port_profile_id=port_profile,
        runtime_dir=runtime_dir,
        port_profiles_path=port_profiles,
        timeout_seconds=timeout_seconds,
    )
    _require_ok(payload)
    _print_json(payload)


if __name__ == "__main__":
    app()
