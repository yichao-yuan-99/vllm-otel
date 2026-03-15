#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Typer CLI frontend for the Apptainer HPC control server."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import typer

try:
    from .client_d import (
        DEFAULT_RUNTIME_DIR as CLIENT_D_RUNTIME_DIR,
        DEFAULT_SERVER_PORT,
        client_d_status,
        resolve_client_d_server_url,
        start_client_d,
        stop_client_d,
    )
except ImportError:  # pragma: no cover
    from client_d import (  # type: ignore[no-redef]
        DEFAULT_RUNTIME_DIR as CLIENT_D_RUNTIME_DIR,
        DEFAULT_SERVER_PORT,
        client_d_status,
        resolve_client_d_server_url,
        start_client_d,
        stop_client_d,
    )

try:
    from .port_profiles import load_port_profiles
except ImportError:  # pragma: no cover
    from port_profiles import load_port_profiles  # type: ignore[no-redef]


app = typer.Typer(add_completion=False, no_args_is_help=True, help="vLLM HPC control client")

START_BLOCKING_TIMEOUT_SECONDS = 24.0 * 60.0 * 60.0
START_CLEANUP_WAIT_FOR_SUBMISSION_SECONDS = 120.0
STOP_BLOCKING_TIMEOUT_SECONDS = 10.0 * 60.0
GATEWAY_WAIT_UP_TIMEOUT_SECONDS = 900
GATEWAY_WAIT_UP_POLL_INTERVAL_SECONDS = 2.0
GATEWAY_STARTUP_PROBE_SECONDS = 3.0

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATEWAY_CONFIG_PATH = REPO_ROOT / "gateway" / "config.toml"
DEFAULT_GATEWAY_CONFIG_EXAMPLE_PATH = REPO_ROOT / "gateway" / "config.example.toml"
DEFAULT_GATEWAY_VENV_DIR = REPO_ROOT / ".venv"
DEFAULT_GATEWAY_HOST = "0.0.0.0"
DEFAULT_GATEWAY_PID_FILE_PREFIX = "gateway_d"
DEFAULT_GATEWAY_LOG_FILE_PREFIX = "gateway_d"
REMOTE_HOST_OPTION_HELP = (
    "SSH target, same as in `ssh <target>`. "
    "`--remote-host`/`-r` is an alias for this option."
)


def _remote_host_option(default: str = "amd-hpc") -> Any:
    return typer.Option(
        default,
        "--ssh-target",
        "--remote-host",
        "-t",
        "-r",
        help=REMOTE_HOST_OPTION_HELP,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _expand_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _gateway_runtime_files(runtime_dir: Path | str, *, port_profile_id: int) -> tuple[Path, Path]:
    runtime_path = _expand_path(runtime_dir)
    suffix = f".{port_profile_id}"
    return (
        runtime_path / f"{DEFAULT_GATEWAY_PID_FILE_PREFIX}{suffix}.pid",
        runtime_path / f"{DEFAULT_GATEWAY_LOG_FILE_PREFIX}{suffix}.log",
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


def _resolve_gateway_config_path(config_path: Path | None = None) -> Path:
    if config_path is not None:
        return _expand_path(config_path)
    if DEFAULT_GATEWAY_CONFIG_PATH.exists():
        return _expand_path(DEFAULT_GATEWAY_CONFIG_PATH)
    return _expand_path(DEFAULT_GATEWAY_CONFIG_EXAMPLE_PATH)


def _build_gateway_command(
    *,
    port_profile_id: int,
    config_path: Path,
    venv_dir: Path,
    host: str,
    skip_install: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "gateway",
        "start",
        "--config",
        str(config_path),
        "--port-profile-id",
        str(port_profile_id),
        "--host",
        host,
        "--venv-dir",
        str(venv_dir),
    ]
    if skip_install:
        command.append("--skip-install")
    return command


def _gateway_record_matches_running_process(record: dict[str, Any]) -> bool:
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
    normalized = cmdline.lower()
    if "gateway" not in normalized or " start " not in f" {normalized} ":
        return False
    profile_id = record.get("port_profile_id")
    if isinstance(profile_id, int):
        token = f"--port-profile-id {profile_id}"
        token_eq = f"--port-profile-id={profile_id}"
        if token not in cmdline and token_eq not in cmdline:
            return False
    return True


def start_gateway_daemon(
    *,
    port_profile_id: int,
    runtime_dir: Path | str = CLIENT_D_RUNTIME_DIR,
    config_path: Path | None = None,
    venv_dir: Path | None = None,
    host: str = DEFAULT_GATEWAY_HOST,
    skip_install: bool = True,
) -> dict[str, Any]:
    resolved_config_path = _resolve_gateway_config_path(config_path)
    resolved_venv_dir = _expand_path(venv_dir or DEFAULT_GATEWAY_VENV_DIR)

    pid_file, log_file = _gateway_runtime_files(runtime_dir, port_profile_id=port_profile_id)
    runtime_path = pid_file.parent
    runtime_path.mkdir(parents=True, exist_ok=True)

    active_record = _load_pid_record(pid_file)
    if isinstance(active_record, dict):
        if _gateway_record_matches_running_process(active_record):
            return {
                "ok": True,
                "code": 0,
                "message": "gateway already running",
                "data": {"pid_file": str(pid_file), "record": active_record},
            }
        pid_file.unlink(missing_ok=True)

    gateway_cmd = _build_gateway_command(
        port_profile_id=port_profile_id,
        config_path=resolved_config_path,
        venv_dir=resolved_venv_dir,
        host=host,
        skip_install=skip_install,
    )
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now_iso()}] starting gateway: {shlex.join(gateway_cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            gateway_cmd,
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    deadline = time.monotonic() + GATEWAY_STARTUP_PROBE_SECONDS
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            tail = _tail_file(log_file)
            return {
                "ok": False,
                "code": 401,
                "message": f"failed to start gateway daemon (exit={rc})",
                "data": {
                    "pid_file": str(pid_file),
                    "log_file": str(log_file),
                    "command": gateway_cmd,
                    "log_tail": tail,
                },
            }
        time.sleep(0.1)

    record = {
        "pid": proc.pid,
        "started_at": _utc_now_iso(),
        "port_profile_id": port_profile_id,
        "config_path": str(resolved_config_path),
        "venv_dir": str(resolved_venv_dir),
        "host": host,
        "skip_install": skip_install,
        "command": gateway_cmd,
        "pid_file": str(pid_file),
        "log_file": str(log_file),
    }
    _write_pid_record(pid_file, record)
    return {
        "ok": True,
        "code": 0,
        "message": "gateway started",
        "data": record,
    }


def stop_gateway_daemon(
    *,
    port_profile_id: int,
    runtime_dir: Path | str = CLIENT_D_RUNTIME_DIR,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    pid_file, log_file = _gateway_runtime_files(runtime_dir, port_profile_id=port_profile_id)
    record = _load_pid_record(pid_file)
    if not isinstance(record, dict):
        pid_file.unlink(missing_ok=True)
        return {
            "ok": True,
            "code": 0,
            "message": f"gateway is not running for port profile {port_profile_id}",
            "data": {"port_profile_id": port_profile_id, "pid_file": str(pid_file), "log_file": str(log_file)},
        }

    raw_pid = record.get("pid")
    if isinstance(raw_pid, bool):
        raw_pid = None
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        pid_file.unlink(missing_ok=True)
        return {
            "ok": True,
            "code": 0,
            "message": "removed invalid gateway pid record",
            "data": {"port_profile_id": port_profile_id, "pid_file": str(pid_file), "record": record},
        }

    if not _gateway_record_matches_running_process(record):
        cmdline = _process_cmdline(pid) if _pid_is_running(pid) else ""
        pid_file.unlink(missing_ok=True)
        return {
            "ok": True,
            "code": 0,
            "message": f"gateway is not running for port profile {port_profile_id} (stale pid file removed)",
            "data": {
                "port_profile_id": port_profile_id,
                "pid": pid,
                "pid_file": str(pid_file),
                "process_cmdline": cmdline,
            },
        }

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
        handle.write(f"[{_utc_now_iso()}] stopped gateway pid={pid} forced={forced}\n")

    return {
        "ok": True,
        "code": 0,
        "message": "gateway stopped",
        "data": {
            "port_profile_id": port_profile_id,
            "pid": pid,
            "forced": forced,
            "pid_file": str(pid_file),
            "log_file": str(log_file),
        },
    }


def _post_json(server_url: str, endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = f"{server_url.rstrip('/')}/{endpoint.lstrip('/')}"
    encoded = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        method="POST",
        data=encoded,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {"ok": False, "message": "empty response", "code": 1}
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                return parsed
            return {"ok": False, "message": "response was not a JSON object", "code": 1}
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {
                "ok": False,
                "code": exc.code,
                "message": f"HTTP {exc.code}: {body.strip() or exc.reason}",
            }
        if isinstance(parsed, dict):
            if "ok" not in parsed:
                parsed["ok"] = False
            if "code" not in parsed:
                parsed["code"] = exc.code
            return parsed
        return {
            "ok": False,
            "code": exc.code,
            "message": f"HTTP {exc.code}: {body.strip() or exc.reason}",
        }
    except urlerror.URLError as exc:
        return {
            "ok": False,
            "code": 1,
            "message": f"failed to reach server: {exc.reason}",
        }


def _print_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _require_ok(payload: dict[str, Any]) -> None:
    if payload.get("ok"):
        return
    _print_json(payload)
    raise typer.Exit(code=1)


def _is_ok(payload: dict[str, Any]) -> bool:
    return bool(payload.get("ok"))


def _clientd_runtime_dir(ctx: typer.Context) -> Path:
    obj = ctx.obj or {}
    raw = obj.get("clientd_runtime_dir", CLIENT_D_RUNTIME_DIR)
    if isinstance(raw, Path):
        return raw
    return Path(raw)


def _has_server_url_override(ctx: typer.Context) -> bool:
    obj = ctx.obj or {}
    raw = obj.get("server_url")
    return isinstance(raw, str) and bool(raw.strip())


def _parse_profile_list(value: str) -> list[int]:
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if not tokens:
        raise ValueError("profile list cannot be empty")
    parsed: list[int] = []
    for token in tokens:
        try:
            profile_id = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid profile id '{token}' in profile list") from exc
        parsed.append(profile_id)
    if len(set(parsed)) != len(parsed):
        raise ValueError("profile list cannot contain duplicate ids")
    return parsed


def _parse_extra_env_list(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_item in values:
        item = raw_item.strip()
        if not item:
            raise ValueError("env value cannot be empty; use KEY=VALUE")
        if "=" not in item:
            raise ValueError(f"env value '{item}' must be in KEY=VALUE format")
        key_raw, value = item.split("=", 1)
        key = key_raw.strip()
        if not key:
            raise ValueError(f"env value '{item}' has an empty key")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ValueError(
                f"env key '{key}' is invalid; expected [A-Za-z_][A-Za-z0-9_]*"
            )
        if key in parsed:
            raise ValueError(f"duplicate env key '{key}'")
        parsed[key] = value
    return parsed


def _sorted_unique_profile_ids(profile_ids: list[int]) -> list[int]:
    return sorted(set(int(value) for value in profile_ids))


def _pick_control_profile_for_group(ctx: typer.Context) -> int:
    runtime_dir = _clientd_runtime_dir(ctx)
    for profile_id in sorted(load_port_profiles().keys()):
        payload = client_d_status(port_profile_id=profile_id, runtime_dir=runtime_dir)
        data = payload.get("data")
        running = bool(data.get("running")) if isinstance(data, dict) else False
        if running:
            return profile_id
    raise RuntimeError(
        "no running client-d tunnel found to reach the control server; "
        "start a tunnel first or pass --server-url"
    )


def _run_group_command_with_timeout(
    ctx: typer.Context,
    endpoint: str,
    *,
    payload: dict[str, Any],
    timeout_seconds: float | None,
    preferred_control_profile: int | None = None,
) -> dict[str, Any]:
    request_payload = dict(payload)
    if not _has_server_url_override(ctx):
        control_profile = (
            preferred_control_profile
            if preferred_control_profile is not None
            else _pick_control_profile_for_group(ctx)
        )
        request_payload.setdefault("port_profile", control_profile)
    return _run_command_with_timeout(
        ctx,
        endpoint,
        payload=request_payload,
        timeout_seconds=timeout_seconds,
    )


def _extract_profile_ids_from_group_status(payload: dict[str, Any]) -> list[int]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    profiles_raw = data.get("profiles")
    if not isinstance(profiles_raw, list):
        return []

    profile_ids: list[int] = []
    for item in profiles_raw:
        if not isinstance(item, dict):
            continue
        profile_id_raw = item.get("port_profile")
        if isinstance(profile_id_raw, int) and not isinstance(profile_id_raw, bool):
            profile_ids.append(profile_id_raw)
    return _sorted_unique_profile_ids(profile_ids)


def _stop_local_profile_daemons(
    ctx: typer.Context,
    *,
    profile_ids: list[int],
    clientd_timeout_seconds: float,
) -> tuple[dict[str, Any], list[int]]:
    results: dict[str, Any] = {}
    failed_profiles: list[int] = []
    for profile_id in _sorted_unique_profile_ids(profile_ids):
        gateway_payload = _stop_gateway_for_profile(
            ctx,
            port_profile=profile_id,
            timeout_seconds=clientd_timeout_seconds,
        )
        clientd_payload = _stop_clientd_for_profile(
            ctx,
            port_profile=profile_id,
            timeout_seconds=clientd_timeout_seconds,
        )
        ok = _is_ok(gateway_payload) and _is_ok(clientd_payload)
        if not ok:
            failed_profiles.append(profile_id)
        results[str(profile_id)] = {
            "ok": ok,
            "gateway": gateway_payload,
            "clientd": clientd_payload,
        }
    return results, failed_profiles


def _gateway_status_for_profile(
    ctx: typer.Context,
    *,
    port_profile: int,
) -> dict[str, Any]:
    runtime_dir = _clientd_runtime_dir(ctx)
    pid_file, log_file = _gateway_runtime_files(runtime_dir, port_profile_id=port_profile)
    record = _load_pid_record(pid_file)
    if not isinstance(record, dict):
        return {
            "running": False,
            "port_profile": port_profile,
            "pid_file": str(pid_file),
            "log_file": str(log_file),
            "record": None,
        }

    raw_pid = record.get("pid")
    if isinstance(raw_pid, bool):
        raw_pid = None
    pid: int | None
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        pid = None

    running = _gateway_record_matches_running_process(record)
    return {
        "running": running,
        "port_profile": port_profile,
        "pid": pid,
        "pid_file": str(pid_file),
        "log_file": str(log_file),
        "record": record,
        "process_cmdline": _process_cmdline(pid) if running and pid is not None else "",
    }


def _collect_profile_liveness(
    ctx: typer.Context,
    *,
    include_remote_status: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    runtime_dir = _clientd_runtime_dir(ctx)
    has_server_override = _has_server_url_override(ctx)
    profiles = load_port_profiles()
    profile_entries: list[dict[str, Any]] = []
    group_entries: dict[str, dict[str, Any]] = {}

    for profile_id, profile in sorted(profiles.items()):
        clientd_payload = client_d_status(
            port_profile_id=profile_id,
            runtime_dir=runtime_dir,
        )
        clientd_data = clientd_payload.get("data")
        clientd_running = bool(clientd_data.get("running")) if isinstance(clientd_data, dict) else False

        gateway_status = _gateway_status_for_profile(ctx, port_profile=profile_id)
        gateway_running = bool(gateway_status.get("running"))

        status_payload: dict[str, Any] | None = None
        status_error: str | None = None
        active_job: dict[str, Any] | None = None
        active_job_status: str | None = None
        if include_remote_status and (clientd_running or has_server_override):
            status_payload = _run_command_with_timeout(
                ctx,
                "/status",
                payload={"port_profile": profile_id},
                timeout_seconds=15.0,
            )
            if _is_ok(status_payload):
                status_data = status_payload.get("data")
                if isinstance(status_data, dict):
                    active_job_data = status_data.get("active_job")
                    if isinstance(active_job_data, dict):
                        active_job = dict(active_job_data)
                    active_job_status_raw = status_data.get("active_job_status")
                    if isinstance(active_job_status_raw, str):
                        active_job_status = active_job_status_raw
            else:
                status_error = str(status_payload.get("message", "status request failed"))

        group_name: str | None = None
        if isinstance(active_job, dict):
            group_name_raw = active_job.pop("group_name", None)
            if isinstance(group_name_raw, str) and group_name_raw.strip():
                group_name = group_name_raw.strip()
            group_profiles_raw = active_job.pop("group_profiles", None)
            group_profiles = (
                [
                    int(value)
                    for value in group_profiles_raw
                    if isinstance(value, int) and not isinstance(value, bool)
                ]
                if isinstance(group_profiles_raw, list)
                else []
            )
            if group_name is not None:
                group_entry = group_entries.setdefault(
                    group_name,
                    {
                        "group_name": group_name,
                        "profiles": [],
                        "job_id": active_job.get("job_id"),
                        "active_job_status_by_profile": {},
                    },
                )
                profiles_payload = group_entry.get("profiles")
                if isinstance(profiles_payload, list):
                    profiles_payload.append(profile_id)
                    profiles_payload.extend(group_profiles)
                active_status_payload = group_entry.get("active_job_status_by_profile")
                if isinstance(active_status_payload, dict):
                    active_status_payload[str(profile_id)] = active_job_status

        compact_active_job: dict[str, Any] | None = None
        if isinstance(active_job, dict):
            compact_active_job = {
                "job_id": active_job.get("job_id"),
                "partition": active_job.get("partition"),
                "model": active_job.get("model"),
                "submitted_at": active_job.get("submitted_at"),
                "service_port": active_job.get("service_port"),
                "jaeger_otlp_port": active_job.get("jaeger_otlp_port"),
                "jaeger_ui_port": active_job.get("jaeger_ui_port"),
            }

        clientd_record = (
            clientd_data.get("record")
            if isinstance(clientd_data, dict)
            else None
        )
        clientd_record = clientd_record if isinstance(clientd_record, dict) else {}

        clientd_summary = {
            "running": clientd_running,
            "pid": clientd_data.get("pid") if isinstance(clientd_data, dict) else None,
            "ssh_target": clientd_record.get("ssh_target"),
            "started_at": clientd_record.get("started_at"),
            "ports": clientd_record.get("ports"),
            "server_url": clientd_data.get("server_url") if isinstance(clientd_data, dict) else None,
        }
        gateway_summary = {
            "running": gateway_running,
            "pid": gateway_status.get("pid"),
            "started_at": (
                gateway_status.get("record", {}).get("started_at")
                if isinstance(gateway_status.get("record"), dict)
                else None
            ),
            "config_path": (
                gateway_status.get("record", {}).get("config_path")
                if isinstance(gateway_status.get("record"), dict)
                else None
            ),
            "host": (
                gateway_status.get("record", {}).get("host")
                if isinstance(gateway_status.get("record"), dict)
                else None
            ),
        }

        alive = bool(clientd_running or gateway_running or active_job is not None)
        status_ok = _is_ok(status_payload) if isinstance(status_payload, dict) else None
        status_code = (
            int(status_payload.get("code"))
            if isinstance(status_payload, dict) and isinstance(status_payload.get("code"), int)
            else None
        )
        status_message = (
            str(status_payload.get("message"))
            if isinstance(status_payload, dict) and isinstance(status_payload.get("message"), str)
            else None
        )
        profile_entries.append(
            {
                "port_profile": profile_id,
                "label": profile.label,
                "alive": alive,
                "group_name": group_name,
                "clientd_running": clientd_running,
                "gateway_running": gateway_running,
                "active_job": compact_active_job,
                "active_job_status": active_job_status,
                "status_error": status_error,
                "status_ok": status_ok,
                "status_code": status_code,
                "status_message": status_message,
                "clientd": clientd_summary,
                "gateway": gateway_summary,
            }
        )
        if verbose:
            profile_entries[-1]["raw"] = {
                "active_job": active_job,
                "clientd": clientd_data if isinstance(clientd_data, dict) else {},
                "gateway": gateway_status,
                "status": status_payload,
            }

    alive_profile_ids = [
        int(entry["port_profile"])
        for entry in profile_entries
        if bool(entry.get("alive"))
    ]
    normalized_group_entries: dict[str, dict[str, Any]] = {}
    for group_name, entry in group_entries.items():
        profiles_payload = entry.get("profiles")
        normalized_profiles = (
            _sorted_unique_profile_ids(
                [
                    int(value)
                    for value in profiles_payload
                    if isinstance(value, int) and not isinstance(value, bool)
                ]
            )
            if isinstance(profiles_payload, list)
            else []
        )
        normalized_group_entries[group_name] = {
            "group_name": group_name,
            "profiles": normalized_profiles,
            "job_id": entry.get("job_id"),
            "active_job_status_by_profile": (
                entry.get("active_job_status_by_profile")
                if isinstance(entry.get("active_job_status_by_profile"), dict)
                else {}
            ),
        }
    return {
        "checked_profiles": [int(profile_id) for profile_id in sorted(profiles.keys())],
        "alive_profile_ids": alive_profile_ids,
        "alive_count": len(alive_profile_ids),
        "groups": normalized_group_entries,
        "profiles": profile_entries,
    }


def _stop_profile_flow(
    ctx: typer.Context,
    *,
    port_profile: int,
    clientd_timeout_seconds: float,
    run_remote_stop: bool,
) -> dict[str, Any]:
    gateway_payload = _stop_gateway_for_profile(
        ctx,
        port_profile=port_profile,
        timeout_seconds=clientd_timeout_seconds,
    )
    if not _is_ok(gateway_payload):
        return {
            "ok": False,
            "code": int(gateway_payload.get("code", 1)),
            "message": str(gateway_payload.get("message", "gateway stop failed")),
            "data": {"gateway": gateway_payload.get("data"), "port_profile": port_profile},
        }

    server_payload: dict[str, Any] | None = None
    if run_remote_stop:
        try:
            server_payload = _run_blocking_with_progress(
                ctx,
                endpoint="/stop",
                payload={"port_profile": port_profile, "block": True},
                progress_endpoint="/stop/status",
                progress_payload={"port_profile": port_profile},
                progress_tag="stop",
                timeout_seconds=STOP_BLOCKING_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            server_payload = {
                "ok": False,
                "code": 1,
                "message": f"failed to issue stop request: {exc}",
                "data": {"port_profile": port_profile},
            }

    clientd_payload = _stop_clientd_for_profile(
        ctx,
        port_profile=port_profile,
        timeout_seconds=clientd_timeout_seconds,
    )
    if run_remote_stop and server_payload is not None:
        return _stop_result_payload(
            server_payload=server_payload,
            clientd_payload=clientd_payload,
            gateway_payload=gateway_payload,
        )

    ok = _is_ok(clientd_payload)
    return {
        "ok": ok,
        "code": 0 if ok else int(clientd_payload.get("code", 1)),
        "message": (
            "remote stop skipped; gateway and client-d stopped"
            if ok
            else str(clientd_payload.get("message", "client-d stop failed"))
        ),
        "data": {
            "port_profile": port_profile,
            "server": None,
            "gateway": gateway_payload.get("data"),
            "clientd": clientd_payload.get("data"),
        },
    }


def _stop_clientd_for_profile(
    ctx: typer.Context,
    *,
    port_profile: int,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    return stop_client_d(
        port_profile_id=port_profile,
        runtime_dir=_clientd_runtime_dir(ctx),
        timeout_seconds=timeout_seconds,
    )


def _stop_gateway_for_profile(
    ctx: typer.Context,
    *,
    port_profile: int,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    return stop_gateway_daemon(
        port_profile_id=port_profile,
        runtime_dir=_clientd_runtime_dir(ctx),
        timeout_seconds=timeout_seconds,
    )


def _server_clientd_payload(
    *,
    ok: bool,
    code: int,
    message: str,
    server_payload: dict[str, Any],
    clientd_payload: dict[str, Any],
    gateway_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "server": server_payload.get("data"),
        "clientd": clientd_payload.get("data"),
    }
    if gateway_payload is not None:
        data["gateway"] = gateway_payload.get("data")
    return {
        "ok": ok,
        "code": code,
        "message": message,
        "data": data,
    }


def _clientd_cleanup_only_payload(
    *,
    ok: bool,
    code: int,
    message: str,
    clientd_payload: dict[str, Any],
    gateway_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"clientd_cleanup": clientd_payload.get("data")}
    if gateway_payload is not None:
        data["gateway_cleanup"] = gateway_payload.get("data")
    return {
        "ok": ok,
        "code": code,
        "message": message,
        "data": data,
    }


def _start_result_payload(
    *,
    server_payload: dict[str, Any],
    clientd_payload: dict[str, Any],
    gateway_payload: dict[str, Any] | None = None,
    wait_up_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _server_clientd_payload(
        ok=True,
        code=0,
        message=str(server_payload.get("message", "start complete")),
        server_payload=server_payload,
        clientd_payload=clientd_payload,
        gateway_payload=gateway_payload,
    )
    if wait_up_payload is not None:
        data = payload.get("data")
        if isinstance(data, dict):
            data["wait_up"] = wait_up_payload.get("data")
    return payload


def _start_failure_payload(
    *,
    server_payload: dict[str, Any],
    clientd_start_payload: dict[str, Any],
    clientd_cleanup_payload: dict[str, Any] | None,
    wait_up_payload: dict[str, Any] | None = None,
    gateway_start_payload: dict[str, Any] | None = None,
    gateway_cleanup_payload: dict[str, Any] | None = None,
    server_cleanup_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "code": int(server_payload.get("code", 1)),
        "message": str(server_payload.get("message", "start failed")),
        "data": {
            "clientd_start": clientd_start_payload.get("data"),
            "server_start": server_payload.get("data"),
            "clientd_cleanup": None if clientd_cleanup_payload is None else clientd_cleanup_payload.get("data"),
            "wait_up": None if wait_up_payload is None else wait_up_payload.get("data"),
            "gateway_start": None if gateway_start_payload is None else gateway_start_payload.get("data"),
            "gateway_cleanup": None if gateway_cleanup_payload is None else gateway_cleanup_payload.get("data"),
            "server_cleanup": None if server_cleanup_payload is None else server_cleanup_payload.get("data"),
        },
    }


def _stop_result_payload(
    *,
    server_payload: dict[str, Any],
    clientd_payload: dict[str, Any],
    gateway_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if gateway_payload is not None and not _is_ok(gateway_payload):
        return {
            "ok": False,
            "code": int(gateway_payload.get("code", 1)),
            "message": str(gateway_payload.get("message", "gateway stop failed")),
            "data": {
                "gateway": gateway_payload.get("data"),
                "server": server_payload.get("data"),
                "clientd": clientd_payload.get("data"),
            },
        }

    server_code = int(server_payload.get("code", 1))
    server_ok = _is_ok(server_payload)
    if server_ok:
        message = str(server_payload.get("message", "stopped"))
        ok = True
        code = 0
    elif server_code == 21:
        message = "no active job; gateway and client-d stopped"
        ok = True
        code = 0
    else:
        message = str(server_payload.get("message", "stop failed"))
        ok = False
        code = server_code
    return _server_clientd_payload(
        ok=ok,
        code=code,
        message=message,
        server_payload=server_payload,
        clientd_payload=clientd_payload,
        gateway_payload=gateway_payload,
    )


@app.callback()
def main(
    ctx: typer.Context,
    server_url: str | None = typer.Option(
        None,
        "--server-url",
        help="Override control server base URL. By default it is derived from --port-profile.",
    ),
    clientd_runtime_dir: Path = typer.Option(
        CLIENT_D_RUNTIME_DIR,
        "--clientd-runtime-dir",
        help="Directory for client-d pid/log files when deriving the control server URL.",
    ),
    timeout_seconds: float = typer.Option(
        120.0,
        "--timeout-seconds",
        help="HTTP timeout for each command.",
    ),
) -> None:
    ctx.obj = {
        "server_url": server_url,
        "clientd_runtime_dir": clientd_runtime_dir,
        "timeout_seconds": timeout_seconds,
    }


def _run_command(ctx: typer.Context, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return _run_command_with_timeout(ctx, endpoint, payload=payload, timeout_seconds=None)


def _run_command_with_timeout(
    ctx: typer.Context,
    endpoint: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    payload = payload or {}
    obj = ctx.obj or {}
    server_url_override = obj.get("server_url")
    if isinstance(server_url_override, str) and server_url_override:
        server_url = server_url_override
    else:
        port_profile = payload.get("port_profile")
        if isinstance(port_profile, bool) or not isinstance(port_profile, int):
            raise RuntimeError("port_profile is required for AMD HPC control commands")
        server_url = resolve_client_d_server_url(
            port_profile_id=port_profile,
            runtime_dir=obj.get("clientd_runtime_dir", CLIENT_D_RUNTIME_DIR),
            remote_server_port=DEFAULT_SERVER_PORT,
        )
    default_timeout_seconds = float(obj.get("timeout_seconds", 120.0))
    request_timeout_seconds = timeout_seconds if timeout_seconds is not None else default_timeout_seconds
    return _post_json(
        server_url=server_url,
        endpoint=endpoint,
        payload=payload,
        timeout=request_timeout_seconds,
    )


def _run_blocking_with_progress(
    ctx: typer.Context,
    *,
    endpoint: str,
    payload: dict[str, Any],
    progress_endpoint: str,
    progress_payload: dict[str, Any] | None = None,
    progress_tag: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def run_command() -> None:
        result["payload"] = _run_command_with_timeout(
            ctx,
            endpoint,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )

    worker = threading.Thread(target=run_command, daemon=True)
    worker.start()

    last_line: str | None = None
    while worker.is_alive():
        progress_response = _run_command(ctx, progress_endpoint, progress_payload)
        data = progress_response.get("data")
        if isinstance(data, dict):
            status = data.get("status")
            phase = data.get("phase")
            message = data.get("message")
            updated_at = data.get("updated_at")
            line = (
                f"[{progress_tag}] status={status} phase={phase} "
                f"updated_at={updated_at} message={message}"
            )
            if line != last_line:
                typer.echo(line)
                last_line = line
        worker.join(timeout=2.0)

    command_payload = result.get("payload")
    if not isinstance(command_payload, dict):
        typer.echo(f"error: {progress_tag} command did not return a valid response", err=True)
        raise typer.Exit(code=1)
    return command_payload


def _format_profile_progress_line(progress_response: dict[str, Any], *, progress_tag: str, profile_id: int) -> str:
    data = progress_response.get("data")
    if isinstance(data, dict):
        status = data.get("status")
        phase = data.get("phase")
        message = data.get("message")
        updated_at = data.get("updated_at")
        return (
            f"[{progress_tag}] profile={profile_id} status={status} "
            f"phase={phase} updated_at={updated_at} message={message}"
        )
    code = progress_response.get("code")
    message = progress_response.get("message")
    return f"[{progress_tag}] profile={profile_id} progress unavailable code={code} message={message}"


def _run_group_blocking_with_progress(
    ctx: typer.Context,
    *,
    endpoint: str,
    payload: dict[str, Any],
    progress_endpoint: str,
    progress_profiles: list[int],
    progress_tag: str,
    timeout_seconds: float,
    preferred_control_profile: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    worker_error: dict[str, Exception] = {}

    def run_command() -> None:
        try:
            result["payload"] = _run_group_command_with_timeout(
                ctx,
                endpoint,
                payload=payload,
                timeout_seconds=timeout_seconds,
                preferred_control_profile=preferred_control_profile,
            )
        except Exception as exc:  # noqa: BLE001
            worker_error["error"] = exc

    worker = threading.Thread(target=run_command, daemon=True)
    worker.start()

    tracked_profiles = _sorted_unique_profile_ids(progress_profiles)
    last_lines: dict[int, str] = {}
    while worker.is_alive():
        for profile_id in tracked_profiles:
            try:
                progress_response = _run_command_with_timeout(
                    ctx,
                    progress_endpoint,
                    payload={"port_profile": profile_id},
                    timeout_seconds=15.0,
                )
                line = _format_profile_progress_line(
                    progress_response,
                    progress_tag=progress_tag,
                    profile_id=profile_id,
                )
            except Exception as exc:  # noqa: BLE001
                line = f"[{progress_tag}] profile={profile_id} progress poll failed: {exc}"

            if last_lines.get(profile_id) != line:
                typer.echo(line)
                last_lines[profile_id] = line

        worker.join(timeout=2.0)

    if "error" in worker_error:
        raise worker_error["error"]

    command_payload = result.get("payload")
    if not isinstance(command_payload, dict):
        typer.echo(f"error: {progress_tag} command did not return a valid response", err=True)
        raise typer.Exit(code=1)
    return command_payload


def _run_blocking_stop_for_profile(
    ctx: typer.Context,
    *,
    port_profile: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    return _run_blocking_with_progress(
        ctx,
        endpoint="/stop",
        payload={"port_profile": port_profile, "block": True},
        progress_endpoint="/stop/status",
        progress_payload={"port_profile": port_profile},
        progress_tag="stop",
        timeout_seconds=timeout_seconds,
    )


def _cleanup_after_start_failure(
    ctx: typer.Context,
    *,
    port_profile: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    gateway_cleanup_payload = _stop_gateway_for_profile(ctx, port_profile=port_profile)
    try:
        server_cleanup_payload = _run_blocking_stop_for_profile(
            ctx,
            port_profile=port_profile,
            timeout_seconds=STOP_BLOCKING_TIMEOUT_SECONDS,
        )
    except KeyboardInterrupt:
        server_cleanup_payload = {
            "ok": False,
            "code": 130,
            "message": "start failure cleanup interrupted",
            "data": {"port_profile": port_profile},
        }
    except Exception as exc:  # noqa: BLE001
        server_cleanup_payload = {
            "ok": False,
            "code": 1,
            "message": f"start failure cleanup stop request failed: {exc}",
            "data": {"port_profile": port_profile},
        }
    clientd_cleanup_payload = _stop_clientd_for_profile(ctx, port_profile=port_profile)
    return server_cleanup_payload, gateway_cleanup_payload, clientd_cleanup_payload


def _emit_cleanup_result(note: str, payload: dict[str, Any]) -> None:
    typer.echo(note, err=True)
    _print_json(payload)


def _stop_clientd_and_emit_cleanup(
    ctx: typer.Context,
    *,
    port_profile: int,
    note: str,
    ok: bool,
    code: int,
    message: str,
    gateway_payload: dict[str, Any] | None = None,
) -> None:
    gateway_cleanup_payload = gateway_payload or _stop_gateway_for_profile(ctx, port_profile=port_profile)
    clientd_payload = _stop_clientd_for_profile(ctx, port_profile=port_profile)
    _emit_cleanup_result(
        note,
        _clientd_cleanup_only_payload(
            ok=ok,
            code=code,
            message=message,
            clientd_payload=clientd_payload,
            gateway_payload=gateway_cleanup_payload,
        ),
    )


def _emit_cleanup_stop_result(
    ctx: typer.Context,
    *,
    port_profile: int,
    server_payload: dict[str, Any],
    success_note: str,
    success_message: str | None = None,
    no_job_note: str | None = None,
    no_job_message: str | None = None,
    error_note: str,
    gateway_payload: dict[str, Any] | None = None,
) -> None:
    gateway_cleanup_payload = gateway_payload or _stop_gateway_for_profile(ctx, port_profile=port_profile)
    clientd_payload = _stop_clientd_for_profile(ctx, port_profile=port_profile)
    server_code = int(server_payload.get("code", 1))
    if _is_ok(server_payload):
        _emit_cleanup_result(
            success_note,
            _server_clientd_payload(
                ok=True,
                code=0,
                message=success_message or str(server_payload.get("message", "cleanup complete")),
                server_payload=server_payload,
                clientd_payload=clientd_payload,
                gateway_payload=gateway_cleanup_payload,
            ),
        )
        return
    if no_job_note is not None and server_code == 21:
        _emit_cleanup_result(
            no_job_note,
            _server_clientd_payload(
                ok=True,
                code=0,
                message=no_job_message or "no active job found; client-d stopped",
                server_payload=server_payload,
                clientd_payload=clientd_payload,
                gateway_payload=gateway_cleanup_payload,
            ),
        )
        return
    _emit_cleanup_result(
        error_note,
        _server_clientd_payload(
            ok=False,
            code=server_code,
            message=str(server_payload.get("message", "cleanup stop failed")),
            server_payload=server_payload,
            clientd_payload=clientd_payload,
            gateway_payload=gateway_cleanup_payload,
        ),
    )


def _cleanup_after_start_interrupt(ctx: typer.Context, *, port_profile: int) -> None:
    typer.echo(
        "[start] interrupt received; entering cleanup and attempting blocking stop.",
        err=True,
    )
    gateway_cleanup_payload = _stop_gateway_for_profile(ctx, port_profile=port_profile)

    try:
        stop_payload = _run_blocking_stop_for_profile(
            ctx,
            port_profile=port_profile,
            timeout_seconds=START_BLOCKING_TIMEOUT_SECONDS,
        )
    except KeyboardInterrupt:
        _stop_clientd_and_emit_cleanup(
            ctx,
            port_profile=port_profile,
            note="[start] cleanup interrupted again; rerun stop with the same --port-profile.",
            ok=False,
            code=130,
            message="start cleanup interrupted",
            gateway_payload=gateway_cleanup_payload,
        )
        return

    if _is_ok(stop_payload) or stop_payload.get("code") != 21:
        _emit_cleanup_stop_result(
            ctx,
            port_profile=port_profile,
            server_payload=stop_payload,
            success_note="[start] cleanup complete: active job canceled.",
            error_note="[start] cleanup stop returned an error:",
            gateway_payload=gateway_cleanup_payload,
        )
        return

    typer.echo(
        "[start] no active job yet; waiting briefly for in-flight submission before retrying stop.",
        err=True,
    )
    deadline = time.monotonic() + START_CLEANUP_WAIT_FOR_SUBMISSION_SECONDS
    last_line: str | None = None

    while time.monotonic() < deadline:
        progress_payload = _run_command(ctx, "/start/status", {"port_profile": port_profile})
        data = progress_payload.get("data")
        if isinstance(data, dict):
            status = data.get("status")
            phase = data.get("phase")
            job_id = data.get("job_id")
            message = data.get("message")
            updated_at = data.get("updated_at")
            line = (
                f"[start-cleanup] status={status} phase={phase} job_id={job_id} "
                f"updated_at={updated_at} message={message}"
            )
            if line != last_line:
                typer.echo(line, err=True)
                last_line = line

            if status == "failed" and not job_id:
                _emit_cleanup_stop_result(
                    ctx,
                    port_profile=port_profile,
                    server_payload={"ok": True, "data": data},
                    success_note="[start] start failed before submission; nothing to cancel.",
                    success_message="start failed before submission; client-d stopped",
                    error_note="[start] cleanup stop returned an error:",
                    gateway_payload=gateway_cleanup_payload,
                )
                return

            if isinstance(job_id, str) and job_id:
                typer.echo(
                    f"[start] detected submitted job {job_id}; retrying blocking stop.",
                    err=True,
                )
                try:
                    retry_payload = _run_blocking_stop_for_profile(
                        ctx,
                        port_profile=port_profile,
                        timeout_seconds=START_BLOCKING_TIMEOUT_SECONDS,
                    )
                except KeyboardInterrupt:
                    _stop_clientd_and_emit_cleanup(
                        ctx,
                        port_profile=port_profile,
                        note="[start] cleanup interrupted again; rerun stop with the same --port-profile.",
                        ok=False,
                        code=130,
                        message="start cleanup interrupted",
                        gateway_payload=gateway_cleanup_payload,
                    )
                    return
                _emit_cleanup_stop_result(
                    ctx,
                    port_profile=port_profile,
                    server_payload=retry_payload,
                    success_note="[start] cleanup complete: active job canceled.",
                    no_job_note="[start] no active job found on retry; cleanup done.",
                    error_note="[start] cleanup stop returned an error on retry:",
                    gateway_payload=gateway_cleanup_payload,
                )
                return

        time.sleep(1.0)

    _stop_clientd_and_emit_cleanup(
        ctx,
        port_profile=port_profile,
        note=(
            "[start] cleanup timed out waiting for submission; "
            "rerun stop with the same --port-profile if needed."
        ),
        ok=False,
        code=1,
        message="start cleanup timed out waiting for submission",
        gateway_payload=gateway_cleanup_payload,
    )


@app.command()
def start(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    partition: str = typer.Option(..., "--partition", "-p", help="Configured partition key."),
    model: str = typer.Option(..., "--model", "-m", help="Configured model key."),
    block: bool = typer.Option(
        False,
        "--block",
        "-b",
        help="Block until services are fully up.",
    ),
    server_port: int = typer.Option(
        DEFAULT_SERVER_PORT,
        "--server-port",
        help="Remote control server port on the login node.",
    ),
    ssh_option: list[str] = typer.Option(
        [],
        "--ssh-option",
        help="Additional raw ssh option token. Repeat to pass multiple tokens.",
    ),
    env: list[str] = typer.Option(
        [],
        "--env",
        help="Additional vLLM environment variable in KEY=VALUE form. Repeat to pass multiple values.",
    ),
    lmcache: int | None = typer.Option(
        None,
        "--lmcache",
        help=(
            "Enable LMCache with a maximum local CPU size. "
            "Sets LMCACHE_MAX_LOCAL_CPU_SIZE and enables kv-transfer-config."
        ),
    ),
) -> None:
    """Start client-d and submit an sbatch job for a configured partition + model."""
    try:
        extra_env = _parse_extra_env_list(env)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--env") from exc
    if lmcache is not None and lmcache <= 0:
        raise typer.BadParameter("--lmcache must be a positive integer", param_hint="--lmcache")
    if lmcache is not None and "LMCACHE_MAX_LOCAL_CPU_SIZE" in extra_env:
        raise typer.BadParameter(
            "cannot combine --lmcache with --env LMCACHE_MAX_LOCAL_CPU_SIZE=...",
            param_hint="--lmcache",
        )

    start_payload: dict[str, Any] = {
        "port_profile": port_profile,
        "partition": partition,
        "model": model,
        "extra_env": extra_env,
    }
    if lmcache is not None:
        start_payload["lmcache"] = lmcache

    clientd_payload = start_client_d(
        ssh_target=ssh_target,
        port_profile_id=port_profile,
        runtime_dir=_clientd_runtime_dir(ctx),
        remote_server_port=server_port,
        ssh_options=ssh_option or None,
    )
    _require_ok(clientd_payload)

    wait_up_payload: dict[str, Any] | None = None
    if block:
        try:
            server_payload = _run_blocking_with_progress(
                ctx,
                endpoint="/start",
                payload={**start_payload, "block": True},
                progress_endpoint="/start/status",
                progress_payload={"port_profile": port_profile},
                progress_tag="start",
                timeout_seconds=START_BLOCKING_TIMEOUT_SECONDS,
            )
        except KeyboardInterrupt:
            _cleanup_after_start_interrupt(ctx, port_profile=port_profile)
            raise typer.Exit(code=130)
    else:
        server_payload = _run_command_with_timeout(
            ctx,
            "/start",
            payload={**start_payload, "block": False},
            timeout_seconds=None,
        )
    if not _is_ok(server_payload):
        gateway_cleanup_payload = _stop_gateway_for_profile(ctx, port_profile=port_profile)
        cleanup_payload = _stop_clientd_for_profile(ctx, port_profile=port_profile)
        _print_json(
            _start_failure_payload(
                server_payload=server_payload,
                clientd_start_payload=clientd_payload,
                clientd_cleanup_payload=cleanup_payload,
                gateway_cleanup_payload=gateway_cleanup_payload,
            )
        )
        raise typer.Exit(code=1)

    if not block:
        typer.echo(
            "[start] waiting for service readiness before launching gateway.",
            err=True,
        )
        try:
            wait_up_payload = _run_command_with_timeout(
                ctx,
                "/wait-up",
                payload={
                    "port_profile": port_profile,
                    "timeout_seconds": GATEWAY_WAIT_UP_TIMEOUT_SECONDS,
                    "poll_interval_seconds": GATEWAY_WAIT_UP_POLL_INTERVAL_SECONDS,
                    "defer_timeout_until_running": True,
                },
                timeout_seconds=START_BLOCKING_TIMEOUT_SECONDS,
            )
        except KeyboardInterrupt:
            _cleanup_after_start_interrupt(ctx, port_profile=port_profile)
            raise typer.Exit(code=130)

        if not _is_ok(wait_up_payload):
            (
                server_cleanup_payload,
                gateway_cleanup_payload,
                clientd_cleanup_payload,
            ) = _cleanup_after_start_failure(ctx, port_profile=port_profile)
            _print_json(
                _start_failure_payload(
                    server_payload=wait_up_payload,
                    clientd_start_payload=clientd_payload,
                    clientd_cleanup_payload=clientd_cleanup_payload,
                    wait_up_payload=wait_up_payload,
                    gateway_cleanup_payload=gateway_cleanup_payload,
                    server_cleanup_payload=server_cleanup_payload,
                )
            )
            raise typer.Exit(code=1)

    gateway_payload = start_gateway_daemon(
        port_profile_id=port_profile,
        runtime_dir=_clientd_runtime_dir(ctx),
    )
    if not _is_ok(gateway_payload):
        (
            server_cleanup_payload,
            gateway_cleanup_payload,
            clientd_cleanup_payload,
        ) = _cleanup_after_start_failure(ctx, port_profile=port_profile)
        _print_json(
            _start_failure_payload(
                server_payload=gateway_payload,
                clientd_start_payload=clientd_payload,
                clientd_cleanup_payload=clientd_cleanup_payload,
                wait_up_payload=wait_up_payload,
                gateway_start_payload=gateway_payload,
                gateway_cleanup_payload=gateway_cleanup_payload,
                server_cleanup_payload=server_cleanup_payload,
            )
        )
        raise typer.Exit(code=1)

    _print_json(
        _start_result_payload(
            server_payload=server_payload,
            clientd_payload=clientd_payload,
            gateway_payload=gateway_payload,
            wait_up_payload=wait_up_payload,
        )
    )


@app.command()
def stop(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    clientd_timeout_seconds: float = typer.Option(
        10.0,
        "--clientd-timeout-seconds",
        help="Seconds to wait before forcing client-d shutdown.",
    ),
) -> None:
    """Stop gateway first, then stop the active sbatch job and client-d."""
    _ = ssh_target
    stop_payload = _stop_profile_flow(
        ctx,
        port_profile=port_profile,
        clientd_timeout_seconds=clientd_timeout_seconds,
        run_remote_stop=True,
    )
    if not _is_ok(stop_payload):
        _print_json(stop_payload)
        raise typer.Exit(code=1)
    _print_json(stop_payload)


@app.command(name="start-group")
def start_group(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    group_name: str = typer.Option(
        ...,
        "--group-name",
        "-g",
        help="Logical name for this grouped service run.",
    ),
    profile_list: str = typer.Option(
        ...,
        "--profile-list",
        "-L",
        help="Comma-separated port profile IDs, e.g. 0,1,2,3.",
    ),
    partition: str = typer.Option(..., "--partition", "-p", help="Configured partition key."),
    model: str = typer.Option(..., "--model", "-m", help="Configured model key."),
    server_port: int = typer.Option(
        DEFAULT_SERVER_PORT,
        "--server-port",
        help="Remote control server port on the login node.",
    ),
    ssh_option: list[str] = typer.Option(
        [],
        "--ssh-option",
        help="Additional raw ssh option token. Repeat to pass multiple tokens.",
    ),
    env: list[str] = typer.Option(
        [],
        "--env",
        help="Additional vLLM environment variable in KEY=VALUE form. Repeat to pass multiple values.",
    ),
    lmcache: int | None = typer.Option(
        None,
        "--lmcache",
        help=(
            "Enable LMCache with a maximum local CPU size. "
            "Sets LMCACHE_MAX_LOCAL_CPU_SIZE and enables kv-transfer-config."
        ),
    ),
    clientd_timeout_seconds: float = typer.Option(
        10.0,
        "--clientd-timeout-seconds",
        help="Seconds to wait before forcing local gateway/client-d shutdown.",
    ),
) -> None:
    """Start grouped services across a list of port profiles."""
    try:
        profile_ids = _parse_profile_list(profile_list)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile-list") from exc
    try:
        extra_env = _parse_extra_env_list(env)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--env") from exc
    if lmcache is not None and lmcache <= 0:
        raise typer.BadParameter("--lmcache must be a positive integer", param_hint="--lmcache")
    if lmcache is not None and "LMCACHE_MAX_LOCAL_CPU_SIZE" in extra_env:
        raise typer.BadParameter(
            "cannot combine --lmcache with --env LMCACHE_MAX_LOCAL_CPU_SIZE=...",
            param_hint="--lmcache",
        )

    group_start_payload: dict[str, Any] = {
        "group_name": group_name,
        "profile_list": profile_ids,
        "partition": partition,
        "model": model,
        "block": True,
        "extra_env": extra_env,
    }
    if lmcache is not None:
        group_start_payload["lmcache"] = lmcache

    clientd_payloads: dict[str, dict[str, Any]] = {}
    started_profiles: list[int] = []
    for profile_id in profile_ids:
        typer.echo(f"[start-group] starting client-d for profile={profile_id}")
        payload = start_client_d(
            ssh_target=ssh_target,
            port_profile_id=profile_id,
            runtime_dir=_clientd_runtime_dir(ctx),
            remote_server_port=server_port,
            ssh_options=ssh_option or None,
        )
        clientd_payloads[str(profile_id)] = payload
        if _is_ok(payload):
            started_profiles.append(profile_id)
            typer.echo(f"[start-group] client-d ready for profile={profile_id}")
            continue

        cleanup_payloads, cleanup_failures = _stop_local_profile_daemons(
            ctx,
            profile_ids=started_profiles,
            clientd_timeout_seconds=clientd_timeout_seconds,
        )
        _print_json(
            {
                "ok": False,
                "code": int(payload.get("code", 1)),
                "message": str(payload.get("message", "failed to start client-d for grouped run")),
                "data": {
                    "group_name": group_name,
                    "profile_list": profile_ids,
                    "failed_profile": profile_id,
                    "clientd": clientd_payloads,
                    "local_cleanup": {
                        "failed_profiles": cleanup_failures,
                        "results": cleanup_payloads,
                    },
                },
            }
        )
        raise typer.Exit(code=1)

    preferred_control_profile = profile_ids[0]
    try:
        server_payload = _run_group_blocking_with_progress(
            ctx,
            endpoint="/group/start",
            payload=group_start_payload,
            progress_endpoint="/start/status",
            progress_profiles=profile_ids,
            progress_tag="start-group",
            timeout_seconds=START_BLOCKING_TIMEOUT_SECONDS,
            preferred_control_profile=preferred_control_profile,
        )
    except Exception as exc:  # noqa: BLE001
        local_cleanup_payloads, cleanup_failures = _stop_local_profile_daemons(
            ctx,
            profile_ids=profile_ids,
            clientd_timeout_seconds=clientd_timeout_seconds,
        )
        _print_json(
            {
                "ok": False,
                "code": 1,
                "message": f"failed to issue grouped start command: {exc}",
                "data": {
                    "group_name": group_name,
                    "profile_list": profile_ids,
                    "clientd": clientd_payloads,
                    "local_cleanup": {
                        "failed_profiles": cleanup_failures,
                        "results": local_cleanup_payloads,
                    },
                },
            }
        )
        raise typer.Exit(code=1)

    if not _is_ok(server_payload):
        try:
            server_cleanup_payload = _run_group_command_with_timeout(
                ctx,
                "/group/stop",
                payload={"group_name": group_name, "block": True},
                timeout_seconds=STOP_BLOCKING_TIMEOUT_SECONDS,
                preferred_control_profile=preferred_control_profile,
            )
        except Exception as exc:  # noqa: BLE001
            server_cleanup_payload = {
                "ok": False,
                "code": 1,
                "message": f"failed to run grouped stop cleanup: {exc}",
                "data": {"group_name": group_name},
            }

        local_cleanup_payloads, cleanup_failures = _stop_local_profile_daemons(
            ctx,
            profile_ids=profile_ids,
            clientd_timeout_seconds=clientd_timeout_seconds,
        )
        _print_json(
            {
                "ok": False,
                "code": int(server_payload.get("code", 1)),
                "message": str(server_payload.get("message", "grouped start failed")),
                "data": {
                    "group_name": group_name,
                    "profile_list": profile_ids,
                    "server_start": server_payload,
                    "server_cleanup": server_cleanup_payload,
                    "clientd": clientd_payloads,
                    "local_cleanup": {
                        "failed_profiles": cleanup_failures,
                        "results": local_cleanup_payloads,
                    },
                },
            }
        )
        raise typer.Exit(code=1)

    gateway_payloads: dict[str, dict[str, Any]] = {}
    failed_gateway_profiles: list[int] = []
    for profile_id in profile_ids:
        typer.echo(f"[start-group] starting gateway for profile={profile_id}")
        gateway_payload = start_gateway_daemon(
            port_profile_id=profile_id,
            runtime_dir=_clientd_runtime_dir(ctx),
        )
        gateway_payloads[str(profile_id)] = gateway_payload
        if not _is_ok(gateway_payload):
            failed_gateway_profiles.append(profile_id)
        else:
            typer.echo(f"[start-group] gateway ready for profile={profile_id}")

    if failed_gateway_profiles:
        try:
            server_cleanup_payload = _run_group_command_with_timeout(
                ctx,
                "/group/stop",
                payload={"group_name": group_name, "block": True},
                timeout_seconds=STOP_BLOCKING_TIMEOUT_SECONDS,
                preferred_control_profile=preferred_control_profile,
            )
        except Exception as exc:  # noqa: BLE001
            server_cleanup_payload = {
                "ok": False,
                "code": 1,
                "message": f"failed to run grouped stop cleanup after gateway failure: {exc}",
                "data": {"group_name": group_name},
            }

        local_cleanup_payloads, cleanup_failures = _stop_local_profile_daemons(
            ctx,
            profile_ids=profile_ids,
            clientd_timeout_seconds=clientd_timeout_seconds,
        )
        _print_json(
            {
                "ok": False,
                "code": 1,
                "message": "grouped start failed while launching local gateways",
                "data": {
                    "group_name": group_name,
                    "profile_list": profile_ids,
                    "failed_gateway_profiles": failed_gateway_profiles,
                    "server_start": server_payload,
                    "server_cleanup": server_cleanup_payload,
                    "clientd": clientd_payloads,
                    "gateway": gateway_payloads,
                    "local_cleanup": {
                        "failed_profiles": cleanup_failures,
                        "results": local_cleanup_payloads,
                    },
                },
            }
        )
        raise typer.Exit(code=1)

    _print_json(
        {
            "ok": True,
            "code": 0,
            "message": str(server_payload.get("message", "grouped services are up")),
            "data": {
                "group_name": group_name,
                "profile_list": profile_ids,
                "server": server_payload.get("data"),
                "clientd": {key: value.get("data") for key, value in clientd_payloads.items()},
                "gateway": {key: value.get("data") for key, value in gateway_payloads.items()},
            },
        }
    )


@app.command(name="group-status")
def group_status(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    group_name: str = typer.Option(
        ...,
        "--group-name",
        "-g",
        help="Logical group name.",
    ),
) -> None:
    """Show status for an active grouped service run."""
    _ = ssh_target
    typer.echo(f"[group-status] querying group={group_name}")
    payload = _run_group_command_with_timeout(
        ctx,
        "/group/status",
        payload={"group_name": group_name},
        timeout_seconds=30.0,
    )
    if _is_ok(payload):
        profile_ids = _extract_profile_ids_from_group_status(payload)
        typer.echo(f"[group-status] received status for {len(profile_ids)} profiles")
    _require_ok(payload)
    _print_json(payload)


@app.command(name="stop-group")
def stop_group(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    group_name: str = typer.Option(
        ...,
        "--group-name",
        "-g",
        help="Logical group name.",
    ),
    clientd_timeout_seconds: float = typer.Option(
        10.0,
        "--clientd-timeout-seconds",
        help="Seconds to wait before forcing local gateway/client-d shutdown.",
    ),
) -> None:
    """Stop a grouped service run and tear down local daemons for all group profiles."""
    _ = ssh_target
    typer.echo(f"[stop-group] querying group={group_name}")
    group_status_payload = _run_group_command_with_timeout(
        ctx,
        "/group/status",
        payload={"group_name": group_name},
        timeout_seconds=30.0,
    )
    if not _is_ok(group_status_payload):
        _print_json(group_status_payload)
        raise typer.Exit(code=1)

    profile_ids = _extract_profile_ids_from_group_status(group_status_payload)
    if not profile_ids:
        _print_json(
            {
                "ok": False,
                "code": 1,
                "message": f"group '{group_name}' has no associated profiles",
                "data": {"group_status": group_status_payload},
            }
        )
        raise typer.Exit(code=1)

    preferred_control_profile = profile_ids[0]
    typer.echo(
        f"[stop-group] stopping group={group_name} profiles={','.join(str(profile_id) for profile_id in profile_ids)}"
    )
    server_payload = _run_group_blocking_with_progress(
        ctx,
        endpoint="/group/stop",
        payload={"group_name": group_name, "block": True},
        progress_endpoint="/stop/status",
        progress_profiles=profile_ids,
        progress_tag="stop-group",
        timeout_seconds=STOP_BLOCKING_TIMEOUT_SECONDS,
        preferred_control_profile=preferred_control_profile,
    )
    typer.echo("[stop-group] tearing down local gateway/client-d daemons")
    local_cleanup_payloads, cleanup_failures = _stop_local_profile_daemons(
        ctx,
        profile_ids=profile_ids,
        clientd_timeout_seconds=clientd_timeout_seconds,
    )

    ok = _is_ok(server_payload) and not cleanup_failures
    payload = {
        "ok": ok,
        "code": 0 if ok else int(server_payload.get("code", 1)),
        "message": (
            str(server_payload.get("message", "grouped services stopped"))
            if ok
            else (
                "group stop completed with local cleanup failures"
                if _is_ok(server_payload)
                else str(server_payload.get("message", "group stop failed"))
            )
        ),
        "data": {
            "group_name": group_name,
            "profile_list": profile_ids,
            "server": server_payload,
            "group_status": group_status_payload,
            "local_cleanup": {
                "failed_profiles": cleanup_failures,
                "results": local_cleanup_payloads,
            },
        },
    }
    _print_json(payload)
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def logs(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    lines: int = typer.Option(200, "--lines", "-n", min=1, help="Lines per log file."),
) -> None:
    """Fetch current tail of Slurm/Jaeger/vLLM logs."""
    _ = ssh_target
    payload = _run_command(ctx, "/logs", {"port_profile": port_profile, "lines": lines})
    _require_ok(payload)

    data = payload.get("data")
    if not isinstance(data, dict):
        _print_json(payload)
        return

    logs_payload = data.get("logs")
    if not isinstance(logs_payload, dict):
        _print_json(payload)
        return

    typer.echo(f"job_id: {data.get('job_id')} ({data.get('job_status')})")
    for name in ["slurm_out", "slurm_err", "jaeger", "vllm"]:
        typer.echo(f"\n===== {name} =====")
        text = logs_payload.get(name)
        if isinstance(text, str) and text:
            typer.echo(text.rstrip("\n"))
        else:
            typer.echo("(empty)")


@app.command()
def up(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
) -> None:
    """Check whether both tunneled vLLM and Jaeger endpoints are up."""
    _ = ssh_target
    payload = _run_command(ctx, "/up", {"port_profile": port_profile})
    _require_ok(payload)
    _print_json(payload)


@app.command(name="wait-up")
def wait_up(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    timeout_seconds: int = typer.Option(
        900,
        "--timeout-seconds",
        min=1,
        help="Maximum seconds to wait for vLLM + Jaeger readiness.",
    ),
    poll_interval_seconds: float = typer.Option(
        2.0,
        "--poll-interval-seconds",
        min=0.1,
        help="Polling interval in seconds.",
    ),
    defer_timeout_until_running: bool = typer.Option(
        True,
        "--defer-timeout-until-running/--timeout-from-submit",
        help="If set, startup timeout begins only after Slurm state reaches RUNNING.",
    ),
) -> None:
    """Block until both tunneled vLLM and Jaeger endpoints are up."""
    _ = ssh_target
    request_timeout_seconds = (
        24.0 * 60.0 * 60.0
        if defer_timeout_until_running
        else max(float(timeout_seconds) + 30.0, 120.0)
    )
    payload = _run_command_with_timeout(
        ctx,
        "/wait-up",
        payload={
            "port_profile": port_profile,
            "timeout_seconds": timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "defer_timeout_until_running": defer_timeout_until_running,
        },
        timeout_seconds=request_timeout_seconds,
    )
    _require_ok(payload)
    _print_json(payload)


@app.command(name="alive-profiles")
def alive_profiles(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    include_remote_status: bool = typer.Option(
        True,
        "--include-remote-status/--local-only",
        help=(
            "Include remote /status checks for profiles that have an active tunnel "
            "or when --server-url is set."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Include full raw per-profile status payloads.",
    ),
) -> None:
    """List all configured port profiles and mark which ones are currently alive."""
    _ = ssh_target
    payload = {
        "ok": True,
        "code": 0,
        "message": "alive profile scan complete",
        "data": _collect_profile_liveness(
            ctx,
            include_remote_status=include_remote_status,
            verbose=verbose,
        ),
    }
    _print_json(payload)


@app.command(name="stop-alive-profiles")
def stop_alive_profiles(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    clientd_timeout_seconds: float = typer.Option(
        10.0,
        "--clientd-timeout-seconds",
        help="Seconds to wait before forcing client-d and gateway shutdown.",
    ),
    include_remote_status: bool = typer.Option(
        True,
        "--include-remote-status/--local-only",
        help=(
            "Include remote /status checks for profiles that have an active tunnel "
            "or when --server-url is set."
        ),
    ),
) -> None:
    """Stop every profile that is currently alive."""
    _ = ssh_target
    liveness = _collect_profile_liveness(
        ctx,
        include_remote_status=include_remote_status,
        verbose=False,
    )
    alive_profile_ids = [int(value) for value in liveness.get("alive_profile_ids", [])]
    if not alive_profile_ids:
        _print_json(
            {
                "ok": True,
                "code": 0,
                "message": "no alive profiles found",
                "data": {
                    "alive_profile_ids": [],
                    "stopped_profile_ids": [],
                    "failed_profile_ids": [],
                    "results": {},
                },
            }
        )
        return

    has_server_override = _has_server_url_override(ctx)
    liveness_by_profile: dict[int, dict[str, Any]] = {}
    for entry in liveness.get("profiles", []):
        if not isinstance(entry, dict):
            continue
        profile_id_raw = entry.get("port_profile")
        if isinstance(profile_id_raw, int):
            liveness_by_profile[profile_id_raw] = entry

    results: dict[str, dict[str, Any]] = {}
    alive_profile_id_set = set(alive_profile_ids)
    groups_payload = liveness.get("groups")
    groups_by_name = groups_payload if isinstance(groups_payload, dict) else {}
    stopped_profile_ids: set[int] = set()
    failed_profile_ids: set[int] = set()
    handled_profiles: set[int] = set()

    for profile_id in sorted(alive_profile_ids):
        if profile_id in handled_profiles:
            continue
        profile_liveness = liveness_by_profile.get(profile_id, {})
        group_name_raw = profile_liveness.get("group_name")
        group_name = (
            group_name_raw.strip()
            if isinstance(group_name_raw, str) and group_name_raw.strip()
            else None
        )

        if group_name is not None:
            group_payload = groups_by_name.get(group_name)
            group_profiles_raw = (
                group_payload.get("profiles")
                if isinstance(group_payload, dict)
                else []
            )
            group_profile_ids = (
                [
                    int(value)
                    for value in group_profiles_raw
                    if isinstance(value, int) and not isinstance(value, bool)
                ]
                if isinstance(group_profiles_raw, list)
                else [profile_id]
            )
            group_profile_ids.append(profile_id)
            group_profile_ids = [
                group_profile_id
                for group_profile_id in _sorted_unique_profile_ids(group_profile_ids)
                if group_profile_id in alive_profile_id_set
            ]
            if not group_profile_ids:
                group_profile_ids = [profile_id]

            run_remote_group_stop = bool(
                has_server_override
                or any(
                    bool(liveness_by_profile.get(group_profile_id, {}).get("clientd_running"))
                    for group_profile_id in group_profile_ids
                )
            )
            preferred_control_profile = next(
                (
                    group_profile_id
                    for group_profile_id in group_profile_ids
                    if bool(liveness_by_profile.get(group_profile_id, {}).get("clientd_running"))
                ),
                group_profile_ids[0],
            )

            if run_remote_group_stop:
                try:
                    group_stop_payload = _run_group_command_with_timeout(
                        ctx,
                        "/group/stop",
                        payload={"group_name": group_name, "block": True},
                        timeout_seconds=STOP_BLOCKING_TIMEOUT_SECONDS,
                        preferred_control_profile=preferred_control_profile,
                    )
                except Exception as exc:  # noqa: BLE001
                    group_stop_payload = {
                        "ok": False,
                        "code": 1,
                        "message": f"failed to issue group stop for '{group_name}': {exc}",
                        "data": {
                            "group_name": group_name,
                            "profile_list": group_profile_ids,
                        },
                    }
            else:
                group_stop_payload = {
                    "ok": True,
                    "code": 0,
                    "message": "group remote stop skipped; no control channel available",
                    "data": {
                        "group_name": group_name,
                        "profile_list": group_profile_ids,
                    },
                }

            local_cleanup_payloads, local_cleanup_failures = _stop_local_profile_daemons(
                ctx,
                profile_ids=group_profile_ids,
                clientd_timeout_seconds=clientd_timeout_seconds,
            )
            group_ok = _is_ok(group_stop_payload) and not local_cleanup_failures
            for group_profile_id in group_profile_ids:
                member_liveness = liveness_by_profile.get(group_profile_id, {})
                results[str(group_profile_id)] = {
                    "mode": "group",
                    "group_name": group_name,
                    "profile_list": group_profile_ids,
                    "run_remote_stop": run_remote_group_stop,
                    "liveness": member_liveness,
                    "group_stop": group_stop_payload,
                    "local_cleanup": local_cleanup_payloads.get(str(group_profile_id)),
                }
                if group_ok:
                    stopped_profile_ids.add(group_profile_id)
                else:
                    failed_profile_ids.add(group_profile_id)
                handled_profiles.add(group_profile_id)
            continue

        run_remote_stop = bool(profile_liveness.get("clientd_running") or has_server_override)
        stop_payload = _stop_profile_flow(
            ctx,
            port_profile=profile_id,
            clientd_timeout_seconds=clientd_timeout_seconds,
            run_remote_stop=run_remote_stop,
        )
        results[str(profile_id)] = {
            "mode": "single",
            "run_remote_stop": run_remote_stop,
            "liveness": profile_liveness,
            "stop": stop_payload,
        }
        if _is_ok(stop_payload):
            stopped_profile_ids.add(profile_id)
        else:
            failed_profile_ids.add(profile_id)
        handled_profiles.add(profile_id)

    stopped_profile_list = sorted(stopped_profile_ids)
    failed_profile_list = sorted(failed_profile_ids)
    ok = not failed_profile_list
    payload = {
        "ok": ok,
        "code": 0 if ok else 1,
        "message": (
            f"stopped {len(stopped_profile_list)} alive profiles"
            if ok
            else (
                f"failed to stop {len(failed_profile_list)} of "
                f"{len(alive_profile_ids)} alive profiles"
            )
        ),
        "data": {
            "alive_profile_ids": alive_profile_ids,
            "stopped_profile_ids": stopped_profile_list,
            "failed_profile_ids": failed_profile_list,
            "results": results,
        },
    }
    _print_json(payload)
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def status(
    ctx: typer.Context,
    ssh_target: str = _remote_host_option(),
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
) -> None:
    """Show control-plane status and configured partitions/models."""
    _ = ssh_target
    payload = _run_command(ctx, "/status", {"port_profile": port_profile})
    _require_ok(payload)
    _print_json(payload)


if __name__ == "__main__":
    app()
