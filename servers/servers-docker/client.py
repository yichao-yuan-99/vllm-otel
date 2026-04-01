#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Typer CLI for Docker daemon/frontend control."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import typer

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "servers" / "servers-docker"
COMPOSE_FILE = DOCKER_DIR / "docker-compose.yml"
STARTUP_LOGS_DIR = DOCKER_DIR / "logs"
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"
PORT_PROFILES_PATH = REPO_ROOT / "configs" / "port_profiles.toml"
LAUNCH_PROFILES_PATH = DOCKER_DIR / "launch_profiles.toml"
IMAGE_CONFIG_PATH = DOCKER_DIR / "service_images.toml"

RUNTIME_DIR = Path.home() / ".cache" / "vllm-otel-docker-daemon"
COMPOSE_GPU_OVERRIDE_FILE = RUNTIME_DIR / "compose.gpu.override.yml"
DAEMON_PID_FILE = RUNTIME_DIR / "daemon.pid.json"
DAEMON_LOG_FILE = RUNTIME_DIR / "daemon.log"
DAEMON_HEARTBEAT_FILE = RUNTIME_DIR / "daemon.heartbeat.json"
STATE_FILE = RUNTIME_DIR / "state.json"
STOP_STATE_FILE = RUNTIME_DIR / "stop_state.json"
STOP_LOG_FILE = RUNTIME_DIR / "stop.log"
COMPOSE_ENV_FILE = RUNTIME_DIR / "compose.env"

DEFAULT_WAIT_UP_TIMEOUT_SECONDS = 900
DEFAULT_WAIT_UP_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_STARTUP_LOG_TAIL_LINES = 40
GATEWAY_STARTUP_PROBE_SECONDS = 3.0
DEFAULT_COMPOSE_PROJECT_NAME = "vllm-otel"
DEFAULT_JAEGER_CONTAINER_NAME = "jaeger"
DEFAULT_VLLM_CONTAINER_NAME = "vllm-openai-otel-lp"
DEFAULT_OTEL_SERVICE_NAME = "vllm-server"
DEFAULT_GATEWAY_CONFIG_PATH = REPO_ROOT / "gateway" / "config.toml"
DEFAULT_GATEWAY_CONFIG_EXAMPLE_PATH = REPO_ROOT / "gateway" / "config.example.toml"
DEFAULT_GATEWAY_VENV_DIR = REPO_ROOT / ".venv"
DEFAULT_GATEWAY_HOST = "0.0.0.0"
DEFAULT_GATEWAY_PID_FILE_PREFIX = "gateway_d"
DEFAULT_GATEWAY_LOG_FILE_PREFIX = "gateway_d"
VLLM_CRASH_LOOP_RESTART_THRESHOLD = 3
LMCACHE_KV_TRANSFER_CONFIG = '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
COMPOSE_MANAGED_ENV_KEYS = frozenset(
    {
        "COMPOSE_PROJECT_NAME",
        "JAEGER_IMAGE",
        "JAEGER_CONTAINER_NAME",
        "JAEGER_API_PORT",
        "JAEGER_OTLP_PORT",
        "OTEL_SERVICE_NAME",
        "PORT_PROFILE_ID",
        "VLLM_CONTAINER_NAME",
        "VLLM_MODEL_EXTRA_ARGS_B64",
        "VLLM_FORCE_SEQ_TRUST_REMOTE_CODE",
        "VLLM_IMAGE_NAME",
        "VLLM_MODEL_NAME",
        "VLLM_SERVED_MODEL_NAME",
        "VLLM_SERVICE_PORT",
        "VLLM_TENSOR_PARALLEL_SIZE",
        "LMCACHE_INTERNAL_API_SERVER_ENABLED",
        "LMCACHE_INTERNAL_API_SERVER_PORT_START",
        "LMCACHE_MAX_LOCAL_CPU_SIZE",
        "PYTHONHASHSEED",
    }
)


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Docker control client (daemon + managed environments).",
)
profiles_app = typer.Typer(add_completion=False, no_args_is_help=True, help="List configured profiles.")
app.add_typer(profiles_app, name="profiles")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _payload(*, ok: bool, code: int, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": ok,
        "code": code,
        "message": message,
        "data": data or {},
    }


def _print_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _emit(payload: dict[str, Any], *, fail_on_error: bool = True) -> None:
    _print_json(payload)
    if fail_on_error and not payload.get("ok"):
        raise typer.Exit(code=1)


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sanitize_path_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    cleaned = cleaned.strip("._-")
    return cleaned or "value"


def _sanitize_runtime_token(value: str) -> str:
    lowered = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered)
    cleaned = cleaned.strip("-")
    return cleaned or "value"


def _bounded_runtime_name(prefix: str, suffix: str, *, max_length: int = 63) -> str:
    candidate = f"{prefix}-{suffix}"
    if len(candidate) <= max_length:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
    keep = max(max_length - len(prefix) - len(digest) - 2, 8)
    shortened_suffix = suffix[:keep].rstrip("-")
    if not shortened_suffix:
        shortened_suffix = "value"
    return f"{prefix}-{shortened_suffix}-{digest}"


def _runtime_names_for_selection(*, model_key: str, launch_profile_key: str) -> dict[str, str]:
    model_token = _sanitize_runtime_token(model_key)
    launch_token = _sanitize_runtime_token(launch_profile_key)
    suffix = f"{model_token}-{launch_token}"
    return {
        "compose_project_name": _bounded_runtime_name(DEFAULT_COMPOSE_PROJECT_NAME, suffix),
        "jaeger_container_name": _bounded_runtime_name(DEFAULT_JAEGER_CONTAINER_NAME, suffix, max_length=96),
        "vllm_container_name": _bounded_runtime_name(DEFAULT_VLLM_CONTAINER_NAME, suffix, max_length=96),
        "otel_service_name": _bounded_runtime_name(DEFAULT_OTEL_SERVICE_NAME, suffix, max_length=96),
    }


def _runtime_names_from_state(state: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "compose_project_name": DEFAULT_COMPOSE_PROJECT_NAME,
        "jaeger_container_name": DEFAULT_JAEGER_CONTAINER_NAME,
        "vllm_container_name": DEFAULT_VLLM_CONTAINER_NAME,
        "otel_service_name": DEFAULT_OTEL_SERVICE_NAME,
    }
    resolved = state.get("resolved")
    if isinstance(resolved, dict):
        runtime_names = resolved.get("runtime_names")
        if isinstance(runtime_names, dict):
            out = dict(defaults)
            for key in out:
                value = runtime_names.get(key)
                if isinstance(value, str) and value.strip():
                    out[key] = value.strip()
            return out
    env_values = _parse_compose_env_file(COMPOSE_ENV_FILE)
    out = dict(defaults)
    env_map = {
        "compose_project_name": "COMPOSE_PROJECT_NAME",
        "jaeger_container_name": "JAEGER_CONTAINER_NAME",
        "vllm_container_name": "VLLM_CONTAINER_NAME",
        "otel_service_name": "OTEL_SERVICE_NAME",
    }
    for out_key, env_key in env_map.items():
        value = env_values.get(env_key)
        if isinstance(value, str) and value.strip():
            out[out_key] = value.strip()
    return out


def _runtime_names_from_selection(selection: dict[str, Any]) -> dict[str, str]:
    identity = _selection_identity(selection)
    if identity is not None:
        defaults = _runtime_names_for_selection(model_key=identity[0], launch_profile_key=identity[2])
    else:
        defaults = {
            "compose_project_name": DEFAULT_COMPOSE_PROJECT_NAME,
            "jaeger_container_name": DEFAULT_JAEGER_CONTAINER_NAME,
            "vllm_container_name": DEFAULT_VLLM_CONTAINER_NAME,
            "otel_service_name": DEFAULT_OTEL_SERVICE_NAME,
        }
    runtime_names = selection.get("runtime_names")
    if not isinstance(runtime_names, dict):
        return defaults
    out = dict(defaults)
    for key in out:
        value = runtime_names.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    return out


def _compose_project_name() -> str:
    env_values = _parse_compose_env_file(COMPOSE_ENV_FILE)
    project_name = env_values.get("COMPOSE_PROJECT_NAME", "").strip()
    if project_name:
        return project_name
    return DEFAULT_COMPOSE_PROJECT_NAME


def _new_startup_log_path(*, model_key: str, port_profile_id: int, launch_profile_key: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_token = _sanitize_path_token(model_key)
    port_token = _sanitize_path_token(str(port_profile_id))
    launch_token = _sanitize_path_token(launch_profile_key)
    return STARTUP_LOGS_DIR / f"start-{stamp}-{model_token}-p{port_token}-{launch_token}.log"


def _new_compose_log_path(startup_log_path: Path) -> Path:
    return startup_log_path.with_suffix(".compose.log")


def _emit_progress(message: str, *, log_path: Path | None = None) -> None:
    line = f"[{_utc_now_iso()}] {message}"
    typer.echo(line, err=True)
    if log_path is not None:
        _append_text(log_path, f"{line}\n")


def _ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default
    return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _run_exec(
    cmd: list[str],
    *,
    timeout_seconds: float | None = None,
    env: dict[str, str] | None = None,
) -> ExecResult:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        return ExecResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except FileNotFoundError:
        return ExecResult(returncode=127, stdout="", stderr="", error=f"command not found: {cmd[0]}")
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "")
        return ExecResult(
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            error=f"timed out after {timeout_seconds} seconds",
        )


def _run_exec_streaming(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    output_path: Path | None = None,
) -> ExecResult:
    try:
        proc = subprocess.Popen(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except FileNotFoundError:
        return ExecResult(returncode=127, stdout="", stderr="", error=f"command not found: {cmd[0]}")

    combined: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            combined.append(line)
            typer.echo(line.rstrip("\n"), err=True)
            if output_path is not None:
                _append_text(output_path, line)
        proc.stdout.close()
        returncode = proc.wait()
        return ExecResult(returncode=returncode, stdout="".join(combined), stderr="")
    except KeyboardInterrupt:
        proc.stdout.close()
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
        raise


def _tail_text(path: Path, *, lines: int = 60) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    split = text.splitlines()
    return "\n".join(split[-lines:])


def _tail_lines(text: str, *, lines: int) -> str:
    split = text.splitlines()
    return "\n".join(split[-lines:])


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


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


def _terminate_pid_or_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
        return
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return


def _stop_pid(pid: int, *, timeout_seconds: float = 10.0) -> bool:
    _terminate_pid_or_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.1)
    _terminate_pid_or_group(pid, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.1)
    return not _pid_is_running(pid)


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


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _selection_identity(selection: dict[str, Any]) -> tuple[str, int, str] | None:
    model_key_raw = selection.get("model_key")
    launch_profile_raw = selection.get("launch_profile_key")
    port_profile_id = _coerce_int(selection.get("port_profile_id"))
    if not isinstance(model_key_raw, str) or not model_key_raw.strip():
        return None
    if not isinstance(launch_profile_raw, str) or not launch_profile_raw.strip():
        return None
    if port_profile_id is None:
        return None
    return model_key_raw.strip(), port_profile_id, launch_profile_raw.strip()


def _selection_identity_payload(selection: dict[str, Any]) -> dict[str, Any] | None:
    identity = _selection_identity(selection)
    if identity is None:
        return None
    model_key, port_profile_id, launch_profile_key = identity
    return {
        "model_key": model_key,
        "port_profile_id": port_profile_id,
        "launch_profile_key": launch_profile_key,
    }


def _state_matches_selection(state: dict[str, Any], selection: dict[str, Any]) -> bool:
    state_selection = state.get("selection")
    if not isinstance(state_selection, dict):
        return False
    return _selection_identity(state_selection) == _selection_identity(selection)


def _compose_context_for_selection(selection: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    identity = _selection_identity(selection)
    if identity is None:
        return False, {}, "invalid selection identity"

    runtime_names = selection.get("runtime_names")
    if not isinstance(runtime_names, dict):
        runtime_names = _runtime_names_for_selection(
            model_key=identity[0],
            launch_profile_key=identity[2],
        )

    project_name_raw = runtime_names.get("compose_project_name")
    if not isinstance(project_name_raw, str) or not project_name_raw.strip():
        return False, {}, "selection missing runtime compose project name"
    project_name = project_name_raw.strip()

    _ensure_runtime_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    token = f"{stamp}-{os.getpid()}-{int(time.time() * 1000) % 1000000}"
    env_file = RUNTIME_DIR / f"compose.target-stop.{token}.env"
    gpu_override_file = RUNTIME_DIR / f"compose.target-stop.{token}.gpu.override.yml"

    try:
        _write_compose_env(_compose_env_values(selection), output_path=env_file)
        _write_compose_gpu_override(selection, output_path=gpu_override_file)
    except Exception as exc:
        env_file.unlink(missing_ok=True)
        gpu_override_file.unlink(missing_ok=True)
        return False, {}, f"failed to materialize target compose context: {exc}"

    return (
        True,
        {
            "project_name": project_name,
            "env_file": env_file,
            "gpu_override_file": gpu_override_file,
        },
        "",
    )


def _cleanup_compose_context(context: dict[str, Any]) -> None:
    for key in ("env_file", "gpu_override_file"):
        path_value = context.get(key)
        if isinstance(path_value, Path):
            path_value.unlink(missing_ok=True)
        elif isinstance(path_value, str):
            Path(path_value).unlink(missing_ok=True)


def _current_port_profile_id(state: dict[str, Any], *, allow_env_fallback: bool = True) -> int | None:
    selection = state.get("selection")
    if isinstance(selection, dict):
        profile_id = _coerce_int(selection.get("port_profile_id"))
        if profile_id is not None:
            return profile_id
    if not allow_env_fallback:
        return None
    env_values = _parse_compose_env_file(COMPOSE_ENV_FILE)
    return _coerce_int(env_values.get("PORT_PROFILE_ID"))


def _gateway_runtime_files(*, port_profile_id: int) -> tuple[Path, Path]:
    suffix = _sanitize_path_token(str(port_profile_id))
    return (
        RUNTIME_DIR / f"{DEFAULT_GATEWAY_PID_FILE_PREFIX}.{suffix}.pid",
        RUNTIME_DIR / f"{DEFAULT_GATEWAY_LOG_FILE_PREFIX}.{suffix}.log",
    )


def _resolve_gateway_config_path() -> Path:
    if DEFAULT_GATEWAY_CONFIG_PATH.exists():
        return DEFAULT_GATEWAY_CONFIG_PATH
    return DEFAULT_GATEWAY_CONFIG_EXAMPLE_PATH


def _gateway_record_matches_running_process(record: dict[str, Any]) -> bool:
    raw_pid = record.get("pid")
    pid = _coerce_int(raw_pid)
    if pid is None or not _pid_is_running(pid):
        return False
    cmdline = _process_cmdline(pid)
    if not cmdline:
        return True
    normalized = cmdline.lower()
    if "gateway" not in normalized or " start " not in f" {normalized} ":
        return False
    profile_id = _coerce_int(record.get("port_profile_id"))
    if profile_id is not None:
        token = f"--port-profile-id {profile_id}"
        token_eq = f"--port-profile-id={profile_id}"
        if token not in cmdline and token_eq not in cmdline:
            return False
    return True


def _gateway_status_for_port_profile(port_profile_id: int) -> dict[str, Any]:
    pid_file, log_file = _gateway_runtime_files(port_profile_id=port_profile_id)
    record_raw = _read_json(pid_file, None)
    record = record_raw if isinstance(record_raw, dict) else None
    running = isinstance(record, dict) and _gateway_record_matches_running_process(record)
    pid = _coerce_int(record.get("pid")) if isinstance(record, dict) else None
    return {
        "running": running,
        "port_profile_id": port_profile_id,
        "pid": pid,
        "pid_file": str(pid_file),
        "log_file": str(log_file),
        "record": record,
    }


def _start_gateway_for_selection(selection: dict[str, Any]) -> dict[str, Any]:
    profile_id = _coerce_int(selection.get("port_profile_id"))
    ports = selection.get("ports")
    if profile_id is None:
        return _payload(
            ok=False,
            code=540,
            message="cannot start gateway: missing port profile id in selection",
        )
    if not isinstance(ports, dict):
        return _payload(
            ok=False,
            code=541,
            message="cannot start gateway: missing resolved ports in selection",
        )

    config_path = _resolve_gateway_config_path()
    if not config_path.exists():
        return _payload(
            ok=False,
            code=542,
            message=(
                "cannot start gateway: config file not found "
                f"(checked {DEFAULT_GATEWAY_CONFIG_PATH} and {DEFAULT_GATEWAY_CONFIG_EXAMPLE_PATH})"
            ),
        )

    _ensure_runtime_dir()
    pid_file, log_file = _gateway_runtime_files(port_profile_id=profile_id)
    active_record = _read_json(pid_file, None)
    if isinstance(active_record, dict):
        if _gateway_record_matches_running_process(active_record):
            return _payload(
                ok=True,
                code=0,
                message="gateway already running",
                data={"pid_file": str(pid_file), "record": active_record},
            )
        pid_file.unlink(missing_ok=True)

    gateway_cmd = [
        sys.executable,
        "-m",
        "gateway",
        "start",
        "--config",
        str(config_path),
        "--port-profile-id",
        str(profile_id),
        "--host",
        DEFAULT_GATEWAY_HOST,
        "--venv-dir",
        str(DEFAULT_GATEWAY_VENV_DIR),
        "--skip-install",
    ]

    env = dict(os.environ)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{py_path}" if py_path else str(REPO_ROOT)
    jaeger_api_port = _coerce_int(ports.get("jaeger_api_port"))
    jaeger_otlp_port = _coerce_int(ports.get("jaeger_otlp_port"))
    if jaeger_api_port is not None:
        env["GATEWAY_JAEGER_API_BASE_URL_OVERRIDE"] = f"http://127.0.0.1:{jaeger_api_port}/api/traces"
    if jaeger_otlp_port is not None:
        env["GATEWAY_OTLP_TRACES_ENDPOINT_OVERRIDE"] = f"grpc://127.0.0.1:{jaeger_otlp_port}"

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now_iso()}] starting gateway: {' '.join(gateway_cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            gateway_cmd,
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=env,
        )

    deadline = time.monotonic() + GATEWAY_STARTUP_PROBE_SECONDS
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            return _payload(
                ok=False,
                code=543,
                message=f"failed to start gateway daemon (exit={rc})",
                data={
                    "port_profile_id": profile_id,
                    "pid_file": str(pid_file),
                    "log_file": str(log_file),
                    "command": gateway_cmd,
                    "log_tail": _tail_text(log_file),
                },
            )
        time.sleep(0.1)

    record = {
        "pid": proc.pid,
        "started_at": _utc_now_iso(),
        "port_profile_id": profile_id,
        "config_path": str(config_path),
        "venv_dir": str(DEFAULT_GATEWAY_VENV_DIR),
        "host": DEFAULT_GATEWAY_HOST,
        "command": gateway_cmd,
        "pid_file": str(pid_file),
        "log_file": str(log_file),
        "jaeger_api_port": jaeger_api_port,
        "jaeger_otlp_port": jaeger_otlp_port,
    }
    _write_json(pid_file, record)
    return _payload(ok=True, code=0, message="gateway started", data=record)


def _stop_gateway_for_port_profile(port_profile_id: int) -> dict[str, Any]:
    _ensure_runtime_dir()
    pid_file, log_file = _gateway_runtime_files(port_profile_id=port_profile_id)
    record_raw = _read_json(pid_file, None)
    if not isinstance(record_raw, dict):
        pid_file.unlink(missing_ok=True)
        return _payload(
            ok=True,
            code=0,
            message=f"gateway is not running for port profile {port_profile_id}",
            data={
                "port_profile_id": port_profile_id,
                "pid_file": str(pid_file),
                "log_file": str(log_file),
            },
        )

    pid = _coerce_int(record_raw.get("pid"))
    if pid is None:
        pid_file.unlink(missing_ok=True)
        return _payload(
            ok=True,
            code=0,
            message="removed invalid gateway pid record",
            data={"port_profile_id": port_profile_id, "pid_file": str(pid_file), "record": record_raw},
        )

    if not _gateway_record_matches_running_process(record_raw):
        cmdline = _process_cmdline(pid) if _pid_is_running(pid) else ""
        pid_file.unlink(missing_ok=True)
        return _payload(
            ok=True,
            code=0,
            message=f"gateway is not running for port profile {port_profile_id} (stale pid file removed)",
            data={
                "port_profile_id": port_profile_id,
                "pid": pid,
                "pid_file": str(pid_file),
                "process_cmdline": cmdline,
            },
        )

    _terminate_pid_or_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            break
        time.sleep(0.1)

    forced = False
    if _pid_is_running(pid):
        forced = True
        _terminate_pid_or_group(pid, signal.SIGKILL)

    pid_file.unlink(missing_ok=True)
    _append_text(log_file, f"[{_utc_now_iso()}] stopped gateway pid={pid} forced={forced}\n")
    return _payload(
        ok=True,
        code=0,
        message="gateway stopped",
        data={
            "port_profile_id": port_profile_id,
            "pid": pid,
            "forced": forced,
            "pid_file": str(pid_file),
            "log_file": str(log_file),
        },
    )


def _load_daemon_record() -> dict[str, Any] | None:
    raw = _read_json(DAEMON_PID_FILE, None)
    return raw if isinstance(raw, dict) else None


def _daemon_running_record() -> dict[str, Any] | None:
    record = _load_daemon_record()
    if not isinstance(record, dict):
        return None
    raw_pid = record.get("pid")
    if isinstance(raw_pid, bool):
        return None
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return None
    return record if _pid_is_running(pid) else None


def _load_stop_state() -> dict[str, Any] | None:
    raw = _read_json(STOP_STATE_FILE, None)
    return raw if isinstance(raw, dict) else None


def _load_state() -> dict[str, Any]:
    raw = _read_json(STATE_FILE, {})
    if not isinstance(raw, dict):
        raw = {}
    state = {
        "active": False,
        "lifecycle_state": "inactive",
        "updated_at": _utc_now_iso(),
    }
    state.update(raw)
    return state


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now_iso()
    _write_json(STATE_FILE, state)


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _parse_port(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < 1 or value > 65535:
        raise ValueError(f"{key} must be in range 1..65535")
    return value


def _parse_float(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _parse_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _parse_non_empty_string(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _compact_text(value: object, *, max_length: int = 240) -> str | None:
    if not isinstance(value, str):
        return None
    compact = " ".join(value.strip().split())
    if not compact:
        return None
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 3]}..."


def _load_models_config() -> tuple[str | None, dict[str, dict[str, Any]]]:
    payload = _load_toml(MODEL_CONFIG_PATH)
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError("configs/model_config.toml must include [models]")
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw_models.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        vllm_model_name = value.get("vllm_model_name")
        served_model_name = value.get("served_model_name")
        extra_args = value.get("extra_args", [])
        if not isinstance(vllm_model_name, str) or not vllm_model_name:
            raise ValueError(f"models.{key}.vllm_model_name must be a non-empty string")
        if not isinstance(served_model_name, str) or not served_model_name:
            raise ValueError(f"models.{key}.served_model_name must be a non-empty string")
        if not isinstance(extra_args, list):
            raise ValueError(f"models.{key}.extra_args must be a list")
        weight_vram_gb = _parse_float(value.get("weight_vram_gb"), f"models.{key}.weight_vram_gb")
        out[key] = {
            "vllm_model_name": vllm_model_name,
            "served_model_name": served_model_name,
            "weight_vram_gb": weight_vram_gb,
            "extra_args": list(extra_args),
        }
    default_model = payload.get("default_model")
    if default_model is not None and not isinstance(default_model, str):
        raise ValueError("default_model in model config must be a string")
    return default_model, out


def _load_port_profiles() -> tuple[str | None, dict[str, dict[str, Any]]]:
    payload = _load_toml(PORT_PROFILES_PATH)
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise ValueError("configs/port_profiles.toml must include [profiles]")
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw_profiles.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        vllm_port = _parse_port(value.get("vllm_port"), f"profiles.{key}.vllm_port")
        gateway_port_raw = value.get("gateway_port")
        gateway_parse_port_raw = value.get("gateway_parse_port")
        gateway_port = _parse_port(gateway_port_raw, f"profiles.{key}.gateway_port") if gateway_port_raw is not None else None
        gateway_parse_port = (
            _parse_port(gateway_parse_port_raw, f"profiles.{key}.gateway_parse_port")
            if gateway_parse_port_raw is not None
            else None
        )
        jaeger_api_port = _parse_port(value.get("jaeger_api_port"), f"profiles.{key}.jaeger_api_port")
        jaeger_otlp_port = _parse_port(value.get("jaeger_otlp_port"), f"profiles.{key}.jaeger_otlp_port")
        lmcache_port = _parse_port(value.get("lmcache_port"), f"profiles.{key}.lmcache_port")
        out[key] = {
            "label": value.get("label"),
            "vllm_port": vllm_port,
            "gateway_port": gateway_port,
            "gateway_parse_port": gateway_parse_port,
            "jaeger_api_port": jaeger_api_port,
            "jaeger_otlp_port": jaeger_otlp_port,
            "lmcache_port": lmcache_port,
        }
    default_profile = payload.get("default_profile")
    if default_profile is not None and not isinstance(default_profile, str):
        raise ValueError("default_profile in port profile config must be a string")
    return default_profile, out


def _load_launch_profiles() -> tuple[str | None, dict[str, dict[str, Any]]]:
    payload = _load_toml(LAUNCH_PROFILES_PATH)
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise ValueError("servers/servers-docker/launch_profiles.toml must include [profiles]")
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw_profiles.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        visible_devices = value.get("visible_devices")
        gpu_type = value.get("gpu_type")
        if not isinstance(visible_devices, str) or not visible_devices:
            raise ValueError(f"profiles.{key}.visible_devices must be a non-empty string")
        visible_device_ids = [part.strip() for part in visible_devices.split(",") if part.strip()]
        if not visible_device_ids:
            raise ValueError(f"profiles.{key}.visible_devices must contain at least one device id")
        if not all(part.isdigit() for part in visible_device_ids):
            raise ValueError(
                f"profiles.{key}.visible_devices must be a comma-separated list of numeric GPU ids"
            )
        if not isinstance(gpu_type, str) or not gpu_type:
            raise ValueError(f"profiles.{key}.gpu_type must be a non-empty string")
        per_gpu_memory_gb = _parse_float(value.get("per_gpu_memory_gb"), f"profiles.{key}.per_gpu_memory_gb")
        total_gpu_memory_gb = _parse_float(value.get("total_gpu_memory_gb"), f"profiles.{key}.total_gpu_memory_gb")
        tensor_parallel_size = _parse_int(value.get("tensor_parallel_size"), f"profiles.{key}.tensor_parallel_size")
        out[key] = {
            "label": value.get("label"),
            "gpu_type": gpu_type,
            "visible_devices": visible_devices,
            "visible_device_ids": visible_device_ids,
            "per_gpu_memory_gb": per_gpu_memory_gb,
            "total_gpu_memory_gb": total_gpu_memory_gb,
            "tensor_parallel_size": tensor_parallel_size,
        }
    default_profile = payload.get("default_profile")
    if default_profile is not None and not isinstance(default_profile, str):
        raise ValueError("default_profile in launch profile config must be a string")
    return default_profile, out


def _load_image_config() -> dict[str, str]:
    payload = _load_toml(IMAGE_CONFIG_PATH)
    raw_images = payload.get("images")
    if not isinstance(raw_images, dict):
        raise ValueError("servers/servers-docker/service_images.toml must include [images]")
    return {
        "jaeger_image": _parse_non_empty_string(raw_images.get("jaeger"), "images.jaeger"),
        "vllm_image_name": _parse_non_empty_string(raw_images.get("vllm"), "images.vllm"),
    }


def _resolve_default_key(default_key: str | None, values: dict[str, Any], *, label: str) -> str:
    if default_key:
        if default_key not in values:
            raise ValueError(f"{label} default '{default_key}' not found in config")
        return default_key
    if not values:
        raise ValueError(f"{label} config is empty")
    return sorted(values.keys())[0]


def _model_requests_trust_remote_code(extra_args: list[Any]) -> bool:
    for arg in extra_args:
        if not isinstance(arg, str):
            continue
        normalized = arg.strip().lower()
        if normalized == "--trust-remote-code":
            return True
        if normalized.startswith("--trust-remote-code="):
            value = normalized.split("=", 1)[1]
            if value in {"1", "true", "yes", "on"}:
                return True
    return False


def _normalize_lmcache_size(lmcache_max_local_cpu_size: int | None) -> int | None:
    if lmcache_max_local_cpu_size is None:
        return None
    if isinstance(lmcache_max_local_cpu_size, bool) or not isinstance(lmcache_max_local_cpu_size, int):
        raise ValueError("lmcache must be a positive integer")
    if lmcache_max_local_cpu_size <= 0:
        raise ValueError("lmcache must be a positive integer")
    return lmcache_max_local_cpu_size


def _normalize_gpu_memory_utilization(gpu_memory_utilization: float | None) -> float | None:
    if gpu_memory_utilization is None:
        return None
    if isinstance(gpu_memory_utilization, bool) or not isinstance(gpu_memory_utilization, (int, float)):
        raise ValueError("gpu-memory-utilization must be > 0 and <= 1")
    normalized = float(gpu_memory_utilization)
    if not math.isfinite(normalized) or normalized <= 0 or normalized > 1:
        raise ValueError("gpu-memory-utilization must be > 0 and <= 1")
    return normalized


def _extract_vllm_option_value(args: list[str], option_name: str) -> str | None:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == option_name:
            index += 1
            if index >= len(args):
                raise ValueError(f"{option_name} requires a value")
            return args[index]
        if arg.startswith(f"{option_name}="):
            return arg.split("=", 1)[1]
        index += 1
    return None


def _apply_lmcache_option(
    *,
    extra_args: list[Any],
    lmcache_max_local_cpu_size: int | None,
) -> tuple[list[str], bool]:
    if not all(isinstance(arg, str) for arg in extra_args):
        raise ValueError("model extra_args must contain only strings")

    normalized_lmcache = _normalize_lmcache_size(lmcache_max_local_cpu_size)
    normalized_args = list(extra_args)
    if normalized_lmcache is None:
        return normalized_args, False

    existing_kv_transfer = _extract_vllm_option_value(normalized_args, "--kv-transfer-config")
    if existing_kv_transfer is not None:
        if existing_kv_transfer != LMCACHE_KV_TRANSFER_CONFIG:
            raise ValueError(
                "model extra_args already configure --kv-transfer-config with a different value"
            )
        return normalized_args, True

    normalized_args.extend(["--kv-transfer-config", LMCACHE_KV_TRANSFER_CONFIG])
    return normalized_args, True


def _apply_gpu_memory_utilization_option(
    *,
    extra_args: list[Any],
    gpu_memory_utilization: float | None,
) -> tuple[list[str], bool]:
    if not all(isinstance(arg, str) for arg in extra_args):
        raise ValueError("model extra_args must contain only strings")

    normalized_gpu_memory_utilization = _normalize_gpu_memory_utilization(gpu_memory_utilization)
    normalized_args: list[str] = []
    if normalized_gpu_memory_utilization is None:
        return list(extra_args), False

    skip_next = False
    for index, arg in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--gpu-memory-utilization":
            if index + 1 >= len(extra_args):
                raise ValueError("--gpu-memory-utilization requires a value")
            skip_next = True
            continue
        if arg.startswith("--gpu-memory-utilization="):
            continue
        normalized_args.append(arg)

    normalized_args.extend(
        [
            "--gpu-memory-utilization",
            str(normalized_gpu_memory_utilization),
        ]
    )
    return normalized_args, True


def _encode_model_extra_args(extra_args: list[Any]) -> str:
    if not all(isinstance(arg, str) for arg in extra_args):
        raise ValueError("model extra_args must contain only strings")
    payload = json.dumps(extra_args, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _default_selection() -> dict[str, Any]:
    default_model, models = _load_models_config()
    default_port_profile, port_profiles = _load_port_profiles()
    default_launch_profile, launch_profiles = _load_launch_profiles()
    images = _load_image_config()

    model_key = _resolve_default_key(default_model, models, label="model")
    port_key = _resolve_default_key(default_port_profile, port_profiles, label="port profile")
    launch_key = _resolve_default_key(default_launch_profile, launch_profiles, label="launch profile")

    return {
        "model_key": model_key,
        "port_profile_id": port_key,
        "launch_profile_key": launch_key,
        "images": images,
        "model": models[model_key],
        "ports": port_profiles[port_key],
        "launch": launch_profiles[launch_key],
        "runtime_names": _runtime_names_for_selection(model_key=model_key, launch_profile_key=launch_key),
    }


def _compose_env_values(selection: dict[str, Any]) -> dict[str, str]:
    images = selection["images"]
    model = selection["model"]
    ports = selection["ports"]
    launch = selection["launch"]
    port_profile_id = selection.get("port_profile_id")
    port_profile_id_text = str(port_profile_id) if port_profile_id is not None else ""
    model_extra_args = model.get("extra_args", [])
    raw_lmcache = selection.get("lmcache")
    if raw_lmcache is None:
        lmcache_max_local_cpu_size = None
    else:
        coerced_lmcache = _coerce_int(raw_lmcache)
        if coerced_lmcache is None:
            raise ValueError("selection.lmcache must be a positive integer")
        lmcache_max_local_cpu_size = _normalize_lmcache_size(coerced_lmcache)
    gpu_memory_utilization = _normalize_gpu_memory_utilization(selection.get("gpu_memory_utilization"))
    effective_extra_args, _ = _apply_lmcache_option(
        extra_args=model_extra_args,
        lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
    )
    effective_extra_args, _ = _apply_gpu_memory_utilization_option(
        extra_args=effective_extra_args,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    trust_remote_code = _model_requests_trust_remote_code(model_extra_args)
    lmcache_port = _parse_port(ports.get("lmcache_port"), "resolved.ports.lmcache_port")
    runtime_names = selection.get("runtime_names")
    if not isinstance(runtime_names, dict):
        model_key = str(selection.get("model_key", "default-model"))
        launch_profile_key = str(selection.get("launch_profile_key", "default-launch"))
        runtime_names = _runtime_names_for_selection(model_key=model_key, launch_profile_key=launch_profile_key)
    return {
        "COMPOSE_PROJECT_NAME": str(runtime_names["compose_project_name"]),
        "JAEGER_IMAGE": str(images["jaeger_image"]),
        "JAEGER_CONTAINER_NAME": str(runtime_names["jaeger_container_name"]),
        "VLLM_IMAGE_NAME": str(images["vllm_image_name"]),
        "VLLM_CONTAINER_NAME": str(runtime_names["vllm_container_name"]),
        "PORT_PROFILE_ID": port_profile_id_text,
        "VLLM_FORCE_SEQ_TRUST_REMOTE_CODE": "true" if trust_remote_code else "false",
        "VLLM_MODEL_EXTRA_ARGS_B64": _encode_model_extra_args(effective_extra_args),
        "VLLM_MODEL_NAME": str(model["vllm_model_name"]),
        "VLLM_SERVED_MODEL_NAME": str(model["served_model_name"]),
        "VLLM_SERVICE_PORT": str(ports["vllm_port"]),
        "VLLM_TENSOR_PARALLEL_SIZE": str(launch["tensor_parallel_size"]),
        "JAEGER_API_PORT": str(ports["jaeger_api_port"]),
        "JAEGER_OTLP_PORT": str(ports["jaeger_otlp_port"]),
        "LMCACHE_INTERNAL_API_SERVER_ENABLED": "1",
        "LMCACHE_INTERNAL_API_SERVER_PORT_START": str(lmcache_port),
        "LMCACHE_MAX_LOCAL_CPU_SIZE": (
            str(lmcache_max_local_cpu_size) if lmcache_max_local_cpu_size is not None else ""
        ),
        "OTEL_SERVICE_NAME": str(runtime_names["otel_service_name"]),
        "PYTHONHASHSEED": "0",
    }


def _compose_env_text(values: dict[str, str]) -> str:
    lines = [
        "# Auto-generated by servers/servers-docker/client.py",
        "# This file is managed internally and should not be edited manually.",
    ]
    for key in sorted(values):
        lines.append(f"{key}={values[key]}")
    return "\n".join(lines) + "\n"


def _write_compose_env(values: dict[str, str], *, output_path: Path = COMPOSE_ENV_FILE) -> None:
    _ensure_runtime_dir()
    output_path.write_text(_compose_env_text(values), encoding="utf-8")


def _compose_gpu_override_yaml(selection: dict[str, Any]) -> str:
    launch = selection["launch"]
    device_ids = launch.get("visible_device_ids")
    if not isinstance(device_ids, list) or not device_ids:
        raise ValueError("launch.visible_device_ids must be a non-empty list")
    lines = [
        "# Auto-generated by servers/servers-docker/client.py",
        "services:",
        "  vllm:",
        "    deploy:",
        "      resources:",
        "        reservations:",
        "          devices:",
        "            - driver: nvidia",
        "              device_ids:",
    ]
    for device_id in device_ids:
        lines.append(f'                - "{device_id}"')
    lines.extend(
        [
            "              capabilities:",
            "                - gpu",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_compose_gpu_override(selection: dict[str, Any], *, output_path: Path = COMPOSE_GPU_OVERRIDE_FILE) -> None:
    _ensure_runtime_dir()
    output_path.write_text(_compose_gpu_override_yaml(selection), encoding="utf-8")


def _parse_compose_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _selection_from_existing_compose_env() -> dict[str, Any] | None:
    env_values = _parse_compose_env_file(COMPOSE_ENV_FILE)
    visible_devices = env_values.get("VLLM_VISIBLE_DEVICES")
    if not visible_devices:
        return None
    device_ids = [part.strip() for part in visible_devices.split(",") if part.strip()]
    if not device_ids or not all(part.isdigit() for part in device_ids):
        return None
    selection = _default_selection()
    selection["launch"] = dict(selection["launch"])
    selection["launch"]["visible_devices"] = visible_devices
    selection["launch"]["visible_device_ids"] = device_ids
    return selection


def _ensure_compose_env_file() -> tuple[bool, str]:
    if COMPOSE_ENV_FILE.exists() and COMPOSE_GPU_OVERRIDE_FILE.exists():
        return True, str(COMPOSE_ENV_FILE)
    try:
        if not COMPOSE_ENV_FILE.exists():
            selection = _default_selection()
            _write_compose_env(_compose_env_values(selection))
            _write_compose_gpu_override(selection)
        elif not COMPOSE_GPU_OVERRIDE_FILE.exists():
            selection = _selection_from_existing_compose_env()
            if selection is None:
                selection = _default_selection()
            _write_compose_gpu_override(selection)
    except Exception as exc:
        return False, str(exc)
    return True, str(COMPOSE_ENV_FILE)


def _compose_cmd(
    args: list[str],
    *,
    project_name: str | None = None,
    compose_env_file: Path = COMPOSE_ENV_FILE,
    compose_gpu_override_file: Path = COMPOSE_GPU_OVERRIDE_FILE,
) -> list[str]:
    effective_project_name = project_name or _compose_project_name()
    return [
        "docker",
        "compose",
        "--project-name",
        effective_project_name,
        "-f",
        str(COMPOSE_FILE),
        "-f",
        str(compose_gpu_override_file),
        "--env-file",
        str(compose_env_file),
        *args,
    ]


def _compose_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in COMPOSE_MANAGED_ENV_KEYS:
        env.pop(key, None)
    return env


def _run_compose(
    args: list[str],
    *,
    timeout_seconds: float | None = None,
    project_name: str | None = None,
    compose_env_file: Path = COMPOSE_ENV_FILE,
    compose_gpu_override_file: Path = COMPOSE_GPU_OVERRIDE_FILE,
) -> ExecResult:
    return _run_exec(
        _compose_cmd(
            args,
            project_name=project_name,
            compose_env_file=compose_env_file,
            compose_gpu_override_file=compose_gpu_override_file,
        ),
        timeout_seconds=timeout_seconds,
        env=_compose_subprocess_env(),
    )


def _run_compose_streaming(
    args: list[str],
    *,
    output_path: Path | None = None,
    project_name: str | None = None,
    compose_env_file: Path = COMPOSE_ENV_FILE,
    compose_gpu_override_file: Path = COMPOSE_GPU_OVERRIDE_FILE,
) -> ExecResult:
    return _run_exec_streaming(
        _compose_cmd(
            args,
            project_name=project_name,
            compose_env_file=compose_env_file,
            compose_gpu_override_file=compose_gpu_override_file,
        ),
        env=_compose_subprocess_env(),
        output_path=output_path,
    )


def _compose_running_services(
    *,
    project_name: str | None = None,
    compose_env_file: Path = COMPOSE_ENV_FILE,
    compose_gpu_override_file: Path = COMPOSE_GPU_OVERRIDE_FILE,
    ensure_env_file: bool = True,
) -> tuple[bool, set[str], str]:
    if ensure_env_file:
        env_ok, env_msg = _ensure_compose_env_file()
        if not env_ok:
            return False, set(), f"failed to materialize compose env file: {env_msg}"
    result = _run_compose(
        ["ps", "--services", "--status", "running"],
        project_name=project_name,
        compose_env_file=compose_env_file,
        compose_gpu_override_file=compose_gpu_override_file,
    )
    if result.error:
        return False, set(), result.error
    if result.returncode != 0:
        err = (result.stderr or result.stdout).strip()
        return False, set(), err or "docker compose ps failed"
    services = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return True, services, ""


def _service_urls_from_ports(ports: dict[str, Any]) -> dict[str, str]:
    urls = {
        "vllm": f"http://127.0.0.1:{ports['vllm_port']}",
        "jaeger_api": f"http://127.0.0.1:{ports['jaeger_api_port']}",
        "jaeger_ui": f"http://127.0.0.1:{ports['jaeger_api_port']}",
        "jaeger_otlp": f"grpc://127.0.0.1:{ports['jaeger_otlp_port']}",
    }
    gateway_port = _coerce_int(ports.get("gateway_port"))
    gateway_parse_port = _coerce_int(ports.get("gateway_parse_port"))
    lmcache_port = _coerce_int(ports.get("lmcache_port"))
    if gateway_port is not None:
        urls["gateway"] = f"http://127.0.0.1:{gateway_port}"
    if gateway_parse_port is not None:
        urls["gateway_parse"] = f"http://127.0.0.1:{gateway_parse_port}"
    if lmcache_port is not None:
        urls["lmcache_metrics"] = f"http://127.0.0.1:{lmcache_port}/metrics"
    return urls


def _refresh_state_from_compose() -> dict[str, Any]:
    state = _load_state()
    compose_ok, services, compose_error = _compose_running_services()
    if compose_ok:
        running = bool({"vllm", "jaeger"} & services)
        if state.get("active") and not running:
            state["active"] = False
            state["lifecycle_state"] = "inactive"
            state["last_transition"] = "compose_not_running"
            _save_state(state)
    else:
        state["compose_check_error"] = compose_error
        _save_state(state)
    return state


def _docker_container_snapshot(container_name: str) -> dict[str, Any]:
    result = _run_exec(["docker", "inspect", container_name], timeout_seconds=10.0)
    if result.error:
        return {
            "name": container_name,
            "present": False,
            "inspect_error": result.error,
        }
    if result.returncode != 0:
        inspect_error = _compact_text(result.stderr or result.stdout) or f"docker inspect failed: rc={result.returncode}"
        return {
            "name": container_name,
            "present": False,
            "inspect_error": inspect_error,
        }
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "name": container_name,
            "present": False,
            "inspect_error": f"failed to parse docker inspect output: {exc}",
        }
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
        return {
            "name": container_name,
            "present": False,
            "inspect_error": "docker inspect returned unexpected payload",
        }

    payload = raw[0]
    state = payload.get("State")
    if not isinstance(state, dict):
        state = {}

    restart_count_raw = payload.get("RestartCount", 0)
    try:
        restart_count = int(restart_count_raw)
    except (TypeError, ValueError):
        restart_count = 0

    return {
        "name": container_name,
        "present": True,
        "restart_count": restart_count,
        "status": state.get("Status"),
        "running": bool(state.get("Running")),
        "restarting": bool(state.get("Restarting")),
        "oom_killed": bool(state.get("OOMKilled")),
        "dead": bool(state.get("Dead")),
        "error": state.get("Error"),
        "exit_code": state.get("ExitCode"),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
    }


def _docker_logs_tail(container_name: str, *, lines: int = DEFAULT_STARTUP_LOG_TAIL_LINES) -> dict[str, Any]:
    result = _run_exec(["docker", "logs", "--tail", str(lines), container_name], timeout_seconds=10.0)
    if result.error:
        return {
            "ok": False,
            "container_name": container_name,
            "lines": lines,
            "logs": "",
            "error": result.error,
        }
    if result.returncode != 0:
        return {
            "ok": False,
            "container_name": container_name,
            "lines": lines,
            "logs": "",
            "error": _compact_text(result.stderr or result.stdout) or f"docker logs failed: rc={result.returncode}",
        }

    merged_logs = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    return {
        "ok": True,
        "container_name": container_name,
        "lines": lines,
        "logs": merged_logs,
        "error": None,
    }


def _docker_logs_full(container_name: str) -> dict[str, Any]:
    result = _run_exec(["docker", "logs", container_name], timeout_seconds=20.0)
    if result.error:
        return {
            "ok": False,
            "container_name": container_name,
            "logs": "",
            "error": result.error,
        }
    if result.returncode != 0:
        return {
            "ok": False,
            "container_name": container_name,
            "logs": "",
            "error": _compact_text(result.stderr or result.stdout) or f"docker logs failed: rc={result.returncode}",
        }
    merged_logs = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    return {
        "ok": True,
        "container_name": container_name,
        "logs": merged_logs,
        "error": None,
    }


def _compose_logs_full(
    *,
    services: list[str] | None = None,
    project_name: str | None = None,
    compose_env_file: Path = COMPOSE_ENV_FILE,
    compose_gpu_override_file: Path = COMPOSE_GPU_OVERRIDE_FILE,
    ensure_env_file: bool = True,
) -> dict[str, Any]:
    if ensure_env_file:
        env_ok, env_msg = _ensure_compose_env_file()
        if not env_ok:
            return {
                "ok": False,
                "logs": "",
                "error": f"failed to materialize compose env file: {env_msg}",
            }
    compose_args = ["logs", "--no-color", "--timestamps", *(services or ["jaeger", "vllm"])]
    result = _run_compose(
        compose_args,
        project_name=project_name,
        compose_env_file=compose_env_file,
        compose_gpu_override_file=compose_gpu_override_file,
    )
    if result.error:
        return {
            "ok": False,
            "logs": "",
            "error": result.error,
        }
    if result.returncode != 0:
        return {
            "ok": False,
            "logs": "",
            "error": _compact_text(result.stderr or result.stdout) or f"docker compose logs failed: rc={result.returncode}",
        }
    merged_logs = "".join(part for part in (result.stdout, result.stderr) if part)
    return {
        "ok": True,
        "logs": _strip_ansi(merged_logs),
        "error": None,
    }


def _sync_compose_logs_to_file(
    *,
    log_path: Path | None,
    previous_logs: str,
    project_name: str | None = None,
    compose_env_file: Path = COMPOSE_ENV_FILE,
    compose_gpu_override_file: Path = COMPOSE_GPU_OVERRIDE_FILE,
    ensure_env_file: bool = True,
) -> str:
    if log_path is None:
        return previous_logs

    captured = _compose_logs_full(
        project_name=project_name,
        compose_env_file=compose_env_file,
        compose_gpu_override_file=compose_gpu_override_file,
        ensure_env_file=ensure_env_file,
    )
    if not captured.get("ok"):
        return previous_logs

    current_logs = str(captured.get("logs", ""))
    if not current_logs.strip():
        return previous_logs
    if current_logs != previous_logs:
        normalized_logs = current_logs if current_logs.endswith("\n") else f"{current_logs}\n"
        _write_text(log_path, normalized_logs)
    return current_logs


def _extract_log_hint(log_text: str) -> str | None:
    interesting_markers = (
        "Value error,",
        "ValidationError",
        "Traceback (most recent call last):",
        "RuntimeError:",
        "ModuleNotFoundError:",
        "ImportError:",
        "CUDA out of memory",
        "Connection reset by peer",
    )
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    for line in reversed(lines):
        for marker in interesting_markers:
            if marker in line:
                return _compact_text(line, max_length=320)
    for line in reversed(lines):
        lowered = line.lower()
        if "error" in lowered or "exception" in lowered:
            return _compact_text(line, max_length=320)
    if not lines:
        return None
    return _compact_text(lines[-1], max_length=320)


def _format_health_service_status(name: str, payload: dict[str, Any]) -> str:
    if payload.get("ok"):
        return f"{name}=ok"
    status_code = payload.get("status_code")
    if status_code is not None:
        return f"{name}=status:{status_code}"
    error_text = _compact_text(payload.get("error"), max_length=120)
    return f"{name}=waiting:{error_text or 'unavailable'}"


def _format_wait_progress(snapshot: dict[str, Any], *, attempts: int, elapsed_seconds: float) -> str:
    services = snapshot.get("services")
    containers = snapshot.get("containers")
    diagnostics = snapshot.get("diagnostics")

    parts = [f"wait-up attempt={attempts} elapsed={elapsed_seconds:.1f}s"]

    if isinstance(services, dict):
        for name in ("jaeger_api", "jaeger_otlp", "vllm", "gateway", "gateway_parse"):
            payload = services.get(name)
            if isinstance(payload, dict):
                parts.append(_format_health_service_status(name, payload))

    if isinstance(containers, dict):
        vllm_container = containers.get("vllm")
        if isinstance(vllm_container, dict):
            status = vllm_container.get("status") or "unknown"
            restart_count = vllm_container.get("restart_count")
            parts.append(f"vllm_container={status}")
            if isinstance(restart_count, int):
                parts.append(f"vllm_restarts={restart_count}")

    if isinstance(diagnostics, dict):
        vllm_diag = diagnostics.get("vllm")
        if isinstance(vllm_diag, dict):
            hint = _compact_text(vllm_diag.get("log_hint"), max_length=180)
            if hint:
                parts.append(f"log_hint={hint}")

    return " | ".join(parts)


def _startup_failure_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    services = snapshot.get("services")
    containers = snapshot.get("containers")
    diagnostics = snapshot.get("diagnostics")
    if not isinstance(services, dict) or not isinstance(containers, dict):
        return None

    vllm_service = services.get("vllm")
    vllm_container = containers.get("vllm")
    if not isinstance(vllm_service, dict) or not isinstance(vllm_container, dict):
        return None
    if vllm_service.get("ok"):
        return None

    restart_count = vllm_container.get("restart_count")
    oom_killed = bool(vllm_container.get("oom_killed"))
    log_hint = None
    logs_tail = None
    if isinstance(diagnostics, dict):
        vllm_diag = diagnostics.get("vllm")
        if isinstance(vllm_diag, dict):
            log_hint = vllm_diag.get("log_hint")
            recent_logs = vllm_diag.get("recent_logs")
            if isinstance(recent_logs, dict):
                logs_tail = recent_logs.get("logs")

    if oom_killed:
        return {
            "code": 507,
            "message": "vllm startup failed: container was OOM-killed",
            "data": {
                "reason": "container OOMKilled during startup",
                "container": vllm_container,
                "service_check": vllm_service,
                "logs_tail": logs_tail,
            },
        }

    if isinstance(restart_count, int) and restart_count >= VLLM_CRASH_LOOP_RESTART_THRESHOLD and log_hint:
        return {
            "code": 506,
            "message": "vllm crash loop detected during startup",
            "data": {
                "reason": log_hint,
                "container": vllm_container,
                "service_check": vllm_service,
                "logs_tail": logs_tail,
            },
        }

    return None


def _daemon_status_payload() -> dict[str, Any]:
    daemon_record = _daemon_running_record()
    state = _refresh_state_from_compose()
    port_profile_id = _current_port_profile_id(state, allow_env_fallback=False)
    gateway_status = _gateway_status_for_port_profile(port_profile_id) if port_profile_id is not None else None
    compose_ok, services, compose_error = _compose_running_services()
    return _payload(
        ok=True,
        code=0,
        message="daemon status",
        data={
            "daemon": {
                "running": daemon_record is not None,
                "record": daemon_record,
                "pid_file": str(DAEMON_PID_FILE),
                "log_file": str(DAEMON_LOG_FILE),
            },
            "active_environment": state if state.get("active") else None,
            "gateway": gateway_status,
            "compose": {
                "ok": compose_ok,
                "running_services": sorted(services) if compose_ok else [],
                "error": compose_error if not compose_ok else None,
            },
        },
    )


def _start_daemon_impl() -> dict[str, Any]:
    _ensure_runtime_dir()
    active = _daemon_running_record()
    if active is not None:
        return _payload(ok=True, code=0, message="daemon already running", data={"daemon": active})

    DAEMON_PID_FILE.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "__daemon-loop__"]
    with DAEMON_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now_iso()}] starting daemon: {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    time.sleep(0.25)
    if proc.poll() is not None:
        return _payload(
            ok=False,
            code=301,
            message=f"failed to start daemon (exit={proc.returncode})",
            data={"log_tail": _tail_text(DAEMON_LOG_FILE)},
        )

    record = {
        "pid": proc.pid,
        "started_at": _utc_now_iso(),
        "pid_file": str(DAEMON_PID_FILE),
        "log_file": str(DAEMON_LOG_FILE),
        "heartbeat_file": str(DAEMON_HEARTBEAT_FILE),
    }
    _write_json(DAEMON_PID_FILE, record)
    return _payload(ok=True, code=0, message="daemon started", data={"daemon": record})


def _ensure_daemon_running() -> tuple[bool, dict[str, Any]]:
    running = _daemon_running_record()
    if running is not None:
        return True, _payload(ok=True, code=0, message="daemon already running", data={"daemon": running})
    started = _start_daemon_impl()
    return bool(started.get("ok")), started


def _http_get_health(url: str, *, timeout_seconds: float = 2.5, allow_redirect: bool = True) -> dict[str, Any]:
    started = time.monotonic()
    req = urlrequest.Request(url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as response:
            status = int(response.status)
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
            ok = 200 <= status < 400 if allow_redirect else 200 <= status < 300
            return {
                "ok": ok,
                "status_code": status,
                "url": url,
                "latency_ms": elapsed_ms,
                "error": None,
            }
    except urlerror.HTTPError as exc:
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
        status = int(exc.code)
        ok = 200 <= status < 400 if allow_redirect else 200 <= status < 300
        return {
            "ok": ok,
            "status_code": status,
            "url": url,
            "latency_ms": elapsed_ms,
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover - network boundary
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
        return {
            "ok": False,
            "status_code": None,
            "url": url,
            "latency_ms": elapsed_ms,
            "error": str(exc),
        }


def _tcp_health(host: str, port: int, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
            return {
                "ok": True,
                "host": host,
                "port": port,
                "latency_ms": elapsed_ms,
                "error": None,
            }
    except Exception as exc:  # pragma: no cover - network boundary
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
        return {
            "ok": False,
            "host": host,
            "port": port,
            "latency_ms": elapsed_ms,
            "error": str(exc),
        }


def _build_health_snapshot(
    state: dict[str, Any],
    *,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if selection is None:
        resolved = state.get("resolved")
        if not isinstance(resolved, dict):
            raise ValueError("state missing resolved config")
        ports = resolved.get("ports")
        if not isinstance(ports, dict):
            raise ValueError("state missing resolved.ports")
        runtime_names = _runtime_names_from_state(state)
        port_profile_id = _current_port_profile_id(state)
    else:
        ports = selection.get("ports")
        if not isinstance(ports, dict):
            raise ValueError("selection missing ports")
        runtime_names = _runtime_names_from_selection(selection)
        port_profile_id = _coerce_int(selection.get("port_profile_id"))

    vllm_port = _parse_port(ports.get("vllm_port"), "resolved.ports.vllm_port")
    gateway_port = _coerce_int(ports.get("gateway_port"))
    gateway_parse_port = _coerce_int(ports.get("gateway_parse_port"))
    jaeger_api_port = _parse_port(ports.get("jaeger_api_port"), "resolved.ports.jaeger_api_port")
    jaeger_otlp_port = _parse_port(ports.get("jaeger_otlp_port"), "resolved.ports.jaeger_otlp_port")

    checks = {
        "vllm": _http_get_health(f"http://127.0.0.1:{vllm_port}/v1/models", allow_redirect=False),
        "jaeger_api": _http_get_health(f"http://127.0.0.1:{jaeger_api_port}/api/services", allow_redirect=False),
        "jaeger_ui": _http_get_health(f"http://127.0.0.1:{jaeger_api_port}/", allow_redirect=True),
        "jaeger_otlp": _tcp_health("127.0.0.1", jaeger_otlp_port),
    }
    if gateway_port is not None:
        checks["gateway"] = _http_get_health(f"http://127.0.0.1:{gateway_port}/healthz", allow_redirect=False)
    if gateway_parse_port is not None:
        checks["gateway_parse"] = _http_get_health(
            f"http://127.0.0.1:{gateway_parse_port}/healthz",
            allow_redirect=False,
        )
    containers = {
        "jaeger": _docker_container_snapshot(runtime_names["jaeger_container_name"]),
        "vllm": _docker_container_snapshot(runtime_names["vllm_container_name"]),
    }
    gateway_daemon = _gateway_status_for_port_profile(port_profile_id) if port_profile_id is not None else None
    diagnostics: dict[str, Any] = {}

    vllm_container = containers["vllm"]
    should_capture_vllm_logs = not checks["vllm"]["ok"] or bool(vllm_container.get("restart_count"))
    if should_capture_vllm_logs:
        vllm_logs = _docker_logs_tail(runtime_names["vllm_container_name"])
        vllm_diag = {
            "log_hint": _extract_log_hint(vllm_logs.get("logs", "")) if vllm_logs.get("ok") else None,
            "recent_logs": {
                "lines": vllm_logs.get("lines"),
                "logs": _tail_lines(vllm_logs.get("logs", ""), lines=DEFAULT_STARTUP_LOG_TAIL_LINES),
                "error": vllm_logs.get("error"),
            },
        }
        diagnostics["vllm"] = vllm_diag

    return {
        "ok": all(bool(item.get("ok")) for item in checks.values()),
        "checked_at": _utc_now_iso(),
        "runtime_names": runtime_names,
        "gateway_daemon": gateway_daemon,
        "containers": containers,
        "diagnostics": diagnostics,
        "services": checks,
    }


def _up_impl(*, selection: dict[str, Any] | None = None) -> dict[str, Any]:
    if selection is None:
        state = _refresh_state_from_compose()
        if not state.get("active"):
            return _payload(ok=False, code=404, message="no active environment")
    else:
        state = _load_state()
    try:
        snapshot = _build_health_snapshot(state, selection=selection)
    except Exception as exc:
        return _payload(ok=False, code=500, message=f"failed to build health snapshot: {exc}")
    if selection is None:
        state["last_health"] = snapshot
        _save_state(state)
    if snapshot["ok"]:
        return _payload(ok=True, code=0, message="all services healthy", data=snapshot)
    return _payload(ok=False, code=503, message="one or more services unhealthy", data=snapshot)


def _wait_up_impl(
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    show_progress: bool = False,
    startup_log_path: Path | None = None,
    compose_log_path: Path | None = None,
    previous_compose_logs: str = "",
    selection: dict[str, Any] | None = None,
    compose_project_name: str | None = None,
    compose_env_file: Path = COMPOSE_ENV_FILE,
    compose_gpu_override_file: Path = COMPOSE_GPU_OVERRIDE_FILE,
    ensure_compose_env_file: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + max(timeout_seconds, 0.0)
    attempts = 0
    last_snapshot: dict[str, Any] | None = None
    captured_compose_logs = previous_compose_logs
    while True:
        attempts += 1
        up_payload = _up_impl(selection=selection)
        elapsed = round(time.monotonic() - started, 3)
        captured_compose_logs = _sync_compose_logs_to_file(
            log_path=compose_log_path,
            previous_logs=captured_compose_logs,
            project_name=compose_project_name,
            compose_env_file=compose_env_file,
            compose_gpu_override_file=compose_gpu_override_file,
            ensure_env_file=ensure_compose_env_file,
        )
        if up_payload.get("ok"):
            data = dict(up_payload.get("data", {}))
            data.update({"attempts": attempts, "elapsed_seconds": elapsed})
            if show_progress:
                _emit_progress(
                    "wait-up completed successfully"
                    f" after {elapsed:.1f}s",
                    log_path=startup_log_path,
                )
            return _payload(ok=True, code=0, message="services are up", data=data)
        last_snapshot = up_payload.get("data") if isinstance(up_payload.get("data"), dict) else None
        if show_progress and isinstance(last_snapshot, dict):
            _emit_progress(
                _format_wait_progress(last_snapshot, attempts=attempts, elapsed_seconds=elapsed),
                log_path=startup_log_path,
            )
        if isinstance(last_snapshot, dict):
            startup_failure = _startup_failure_from_snapshot(last_snapshot)
            if startup_failure is not None:
                data = dict(startup_failure.get("data", {}))
                data.update(
                    {
                        "attempts": attempts,
                        "elapsed_seconds": elapsed,
                        "last_snapshot": last_snapshot,
                    }
                )
                return _payload(
                    ok=False,
                    code=int(startup_failure.get("code", 506)),
                    message=str(startup_failure.get("message", "startup failed")),
                    data=data,
                )
        if time.monotonic() >= deadline:
            return _payload(
                ok=False,
                code=408,
                message=f"timed out waiting for healthy services after {elapsed}s",
                data={
                    "attempts": attempts,
                    "elapsed_seconds": elapsed,
                    "last_snapshot": last_snapshot,
                },
            )
        time.sleep(max(poll_interval_seconds, 0.2))


def _validate_start_selection(
    *,
    model_key: str,
    port_profile_id: int,
    launch_profile_key: str,
    lmcache: int | None = None,
    gpu_memory_utilization: float | None = None,
    enforce_weight_limit: bool = True,
) -> tuple[bool, dict[str, Any]]:
    try:
        images = _load_image_config()
        _, models = _load_models_config()
        _, port_profiles = _load_port_profiles()
        _, launch_profiles = _load_launch_profiles()
    except Exception as exc:
        return False, _payload(ok=False, code=410, message=f"failed loading configs: {exc}")
    try:
        normalized_lmcache = _normalize_lmcache_size(lmcache)
        normalized_gpu_memory_utilization = _normalize_gpu_memory_utilization(gpu_memory_utilization)
    except ValueError as exc:
        return False, _payload(ok=False, code=414, message=str(exc))

    port_key = str(port_profile_id)
    model = models.get(model_key)
    if model is None:
        return False, _payload(ok=False, code=411, message=f"unknown model key: {model_key}")

    port_profile = port_profiles.get(port_key)
    if port_profile is None:
        return False, _payload(ok=False, code=412, message=f"unknown port profile id: {port_profile_id}")

    launch_profile = launch_profiles.get(launch_profile_key)
    if launch_profile is None:
        return False, _payload(ok=False, code=413, message=f"unknown launch profile: {launch_profile_key}")

    model_weight = float(model["weight_vram_gb"])
    total_gpu_memory = float(launch_profile["total_gpu_memory_gb"])
    if enforce_weight_limit and model_weight > (0.75 * total_gpu_memory):
        return False, _payload(
            ok=False,
            code=422,
            message=(
                "launch rejected: model weight exceeds 75% of launch profile total GPU memory "
                f"(weight={model_weight}GB, limit={round(0.75 * total_gpu_memory, 3)}GB)"
            ),
            data={
                "model_key": model_key,
                "weight_vram_gb": model_weight,
                "launch_profile": launch_profile_key,
                "total_gpu_memory_gb": total_gpu_memory,
                "max_allowed_weight_gb": round(0.75 * total_gpu_memory, 3),
            },
        )

    resolved = {
        "model_key": model_key,
        "port_profile_id": port_key,
        "launch_profile_key": launch_profile_key,
        "lmcache": normalized_lmcache,
        "gpu_memory_utilization": normalized_gpu_memory_utilization,
        "images": images,
        "model": model,
        "ports": port_profile,
        "launch": launch_profile,
        "service_urls": _service_urls_from_ports(port_profile),
        "runtime_names": _runtime_names_for_selection(model_key=model_key, launch_profile_key=launch_profile_key),
    }
    return True, _payload(ok=True, code=0, message="selection validated", data=resolved)


def _resolve_stop_selection(
    *,
    model_key: str | None,
    port_profile_id: int | None,
    launch_profile_key: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    provided_count = sum(value is not None for value in (model_key, port_profile_id, launch_profile_key))
    if provided_count == 0:
        return None, None
    if provided_count != 3:
        return None, _payload(
            ok=False,
            code=532,
            message="stop selection requires --model, --port-profile, and --launch-profile together",
        )

    assert model_key is not None
    assert port_profile_id is not None
    assert launch_profile_key is not None
    normalized_model = model_key.strip()
    normalized_launch = launch_profile_key.strip()
    if not normalized_model:
        return None, _payload(ok=False, code=533, message="--model cannot be empty")
    if not normalized_launch:
        return None, _payload(ok=False, code=534, message="--launch-profile cannot be empty")

    valid, selection_payload = _validate_start_selection(
        model_key=normalized_model,
        port_profile_id=port_profile_id,
        launch_profile_key=normalized_launch,
        lmcache=None,
        enforce_weight_limit=False,
    )
    if not valid:
        return None, selection_payload
    selection = selection_payload.get("data")
    if not isinstance(selection, dict):
        return None, _payload(ok=False, code=535, message="failed to resolve stop selection")
    return selection, None


def _materialize_env_for_selection(selection: dict[str, Any]) -> tuple[bool, str]:
    try:
        _write_compose_env(_compose_env_values(selection))
        _write_compose_gpu_override(selection)
    except Exception as exc:
        return False, str(exc)
    return True, str(COMPOSE_ENV_FILE)


def _compose_up(*, startup_log_path: Path | None = None) -> ExecResult:
    env_ok, env_msg = _ensure_compose_env_file()
    if not env_ok:
        return ExecResult(
            returncode=503,
            stdout="",
            stderr="",
            error=f"failed to materialize compose env file: {env_msg}",
        )
    return _run_compose_streaming(["up", "-d", "--pull", "always", "jaeger", "vllm"], output_path=startup_log_path)


def _compose_down(
    *,
    stream_output: bool = False,
    output_path: Path | None = None,
    project_name: str | None = None,
    compose_env_file: Path = COMPOSE_ENV_FILE,
    compose_gpu_override_file: Path = COMPOSE_GPU_OVERRIDE_FILE,
    ensure_env_file: bool = True,
) -> ExecResult:
    if ensure_env_file:
        env_ok, env_msg = _ensure_compose_env_file()
        if not env_ok:
            return ExecResult(
                returncode=503,
                stdout="",
                stderr="",
                error=f"failed to materialize compose env file: {env_msg}",
            )
    if stream_output:
        return _run_compose_streaming(
            ["down", "--remove-orphans"],
            output_path=output_path,
            project_name=project_name,
            compose_env_file=compose_env_file,
            compose_gpu_override_file=compose_gpu_override_file,
        )
    return _run_compose(
        ["down", "--remove-orphans"],
        project_name=project_name,
        compose_env_file=compose_env_file,
        compose_gpu_override_file=compose_gpu_override_file,
    )


def _stop_environment_blocking_impl(
    reason: str,
    *,
    show_progress: bool = False,
    log_path: Path | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selection_identity = _selection_identity_payload(selection) if isinstance(selection, dict) else None
    if selection is not None and selection_identity is None:
        return _payload(ok=False, code=523, message="invalid stop selection")

    compose_context: dict[str, Any] | None = None
    compose_project_name: str | None = None
    compose_env_file = COMPOSE_ENV_FILE
    compose_gpu_override_file = COMPOSE_GPU_OVERRIDE_FILE
    ensure_env_file = True
    if selection is not None:
        context_ok, context_payload, context_error = _compose_context_for_selection(selection)
        if not context_ok:
            return _payload(
                ok=False,
                code=524,
                message=f"failed to prepare stop selection compose context: {context_error}",
                data={"selection": selection_identity},
            )
        compose_context = context_payload
        compose_project_name = str(context_payload["project_name"])
        compose_env_file = Path(context_payload["env_file"])
        compose_gpu_override_file = Path(context_payload["gpu_override_file"])
        ensure_env_file = False

    try:
        if show_progress:
            if selection_identity is None:
                _emit_progress(f"stop begin reason={reason}", log_path=log_path)
            else:
                _emit_progress(
                    "stop begin"
                    f" reason={reason}"
                    f" model={selection_identity['model_key']}"
                    f" port_profile={selection_identity['port_profile_id']}"
                    f" launch={selection_identity['launch_profile_key']}",
                    log_path=log_path,
                )
        state = _load_state()
        state_matches_selection = selection is None or _state_matches_selection(state, selection)
        port_profile_id = _current_port_profile_id(state) if selection is None else _coerce_int(selection.get("port_profile_id"))
        if show_progress:
            _emit_progress("stopping local gateway daemon", log_path=log_path)
        if port_profile_id is None:
            gateway_stop = _payload(
                ok=True,
                code=0,
                message="gateway stop skipped: no active port profile id",
            )
        else:
            gateway_stop = _stop_gateway_for_port_profile(port_profile_id)
        if show_progress:
            _emit_progress(f"gateway stop result: {gateway_stop.get('message')}", log_path=log_path)
            _emit_progress("checking compose services before stop", log_path=log_path)
        compose_ok, running_before, compose_error = _compose_running_services(
            project_name=compose_project_name,
            compose_env_file=compose_env_file,
            compose_gpu_override_file=compose_gpu_override_file,
            ensure_env_file=ensure_env_file,
        )
        if compose_ok and not running_before and not state.get("active"):
            if state_matches_selection:
                state["active"] = False
                state["lifecycle_state"] = "inactive"
                state["last_stop_reason"] = reason
                _save_state(state)
            elif selection_identity is not None:
                state["last_targeted_stop"] = {
                    "selection": selection_identity,
                    "reason": reason,
                    "stopped_at": _utc_now_iso(),
                    "compose_project_name": compose_project_name,
                    "result": "already_stopped",
                }
                _save_state(state)
            if show_progress:
                _emit_progress("stop skipped because no managed services are running", log_path=log_path)
            return _payload(
                ok=bool(gateway_stop.get("ok", False)),
                code=0 if gateway_stop.get("ok", False) else 522,
                message="environment already stopped" if gateway_stop.get("ok", False) else "gateway stop failed",
                data={
                    "reason": reason,
                    "gateway_stop": gateway_stop,
                    "selection": selection_identity,
                    "target_compose_project_name": compose_project_name,
                },
            )

        if show_progress:
            running_text = ",".join(sorted(running_before)) if compose_ok and running_before else "none"
            _emit_progress(
                f"running docker compose down --remove-orphans services_before={running_text}",
                log_path=log_path,
            )
        down = _compose_down(
            stream_output=show_progress,
            output_path=log_path,
            project_name=compose_project_name,
            compose_env_file=compose_env_file,
            compose_gpu_override_file=compose_gpu_override_file,
            ensure_env_file=ensure_env_file,
        )
        if show_progress:
            _emit_progress(f"docker compose down finished with returncode={down.returncode}", log_path=log_path)
            _emit_progress("verifying compose services after stop", log_path=log_path)
        compose_ok_after, running_after, compose_after_error = _compose_running_services(
            project_name=compose_project_name,
            compose_env_file=compose_env_file,
            compose_gpu_override_file=compose_gpu_override_file,
            ensure_env_file=ensure_env_file,
        )
        state = _load_state()
        if state_matches_selection:
            state["last_stop_reason"] = reason
            state["last_stop_completed_at"] = _utc_now_iso()

        still_running = compose_ok_after and bool({"jaeger", "vllm"} & running_after)
        if state_matches_selection:
            if not still_running:
                state["active"] = False
                state["lifecycle_state"] = "inactive"
            else:
                state["active"] = True
                state["lifecycle_state"] = "stop_failed"
            _save_state(state)
        elif selection_identity is not None:
            state["last_targeted_stop"] = {
                "selection": selection_identity,
                "reason": reason,
                "stopped_at": _utc_now_iso(),
                "compose_project_name": compose_project_name,
                "result": "stopped" if (down.returncode == 0 and not still_running and gateway_stop.get("ok")) else "failed",
            }
            _save_state(state)

        data = {
            "reason": reason,
            "gateway_stop": gateway_stop,
            "compose_down": {
                "returncode": down.returncode,
                "stdout": down.stdout,
                "stderr": down.stderr,
                "error": down.error,
                "timed_out": down.timed_out,
            },
            "compose_before": {
                "ok": compose_ok,
                "running_services": sorted(running_before) if compose_ok else [],
                "error": compose_error if not compose_ok else None,
            },
            "compose_after": {
                "ok": compose_ok_after,
                "running_services": sorted(running_after) if compose_ok_after else [],
                "error": compose_after_error if not compose_ok_after else None,
            },
            "selection": selection_identity,
            "target_matches_active_state": state_matches_selection,
            "target_compose_project_name": compose_project_name,
        }

        if down.error:
            if show_progress:
                _emit_progress(f"stop failed running docker compose down: {down.error}", log_path=log_path)
            return _payload(ok=False, code=520, message=f"failed running docker compose down: {down.error}", data=data)
        if down.returncode != 0 or still_running:
            if show_progress:
                remaining_text = ",".join(sorted(running_after)) if compose_ok_after and running_after else "unknown"
                _emit_progress(f"stop verification failed remaining_services={remaining_text}", log_path=log_path)
            return _payload(
                ok=False,
                code=521,
                message="docker compose down failed or services still running",
                data=data,
            )
        if not gateway_stop.get("ok"):
            if show_progress:
                _emit_progress(f"gateway stop failed: {gateway_stop.get('message')}", log_path=log_path)
            return _payload(ok=False, code=522, message="gateway stop failed", data=data)
        if show_progress:
            _emit_progress("stop completed successfully", log_path=log_path)
        return _payload(ok=True, code=0, message="environment stopped", data=data)
    finally:
        if compose_context is not None:
            _cleanup_compose_context(compose_context)


def _start_stop_worker() -> dict[str, Any]:
    stop_state = _load_stop_state()
    if isinstance(stop_state, dict):
        raw_pid = stop_state.get("pid")
        if isinstance(raw_pid, int) and _pid_is_running(raw_pid) and stop_state.get("status") == "running":
            return _payload(
                ok=False,
                code=530,
                message="a stop operation is already running",
                data={"stop_state": stop_state},
            )

    cmd = [sys.executable, str(Path(__file__).resolve()), "__stop-worker__"]
    with STOP_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now_iso()}] starting stop worker: {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    stop_payload = {
        "status": "running",
        "pid": proc.pid,
        "started_at": _utc_now_iso(),
        "log_file": str(STOP_LOG_FILE),
    }
    _write_json(STOP_STATE_FILE, stop_payload)

    state = _load_state()
    state["lifecycle_state"] = "stopping"
    _save_state(state)
    return _payload(ok=True, code=0, message="stop requested", data=stop_payload)


def _start_impl(
    *,
    model_key: str,
    port_profile_id: int,
    launch_profile_key: str,
    lmcache: int | None,
    gpu_memory_utilization: float | None,
    block: bool,
    timeout_seconds: float,
    startup_log_path: Path,
    compose_log_path: Path,
) -> dict[str, Any]:
    daemon_ok, daemon_payload = _ensure_daemon_running()
    if not daemon_ok:
        return daemon_payload

    _refresh_state_from_compose()

    valid, selection_payload = _validate_start_selection(
        model_key=model_key,
        port_profile_id=port_profile_id,
        launch_profile_key=launch_profile_key,
        lmcache=lmcache,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if not valid:
        return selection_payload
    selection = selection_payload["data"]
    compose_context: dict[str, Any] | None = None
    compose_project_name: str | None = None
    compose_env_file = COMPOSE_ENV_FILE
    compose_gpu_override_file = COMPOSE_GPU_OVERRIDE_FILE
    ensure_compose_env_file = True
    if block:
        context_ok, context_payload, context_error = _compose_context_for_selection(selection)
        if not context_ok:
            return _payload(
                ok=False,
                code=525,
                message=f"failed to prepare compose context for blocking wait: {context_error}",
            )
        compose_context = context_payload
        compose_project_name = str(context_payload["project_name"])
        compose_env_file = Path(context_payload["env_file"])
        compose_gpu_override_file = Path(context_payload["gpu_override_file"])
        ensure_compose_env_file = False

    try:
        _append_text(
            startup_log_path,
            (
                f"[{_utc_now_iso()}] startup log created\n"
                f"[{_utc_now_iso()}] startup_log_file={startup_log_path}\n"
                f"[{_utc_now_iso()}] compose_log_file={compose_log_path}\n"
            ),
        )
        _emit_progress(
            "resolved selection"
            f" model={selection['model_key']}"
            f" launch={selection['launch_profile_key']}"
            f" port_profile={selection['port_profile_id']}"
            f" lmcache={selection['lmcache'] if selection.get('lmcache') is not None else 'disabled'}"
            f" gpu_memory_utilization={selection['gpu_memory_utilization'] if selection.get('gpu_memory_utilization') is not None else 'default'}"
            f" vllm_port={selection['ports']['vllm_port']}"
            f" image={selection['images']['vllm_image_name']}",
            log_path=startup_log_path,
        )

        materialized_ok, materialized_msg = _materialize_env_for_selection(selection)
        if not materialized_ok:
            return _payload(
                ok=False,
                code=500,
                message=f"failed to materialize docker env: {materialized_msg}",
            )

        _emit_progress(f"startup log file={startup_log_path}", log_path=startup_log_path)
        _emit_progress(f"compose log file={compose_log_path}", log_path=startup_log_path)
        _emit_progress("running docker compose up -d --pull always jaeger vllm", log_path=startup_log_path)
        up = _compose_up(startup_log_path=startup_log_path)
        _emit_progress(f"docker compose up finished with returncode={up.returncode}", log_path=startup_log_path)
        captured_compose_logs = _sync_compose_logs_to_file(
            log_path=compose_log_path,
            previous_logs="",
            project_name=compose_project_name,
            compose_env_file=compose_env_file,
            compose_gpu_override_file=compose_gpu_override_file,
            ensure_env_file=ensure_compose_env_file,
        )
        if up.error:
            return _payload(
                ok=False,
                code=501,
                message=f"docker compose up failed: {up.error}",
                data={"stdout": up.stdout, "stderr": up.stderr},
            )
        if up.returncode != 0:
            return _payload(
                ok=False,
                code=502,
                message="docker compose up failed",
                data={"returncode": up.returncode, "stdout": up.stdout, "stderr": up.stderr},
            )

        _emit_progress("starting local gateway daemon", log_path=startup_log_path)
        gateway_start = _start_gateway_for_selection(selection)
        if not gateway_start.get("ok"):
            _emit_progress(
                f"gateway startup failed: {gateway_start.get('message')}",
                log_path=startup_log_path,
            )
            compose_cleanup = _compose_down(
                stream_output=True,
                output_path=startup_log_path,
                project_name=compose_project_name,
                compose_env_file=compose_env_file,
                compose_gpu_override_file=compose_gpu_override_file,
                ensure_env_file=ensure_compose_env_file,
            )
            return _payload(
                ok=False,
                code=544,
                message="gateway startup failed",
                data={
                    "gateway_start": gateway_start.get("data"),
                    "gateway_error": gateway_start.get("message"),
                    "compose_cleanup": {
                        "returncode": compose_cleanup.returncode,
                        "stdout": compose_cleanup.stdout,
                        "stderr": compose_cleanup.stderr,
                        "error": compose_cleanup.error,
                    },
                },
            )
        _emit_progress("gateway daemon started", log_path=startup_log_path)

        new_state = {
            "active": True,
            "lifecycle_state": "starting" if block else "running",
            "started_at": _utc_now_iso(),
            "selection": {
                "model_key": selection["model_key"],
                "port_profile_id": selection["port_profile_id"],
                "launch_profile_key": selection["launch_profile_key"],
                "lmcache": selection.get("lmcache"),
                "gpu_memory_utilization": selection.get("gpu_memory_utilization"),
            },
            "resolved": {
                "images": selection["images"],
                "model": selection["model"],
                "ports": selection["ports"],
                "launch": selection["launch"],
                "lmcache_enabled": selection.get("lmcache") is not None,
                "lmcache_max_local_cpu_size": selection.get("lmcache"),
                "gpu_memory_utilization": selection.get("gpu_memory_utilization"),
                "service_urls": selection["service_urls"],
                "runtime_names": selection["runtime_names"],
            },
            "compose": {
                "file": str(COMPOSE_FILE),
                "env_file": str(COMPOSE_ENV_FILE),
            },
            "gateway": gateway_start.get("data"),
        }
        _save_state(new_state)

        response_data = {
            "daemon": daemon_payload.get("data", {}).get("daemon"),
            "selection": new_state["selection"],
            "resolved": new_state["resolved"],
            "compose_up": {"returncode": up.returncode, "stdout": up.stdout, "stderr": up.stderr},
            "gateway_start": gateway_start.get("data"),
            "env_file": materialized_msg,
            "startup_log_file": str(startup_log_path),
            "compose_log_file": str(compose_log_path),
        }

        if not block:
            return _payload(ok=True, code=0, message="environment start requested", data=response_data)

        _emit_progress(
            "waiting for health checks"
            f" timeout={timeout_seconds}s"
            f" vllm={selection['service_urls']['vllm']}/v1/models"
            f" jaeger={selection['service_urls']['jaeger_api']}"
            f" gateway={selection['service_urls'].get('gateway', 'n/a')}/healthz",
            log_path=startup_log_path,
        )
        wait_payload = _wait_up_impl(
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=DEFAULT_WAIT_UP_POLL_INTERVAL_SECONDS,
            show_progress=True,
            startup_log_path=startup_log_path,
            compose_log_path=compose_log_path,
            previous_compose_logs=captured_compose_logs,
            selection=selection,
            compose_project_name=compose_project_name,
            compose_env_file=compose_env_file,
            compose_gpu_override_file=compose_gpu_override_file,
            ensure_compose_env_file=ensure_compose_env_file,
        )
        state_after_wait = _load_state()
        if wait_payload.get("ok"):
            state_after_wait["lifecycle_state"] = "running"
            _save_state(state_after_wait)
            response_data["wait_up"] = wait_payload.get("data", {})
            return _payload(ok=True, code=0, message="environment started and healthy", data=response_data)

        wait_code = wait_payload.get("code")
        state_after_wait["lifecycle_state"] = "degraded" if wait_code == 408 else "failed"
        _save_state(state_after_wait)
        response_data["wait_up"] = wait_payload.get("data", {})
        return _payload(
            ok=False,
            code=int(wait_code) if isinstance(wait_code, int) else 504,
            message=str(wait_payload.get("message") or "environment failed to become healthy"),
            data=response_data,
        )
    finally:
        if compose_context is not None:
            _cleanup_compose_context(compose_context)


def _status_impl() -> dict[str, Any]:
    state = _refresh_state_from_compose()
    daemon = _daemon_running_record()
    port_profile_id = _current_port_profile_id(state, allow_env_fallback=False)
    gateway_status = _gateway_status_for_port_profile(port_profile_id) if port_profile_id is not None else None
    compose_ok, services, compose_error = _compose_running_services()
    data = {
        "daemon_running": daemon is not None,
        "daemon": daemon,
        "active_environment": state if state.get("active") else None,
        "lifecycle_state": state.get("lifecycle_state"),
        "gateway": gateway_status,
        "compose": {
            "ok": compose_ok,
            "running_services": sorted(services) if compose_ok else [],
            "error": compose_error if not compose_ok else None,
        },
    }
    return _payload(ok=True, code=0, message="status", data=data)


def _logs_impl(lines: int) -> dict[str, Any]:
    if lines <= 0:
        return _payload(ok=False, code=600, message="lines must be > 0")
    env_ok, env_msg = _ensure_compose_env_file()
    if not env_ok:
        return _payload(ok=False, code=601, message=f"failed to materialize compose env file: {env_msg}")
    result = _run_compose(["logs", "--tail", str(lines), "vllm", "jaeger"])
    if result.error:
        return _payload(ok=False, code=602, message=f"failed to run docker compose logs: {result.error}")
    if result.returncode != 0:
        return _payload(
            ok=False,
            code=603,
            message="docker compose logs failed",
            data={"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
        )
    state = _load_state()
    port_profile_id = _current_port_profile_id(state, allow_env_fallback=False)
    gateway_logs = None
    if port_profile_id is not None:
        gateway_status = _gateway_status_for_port_profile(port_profile_id)
        log_file_path = gateway_status.get("log_file")
        if isinstance(log_file_path, str):
            gateway_logs = {
                "status": gateway_status,
                "lines": lines,
                "logs": _tail_text(Path(log_file_path), lines=lines),
            }
    return _payload(
        ok=True,
        code=0,
        message="logs",
        data={"lines": lines, "logs": result.stdout, "gateway": gateway_logs},
    )


def _daemon_loop_main() -> int:
    _ensure_runtime_dir()
    keep_running = True

    def _handle_stop(_signum: int, _frame: Any) -> None:
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    while keep_running:
        heartbeat = {
            "pid": os.getpid(),
            "updated_at": _utc_now_iso(),
        }
        _write_json(DAEMON_HEARTBEAT_FILE, heartbeat)
        time.sleep(1.0)
    return 0


def _stop_worker_main() -> int:
    _ensure_runtime_dir()
    result = _stop_environment_blocking_impl("async_stop")
    current = _load_stop_state() or {}
    current.update(
        {
            "status": "succeeded" if result.get("ok") else "failed",
            "finished_at": _utc_now_iso(),
            "result": result,
        }
    )
    _write_json(STOP_STATE_FILE, current)
    return 0 if result.get("ok") else 1


@app.command(name="daemon-start")
def daemon_start() -> None:
    """Ensure daemon exists (idempotent)."""
    _emit(_start_daemon_impl())


@app.command(name="daemon-status")
def daemon_status() -> None:
    """Show daemon liveness and active environment metadata."""
    _emit(_daemon_status_payload())


@app.command(name="daemon-stop")
def daemon_stop() -> None:
    """Stop daemon; stop active environment first."""
    env_stop_payload = _stop_environment_blocking_impl("daemon_stop")

    record = _load_daemon_record()
    if not isinstance(record, dict):
        _emit(
            _payload(
                ok=True,
                code=0,
                message="daemon is not running",
                data={"environment_stop": env_stop_payload},
            )
        )
        return

    raw_pid = record.get("pid")
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        DAEMON_PID_FILE.unlink(missing_ok=True)
        _emit(
            _payload(
                ok=True,
                code=0,
                message="removed invalid daemon pid record",
                data={"environment_stop": env_stop_payload},
            )
        )
        return

    if not _pid_is_running(pid):
        DAEMON_PID_FILE.unlink(missing_ok=True)
        _emit(
            _payload(
                ok=True,
                code=0,
                message="daemon is not running",
                data={"environment_stop": env_stop_payload},
            )
        )
        return

    stopped = _stop_pid(pid, timeout_seconds=10.0)
    if stopped:
        DAEMON_PID_FILE.unlink(missing_ok=True)
        _emit(
            _payload(
                ok=True,
                code=0,
                message="daemon stopped",
                data={"environment_stop": env_stop_payload},
            )
        )
        return
    _emit(
        _payload(
            ok=False,
            code=302,
            message=f"failed to stop daemon pid={pid}",
            data={"environment_stop": env_stop_payload},
        )
    )


@profiles_app.command(name="models")
def profiles_models() -> None:
    """List allowed model keys."""
    try:
        default_model, models = _load_models_config()
    except Exception as exc:
        _emit(_payload(ok=False, code=800, message=f"failed to load model profiles: {exc}"))
        return
    _emit(
        _payload(
            ok=True,
            code=0,
            message="model profiles",
            data={"default_model": default_model, "models": models},
        )
    )


@profiles_app.command(name="ports")
def profiles_ports() -> None:
    """List allowed port profile IDs."""
    try:
        default_profile, profiles = _load_port_profiles()
    except Exception as exc:
        _emit(_payload(ok=False, code=801, message=f"failed to load port profiles: {exc}"))
        return
    _emit(
        _payload(
            ok=True,
            code=0,
            message="port profiles",
            data={"default_profile": default_profile, "profiles": profiles},
        )
    )


@profiles_app.command(name="launches")
def profiles_launches() -> None:
    """List allowed launch profile keys."""
    try:
        default_profile, profiles = _load_launch_profiles()
    except Exception as exc:
        _emit(_payload(ok=False, code=802, message=f"failed to load launch profiles: {exc}"))
        return
    _emit(
        _payload(
            ok=True,
            code=0,
            message="launch profiles",
            data={"default_profile": default_profile, "profiles": profiles},
        )
    )


@app.command(name="start")
def start(
    model: str = typer.Option(..., "--model", "-m", help="Model key from configs/model_config.toml."),
    port_profile: int = typer.Option(..., "--port-profile", "-p", help="Port profile numeric ID."),
    launch_profile: str = typer.Option(..., "--launch-profile", "-l", help="Launch profile key."),
    lmcache: int | None = typer.Option(
        None,
        "--lmcache",
        help=(
            "Enable LMCache with a maximum local CPU size. "
            "Sets LMCACHE_MAX_LOCAL_CPU_SIZE and enables kv-transfer-config."
        ),
    ),
    gpu_memory_utilization: float | None = typer.Option(
        None,
        "--gpu-memory-utilization",
        help="Pass through to vLLM as --gpu-memory-utilization <value> (0 < value <= 1).",
    ),
    block: bool = typer.Option(False, "--block", "-b", help="Block until endpoints are healthy."),
    timeout_seconds: float = typer.Option(
        DEFAULT_WAIT_UP_TIMEOUT_SECONDS,
        "--timeout-seconds",
        help="Timeout for --block health wait.",
    ),
) -> None:
    """Start one environment (auto-start daemon if needed)."""
    if lmcache is not None and lmcache <= 0:
        raise typer.BadParameter("--lmcache must be a positive integer", param_hint="--lmcache")
    try:
        _normalize_gpu_memory_utilization(gpu_memory_utilization)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--gpu-memory-utilization") from exc
    startup_log_path = _new_startup_log_path(
        model_key=model,
        port_profile_id=port_profile,
        launch_profile_key=launch_profile,
    )
    compose_log_path = _new_compose_log_path(startup_log_path)
    try:
        _emit(
            _start_impl(
                model_key=model,
                port_profile_id=port_profile,
                launch_profile_key=launch_profile,
                lmcache=lmcache,
                gpu_memory_utilization=gpu_memory_utilization,
                block=block,
                timeout_seconds=timeout_seconds,
                startup_log_path=startup_log_path,
                compose_log_path=compose_log_path,
            )
        )
    except KeyboardInterrupt:
        if block:
            _emit_progress("start interrupted by user; cleaning up managed environment", log_path=startup_log_path)
            cleanup_payload = _stop_environment_blocking_impl(
                "start_interrupted",
                show_progress=True,
                log_path=startup_log_path,
            )
            _emit(
                _payload(
                    ok=False,
                    code=130,
                    message="start interrupted by user",
                    data={
                        "cleanup": cleanup_payload,
                        "startup_log_file": str(startup_log_path),
                        "compose_log_file": str(compose_log_path),
                    },
                ),
                fail_on_error=False,
            )
            raise typer.Exit(code=130)
        raise typer.Exit(code=130)


@app.command(name="status")
def status() -> None:
    """Show active environment status and resolved config."""
    _emit(_status_impl())


@app.command(name="up")
def up() -> None:
    """Immediate health check for expected endpoints."""
    _emit(_up_impl())


@app.command(name="wait-up")
def wait_up(
    timeout_seconds: float = typer.Option(
        DEFAULT_WAIT_UP_TIMEOUT_SECONDS,
        "--timeout-seconds",
        help="Maximum seconds to wait.",
    ),
    poll_interval_seconds: float = typer.Option(
        DEFAULT_WAIT_UP_POLL_INTERVAL_SECONDS,
        "--poll-interval-seconds",
        help="Poll interval seconds.",
    ),
) -> None:
    """Poll health checks until ready or timeout."""
    _emit(_wait_up_impl(timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds, show_progress=True))


@app.command(name="logs")
def logs(
    lines: int = typer.Option(200, "--lines", "-n", help="Number of lines to tail."),
) -> None:
    """Tail recent vLLM + Jaeger compose logs."""
    _emit(_logs_impl(lines))


@app.command(name="stop")
def stop(
    block: bool = typer.Option(False, "--block", "-b", help="Block until teardown completes."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model key used at start time."),
    port_profile: int | None = typer.Option(None, "--port-profile", "-p", help="Port profile ID used at start time."),
    launch_profile: str | None = typer.Option(None, "--launch-profile", "-l", help="Launch profile key used at start time."),
) -> None:
    """Stop environment; async by default."""
    selection, selection_error = _resolve_stop_selection(
        model_key=model,
        port_profile_id=port_profile,
        launch_profile_key=launch_profile,
    )
    if selection_error is not None:
        _emit(selection_error)
        return
    if selection is not None and not block:
        _emit(
            _payload(
                ok=False,
                code=536,
                message="targeted stop requires --block",
            )
        )
        return
    if block:
        reason = "stop_blocking_selected" if selection is not None else "stop_blocking"
        _emit(_stop_environment_blocking_impl(reason, show_progress=True, selection=selection))
        return
    _emit(_start_stop_worker())


@app.command(name="stop-poll")
def stop_poll() -> None:
    """Poll status of a previously requested async stop."""
    state = _load_stop_state()
    if not isinstance(state, dict):
        _emit(_payload(ok=True, code=0, message="no async stop in progress"))
        return

    status_value = state.get("status")
    raw_pid = state.get("pid")
    if status_value == "running" and isinstance(raw_pid, int) and _pid_is_running(raw_pid):
        _emit(_payload(ok=True, code=0, message="stop still running", data=state))
        return

    refreshed = _load_stop_state() or state
    if refreshed.get("status") == "running":
        refreshed["status"] = "failed"
        refreshed["finished_at"] = _utc_now_iso()
        refreshed["result"] = _payload(
            ok=False,
            code=531,
            message="stop worker exited without writing terminal status",
            data={"log_tail": _tail_text(STOP_LOG_FILE)},
        )
        _write_json(STOP_STATE_FILE, refreshed)
    _emit(_payload(ok=True, code=0, message="stop status", data=refreshed))


def _main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "__daemon-loop__":
        return _daemon_loop_main()
    if len(sys.argv) >= 2 and sys.argv[1] == "__stop-worker__":
        return _stop_worker_main()
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
