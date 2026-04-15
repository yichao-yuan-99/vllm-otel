#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Typer CLI for Docker multi-backend control."""

from __future__ import annotations

from concurrent import futures
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Sequence

import typer


REPO_ROOT = Path(__file__).resolve().parents[2]
MULTI_DIR = REPO_ROOT / "servers" / "servers-docker-multi"
SINGLE_CLIENT_PATH = REPO_ROOT / "servers" / "servers-docker" / "client.py"
STARTUP_LOGS_DIR = MULTI_DIR / "logs"

RUNTIME_DIR = Path.home() / ".cache" / "vllm-otel-docker-multi"
STATE_FILE = RUNTIME_DIR / "state.json"
GATEWAY_MULTI_PID_FILE = RUNTIME_DIR / "gateway_multi.pid.json"
GATEWAY_MULTI_LOG_FILE = RUNTIME_DIR / "gateway_multi.log"

DEFAULT_WAIT_UP_TIMEOUT_SECONDS = 900
DEFAULT_WAIT_UP_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_GATEWAY_MULTI_MODULE_NAME = "gateway_multi"
DEFAULT_GATEWAY_MULTI_CONFIG_PATH = REPO_ROOT / "gateway_multi" / "config.toml"
DEFAULT_GATEWAY_MULTI_CONFIG_EXAMPLE_PATH = REPO_ROOT / "gateway_multi" / "config.example.toml"
DEFAULT_GATEWAY_MULTI_HOST = "0.0.0.0"
DEFAULT_GATEWAY_MULTI_VENV_DIR = REPO_ROOT / ".venv"
DEFAULT_ASSIGNMENT_POLICY = "round_robin"


def _load_single_client() -> Any:
    spec = importlib.util.spec_from_file_location(
        "servers_docker_single_client",
        SINGLE_CLIENT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base client from {SINGLE_CLIENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_single = _load_single_client()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Docker multi-backend control client.",
)
profiles_app = typer.Typer(add_completion=False, no_args_is_help=True, help="List configured profiles.")
app.add_typer(profiles_app, name="profiles")


def _payload(*, ok: bool, code: int, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return _single._payload(ok=ok, code=code, message=message, data=data)


def _emit(payload: dict[str, Any], *, fail_on_error: bool = True) -> None:
    _single._emit(payload, fail_on_error=fail_on_error)


def _utc_now_iso() -> str:
    return _single._utc_now_iso()


def _write_json(path: Path, data: Any) -> None:
    _single._write_json(path, data)


def _read_json(path: Path, default: Any) -> Any:
    return _single._read_json(path, default)


def _append_text(path: Path, text: str) -> None:
    _single._append_text(path, text)


def _tail_text(path: Path, *, lines: int = 60) -> str:
    return _single._tail_text(path, lines=lines)


def _ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_path_token(value: str) -> str:
    return _single._sanitize_path_token(value)


def _port_profile_ids_suffix(port_profile_ids: Sequence[int | str]) -> str:
    joined = "-".join(str(profile_id) for profile_id in port_profile_ids)
    suffix = _sanitize_path_token(joined)
    return suffix or "default"


def _state_file_for_port_profile_ids(
    port_profile_ids: Sequence[int | str] | None = None,
) -> Path:
    if not port_profile_ids:
        return STATE_FILE
    return RUNTIME_DIR / f"state.{_port_profile_ids_suffix(port_profile_ids)}.json"


def _state_file_for_selection(selection: dict[str, Any] | None) -> Path:
    if not isinstance(selection, dict):
        return STATE_FILE
    port_profile_ids = selection.get("port_profile_ids")
    if (
        isinstance(port_profile_ids, list)
        and port_profile_ids
        and all(not isinstance(value, bool) and isinstance(value, (int, str)) for value in port_profile_ids)
    ):
        return _state_file_for_port_profile_ids(port_profile_ids)
    return STATE_FILE


def _ensure_list_of_strings(values: list[Any], *, key: str) -> list[str]:
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{key} must contain only strings")
    return [str(value) for value in values]


def _parse_port_profile_ids(raw_values: Sequence[str]) -> list[int]:
    parsed: list[int] = []
    seen: set[int] = set()
    for raw in raw_values:
        for part in raw.split(","):
            stripped = part.strip()
            if not stripped:
                continue
            try:
                profile_id = int(stripped)
            except ValueError as exc:
                raise ValueError(f"invalid port profile id: {stripped}") from exc
            if profile_id < 0:
                raise ValueError(f"invalid port profile id: {stripped}")
            if profile_id in seen:
                raise ValueError(f"duplicate port profile id: {profile_id}")
            parsed.append(profile_id)
            seen.add(profile_id)
    if not parsed:
        raise ValueError("at least one port profile id is required")
    return parsed


def _backend_runtime_names_for_selection(
    *,
    model_key: str,
    launch_profile_key: str,
    port_profile_id: int,
) -> dict[str, str]:
    model_token = _single._sanitize_runtime_token(model_key)
    launch_token = _single._sanitize_runtime_token(launch_profile_key)
    suffix = f"{model_token}-{launch_token}-p{port_profile_id}"
    return {
        "compose_project_name": _single._bounded_runtime_name(
            _single.DEFAULT_COMPOSE_PROJECT_NAME,
            suffix,
        ),
        "jaeger_container_name": _single._bounded_runtime_name(
            _single.DEFAULT_JAEGER_CONTAINER_NAME,
            suffix,
            max_length=96,
        ),
        "vllm_container_name": _single._bounded_runtime_name(
            _single.DEFAULT_VLLM_CONTAINER_NAME,
            suffix,
            max_length=96,
        ),
        "otel_service_name": _single._bounded_runtime_name(
            _single.DEFAULT_OTEL_SERVICE_NAME,
            suffix,
            max_length=96,
        ),
    }


def _split_launch_profile_across_backends(
    *,
    launch_profile: dict[str, Any],
    backend_count: int,
) -> list[dict[str, Any]]:
    device_ids_raw = launch_profile.get("visible_device_ids")
    if not isinstance(device_ids_raw, list) or not device_ids_raw:
        raise ValueError("launch.visible_device_ids must be a non-empty list")
    device_ids = _ensure_list_of_strings(device_ids_raw, key="launch.visible_device_ids")
    if len(device_ids) != backend_count:
        raise ValueError(
            "launch profile GPU count must match the number of selected port profiles "
            f"(gpus={len(device_ids)}, profiles={backend_count})"
        )

    per_gpu_memory_gb = float(launch_profile["per_gpu_memory_gb"])
    out: list[dict[str, Any]] = []
    for device_id in device_ids:
        out.append(
            {
                "label": launch_profile.get("label"),
                "gpu_type": launch_profile["gpu_type"],
                "visible_devices": device_id,
                "visible_device_ids": [device_id],
                "per_gpu_memory_gb": per_gpu_memory_gb,
                "total_gpu_memory_gb": per_gpu_memory_gb,
                "tensor_parallel_size": 1,
            }
        )
    return out


def _resolve_gateway_multi_config_path() -> Path:
    if DEFAULT_GATEWAY_MULTI_CONFIG_PATH.exists():
        return DEFAULT_GATEWAY_MULTI_CONFIG_PATH
    return DEFAULT_GATEWAY_MULTI_CONFIG_EXAMPLE_PATH


def _gateway_multi_runtime_files(
    port_profile_ids: Sequence[int | str] | None = None,
) -> tuple[Path, Path]:
    if not port_profile_ids:
        return GATEWAY_MULTI_PID_FILE, GATEWAY_MULTI_LOG_FILE
    suffix = _port_profile_ids_suffix(port_profile_ids)
    return (
        RUNTIME_DIR / f"gateway_multi.{suffix}.pid.json",
        RUNTIME_DIR / f"gateway_multi.{suffix}.log",
    )


def _empty_state() -> dict[str, Any]:
    return {
        "active": False,
        "lifecycle_state": "inactive",
        "updated_at": _utc_now_iso(),
    }


def _selection_state_files() -> list[Path]:
    if not RUNTIME_DIR.exists():
        return []
    return sorted(path for path in RUNTIME_DIR.glob("state.*.json") if path.is_file())


def _legacy_selection_state_from_alias() -> dict[str, Any] | None:
    raw = _read_json(STATE_FILE, None)
    if not isinstance(raw, dict):
        return None
    selection = raw.get("selection")
    if not isinstance(selection, dict):
        return None
    if _selection_identity(selection) is None:
        return None
    return raw


def _migrate_legacy_selection_state_alias() -> dict[str, Any] | None:
    legacy_state = _legacy_selection_state_from_alias()
    if legacy_state is None:
        return None
    target_path = _state_file_for_selection(legacy_state.get("selection"))
    if target_path != STATE_FILE and not target_path.exists():
        _write_json(target_path, legacy_state)
    return legacy_state


def _latest_active_saved_state(
    *,
    exclude_selection: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    _migrate_legacy_selection_state_alias()
    latest: dict[str, Any] | None = None
    latest_updated_at = ""
    for state_file in _selection_state_files():
        raw = _read_json(state_file, None)
        if not isinstance(raw, dict) or not raw.get("active"):
            continue
        if _selection_from_state(raw) is None:
            continue
        if exclude_selection is not None and _state_matches_selection(raw, exclude_selection):
            continue
        updated_at = str(raw.get("updated_at") or "")
        if latest is None or updated_at >= latest_updated_at:
            latest = raw
            latest_updated_at = updated_at
    return latest


def _write_current_state_alias(state: dict[str, Any] | None) -> None:
    alias_state = dict(state) if isinstance(state, dict) else _empty_state()
    alias_state["updated_at"] = _utc_now_iso()
    _write_json(STATE_FILE, alias_state)


def _load_gateway_multi_record(
    port_profile_ids: Sequence[int] | None = None,
) -> tuple[dict[str, Any] | None, Path, Path]:
    pid_file, log_file = _gateway_multi_runtime_files(port_profile_ids)
    record_raw = _read_json(pid_file, None)
    if isinstance(record_raw, dict):
        return record_raw, pid_file, log_file
    if port_profile_ids:
        legacy_record_raw = _read_json(GATEWAY_MULTI_PID_FILE, None)
        if isinstance(legacy_record_raw, dict) and legacy_record_raw.get("port_profile_ids") == list(port_profile_ids):
            return legacy_record_raw, GATEWAY_MULTI_PID_FILE, GATEWAY_MULTI_LOG_FILE
    return None, pid_file, log_file


def _gateway_multi_record_matches_running_process(record: dict[str, Any]) -> bool:
    pid = _single._coerce_int(record.get("pid"))
    if pid is None or not _single._pid_is_running(pid):
        return False
    cmdline = _single._process_cmdline(pid)
    if not cmdline:
        return True
    normalized = cmdline.lower()
    if DEFAULT_GATEWAY_MULTI_MODULE_NAME not in normalized or " start " not in f" {normalized} ":
        return False
    port_profile_ids = record.get("port_profile_ids")
    if isinstance(port_profile_ids, list):
        for profile_id in port_profile_ids:
            token = f"--port-profile-id {profile_id}"
            token_eq = f"--port-profile-id={profile_id}"
            if token not in cmdline and token_eq not in cmdline:
                return False
    return True


def _gateway_multi_status(selection: dict[str, Any] | None = None) -> dict[str, Any]:
    port_profile_ids = None
    if isinstance(selection, dict):
        raw_port_profile_ids = selection.get("port_profile_ids")
        if isinstance(raw_port_profile_ids, list) and all(isinstance(value, int) for value in raw_port_profile_ids):
            port_profile_ids = raw_port_profile_ids
    record, pid_file, log_file = _load_gateway_multi_record(port_profile_ids)
    running = isinstance(record, dict) and _gateway_multi_record_matches_running_process(record)
    pid = _single._coerce_int(record.get("pid")) if isinstance(record, dict) else None
    return {
        "running": running,
        "pid": pid,
        "pid_file": str(pid_file),
        "log_file": str(log_file),
        "record": record,
    }


def _resolve_gateway_multi_ipc_socket_paths(port_profile_ids: list[int]) -> list[Path | None]:
    try:
        from gateway_multi.cli import _resolve_ipc_socket_path as _gateway_multi_resolve_ipc_socket_path
        from gateway_multi.runtime_config import load_runtime_settings as _load_gateway_multi_runtime_settings

        settings = _load_gateway_multi_runtime_settings(_resolve_gateway_multi_config_path())
        return [
            _gateway_multi_resolve_ipc_socket_path(
                ipc_enabled=settings.ipc_enabled,
                configured_socket_path_template=settings.ipc_socket_path_template,
                profile_id=profile_id,
            )
            for profile_id in port_profile_ids
        ]
    except Exception:
        return [
            Path(f"/tmp/vllm-gateway-profile-{profile_id}.sock")
            for profile_id in port_profile_ids
        ]


def _cleanup_stale_gateway_multi_ipc_sockets(port_profile_ids: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for socket_path in _resolve_gateway_multi_ipc_socket_paths(port_profile_ids):
        result = {
            "socket_path": str(socket_path) if socket_path is not None else None,
            "enabled": socket_path is not None,
            "exists": False,
            "active_listener": False,
            "removed": False,
            "error": None,
        }
        if socket_path is None:
            results.append(result)
            continue
        try:
            mode = socket_path.stat().st_mode
        except FileNotFoundError:
            results.append(result)
            continue
        except OSError as exc:
            result["error"] = str(exc)
            results.append(result)
            continue
        result["exists"] = True
        if not os.path.exists(socket_path) or not stat.S_ISSOCK(mode):
            result["error"] = f"path exists but is not a socket: {socket_path}"
            results.append(result)
            continue
        try:
            active_listener = _single._unix_socket_accepting_connections(socket_path)
        except Exception as exc:
            result["error"] = str(exc)
            results.append(result)
            continue
        result["active_listener"] = active_listener
        if not active_listener:
            socket_path.unlink(missing_ok=True)
            result["removed"] = True
        results.append(result)
    return results


def _start_gateway_multi(selection: dict[str, Any]) -> dict[str, Any]:
    port_profile_ids = selection["port_profile_ids"]
    config_path = _resolve_gateway_multi_config_path()
    if not config_path.exists():
        return _payload(
            ok=False,
            code=742,
            message=(
                "cannot start gateway_multi: config file not found "
                f"(checked {DEFAULT_GATEWAY_MULTI_CONFIG_PATH} and {DEFAULT_GATEWAY_MULTI_CONFIG_EXAMPLE_PATH})"
            ),
        )

    _ensure_runtime_dir()
    pid_file, log_file = _gateway_multi_runtime_files(port_profile_ids)
    active_record, active_pid_file, _ = _load_gateway_multi_record(port_profile_ids)
    if isinstance(active_record, dict):
        if _gateway_multi_record_matches_running_process(active_record):
            active_profiles = active_record.get("port_profile_ids")
            active_policy = active_record.get("assignment_policy")
            if active_profiles != list(port_profile_ids) or active_policy != selection["assignment_policy"]:
                return _payload(
                    ok=False,
                    code=745,
                    message=(
                        "gateway_multi is already running with a different selection "
                        f"(running_profiles={active_profiles}, requested_profiles={port_profile_ids})"
                    ),
                    data={"record": active_record},
                )
            return _payload(
                ok=True,
                code=0,
                message="gateway_multi already running",
                data={"pid_file": str(active_pid_file), "record": active_record},
            )
        active_pid_file.unlink(missing_ok=True)

    gateway_cmd = [
        sys.executable,
        "-m",
        DEFAULT_GATEWAY_MULTI_MODULE_NAME,
        "start",
        "--config",
        str(config_path),
        "--host",
        DEFAULT_GATEWAY_MULTI_HOST,
        "--venv-dir",
        str(DEFAULT_GATEWAY_MULTI_VENV_DIR),
        "--skip-install",
    ]
    for profile_id in port_profile_ids:
        gateway_cmd.extend(["--port-profile-id", str(profile_id)])
    gateway_cmd.extend(["--policy", str(selection["assignment_policy"])])

    env = dict(os.environ)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{py_path}" if py_path else str(REPO_ROOT)

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now_iso()}] starting gateway_multi: {' '.join(gateway_cmd)}\n")
        handle.flush()
        proc = _single.subprocess.Popen(
            gateway_cmd,
            cwd=str(REPO_ROOT),
            stdin=_single.subprocess.DEVNULL,
            stdout=handle,
            stderr=_single.subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=env,
        )

    deadline = time.monotonic() + _single.GATEWAY_STARTUP_PROBE_SECONDS
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            return _payload(
                ok=False,
                code=743,
                message=f"failed to start gateway_multi daemon (exit={rc})",
                data={
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
        "port_profile_ids": list(port_profile_ids),
        "assignment_policy": selection["assignment_policy"],
        "module": DEFAULT_GATEWAY_MULTI_MODULE_NAME,
        "config_path": str(config_path),
        "venv_dir": str(DEFAULT_GATEWAY_MULTI_VENV_DIR),
        "host": DEFAULT_GATEWAY_MULTI_HOST,
        "command": gateway_cmd,
        "pid_file": str(pid_file),
        "log_file": str(log_file),
    }
    _write_json(pid_file, record)
    return _payload(ok=True, code=0, message="gateway_multi started", data=record)


def _stop_gateway_multi(port_profile_ids: list[int]) -> dict[str, Any]:
    _ensure_runtime_dir()
    record_raw, pid_file, log_file = _load_gateway_multi_record(port_profile_ids)
    if not isinstance(record_raw, dict):
        pid_file.unlink(missing_ok=True)
        ipc_cleanup = _cleanup_stale_gateway_multi_ipc_sockets(port_profile_ids)
        message = "gateway_multi is not running"
        if any(item.get("removed") for item in ipc_cleanup):
            message += " (removed stale IPC sockets)"
        return _payload(
            ok=True,
            code=0,
            message=message,
            data={
                "pid_file": str(pid_file),
                "log_file": str(log_file),
                "ipc_socket_cleanup": ipc_cleanup,
            },
        )

    pid = _single._coerce_int(record_raw.get("pid"))
    if pid is None or not _gateway_multi_record_matches_running_process(record_raw):
        pid_file.unlink(missing_ok=True)
        ipc_cleanup = _cleanup_stale_gateway_multi_ipc_sockets(port_profile_ids)
        message = "gateway_multi is not running (stale pid file removed)"
        if any(item.get("removed") for item in ipc_cleanup):
            message += " and stale IPC sockets removed"
        return _payload(
            ok=True,
            code=0,
            message=message,
            data={
                "pid_file": str(pid_file),
                "record": record_raw,
                "ipc_socket_cleanup": ipc_cleanup,
            },
        )

    stopped = _single._stop_pid(pid, timeout_seconds=10.0)
    pid_file.unlink(missing_ok=True)
    _append_text(log_file, f"[{_utc_now_iso()}] stopped gateway_multi pid={pid} forced={not stopped}\n")
    ipc_cleanup = _cleanup_stale_gateway_multi_ipc_sockets(port_profile_ids)
    if stopped:
        return _payload(
            ok=True,
            code=0,
            message="gateway_multi stopped",
            data={
                "pid": pid,
                "pid_file": str(pid_file),
                "log_file": str(log_file),
                "ipc_socket_cleanup": ipc_cleanup,
            },
        )
    return _payload(
        ok=False,
        code=744,
        message=f"failed to stop gateway_multi pid={pid}",
        data={
            "pid": pid,
            "pid_file": str(pid_file),
            "log_file": str(log_file),
            "ipc_socket_cleanup": ipc_cleanup,
        },
    )


def _selection_identity(selection: dict[str, Any]) -> tuple[str, tuple[int, ...], str] | None:
    model_key = selection.get("model_key")
    launch_profile_key = selection.get("launch_profile_key")
    port_profile_ids = selection.get("port_profile_ids")
    if not isinstance(model_key, str) or not model_key.strip():
        return None
    if not isinstance(launch_profile_key, str) or not launch_profile_key.strip():
        return None
    if not isinstance(port_profile_ids, list) or not all(isinstance(value, int) for value in port_profile_ids):
        return None
    return model_key.strip(), tuple(port_profile_ids), launch_profile_key.strip()


def _state_matches_selection(state: dict[str, Any], selection: dict[str, Any]) -> bool:
    state_selection = state.get("selection")
    if not isinstance(state_selection, dict):
        return False
    return _selection_identity(state_selection) == _selection_identity(selection)


def _load_state(selection: dict[str, Any] | None = None) -> dict[str, Any]:
    legacy_state = _migrate_legacy_selection_state_alias()
    state_file = _state_file_for_selection(selection)
    raw = _read_json(state_file, None)
    if raw is None and state_file != STATE_FILE:
        # Keep older unsuffixed state files readable after the suffix migration.
        legacy_raw = legacy_state if legacy_state is not None else _read_json(STATE_FILE, None)
        if isinstance(legacy_raw, dict) and selection is not None and _state_matches_selection(legacy_raw, selection):
            raw = legacy_raw
    if raw is None and selection is None:
        raw = _read_json(STATE_FILE, None)
    if not isinstance(raw, dict):
        raw = {}
    state = _empty_state()
    state.update(raw)
    if selection is None and (not state.get("active") or _selection_from_state(state) is None):
        fallback_state = _latest_active_saved_state()
        if fallback_state is not None:
            return fallback_state
    return state


def _save_state(state: dict[str, Any], *, update_current_alias: bool = True) -> None:
    if update_current_alias:
        # Preserve any older unsuffixed selection record before reusing state.json
        # as the current-selection alias for a different environment.
        _migrate_legacy_selection_state_alias()
    state["updated_at"] = _utc_now_iso()
    state_file = _state_file_for_selection(state.get("selection") if isinstance(state, dict) else None)
    _write_json(state_file, state)
    if update_current_alias and state_file != STATE_FILE:
        # Preserve the current-selection alias used by commands that do not take
        # an explicit port-profile selection.
        _write_json(STATE_FILE, state)


def _service_urls_for_multi_selection(backend_selections: list[dict[str, Any]]) -> dict[str, Any]:
    backend_urls: dict[str, Any] = {}
    for backend in backend_selections:
        profile_id = str(backend["port_profile_id"])
        backend_urls[profile_id] = backend["service_urls"]
    control_urls = backend_selections[0]["service_urls"] if backend_selections else {}
    return {
        "backend_service_urls": backend_urls,
        "gateway": control_urls.get("gateway"),
        "gateway_parse": control_urls.get("gateway_parse"),
    }


def _validate_start_selection_multi(
    *,
    model_key: str,
    port_profile_ids: list[int],
    launch_profile_key: str,
    lmcache: int | None = None,
    gpu_memory_utilization: float | None = None,
    enforce_weight_limit: bool = True,
) -> tuple[bool, dict[str, Any]]:
    try:
        images = _single._load_image_config()
        _, models = _single._load_models_config()
        _, port_profiles = _single._load_port_profiles()
        _, launch_profiles = _single._load_launch_profiles()
        normalized_lmcache = _single._normalize_lmcache_size(lmcache)
        normalized_gpu_memory_utilization = _single._normalize_gpu_memory_utilization(
            gpu_memory_utilization
        )
    except Exception as exc:
        return False, _payload(ok=False, code=710, message=f"failed loading configs: {exc}")

    if not port_profile_ids:
        return False, _payload(ok=False, code=711, message="at least one port profile id is required")

    model = models.get(model_key)
    if model is None:
        return False, _payload(ok=False, code=712, message=f"unknown model key: {model_key}")

    launch_profile = launch_profiles.get(launch_profile_key)
    if launch_profile is None:
        return False, _payload(ok=False, code=713, message=f"unknown launch profile: {launch_profile_key}")

    try:
        split_launches = _split_launch_profile_across_backends(
            launch_profile=launch_profile,
            backend_count=len(port_profile_ids),
        )
    except ValueError as exc:
        return False, _payload(ok=False, code=714, message=str(exc))

    model_weight = float(model["weight_vram_gb"])
    backend_selections: list[dict[str, Any]] = []
    resolved_ports: list[dict[str, Any]] = []
    for profile_id, backend_launch in zip(port_profile_ids, split_launches):
        port_key = str(profile_id)
        port_profile = port_profiles.get(port_key)
        if port_profile is None:
            return False, _payload(ok=False, code=715, message=f"unknown port profile id: {profile_id}")

        if enforce_weight_limit:
            total_gpu_memory = float(backend_launch["total_gpu_memory_gb"])
            limit = 0.75 * total_gpu_memory
            if model_weight > limit:
                return False, _payload(
                    ok=False,
                    code=716,
                    message=(
                        "launch rejected: model weight exceeds 75% of per-backend GPU memory "
                        f"(weight={model_weight}GB, limit={round(limit, 3)}GB, port_profile={profile_id})"
                    ),
                    data={
                        "model_key": model_key,
                        "weight_vram_gb": model_weight,
                        "launch_profile": launch_profile_key,
                        "port_profile_id": profile_id,
                        "backend_total_gpu_memory_gb": total_gpu_memory,
                        "max_allowed_weight_gb": round(limit, 3),
                    },
                )

        backend_selection = {
            "model_key": model_key,
            "port_profile_id": port_key,
            "launch_profile_key": launch_profile_key,
            "lmcache": normalized_lmcache,
            "gpu_memory_utilization": normalized_gpu_memory_utilization,
            "images": images,
            "model": model,
            "ports": port_profile,
            "launch": backend_launch,
            "runtime_names": _backend_runtime_names_for_selection(
                model_key=model_key,
                launch_profile_key=launch_profile_key,
                port_profile_id=profile_id,
            ),
        }
        backend_selection["service_urls"] = _single._service_urls_from_ports(port_profile)
        backend_selections.append(backend_selection)
        resolved_ports.append(port_profile)

    resolved = {
        "model_key": model_key,
        "port_profile_ids": list(port_profile_ids),
        "launch_profile_key": launch_profile_key,
        "lmcache": normalized_lmcache,
        "gpu_memory_utilization": normalized_gpu_memory_utilization,
        "assignment_policy": DEFAULT_ASSIGNMENT_POLICY,
        "gateway_mode": DEFAULT_GATEWAY_MULTI_MODULE_NAME,
        "gateway_ctx": False,
        "control_port_profile_id": port_profile_ids[0],
        "images": images,
        "model": model,
        "launch": launch_profile,
        "backend_launches": split_launches,
        "backend_selections": backend_selections,
        "ports_by_profile": {str(pid): port for pid, port in zip(port_profile_ids, resolved_ports)},
        "service_urls": _service_urls_for_multi_selection(backend_selections),
    }
    return True, _payload(ok=True, code=0, message="selection validated", data=resolved)


def _new_startup_log_path(
    *,
    model_key: str,
    port_profile_ids: list[int],
    launch_profile_key: str,
) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    model_token = _sanitize_path_token(model_key)
    ports_token = _sanitize_path_token("-".join(str(value) for value in port_profile_ids))
    launch_token = _sanitize_path_token(launch_profile_key)
    return STARTUP_LOGS_DIR / f"start-{stamp}-{model_token}-p{ports_token}-{launch_token}.log"


def _new_compose_log_path(startup_log_path: Path) -> Path:
    return startup_log_path.with_suffix(".compose.log")


def _compose_contexts_for_multi_selection(
    selection: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]], str]:
    contexts: list[dict[str, Any]] = []
    for backend_selection in selection["backend_selections"]:
        ok, context, error = _single._compose_context_for_selection(backend_selection)
        if not ok:
            for existing in contexts:
                _single._cleanup_compose_context(existing)
            return False, [], error
        context["selection"] = backend_selection
        contexts.append(context)
    return True, contexts, ""


def _compose_up_backend(context: dict[str, Any], *, startup_log_path: Path | None = None) -> Any:
    return _single._run_compose_streaming(
        ["up", "-d", "--pull", "always", "jaeger", "vllm"],
        output_path=startup_log_path,
        project_name=str(context["project_name"]),
        compose_env_file=Path(context["env_file"]),
        compose_gpu_override_file=Path(context["gpu_override_file"]),
    )


def _compose_up_backends_parallel(
    contexts: list[dict[str, Any]],
    *,
    startup_log_path: Path,
    compose_log_path: Path,
) -> list[dict[str, Any]]:
    if not contexts:
        return []

    for context in contexts:
        backend_selection = context["selection"]
        _single._emit_progress(
            "running docker compose up -d --pull always jaeger vllm"
            f" port_profile={backend_selection['port_profile_id']}"
            f" visible_devices={backend_selection['launch']['visible_devices']}",
            log_path=startup_log_path,
        )

    results: dict[int, dict[str, Any]] = {}
    with futures.ThreadPoolExecutor(max_workers=len(contexts)) as executor:
        future_to_index = {
            executor.submit(_compose_up_backend, context, startup_log_path=startup_log_path): index
            for index, context in enumerate(contexts)
        }
        for future in futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                up = future.result()
            except Exception as exc:
                up = _single.ExecResult(
                    returncode=1,
                    stdout="",
                    stderr="",
                    error=f"unexpected error during docker compose up: {exc}",
                )
            results[index] = {
                "context": contexts[index],
                "up": up,
            }

    ordered_results: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        _append_compose_logs_for_backend(context, compose_log_path)
        ordered_results.append(results[index])
    return ordered_results


def _compose_down_backend(
    context: dict[str, Any],
    *,
    stream_output: bool = False,
    output_path: Path | None = None,
) -> Any:
    return _single._compose_down(
        stream_output=stream_output,
        output_path=output_path,
        project_name=str(context["project_name"]),
        compose_env_file=Path(context["env_file"]),
        compose_gpu_override_file=Path(context["gpu_override_file"]),
        ensure_env_file=False,
    )


def _append_compose_logs_for_backend(context: dict[str, Any], compose_log_path: Path) -> None:
    backend_selection = context["selection"]
    captured = _single._compose_logs_full(
        project_name=str(context["project_name"]),
        compose_env_file=Path(context["env_file"]),
        compose_gpu_override_file=Path(context["gpu_override_file"]),
        ensure_env_file=False,
    )
    profile_id = backend_selection["port_profile_id"]
    header = f"[{_utc_now_iso()}] backend port_profile={profile_id} compose logs\n"
    _append_text(compose_log_path, header)
    if captured.get("ok"):
        logs = str(captured.get("logs", ""))
        if logs and not logs.endswith("\n"):
            logs += "\n"
        _append_text(compose_log_path, logs)
    else:
        _append_text(compose_log_path, f"compose log capture failed: {captured.get('error')}\n")


def _backend_health_snapshot(backend_selection: dict[str, Any]) -> dict[str, Any]:
    ports = backend_selection["ports"]
    runtime_names = backend_selection["runtime_names"]
    vllm_port = _single._parse_port(ports.get("vllm_port"), "resolved.ports.vllm_port")
    jaeger_api_port = _single._parse_port(ports.get("jaeger_api_port"), "resolved.ports.jaeger_api_port")
    jaeger_otlp_port = _single._parse_port(ports.get("jaeger_otlp_port"), "resolved.ports.jaeger_otlp_port")
    checks = {
        "vllm": _single._http_get_health(
            f"http://127.0.0.1:{vllm_port}/v1/models",
            allow_redirect=False,
        ),
        "jaeger_api": _single._http_get_health(
            f"http://127.0.0.1:{jaeger_api_port}/api/services",
            allow_redirect=False,
        ),
        "jaeger_ui": _single._http_get_health(
            f"http://127.0.0.1:{jaeger_api_port}/",
            allow_redirect=True,
        ),
        "jaeger_otlp": _single._tcp_health("127.0.0.1", jaeger_otlp_port),
    }
    containers = {
        "jaeger": _single._docker_container_snapshot(runtime_names["jaeger_container_name"]),
        "vllm": _single._docker_container_snapshot(runtime_names["vllm_container_name"]),
    }
    diagnostics: dict[str, Any] = {}
    vllm_container = containers["vllm"]
    should_capture_vllm_logs = not checks["vllm"]["ok"] or bool(vllm_container.get("restart_count"))
    if should_capture_vllm_logs:
        vllm_logs = _single._docker_logs_tail(runtime_names["vllm_container_name"])
        diagnostics["vllm"] = {
            "log_hint": _single._extract_log_hint(vllm_logs.get("logs", ""))
            if vllm_logs.get("ok")
            else None,
            "recent_logs": {
                "lines": vllm_logs.get("lines"),
                "logs": _single._tail_lines(
                    vllm_logs.get("logs", ""),
                    lines=_single.DEFAULT_STARTUP_LOG_TAIL_LINES,
                ),
                "error": vllm_logs.get("error"),
            },
        }
    return {
        "port_profile_id": backend_selection["port_profile_id"],
        "ok": all(bool(item.get("ok")) for item in checks.values()),
        "runtime_names": runtime_names,
        "containers": containers,
        "diagnostics": diagnostics,
        "services": checks,
    }


def _find_residual_vllm_processes(backend_selection: dict[str, Any]) -> list[dict[str, Any]]:
    ports = backend_selection.get("ports")
    if not isinstance(ports, dict):
        return []
    vllm_port = _single._parse_port(ports.get("vllm_port"), "resolved.ports.vllm_port")
    proc_listing = _single._run_exec(["ps", "-eo", "pid=,args="], timeout_seconds=10.0)
    if proc_listing.error or proc_listing.returncode != 0:
        return []

    port_tokens = (f"--port {vllm_port}", f"--port={vllm_port}")
    matches: list[dict[str, Any]] = []
    for raw_line in proc_listing.stdout.splitlines():
        line = raw_line.strip()
        if not line or "vllm.entrypoints.openai.api_server" not in line:
            continue
        if not any(token in line for token in port_tokens):
            continue
        parts = line.split(None, 1)
        pid = _single._coerce_int(parts[0]) if parts else None
        if pid is None:
            continue
        matches.append(
            {
                "pid": pid,
                "cmdline": parts[1] if len(parts) > 1 else "",
            }
        )
    return matches


def _build_multi_health_snapshot(selection: dict[str, Any]) -> dict[str, Any]:
    backends = [_backend_health_snapshot(item) for item in selection["backend_selections"]]
    control_ports = selection["backend_selections"][0]["ports"]
    gateway_port = _single._coerce_int(control_ports.get("gateway_port"))
    gateway_parse_port = _single._coerce_int(control_ports.get("gateway_parse_port"))
    gateway_checks: dict[str, Any] = {}
    if gateway_port is not None:
        gateway_checks["gateway"] = _single._http_get_health(
            f"http://127.0.0.1:{gateway_port}/healthz",
            allow_redirect=False,
        )
    if gateway_parse_port is not None:
        gateway_checks["gateway_parse"] = _single._http_get_health(
            f"http://127.0.0.1:{gateway_parse_port}/healthz",
            allow_redirect=False,
        )
    overall_ok = all(backend["ok"] for backend in backends) and all(
        bool(item.get("ok")) for item in gateway_checks.values()
    )
    return {
        "ok": overall_ok,
        "checked_at": _utc_now_iso(),
        "selection_identity": {
            "model_key": selection["model_key"],
            "port_profile_ids": selection["port_profile_ids"],
            "launch_profile_key": selection["launch_profile_key"],
        },
        "assignment_policy": selection["assignment_policy"],
        "control_port_profile_id": selection["control_port_profile_id"],
        "gateway_daemon": _gateway_multi_status(selection),
        "gateway_services": gateway_checks,
        "backends": backends,
    }


def _build_multi_stop_residual_snapshot(selection: dict[str, Any]) -> dict[str, Any]:
    backend_residuals: list[dict[str, Any]] = []
    for backend_selection in selection["backend_selections"]:
        snapshot = _backend_health_snapshot(backend_selection)
        live_processes = _find_residual_vllm_processes(backend_selection)
        alive_services = [
            name
            for name, payload in snapshot["services"].items()
            if isinstance(payload, dict) and payload.get("ok")
        ]
        live_containers = [
            name
            for name, payload in snapshot["containers"].items()
            if isinstance(payload, dict)
            and payload.get("present")
            and (payload.get("running") or payload.get("restarting"))
        ]
        if alive_services or live_containers or live_processes:
            backend_residuals.append(
                {
                    "port_profile_id": snapshot["port_profile_id"],
                    "alive_services": alive_services,
                    "live_containers": live_containers,
                    "live_processes": live_processes,
                    "snapshot": snapshot,
                }
            )

    gateway_status = _gateway_multi_status(selection)
    gateway_checks: dict[str, Any] = {}
    backend_selections = selection.get("backend_selections")
    control_ports = backend_selections[0]["ports"] if isinstance(backend_selections, list) and backend_selections else {}
    gateway_port = _single._coerce_int(control_ports.get("gateway_port")) if isinstance(control_ports, dict) else None
    gateway_parse_port = (
        _single._coerce_int(control_ports.get("gateway_parse_port")) if isinstance(control_ports, dict) else None
    )
    if gateway_port is not None:
        gateway_checks["gateway"] = _single._http_get_health(
            f"http://127.0.0.1:{gateway_port}/healthz",
            allow_redirect=False,
        )
    if gateway_parse_port is not None:
        gateway_checks["gateway_parse"] = _single._http_get_health(
            f"http://127.0.0.1:{gateway_parse_port}/healthz",
            allow_redirect=False,
        )
    alive_gateway_services = [
        name
        for name, payload in gateway_checks.items()
        if isinstance(payload, dict) and payload.get("ok")
    ]
    gateway_residual = bool(gateway_status.get("running")) or bool(alive_gateway_services)
    return {
        "ok": not backend_residuals and not gateway_residual,
        "backends": backend_residuals,
        "gateway_multi": {
            "running": bool(gateway_status.get("running")),
            "alive_services": alive_gateway_services,
            "status": gateway_status,
            "services": gateway_checks,
        },
    }


def _stop_residual_vllm_processes(
    residual_snapshot: dict[str, Any],
    *,
    show_progress: bool = False,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    backends = residual_snapshot.get("backends")
    if not isinstance(backends, list):
        return []

    seen_pids: set[int] = set()
    results: list[dict[str, Any]] = []
    for backend in backends:
        if not isinstance(backend, dict):
            continue
        port_profile_id = backend.get("port_profile_id")
        live_processes = backend.get("live_processes")
        if not isinstance(live_processes, list):
            continue
        for process in live_processes:
            if not isinstance(process, dict):
                continue
            pid = _single._coerce_int(process.get("pid"))
            if pid is None or pid in seen_pids:
                continue
            seen_pids.add(pid)
            if show_progress:
                _single._emit_progress(
                    f"stopping residual vllm process pid={pid} port_profile={port_profile_id}",
                    log_path=log_path,
                )
            stopped = _single._stop_pid(pid, timeout_seconds=10.0)
            results.append(
                {
                    "port_profile_id": port_profile_id,
                    "pid": pid,
                    "cmdline": process.get("cmdline"),
                    "stopped": stopped,
                }
            )
    return results


def _format_wait_progress(snapshot: dict[str, Any], *, attempts: int, elapsed_seconds: float) -> str:
    parts = [f"wait-up attempt={attempts} elapsed={elapsed_seconds:.1f}s"]
    gateway_services = snapshot.get("gateway_services")
    if isinstance(gateway_services, dict):
        for name in ("gateway", "gateway_parse"):
            payload = gateway_services.get(name)
            if isinstance(payload, dict):
                parts.append(_single._format_health_service_status(name, payload))
    backends = snapshot.get("backends")
    if isinstance(backends, list):
        for backend in backends:
            if not isinstance(backend, dict):
                continue
            profile_id = backend.get("port_profile_id")
            services = backend.get("services")
            if isinstance(services, dict):
                service_parts: list[str] = []
                for service_name in ("vllm", "jaeger_api", "jaeger_ui", "jaeger_otlp"):
                    payload = services.get(service_name)
                    if isinstance(payload, dict):
                        service_parts.append(
                            _single._format_health_service_status(service_name, payload)
                        )
                if service_parts:
                    parts.append(f"p{profile_id}:{','.join(service_parts)}")
    return " | ".join(parts)


def _wait_up_multi_impl(
    *,
    selection: dict[str, Any],
    timeout_seconds: float,
    poll_interval_seconds: float,
    show_progress: bool = False,
    startup_log_path: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + max(timeout_seconds, 0.0)
    attempts = 0
    last_snapshot: dict[str, Any] | None = None
    while True:
        attempts += 1
        snapshot = _build_multi_health_snapshot(selection)
        elapsed = round(time.monotonic() - started, 3)
        if snapshot["ok"]:
            if show_progress:
                _single._emit_progress(
                    f"wait-up completed successfully after {elapsed:.1f}s",
                    log_path=startup_log_path,
                )
            return _payload(
                ok=True,
                code=0,
                message="services are up",
                data={
                    "attempts": attempts,
                    "elapsed_seconds": elapsed,
                    "snapshot": snapshot,
                },
            )
        last_snapshot = snapshot
        if show_progress:
            _single._emit_progress(
                _format_wait_progress(snapshot, attempts=attempts, elapsed_seconds=elapsed),
                log_path=startup_log_path,
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


def _start_impl(
    *,
    model_key: str,
    port_profile_ids: list[int],
    launch_profile_key: str,
    lmcache: int | None,
    gpu_memory_utilization: float | None,
    block: bool,
    timeout_seconds: float,
    startup_log_path: Path,
    compose_log_path: Path,
) -> dict[str, Any]:
    valid, selection_payload = _validate_start_selection_multi(
        model_key=model_key,
        port_profile_ids=port_profile_ids,
        launch_profile_key=launch_profile_key,
        lmcache=lmcache,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if not valid:
        return selection_payload
    selection = selection_payload["data"]

    context_ok, contexts, context_error = _compose_contexts_for_multi_selection(selection)
    if not context_ok:
        return _payload(
            ok=False,
            code=720,
            message=f"failed to prepare compose contexts: {context_error}",
        )

    started_contexts: list[dict[str, Any]] = []
    start_failed = False
    try:
        _append_text(
            startup_log_path,
            (
                f"[{_utc_now_iso()}] startup log created\n"
                f"[{_utc_now_iso()}] startup_log_file={startup_log_path}\n"
                f"[{_utc_now_iso()}] compose_log_file={compose_log_path}\n"
            ),
        )
        _single._emit_progress(
            "resolved multi selection"
            f" model={selection['model_key']}"
            f" launch={selection['launch_profile_key']}"
            f" port_profiles={selection['port_profile_ids']}"
            f" assignment_policy={selection['assignment_policy']}"
            f" lmcache={selection['lmcache'] if selection.get('lmcache') is not None else 'disabled'}"
            f" gpu_memory_utilization={selection['gpu_memory_utilization'] if selection.get('gpu_memory_utilization') is not None else 'default'}",
            log_path=startup_log_path,
        )

        compose_results = _compose_up_backends_parallel(
            contexts,
            startup_log_path=startup_log_path,
            compose_log_path=compose_log_path,
        )
        started_contexts = [
            result["context"]
            for result in compose_results
            if not result["up"].error and result["up"].returncode == 0
        ]
        for result in compose_results:
            context = result["context"]
            backend_selection = context["selection"]
            up = result["up"]
            if up.error:
                start_failed = True
                return _payload(
                    ok=False,
                    code=721,
                    message=f"docker compose up failed for port profile {backend_selection['port_profile_id']}: {up.error}",
                    data={
                        "port_profile_id": backend_selection["port_profile_id"],
                        "stdout": up.stdout,
                        "stderr": up.stderr,
                    },
                )
            if up.returncode != 0:
                start_failed = True
                return _payload(
                    ok=False,
                    code=722,
                    message=f"docker compose up failed for port profile {backend_selection['port_profile_id']}",
                    data={
                        "port_profile_id": backend_selection["port_profile_id"],
                        "returncode": up.returncode,
                        "stdout": up.stdout,
                        "stderr": up.stderr,
                    },
                )

        _single._emit_progress("starting local gateway_multi daemon", log_path=startup_log_path)
        gateway_start = _start_gateway_multi(selection)
        if not gateway_start.get("ok"):
            start_failed = True
            _single._emit_progress(
                f"gateway_multi startup failed: {gateway_start.get('message')}",
                log_path=startup_log_path,
            )
            return _payload(
                ok=False,
                code=723,
                message="gateway_multi startup failed",
                data={
                    "gateway_start": gateway_start.get("data"),
                    "gateway_error": gateway_start.get("message"),
                },
            )
        _single._emit_progress("gateway_multi daemon started", log_path=startup_log_path)

        new_state = {
            "active": True,
            "lifecycle_state": "starting" if block else "running",
            "started_at": _utc_now_iso(),
            "selection": {
                "model_key": selection["model_key"],
                "port_profile_ids": selection["port_profile_ids"],
                "launch_profile_key": selection["launch_profile_key"],
                "lmcache": selection.get("lmcache"),
                "gpu_memory_utilization": selection.get("gpu_memory_utilization"),
                "assignment_policy": selection["assignment_policy"],
            },
            "resolved": {
                "images": selection["images"],
                "model": selection["model"],
                "launch": selection["launch"],
                "gateway_mode": selection["gateway_mode"],
                "service_urls": selection["service_urls"],
                "control_port_profile_id": selection["control_port_profile_id"],
                "backend_selections": [
                    {
                        "model_key": item["model_key"],
                        "port_profile_id": item["port_profile_id"],
                        "launch_profile_key": item["launch_profile_key"],
                        "ports": item["ports"],
                        "launch": item["launch"],
                        "runtime_names": item["runtime_names"],
                        "service_urls": item["service_urls"],
                    }
                    for item in selection["backend_selections"]
                ],
            },
            "gateway": gateway_start.get("data"),
        }
        _save_state(new_state)

        response_data = {
            "selection": new_state["selection"],
            "resolved": new_state["resolved"],
            "gateway_start": gateway_start.get("data"),
            "startup_log_file": str(startup_log_path),
            "compose_log_file": str(compose_log_path),
        }

        if not block:
            return _payload(ok=True, code=0, message="environment start requested", data=response_data)

        _single._emit_progress(
            "waiting for health checks"
            f" timeout={timeout_seconds}s"
            f" control_port_profile={selection['control_port_profile_id']}"
            f" backend_profiles={selection['port_profile_ids']}",
            log_path=startup_log_path,
        )
        wait_payload = _wait_up_multi_impl(
            selection=selection,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=DEFAULT_WAIT_UP_POLL_INTERVAL_SECONDS,
            show_progress=True,
            startup_log_path=startup_log_path,
        )
        state_after_wait = _load_state(selection)
        if wait_payload.get("ok"):
            state_after_wait["lifecycle_state"] = "running"
            _save_state(state_after_wait)
            response_data["wait_up"] = wait_payload.get("data", {})
            return _payload(ok=True, code=0, message="environment started and healthy", data=response_data)

        state_after_wait["lifecycle_state"] = "degraded"
        _save_state(state_after_wait)
        response_data["wait_up"] = wait_payload.get("data", {})
        return _payload(
            ok=False,
            code=int(wait_payload.get("code", 504)),
            message=str(wait_payload.get("message") or "environment failed to become healthy"),
            data=response_data,
        )
    finally:
        if start_failed:
            for context in reversed(started_contexts):
                _compose_down_backend(context, stream_output=True, output_path=startup_log_path)
        for context in contexts:
            _single._cleanup_compose_context(context)


def _selection_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    selection = state.get("selection")
    if not isinstance(selection, dict):
        return None
    identity = _selection_identity(selection)
    if identity is None:
        return None
    valid, selection_payload = _validate_start_selection_multi(
        model_key=identity[0],
        port_profile_ids=list(identity[1]),
        launch_profile_key=identity[2],
        lmcache=_single._coerce_int(selection.get("lmcache")),
        gpu_memory_utilization=selection.get("gpu_memory_utilization"),
        enforce_weight_limit=False,
    )
    if not valid:
        return None
    resolved = selection_payload.get("data")
    return resolved if isinstance(resolved, dict) else None


def _status_impl() -> dict[str, Any]:
    state = _load_state()
    selection = _selection_from_state(state) if state.get("active") else None
    health = _build_multi_health_snapshot(selection) if selection is not None else None
    data = {
        "active_environment": state if state.get("active") else None,
        "lifecycle_state": state.get("lifecycle_state"),
        "gateway_multi": _gateway_multi_status(selection),
        "health": health,
    }
    return _payload(ok=True, code=0, message="status", data=data)


def _up_impl() -> dict[str, Any]:
    state = _load_state()
    if not state.get("active"):
        return _payload(ok=False, code=404, message="no active environment")
    selection = _selection_from_state(state)
    if selection is None:
        return _payload(ok=False, code=500, message="failed to reconstruct active selection")
    snapshot = _build_multi_health_snapshot(selection)
    if snapshot["ok"]:
        return _payload(ok=True, code=0, message="all services healthy", data=snapshot)
    return _payload(ok=False, code=503, message="one or more services unhealthy", data=snapshot)


def _logs_impl(lines: int) -> dict[str, Any]:
    if lines <= 0:
        return _payload(ok=False, code=760, message="lines must be > 0")
    state = _load_state()
    selection = _selection_from_state(state)
    if selection is None:
        return _payload(ok=False, code=761, message="no active environment")
    backend_logs: dict[str, Any] = {}
    for backend in selection["backend_selections"]:
        runtime_names = backend["runtime_names"]
        backend_logs[str(backend["port_profile_id"])] = {
            "vllm": _single._docker_logs_tail(
                runtime_names["vllm_container_name"],
                lines=lines,
            ),
            "jaeger": _single._docker_logs_tail(
                runtime_names["jaeger_container_name"],
                lines=lines,
            ),
        }
    _, gateway_pid_file, gateway_log_file = _load_gateway_multi_record(selection["port_profile_ids"])
    gateway_logs = {
        "status": _gateway_multi_status(selection),
        "lines": lines,
        "logs": _tail_text(gateway_log_file, lines=lines),
        "pid_file": str(gateway_pid_file),
        "log_file": str(gateway_log_file),
    }
    return _payload(
        ok=True,
        code=0,
        message="logs",
        data={
            "lines": lines,
            "backend_logs": backend_logs,
            "gateway_multi": gateway_logs,
        },
    )


def _stop_impl(
    *,
    selection: dict[str, Any] | None,
    reason: str,
    show_progress: bool = False,
    log_path: Path | None = None,
) -> dict[str, Any]:
    current_state = _load_state()
    target_selection = selection or _selection_from_state(current_state)
    if target_selection is None:
        return _payload(ok=True, code=0, message="environment already stopped")
    target_state = _load_state(target_selection)
    current_matches_target = _state_matches_selection(current_state, target_selection)

    context_ok, contexts, context_error = _compose_contexts_for_multi_selection(target_selection)
    if not context_ok:
        return _payload(
            ok=False,
            code=730,
            message=f"failed to prepare stop compose contexts: {context_error}",
        )

    try:
        if show_progress:
            _single._emit_progress(
                "stopping gateway_multi and backend compose projects"
                f" port_profiles={target_selection['port_profile_ids']}",
                log_path=log_path,
            )
        gateway_stop = _stop_gateway_multi(target_selection["port_profile_ids"])
        compose_results: list[dict[str, Any]] = []
        ok = bool(gateway_stop.get("ok"))
        for context in contexts:
            backend_selection = context["selection"]
            if show_progress:
                _single._emit_progress(
                    f"running docker compose down for port_profile={backend_selection['port_profile_id']}",
                    log_path=log_path,
                )
            down = _compose_down_backend(context, stream_output=show_progress, output_path=log_path)
            compose_results.append(
                {
                    "port_profile_id": backend_selection["port_profile_id"],
                    "project_name": context["project_name"],
                    "returncode": down.returncode,
                    "stdout": down.stdout,
                    "stderr": down.stderr,
                    "error": down.error,
                }
            )
            if down.error or down.returncode != 0:
                ok = False

        residual_snapshot = _build_multi_stop_residual_snapshot(target_selection)
        residual_cleanup: list[dict[str, Any]] = []
        if not residual_snapshot["ok"]:
            residual_cleanup = _stop_residual_vllm_processes(
                residual_snapshot,
                show_progress=show_progress,
                log_path=log_path,
            )
            if residual_cleanup:
                residual_snapshot = _build_multi_stop_residual_snapshot(target_selection)
        if not residual_snapshot["ok"]:
            ok = False

        if _state_matches_selection(target_state, target_selection):
            if ok:
                target_state["active"] = False
                target_state["lifecycle_state"] = "inactive"
                target_state["last_stop_reason"] = reason
                target_state["last_stop_completed_at"] = _utc_now_iso()
            else:
                target_state["active"] = True
                target_state["lifecycle_state"] = "degraded"
                target_state["last_stop_attempt_reason"] = reason
                target_state["last_stop_attempted_at"] = _utc_now_iso()
            _save_state(target_state, update_current_alias=False)
        if current_matches_target:
            if ok:
                _write_current_state_alias(_latest_active_saved_state(exclude_selection=target_selection))
            else:
                _write_current_state_alias(target_state)

        return _payload(
            ok=ok,
            code=0 if ok else 731,
            message="environment stopped" if ok else "failed to stop one or more services",
            data={
                "reason": reason,
                "selection": {
                    "model_key": target_selection["model_key"],
                    "port_profile_ids": target_selection["port_profile_ids"],
                    "launch_profile_key": target_selection["launch_profile_key"],
                },
                "gateway_stop": gateway_stop,
                "compose_down": compose_results,
                "residual_cleanup": residual_cleanup,
                "residual_services": residual_snapshot,
            },
        )
    finally:
        for context in contexts:
            _single._cleanup_compose_context(context)


@profiles_app.command(name="models")
def profiles_models() -> None:
    """List allowed model keys."""
    try:
        default_model, models = _single._load_models_config()
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
        default_profile, profiles = _single._load_port_profiles()
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
        default_profile, profiles = _single._load_launch_profiles()
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
    port_profile: list[str] = typer.Option(
        ...,
        "--port-profile",
        "-p",
        help="Port profile IDs. Repeat or pass a comma-separated list like 0,1.",
    ),
    launch_profile: str = typer.Option(..., "--launch-profile", "-l", help="Launch profile key."),
    lmcache: int | None = typer.Option(
        None,
        "--lmcache",
        help="Enable LMCache with a maximum local CPU size for each backend.",
    ),
    gpu_memory_utilization: float | None = typer.Option(
        None,
        "--gpu-memory-utilization",
        help="Pass through to each backend vLLM as --gpu-memory-utilization <value> (0 < value <= 1).",
    ),
    block: bool = typer.Option(False, "--block", "-b", help="Block until endpoints are healthy."),
    timeout_seconds: float = typer.Option(
        DEFAULT_WAIT_UP_TIMEOUT_SECONDS,
        "--timeout-seconds",
        help="Timeout for --block health wait.",
    ),
) -> None:
    """Start one multi-backend environment."""
    try:
        port_profile_ids = _parse_port_profile_ids(port_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--port-profile") from exc
    if lmcache is not None and lmcache <= 0:
        raise typer.BadParameter("--lmcache must be a positive integer", param_hint="--lmcache")
    try:
        _single._normalize_gpu_memory_utilization(gpu_memory_utilization)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--gpu-memory-utilization") from exc

    startup_log_path = _new_startup_log_path(
        model_key=model,
        port_profile_ids=port_profile_ids,
        launch_profile_key=launch_profile,
    )
    compose_log_path = _new_compose_log_path(startup_log_path)
    interrupt_selection: dict[str, Any] | None = None
    valid_interrupt_selection, interrupt_selection_payload = _validate_start_selection_multi(
        model_key=model,
        port_profile_ids=port_profile_ids,
        launch_profile_key=launch_profile,
        lmcache=lmcache,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_weight_limit=False,
    )
    if valid_interrupt_selection:
        resolved_interrupt_selection = interrupt_selection_payload.get("data")
        if isinstance(resolved_interrupt_selection, dict):
            interrupt_selection = resolved_interrupt_selection
    try:
        _emit(
            _start_impl(
                model_key=model,
                port_profile_ids=port_profile_ids,
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
            _single._emit_progress(
                "start interrupted by user; cleaning up managed environment",
                log_path=startup_log_path,
            )
            cleanup_payload = _stop_impl(
                selection=interrupt_selection,
                reason="start_interrupted",
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
    state = _load_state()
    selection = _selection_from_state(state)
    if selection is None:
        _emit(_payload(ok=False, code=404, message="no active environment"))
        return
    _emit(
        _wait_up_multi_impl(
            selection=selection,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            show_progress=True,
        )
    )


@app.command(name="logs")
def logs(
    lines: int = typer.Option(200, "--lines", "-n", help="Number of lines to tail."),
) -> None:
    """Tail recent vLLM + Jaeger docker logs and gateway_multi log."""
    _emit(_logs_impl(lines))


@app.command(name="stop")
def stop(
    block: bool = typer.Option(False, "--block", "-b", help="Block until teardown completes."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model key used at start time."),
    port_profile: list[str] | None = typer.Option(
        None,
        "--port-profile",
        "-p",
        help="Port profile IDs used at start time. Repeat or pass a comma-separated list.",
    ),
    launch_profile: str | None = typer.Option(None, "--launch-profile", "-l", help="Launch profile key used at start time."),
) -> None:
    """Stop environment. Multi stop runs blocking even without --block."""
    selection: dict[str, Any] | None = None
    provided_count = int(model is not None) + int(bool(port_profile)) + int(launch_profile is not None)
    if provided_count not in {0, 3}:
        _emit(
            _payload(
                ok=False,
                code=732,
                message="stop selection requires --model, --port-profile, and --launch-profile together",
            )
        )
        return
    if provided_count == 3:
        assert model is not None
        assert port_profile is not None
        assert launch_profile is not None
        try:
            port_profile_ids = _parse_port_profile_ids(port_profile)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--port-profile") from exc
        valid, selection_payload = _validate_start_selection_multi(
            model_key=model.strip(),
            port_profile_ids=port_profile_ids,
            launch_profile_key=launch_profile.strip(),
            enforce_weight_limit=False,
        )
        if not valid:
            _emit(selection_payload)
            return
        selection = selection_payload["data"]

    _ = block
    _emit(_stop_impl(selection=selection, reason="stop_blocking", show_progress=True))


@app.command(name="daemon-stop")
def daemon_stop() -> None:
    """Compatibility alias for blocking stop."""
    _emit(_stop_impl(selection=None, reason="daemon_stop", show_progress=True))


def _main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
