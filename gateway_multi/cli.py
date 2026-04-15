from __future__ import annotations

import asyncio
from dataclasses import replace
import errno
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from typing import Sequence

import typer
import uvicorn

from gateway.model_configs import load_model_registry
from gateway.port_profiles import load_port_profile
from gateway_multi.app import (
    create_control_app,
    create_gateway_service,
    create_ipc_app,
    normalize_assignment_policy,
    normalize_balanced_usage_threshold_tokens,
)
from gateway_multi.runtime_config import (
    DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS,
    DEFAULT_CONFIG_PATH,
    load_runtime_settings,
)


app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_VENV_DIR = REPO_ROOT / ".venv"
_BOOTSTRAP_ENV = "VLLM_OTEL_GATEWAY_MULTI_BOOTSTRAPPED"
DEFAULT_IPC_SOCKET_DIR = Path("/tmp")


@app.callback()
def gateway_app() -> None:
    """Gateway multi CLI."""


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_config_exists(config_path: Path) -> None:
    if config_path.exists():
        return
    typer.secho(f"error: config file not found: {config_path}", err=True, fg=typer.colors.RED)
    typer.secho(
        "hint: copy gateway_multi/config.example.toml to gateway_multi/config.toml and edit as needed",
        err=True,
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=1)


def _create_venv(venv_dir: Path) -> None:
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)


def _install_gateway(venv_python: Path) -> None:
    subprocess.run([str(venv_python), "-m", "pip", "install", "-e", str(PACKAGE_ROOT)], check=True)


def _reexec_in_venv(
    *,
    venv_python: Path,
    config_path: Path,
    host: str,
    venv_dir: Path,
    port_profile_ids: Sequence[int],
    policy: str | None,
    balanced_usage_threshold_tokens: int | None,
    skip_install: bool,
) -> None:
    command = [
        str(venv_python),
        "-m",
        "gateway_multi",
        "start",
        "--config",
        str(config_path),
        "--host",
        host,
        "--venv-dir",
        str(venv_dir),
    ]
    for port_profile_id in port_profile_ids:
        command.extend(["--port-profile-id", str(port_profile_id)])
    if policy is not None:
        command.extend(["--policy", policy])
    if balanced_usage_threshold_tokens is not None:
        command.extend(
            [
                "--balanced-usage-threshold-tokens",
                str(balanced_usage_threshold_tokens),
            ]
        )
    if skip_install:
        command.append("--skip-install")
    env = os.environ.copy()
    env[_BOOTSTRAP_ENV] = "1"
    os.execvpe(str(venv_python), command, env)


def _create_listen_socket(host: str, port: int) -> socket.socket:
    return socket.create_server((host, port), reuse_port=False)


def _unix_socket_accepting_connections(socket_path: Path) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.2)
        probe.connect(str(socket_path))
    except FileNotFoundError:
        return False
    except ConnectionRefusedError:
        return False
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ECONNREFUSED}:
            return False
        raise RuntimeError(f"failed to probe IPC socket path {socket_path}: {exc}") from exc
    else:
        return True
    finally:
        probe.close()


def _prepare_unix_socket_path(socket_path: Path) -> None:
    try:
        mode = socket_path.stat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(f"IPC socket path already exists and is not a socket: {socket_path}")
    if _unix_socket_accepting_connections(socket_path):
        raise RuntimeError(
            f"IPC socket path already exists and is active: {socket_path}. "
            "Stop the existing gateway or remove the socket and restart."
        )
    socket_path.unlink(missing_ok=True)


def _create_unix_listen_socket(
    socket_path: Path,
    *,
    permissions: int,
    uid: int | None,
    gid: int | None,
) -> socket.socket:
    resolved_socket_path = _resolve_path(socket_path)
    resolved_socket_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_unix_socket_path(resolved_socket_path)

    listen_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listen_socket.bind(str(resolved_socket_path))
        os.chmod(resolved_socket_path, permissions)
        if uid is not None or gid is not None:
            os.chown(
                resolved_socket_path,
                uid if uid is not None else -1,
                gid if gid is not None else -1,
            )
        listen_socket.set_inheritable(True)
        return listen_socket
    except Exception:
        listen_socket.close()
        if resolved_socket_path.exists():
            resolved_socket_path.unlink()
        raise


def _default_ipc_socket_path(profile_id: int | str | None) -> Path:
    resolved_profile_id = "0" if profile_id is None else str(profile_id)
    return DEFAULT_IPC_SOCKET_DIR / f"vllm-gateway-profile-{resolved_profile_id}.sock"


def _resolve_ipc_socket_path(
    *,
    ipc_enabled: bool,
    configured_socket_path_template: str | None,
    profile_id: int | str | None,
) -> Path | None:
    if not ipc_enabled:
        return None
    if configured_socket_path_template:
        try:
            rendered = configured_socket_path_template.format(profile_id=str(profile_id))
        except KeyError as exc:
            raise ValueError(
                "ipc.socket_path_template may only reference {profile_id}"
            ) from exc
        return _resolve_path(Path(rendered))
    return _resolve_path(_default_ipc_socket_path(profile_id))


async def _serve_app(
    app_instance: object,
    host: str,
    ports: list[int],
    *,
    ipc_socket_path: Path | None = None,
    ipc_socket_permissions: int = 0o660,
    ipc_socket_uid: int | None = None,
    ipc_socket_gid: int | None = None,
) -> None:
    sockets = [_create_listen_socket(host, port) for port in ports]
    cleanup_socket_paths: list[Path] = []
    if ipc_socket_path is not None:
        resolved_ipc_socket_path = _resolve_path(ipc_socket_path)
        sockets.append(
            _create_unix_listen_socket(
                resolved_ipc_socket_path,
                permissions=ipc_socket_permissions,
                uid=ipc_socket_uid,
                gid=ipc_socket_gid,
            )
        )
        cleanup_socket_paths.append(resolved_ipc_socket_path)
    if not sockets:
        raise ValueError("at least one TCP or IPC socket is required")

    config = uvicorn.Config(app_instance, host=host, port=ports[0] if ports else 0)
    server = uvicorn.Server(config)
    try:
        await server.serve(sockets=sockets)
    finally:
        for listen_socket in sockets:
            listen_socket.close()
        for socket_path in cleanup_socket_paths:
            if socket_path.exists():
                socket_path.unlink()


async def _serve_all(
    *,
    control_app: object,
    host: str,
    control_ports: list[int],
    ipc_apps: list[tuple[object, Path]],
    ipc_socket_permissions: int,
    ipc_socket_uid: int | None,
    ipc_socket_gid: int | None,
) -> None:
    async with asyncio.TaskGroup() as task_group:
        task_group.create_task(
            _serve_app(
                control_app,
                host,
                control_ports,
            )
        )
        for ipc_app, ipc_socket_path in ipc_apps:
            task_group.create_task(
                _serve_app(
                    ipc_app,
                    host,
                    [],
                    ipc_socket_path=ipc_socket_path,
                    ipc_socket_permissions=ipc_socket_permissions,
                    ipc_socket_uid=ipc_socket_uid,
                    ipc_socket_gid=ipc_socket_gid,
                )
            )


def _run_gateway(
    config_path: Path,
    host: str,
    port_profile_ids_override: Sequence[int],
    policy_override: str | None,
    balanced_usage_threshold_tokens_override: int | None,
) -> None:
    runtime_settings = load_runtime_settings(config_path)
    if port_profile_ids_override:
        runtime_settings = replace(
            runtime_settings,
            port_profile_ids=tuple(str(profile_id) for profile_id in port_profile_ids_override),
        )
    if policy_override is not None:
        runtime_settings = replace(
            runtime_settings,
            assignment_policy=normalize_assignment_policy(policy_override),
        )
    if balanced_usage_threshold_tokens_override is not None:
        runtime_settings = replace(
            runtime_settings,
            balanced_usage_threshold_tokens=normalize_balanced_usage_threshold_tokens(
                balanced_usage_threshold_tokens_override
            ),
        )

    profiles = [
        load_port_profile(profile_id)
        for profile_id in runtime_settings.port_profile_ids
    ]
    model_registry = load_model_registry()
    service = create_gateway_service(
        runtime_settings=runtime_settings,
        profiles=profiles,
    )

    control_profile = profiles[0]
    control_app = create_control_app(
        service,
        gateway_parse_port=control_profile.gateway_parse_port,
        model_registry=model_registry,
    )

    ipc_apps: list[tuple[object, Path]] = []
    seen_socket_paths: set[Path] = set()
    for backend in service.backends:
        ipc_socket_path = _resolve_ipc_socket_path(
            ipc_enabled=runtime_settings.ipc_enabled,
            configured_socket_path_template=runtime_settings.ipc_socket_path_template,
            profile_id=backend.profile.profile_id,
        )
        if ipc_socket_path is None:
            continue
        if ipc_socket_path in seen_socket_paths:
            raise ValueError(f"duplicate IPC socket path resolved: {ipc_socket_path}")
        seen_socket_paths.add(ipc_socket_path)
        ipc_apps.append((create_ipc_app(backend.service), ipc_socket_path))

    asyncio.run(
        _serve_all(
            control_app=control_app,
            host=host,
            control_ports=[control_profile.gateway_port, control_profile.gateway_parse_port],
            ipc_apps=ipc_apps,
            ipc_socket_permissions=runtime_settings.ipc_socket_permissions,
            ipc_socket_uid=runtime_settings.ipc_socket_uid,
            ipc_socket_gid=runtime_settings.ipc_socket_gid,
        )
    )


@app.command()
def start(
    config: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to gateway_multi TOML config.",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host interface to bind the control listener sockets.",
    ),
    venv_dir: Path = typer.Option(
        DEFAULT_VENV_DIR,
        "--venv-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Shared virtual environment used to bootstrap the gateway package.",
    ),
    port_profile_ids: list[int] | None = typer.Option(
        None,
        "--port-profile-id",
        help=(
            "Port profile numeric ID from configs/port_profiles.toml. "
            "Repeat to select multiple profiles. Overrides run.port_profile_ids "
            "in gateway_multi/config.toml."
        ),
    ),
    policy: str | None = typer.Option(
        None,
        "--policy",
        help=(
            "Agent assignment policy. Supported values: balanced, round_robin, "
            "lowest_usage, lowest_profile_without_pending."
        ),
    ),
    balanced_usage_threshold_tokens: int | None = typer.Option(
        None,
        "--balanced-usage-threshold-tokens",
        help=(
            "Context-token threshold used by the balanced assignment policy. "
            f"Defaults to run.balanced_usage_threshold_tokens or "
            f"{DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS}."
        ),
    ),
    skip_install: bool = typer.Option(
        False,
        "--skip-install",
        help="Skip editable install into the shared .venv before starting.",
    ),
) -> None:
    config_path = _resolve_path(config)
    venv_dir = _resolve_path(venv_dir)
    _ensure_config_exists(config_path)
    resolved_port_profile_ids = list(port_profile_ids or [])

    if os.environ.get(_BOOTSTRAP_ENV) != "1":
        venv_python = _venv_python(venv_dir)
        if not venv_python.exists():
            if skip_install:
                typer.secho(
                    f"error: shared virtual environment not found: {venv_dir}",
                    err=True,
                    fg=typer.colors.RED,
                )
                typer.secho(
                    "hint: rerun without --skip-install so the gateway can bootstrap .venv",
                    err=True,
                    fg=typer.colors.YELLOW,
                )
                raise typer.Exit(code=1)
            _create_venv(venv_dir)
        if not skip_install:
            _install_gateway(venv_python)
        _reexec_in_venv(
            venv_python=venv_python,
            config_path=config_path,
            host=host,
            venv_dir=venv_dir,
            port_profile_ids=resolved_port_profile_ids,
            policy=policy,
            balanced_usage_threshold_tokens=balanced_usage_threshold_tokens,
            skip_install=True,
        )
        raise typer.Exit()

    _run_gateway(
        config_path=config_path,
        host=host,
        port_profile_ids_override=resolved_port_profile_ids,
        policy_override=policy,
        balanced_usage_threshold_tokens_override=balanced_usage_threshold_tokens,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
