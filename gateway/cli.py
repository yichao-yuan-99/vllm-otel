from __future__ import annotations

import asyncio
import errno
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

import typer
import uvicorn

from gateway.app import GatewayConfig, create_app, create_gateway_service
from gateway.model_configs import ModelRegistry, load_model_registry
from gateway.port_profiles import PortProfile, load_port_profile
from gateway.runtime_config import DEFAULT_CONFIG_PATH, load_runtime_settings


app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_VENV_DIR = REPO_ROOT / ".venv"
_BOOTSTRAP_ENV = "VLLM_OTEL_GATEWAY_BOOTSTRAPPED"
DEFAULT_IPC_SOCKET_DIR = Path("/tmp")


@app.callback()
def gateway_app() -> None:
    """Gateway CLI."""


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
        "hint: copy gateway/config.example.toml to gateway/config.toml and edit as needed",
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
    port_profile_id: int | None,
    skip_install: bool,
) -> None:
    command = [
        str(venv_python),
        "-m",
        "gateway",
        "start",
        "--config",
        str(config_path),
        "--host",
        host,
        "--venv-dir",
        str(venv_dir),
    ]
    if port_profile_id is not None:
        command.extend(["--port-profile-id", str(port_profile_id)])
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
    configured_socket_path: str | None,
    profile_id: int | str | None,
) -> Path | None:
    if not ipc_enabled:
        return None
    if configured_socket_path:
        return _resolve_path(Path(configured_socket_path))
    return _resolve_path(_default_ipc_socket_path(profile_id))


def resolve_listener_ports(
    *,
    gateway_config: GatewayConfig,
    profile: PortProfile,
    model_registry: ModelRegistry,
    served_model_name: str | None = None,
) -> tuple[int, int]:
    # Always expose both configured gateway listeners. When the served model has
    # no reasoning parser, the parse-port listener remains a no-op view of the
    # same upstream response.
    _ = gateway_config
    _ = model_registry
    _ = served_model_name
    return profile.gateway_port, profile.gateway_parse_port


async def _serve_gateway(
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
    config = uvicorn.Config(app_instance, host=host, port=ports[0])
    server = uvicorn.Server(config)
    try:
        await server.serve(sockets=sockets)
    finally:
        for listen_socket in sockets:
            listen_socket.close()
        for socket_path in cleanup_socket_paths:
            if socket_path.exists():
                socket_path.unlink()


def _run_gateway(config_path: Path, host: str, port_profile_id: int | None) -> None:
    runtime_settings = load_runtime_settings(config_path)
    selected_profile_id = (
        port_profile_id if port_profile_id is not None else runtime_settings.port_profile_id
    )
    profile = load_port_profile(selected_profile_id)
    config = GatewayConfig.from_port_profile(
        profile.profile_id,
        runtime_settings=runtime_settings,
    )
    model_registry = load_model_registry()
    gateway_port, gateway_parse_port = resolve_listener_ports(
        gateway_config=config,
        profile=profile,
        model_registry=model_registry,
    )
    app_instance = create_app(
        service=create_gateway_service(config=config),
        gateway_parse_port=gateway_parse_port,
        model_registry=model_registry,
    )
    ipc_socket_path = _resolve_ipc_socket_path(
        ipc_enabled=runtime_settings.ipc_enabled,
        configured_socket_path=runtime_settings.ipc_socket_path,
        profile_id=profile.profile_id,
    )
    typer.echo(
        "gateway listeners: "
        f"raw=http://{host}:{gateway_port} "
        f"parsed=http://{host}:{gateway_parse_port}"
    )
    if ipc_socket_path is not None:
        typer.echo(
            "gateway ipc listener: "
            f"unix://{ipc_socket_path}"
        )
    ports = [gateway_port]
    if gateway_parse_port != gateway_port:
        ports.append(gateway_parse_port)
    asyncio.run(
        _serve_gateway(
            app_instance,
            host,
            ports,
            ipc_socket_path=ipc_socket_path,
            ipc_socket_permissions=runtime_settings.ipc_socket_permissions,
            ipc_socket_uid=runtime_settings.ipc_socket_uid,
            ipc_socket_gid=runtime_settings.ipc_socket_gid,
        )
    )


@app.command()
def start(
    config: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        exists=False,
        dir_okay=False,
        file_okay=True,
        help="Path to the gateway TOML config file.",
    ),
    port_profile_id: int | None = typer.Option(
        None,
        help="Port profile numeric ID from configs/port_profiles.toml. Overrides run.port_profile_id in gateway/config.toml.",
    ),
    host: str = typer.Option("0.0.0.0", help="Host address for the gateway listener."),
    venv_dir: Path = typer.Option(
        DEFAULT_VENV_DIR,
        exists=False,
        file_okay=False,
        dir_okay=True,
        help="Shared virtual environment used to run the gateway.",
    ),
    skip_install: bool = typer.Option(
        False,
        help="Skip installing or updating the gateway package inside the shared virtual environment.",
    ),
) -> None:
    """Start the gateway, bootstrapping the shared .venv when needed."""
    config_path = _resolve_path(config)
    resolved_venv_dir = _resolve_path(venv_dir)
    _ensure_config_exists(config_path)

    venv_python = _venv_python(resolved_venv_dir)
    current_python = Path(sys.executable).resolve()
    in_bootstrapped_process = os.environ.get(_BOOTSTRAP_ENV) == "1"

    if current_python != venv_python.resolve() and not in_bootstrapped_process:
        if not venv_python.exists():
            if skip_install:
                typer.secho(
                    f"error: shared virtual environment not found: {resolved_venv_dir}",
                    err=True,
                    fg=typer.colors.RED,
                )
                typer.secho(
                    "hint: rerun without --skip-install so the gateway can bootstrap .venv",
                    err=True,
                    fg=typer.colors.YELLOW,
                )
                raise typer.Exit(code=1)
            _create_venv(resolved_venv_dir)
        if not skip_install:
            _install_gateway(venv_python)
        _reexec_in_venv(
            venv_python=venv_python,
            config_path=config_path,
            host=host,
            venv_dir=resolved_venv_dir,
            port_profile_id=port_profile_id,
            skip_install=skip_install,
        )

    if current_python == venv_python.resolve() and not skip_install and not in_bootstrapped_process:
        _install_gateway(venv_python)

    _run_gateway(config_path, host, port_profile_id)


def main() -> None:
    app()
