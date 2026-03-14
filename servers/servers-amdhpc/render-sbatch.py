#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render sbatch scripts for start/start-group without submitting jobs."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import typer

try:
    from .client_d import DEFAULT_SERVER_PORT
    from .control_plane import ControlPlane, ControlPlaneError, command_result_payload, error_payload
except ImportError:  # pragma: no cover
    from client_d import DEFAULT_SERVER_PORT  # type: ignore[no-redef]
    from control_plane import (  # type: ignore[no-redef]
        ControlPlane,
        ControlPlaneError,
        command_result_payload,
        error_payload,
    )


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "server_config.toml"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Render sbatch scripts using the same interface as client start/start-group, "
        "without starting client-d, gateway, or submitting jobs."
    ),
)


def _print_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


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


def _check_port_availability_from_ctx(ctx: typer.Context) -> bool:
    obj = ctx.obj or {}
    return bool(obj.get("check_port_availability", False))


def _resolve_local_mode_script(value: Path | None) -> str | None:
    if value is None:
        return None
    script_path = value.expanduser().resolve()
    if not script_path.exists():
        raise ValueError(f"local-mode script not found: {script_path}")
    if not script_path.is_file():
        raise ValueError(f"local-mode script must be a file: {script_path}")
    return str(script_path)


def _control_plane_from_ctx(ctx: typer.Context) -> ControlPlane:
    obj = ctx.obj or {}
    cached = obj.get("control_plane")
    if isinstance(cached, ControlPlane):
        return cached

    config_raw = obj.get("config", DEFAULT_CONFIG_PATH)
    config_path = Path(config_raw).expanduser().resolve()
    control_plane = ControlPlane(config_path, archive_previous_artifacts=False)
    obj["control_plane"] = control_plane
    ctx.obj = obj
    return control_plane


def _emit_control_plane_error(exc: ControlPlaneError) -> None:
    _print_json(error_payload(exc))
    raise typer.Exit(code=1)


@app.callback()
def main(
    ctx: typer.Context,
    config: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        help=f"Path to server config TOML (default: {DEFAULT_CONFIG_PATH})",
    ),
    check_port_availability: bool = typer.Option(
        False,
        "--check-port-availability",
        help=(
            "Validate selected profile ports are currently free on the login node. "
            "Disabled by default for offline rendering."
        ),
    ),
) -> None:
    ctx.obj = {
        "config": config,
        "check_port_availability": check_port_availability,
    }


@app.command()
def start(
    ctx: typer.Context,
    ssh_target: str = typer.Option(
        "amd-hpc",
        "--ssh-target",
        "-t",
        help="Compatibility flag from client start; ignored by render-sbatch.",
    ),
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
        help="Compatibility flag from client start; ignored by render-sbatch.",
    ),
    server_port: int = typer.Option(
        DEFAULT_SERVER_PORT,
        "--server-port",
        help="Compatibility flag from client start; ignored by render-sbatch.",
    ),
    ssh_option: list[str] = typer.Option(
        [],
        "--ssh-option",
        help="Compatibility flag from client start; ignored by render-sbatch.",
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
    local_mode: Path | None = typer.Option(
        None,
        "--local-mode",
        help=(
            "Path to a script to execute directly on the allocated node after vLLM/Jaeger "
            "and gateway are ready. Disables reverse-tunnel assumptions in rendered sbatch."
        ),
    ),
) -> None:
    del ssh_target, block, server_port, ssh_option
    try:
        extra_env = _parse_extra_env_list(env)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--env") from exc
    try:
        local_mode_script = _resolve_local_mode_script(local_mode)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--local-mode") from exc
    if lmcache is not None and lmcache <= 0:
        raise typer.BadParameter("--lmcache must be a positive integer", param_hint="--lmcache")
    if lmcache is not None and "LMCACHE_MAX_LOCAL_CPU_SIZE" in extra_env:
        raise typer.BadParameter(
            "cannot combine --lmcache with --env LMCACHE_MAX_LOCAL_CPU_SIZE=...",
            param_hint="--lmcache",
        )

    try:
        control_plane = _control_plane_from_ctx(ctx)
        result = control_plane.render_start_sbatch(
            port_profile_id=port_profile,
            partition=partition,
            model=model,
            extra_env=extra_env,
            lmcache_max_local_cpu_size=str(lmcache) if lmcache is not None else None,
            local_mode_script=local_mode_script,
            check_port_availability=_check_port_availability_from_ctx(ctx),
        )
    except ControlPlaneError as exc:
        _emit_control_plane_error(exc)

    _print_json(command_result_payload(result))


@app.command(name="start-group")
def start_group(
    ctx: typer.Context,
    ssh_target: str = typer.Option(
        "amd-hpc",
        "--ssh-target",
        "-t",
        help="Compatibility flag from client start-group; ignored by render-sbatch.",
    ),
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
        help="Compatibility flag from client start-group; ignored by render-sbatch.",
    ),
    ssh_option: list[str] = typer.Option(
        [],
        "--ssh-option",
        help="Compatibility flag from client start-group; ignored by render-sbatch.",
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
        help="Compatibility flag from client start-group; ignored by render-sbatch.",
    ),
    local_mode: Path | None = typer.Option(
        None,
        "--local-mode",
        help=(
            "Reserved for single-profile local execution mode. "
            "Grouped local-mode rendering is not supported yet."
        ),
    ),
) -> None:
    del ssh_target, server_port, ssh_option, clientd_timeout_seconds
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
    if local_mode is not None:
        raise typer.BadParameter(
            "--local-mode is only supported with `render-sbatch.py start` right now",
            param_hint="--local-mode",
        )

    try:
        control_plane = _control_plane_from_ctx(ctx)
        result = control_plane.render_start_group_sbatch(
            group_name=group_name,
            port_profile_ids=profile_ids,
            partition=partition,
            model=model,
            extra_env=extra_env,
            lmcache_max_local_cpu_size=str(lmcache) if lmcache is not None else None,
            check_port_availability=_check_port_availability_from_ctx(ctx),
        )
    except ControlPlaneError as exc:
        _emit_control_plane_error(exc)

    _print_json(command_result_payload(result))


if __name__ == "__main__":
    app()
