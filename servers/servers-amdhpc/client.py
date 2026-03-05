#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Typer CLI frontend for the Apptainer HPC control server."""

from __future__ import annotations

import json
from pathlib import Path
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
        resolve_client_d_server_url,
        start_client_d,
        stop_client_d,
    )
except ImportError:  # pragma: no cover
    from client_d import (  # type: ignore[no-redef]
        DEFAULT_RUNTIME_DIR as CLIENT_D_RUNTIME_DIR,
        DEFAULT_SERVER_PORT,
        resolve_client_d_server_url,
        start_client_d,
        stop_client_d,
    )


app = typer.Typer(add_completion=False, no_args_is_help=True, help="vLLM HPC control client")

START_BLOCKING_TIMEOUT_SECONDS = 24.0 * 60.0 * 60.0
START_CLEANUP_WAIT_FOR_SUBMISSION_SECONDS = 120.0


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


def _server_clientd_payload(
    *,
    ok: bool,
    code: int,
    message: str,
    server_payload: dict[str, Any],
    clientd_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": ok,
        "code": code,
        "message": message,
        "data": {
            "server": server_payload.get("data"),
            "clientd": clientd_payload.get("data"),
        },
    }


def _clientd_cleanup_only_payload(
    *,
    ok: bool,
    code: int,
    message: str,
    clientd_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": ok,
        "code": code,
        "message": message,
        "data": {"clientd_cleanup": clientd_payload.get("data")},
    }


def _start_result_payload(
    *,
    server_payload: dict[str, Any],
    clientd_payload: dict[str, Any],
) -> dict[str, Any]:
    return _server_clientd_payload(
        ok=True,
        code=0,
        message=str(server_payload.get("message", "start complete")),
        server_payload=server_payload,
        clientd_payload=clientd_payload,
    )


def _start_failure_payload(
    *,
    server_payload: dict[str, Any],
    clientd_start_payload: dict[str, Any],
    clientd_cleanup_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "code": int(server_payload.get("code", 1)),
        "message": str(server_payload.get("message", "start failed")),
        "data": {
            "clientd_start": clientd_start_payload.get("data"),
            "server_start": server_payload.get("data"),
            "clientd_cleanup": None if clientd_cleanup_payload is None else clientd_cleanup_payload.get("data"),
        },
    }


def _stop_result_payload(
    *,
    server_payload: dict[str, Any],
    clientd_payload: dict[str, Any],
) -> dict[str, Any]:
    server_code = int(server_payload.get("code", 1))
    server_ok = _is_ok(server_payload)
    if server_ok:
        message = str(server_payload.get("message", "stopped"))
        ok = True
        code = 0
    elif server_code == 21:
        message = "no active job; client-d stopped"
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
) -> None:
    clientd_payload = _stop_clientd_for_profile(ctx, port_profile=port_profile)
    _emit_cleanup_result(
        note,
        _clientd_cleanup_only_payload(
            ok=ok,
            code=code,
            message=message,
            clientd_payload=clientd_payload,
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
) -> None:
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
        ),
    )


def _cleanup_after_start_interrupt(ctx: typer.Context, *, port_profile: int) -> None:
    typer.echo(
        "[start] interrupt received; entering cleanup and attempting blocking stop.",
        err=True,
    )

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
        )
        return

    if _is_ok(stop_payload) or stop_payload.get("code") != 21:
        _emit_cleanup_stop_result(
            ctx,
            port_profile=port_profile,
            server_payload=stop_payload,
            success_note="[start] cleanup complete: active job canceled.",
            error_note="[start] cleanup stop returned an error:",
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
                    )
                    return
                _emit_cleanup_stop_result(
                    ctx,
                    port_profile=port_profile,
                    server_payload=retry_payload,
                    success_note="[start] cleanup complete: active job canceled.",
                    no_job_note="[start] no active job found on retry; cleanup done.",
                    error_note="[start] cleanup stop returned an error on retry:",
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
    )


@app.command()
def start(
    ctx: typer.Context,
    ssh_target: str = typer.Option(
        "amd-hpc",
        "--ssh-target",
        "-t",
        help="SSH target, same as in `ssh <target>`.",
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
) -> None:
    """Start client-d and submit an sbatch job for a configured partition + model."""
    clientd_payload = start_client_d(
        ssh_target=ssh_target,
        port_profile_id=port_profile,
        runtime_dir=_clientd_runtime_dir(ctx),
        remote_server_port=server_port,
        ssh_options=ssh_option or None,
    )
    _require_ok(clientd_payload)

    if block:
        try:
            server_payload = _run_blocking_with_progress(
                ctx,
                endpoint="/start",
                payload={"port_profile": port_profile, "partition": partition, "model": model, "block": True},
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
            payload={"port_profile": port_profile, "partition": partition, "model": model, "block": False},
            timeout_seconds=None,
        )
    if not _is_ok(server_payload):
        cleanup_payload = _stop_clientd_for_profile(ctx, port_profile=port_profile)
        _print_json(
            _start_failure_payload(
                server_payload=server_payload,
                clientd_start_payload=clientd_payload,
                clientd_cleanup_payload=cleanup_payload,
            )
        )
        raise typer.Exit(code=1)
    _print_json(_start_result_payload(server_payload=server_payload, clientd_payload=clientd_payload))


@app.command()
def stop(
    ctx: typer.Context,
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
    """Stop the active sbatch job, wait for completion, then stop client-d."""
    server_payload = _run_blocking_with_progress(
        ctx,
        endpoint="/stop",
        payload={"port_profile": port_profile, "block": True},
        progress_endpoint="/stop/status",
        progress_payload={"port_profile": port_profile},
        progress_tag="stop",
        timeout_seconds=10.0 * 60.0,
    )
    clientd_payload = _stop_clientd_for_profile(
        ctx,
        port_profile=port_profile,
        timeout_seconds=clientd_timeout_seconds,
    )
    stop_payload = _stop_result_payload(server_payload=server_payload, clientd_payload=clientd_payload)
    if not _is_ok(stop_payload):
        _print_json(stop_payload)
        raise typer.Exit(code=1)
    _print_json(stop_payload)


@app.command()
def logs(
    ctx: typer.Context,
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    lines: int = typer.Option(200, "--lines", "-n", min=1, help="Lines per log file."),
) -> None:
    """Fetch current tail of Slurm/Jaeger/vLLM logs."""
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
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
) -> None:
    """Check whether both tunneled vLLM and Jaeger endpoints are up."""
    payload = _run_command(ctx, "/up", {"port_profile": port_profile})
    _require_ok(payload)
    _print_json(payload)


@app.command(name="wait-up")
def wait_up(
    ctx: typer.Context,
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


@app.command()
def status(
    ctx: typer.Context,
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-P",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
) -> None:
    """Show control-plane status and configured partitions/models."""
    payload = _run_command(ctx, "/status", {"port_profile": port_profile})
    _require_ok(payload)
    _print_json(payload)


if __name__ == "__main__":
    app()
