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
        client_d_status,
        start_client_d,
        stop_client_d,
    )
except ImportError:  # pragma: no cover
    from client_d import (  # type: ignore[no-redef]
        DEFAULT_RUNTIME_DIR as CLIENT_D_RUNTIME_DIR,
        DEFAULT_SERVER_PORT,
        client_d_status,
        start_client_d,
        stop_client_d,
    )


app = typer.Typer(add_completion=False, no_args_is_help=True, help="vLLM HPC control client")


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


@app.callback()
def main(
    ctx: typer.Context,
    server_url: str = typer.Option(
        "http://127.0.0.1:23971",
        "--server-url",
        help="Control server base URL.",
    ),
    timeout_seconds: float = typer.Option(
        120.0,
        "--timeout-seconds",
        help="HTTP timeout for each command.",
    ),
) -> None:
    ctx.obj = {
        "server_url": server_url,
        "timeout_seconds": timeout_seconds,
    }


@app.command(name="clientd-start")
def clientd_start(
    ssh_target: str = typer.Option(..., "--ssh-target", "-t", help="SSH target, same as in `ssh <target>`."),
    port_profile: int = typer.Option(
        ...,
        "--port-profile",
        "-p",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    runtime_dir: Path = typer.Option(
        CLIENT_D_RUNTIME_DIR,
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
    """Start client-d SSH tunnels."""
    payload = start_client_d(
        ssh_target=ssh_target,
        port_profile_id=port_profile,
        runtime_dir=runtime_dir,
        server_port=server_port,
        ssh_options=ssh_option or None,
    )
    _require_ok(payload)
    _print_json(payload)


@app.command(name="clientd-stop")
def clientd_stop(
    runtime_dir: Path = typer.Option(
        CLIENT_D_RUNTIME_DIR,
        "--runtime-dir",
        help="Directory for client-d pid and log files.",
    ),
    timeout_seconds: float = typer.Option(
        10.0,
        "--timeout-seconds",
        help="Seconds to wait before forcing kill.",
    ),
) -> None:
    """Stop client-d SSH tunnels."""
    payload = stop_client_d(runtime_dir=runtime_dir, timeout_seconds=timeout_seconds)
    _require_ok(payload)
    _print_json(payload)


@app.command(name="clientd-status")
def clientd_status(
    runtime_dir: Path = typer.Option(
        CLIENT_D_RUNTIME_DIR,
        "--runtime-dir",
        help="Directory for client-d pid and log files.",
    ),
) -> None:
    """Show client-d status."""
    payload = client_d_status(runtime_dir=runtime_dir)
    _require_ok(payload)
    _print_json(payload)


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
    server_url = str(obj.get("server_url", "http://127.0.0.1:23971"))
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
        progress_payload = _run_command(ctx, progress_endpoint)
        data = progress_payload.get("data")
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


def _cleanup_after_start_interrupt(ctx: typer.Context) -> None:
    typer.echo(
        "[start] interrupt received; entering cleanup and attempting blocking stop.",
        err=True,
    )

    def run_blocking_stop() -> dict[str, Any]:
        return _run_blocking_with_progress(
            ctx,
            endpoint="/stop",
            payload={"block": True},
            progress_endpoint="/stop/status",
            progress_tag="stop",
            timeout_seconds=24.0 * 60.0 * 60.0,
        )

    try:
        stop_payload = run_blocking_stop()
    except KeyboardInterrupt:
        typer.echo(
            "[start] cleanup interrupted again; run `python3 servers/servers-amdhpc/client.py stop -b`.",
            err=True,
        )
        return

    if stop_payload.get("ok"):
        typer.echo("[start] cleanup complete: active job canceled.", err=True)
        _print_json(stop_payload)
        return

    if stop_payload.get("code") != 21:
        typer.echo("[start] cleanup stop returned an error:", err=True)
        _print_json(stop_payload)
        return

    typer.echo(
        "[start] no active job yet; waiting briefly for in-flight submission before retrying stop.",
        err=True,
    )
    deadline = time.monotonic() + 120.0
    last_line: str | None = None

    while time.monotonic() < deadline:
        progress_payload = _run_command(ctx, "/start/status")
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
                typer.echo("[start] start failed before submission; nothing to cancel.", err=True)
                return

            if isinstance(job_id, str) and job_id:
                typer.echo(
                    f"[start] detected submitted job {job_id}; retrying blocking stop.",
                    err=True,
                )
                try:
                    retry_payload = run_blocking_stop()
                except KeyboardInterrupt:
                    typer.echo(
                        "[start] cleanup interrupted again; run `python3 servers/servers-amdhpc/client.py stop -b`.",
                        err=True,
                    )
                    return
                if retry_payload.get("ok"):
                    typer.echo("[start] cleanup complete: active job canceled.", err=True)
                    _print_json(retry_payload)
                elif retry_payload.get("code") == 21:
                    typer.echo("[start] no active job found on retry; cleanup done.", err=True)
                else:
                    typer.echo("[start] cleanup stop returned an error on retry:", err=True)
                    _print_json(retry_payload)
                return

        time.sleep(1.0)

    typer.echo(
        "[start] cleanup timed out waiting for submission; run `python3 servers/servers-amdhpc/client.py stop -b` if needed.",
        err=True,
    )


@app.command()
def start(
    ctx: typer.Context,
    partition: str = typer.Option(..., "--partition", "-p", help="Configured partition key."),
    model: str = typer.Option(..., "--model", "-m", help="Configured model key."),
    block: bool = typer.Option(
        False,
        "--block",
        "-b",
        help="Block until services are fully up.",
    ),
) -> None:
    """Submit an sbatch job for a configured partition + model."""
    if block:
        try:
            payload = _run_blocking_with_progress(
                ctx,
                endpoint="/start",
                payload={"partition": partition, "model": model, "block": True},
                progress_endpoint="/start/status",
                progress_tag="start",
                timeout_seconds=24.0 * 60.0 * 60.0,
            )
        except KeyboardInterrupt:
            _cleanup_after_start_interrupt(ctx)
            raise typer.Exit(code=130)
    else:
        payload = _run_command_with_timeout(
            ctx,
            "/start",
            payload={"partition": partition, "model": model, "block": False},
            timeout_seconds=None,
        )
    _require_ok(payload)
    _print_json(payload)


@app.command()
def stop(
    ctx: typer.Context,
    block: bool = typer.Option(
        False,
        "--block",
        "-b",
        help="Block until the job disappears from Slurm.",
    ),
) -> None:
    """Stop the currently active sbatch job."""
    if block:
        payload = _run_blocking_with_progress(
            ctx,
            endpoint="/stop",
            payload={"block": True},
            progress_endpoint="/stop/status",
            progress_tag="stop",
            timeout_seconds=10.0 * 60.0,
        )
    else:
        payload = _run_command_with_timeout(
            ctx,
            "/stop",
            payload={"block": False},
            timeout_seconds=None,
        )
    _require_ok(payload)
    _print_json(payload)


@app.command()
def stop_poll(ctx: typer.Context) -> None:
    """Check whether a previous non-block stop has fully finished."""
    payload = _run_command(ctx, "/stop/poll")
    _require_ok(payload)
    _print_json(payload)


@app.command()
def logs(
    ctx: typer.Context,
    lines: int = typer.Option(200, "--lines", "-n", min=1, help="Lines per log file."),
) -> None:
    """Fetch current tail of Slurm/Jaeger/vLLM logs."""
    payload = _run_command(ctx, "/logs", {"lines": lines})
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
def up(ctx: typer.Context) -> None:
    """Check whether both tunneled vLLM and Jaeger endpoints are up."""
    payload = _run_command(ctx, "/up")
    _require_ok(payload)
    _print_json(payload)


@app.command(name="wait-up")
def wait_up(
    ctx: typer.Context,
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
            "timeout_seconds": timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "defer_timeout_until_running": defer_timeout_until_running,
        },
        timeout_seconds=request_timeout_seconds,
    )
    _require_ok(payload)
    _print_json(payload)


@app.command()
def status(ctx: typer.Context) -> None:
    """Show control-plane status and configured partitions/models."""
    payload = _run_command(ctx, "/status")
    _require_ok(payload)
    _print_json(payload)


if __name__ == "__main__":
    app()
