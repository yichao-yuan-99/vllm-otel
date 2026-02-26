#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Typer CLI frontend for the Apptainer HPC control server."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import typer

try:
    from .client_d import (
        DEFAULT_JAEGER_OTLP_PORT,
        DEFAULT_JAEGER_UI_PORT,
        DEFAULT_RUNTIME_DIR as CLIENT_D_RUNTIME_DIR,
        DEFAULT_SERVER_PORT,
        DEFAULT_VLLM_PORT,
        client_d_status,
        start_client_d,
        stop_client_d,
    )
except ImportError:  # pragma: no cover
    from client_d import (  # type: ignore[no-redef]
        DEFAULT_JAEGER_OTLP_PORT,
        DEFAULT_JAEGER_UI_PORT,
        DEFAULT_RUNTIME_DIR as CLIENT_D_RUNTIME_DIR,
        DEFAULT_SERVER_PORT,
        DEFAULT_VLLM_PORT,
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
    runtime_dir: Path = typer.Option(
        CLIENT_D_RUNTIME_DIR,
        "--runtime-dir",
        help="Directory for client-d pid and log files.",
    ),
    server_port: int = typer.Option(DEFAULT_SERVER_PORT, "--server-port", help="Control server port."),
    vllm_port: int = typer.Option(DEFAULT_VLLM_PORT, "--vllm-port", help="vLLM service port."),
    jaeger_otlp_port: int = typer.Option(
        DEFAULT_JAEGER_OTLP_PORT,
        "--jaeger-otlp-port",
        help="Jaeger OTLP gRPC port.",
    ),
    jaeger_ui_port: int = typer.Option(
        DEFAULT_JAEGER_UI_PORT,
        "--jaeger-ui-port",
        help="Jaeger UI/API port.",
    ),
    ssh_option: list[str] = typer.Option(
        [],
        "--ssh-option",
        help="Additional raw ssh option token. Repeat to pass multiple tokens.",
    ),
) -> None:
    """Start client-d SSH tunnels."""
    payload = start_client_d(
        ssh_target=ssh_target,
        runtime_dir=runtime_dir,
        server_port=server_port,
        vllm_port=vllm_port,
        jaeger_otlp_port=jaeger_otlp_port,
        jaeger_ui_port=jaeger_ui_port,
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


@app.command()
def pull(ctx: typer.Context) -> None:
    """Pull Jaeger + vLLM OCI images into SIF files."""
    payload = _run_command(ctx, "/pull")
    _require_ok(payload)
    _print_json(payload)


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
        payload = _run_blocking_with_progress(
            ctx,
            endpoint="/start",
            payload={"partition": partition, "model": model, "block": True},
            progress_endpoint="/start/status",
            progress_tag="start",
            timeout_seconds=30.0 * 60.0,
        )
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
) -> None:
    """Block until both tunneled vLLM and Jaeger endpoints are up."""
    payload = _run_command_with_timeout(
        ctx,
        "/wait-up",
        payload={
            "timeout_seconds": timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
        },
        timeout_seconds=max(float(timeout_seconds) + 30.0, 120.0),
    )
    _require_ok(payload)
    _print_json(payload)


@app.command()
def test(ctx: typer.Context) -> None:
    """Run OTEL + force-sequence smoke tests against the active job."""
    result: dict[str, Any] = {}

    def run_test() -> None:
        result["payload"] = _run_command(ctx, "/test")

    worker = threading.Thread(target=run_test, daemon=True)
    worker.start()

    last_line: str | None = None
    while worker.is_alive():
        progress_payload = _run_command(ctx, "/test/status")
        data = progress_payload.get("data")
        if isinstance(data, dict):
            status = data.get("status")
            phase = data.get("phase")
            message = data.get("message")
            updated_at = data.get("updated_at")
            line = f"[test] status={status} phase={phase} updated_at={updated_at} message={message}"
            if line != last_line:
                typer.echo(line)
                last_line = line
        worker.join(timeout=2.0)

    payload = result.get("payload")
    if not isinstance(payload, dict):
        typer.echo("error: test command did not return a valid response", err=True)
        raise typer.Exit(code=1)

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
