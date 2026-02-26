#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HTTP control server for Slurm + Apptainer vLLM orchestration."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
from typing import Any

try:
    from .control_plane import (
        CommandResult,
        ControlPlane,
        ControlPlaneError,
        command_result_payload,
        error_payload,
    )
except ImportError:  # pragma: no cover
    from control_plane import (  # type: ignore[no-redef]
        CommandResult,
        ControlPlane,
        ControlPlaneError,
        command_result_payload,
        error_payload,
    )


def _parse_request_payload(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length_raw = handler.headers.get("Content-Length", "0")
    try:
        length = int(length_raw)
    except ValueError as exc:  # pragma: no cover - malformed HTTP length
        raise ControlPlaneError(message="invalid Content-Length", code=200, http_status=400) from exc

    if length == 0:
        return {}

    raw = handler.rfile.read(length)
    if not raw:
        return {}

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ControlPlaneError(message="request body must be valid JSON", code=201, http_status=400) from exc

    if not isinstance(payload, dict):
        raise ControlPlaneError(
            message="request JSON must be an object",
            code=202,
            http_status=400,
        )
    return payload


class RequestHandler(BaseHTTPRequestHandler):
    control_plane: ControlPlane

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._write_json(200, {"ok": True, "message": "healthy"})
            return
        if path == "/up":
            self._handle_with_result(lambda: self.control_plane.up())
            return
        if path == "/stop/poll":
            self._handle_with_result(lambda: self.control_plane.stop_poll())
            return
        if path == "/start/status":
            self._handle_with_result(lambda: self.control_plane.start_status())
            return
        if path == "/stop/status":
            self._handle_with_result(lambda: self.control_plane.stop_status())
            return
        if path == "/status":
            self._handle_with_result(lambda: self.control_plane.status())
            return
        if path == "/test/status":
            self._handle_with_result(lambda: self.control_plane.test_status())
            return

        self._write_json(
            404,
            {
                "ok": False,
                "code": 404,
                "message": f"unknown endpoint: {path}",
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]

        try:
            payload = _parse_request_payload(self)
        except ControlPlaneError as exc:
            self._write_json(exc.http_status, error_payload(exc))
            return

        routes = {
            "/pull": lambda: self.control_plane.pull(),
            "/start": lambda: self._start_command(payload),
            "/stop": lambda: self._stop_command(payload),
            "/stop/poll": lambda: self.control_plane.stop_poll(),
            "/start/status": lambda: self.control_plane.start_status(),
            "/stop/status": lambda: self.control_plane.stop_status(),
            "/logs": lambda: self._logs_command(payload),
            "/up": lambda: self.control_plane.up(),
            "/wait-up": lambda: self._wait_up_command(payload),
            "/test": lambda: self.control_plane.test(),
            "/test/status": lambda: self.control_plane.test_status(),
            "/status": lambda: self.control_plane.status(),
            "/command/pull": lambda: self.control_plane.pull(),
            "/command/start": lambda: self._start_command(payload),
            "/command/stop": lambda: self._stop_command(payload),
            "/command/stop/poll": lambda: self.control_plane.stop_poll(),
            "/command/start/status": lambda: self.control_plane.start_status(),
            "/command/stop/status": lambda: self.control_plane.stop_status(),
            "/command/logs": lambda: self._logs_command(payload),
            "/command/up": lambda: self.control_plane.up(),
            "/command/wait-up": lambda: self._wait_up_command(payload),
            "/command/test": lambda: self.control_plane.test(),
            "/command/test/status": lambda: self.control_plane.test_status(),
            "/command/status": lambda: self.control_plane.status(),
        }

        action = routes.get(path)
        if action is None:
            self._write_json(
                404,
                {
                    "ok": False,
                    "code": 404,
                    "message": f"unknown endpoint: {path}",
                },
            )
            return

        self._handle_with_result(action)

    def _start_command(self, payload: dict[str, Any]) -> CommandResult:
        partition = payload.get("partition")
        model = payload.get("model")
        block = payload.get("block", False)
        if not isinstance(partition, str) or not partition:
            raise ControlPlaneError(
                message="start requires 'partition'",
                code=203,
                http_status=400,
            )
        if not isinstance(model, str) or not model:
            raise ControlPlaneError(
                message="start requires 'model'",
                code=204,
                http_status=400,
            )
        if not isinstance(block, bool):
            raise ControlPlaneError(
                message="start.block must be a boolean",
                code=209,
                http_status=400,
            )
        return self.control_plane.start(partition=partition, model=model, block=block)

    def _stop_command(self, payload: dict[str, Any]) -> CommandResult:
        block = payload.get("block", False)
        if not isinstance(block, bool):
            raise ControlPlaneError(
                message="stop.block must be a boolean",
                code=210,
                http_status=400,
            )
        return self.control_plane.stop(block=block)

    def _logs_command(self, payload: dict[str, Any]) -> CommandResult:
        lines = payload.get("lines", 200)
        if isinstance(lines, bool):
            raise ControlPlaneError(
                message="logs.lines must be an integer",
                code=205,
                http_status=400,
            )
        if not isinstance(lines, int):
            raise ControlPlaneError(
                message="logs.lines must be an integer",
                code=206,
                http_status=400,
            )
        return self.control_plane.logs(lines=lines)

    def _wait_up_command(self, payload: dict[str, Any]) -> CommandResult:
        timeout_seconds = payload.get("timeout_seconds", self.control_plane.config.startup_timeout)
        poll_interval_seconds = payload.get("poll_interval_seconds", 2)
        defer_timeout_until_running = payload.get(
            "defer_timeout_until_running",
            self.control_plane.config.startup_timeout_after_running,
        )

        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
            raise ControlPlaneError(
                message="wait-up.timeout_seconds must be an integer",
                code=207,
                http_status=400,
            )
        if isinstance(poll_interval_seconds, bool) or not isinstance(
            poll_interval_seconds, (int, float)
        ):
            raise ControlPlaneError(
                message="wait-up.poll_interval_seconds must be a number",
                code=208,
                http_status=400,
            )
        if not isinstance(defer_timeout_until_running, bool):
            raise ControlPlaneError(
                message="wait-up.defer_timeout_until_running must be a boolean",
                code=211,
                http_status=400,
            )
        return self.control_plane.wait_up(
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=float(poll_interval_seconds),
            defer_timeout_until_running=defer_timeout_until_running,
        )

    def _handle_with_result(self, action: Any) -> None:
        try:
            result = action()
        except ControlPlaneError as exc:
            self._write_json(exc.http_status, error_payload(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._write_json(
                500,
                {
                    "ok": False,
                    "code": 500,
                    "message": f"internal server error: {exc}",
                },
            )
            return

        self._write_json(200, command_result_payload(result))

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected (common when CLI is interrupted with Ctrl-C).
            self.log_message("client disconnected before response could be sent")

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep default HTTP logs concise.
        super().log_message(fmt, *args)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM HPC control server")
    default_config = Path(__file__).resolve().parent / "server_config.toml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to server config TOML (default: {default_config})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override bind host from config",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override bind port from config",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    control_plane = ControlPlane(args.config)
    host = args.host or control_plane.config.host
    port = args.port or control_plane.config.port

    RequestHandler.control_plane = control_plane

    server = ReusableThreadingHTTPServer((host, port), RequestHandler)
    # Convert SIGTERM into KeyboardInterrupt so shutdown cleanup runs.
    def _signal_to_keyboard_interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_to_keyboard_interrupt)

    print(f"control server listening on http://{host}:{port}")
    print(f"config: {args.config}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            cleanup_result = control_plane.stop(reason="server_exit", block=True)
            print(
                "shutdown cleanup: stopped active job "
                f"{cleanup_result.data.get('job_id')} (previous_status={cleanup_result.data.get('previous_status')})"
            )
        except ControlPlaneError as exc:
            if exc.code == 21:
                print("shutdown cleanup: no active job")
            else:
                print(f"shutdown cleanup failed: code={exc.code} message={exc.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
