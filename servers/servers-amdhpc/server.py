#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HTTP control server for Slurm + Apptainer vLLM orchestration."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

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


def _parse_request_path(handler: BaseHTTPRequestHandler) -> tuple[str, dict[str, Any]]:
    parsed = urlparse(handler.path)
    query_payload = {key: values[-1] for key, values in parse_qs(parsed.query).items() if values}
    return parsed.path, query_payload


class RequestHandler(BaseHTTPRequestHandler):
    control_plane: ControlPlane
    GET_ROUTE_HANDLERS = {
        "/up": "_up_command",
        "/stop/poll": "_stop_poll_command",
        "/start/status": "_start_status_command",
        "/stop/status": "_stop_status_command",
        "/status": "_status_command",
    }
    POST_ROUTE_HANDLERS = {
        "/start": "_start_command",
        "/stop": "_stop_command",
        "/group/start": "_group_start_command",
        "/group/stop": "_group_stop_command",
        "/group/status": "_group_status_command",
        "/stop/poll": "_stop_poll_command",
        "/start/status": "_start_status_command",
        "/stop/status": "_stop_status_command",
        "/logs": "_logs_command",
        "/up": "_up_command",
        "/wait-up": "_wait_up_command",
        "/status": "_status_command",
        "/command/start": "_start_command",
        "/command/stop": "_stop_command",
        "/command/group/start": "_group_start_command",
        "/command/group/stop": "_group_stop_command",
        "/command/group/status": "_group_status_command",
        "/command/stop/poll": "_stop_poll_command",
        "/command/start/status": "_start_status_command",
        "/command/stop/status": "_stop_status_command",
        "/command/logs": "_logs_command",
        "/command/up": "_up_command",
        "/command/wait-up": "_wait_up_command",
        "/command/status": "_status_command",
    }

    def do_GET(self) -> None:  # noqa: N802
        path, payload = _parse_request_path(self)
        if path == "/health":
            self._write_json(200, {"ok": True, "message": "healthy"})
            return
        self._dispatch_route(path=path, payload=payload, routes=self.GET_ROUTE_HANDLERS)

    def do_POST(self) -> None:  # noqa: N802
        path, _ = _parse_request_path(self)

        try:
            payload = _parse_request_payload(self)
        except ControlPlaneError as exc:
            self._write_json(exc.http_status, error_payload(exc))
            return

        self._dispatch_route(path=path, payload=payload, routes=self.POST_ROUTE_HANDLERS)

    def _dispatch_route(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        routes: dict[str, str],
    ) -> None:
        handler_name = routes.get(path)
        if handler_name is None:
            self._write_unknown_endpoint(path)
            return
        self._handle_with_result(lambda: getattr(self, handler_name)(payload))

    def _write_unknown_endpoint(self, path: str) -> None:
        self._write_json(
            404,
            {
                "ok": False,
                "code": 404,
                "message": f"unknown endpoint: {path}",
            },
        )

    def _require_port_profile(self, payload: dict[str, Any], *, command_name: str) -> int:
        raw_port_profile = payload.get("port_profile")
        if raw_port_profile is None or raw_port_profile == "":
            raise ControlPlaneError(
                message=f"{command_name} requires 'port_profile'",
                code=212,
                http_status=400,
            )
        if isinstance(raw_port_profile, bool):
            raise ControlPlaneError(
                message=f"{command_name}.port_profile must be an integer",
                code=213,
                http_status=400,
            )
        if isinstance(raw_port_profile, int):
            return raw_port_profile
        if isinstance(raw_port_profile, str):
            try:
                return int(raw_port_profile)
            except ValueError as exc:
                raise ControlPlaneError(
                    message=f"{command_name}.port_profile must be an integer",
                    code=214,
                    http_status=400,
                ) from exc
        raise ControlPlaneError(
            message=f"{command_name}.port_profile must be an integer",
            code=215,
            http_status=400,
        )

    def _require_group_name(self, payload: dict[str, Any], *, command_name: str) -> str:
        raw_group_name = payload.get("group_name")
        if not isinstance(raw_group_name, str) or not raw_group_name.strip():
            raise ControlPlaneError(
                message=f"{command_name} requires non-empty 'group_name'",
                code=216,
                http_status=400,
            )
        return raw_group_name.strip()

    def _require_profile_list(self, payload: dict[str, Any], *, command_name: str) -> list[int]:
        raw_profile_list = payload.get("profile_list")
        values: list[Any]
        if isinstance(raw_profile_list, list):
            values = list(raw_profile_list)
        elif isinstance(raw_profile_list, str):
            values = [item.strip() for item in raw_profile_list.split(",") if item.strip()]
        else:
            raise ControlPlaneError(
                message=(
                    f"{command_name} requires 'profile_list' as a list of integers "
                    "or comma-separated string"
                ),
                code=217,
                http_status=400,
            )

        if not values:
            raise ControlPlaneError(
                message=f"{command_name}.profile_list cannot be empty",
                code=218,
                http_status=400,
            )

        parsed: list[int] = []
        for value in values:
            if isinstance(value, bool):
                raise ControlPlaneError(
                    message=f"{command_name}.profile_list values must be integers",
                    code=219,
                    http_status=400,
                )
            if isinstance(value, int):
                parsed.append(value)
                continue
            if isinstance(value, str):
                try:
                    parsed.append(int(value))
                    continue
                except ValueError as exc:
                    raise ControlPlaneError(
                        message=f"{command_name}.profile_list values must be integers",
                        code=220,
                        http_status=400,
                    ) from exc
            raise ControlPlaneError(
                message=f"{command_name}.profile_list values must be integers",
                code=221,
                http_status=400,
            )
        return parsed

    def _optional_extra_env(self, payload: dict[str, Any], *, command_name: str) -> dict[str, str]:
        raw_extra_env = payload.get("extra_env")
        if raw_extra_env is None:
            return {}

        parsed: dict[str, str] = {}
        if isinstance(raw_extra_env, dict):
            items = raw_extra_env.items()
        elif isinstance(raw_extra_env, list):
            items = []
            for item in raw_extra_env:
                if not isinstance(item, str):
                    raise ControlPlaneError(
                        message=f"{command_name}.extra_env list values must be strings in KEY=VALUE form",
                        code=226,
                        http_status=400,
                    )
                if "=" not in item:
                    raise ControlPlaneError(
                        message=f"{command_name}.extra_env value '{item}' must be KEY=VALUE",
                        code=227,
                        http_status=400,
                    )
                key, value = item.split("=", 1)
                items.append((key, value))
        else:
            raise ControlPlaneError(
                message=f"{command_name}.extra_env must be an object or list of KEY=VALUE strings",
                code=228,
                http_status=400,
            )

        for raw_key, raw_value in items:
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ControlPlaneError(
                    message=f"{command_name}.extra_env keys must be non-empty strings",
                    code=229,
                    http_status=400,
                )
            key = raw_key.strip()
            if key in parsed:
                raise ControlPlaneError(
                    message=f"{command_name}.extra_env key '{key}' is duplicated",
                    code=230,
                    http_status=400,
                )
            parsed[key] = str(raw_value)
        return parsed

    def _optional_lmcache_size(self, payload: dict[str, Any], *, command_name: str) -> str | None:
        raw_size = payload.get("lmcache")
        if raw_size is None:
            return None

        if isinstance(raw_size, bool):
            raise ControlPlaneError(
                message=f"{command_name}.lmcache must be a positive integer",
                code=231,
                http_status=400,
            )

        if isinstance(raw_size, int):
            size_value = raw_size
        elif isinstance(raw_size, str):
            text = raw_size.strip()
            if not text:
                raise ControlPlaneError(
                    message=f"{command_name}.lmcache must be a positive integer",
                    code=232,
                    http_status=400,
                )
            try:
                size_value = int(text)
            except ValueError as exc:
                raise ControlPlaneError(
                    message=f"{command_name}.lmcache must be a positive integer",
                    code=233,
                    http_status=400,
                ) from exc
        else:
            raise ControlPlaneError(
                message=f"{command_name}.lmcache must be a positive integer",
                code=234,
                http_status=400,
            )

        if size_value <= 0:
            raise ControlPlaneError(
                message=f"{command_name}.lmcache must be a positive integer",
                code=235,
                http_status=400,
            )

        return str(size_value)

    def _start_command(self, payload: dict[str, Any]) -> CommandResult:
        port_profile = self._require_port_profile(payload, command_name="start")
        partition = payload.get("partition")
        model = payload.get("model")
        block = payload.get("block", False)
        extra_env = self._optional_extra_env(payload, command_name="start")
        lmcache_size = self._optional_lmcache_size(payload, command_name="start")
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
        return self.control_plane.start(
            port_profile_id=port_profile,
            partition=partition,
            model=model,
            block=block,
            extra_env=extra_env,
            lmcache_max_local_cpu_size=lmcache_size,
        )

    def _stop_command(self, payload: dict[str, Any]) -> CommandResult:
        port_profile = self._require_port_profile(payload, command_name="stop")
        block = payload.get("block", False)
        if not isinstance(block, bool):
            raise ControlPlaneError(
                message="stop.block must be a boolean",
                code=210,
                http_status=400,
            )
        return self.control_plane.stop(port_profile_id=port_profile, block=block)

    def _group_start_command(self, payload: dict[str, Any]) -> CommandResult:
        group_name = self._require_group_name(payload, command_name="group/start")
        profile_list = self._require_profile_list(payload, command_name="group/start")
        partition = payload.get("partition")
        model = payload.get("model")
        block = payload.get("block", True)
        extra_env = self._optional_extra_env(payload, command_name="group/start")
        lmcache_size = self._optional_lmcache_size(payload, command_name="group/start")
        if not isinstance(partition, str) or not partition:
            raise ControlPlaneError(
                message="group/start requires 'partition'",
                code=222,
                http_status=400,
            )
        if not isinstance(model, str) or not model:
            raise ControlPlaneError(
                message="group/start requires 'model'",
                code=223,
                http_status=400,
            )
        if not isinstance(block, bool):
            raise ControlPlaneError(
                message="group/start.block must be a boolean",
                code=224,
                http_status=400,
            )
        return self.control_plane.start_group(
            group_name=group_name,
            port_profile_ids=profile_list,
            partition=partition,
            model=model,
            block=block,
            extra_env=extra_env,
            lmcache_max_local_cpu_size=lmcache_size,
        )

    def _group_stop_command(self, payload: dict[str, Any]) -> CommandResult:
        group_name = self._require_group_name(payload, command_name="group/stop")
        block = payload.get("block", True)
        if not isinstance(block, bool):
            raise ControlPlaneError(
                message="group/stop.block must be a boolean",
                code=225,
                http_status=400,
            )
        return self.control_plane.stop_group(group_name=group_name, block=block)

    def _group_status_command(self, payload: dict[str, Any]) -> CommandResult:
        group_name = self._require_group_name(payload, command_name="group/status")
        return self.control_plane.group_status(group_name=group_name)

    def _stop_poll_command(self, payload: dict[str, Any]) -> CommandResult:
        port_profile = self._require_port_profile(payload, command_name="stop-poll")
        return self.control_plane.stop_poll(port_profile_id=port_profile)

    def _start_status_command(self, payload: dict[str, Any]) -> CommandResult:
        port_profile = self._require_port_profile(payload, command_name="start/status")
        return self.control_plane.start_status(port_profile_id=port_profile)

    def _stop_status_command(self, payload: dict[str, Any]) -> CommandResult:
        port_profile = self._require_port_profile(payload, command_name="stop/status")
        return self.control_plane.stop_status(port_profile_id=port_profile)

    def _logs_command(self, payload: dict[str, Any]) -> CommandResult:
        port_profile = self._require_port_profile(payload, command_name="logs")
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
        return self.control_plane.logs(port_profile_id=port_profile, lines=lines)

    def _up_command(self, payload: dict[str, Any]) -> CommandResult:
        port_profile = self._require_port_profile(payload, command_name="up")
        return self.control_plane.up(port_profile_id=port_profile)

    def _wait_up_command(self, payload: dict[str, Any]) -> CommandResult:
        port_profile = self._require_port_profile(payload, command_name="wait-up")
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
            port_profile_id=port_profile,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=float(poll_interval_seconds),
            defer_timeout_until_running=defer_timeout_until_running,
        )

    def _status_command(self, payload: dict[str, Any]) -> CommandResult:
        port_profile = self._require_port_profile(payload, command_name="status")
        return self.control_plane.status(port_profile_id=port_profile)

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

    try:
        control_plane = ControlPlane(args.config)
        image_actions = control_plane.validate_startup_requirements()
    except ControlPlaneError as exc:
        print(f"startup check failed: code={exc.code} message={exc.message}")
        if exc.details:
            print(json.dumps(exc.details, indent=2, sort_keys=True))
        return 2

    for action in image_actions:
        print(f"startup image prep: {action}")

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
        stop_requested_profiles: set[int] = set()
        stop_request_failures: set[int] = set()
        stop_poll_failures: set[int] = set()
        stopped_count = 0

        # Phase 1: request cancellation for all profiles first (non-blocking).
        for port_profile_id in sorted(control_plane.config.port_profiles.keys()):
            try:
                cleanup_result = control_plane.stop(
                    port_profile_id=port_profile_id,
                    reason="server_exit",
                    block=False,
                    allow_group=True,
                )
                result_data = cleanup_result.data if isinstance(cleanup_result.data, dict) else {}
                final_status = str(result_data.get("final_status", "unknown"))
                job_id = result_data.get("job_id")
                if final_status == "cancelling":
                    stop_requested_profiles.add(port_profile_id)
                    print(
                        "shutdown cleanup: stop requested for profile "
                        f"{port_profile_id}: {job_id} "
                        f"(previous_status={result_data.get('previous_status')})"
                    )
                else:
                    print(
                        "shutdown cleanup: profile "
                        f"{port_profile_id} already inactive "
                        f"(final_status={final_status})"
                    )
            except ControlPlaneError as exc:
                if exc.code == 21:
                    continue
                stop_request_failures.add(port_profile_id)
                stop_requested_profiles.add(port_profile_id)
                print(
                    "shutdown cleanup failed to request stop for profile "
                    f"{port_profile_id}: code={exc.code} message={exc.message}"
                )

        # Phase 2: wait for all previously requested stops to complete.
        if stop_requested_profiles:
            deadline = time.monotonic() + control_plane.config.stop_wait_timeout_seconds
            pending_profiles = set(stop_requested_profiles)
            last_reported_status: dict[int, tuple[str | None, str | None]] = {}
            while pending_profiles:
                finished_profiles: set[int] = set()
                for port_profile_id in sorted(pending_profiles):
                    try:
                        poll_result = control_plane.stop_poll(
                            port_profile_id=port_profile_id,
                            allow_group=True,
                        )
                    except ControlPlaneError as exc:
                        if port_profile_id not in stop_poll_failures:
                            stop_poll_failures.add(port_profile_id)
                            print(
                                "shutdown cleanup polling failed for profile "
                                f"{port_profile_id}: code={exc.code} message={exc.message}"
                            )
                        continue

                    poll_data = poll_result.data if isinstance(poll_result.data, dict) else {}
                    done = bool(poll_data.get("done"))
                    job_id = poll_data.get("job_id")
                    job_status = poll_data.get("job_status")
                    if done:
                        finished_profiles.add(port_profile_id)
                        stopped_count += 1
                        print(
                            "shutdown cleanup: stopped active job for profile "
                            f"{port_profile_id}: {job_id} "
                            f"(final_status={job_status})"
                        )
                        continue

                    current_status = (
                        str(job_id) if job_id is not None else None,
                        str(job_status) if job_status is not None else None,
                    )
                    if last_reported_status.get(port_profile_id) != current_status:
                        print(
                            "shutdown cleanup: waiting on profile "
                            f"{port_profile_id} job={job_id} status={job_status}"
                        )
                        last_reported_status[port_profile_id] = current_status

                pending_profiles -= finished_profiles
                if not pending_profiles:
                    break
                if time.monotonic() >= deadline:
                    print(
                        "shutdown cleanup timed out waiting for profiles: "
                        f"{sorted(pending_profiles)}"
                    )
                    stop_poll_failures.update(pending_profiles)
                    break
                time.sleep(control_plane.config.stop_poll_interval_seconds)

        failure_count = len(stop_request_failures | stop_poll_failures)
        if stopped_count == 0 and failure_count == 0:
            print("shutdown cleanup: no active jobs")
        elif failure_count > 0:
            print(
                f"shutdown cleanup completed with {failure_count} failure(s); "
                f"stopped {stopped_count} active job(s)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
