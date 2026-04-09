#!/usr/bin/env python3
"""AMD SMI power daemon helpers and JSON-over-UDS IPC utilities."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import socketserver
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import amdsmi as _amdsmi
except Exception as exc:  # pragma: no cover - depends on local ROCm install
    _amdsmi = None
    _AMDSMI_IMPORT_ERROR = exc
else:
    _AMDSMI_IMPORT_ERROR = None


DEFAULT_SOCKET_PATH = "/tmp/amdsmi-power-reader.sock"
LOGGER = logging.getLogger("amdsmi-power-daemon")


def utc_timestamp() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _json_safe(value: Any) -> Any:
    """Convert AMD SMI return values into JSON-safe values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _extract_socket_power_w(power_info: Mapping[str, Any]) -> float | None:
    """Pick the best socket power field from the AMD SMI payload."""
    for key in ("socket_power", "current_socket_power", "average_socket_power"):
        value = power_info.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.upper() != "N/A":
            try:
                return float(value)
            except ValueError:
                continue
    return None


@dataclass(frozen=True)
class GpuIdentity:
    """Serializable GPU metadata cached by the daemon."""

    index: int
    bdf: str | None
    uuid: str | None
    hip_id: int | None
    hip_uuid: str | None
    drm_card: int | None
    drm_render: int | None
    hsa_id: int | None


class AmdSmiPowerService:
    """Holds AMD SMI state and serves power requests."""

    def __init__(self, amdsmi_module: Any | None = None) -> None:
        if amdsmi_module is None:
            if _amdsmi is None:
                raise RuntimeError(
                    "amdsmi import failed; install ROCm AMD SMI Python bindings first"
                ) from _AMDSMI_IMPORT_ERROR
            amdsmi_module = _amdsmi

        self._amdsmi = amdsmi_module
        self._lock = threading.RLock()
        self._initialized = False
        self._handles: list[Any] = []
        self._gpu_identities: list[GpuIdentity] = []

    def start(self) -> None:
        """Initialize AMD SMI and cache GPU handles."""
        with self._lock:
            if self._initialized:
                return

            self._amdsmi.amdsmi_init()
            self._handles = list(self._amdsmi.amdsmi_get_processor_handles())
            self._gpu_identities = [
                self._build_gpu_identity(index, handle)
                for index, handle in enumerate(self._handles)
            ]
            self._initialized = True

    def close(self) -> None:
        """Release AMD SMI resources."""
        with self._lock:
            if not self._initialized:
                return

            try:
                self._amdsmi.amdsmi_shut_down()
            finally:
                self._handles = []
                self._gpu_identities = []
                self._initialized = False

    def handle_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Handle a JSON request payload from the IPC layer."""
        command = request.get("command")

        if command == "ping":
            return {
                "ok": True,
                "timestamp": utc_timestamp(),
                "gpu_count": len(self._gpu_identities),
            }

        if command == "list_gpus":
            return {
                "ok": True,
                "timestamp": utc_timestamp(),
                "gpus": self.list_gpus(),
            }

        if command == "get_power":
            gpu_index = request.get("gpu_index", 0)
            if not isinstance(gpu_index, int):
                raise ValueError("gpu_index must be an integer")
            return self.read_power(gpu_index)

        raise ValueError(f"Unsupported command: {command!r}")

    def list_gpus(self) -> list[dict[str, Any]]:
        """Return cached GPU metadata."""
        with self._lock:
            return [asdict(identity) for identity in self._gpu_identities]

    def read_power(self, gpu_index: int) -> dict[str, Any]:
        """Read power info for a single GPU index."""
        with self._lock:
            self._require_started()

            if gpu_index < 0 or gpu_index >= len(self._handles):
                raise IndexError(
                    f"gpu_index {gpu_index} out of range; found {len(self._handles)} GPU(s)"
                )

            handle = self._handles[gpu_index]
            identity = self._gpu_identities[gpu_index]
            raw_power_info = _json_safe(self._amdsmi.amdsmi_get_power_info(handle))

        return {
            "ok": True,
            "timestamp": utc_timestamp(),
            "gpu": asdict(identity),
            "power_w": _extract_socket_power_w(raw_power_info),
            "power_info": raw_power_info,
        }

    def _build_gpu_identity(self, index: int, handle: Any) -> GpuIdentity:
        """Build a serializable identity object for one GPU handle."""
        enumeration = self._safe_call("amdsmi_get_gpu_enumeration_info", handle) or {}
        return GpuIdentity(
            index=index,
            bdf=self._safe_call("amdsmi_get_gpu_device_bdf", handle),
            uuid=self._safe_call("amdsmi_get_gpu_device_uuid", handle),
            hip_id=enumeration.get("hip_id"),
            hip_uuid=enumeration.get("hip_uuid"),
            drm_card=enumeration.get("drm_card"),
            drm_render=enumeration.get("drm_render"),
            hsa_id=enumeration.get("hsa_id"),
        )

    def _require_started(self) -> None:
        if not self._initialized:
            raise RuntimeError("AMD SMI service has not been started")

    def _safe_call(self, name: str, *args: Any) -> Any:
        func = getattr(self._amdsmi, name)
        try:
            return func(*args)
        except Exception as exc:  # pragma: no cover - defensive logging path
            LOGGER.warning("%s failed: %s", name, exc)
            return None


class _ThreadingUnixStreamServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Threaded Unix domain socket server for JSON line requests."""

    daemon_threads = True


class _JsonLineRequestHandler(socketserver.StreamRequestHandler):
    """Handle a single JSON request and return a single JSON response."""

    def handle(self) -> None:
        line = self.rfile.readline()
        if not line:
            return

        try:
            request = json.loads(line.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request payload must be a JSON object")
            response = self.server.service.handle_request(request)  # type: ignore[attr-defined]
        except Exception as exc:
            response = {
                "ok": False,
                "timestamp": utc_timestamp(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

        self.wfile.write((json.dumps(response, ensure_ascii=True) + "\n").encode("utf-8"))
        self.wfile.flush()


def _remove_socket_file(socket_path: str) -> None:
    path = Path(socket_path)
    if not path.exists():
        return
    if not path.is_socket():
        raise RuntimeError(f"Refusing to remove non-socket path: {socket_path}")
    path.unlink()


def run_daemon(
    socket_path: str = DEFAULT_SOCKET_PATH,
    service: AmdSmiPowerService | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Run the Unix domain socket daemon until interrupted."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    service = service or AmdSmiPowerService()
    old_handlers: dict[int, Any] = {}
    stop_event = stop_event or threading.Event()

    try:
        service.start()

        socket_dir = Path(socket_path).parent
        socket_dir.mkdir(parents=True, exist_ok=True)
        _remove_socket_file(socket_path)

        with _ThreadingUnixStreamServer(socket_path, _JsonLineRequestHandler) as server:
            server.service = service  # type: ignore[attr-defined]
            server.timeout = 0.5

            def _handle_signal(signum: int, _frame: Any) -> None:
                LOGGER.info("Received signal %s, stopping daemon", signum)
                stop_event.set()

            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGINT, signal.SIGTERM):
                    old_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, _handle_signal)

            try:
                LOGGER.info(
                    "AMD SMI power daemon listening on %s for %d GPU(s)",
                    socket_path,
                    len(service.list_gpus()),
                )
                while not stop_event.is_set():
                    server.handle_request()
            finally:
                server.server_close()
                _remove_socket_file(socket_path)
    finally:
        service.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def send_request(
    request: Mapping[str, Any],
    socket_path: str = DEFAULT_SOCKET_PATH,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Send one request to the daemon and parse the JSON response."""
    payload = json.dumps(dict(request), ensure_ascii=True).encode("utf-8") + b"\n"

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(socket_path)
        client.sendall(payload)

        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            buffer += chunk

    if not buffer:
        raise RuntimeError("Daemon closed the connection without a response")

    response = json.loads(buffer.decode("utf-8"))
    if not isinstance(response, dict):
        raise RuntimeError("Daemon response was not a JSON object")
    return response


def request_power(
    gpu_index: int = 0,
    socket_path: str = DEFAULT_SOCKET_PATH,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Request a power sample from the daemon."""
    return send_request(
        {"command": "get_power", "gpu_index": gpu_index},
        socket_path=socket_path,
        timeout=timeout,
    )


def request_gpu_list(
    socket_path: str = DEFAULT_SOCKET_PATH,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Request the daemon's cached GPU inventory."""
    return send_request(
        {"command": "list_gpus"},
        socket_path=socket_path,
        timeout=timeout,
    )


def daemon_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the daemon process."""
    parser = argparse.ArgumentParser(
        prog="amd-smi-power-daemon",
        description="Expose AMD GPU power readings over a Unix domain socket.",
    )
    parser.add_argument(
        "--socket-path",
        "-s",
        default=DEFAULT_SOCKET_PATH,
        help=f"Unix socket path for IPC (default: {DEFAULT_SOCKET_PATH}).",
    )
    args = parser.parse_args(argv)

    try:
        run_daemon(socket_path=args.socket_path)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def client_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the client process."""
    parser = argparse.ArgumentParser(
        prog="amd-smi-power-client",
        description="Request AMD GPU power from the local daemon.",
    )
    parser.add_argument(
        "--socket-path",
        "-s",
        default=DEFAULT_SOCKET_PATH,
        help=f"Unix socket path for IPC (default: {DEFAULT_SOCKET_PATH}).",
    )
    parser.add_argument(
        "--gpu-index",
        "-g",
        type=int,
        default=0,
        help="GPU index to query (default: 0).",
    )
    parser.add_argument(
        "--list-gpus",
        action="store_true",
        help="List GPUs known to the daemon instead of reading power.",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=5.0,
        help="IPC timeout in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--power-only",
        action="store_true",
        help="Print only the selected GPU socket power in watts.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON responses.",
    )
    args = parser.parse_args(argv)

    try:
        response = (
            request_gpu_list(socket_path=args.socket_path, timeout=args.timeout)
            if args.list_gpus
            else request_power(
                gpu_index=args.gpu_index,
                socket_path=args.socket_path,
                timeout=args.timeout,
            )
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not response.get("ok"):
        print(json.dumps(response, ensure_ascii=True, indent=2 if args.pretty else None))
        return 1

    if args.power_only:
        print(response.get("power_w"))
        return 0

    print(json.dumps(response, ensure_ascii=True, indent=2 if args.pretty else None))
    return 0
