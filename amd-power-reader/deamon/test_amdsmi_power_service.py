from __future__ import annotations

import threading
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import amdsmi_power_service as power_service


class FakeAmdSmi:
    def __init__(self) -> None:
        self.init_calls = 0
        self.shutdown_calls = 0
        self.handles = ["gpu0", "gpu1"]

    def amdsmi_init(self) -> None:
        self.init_calls += 1

    def amdsmi_shut_down(self) -> None:
        self.shutdown_calls += 1

    def amdsmi_get_processor_handles(self) -> list[str]:
        return list(self.handles)

    def amdsmi_get_gpu_device_bdf(self, handle: str) -> str:
        return {
            "gpu0": "0000:01:00.0",
            "gpu1": "0000:02:00.0",
        }[handle]

    def amdsmi_get_gpu_device_uuid(self, handle: str) -> str:
        return {
            "gpu0": "uuid-0",
            "gpu1": "uuid-1",
        }[handle]

    def amdsmi_get_gpu_enumeration_info(self, handle: str) -> dict[str, int | str]:
        return {
            "gpu0": {
                "hip_id": 0,
                "hip_uuid": "GPU-uuid-0",
                "drm_card": 0,
                "drm_render": 128,
                "hsa_id": 10,
            },
            "gpu1": {
                "hip_id": 1,
                "hip_uuid": "GPU-uuid-1",
                "drm_card": 1,
                "drm_render": 129,
                "hsa_id": 11,
            },
        }[handle]

    def amdsmi_get_power_info(self, handle: str) -> dict[str, int | str]:
        return {
            "gpu0": {
                "socket_power": 41,
                "average_socket_power": 40,
                "current_socket_power": "N/A",
                "power_limit": 300000000,
            },
            "gpu1": {
                "socket_power": 57,
                "average_socket_power": 56,
                "current_socket_power": "N/A",
                "power_limit": 300000000,
            },
        }[handle]


def _start_daemon(
    tmp_path: Path,
    backend: FakeAmdSmi,
) -> tuple[str, threading.Event, threading.Thread]:
    socket_path = str(tmp_path / "amdsmi-power.sock")
    stop_event = threading.Event()
    service = power_service.AmdSmiPowerService(amdsmi_module=backend)
    thread = threading.Thread(
        target=power_service.run_daemon,
        kwargs={
            "socket_path": socket_path,
            "service": service,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    thread.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if Path(socket_path).exists():
            return socket_path, stop_event, thread
        if not thread.is_alive():
            raise RuntimeError("daemon thread exited before creating the socket")
        time.sleep(0.05)

    raise RuntimeError("timed out waiting for daemon socket")


def _stop_daemon(stop_event: threading.Event, thread: threading.Thread) -> None:
    stop_event.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_round_trip_power_request(tmp_path: Path) -> None:
    backend = FakeAmdSmi()
    socket_path, stop_event, thread = _start_daemon(tmp_path, backend)

    try:
        response = power_service.request_power(gpu_index=1, socket_path=socket_path)
    finally:
        _stop_daemon(stop_event, thread)

    assert backend.init_calls == 1
    assert backend.shutdown_calls == 1
    assert response["ok"] is True
    assert response["gpu"]["index"] == 1
    assert response["gpu"]["uuid"] == "uuid-1"
    assert response["power_w"] == 57.0
    assert response["power_info"]["socket_power"] == 57


def test_list_gpus_and_invalid_index_error(tmp_path: Path) -> None:
    backend = FakeAmdSmi()
    socket_path, stop_event, thread = _start_daemon(tmp_path, backend)

    try:
        gpu_response = power_service.request_gpu_list(socket_path=socket_path)
        error_response = power_service.send_request(
            {"command": "get_power", "gpu_index": 99},
            socket_path=socket_path,
        )
    finally:
        _stop_daemon(stop_event, thread)

    assert gpu_response["ok"] is True
    assert [gpu["bdf"] for gpu in gpu_response["gpus"]] == [
        "0000:01:00.0",
        "0000:02:00.0",
    ]
    assert error_response["ok"] is False
    assert error_response["error_type"] == "IndexError"
