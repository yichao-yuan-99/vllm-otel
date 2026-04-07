#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Docker stop service discovery."""

from __future__ import annotations

from pathlib import Path
import signal
import socket
import sys
import tempfile
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import client  # type: ignore[import-not-found]


class DockerStopManagedServicesTest(unittest.TestCase):
    def test_terminate_pid_or_group_does_not_fall_back_to_os_kill_on_permission_error(self) -> None:
        with (
            mock.patch.object(client.os, "killpg", side_effect=PermissionError),
            mock.patch.object(client.os, "kill") as kill,
        ):
            client._terminate_pid_or_group(1234, signal.SIGTERM)

        kill.assert_not_called()

    def test_stop_reports_managed_services_alongside_running_services(self) -> None:
        saved_states: list[dict[str, object]] = []
        load_state_values = [
            {"active": True, "lifecycle_state": "running"},
            {"active": True, "lifecycle_state": "running"},
        ]

        with (
            mock.patch.object(client, "_load_state", side_effect=lambda: dict(load_state_values.pop(0))),
            mock.patch.object(client, "_save_state", side_effect=saved_states.append),
            mock.patch.object(client, "_current_port_profile_id", return_value=2),
            mock.patch.object(
                client,
                "_stop_gateway_for_port_profile",
                return_value=client._payload(ok=True, code=0, message="gateway stopped"),
            ),
            mock.patch.object(
                client,
                "_compose_running_services",
                side_effect=[
                    (True, {"jaeger"}, ""),
                    (True, set(), ""),
                ],
            ),
            mock.patch.object(
                client,
                "_compose_managed_services",
                side_effect=[
                    (True, {"jaeger", "vllm"}, ""),
                    (True, set(), ""),
                ],
            ),
            mock.patch.object(
                client,
                "_compose_down",
                return_value=client.ExecResult(returncode=0, stdout="", stderr=""),
            ),
            mock.patch.object(client, "_utc_now_iso", return_value="2026-04-01T21:54:48+00:00"),
        ):
            payload = client._stop_environment_blocking_impl("stop_blocking_selected")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["message"], "environment stopped")
        self.assertEqual(payload["data"]["compose_before"]["running_services"], ["jaeger"])
        self.assertEqual(payload["data"]["compose_before"]["managed_services"], ["jaeger", "vllm"])
        self.assertEqual(payload["data"]["compose_after"]["running_services"], [])
        self.assertEqual(payload["data"]["compose_after"]["managed_services"], [])
        self.assertEqual(saved_states[-1]["active"], False)
        self.assertEqual(saved_states[-1]["lifecycle_state"], "inactive")

    def test_stop_gateway_without_pid_record_removes_stale_ipc_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pid_file = tmp_path / "gateway.pid"
            log_file = tmp_path / "gateway.log"
            socket_path = tmp_path / "gateway.sock"

            stale_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stale_socket.bind(str(socket_path))
            stale_socket.close()

            with (
                mock.patch.object(client, "_gateway_runtime_files", return_value=(pid_file, log_file)),
                mock.patch.object(client, "_resolve_gateway_ipc_socket_path", return_value=socket_path),
            ):
                payload = client._stop_gateway_for_port_profile(2)

            self.assertFalse(socket_path.exists())

        self.assertTrue(payload["ok"])
        self.assertEqual(
            payload["message"],
            "gateway is not running for port profile 2 (removed stale IPC socket)",
        )
        self.assertTrue(payload["data"]["ipc_socket_cleanup"]["removed"])

    def test_stop_gateway_without_pid_record_preserves_active_ipc_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pid_file = tmp_path / "gateway.pid"
            log_file = tmp_path / "gateway.log"
            socket_path = tmp_path / "gateway.sock"

            active_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                active_socket.bind(str(socket_path))
                active_socket.listen(1)

                with (
                    mock.patch.object(client, "_gateway_runtime_files", return_value=(pid_file, log_file)),
                    mock.patch.object(client, "_resolve_gateway_ipc_socket_path", return_value=socket_path),
                ):
                    payload = client._stop_gateway_for_port_profile(2)
            finally:
                active_socket.close()
                if socket_path.exists():
                    socket_path.unlink()

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["message"], "gateway is not running for port profile 2")
        self.assertTrue(payload["data"]["ipc_socket_cleanup"]["active_listener"])
        self.assertFalse(payload["data"]["ipc_socket_cleanup"]["removed"])

    def test_stop_does_not_skip_when_only_stopped_services_remain(self) -> None:
        saved_states: list[dict[str, object]] = []
        load_state_values = [
            {"active": False, "lifecycle_state": "inactive"},
            {"active": False, "lifecycle_state": "inactive"},
        ]

        with (
            mock.patch.object(client, "_load_state", side_effect=lambda: dict(load_state_values.pop(0))),
            mock.patch.object(client, "_save_state", side_effect=saved_states.append),
            mock.patch.object(client, "_current_port_profile_id", return_value=2),
            mock.patch.object(
                client,
                "_stop_gateway_for_port_profile",
                return_value=client._payload(ok=True, code=0, message="gateway stopped"),
            ),
            mock.patch.object(
                client,
                "_compose_running_services",
                side_effect=[
                    (True, set(), ""),
                    (True, set(), ""),
                ],
            ),
            mock.patch.object(
                client,
                "_compose_managed_services",
                side_effect=[
                    (True, {"vllm"}, ""),
                    (True, set(), ""),
                ],
            ),
            mock.patch.object(
                client,
                "_compose_down",
                return_value=client.ExecResult(returncode=0, stdout="", stderr=""),
            ) as compose_down,
            mock.patch.object(client, "_utc_now_iso", return_value="2026-04-01T21:54:48+00:00"),
        ):
            payload = client._stop_environment_blocking_impl("stop_blocking_selected")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["message"], "environment stopped")
        self.assertEqual(payload["data"]["compose_before"]["running_services"], [])
        self.assertEqual(payload["data"]["compose_before"]["managed_services"], ["vllm"])
        compose_down.assert_called_once()
        self.assertEqual(saved_states[-1]["active"], False)
        self.assertEqual(saved_states[-1]["lifecycle_state"], "inactive")


if __name__ == "__main__":
    unittest.main()
