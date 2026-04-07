#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for gateway_ctx launch support."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import client  # type: ignore[import-not-found]


class GatewayCtxOptionTest(unittest.TestCase):
    def test_resolve_gateway_ipc_socket_path_uses_gateway_ctx_default_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            missing_primary = tmp_path / "missing-gateway-ctx.toml"
            missing_example = tmp_path / "missing-gateway-ctx.example.toml"

            with (
                mock.patch.object(client, "DEFAULT_GATEWAY_CTX_CONFIG_PATH", missing_primary),
                mock.patch.object(client, "DEFAULT_GATEWAY_CTX_CONFIG_EXAMPLE_PATH", missing_example),
            ):
                socket_path = client._resolve_gateway_ipc_socket_path(
                    3,
                    gateway_mode=client.DEFAULT_GATEWAY_CTX_MODULE_NAME,
                )

        self.assertEqual(socket_path, Path("/tmp/vllm-gateway-ctx-profile-3.sock"))

    def test_gateway_mode_for_port_profile_uses_state_selection_when_pid_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pid_file = tmp_path / "gateway.pid"
            log_file = tmp_path / "gateway.log"

            with (
                mock.patch.object(client, "_gateway_runtime_files", return_value=(pid_file, log_file)),
                mock.patch.object(
                    client,
                    "_load_state",
                    return_value={"selection": {"port_profile_id": "2", "gateway_ctx": True}},
                ),
            ):
                gateway_mode = client._gateway_mode_for_port_profile(2)

        self.assertEqual(gateway_mode, client.DEFAULT_GATEWAY_CTX_MODULE_NAME)

    def test_gateway_record_matches_running_process_for_gateway_ctx(self) -> None:
        record = {
            "pid": 1234,
            "port_profile_id": 2,
            "gateway_mode": client.DEFAULT_GATEWAY_CTX_MODULE_NAME,
        }
        with (
            mock.patch.object(client, "_pid_is_running", return_value=True),
            mock.patch.object(
                client,
                "_process_cmdline",
                return_value="python -m gateway_ctx start --port-profile-id 2",
            ),
        ):
            matched = client._gateway_record_matches_running_process(record)

        self.assertTrue(matched)

    def test_start_gateway_for_selection_uses_gateway_ctx_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "gateway_ctx.toml"
            config_path.write_text("schema_version = 1\n", encoding="utf-8")
            pid_file = tmp_path / "gateway.pid"
            log_file = tmp_path / "gateway.log"

            class FakeProc:
                pid = 4321

                def poll(self) -> None:
                    return None

            selection = {
                "port_profile_id": "2",
                "gateway_ctx": True,
                "ports": {
                    "jaeger_api_port": 16686,
                    "jaeger_otlp_port": 4317,
                },
            }

            with (
                mock.patch.object(client, "_gateway_runtime_files", return_value=(pid_file, log_file)),
                mock.patch.object(
                    client,
                    "_resolve_gateway_config_path",
                    return_value=config_path,
                ),
                mock.patch.object(client, "_ensure_runtime_dir"),
                mock.patch.object(client, "_utc_now_iso", return_value="2026-04-03T06:00:00+00:00"),
                mock.patch.object(client.time, "sleep"),
                mock.patch.object(client.time, "monotonic", side_effect=[0.0, 0.1, 3.1]),
                mock.patch.object(client.subprocess, "Popen", return_value=FakeProc()),
            ):
                payload = client._start_gateway_for_selection(selection)

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["data"]["gateway_mode"], client.DEFAULT_GATEWAY_CTX_MODULE_NAME)
            self.assertEqual(payload["data"]["module"], client.DEFAULT_GATEWAY_CTX_MODULE_NAME)
            self.assertEqual(payload["data"]["config_path"], str(config_path))
            self.assertEqual(payload["data"]["command"][2], client.DEFAULT_GATEWAY_CTX_MODULE_NAME)
            self.assertIn("gateway_ctx start", log_file.read_text(encoding="utf-8"))

            record = json.loads(pid_file.read_text(encoding="utf-8"))
            self.assertEqual(record["gateway_mode"], client.DEFAULT_GATEWAY_CTX_MODULE_NAME)
            self.assertEqual(record["module"], client.DEFAULT_GATEWAY_CTX_MODULE_NAME)
            self.assertEqual(record["command"][2], client.DEFAULT_GATEWAY_CTX_MODULE_NAME)


if __name__ == "__main__":
    unittest.main()
