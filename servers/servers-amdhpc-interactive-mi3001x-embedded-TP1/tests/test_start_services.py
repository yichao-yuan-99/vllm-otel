#!/usr/bin/env python3
"""Smoke tests for the interactive embedded TP1 launcher script/docs."""

from __future__ import annotations

from pathlib import Path
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = MODULE_ROOT / "start-services.sh"
README = MODULE_ROOT / "README.md"


class StartServicesScriptTest(unittest.TestCase):
    def test_start_services_includes_amd_power_daemon_lifecycle(self) -> None:
        script_text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('AMD_SMI_POWER_DAEMON_BIN="${AMD_SMI_POWER_DAEMON_BIN:-amd-smi-power-daemon}"', script_text)
        self.assertIn('AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.sock}"', script_text)
        self.assertIn('command -v "${AMD_SMI_POWER_DAEMON_BIN}"', script_text)
        self.assertIn("probe_unix_socket()", script_text)
        self.assertIn("start_amd_smi_power_daemon()", script_text)
        self.assertIn('terminate_process "amd-smi-power-daemon" "${AMD_SMI_POWER_DAEMON_PID:-}"', script_text)
        self.assertIn('echo "  amd-smi-power-socket: ${AMD_SMI_POWER_SOCKET_PATH}"', script_text)
        self.assertIn("start_amd_smi_power_daemon\nstart_shared_jaeger", script_text)

    def test_start_services_launches_gateway_ctx(self) -> None:
        script_text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('GATEWAY_CONFIG_DEFAULT="${GATEWAY_CONFIG_DEFAULT:-${REPO_ROOT}/gateway_ctx/config.toml}"', script_text)
        self.assertIn('GATEWAY_CONFIG_FALLBACK="${GATEWAY_CONFIG_FALLBACK:-${REPO_ROOT}/gateway_ctx/config.example.toml}"', script_text)
        self.assertIn("-m gateway_ctx", script_text)
        self.assertIn('echo "Launching gateway-ctx profile=${PROFILE_ID} raw=${GATEWAY_PORT} parsed=${GATEWAY_PARSE_PORT}"', script_text)

    def test_readme_documents_amd_power_daemon_usage(self) -> None:
        readme_text = README.read_text(encoding="utf-8")

        self.assertIn("one shared AMD SMI power daemon for `amd-power-reader`", readme_text)
        self.assertIn("AMD_SMI_POWER_DAEMON_BIN", readme_text)
        self.assertIn("AMD_SMI_POWER_SOCKET_PATH", readme_text)
        self.assertIn("pip install -e ./amd-power-reader", readme_text)
        self.assertIn("amd-power-reader --socket-path", readme_text)

    def test_readme_documents_gateway_ctx_usage(self) -> None:
        readme_text = README.read_text(encoding="utf-8")

        self.assertIn("one ctx-aware gateway (`gateway_ctx`) on port profile `0`", readme_text)
        self.assertIn("launching Jaeger, vLLM, and", readme_text)
        self.assertIn("`gateway_ctx`", readme_text)
        self.assertIn("pip install -e ./gateway_ctx", readme_text)


if __name__ == "__main__":
    unittest.main()
