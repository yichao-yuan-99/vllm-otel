#!/usr/bin/env python3
"""Smoke tests for the interactive mi3008x embedded TP1 launcher script/docs."""

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

    def test_start_services_unrolls_all_mi3008x_profiles(self) -> None:
        script_text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("PROFILE_IDS=(0 1 2 3 4 5 6 7)", script_text)
        self.assertIn("declare -A PROFILE_GPU_IDS=(", script_text)
        for profile_id in range(8):
            self.assertIn(f"launch_vllm {profile_id} ", script_text)
            self.assertIn(f"wait_for_vllm_ready {profile_id} ", script_text)
            self.assertIn(f"launch_gateway {profile_id} ", script_text)
            self.assertIn(f"wait_for_gateway_ready {profile_id} ", script_text)
        self.assertIn("--env ROCR_VISIBLE_DEVICES", script_text)
        self.assertIn('echo "  profile ${profile_id}: gpu=${PROFILE_GPU_IDS[$profile_id]}', script_text)

    def test_start_services_launches_gateway_ctx(self) -> None:
        script_text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('GATEWAY_CONFIG_DEFAULT="${GATEWAY_CONFIG_DEFAULT:-${REPO_ROOT}/gateway_ctx/config.toml}"', script_text)
        self.assertIn('GATEWAY_CONFIG_FALLBACK="${GATEWAY_CONFIG_FALLBACK:-${REPO_ROOT}/gateway_ctx/config.example.toml}"', script_text)
        self.assertIn("-m gateway_ctx", script_text)
        self.assertIn('echo "Launching gateway-ctx profile=${profile_id} raw=${gateway_port} parsed=${gateway_parse_port}"', script_text)

    def test_readme_documents_mi3008x_stack(self) -> None:
        readme_text = README.read_text(encoding="utf-8")

        self.assertIn("one shared AMD SMI power daemon for `amd-power-reader`", readme_text)
        self.assertIn("one TP=1 vLLM per port profile `0..7`, pinned to GPU `0..7`", readme_text)
        self.assertIn("one ctx-aware gateway (`gateway_ctx`) per port profile `0..7`", readme_text)
        self.assertIn("pip install -e ./amd-power-reader", readme_text)
        self.assertIn("pip install -e ./gateway_ctx", readme_text)
        self.assertIn("amd-power-reader --socket-path", readme_text)


if __name__ == "__main__":
    unittest.main()
