#!/usr/bin/env python3
"""Smoke tests for the mi3001x embedded TP1 launcher script/docs."""

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
        self.assertIn('AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.${RUN_ID}.sock}"', script_text)
        self.assertIn('command -v "${AMD_SMI_POWER_DAEMON_BIN}"', script_text)
        self.assertIn("probe_unix_socket()", script_text)
        self.assertIn("start_amd_smi_power_daemon()", script_text)
        self.assertIn('terminate_process "amd-smi-power-daemon" "${AMD_SMI_POWER_DAEMON_PID:-}"', script_text)
        self.assertIn('echo "  amd-smi-power-socket: ${AMD_SMI_POWER_SOCKET_PATH}"', script_text)
        self.assertIn("start_amd_smi_power_daemon\nstart_shared_jaeger", script_text)

    def test_start_services_launches_gateway_and_experiment(self) -> None:
        script_text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('GATEWAY_CONFIG_DEFAULT="${GATEWAY_CONFIG_DEFAULT:-${REPO_ROOT}/gateway_ctx/config.toml}"', script_text)
        self.assertIn("-m gateway_ctx", script_text)
        self.assertIn('echo "Launching gateway-ctx profile=${PROFILE_ID} raw=${GATEWAY_PORT} parsed=${GATEWAY_PARSE_PORT}"', script_text)
        self.assertIn('launch_experiment()', script_text)
        self.assertIn('export PORT_PROFILE_ID="${PROFILE_ID}"', script_text)
        self.assertIn('export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}"', script_text)
        self.assertIn('export GATEWAY_BASE_URL="http://127.0.0.1:${GATEWAY_PORT}"', script_text)
        self.assertIn('export AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH}"', script_text)
        self.assertIn('wait_for_experiment_phase()', script_text)
        self.assertIn('GATEWAY_LOG="${JOB_LOG_DIR}/gateway-ctx.${RUN_ID}.p${PROFILE_ID}.log"', script_text)
        self.assertIn('echo "  gateway-ctx: http://${GATEWAY_HOST}:${GATEWAY_PORT}"', script_text)
        self.assertIn('echo "  gateway-ctx-parse: http://${GATEWAY_HOST}:${GATEWAY_PARSE_PORT}"', script_text)

    def test_start_services_supports_forwarding_extra_vllm_env(self) -> None:
        script_text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('VLLM_EXTRA_ENV_B64="${VLLM_EXTRA_ENV_B64:-}"', script_text)
        self.assertIn("load_vllm_extra_env_args()", script_text)
        self.assertIn('error: invalid VLLM_EXTRA_ENV_B64 payload', script_text)
        self.assertIn('"${VLLM_EXTRA_ENV_ARGS[@]}"', script_text)

    def test_readme_documents_embedded_batch_workflow(self) -> None:
        readme_text = README.read_text(encoding="utf-8")

        self.assertIn("one shared AMD SMI power daemon for `amd-power-reader`", readme_text)
        self.assertIn("one ctx-aware gateway (`gateway_ctx`) on port profile `0`", readme_text)
        self.assertIn("one experiment-script invocation after the services are ready", readme_text)
        self.assertIn("python3 servers/servers-amdhpc-mi3001x-embedded-TP1/launch.py render", readme_text)
        self.assertIn("python3 servers/servers-amdhpc-mi3001x-embedded-TP1/launch.py submit", readme_text)
        self.assertIn("pip install -e ./amd-power-reader", readme_text)
        self.assertIn("pip install -e ./gateway_ctx", readme_text)
        self.assertIn("gateway-ctx: `11457`", readme_text)


if __name__ == "__main__":
    unittest.main()
