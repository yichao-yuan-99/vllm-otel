#!/usr/bin/env python3
"""Smoke tests for the mi2508x embedded TP2 launcher script/docs."""

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

    def test_start_services_unrolls_all_mi2508x_tp2_profiles(self) -> None:
        script_text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("PROFILE_IDS=(0 1 2 3)", script_text)
        self.assertIn("VLLM_TENSOR_PARALLEL_SIZE=2", script_text)
        self.assertIn("declare -A PROFILE_GPU_IDS=(", script_text)
        self.assertIn('local vllm_hip_visible_devices="$(compute_hip_visible_devices "${rocr_visible_devices}")"', script_text)
        self.assertIn('--env HIP_VISIBLE_DEVICES="${vllm_hip_visible_devices}"', script_text)
        for profile_id in range(4):
            self.assertIn(f'launch_vllm {profile_id} "${{PROFILE_VLLM_PORTS[{profile_id}]}}"', script_text)
            self.assertIn(f'wait_for_vllm_ready {profile_id} "${{PROFILE_VLLM_PORTS[{profile_id}]}}"', script_text)
            self.assertIn(f'launch_gateway {profile_id} "${{PROFILE_GATEWAY_PORTS[{profile_id}]}}"', script_text)
            self.assertIn(f'wait_for_gateway_ready {profile_id} "${{PROFILE_GATEWAY_PORTS[{profile_id}]}}"', script_text)
            self.assertIn(f"launch_experiment {profile_id}", script_text)
        self.assertIn("--env ROCR_VISIBLE_DEVICES", script_text)
        self.assertIn('echo "  profile ${profile_id}: gpus=${PROFILE_GPU_IDS[$profile_id]}', script_text)
        self.assertIn('gateway-ctx=http://${GATEWAY_HOST}:${PROFILE_GATEWAY_PORTS[$profile_id]}', script_text)
        self.assertIn('gateway-ctx-parse=http://${GATEWAY_HOST}:${PROFILE_GATEWAY_PARSE_PORTS[$profile_id]}', script_text)

    def test_start_services_launches_gateway_and_experiments(self) -> None:
        script_text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('GATEWAY_CONFIG_DEFAULT="${GATEWAY_CONFIG_DEFAULT:-${REPO_ROOT}/gateway_ctx/config.toml}"', script_text)
        self.assertIn("-m gateway_ctx", script_text)
        self.assertIn('echo "Launching gateway-ctx profile=${profile_id} raw=${gateway_port} parsed=${gateway_parse_port}"', script_text)
        self.assertIn('local gateway_log="${JOB_LOG_DIR}/gateway-ctx.${RUN_ID}.p${profile_id}.log"', script_text)
        self.assertIn('launch_experiment()', script_text)
        self.assertIn('export PORT_PROFILE_ID="${profile_id}"', script_text)
        self.assertIn('export VLLM_BASE_URL="http://127.0.0.1:${PROFILE_VLLM_PORTS[$profile_id]}"', script_text)
        self.assertIn('export GATEWAY_BASE_URL="http://127.0.0.1:${PROFILE_GATEWAY_PORTS[$profile_id]}"', script_text)
        self.assertIn('export AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH}"', script_text)
        self.assertIn('wait_for_experiment_phase()', script_text)
        self.assertIn("if wait_for_experiment_phase; then", script_text)
        self.assertNotIn("if ! wait_for_experiment_phase; then", script_text)

    def test_start_services_supports_forwarding_extra_vllm_env(self) -> None:
        script_text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('VLLM_EXTRA_ENV_B64="${VLLM_EXTRA_ENV_B64:-}"', script_text)
        self.assertIn("load_vllm_extra_env_args()", script_text)
        self.assertIn('error: invalid VLLM_EXTRA_ENV_B64 payload', script_text)
        self.assertIn('"${VLLM_EXTRA_ENV_ARGS[@]}"', script_text)

    def test_readme_documents_embedded_batch_workflow(self) -> None:
        readme_text = README.read_text(encoding="utf-8")

        self.assertIn("one shared AMD SMI power daemon for `amd-power-reader`", readme_text)
        self.assertIn("one TP=2 vLLM per port profile `0..3`, pinned to GPU pairs `0,1`, `2,3`,", readme_text)
        self.assertIn("one ctx-aware gateway (`gateway_ctx`) per port profile `0..3`", readme_text)
        self.assertIn("one experiment-script invocation per port profile after the services are ready", readme_text)
        self.assertIn("python3 servers/servers-amdhpc-mi2508x-embedded-TP2/launch.py render", readme_text)
        self.assertIn("python3 servers/servers-amdhpc-mi2508x-embedded-TP2/launch.py submit", readme_text)
        self.assertIn("pip install -e ./amd-power-reader", readme_text)
        self.assertIn("pip install -e ./gateway_ctx", readme_text)
        self.assertIn("gateway-ctx `11457`", readme_text)


if __name__ == "__main__":
    unittest.main()
