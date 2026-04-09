#!/usr/bin/env python3
"""Smoke tests for the shared interactive mi3008x experiment env helper."""

from __future__ import annotations

from pathlib import Path
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
HELPER_SCRIPT = MODULE_ROOT / "experiment-env.sh"


class ExperimentEnvScriptTest(unittest.TestCase):
    def test_helper_defines_shared_interactive_functions(self) -> None:
        script_text = HELPER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('INTERACTIVE_CLIENT_SCRIPT="${INTERACTIVE_CLIENT_SCRIPT:-servers/servers-amdhpc-interactive-mi3008x-embedded-TP1/client.py}"', script_text)
        self.assertIn('INTERACTIVE_START_SERVICES_SCRIPT="${INTERACTIVE_START_SERVICES_SCRIPT:-servers/servers-amdhpc-interactive-mi3008x-embedded-TP1/start-services.sh}"', script_text)
        self.assertIn('INTERACTIVE_START_SERVICES_COMMAND="${INTERACTIVE_START_SERVICES_COMMAND:-python3 ${INTERACTIVE_CLIENT_SCRIPT} start}"', script_text)
        self.assertIn("interactive_embedded_tp1_normalize_port_profile_id()", script_text)
        self.assertIn("interactive_embedded_tp1_normalize_gpu_index()", script_text)
        self.assertIn("interactive_embedded_tp1_resolve_profile_ports()", script_text)
        self.assertIn("interactive_embedded_tp1_wait_for_services()", script_text)
        self.assertIn("interactive_embedded_tp1_resolve_gateway_base_url()", script_text)
        self.assertIn("supported mi3008x port profile ids are 0..7", script_text)
        self.assertIn("supported mi3008x gpu indices are 0..7", script_text)
        self.assertIn('configs/port_profiles.toml', script_text)


if __name__ == "__main__":
    unittest.main()
