#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Docker vLLM option wiring."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from client import (  # type: ignore[import-not-found]
    LMCACHE_KV_TRANSFER_CONFIG,
    _apply_gpu_memory_utilization_option,
    _apply_lmcache_option,
    _compose_env_values,
)


class DockerLMCacheOptionTest(unittest.TestCase):
    def test_lmcache_option_disabled_when_not_provided(self) -> None:
        args, enabled = _apply_lmcache_option(
            extra_args=["--trust-remote-code"],
            lmcache_max_local_cpu_size=None,
        )
        self.assertFalse(enabled)
        self.assertEqual(args, ["--trust-remote-code"])

    def test_lmcache_option_appends_kv_transfer_config(self) -> None:
        args, enabled = _apply_lmcache_option(
            extra_args=["--trust-remote-code"],
            lmcache_max_local_cpu_size=100,
        )
        self.assertTrue(enabled)
        self.assertEqual(
            args,
            [
                "--trust-remote-code",
                "--kv-transfer-config",
                LMCACHE_KV_TRANSFER_CONFIG,
            ],
        )

    def test_lmcache_option_accepts_matching_existing_kv_transfer_config(self) -> None:
        args, enabled = _apply_lmcache_option(
            extra_args=[
                "--kv-transfer-config",
                LMCACHE_KV_TRANSFER_CONFIG,
            ],
            lmcache_max_local_cpu_size=100,
        )
        self.assertTrue(enabled)
        self.assertEqual(
            args,
            [
                "--kv-transfer-config",
                LMCACHE_KV_TRANSFER_CONFIG,
            ],
        )

    def test_lmcache_option_rejects_conflicting_existing_kv_transfer_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "already configure --kv-transfer-config"):
            _apply_lmcache_option(
                extra_args=[
                    "--kv-transfer-config",
                    '{"kv_connector":"OtherConnector"}',
                ],
                lmcache_max_local_cpu_size=100,
            )

    def test_compose_env_values_include_lmcache_settings(self) -> None:
        selection = {
            "model_key": "test_model",
            "launch_profile_key": "test_launch",
            "port_profile_id": "0",
            "lmcache": 256,
            "images": {
                "jaeger_image": "jaegertracing/all-in-one:latest",
                "vllm_image_name": "example/vllm:latest",
            },
            "model": {
                "vllm_model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "served_model_name": "Qwen3-Coder-30B-A3B-Instruct",
                "extra_args": ["--trust-remote-code"],
            },
            "ports": {
                "vllm_port": 11451,
                "jaeger_api_port": 16686,
                "jaeger_otlp_port": 4317,
                "lmcache_port": 29411,
            },
            "launch": {
                "tensor_parallel_size": 2,
            },
            "runtime_names": {
                "compose_project_name": "vllm-otel-test",
                "jaeger_container_name": "jaeger-test",
                "vllm_container_name": "vllm-test",
                "otel_service_name": "vllm-server-test",
            },
        }

        env = _compose_env_values(selection)
        decoded_args = json.loads(base64.b64decode(env["VLLM_MODEL_EXTRA_ARGS_B64"]).decode("utf-8"))

        self.assertEqual(env["LMCACHE_INTERNAL_API_SERVER_ENABLED"], "1")
        self.assertEqual(env["LMCACHE_INTERNAL_API_SERVER_PORT_START"], "29411")
        self.assertEqual(env["LMCACHE_MAX_LOCAL_CPU_SIZE"], "256")
        self.assertEqual(env["PYTHONHASHSEED"], "0")
        self.assertEqual(
            decoded_args,
            [
                "--trust-remote-code",
                "--kv-transfer-config",
                LMCACHE_KV_TRANSFER_CONFIG,
            ],
        )


class DockerGpuMemoryUtilizationOptionTest(unittest.TestCase):
    def test_gpu_memory_utilization_disabled_when_not_provided(self) -> None:
        args, enabled = _apply_gpu_memory_utilization_option(
            extra_args=["--trust-remote-code"],
            gpu_memory_utilization=None,
        )
        self.assertFalse(enabled)
        self.assertEqual(args, ["--trust-remote-code"])

    def test_gpu_memory_utilization_appends_option(self) -> None:
        args, enabled = _apply_gpu_memory_utilization_option(
            extra_args=["--trust-remote-code"],
            gpu_memory_utilization=0.75,
        )
        self.assertTrue(enabled)
        self.assertEqual(
            args,
            [
                "--trust-remote-code",
                "--gpu-memory-utilization",
                "0.75",
            ],
        )

    def test_gpu_memory_utilization_replaces_existing_option(self) -> None:
        args, enabled = _apply_gpu_memory_utilization_option(
            extra_args=[
                "--trust-remote-code",
                "--gpu-memory-utilization",
                "0.9",
            ],
            gpu_memory_utilization=0.75,
        )
        self.assertTrue(enabled)
        self.assertEqual(
            args,
            [
                "--trust-remote-code",
                "--gpu-memory-utilization",
                "0.75",
            ],
        )

    def test_gpu_memory_utilization_rejects_out_of_range_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "gpu-memory-utilization"):
            _apply_gpu_memory_utilization_option(
                extra_args=["--trust-remote-code"],
                gpu_memory_utilization=1.5,
            )

    def test_compose_env_values_include_gpu_memory_utilization_arg(self) -> None:
        selection = {
            "model_key": "test_model",
            "launch_profile_key": "test_launch",
            "port_profile_id": "0",
            "gpu_memory_utilization": 0.75,
            "images": {
                "jaeger_image": "jaegertracing/all-in-one:latest",
                "vllm_image_name": "example/vllm:latest",
            },
            "model": {
                "vllm_model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "served_model_name": "Qwen3-Coder-30B-A3B-Instruct",
                "extra_args": ["--trust-remote-code"],
            },
            "ports": {
                "vllm_port": 11451,
                "jaeger_api_port": 16686,
                "jaeger_otlp_port": 4317,
                "lmcache_port": 29411,
            },
            "launch": {
                "tensor_parallel_size": 2,
            },
            "runtime_names": {
                "compose_project_name": "vllm-otel-test",
                "jaeger_container_name": "jaeger-test",
                "vllm_container_name": "vllm-test",
                "otel_service_name": "vllm-server-test",
            },
        }

        env = _compose_env_values(selection)
        decoded_args = json.loads(base64.b64decode(env["VLLM_MODEL_EXTRA_ARGS_B64"]).decode("utf-8"))

        self.assertEqual(
            decoded_args,
            [
                "--trust-remote-code",
                "--gpu-memory-utilization",
                "0.75",
            ],
        )


if __name__ == "__main__":
    unittest.main()
