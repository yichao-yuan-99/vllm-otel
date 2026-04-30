#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Docker image override handling."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import client  # type: ignore[import-not-found]


class DockerImageOverrideTest(unittest.TestCase):
    def test_validate_start_selection_uses_vllm_image_override(self) -> None:
        with (
            mock.patch.object(
                client,
                "_load_image_config",
                return_value={
                    "jaeger_image": "jaegertracing/all-in-one:latest",
                    "vllm_image_name": "example/default:latest",
                },
            ),
            mock.patch.object(
                client,
                "_load_models_config",
                return_value=(
                    None,
                    {
                        "test_model": {
                            "vllm_model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                            "served_model_name": "Qwen3-Coder-30B-A3B-Instruct",
                            "weight_vram_gb": 20.0,
                            "extra_args": [],
                        }
                    },
                ),
            ),
            mock.patch.object(
                client,
                "_load_port_profiles",
                return_value=(
                    None,
                    {
                        "0": {
                            "vllm_port": 11451,
                            "jaeger_api_port": 16686,
                            "jaeger_otlp_port": 4317,
                            "lmcache_port": 29411,
                        }
                    },
                ),
            ),
            mock.patch.object(
                client,
                "_load_launch_profiles",
                return_value=(
                    None,
                    {
                        "test_launch": {
                            "gpu_type": "h100",
                            "visible_devices": "0",
                            "visible_device_ids": ["0"],
                            "per_gpu_memory_gb": 80.0,
                            "total_gpu_memory_gb": 80.0,
                            "tensor_parallel_size": 1,
                        }
                    },
                ),
            ),
        ):
            valid, payload = client._validate_start_selection(
                model_key="test_model",
                port_profile_id=0,
                launch_profile_key="test_launch",
                vllm_image="yichaoyuan/vllm-vllm-openai:v0.19.0-otel-lp",
            )

        self.assertTrue(valid)
        selection = payload["data"]
        self.assertEqual(selection["images"]["jaeger_image"], "jaegertracing/all-in-one:latest")
        self.assertEqual(
            selection["images"]["vllm_image_name"],
            "yichaoyuan/vllm-vllm-openai:v0.19.0-otel-lp",
        )
        self.assertEqual(
            selection["vllm_image_override"],
            "yichaoyuan/vllm-vllm-openai:v0.19.0-otel-lp",
        )

    def test_validate_start_selection_rejects_blank_vllm_image_override(self) -> None:
        valid, payload = client._validate_start_selection(
            model_key="test_model",
            port_profile_id=0,
            launch_profile_key="test_launch",
            vllm_image="   ",
        )

        self.assertFalse(valid)
        self.assertEqual(payload["code"], 415)
        self.assertEqual(payload["message"], "--image must be a non-empty string")


if __name__ == "__main__":
    unittest.main()
