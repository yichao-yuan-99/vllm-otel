#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AMD HPC vLLM arg normalization."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from control_plane import (  # type: ignore[import-not-found]
    _effective_vllm_extra_args,
    _normalize_user_extra_vllm_args,
)


class EffectiveVllmExtraArgsTest(unittest.TestCase):
    def test_single_gpu_always_adds_trust_remote_code(self) -> None:
        args = _effective_vllm_extra_args(extra_args=["--max-model-len", "32768"], gpus_per_node=1)
        self.assertEqual(args, ["--max-model-len", "32768", "--trust-remote-code"])

    def test_multi_gpu_adds_ray_and_trust_remote_code(self) -> None:
        args = _effective_vllm_extra_args(extra_args=[], gpus_per_node=4)
        self.assertEqual(args, ["--trust-remote-code", "--distributed_executor_backend", "ray"])

    def test_existing_conflicting_flags_are_replaced(self) -> None:
        args = _effective_vllm_extra_args(
            extra_args=[
                "--trust-remote-code=false",
                "--distributed_executor_backend",
                "mp",
                "--gpu-memory-utilization",
                "0.9",
            ],
            gpus_per_node=8,
        )
        self.assertEqual(
            args,
            [
                "--gpu-memory-utilization",
                "0.9",
                "--trust-remote-code",
                "--distributed_executor_backend",
                "ray",
            ],
        )


class NormalizeUserExtraVllmArgsTest(unittest.TestCase):
    def test_missing_or_empty_returns_empty(self) -> None:
        self.assertEqual(_normalize_user_extra_vllm_args(None), [])
        self.assertEqual(_normalize_user_extra_vllm_args([]), [])

    def test_preserves_valid_values(self) -> None:
        self.assertEqual(
            _normalize_user_extra_vllm_args(["--enable-expert-parallel", "--max-model-len", "32768"]),
            ["--enable-expert-parallel", "--max-model-len", "32768"],
        )

    def test_rejects_empty_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            _normalize_user_extra_vllm_args(["--ok", "   "])

    def test_rejects_nul_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "NUL"):
            _normalize_user_extra_vllm_args(["--ok", "bad\x00arg"])

    def test_rejects_non_list_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "list of strings"):
            _normalize_user_extra_vllm_args("--enable-expert-parallel")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
