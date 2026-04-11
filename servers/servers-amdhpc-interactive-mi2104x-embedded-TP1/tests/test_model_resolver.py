#!/usr/bin/env python3
"""Unit tests for the interactive embedded TP1 model resolver."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from model_resolver import resolve_model_launch_config  # type: ignore[import-not-found]


def _decode_extra_args(encoded: str) -> list[str]:
    payload = base64.b64decode(encoded.encode("ascii")).decode("utf-8")
    parsed = json.loads(payload)
    return [str(item) for item in parsed]


class ModelResolverTest(unittest.TestCase):
    def _write_model_config(self, root: Path) -> Path:
        config_path = root / "model_config.toml"
        config_path.write_text(
            textwrap.dedent(
                """\
                schema_version = 1
                default_model = "qwen3_14b"

                [models.qwen3_14b]
                vllm_model_name = "Qwen/Qwen3-14B-FP8"
                served_model_name = "Qwen3-14B-FP8"
                extra_args = []

                [models.minimax_m_2_5]
                vllm_model_name = "MiniMaxAI/MiniMax-M2.5"
                served_model_name = "MiniMax-M2.5"
                extra_args = [
                  "--trust-remote-code",
                  "--distributed_executor_backend", "ray",
                  "--reasoning-parser", "minimax_m2"
                ]
                """
            ),
            encoding="utf-8",
        )
        return config_path

    def test_resolves_config_key_to_launch_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_model_config(Path(tmpdir))

            resolved = resolve_model_launch_config(
                config_path=config_path,
                selector="qwen3_14b",
            )

            self.assertEqual(resolved.resolved_model_key, "qwen3_14b")
            self.assertEqual(
                resolved.vllm_model_name,
                "Qwen/Qwen3-14B-FP8",
            )
            self.assertEqual(
                resolved.served_model_name,
                "Qwen3-14B-FP8",
            )
            self.assertEqual(
                _decode_extra_args(resolved.model_extra_args_b64),
                ["--trust-remote-code"],
            )

    def test_resolves_alias_and_normalizes_tp1_extra_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_model_config(Path(tmpdir))

            resolved = resolve_model_launch_config(
                config_path=config_path,
                selector="MiniMaxAI/MiniMax-M2.5",
            )

            self.assertEqual(resolved.resolved_model_key, "minimax_m_2_5")
            self.assertEqual(
                _decode_extra_args(resolved.model_extra_args_b64),
                ["--reasoning-parser", "minimax_m2", "--trust-remote-code"],
            )

    def test_preserves_explicit_served_name_and_extra_args_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_model_config(Path(tmpdir))
            explicit_extra_args = base64.b64encode(b"[\"--foo\"]").decode("ascii")

            resolved = resolve_model_launch_config(
                config_path=config_path,
                selector="qwen3_14b",
                served_model_name="custom-served-name",
                extra_args_b64=explicit_extra_args,
            )

            self.assertEqual(resolved.served_model_name, "custom-served-name")
            self.assertEqual(resolved.model_extra_args_b64, explicit_extra_args)

    def test_requires_served_name_for_raw_model_not_in_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_model_config(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "VLLM_SERVED_MODEL_NAME"):
                resolve_model_launch_config(
                    config_path=config_path,
                    selector="org/custom-model",
                )

    def test_allows_raw_model_when_served_name_is_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_model_config(Path(tmpdir))

            resolved = resolve_model_launch_config(
                config_path=config_path,
                selector="org/custom-model",
                served_model_name="Custom-Served-Name",
            )

            self.assertEqual(resolved.resolved_model_key, "")
            self.assertEqual(resolved.vllm_model_name, "org/custom-model")
            self.assertEqual(resolved.served_model_name, "Custom-Served-Name")
            self.assertEqual(
                _decode_extra_args(resolved.model_extra_args_b64),
                ["--trust-remote-code"],
            )


if __name__ == "__main__":
    unittest.main()
