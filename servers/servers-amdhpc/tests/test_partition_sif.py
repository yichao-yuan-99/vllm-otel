#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for partition-level vLLM SIF override behavior."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from control_plane import (  # type: ignore[import-not-found]
    ControlPlaneError,
    _effective_partition_vllm_sif,
    load_runtime_config,
)


def _write_minimal_server_config(path: Path, *, partition_body: str) -> None:
    path.write_text(
        (
            "[server]\n"
            "host = \"127.0.0.1\"\n"
            "port = 23971\n"
            "\n"
            "[cluster]\n"
            "login_host = \"login\"\n"
            "\n"
            "[partition.test]\n"
            f"{partition_body}\n"
        ),
        encoding="utf-8",
    )


class PartitionSifOverrideTest(unittest.TestCase):
    def test_effective_partition_vllm_sif_prefers_partition_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            default_sif = (tmp_root / "default.sif").resolve()
            partition_sif = (tmp_root / "partition.sif").resolve()
            default_sif.write_text("", encoding="utf-8")
            partition_sif.write_text("", encoding="utf-8")
            selected = _effective_partition_vllm_sif(
                partition_vllm_sif=partition_sif,
                default_vllm_sif=default_sif,
            )
            self.assertEqual(selected, partition_sif)

    def test_effective_partition_vllm_sif_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            default_sif = (tmp_root / "default.sif").resolve()
            partition_sif = (tmp_root / "partition.sif").resolve()
            default_sif.write_text("", encoding="utf-8")
            selected = _effective_partition_vllm_sif(
                partition_vllm_sif=partition_sif,
                default_vllm_sif=default_sif,
            )
            self.assertEqual(selected, default_sif)

    def test_load_runtime_config_reads_partition_sif_img(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            partition_sif = (tmp_root / "partition-vllm.sif").resolve()
            partition_sif.write_text("", encoding="utf-8")
            config_path = (tmp_root / "server_config.toml").resolve()
            _write_minimal_server_config(
                config_path,
                partition_body=(
                    "gpus_per_node = 1\n"
                    "gpu_memory_gb = 192\n"
                    "total_vram_gb = 192\n"
                    "max_time = \"01:00:00\"\n"
                    f"sif_img = \"{partition_sif}\"\n"
                ),
            )

            runtime_config = load_runtime_config(config_path)
            self.assertEqual(runtime_config.partitions["test"].vllm_sif, partition_sif)

    def test_load_runtime_config_rejects_empty_partition_sif_img(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = (tmp_root / "server_config.toml").resolve()
            _write_minimal_server_config(
                config_path,
                partition_body=(
                    "gpus_per_node = 1\n"
                    "gpu_memory_gb = 192\n"
                    "total_vram_gb = 192\n"
                    "max_time = \"01:00:00\"\n"
                    "sif_img = \"   \"\n"
                ),
            )

            with self.assertRaisesRegex(
                ControlPlaneError,
                "partition\\.test\\.sif_img must be non-empty",
            ):
                load_runtime_config(config_path)


if __name__ == "__main__":
    unittest.main()
