#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for grouped single-node GPU split logic."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from control_plane import _compute_even_group_gpu_split  # type: ignore[import-not-found]


class ComputeEvenGroupGpuSplitTest(unittest.TestCase):
    def test_four_profiles_on_eight_gpus_get_two_each(self) -> None:
        gpus_per_profile, total_vram_per_profile_gb, visible_devices_by_profile = (
            _compute_even_group_gpu_split(
                gpus_per_node=8,
                gpu_memory_gb=192.0,
                group_size=4,
            )
        )

        self.assertEqual(gpus_per_profile, 2)
        self.assertEqual(total_vram_per_profile_gb, 384.0)
        self.assertEqual(
            visible_devices_by_profile,
            ["0,1", "2,3", "4,5", "6,7"],
        )

    def test_eight_profiles_on_eight_gpus_get_one_each(self) -> None:
        gpus_per_profile, total_vram_per_profile_gb, visible_devices_by_profile = (
            _compute_even_group_gpu_split(
                gpus_per_node=8,
                gpu_memory_gb=192.0,
                group_size=8,
            )
        )

        self.assertEqual(gpus_per_profile, 1)
        self.assertEqual(total_vram_per_profile_gb, 192.0)
        self.assertEqual(
            visible_devices_by_profile,
            ["0", "1", "2", "3", "4", "5", "6", "7"],
        )

    def test_rejects_uneven_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not evenly divide"):
            _compute_even_group_gpu_split(
                gpus_per_node=8,
                gpu_memory_gb=192.0,
                group_size=3,
            )

    def test_rejects_group_larger_than_gpu_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds gpus_per_node"):
            _compute_even_group_gpu_split(
                gpus_per_node=4,
                gpu_memory_gb=64.0,
                group_size=5,
            )


if __name__ == "__main__":
    unittest.main()
