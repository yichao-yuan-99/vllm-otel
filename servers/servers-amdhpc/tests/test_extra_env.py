#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for start/start-group extra env parsing and rendering."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from control_plane import (  # type: ignore[import-not-found]
    _apply_lmcache_option,
    _normalize_service_extra_env,
    _render_apptainer_extra_env_flags,
)


class NormalizeServiceExtraEnvTest(unittest.TestCase):
    def test_normalize_and_sort(self) -> None:
        normalized = _normalize_service_extra_env(
            {"B_VAR": "2", "A_VAR": "1"},
        )
        self.assertEqual(list(normalized.keys()), ["A_VAR", "B_VAR"])
        self.assertEqual(normalized["A_VAR"], "1")
        self.assertEqual(normalized["B_VAR"], "2")

    def test_render_apptainer_flags(self) -> None:
        rendered = _render_apptainer_extra_env_flags(
            extra_env={
                "A_VAR": "1",
                "B_VAR": "hello world",
            },
            indent="    ",
        )
        self.assertIn("    --env A_VAR=1 \\\n", rendered)
        self.assertIn("    --env 'B_VAR=hello world' \\\n", rendered)

    def test_reject_invalid_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid environment variable key"):
            _normalize_service_extra_env({"BAD-KEY": "1"})

    def test_lmcache_option_disabled_when_not_provided(self) -> None:
        env, enabled = _apply_lmcache_option(
            extra_env={"B_VAR": "2", "A_VAR": "1"},
            lmcache_max_local_cpu_size=None,
        )
        self.assertFalse(enabled)
        self.assertEqual(env, {"A_VAR": "1", "B_VAR": "2"})

    def test_lmcache_option_sets_env_and_enables_connector(self) -> None:
        env, enabled = _apply_lmcache_option(
            extra_env={"A_VAR": "1"},
            lmcache_max_local_cpu_size="100",
        )
        self.assertTrue(enabled)
        self.assertEqual(
            env,
            {
                "A_VAR": "1",
                "LMCACHE_MAX_LOCAL_CPU_SIZE": "100",
            },
        )

    def test_lmcache_option_accepts_matching_existing_value(self) -> None:
        env, enabled = _apply_lmcache_option(
            extra_env={"LMCACHE_MAX_LOCAL_CPU_SIZE": "200"},
            lmcache_max_local_cpu_size="200",
        )
        self.assertTrue(enabled)
        self.assertEqual(env, {"LMCACHE_MAX_LOCAL_CPU_SIZE": "200"})

    def test_lmcache_option_rejects_conflicting_existing_value(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "already set in extra_env with a different value",
        ):
            _apply_lmcache_option(
                extra_env={"LMCACHE_MAX_LOCAL_CPU_SIZE": "200"},
                lmcache_max_local_cpu_size="100",
            )

    def test_lmcache_option_rejects_empty_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            _apply_lmcache_option(
                extra_env={},
                lmcache_max_local_cpu_size="   ",
            )


if __name__ == "__main__":
    unittest.main()
