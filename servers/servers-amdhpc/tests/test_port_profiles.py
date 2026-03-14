#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AMD HPC port profile parsing."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from port_profiles import load_port_profile, load_port_profiles  # type: ignore[import-not-found]


class PortProfilesTest(unittest.TestCase):
    def test_single_profile_has_lmcache_port(self) -> None:
        profile = load_port_profile(0)
        self.assertIsInstance(profile.lmcache_port, int)
        self.assertGreaterEqual(profile.lmcache_port, 1)
        self.assertLessEqual(profile.lmcache_port, 65535)

    def test_all_profiles_have_unique_lmcache_port(self) -> None:
        profiles = load_port_profiles()
        lmcache_ports = [profile.lmcache_port for profile in profiles.values()]
        self.assertEqual(len(lmcache_ports), len(set(lmcache_ports)))


if __name__ == "__main__":
    unittest.main()
