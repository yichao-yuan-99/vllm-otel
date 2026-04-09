#!/usr/bin/env python3
"""CLI wrapper for the AMD SMI power daemon."""

from __future__ import annotations

import sys

try:
    from .amdsmi_power_service import daemon_main
except ImportError:  # pragma: no cover - direct script execution path
    from amdsmi_power_service import daemon_main


if __name__ == "__main__":
    sys.exit(daemon_main())
