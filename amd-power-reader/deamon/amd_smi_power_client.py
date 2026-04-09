#!/usr/bin/env python3
"""CLI wrapper for the AMD SMI power client."""

from __future__ import annotations

import sys

try:
    from .amdsmi_power_service import client_main
except ImportError:  # pragma: no cover - direct script execution path
    from amdsmi_power_service import client_main


if __name__ == "__main__":
    sys.exit(client_main())
