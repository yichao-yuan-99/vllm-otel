#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Jaeger OTLP reachability smoke test."""

from __future__ import annotations

import socket
import time
from typing import Any


def run(*, jaeger_otlp_port: int, timeout_seconds: float) -> tuple[bool, dict[str, Any]]:
    started = time.monotonic()
    try:
        with socket.create_connection(("127.0.0.1", jaeger_otlp_port), timeout=timeout_seconds):
            return True, {
                "host": "127.0.0.1",
                "port": jaeger_otlp_port,
                "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
            }
    except Exception as exc:  # pragma: no cover - network boundary
        return False, {
            "host": "127.0.0.1",
            "port": jaeger_otlp_port,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
            "error": str(exc),
        }
