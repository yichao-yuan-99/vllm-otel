#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Jaeger service-index smoke test."""

from __future__ import annotations

import time
from typing import Any

from common import DEFAULT_POLL_INTERVAL_SECONDS, http_json


def run(
    *,
    jaeger_api_port: int,
    service_name: str,
    wait_seconds: float,
    request_timeout_seconds: float,
) -> tuple[bool, dict[str, Any]]:
    url = f"http://127.0.0.1:{jaeger_api_port}/api/services"
    started = time.monotonic()
    deadline = started + max(wait_seconds, 0.0)
    attempts = 0
    last_result: dict[str, Any] | None = None
    while True:
        attempts += 1
        ok, result = http_json("GET", url, timeout_seconds=request_timeout_seconds)
        last_result = result
        if ok:
            body = result.get("body")
            data = body.get("data") if isinstance(body, dict) else None
            if isinstance(data, list) and service_name in data:
                return True, {
                    "attempts": attempts,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "service_name": service_name,
                    "services_count": len(data),
                }
        if time.monotonic() >= deadline:
            return False, {
                "attempts": attempts,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "service_name": service_name,
                "last_result": last_result,
            }
        time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)
