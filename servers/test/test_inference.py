#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Basic inference smoke test."""

from __future__ import annotations

from typing import Any

from common import http_json


def run(
    *,
    vllm_port: int,
    model_name: str,
    request_timeout_seconds: float,
) -> tuple[bool, dict[str, Any]]:
    ok, result = http_json(
        "POST",
        f"http://127.0.0.1:{vllm_port}/v1/chat/completions",
        payload_obj={
            "model": model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "chat_template_kwargs": {"thinking": False},
            "max_tokens": 8,
            "temperature": 0,
        },
        timeout_seconds=request_timeout_seconds,
    )
    if not ok:
        return False, result

    body = result.get("body")
    if not isinstance(body, dict):
        return False, {"error": "chat completion response body is not an object", "response": result}
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, {"error": "chat completion returned no choices", "response": result}
    return True, result
