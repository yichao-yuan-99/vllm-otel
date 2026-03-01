#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for server smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
PORT_PROFILES_PATH = REPO_ROOT / "configs" / "port_profiles.toml"

DEFAULT_STARTUP_TIMEOUT_SECONDS = 600
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_TRACE_WAIT_SECONDS = 30
DEFAULT_POLL_INTERVAL_SECONDS = 2
DEFAULT_FORCE_SEQUENCE_TEST_TEXT = "FORCE_SEQ_SMOKE_OK"
EXPECTED_VLLM_SERVICE_NAME = "vllm-server"


def payload(*, ok: bool, code: int, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": ok,
        "code": code,
        "message": message,
        "data": data or {},
    }


def emit(payload_obj: dict[str, Any], *, fail_on_error: bool = True) -> int:
    print(json.dumps(payload_obj, indent=2, sort_keys=True))
    if fail_on_error and not payload_obj.get("ok"):
        return 1
    return 0


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _parse_port(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < 1 or value > 65535:
        raise ValueError(f"{key} must be in range 1..65535")
    return value


def load_port_profile(profile_id: int) -> tuple[str, dict[str, Any]]:
    payload_obj = _load_toml(PORT_PROFILES_PATH)
    raw_profiles = payload_obj.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise ValueError("configs/port_profiles.toml must include [profiles]")

    key = str(profile_id)
    raw = raw_profiles.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"unknown port profile id: {profile_id}")

    return key, {
        "label": raw.get("label"),
        "vllm_port": _parse_port(raw.get("vllm_port"), f"profiles.{key}.vllm_port"),
        "jaeger_api_port": _parse_port(raw.get("jaeger_api_port"), f"profiles.{key}.jaeger_api_port"),
        "jaeger_otlp_port": _parse_port(raw.get("jaeger_otlp_port"), f"profiles.{key}.jaeger_otlp_port"),
    }


def http_json(
    method: str,
    url: str,
    *,
    payload_obj: dict[str, Any] | None = None,
    timeout_seconds: float,
) -> tuple[bool, dict[str, Any]]:
    body: bytes | None = None
    headers: dict[str, str] = {}
    if payload_obj is not None:
        body = json.dumps(payload_obj).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urlrequest.Request(url, method=method, data=body, headers=headers)
    started = time.monotonic()
    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8", errors="replace")
            elapsed = round(time.monotonic() - started, 3)
            parsed: Any
            if text:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = {"raw": text}
            else:
                parsed = {}
            return True, {
                "status_code": int(response.status),
                "elapsed_seconds": elapsed,
                "body": parsed,
            }
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        elapsed = round(time.monotonic() - started, 3)
        parsed_error: Any = text
        if text:
            try:
                parsed_error = json.loads(text)
            except json.JSONDecodeError:
                parsed_error = text
        return False, {
            "status_code": int(exc.code),
            "elapsed_seconds": elapsed,
            "error": parsed_error,
        }
    except Exception as exc:  # pragma: no cover - network boundary
        elapsed = round(time.monotonic() - started, 3)
        return False, {
            "status_code": None,
            "elapsed_seconds": elapsed,
            "error": str(exc),
        }


def wait_for_vllm_models(
    *,
    vllm_port: int,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
) -> tuple[bool, dict[str, Any]]:
    url = f"http://127.0.0.1:{vllm_port}/v1/models"
    started = time.monotonic()
    deadline = started + max(startup_timeout_seconds, 0.0)
    attempts = 0
    last_result: dict[str, Any] | None = None
    while True:
        attempts += 1
        ok, result = http_json("GET", url, timeout_seconds=request_timeout_seconds)
        last_result = result
        if ok:
            data = {
                "attempts": attempts,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "result": result,
            }
            return True, data
        if time.monotonic() >= deadline:
            data = {
                "attempts": attempts,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "last_result": last_result,
            }
            return False, data
        time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)


def extract_model_name(models_response: dict[str, Any]) -> str:
    body = models_response.get("body")
    if not isinstance(body, dict):
        raise ValueError("vLLM /v1/models response body is not an object")
    data = body.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("vLLM /v1/models returned no models")
    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("vLLM /v1/models item is not an object")
    model_name = first.get("id")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("vLLM /v1/models first model has invalid id")
    return model_name
