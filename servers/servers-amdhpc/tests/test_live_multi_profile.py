#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live AMD HPC integration test for concurrent multi-profile operation.

This test launches real cluster jobs. It is skipped unless
`RUN_AMDHPC_LIVE_TESTS=1` is set in the environment.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import tomllib
import unittest
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = MODULE_ROOT / "client.py"
PORT_PROFILES_PATH = REPO_ROOT / "configs" / "port_profiles.toml"

LIVE_TEST_ENV = "RUN_AMDHPC_LIVE_TESTS"

SSH_TARGET = os.environ.get("AMDHPC_SSH_TARGET", "amd-hpc")
PORT_PROFILES = (5, 6, 7, 8, 9)
PARTITION = "mi2104x"
MODEL_KEY = "qwen3_coder_30b"

START_TIMEOUT_SECONDS = float(os.environ.get("AMDHPC_START_TIMEOUT_SECONDS", "10800"))
STOP_TIMEOUT_SECONDS = float(os.environ.get("AMDHPC_STOP_TIMEOUT_SECONDS", "1800"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("AMDHPC_REQUEST_TIMEOUT_SECONDS", "120"))
MODEL_READY_TIMEOUT_SECONDS = float(os.environ.get("AMDHPC_MODEL_READY_TIMEOUT_SECONDS", "120"))
QUERY_DURATION_SECONDS = float(os.environ.get("AMDHPC_QUERY_DURATION_SECONDS", "120"))


def _load_port_profile(profile_id: int) -> dict[str, Any]:
    payload = tomllib.loads(PORT_PROFILES_PATH.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"missing [profiles] in {PORT_PROFILES_PATH}")
    raw = profiles.get(str(profile_id))
    if not isinstance(raw, dict):
        raise ValueError(f"unknown port profile: {profile_id}")
    return {
        "profile_id": profile_id,
        "label": raw.get("label", str(profile_id)),
        "vllm_port": int(raw["vllm_port"]),
        "jaeger_api_port": int(raw["jaeger_api_port"]),
        "jaeger_otlp_port": int(raw["jaeger_otlp_port"]),
    }


def _http_json(
    method: str,
    url: str,
    *,
    payload_obj: dict[str, Any] | None,
    timeout_seconds: float,
) -> tuple[bool, dict[str, Any]]:
    body: bytes | None = None
    headers: dict[str, str] = {}
    if payload_obj is not None:
        body = json.dumps(payload_obj).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urlrequest.Request(url, method=method, data=body, headers=headers)
    started = time.monotonic()
    try:
        with urlrequest.urlopen(request, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(text) if text else {}
            return True, {
                "status_code": int(response.status),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "body": parsed,
            }
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        parsed_error: Any = text
        if text:
            try:
                parsed_error = json.loads(text)
            except json.JSONDecodeError:
                parsed_error = text
        return False, {
            "status_code": int(exc.code),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": parsed_error,
        }
    except Exception as exc:  # pragma: no cover - live network boundary
        return False, {
            "status_code": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": str(exc),
        }


def _extract_last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index in range(len(text) - 1, -1, -1):
        if text[index] != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if text[index + end :].strip():
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("could not locate trailing JSON payload in command output")


def _run_client_command(args: list[str], *, timeout_seconds: float) -> dict[str, Any]:
    command = ["python3", str(CLIENT_PATH), *args]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "payload": None,
            "error": f"command timed out after {timeout_seconds} seconds",
        }

    payload: dict[str, Any] | None = None
    parse_error: str | None = None
    if completed.stdout.strip():
        try:
            payload = _extract_last_json_object(completed.stdout)
        except ValueError as exc:
            parse_error = str(exc)

    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "payload": payload,
        "error": parse_error,
    }


def _run_profile_start(profile_id: int) -> dict[str, Any]:
    return _run_client_command(
        [
            "start",
            "--ssh-target",
            SSH_TARGET,
            "-P",
            str(profile_id),
            "-p",
            PARTITION,
            "-m",
            MODEL_KEY,
            "-b",
        ],
        timeout_seconds=START_TIMEOUT_SECONDS,
    )


def _run_profile_stop(profile_id: int) -> dict[str, Any]:
    return _run_client_command(
        ["stop", "-P", str(profile_id)],
        timeout_seconds=STOP_TIMEOUT_SECONDS,
    )


def _probe_tcp_port(port: int, *, timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_seconds):
            return {
                "ok": True,
                "port": port,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
    except Exception as exc:  # pragma: no cover - live network boundary
        return {
            "ok": False,
            "port": port,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": str(exc),
        }


def _wait_for_models(vllm_port: int, *, timeout_seconds: float) -> dict[str, Any]:
    url = f"http://127.0.0.1:{vllm_port}/v1/models"
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    last_result: dict[str, Any] | None = None
    while True:
        attempts += 1
        ok, result = _http_json("GET", url, payload_obj=None, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
        last_result = result
        if ok:
            return {
                "ok": True,
                "attempts": attempts,
                "result": result,
            }
        if time.monotonic() >= deadline:
            return {
                "ok": False,
                "attempts": attempts,
                "last_result": last_result,
            }
        time.sleep(2.0)


def _extract_model_name(models_response: dict[str, Any]) -> str:
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
        raise ValueError("vLLM /v1/models first item has invalid id")
    return model_name


def _run_chat_completion(vllm_port: int, model_name: str) -> tuple[bool, dict[str, Any]]:
    return _http_json(
        "POST",
        f"http://127.0.0.1:{vllm_port}/v1/chat/completions",
        payload_obj={
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with the word pong."}],
            "chat_template_kwargs": {"thinking": False},
            "max_tokens": 16,
            "temperature": 0,
        },
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )


def _check_profile_health(profile_id: int) -> dict[str, Any]:
    up_result = _run_client_command(["up", "-P", str(profile_id)], timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    payload = up_result.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"profile {profile_id} up command returned no JSON payload: {up_result}")
    if up_result.get("returncode") != 0 or not payload.get("ok"):
        raise RuntimeError(f"profile {profile_id} up command failed: {json.dumps(up_result, indent=2)}")

    ready_data = payload.get("data")
    if not isinstance(ready_data, dict) or not ready_data.get("ready"):
        raise RuntimeError(f"profile {profile_id} is not ready: {json.dumps(up_result, indent=2)}")

    ports = _load_port_profile(profile_id)
    otlp_probe = _probe_tcp_port(ports["jaeger_otlp_port"], timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    if not otlp_probe.get("ok"):
        raise RuntimeError(f"profile {profile_id} OTLP port probe failed: {json.dumps(otlp_probe, indent=2)}")

    models_probe = _wait_for_models(ports["vllm_port"], timeout_seconds=MODEL_READY_TIMEOUT_SECONDS)
    if not models_probe.get("ok"):
        raise RuntimeError(f"profile {profile_id} /v1/models is not ready: {json.dumps(models_probe, indent=2)}")

    model_name = _extract_model_name(models_probe["result"])
    return {
        "profile_id": profile_id,
        "ports": ports,
        "up_payload": payload,
        "otlp_probe": otlp_probe,
        "models_probe": models_probe,
        "model_name": model_name,
    }


def _query_profile_for_duration(
    profile_id: int,
    model_name: str,
    duration_seconds: float,
) -> dict[str, Any]:
    ports = _load_port_profile(profile_id)
    deadline = time.monotonic() + duration_seconds
    success_count = 0
    attempts = 0
    last_response: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        attempts += 1
        ok, result = _run_chat_completion(ports["vllm_port"], model_name)
        if not ok:
            raise RuntimeError(
                f"profile {profile_id} inference request failed after {success_count} successes: "
                f"{json.dumps(result, indent=2)}"
            )
        body = result.get("body")
        if not isinstance(body, dict):
            raise RuntimeError(f"profile {profile_id} returned non-object completion body: {json.dumps(result, indent=2)}")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"profile {profile_id} returned no choices: {json.dumps(result, indent=2)}")
        success_count += 1
        last_response = result

    return {
        "profile_id": profile_id,
        "duration_seconds": duration_seconds,
        "attempts": attempts,
        "success_count": success_count,
        "last_response": last_response,
    }


def _collect_parallel_results(
    func: Any,
    items: tuple[int, ...] | list[int],
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as executor:
        future_to_item = {executor.submit(func, item): item for item in items}
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            results[item] = future.result()
    return results


@unittest.skipUnless(
    os.environ.get(LIVE_TEST_ENV) == "1",
    (
        f"set {LIVE_TEST_ENV}=1 to run this live AMD HPC integration test; "
        "it assumes the control server is already running and uses ssh target amd-hpc"
    ),
)
class LiveMultiProfileIntegrationTest(unittest.TestCase):
    maxDiff = None

    def test_profiles_5_to_9_can_start_query_and_stop_concurrently(self) -> None:
        self.assertTrue(CLIENT_PATH.exists(), f"missing client CLI: {CLIENT_PATH}")

        started_profiles: list[int] = []
        cleanup_failures: dict[int, dict[str, Any]] = {}
        body_error: tuple[type[BaseException], BaseException, Any] | None = None

        try:
            start_results = _collect_parallel_results(_run_profile_start, list(PORT_PROFILES))
            start_failures = {
                profile_id: result
                for profile_id, result in start_results.items()
                if result.get("returncode") != 0
                or not isinstance(result.get("payload"), dict)
                or not result["payload"].get("ok")
            }
            started_profiles = sorted(
                profile_id
                for profile_id, result in start_results.items()
                if profile_id not in start_failures
            )
            self.assertFalse(
                start_failures,
                "concurrent start failed:\n" + json.dumps(start_failures, indent=2, sort_keys=True),
            )

            health_results = _collect_parallel_results(_check_profile_health, list(PORT_PROFILES))
            model_names = {
                profile_id: str(result["model_name"])
                for profile_id, result in health_results.items()
            }

            query_results: dict[int, dict[str, Any]] = {}
            query_failures: dict[int, str] = {}
            mp_context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(PORT_PROFILES),
                mp_context=mp_context,
            ) as executor:
                future_to_profile = {
                    executor.submit(
                        _query_profile_for_duration,
                        profile_id,
                        model_names[profile_id],
                        QUERY_DURATION_SECONDS,
                    ): profile_id
                    for profile_id in PORT_PROFILES
                }
                for future in concurrent.futures.as_completed(future_to_profile):
                    profile_id = future_to_profile[future]
                    try:
                        query_results[profile_id] = future.result()
                    except Exception as exc:  # pragma: no cover - worker boundary
                        query_failures[profile_id] = str(exc)

            self.assertFalse(
                query_failures,
                "concurrent query phase failed:\n" + json.dumps(query_failures, indent=2, sort_keys=True),
            )
            for profile_id, result in sorted(query_results.items()):
                self.assertGreater(
                    int(result["success_count"]),
                    0,
                    f"profile {profile_id} did not complete any successful requests",
                )
        except Exception:  # noqa: BLE001
            body_error = sys.exc_info()
        finally:
            if started_profiles:
                stop_results = _collect_parallel_results(_run_profile_stop, started_profiles)
                cleanup_failures = {
                    profile_id: result
                    for profile_id, result in stop_results.items()
                    if result.get("returncode") != 0
                    or not isinstance(result.get("payload"), dict)
                    or not result["payload"].get("ok")
                }

        if body_error is not None:
            if cleanup_failures:
                print(
                    "cleanup failures after test body error:\n"
                    + json.dumps(cleanup_failures, indent=2, sort_keys=True),
                    file=sys.stderr,
                )
            raise body_error[1].with_traceback(body_error[2])

        self.assertFalse(
            cleanup_failures,
            "graceful stop failed:\n" + json.dumps(cleanup_failures, indent=2, sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
