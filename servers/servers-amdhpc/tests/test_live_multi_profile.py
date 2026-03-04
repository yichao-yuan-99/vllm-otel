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
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
import unittest
from dataclasses import dataclass
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
QUERY_PROGRESS_INTERVAL_SECONDS = float(os.environ.get("AMDHPC_QUERY_PROGRESS_INTERVAL_SECONDS", "30"))
COMMAND_HEARTBEAT_SECONDS = float(os.environ.get("AMDHPC_COMMAND_HEARTBEAT_SECONDS", "30"))

_ACTIVE_PROCESSES_LOCK = threading.Lock()
_ACTIVE_PROCESSES: dict[int, tuple[subprocess.Popen[str], int | None, str]] = {}


@dataclass
class PhaseInterruptedError(RuntimeError):
    phase_name: str
    partial_results: dict[int, dict[str, Any]]

    def __str__(self) -> str:
        return f"phase {self.phase_name} interrupted by user"


def _log(message: str, *, profile_id: int | None = None) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    prefix = f"[amdhpc-live-test {timestamp}]"
    if profile_id is not None:
        prefix = f"{prefix} [profile={profile_id}]"
    print(f"{prefix} {message}", flush=True)


def _register_active_process(
    process: subprocess.Popen[str],
    *,
    profile_id: int | None,
    command_label: str,
) -> None:
    with _ACTIVE_PROCESSES_LOCK:
        _ACTIVE_PROCESSES[id(process)] = (process, profile_id, command_label)


def _unregister_active_process(process: subprocess.Popen[str]) -> None:
    with _ACTIVE_PROCESSES_LOCK:
        _ACTIVE_PROCESSES.pop(id(process), None)


def _interrupt_active_processes(*, reason: str) -> None:
    with _ACTIVE_PROCESSES_LOCK:
        active = list(_ACTIVE_PROCESSES.values())
    if not active:
        _log(f"{reason}: no active client subprocesses to interrupt")
        return

    _log(f"{reason}: interrupting {len(active)} active client subprocesses")
    for process, profile_id, command_label in active:
        if process.poll() is not None:
            continue
        try:
            process.send_signal(signal.SIGINT)
            _log(f"sent SIGINT to running command {command_label}", profile_id=profile_id)
        except OSError as exc:
            _log(f"failed to send SIGINT to {command_label}: {exc}", profile_id=profile_id)


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


def _stream_process_output(
    stream: Any,
    *,
    sink: list[str],
    profile_id: int | None,
    command_label: str,
) -> None:
    try:
        for line in stream:
            sink.append(line)
            text = line.rstrip()
            if text:
                _log(f"{command_label}: {text}", profile_id=profile_id)
    finally:
        stream.close()


def _run_client_command(
    args: list[str],
    *,
    timeout_seconds: float,
    profile_id: int | None = None,
) -> dict[str, Any]:
    command = ["python3", str(CLIENT_PATH), *args]
    command_label = " ".join(args[:3]) if len(args) >= 3 else " ".join(args)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    stdout_lines: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "payload": None,
            "error": f"failed to start command: {exc}",
        }

    _register_active_process(process, profile_id=profile_id, command_label=command_label)

    reader: threading.Thread | None = None
    try:
        if process.stdout is not None:
            reader = threading.Thread(
                target=_stream_process_output,
                kwargs={
                    "stream": process.stdout,
                    "sink": stdout_lines,
                    "profile_id": profile_id,
                    "command_label": command_label,
                },
                daemon=True,
            )
            reader.start()

        started = time.monotonic()
        next_heartbeat_at = started + COMMAND_HEARTBEAT_SECONDS
        timeout_hit = False

        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            now = time.monotonic()
            if now >= next_heartbeat_at:
                _log(
                    (
                        f"{command_label}: still running "
                        f"elapsed_seconds={round(now - started, 1)}"
                    ),
                    profile_id=profile_id,
                )
                next_heartbeat_at = now + COMMAND_HEARTBEAT_SECONDS
            if now - started >= timeout_seconds:
                timeout_hit = True
                _log(
                    f"{command_label}: timeout reached after {timeout_seconds} seconds; terminating",
                    profile_id=profile_id,
                )
                process.terminate()
                break
            time.sleep(1.0)

        if timeout_hit:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _log(f"{command_label}: did not terminate promptly; killing", profile_id=profile_id)
                process.kill()
                process.wait(timeout=10)
        else:
            process.wait()

        if reader is not None:
            reader.join(timeout=5)

        stdout_text = "".join(stdout_lines)
        stderr_text = ""
        if timeout_hit:
            return {
                "command": command,
                "returncode": None,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "payload": None,
                "error": f"command timed out after {timeout_seconds} seconds",
            }

        payload: dict[str, Any] | None = None
        parse_error: str | None = None
        if stdout_text.strip():
            try:
                payload = _extract_last_json_object(stdout_text)
            except ValueError as exc:
                parse_error = str(exc)

        return {
            "command": command,
            "returncode": process.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "payload": payload,
            "error": parse_error,
        }
    finally:
        _unregister_active_process(process)


def _run_profile_start(profile_id: int) -> dict[str, Any]:
    _log(
        f"starting profile with ssh_target={SSH_TARGET} partition={PARTITION} model={MODEL_KEY}",
        profile_id=profile_id,
    )
    result = _run_client_command(
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
        profile_id=profile_id,
    )
    payload = result.get("payload")
    if isinstance(payload, dict) and payload.get("ok"):
        _log("start completed successfully", profile_id=profile_id)
    else:
        _log(f"start failed returncode={result.get('returncode')}", profile_id=profile_id)
    return result


def _run_profile_stop(profile_id: int) -> dict[str, Any]:
    _log("stopping profile", profile_id=profile_id)
    result = _run_client_command(
        ["stop", "-P", str(profile_id)],
        timeout_seconds=STOP_TIMEOUT_SECONDS,
        profile_id=profile_id,
    )
    payload = result.get("payload")
    if isinstance(payload, dict) and payload.get("ok"):
        _log("stop completed successfully", profile_id=profile_id)
    else:
        _log(f"stop failed returncode={result.get('returncode')}", profile_id=profile_id)
    return result


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
    _log("checking health via control-plane up/status", profile_id=profile_id)
    up_result = _run_client_command(
        ["up", "-P", str(profile_id)],
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        profile_id=profile_id,
    )
    payload = up_result.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"profile {profile_id} up command returned no JSON payload: {up_result}")
    if up_result.get("returncode") != 0 or not payload.get("ok"):
        raise RuntimeError(f"profile {profile_id} up command failed: {json.dumps(up_result, indent=2)}")

    ready_data = payload.get("data")
    if not isinstance(ready_data, dict) or not ready_data.get("ready"):
        raise RuntimeError(f"profile {profile_id} is not ready: {json.dumps(up_result, indent=2)}")

    ports = _load_port_profile(profile_id)
    _log(
        (
            "probing local ports "
            f"vllm={ports['vllm_port']} jaeger_ui={ports['jaeger_api_port']} otlp={ports['jaeger_otlp_port']}"
        ),
        profile_id=profile_id,
    )
    otlp_probe = _probe_tcp_port(ports["jaeger_otlp_port"], timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    if not otlp_probe.get("ok"):
        raise RuntimeError(f"profile {profile_id} OTLP port probe failed: {json.dumps(otlp_probe, indent=2)}")

    models_probe = _wait_for_models(ports["vllm_port"], timeout_seconds=MODEL_READY_TIMEOUT_SECONDS)
    if not models_probe.get("ok"):
        raise RuntimeError(f"profile {profile_id} /v1/models is not ready: {json.dumps(models_probe, indent=2)}")

    model_name = _extract_model_name(models_probe["result"])
    _log(f"health check passed model={model_name}", profile_id=profile_id)
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
    next_progress_at = time.monotonic() + QUERY_PROGRESS_INTERVAL_SECONDS
    success_count = 0
    attempts = 0
    last_response: dict[str, Any] | None = None

    _log(
        (
            f"starting query loop for {round(duration_seconds, 1)}s "
            f"against vllm_port={ports['vllm_port']} model={model_name}"
        ),
        profile_id=profile_id,
    )
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
        if time.monotonic() >= next_progress_at:
            remaining_seconds = max(deadline - time.monotonic(), 0.0)
            _log(
                (
                    f"query progress successes={success_count} attempts={attempts} "
                    f"remaining_seconds={round(remaining_seconds, 1)}"
                ),
                profile_id=profile_id,
            )
            next_progress_at = time.monotonic() + QUERY_PROGRESS_INTERVAL_SECONDS

    _log(
        f"query loop complete successes={success_count} attempts={attempts}",
        profile_id=profile_id,
    )
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
    *,
    phase_name: str,
    on_result: Any | None = None,
) -> dict[int, dict[str, Any]]:
    _log(f"phase={phase_name} launching {len(items)} parallel tasks")
    results: dict[int, dict[str, Any]] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(items))
    future_to_item: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
    try:
        future_to_item = {executor.submit(func, item): item for item in items}
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                results[item] = future.result()
            except Exception as exc:
                _log(f"phase={phase_name} task failed: {exc}", profile_id=item)
                raise
            if on_result is not None:
                on_result(item, results[item])
            _log(f"phase={phase_name} task finished", profile_id=item)
        executor.shutdown(wait=True, cancel_futures=False)
        return results
    except KeyboardInterrupt as exc:
        _log(f"phase={phase_name} interrupted by user")
        _interrupt_active_processes(reason=f"phase={phase_name}")
        for future in future_to_item:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise PhaseInterruptedError(phase_name=phase_name, partial_results=dict(results)) from exc
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise


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
        _log(
            (
                "starting live multi-profile test "
                f"profiles={list(PORT_PROFILES)} ssh_target={SSH_TARGET} "
                f"partition={PARTITION} model={MODEL_KEY}"
            )
        )

        started_profiles: set[int] = set()
        cleanup_failures: dict[int, dict[str, Any]] = {}
        body_error: tuple[type[BaseException], BaseException, Any] | None = None
        interrupted_phase: str | None = None

        def _record_started_profile(profile_id: int, result: dict[str, Any]) -> None:
            payload = result.get("payload")
            if (
                result.get("returncode") == 0
                and isinstance(payload, dict)
                and payload.get("ok")
            ):
                started_profiles.add(profile_id)

        try:
            start_results = _collect_parallel_results(
                _run_profile_start,
                list(PORT_PROFILES),
                phase_name="start",
                on_result=_record_started_profile,
            )
            start_failures = {
                profile_id: result
                for profile_id, result in start_results.items()
                if result.get("returncode") != 0
                or not isinstance(result.get("payload"), dict)
                or not result["payload"].get("ok")
            }
            self.assertFalse(
                start_failures,
                "concurrent start failed:\n" + json.dumps(start_failures, indent=2, sort_keys=True),
            )

            _log("all profile starts completed; beginning health checks")
            health_results = _collect_parallel_results(
                _check_profile_health,
                list(PORT_PROFILES),
                phase_name="health",
            )
            model_names = {
                profile_id: str(result["model_name"])
                for profile_id, result in health_results.items()
            }
            _log(
                "all profile health checks passed "
                + json.dumps(model_names, indent=2, sort_keys=True)
            )

            query_results: dict[int, dict[str, Any]] = {}
            query_failures: dict[int, str] = {}
            _log(
                f"starting concurrent query phase for {round(QUERY_DURATION_SECONDS, 1)} seconds"
            )
            mp_context = multiprocessing.get_context("spawn")
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=len(PORT_PROFILES),
                mp_context=mp_context,
            )
            try:
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
                        _log(
                            (
                                "query worker finished "
                                f"successes={query_results[profile_id]['success_count']} "
                                f"attempts={query_results[profile_id]['attempts']}"
                            ),
                            profile_id=profile_id,
                        )
                    except Exception as exc:  # pragma: no cover - worker boundary
                        query_failures[profile_id] = str(exc)
                        _log(f"query worker failed: {exc}", profile_id=profile_id)
                executor.shutdown(wait=True, cancel_futures=False)
            except KeyboardInterrupt as exc:
                _log("phase=query interrupted by user")
                executor.shutdown(wait=False, cancel_futures=True)
                raise PhaseInterruptedError(
                    phase_name="query",
                    partial_results=dict(query_results),
                ) from exc

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
            _log("query phase completed successfully for all profiles")
        except PhaseInterruptedError as exc:
            interrupted_phase = exc.phase_name
            body_error = sys.exc_info()
        except Exception:  # noqa: BLE001
            body_error = sys.exc_info()
        finally:
            if started_profiles:
                profiles_to_stop = sorted(started_profiles)
                _log(f"starting cleanup stop for profiles={profiles_to_stop}")
                try:
                    stop_results = _collect_parallel_results(
                        _run_profile_stop,
                        profiles_to_stop,
                        phase_name="stop",
                    )
                except PhaseInterruptedError:
                    interrupted_phase = interrupted_phase or "stop"
                    body_error = body_error or sys.exc_info()
                    _log("cleanup stop was interrupted by user")
                else:
                    cleanup_failures = {
                        profile_id: result
                        for profile_id, result in stop_results.items()
                        if result.get("returncode") != 0
                        or not isinstance(result.get("payload"), dict)
                        or not result["payload"].get("ok")
                    }
                    if cleanup_failures:
                        _log("cleanup stop completed with failures")
                    else:
                        _log("cleanup stop completed successfully")

        if body_error is not None:
            if interrupted_phase is not None:
                if cleanup_failures:
                    self.fail(
                        "interrupted by user during "
                        f"{interrupted_phase}; cleanup had failures:\n"
                        + json.dumps(cleanup_failures, indent=2, sort_keys=True)
                    )
                self.skipTest(f"interrupted by user during {interrupted_phase}; cleanup attempted")
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
        _log("live multi-profile test completed successfully")


if __name__ == "__main__":
    unittest.main()
