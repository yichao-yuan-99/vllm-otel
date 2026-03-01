#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Entrypoint for standalone smoke tests on local vLLM + Jaeger."""

from __future__ import annotations

import argparse

from common import (
    DEFAULT_FORCE_SEQUENCE_TEST_TEXT,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    DEFAULT_TRACE_WAIT_SECONDS,
    EXPECTED_VLLM_SERVICE_NAME,
    emit,
    extract_model_name,
    load_port_profile,
    payload,
    wait_for_vllm_models,
)
from test_force_sequence import run as run_force_sequence_test
from test_inference import run as run_inference_test
from test_jaeger_otlp import run as run_jaeger_otlp_test
from test_jaeger_service import run as run_jaeger_service_test


def run_smoke_tests(
    *,
    port_profile_id: int,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
    trace_wait_seconds: float,
    force_sequence_test_text: str,
) -> dict[str, object]:
    try:
        profile_key, ports = load_port_profile(port_profile_id)
    except Exception as exc:
        return payload(ok=False, code=900, message=f"failed to load port profile: {exc}")

    checks: dict[str, object] = {
        "profile": {"id": profile_key, "label": ports.get("label"), "ports": ports},
    }

    otlp_ok, otlp_detail = run_jaeger_otlp_test(
        jaeger_otlp_port=ports["jaeger_otlp_port"],
        timeout_seconds=request_timeout_seconds,
    )
    checks["jaeger_otlp_port"] = {"ok": otlp_ok, "detail": otlp_detail}
    if not otlp_ok:
        return payload(ok=False, code=901, message="jaeger OTLP port is not reachable", data=checks)

    models_ok, models_detail = wait_for_vllm_models(
        vllm_port=ports["vllm_port"],
        startup_timeout_seconds=startup_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )
    checks["vllm_models"] = {"ok": models_ok, "detail": models_detail}
    if not models_ok:
        return payload(ok=False, code=902, message="vLLM /v1/models is not ready", data=checks)

    try:
        model_name = extract_model_name(models_detail["result"])
    except Exception as exc:
        checks["model_name"] = {"ok": False, "error": str(exc)}
        return payload(ok=False, code=903, message="failed to resolve model name from /v1/models", data=checks)
    checks["model_name"] = {"ok": True, "value": model_name}

    inference_ok, inference_detail = run_inference_test(
        vllm_port=ports["vllm_port"],
        model_name=model_name,
        request_timeout_seconds=request_timeout_seconds,
    )
    checks["inference"] = {"ok": inference_ok, "detail": inference_detail}
    if not inference_ok:
        return payload(ok=False, code=904, message="inference smoke test failed", data=checks)

    force_ok, force_detail = run_force_sequence_test(
        vllm_port=ports["vllm_port"],
        model_name=model_name,
        test_text=force_sequence_test_text,
        request_timeout_seconds=request_timeout_seconds,
    )
    checks["force_sequence"] = {"ok": force_ok, "detail": force_detail}
    if not force_ok:
        return payload(ok=False, code=905, message="force-sequence smoke test failed", data=checks)

    jaeger_ok, jaeger_detail = run_jaeger_service_test(
        jaeger_api_port=ports["jaeger_api_port"],
        service_name=EXPECTED_VLLM_SERVICE_NAME,
        wait_seconds=trace_wait_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )
    checks["jaeger_services"] = {"ok": jaeger_ok, "detail": jaeger_detail}
    if not jaeger_ok:
        return payload(
            ok=False,
            code=906,
            message=f"jaeger missing expected service '{EXPECTED_VLLM_SERVICE_NAME}'",
            data=checks,
        )

    return payload(
        ok=True,
        code=0,
        message="smoke checks passed (vllm, force-sequence, jaeger)",
        data=checks,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run smoke tests against a running local vLLM + Jaeger environment "
            "selected by port profile ID."
        )
    )
    parser.add_argument(
        "--port-profile",
        "-p",
        required=True,
        type=int,
        help="Port profile numeric ID from configs/port_profiles.toml.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help="Timeout while waiting for vLLM /v1/models readiness.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Per-request timeout for HTTP/TCP checks.",
    )
    parser.add_argument(
        "--trace-wait-seconds",
        type=float,
        default=DEFAULT_TRACE_WAIT_SECONDS,
        help="How long to wait for Jaeger service indexing after inference.",
    )
    parser.add_argument(
        "--force-sequence-test-text",
        default=DEFAULT_FORCE_SEQUENCE_TEST_TEXT,
        help="Text used to build forced token sequence for force-sequence smoke check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_smoke_tests(
        port_profile_id=args.port_profile,
        startup_timeout_seconds=args.startup_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        trace_wait_seconds=args.trace_wait_seconds,
        force_sequence_test_text=args.force_sequence_test_text,
    )
    return emit(result, fail_on_error=True)


if __name__ == "__main__":
    raise SystemExit(main())
