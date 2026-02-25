#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time

import requests
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, set_tracer_provider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {value}") from exc


def _wait_for_vllm(base_url: str, timeout_seconds: int) -> None:
    models_url = f"{base_url.rstrip('/')}/v1/models"
    deadline = time.time() + timeout_seconds
    last_error = "vLLM not ready yet"
    while time.time() < deadline:
        try:
            response = requests.get(models_url, timeout=10)
            if response.ok:
                print(f"vLLM is ready: {models_url}")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(5)
    raise RuntimeError(
        f"Timed out after {timeout_seconds}s waiting for vLLM at {models_url}. "
        f"Last error: {last_error}"
    )


def main() -> int:
    base_url = os.getenv(
        "VLLM_BASE_URL", f"http://localhost:{os.getenv('VLLM_SERVICE_PORT', '11451')}"
    )
    model_name = (
        os.getenv("VLLM_TEST_MODEL_NAME")
        or os.getenv("VLLM_SERVED_MODEL_NAME")
        or os.getenv("VLLM_MODEL_NAME")
    )
    if not model_name:
        print(
            "Set one of VLLM_TEST_MODEL_NAME, VLLM_SERVED_MODEL_NAME, or VLLM_MODEL_NAME",
            file=sys.stderr,
        )
        return 2

    prompt = os.getenv("VLLM_TEST_PROMPT", "San Francisco is a")
    max_tokens = _int_env("VLLM_TEST_MAX_TOKENS", 32)
    startup_timeout = _int_env("VLLM_STARTUP_TIMEOUT", 600)

    _wait_for_vllm(base_url, startup_timeout)

    trace_provider = TracerProvider()
    set_tracer_provider(trace_provider)
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    tracer = trace_provider.get_tracer("docker-smoke-test")

    completion_url = f"{base_url.rstrip('/')}/v1/completions"
    headers = {}
    with tracer.start_as_current_span(
        "dummy-client-completion", kind=SpanKind.CLIENT
    ) as span:
        span.set_attribute("smoke_test", True)
        span.set_attribute("vllm.base_url", base_url)
        span.set_attribute("vllm.model_name", model_name)
        TraceContextTextMapPropagator().inject(headers)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        response = requests.post(
            completion_url, headers=headers, json=payload, timeout=120
        )
        response.raise_for_status()

        result = response.json()
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError("Completion response did not include any choices")
        completion_text = (choices[0].get("text") or "").strip()
        trace_id = f"{span.get_span_context().trace_id:032x}"

    trace_provider.force_flush()
    print(f"Smoke test OK. Model={model_name}")
    print(f"Completion preview: {completion_text[:200]}")
    print(f"Trace ID: {trace_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as exc:
        response = exc.response
        body = response.text[:400] if response is not None else str(exc)
        print(f"HTTP request failed: {body}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
