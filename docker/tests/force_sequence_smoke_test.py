#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from typing import Any

import requests


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw}") from exc


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


def _require_model() -> str:
    model_name = (
        os.getenv("VLLM_TEST_MODEL_NAME")
        or os.getenv("VLLM_SERVED_MODEL_NAME")
        or os.getenv("VLLM_MODEL_NAME")
    )
    if model_name:
        return model_name
    raise RuntimeError(
        "Set one of VLLM_TEST_MODEL_NAME, VLLM_SERVED_MODEL_NAME, or VLLM_MODEL_NAME"
    )


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def main() -> int:
    base_url = os.getenv(
        "VLLM_BASE_URL", f"http://localhost:{os.getenv('VLLM_SERVICE_PORT', '11451')}"
    )
    model_name = _require_model()
    startup_timeout = _int_env("VLLM_STARTUP_TIMEOUT", 600)
    test_text = os.getenv("VLLM_FORCE_SEQUENCE_TEST_TEXT", "FORCE_SEQ_SMOKE_OK")

    _wait_for_vllm(base_url, startup_timeout)

    tokenize_payload = {
        "model": model_name,
        "prompt": test_text,
        "add_special_tokens": False,
    }
    tokenize_resp = _post_json(f"{base_url.rstrip('/')}/tokenize", tokenize_payload)
    forced_token_ids = tokenize_resp.get("tokens")
    if not isinstance(forced_token_ids, list) or not forced_token_ids:
        raise RuntimeError(f"Tokenization failed: {tokenize_resp}")
    if not all(isinstance(token_id, int) for token_id in forced_token_ids):
        raise RuntimeError(f"Tokenization returned non-int token ids: {tokenize_resp}")

    detok_payload = {"model": model_name, "tokens": forced_token_ids}
    detok_resp = _post_json(f"{base_url.rstrip('/')}/detokenize", detok_payload)
    expected_text = detok_resp.get("prompt")
    if not isinstance(expected_text, str) or expected_text == "":
        raise RuntimeError(f"Detokenization failed: {detok_resp}")

    chat_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Return any short answer."}],
        "temperature": 0.0,
        "max_tokens": len(forced_token_ids) + 1,
        "vllm_xargs": {
            "forced_token_ids": forced_token_ids,
            "force_eos_after_sequence": True,
        },
    }
    chat_resp = _post_json(f"{base_url.rstrip('/')}/v1/chat/completions", chat_payload)
    choices = chat_resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"Missing choices in chat response: {chat_resp}")
    message = choices[0].get("message") or {}
    generated_text = message.get("content")
    if not isinstance(generated_text, str):
        raise RuntimeError(f"Missing text content in chat response: {chat_resp}")

    if generated_text != expected_text:
        raise RuntimeError(
            "Forced generation mismatch.\n"
            f"Expected: {expected_text!r}\n"
            f"Got:      {generated_text!r}\n"
            "This usually means the force-sequence logits processor is not active."
        )

    print("Force-sequence smoke test OK.")
    print(f"Model={model_name}")
    print(f"Forced text={expected_text!r}")
    print(f"Forced token count={len(forced_token_ids)}")
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
        print(f"Force-sequence smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
