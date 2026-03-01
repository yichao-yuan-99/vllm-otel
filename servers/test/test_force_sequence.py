#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Force-sequence logits-processor smoke test."""

from __future__ import annotations

from typing import Any

from common import http_json


def run(
    *,
    vllm_port: int,
    model_name: str,
    test_text: str,
    request_timeout_seconds: float,
) -> tuple[bool, dict[str, Any]]:
    tok_ok, tok = http_json(
        "POST",
        f"http://127.0.0.1:{vllm_port}/tokenize",
        payload_obj={
            "model": model_name,
            "prompt": test_text,
            "add_special_tokens": False,
        },
        timeout_seconds=request_timeout_seconds,
    )
    if not tok_ok:
        return False, {"stage": "tokenize", "detail": tok}

    tok_body = tok.get("body")
    if not isinstance(tok_body, dict):
        return False, {"stage": "tokenize", "detail": "response body is not an object", "response": tok}
    tokens = tok_body.get("tokens")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(token, int) for token in tokens):
        return False, {"stage": "tokenize", "detail": "invalid token list", "response": tok}

    detok_ok, detok = http_json(
        "POST",
        f"http://127.0.0.1:{vllm_port}/detokenize",
        payload_obj={"model": model_name, "tokens": tokens},
        timeout_seconds=request_timeout_seconds,
    )
    if not detok_ok:
        return False, {"stage": "detokenize", "detail": detok}
    detok_body = detok.get("body")
    if not isinstance(detok_body, dict):
        return False, {"stage": "detokenize", "detail": "response body is not an object", "response": detok}
    expected_text = detok_body.get("prompt")
    if not isinstance(expected_text, str) or expected_text == "":
        return False, {"stage": "detokenize", "detail": "missing prompt text", "response": detok}

    chat_ok, chat = http_json(
        "POST",
        f"http://127.0.0.1:{vllm_port}/v1/chat/completions",
        payload_obj={
            "model": model_name,
            "messages": [{"role": "user", "content": "Return any short answer."}],
            "chat_template_kwargs": {"thinking": False},
            "temperature": 0.0,
            "max_tokens": len(tokens) + 1,
            "vllm_xargs": {
                "forced_token_ids": tokens,
                "force_eos_after_sequence": True,
            },
        },
        timeout_seconds=request_timeout_seconds,
    )
    if not chat_ok:
        return False, {"stage": "chat_completions", "detail": chat}

    chat_body = chat.get("body")
    if not isinstance(chat_body, dict):
        return False, {"stage": "chat_completions", "detail": "response body is not an object", "response": chat}
    choices = chat_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, {"stage": "chat_completions", "detail": "no choices", "response": chat}
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return False, {"stage": "chat_completions", "detail": "invalid choice payload", "response": chat}
    message = choice0.get("message")
    if not isinstance(message, dict):
        return False, {"stage": "chat_completions", "detail": "missing message object", "response": chat}
    generated_text = message.get("content")
    if not isinstance(generated_text, str):
        return False, {"stage": "chat_completions", "detail": "missing message.content", "response": chat}

    expected_tokens_len = len(tokens)
    actual_tok_ok, actual_tok = http_json(
        "POST",
        f"http://127.0.0.1:{vllm_port}/tokenize",
        payload_obj={
            "model": model_name,
            "prompt": generated_text,
            "add_special_tokens": False,
        },
        timeout_seconds=request_timeout_seconds,
    )
    if not actual_tok_ok:
        return False, {"stage": "tokenize_actual", "detail": actual_tok}
    actual_tok_body = actual_tok.get("body")
    if not isinstance(actual_tok_body, dict):
        return False, {
            "stage": "tokenize_actual",
            "detail": "response body is not an object",
            "response": actual_tok,
        }
    actual_tokens = actual_tok_body.get("tokens")
    if not isinstance(actual_tokens, list) or not all(isinstance(token, int) for token in actual_tokens):
        return False, {
            "stage": "tokenize_actual",
            "detail": "invalid token list",
            "response": actual_tok,
        }
    actual_tokens_len = len(actual_tokens)

    if actual_tokens_len != expected_tokens_len:
        return False, {
            "stage": "assert",
            "detail": "forced token length mismatch",
            "expected_tokens_len": expected_tokens_len,
            "actual_tokens_len": actual_tokens_len,
            "expected": expected_text,
            "actual": generated_text,
        }

    if generated_text != expected_text:
        return False, {
            "stage": "assert",
            "detail": "forced generation mismatch",
            "expected": expected_text,
            "actual": generated_text,
            "expected_tokens_len": expected_tokens_len,
            "actual_tokens_len": actual_tokens_len,
        }

    return True, {
        "expected_text": expected_text,
        "expected_tokens_len": expected_tokens_len,
        "actual_tokens_len": actual_tokens_len,
    }
