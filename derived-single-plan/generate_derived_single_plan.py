#!/usr/bin/env python3
"""Scale a single-worker replay plan into a synthetic multi-worker plan."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from random import Random
import sys
from typing import Any, Callable
from urllib import error as url_error
from urllib import request as url_request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway.port_profiles import load_port_profile


DEFAULT_SIZE = 500
DEFAULT_REQUEST_TIMEOUT_S = 60.0
MAX_SAMPLE_ATTEMPTS = 32
REPLAY_PLAN_SCHEMA_VERSION = "replay-plan.v1"


class ProgressBar:
    def __init__(self, *, total: int, label: str, stream: Any = None) -> None:
        self.total = max(0, int(total))
        self.completed = 0
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.message: str | None = None
        self._last_len = 0
        self.enabled = self.total > 0 and bool(getattr(self.stream, "isatty", lambda: False)())
        if self.enabled:
            self._render()

    def set_message(self, message: str | None) -> None:
        if not self.enabled:
            return
        self.message = message
        self._render()

    def advance(self, step: int = 1, message: str | None = None) -> None:
        if not self.enabled:
            return
        if message is not None:
            self.message = message
        self.completed = min(self.total, self.completed + max(0, int(step)))
        self._render()

    def close(self) -> None:
        if not self.enabled:
            return
        self._render(final=True)

    def _render(self, *, final: bool = False) -> None:
        ratio = 1.0 if self.total == 0 else float(self.completed) / float(self.total)
        ratio = max(0.0, min(1.0, ratio))
        width = 32
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        line = f"\r{self.label} [{bar}] {self.completed}/{self.total}"
        if self.message:
            line += f" {self.message}"
        padding = max(0, self._last_len - len(line))
        self.stream.write(line + (" " * padding))
        self.stream.flush()
        self._last_len = len(line)
        if final or self.completed >= self.total:
            self.stream.write("\n")
            self.stream.flush()
            self._last_len = 0


def now_iso8601_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    chars: list[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars) or "value"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def decode_json_response_body(raw: str) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {"raw_body": raw}


def http_json(
    *,
    method: str,
    url: str,
    payload: Any,
    timeout_s: float | None,
) -> tuple[int, Any]:
    req = url_request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"content-type": "application/json"},
        method=method.upper(),
    )
    try:
        if timeout_s is None:
            response_cm = url_request.urlopen(req)
        else:
            response_cm = url_request.urlopen(req, timeout=timeout_s)
        with response_cm as response:
            status = int(response.getcode())
            raw = response.read().decode("utf-8", errors="replace")
    except url_error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")
    except url_error.URLError as exc:
        raise RuntimeError(f"HTTP request failed: {method} {url}: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"HTTP request timed out: {method} {url}") from exc
    return status, decode_json_response_body(raw)


def _is_int_token(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_token_sequence(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    if not all(_is_int_token(item) for item in value):
        return None
    return [int(item) for item in value]


def extract_forced_token_ids(request_payload: dict[str, Any]) -> list[int] | None:
    direct = _normalize_token_sequence(request_payload.get("forced_token_ids"))
    if direct is not None:
        return direct

    body = request_payload.get("body")
    if not isinstance(body, dict):
        return None
    vllm_xargs = body.get("vllm_xargs")
    if not isinstance(vllm_xargs, dict):
        return None
    return _normalize_token_sequence(vllm_xargs.get("forced_token_ids"))


def set_forced_token_ids(request_payload: dict[str, Any], forced_token_ids: list[int]) -> None:
    body = request_payload.get("body")
    if not isinstance(body, dict):
        raise ValueError("deterministic request is missing body object")

    vllm_xargs = body.get("vllm_xargs")
    if not isinstance(vllm_xargs, dict):
        vllm_xargs = {}
    vllm_xargs["forced_token_ids"] = list(forced_token_ids)
    vllm_xargs["force_eos_after_sequence"] = True
    body["vllm_xargs"] = vllm_xargs

    max_tokens_required = len(forced_token_ids) + 1
    current_max_tokens = body.get("max_tokens")
    if not isinstance(current_max_tokens, int) or isinstance(current_max_tokens, bool):
        current_max_tokens = 0
    if current_max_tokens < max_tokens_required:
        body["max_tokens"] = max_tokens_required

    request_payload["forced_token_ids"] = list(forced_token_ids)
    request_payload["force_eos_after_sequence"] = True
    request_payload["cancel_after_s"] = None
    request_payload["replay_mode"] = "deterministic_forced_tokens"


def extract_expected_response_text(request_payload: dict[str, Any]) -> str | None:
    expected_response_text = request_payload.get("expected_response_text")
    if not isinstance(expected_response_text, str):
        return None
    return expected_response_text


def extract_request_messages(request_payload: dict[str, Any]) -> list[Any] | None:
    body = request_payload.get("body")
    if not isinstance(body, dict):
        return None
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    return messages


def extract_message_text(message_payload: dict[str, Any]) -> str | None:
    content = message_payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return None
    if not isinstance(content, list):
        return None
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str):
            return text
    return None


def replace_message_text_content(
    message_payload: dict[str, Any],
    *,
    source_text: str,
    replacement_text: str,
    expected_role: str | None = None,
) -> bool:
    if expected_role is not None and message_payload.get("role") != expected_role:
        return False

    content = message_payload.get("content")
    if isinstance(content, str):
        if content != source_text:
            return False
        message_payload["content"] = replacement_text
        return True

    if isinstance(content, dict):
        text = content.get("text")
        if text != source_text:
            return False
        content["text"] = replacement_text
        return True

    if not isinstance(content, list):
        return False

    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        if part.get("text") != source_text:
            continue
        part["text"] = replacement_text
        return True
    return False


def replace_assistant_message_content(
    message_payload: dict[str, Any],
    *,
    source_text: str,
    replacement_text: str,
) -> bool:
    return replace_message_text_content(
        message_payload,
        source_text=source_text,
        replacement_text=replacement_text,
        expected_role="assistant",
    )


def rewrite_initial_messages(
    *,
    request_payload: dict[str, Any],
    source_initial_messages: list[Any],
    replacement_initial_message_texts: list[str | None],
) -> None:
    messages = extract_request_messages(request_payload)
    if messages is None:
        return

    for message_index, source_message_payload in enumerate(source_initial_messages):
        if message_index >= len(messages):
            break
        if not isinstance(source_message_payload, dict):
            continue
        current_message_payload = messages[message_index]
        if not isinstance(current_message_payload, dict):
            continue
        replacement_text = replacement_initial_message_texts[message_index]
        if replacement_text is None:
            continue
        source_text = extract_message_text(source_message_payload)
        if source_text is None:
            continue
        source_role = source_message_payload.get("role")
        expected_role = source_role if isinstance(source_role, str) else None
        replace_message_text_content(
            current_message_payload,
            source_text=source_text,
            replacement_text=replacement_text,
            expected_role=expected_role,
        )


def rewrite_request_message_history(
    *,
    request_payload: dict[str, Any],
    source_expected_response_texts: list[str | None],
    replacement_expected_response_texts: list[str | None],
) -> None:
    messages = extract_request_messages(request_payload)
    if messages is None:
        return

    pending_replacements = [
        (source_text, replacement_text)
        for source_text, replacement_text in zip(
            source_expected_response_texts,
            replacement_expected_response_texts,
            strict=False,
        )
        if source_text is not None and replacement_text is not None
    ]
    if not pending_replacements:
        return

    pending_index = 0
    for message_payload in messages:
        if pending_index >= len(pending_replacements):
            break
        if not isinstance(message_payload, dict):
            continue
        source_text, replacement_text = pending_replacements[pending_index]
        if replace_assistant_message_content(
            message_payload,
            source_text=source_text,
            replacement_text=replacement_text,
        ):
            pending_index += 1


def resolve_model_name(
    *,
    request_payload: dict[str, Any],
    plan_payload: dict[str, Any],
) -> str:
    model_for_tokenize = request_payload.get("model_for_tokenize")
    if isinstance(model_for_tokenize, str) and model_for_tokenize.strip():
        return model_for_tokenize.strip()

    body = request_payload.get("body")
    if isinstance(body, dict):
        body_model = body.get("model")
        if isinstance(body_model, str) and body_model.strip():
            return body_model.strip()

    replay_target = plan_payload.get("replay_target")
    if isinstance(replay_target, dict):
        target_model = replay_target.get("model")
        if isinstance(target_model, str) and target_model.strip():
            return target_model.strip()

    raise ValueError("unable to resolve model name for request detokenization")


def detokenize_tokens(
    *,
    detokenize_endpoint: str,
    model_name: str,
    forced_token_ids: list[int],
    timeout_s: float,
) -> str:
    status, payload = http_json(
        method="POST",
        url=detokenize_endpoint,
        payload={
            "model": model_name,
            "tokens": forced_token_ids,
        },
        timeout_s=timeout_s,
    )
    if status >= 400:
        raise ValueError(
            f"/detokenize failed for model={model_name!r}: "
            f"HTTP {status}, payload={payload}"
        )
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected /detokenize payload type: {type(payload)!r}")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"Invalid /detokenize payload: {payload}")
    return prompt


def tokenize_text(
    *,
    tokenize_endpoint: str,
    model_name: str,
    text: str,
    timeout_s: float,
) -> list[int]:
    status, payload = http_json(
        method="POST",
        url=tokenize_endpoint,
        payload={
            "model": model_name,
            "prompt": text,
            "add_special_tokens": False,
        },
        timeout_s=timeout_s,
    )
    if status >= 400:
        raise ValueError(
            f"/tokenize failed for model={model_name!r}: "
            f"HTTP {status}, payload={payload}"
        )
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected /tokenize payload type: {type(payload)!r}")
    tokens = _normalize_token_sequence(payload.get("tokens"))
    if tokens is None:
        raise ValueError(f"Invalid /tokenize payload: {payload}")
    return tokens


def resolve_detokenize_endpoint(port_profile_id: int) -> str:
    profile = load_port_profile(port_profile_id)
    return f"http://127.0.0.1:{profile.vllm_port}/detokenize"


def resolve_tokenize_endpoint(port_profile_id: int) -> str:
    profile = load_port_profile(port_profile_id)
    return f"http://127.0.0.1:{profile.vllm_port}/tokenize"


def extract_compile_single_trail(plan_payload: Any) -> str | None:
    if not isinstance(plan_payload, dict):
        return None
    compile_options = plan_payload.get("compile_options")
    if not isinstance(compile_options, dict):
        return None
    single_trail = compile_options.get("single_trail")
    if not isinstance(single_trail, str):
        return None
    stripped = single_trail.strip()
    return stripped or None


def validate_single_worker_plan(plan_payload: Any) -> dict[str, Any]:
    if not isinstance(plan_payload, dict):
        raise ValueError("plan payload root must be an object")
    if plan_payload.get("schema_version") != REPLAY_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"plan schema_version must be {REPLAY_PLAN_SCHEMA_VERSION!r}, "
            f"got {plan_payload.get('schema_version')!r}"
        )
    workers = plan_payload.get("workers")
    if not isinstance(workers, list):
        raise ValueError("plan workers must be a list")
    if len(workers) != 1:
        raise ValueError(
            f"derived-single-plan requires exactly 1 worker, found {len(workers)}"
        )
    worker = workers[0]
    if not isinstance(worker, dict):
        raise ValueError("plan worker must be an object")
    requests = worker.get("requests")
    if not isinstance(requests, list):
        raise ValueError("plan worker.requests must be a list")
    if not requests:
        raise ValueError("plan worker.requests must not be empty")
    if extract_compile_single_trail(plan_payload) is None:
        raise ValueError(
            "plan must be compiled with --single-trail "
            "(missing compile_options.single_trail)"
        )
    return plan_payload


def collect_source_token_sequences(
    plan_payload: dict[str, Any],
) -> list[list[int]]:
    worker = plan_payload["workers"][0]
    requests = worker["requests"]
    assert isinstance(requests, list)

    token_sequences: list[list[int]] = []
    for request_index, request_payload in enumerate(requests):
        if not isinstance(request_payload, dict):
            raise ValueError(f"request[{request_index}] must be an object")
        replay_mode = request_payload.get("replay_mode")
        if replay_mode == "client_disconnect_after_duration":
            raise ValueError(
                "derived-single-plan only supports deterministic requests with forced_token_ids"
            )
        forced_token_ids = extract_forced_token_ids(request_payload)
        if forced_token_ids is None:
            raise ValueError(
                f"request[{request_index}] is missing deterministic forced_token_ids"
            )
        token_sequences.append(forced_token_ids)

    if not token_sequences:
        raise ValueError("source plan does not contain any forced tokens to sample from")
    return token_sequences


def build_derived_identifier(base: str, *, index: int, size: int) -> str:
    width = max(2, len(str(size)))
    return f"{base}__derived{index + 1:0{width}d}"


def build_request_progress_message(
    *,
    plan_label: str,
    worker_index: int,
    worker_count: int,
    request_index: int,
    request_count: int,
) -> str:
    return (
        f"{plan_label} "
        f"worker {worker_index + 1}/{worker_count} "
        f"request {request_index + 1}/{request_count}"
    )


def sample_token_window(
    *,
    rng: Random,
    token_corpus: list[int],
    token_count: int,
) -> list[int]:
    if token_count <= 0:
        raise ValueError("replacement token window size must be > 0")
    if len(token_corpus) < token_count:
        raise ValueError(
            "replacement token corpus is too short: "
            f"need at least {token_count} tokens, found {len(token_corpus)}"
        )
    max_offset = len(token_corpus) - token_count
    start_offset = 0 if max_offset == 0 else rng.randint(0, max_offset)
    return list(token_corpus[start_offset : start_offset + token_count])


def validate_replay_stable_sequence(
    *,
    model_name: str,
    forced_token_ids: list[int],
    detokenize_fn: Callable[[str, list[int]], str],
    tokenize_fn: Callable[[str, str], list[int]],
) -> str:
    if not forced_token_ids:
        raise ValueError("cannot validate an empty forced-token sequence")

    expected_response_text = detokenize_fn(model_name, list(forced_token_ids))
    if expected_response_text.endswith("\ufffd"):
        raise ValueError(
            "generated forced-token sequence detokenizes to text ending with "
            "a Unicode replacement character"
        )
    canonical_tokens = tokenize_fn(model_name, expected_response_text)
    if not canonical_tokens:
        raise ValueError(
            "generated forced-token sequence detokenized to text that tokenizes "
            "back to an empty sequence"
        )
    if canonical_tokens != list(forced_token_ids):
        raise ValueError(
            "generated forced-token sequence does not round-trip without "
            "changing token ids"
        )
    return expected_response_text


def sample_replay_stable_sequence(
    *,
    rng: Random,
    token_corpus: list[int],
    token_count: int,
    model_name: str,
    detokenize_fn: Callable[[str, list[int]], str],
    tokenize_fn: Callable[[str, str], list[int]],
) -> tuple[list[int], str]:
    last_error: Exception | None = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        sampled_token_ids = sample_token_window(
            rng=rng,
            token_corpus=token_corpus,
            token_count=token_count,
        )
        try:
            expected_response_text = validate_replay_stable_sequence(
                model_name=model_name,
                forced_token_ids=sampled_token_ids,
                detokenize_fn=detokenize_fn,
                tokenize_fn=tokenize_fn,
            )
            return sampled_token_ids, expected_response_text
        except ValueError as exc:
            last_error = exc

    raise ValueError(
        "unable to generate replay-stable forced_token_ids after "
        f"{MAX_SAMPLE_ATTEMPTS} attempts"
        + (f": {last_error}" if last_error is not None else "")
    )


def derive_plan_payload(
    *,
    plan_payload: dict[str, Any],
    seed: int,
    size: int,
    detokenize_fn: Callable[[str, list[int]], str],
    tokenize_fn: Callable[[str, str], list[int]],
    replacement_tokens_provider: Callable[[str], list[int]],
    source_plan_path: Path | None = None,
    replacement_text_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
    progress_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_single_worker_plan(plan_payload)
    if size <= 0:
        raise ValueError("size must be > 0")

    token_sequences = collect_source_token_sequences(plan_payload)
    source_worker = plan_payload["workers"][0]
    source_requests = source_worker["requests"]
    assert isinstance(source_requests, list)
    source_expected_response_texts = [
        extract_expected_response_text(request_payload)
        if isinstance(request_payload, dict)
        else None
        for request_payload in source_requests
    ]
    source_initial_request_payload = source_requests[0]
    assert isinstance(source_initial_request_payload, dict)
    source_initial_messages = (
        list(extract_request_messages(source_initial_request_payload) or [])
    )
    source_initial_message_model_name = resolve_model_name(
        request_payload=source_initial_request_payload,
        plan_payload=plan_payload,
    )
    source_initial_message_texts: list[str | None] = []
    source_initial_message_token_counts: list[int | None] = []
    for message_payload in source_initial_messages:
        if not isinstance(message_payload, dict):
            source_initial_message_texts.append(None)
            source_initial_message_token_counts.append(None)
            continue
        source_text = extract_message_text(message_payload)
        source_initial_message_texts.append(source_text)
        if source_text is None:
            source_initial_message_token_counts.append(None)
            continue
        source_tokens = tokenize_fn(source_initial_message_model_name, source_text)
        source_initial_message_token_counts.append(len(source_tokens))
    source_worker_id = str(
        source_worker.get("worker_id") or source_worker.get("trial_id") or "worker"
    )
    source_trial_id = str(source_worker.get("trial_id") or source_worker_id)
    source_api_token_value = source_worker.get("api_token")
    if isinstance(source_api_token_value, str) and source_api_token_value:
        source_api_token = source_api_token_value
    else:
        source_api_token = f"derived-single-plan:{safe_name(source_worker_id)}"

    rng = Random(seed)
    derived_workers: list[dict[str, Any]] = []
    request_count = len(token_sequences)
    total_generated_token_count = 0
    total_generated_initial_message_token_count = 0
    replacement_token_count_by_model: dict[str, int] = {}
    plan_progress_label = progress_label or (
        source_plan_path.name if source_plan_path is not None else source_worker_id
    )

    for worker_index in range(size):
        worker_copy = copy.deepcopy(source_worker)
        worker_copy["launch_priority"] = worker_index
        worker_copy["worker_id"] = build_derived_identifier(
            source_worker_id,
            index=worker_index,
            size=size,
        )
        worker_copy["trial_id"] = build_derived_identifier(
            source_trial_id,
            index=worker_index,
            size=size,
        )
        worker_copy["api_token"] = build_derived_identifier(
            source_api_token,
            index=worker_index,
            size=size,
        )
        worker_copy["api_token_hash"] = sha256_hex(worker_copy["api_token"])
        worker_copy["derived_from_worker_id"] = source_worker_id
        worker_copy["derived_from_trial_id"] = source_trial_id
        worker_copy["derived_worker_index"] = worker_index

        requests = worker_copy.get("requests")
        assert isinstance(requests, list)
        replacement_initial_message_texts: list[str | None] = []
        if source_initial_messages:
            initial_message_rng = Random(
                int(
                    sha256_hex(
                        f"derived-single-plan:initial-messages:{seed}:{source_worker_id}:{worker_index}"
                    ),
                    16,
                )
            )
            replacement_token_corpus = replacement_tokens_provider(
                source_initial_message_model_name
            )
            replacement_token_count_by_model.setdefault(
                source_initial_message_model_name,
                len(replacement_token_corpus),
            )
            for message_index, token_count in enumerate(source_initial_message_token_counts):
                source_text = source_initial_message_texts[message_index]
                if source_text is None or token_count is None:
                    replacement_initial_message_texts.append(None)
                    continue
                if token_count <= 0:
                    replacement_initial_message_texts.append(source_text)
                    continue
                _, replacement_text = sample_replay_stable_sequence(
                    rng=initial_message_rng,
                    token_corpus=replacement_token_corpus,
                    token_count=token_count,
                    model_name=source_initial_message_model_name,
                    detokenize_fn=detokenize_fn,
                    tokenize_fn=tokenize_fn,
                )
                replacement_initial_message_texts.append(replacement_text)
                total_generated_initial_message_token_count += token_count
        replacement_expected_response_texts: list[str | None] = []
        for request_index, request_payload in enumerate(requests):
            if not isinstance(request_payload, dict):
                raise ValueError(f"request[{request_index}] must be an object")
            source_forced_token_ids = token_sequences[request_index]
            model_name = resolve_model_name(
                request_payload=request_payload,
                plan_payload=plan_payload,
            )
            replacement_token_corpus = replacement_tokens_provider(model_name)
            replacement_token_count_by_model.setdefault(
                model_name,
                len(replacement_token_corpus),
            )
            stable_forced_token_ids, expected_response_text = sample_replay_stable_sequence(
                rng=rng,
                token_corpus=replacement_token_corpus,
                token_count=len(source_forced_token_ids),
                model_name=model_name,
                detokenize_fn=detokenize_fn,
                tokenize_fn=tokenize_fn,
            )
            total_generated_token_count += len(stable_forced_token_ids)
            set_forced_token_ids(request_payload, stable_forced_token_ids)
            request_payload["expected_response_text"] = expected_response_text
            replacement_expected_response_texts.append(expected_response_text)
            if progress_callback is not None:
                progress_callback(
                    build_request_progress_message(
                        plan_label=plan_progress_label,
                        worker_index=worker_index,
                        worker_count=size,
                        request_index=request_index,
                        request_count=request_count,
                    )
                )
        for request_index, request_payload in enumerate(requests):
            if not isinstance(request_payload, dict):
                continue
            rewrite_initial_messages(
                request_payload=request_payload,
                source_initial_messages=source_initial_messages,
                replacement_initial_message_texts=replacement_initial_message_texts,
            )
            rewrite_request_message_history(
                request_payload=request_payload,
                source_expected_response_texts=source_expected_response_texts[:request_index],
                replacement_expected_response_texts=(
                    replacement_expected_response_texts[:request_index]
                ),
            )
        derived_workers.append(worker_copy)

    derived_payload = copy.deepcopy(plan_payload)
    derived_payload["workers"] = derived_workers
    derived_payload["is_derived"] = True
    derived_payload["derived_single_plan"] = {
        "seed": seed,
        "size": size,
        "derived_at": now_iso8601_utc(),
        "source_plan_path": str(source_plan_path) if source_plan_path is not None else None,
        "source_worker_id": source_worker_id,
        "source_trial_id": source_trial_id,
        "source_worker_count": 1,
        "initial_message_count": len(source_initial_messages),
        "request_count_per_worker": request_count,
        "output_worker_count": size,
        "replacement_text_path": (
            str(replacement_text_path) if replacement_text_path is not None else None
        ),
        "replacement_token_count_by_model": replacement_token_count_by_model,
        "total_generated_token_count": total_generated_token_count,
        "total_generated_initial_message_token_count": (
            total_generated_initial_message_token_count
        ),
    }
    summary = {
        "status": "ok",
        "seed": seed,
        "size": size,
        "source_plan_path": str(source_plan_path) if source_plan_path is not None else None,
        "source_worker_id": source_worker_id,
        "initial_message_count": len(source_initial_messages),
        "request_count_per_worker": request_count,
        "output_worker_count": size,
        "replacement_text_path": (
            str(replacement_text_path) if replacement_text_path is not None else None
        ),
        "replacement_token_count_by_model": replacement_token_count_by_model,
        "total_generated_token_count": total_generated_token_count,
        "total_generated_initial_message_token_count": (
            total_generated_initial_message_token_count
        ),
    }
    return derived_payload, summary


def build_output_plan_path(*, plan_path: Path, seed: int, size: int) -> Path:
    return plan_path.parent / f"seed-{seed}-size-{size}.{plan_path.name}"


def build_worker_label(worker_payload: Any, worker_index: int) -> str:
    if isinstance(worker_payload, dict):
        worker_id = worker_payload.get("worker_id")
        if isinstance(worker_id, str) and worker_id:
            return worker_id
        trial_id = worker_payload.get("trial_id")
        if isinstance(trial_id, str) and trial_id:
            return trial_id
    return f"worker[{worker_index}]"


def canonicalize_messages(messages: list[Any]) -> str:
    return json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def verify_derived_plan_payload(
    *,
    source_plan_payload: dict[str, Any],
    derived_plan_payload: dict[str, Any],
    tokenize_fn: Callable[[str, str], list[int]],
) -> dict[str, Any]:
    validate_single_worker_plan(source_plan_payload)

    derived_workers = derived_plan_payload.get("workers")
    if not isinstance(derived_workers, list) or not derived_workers:
        raise ValueError("derived plan must contain at least one worker")

    source_worker = source_plan_payload["workers"][0]
    source_requests = source_worker["requests"]
    assert isinstance(source_requests, list)

    token_count_cache: dict[tuple[str, str], int] = {}

    def _token_count(model_name: str, text: str) -> int:
        cache_key = (model_name, text)
        cached = token_count_cache.get(cache_key)
        if cached is not None:
            return cached
        token_count = len(tokenize_fn(model_name, text))
        token_count_cache[cache_key] = token_count
        return token_count

    source_initial_messages = list(extract_request_messages(source_requests[0]) or [])
    source_request_specs: list[dict[str, Any]] = []
    for request_index, source_request_payload in enumerate(source_requests):
        if not isinstance(source_request_payload, dict):
            raise ValueError(f"source request[{request_index}] must be an object")
        source_messages = extract_request_messages(source_request_payload)
        if source_messages is None:
            raise ValueError(f"source request[{request_index}] is missing body.messages")
        model_name = resolve_model_name(
            request_payload=source_request_payload,
            plan_payload=source_plan_payload,
        )
        source_forced_token_ids = extract_forced_token_ids(source_request_payload)
        if source_forced_token_ids is None:
            raise ValueError(
                f"source request[{request_index}] is missing deterministic forced_token_ids"
            )

        message_specs: list[dict[str, Any]] = []
        prompt_token_count = 0
        assistant_ordinal = 0
        for message_index, source_message_payload in enumerate(source_messages):
            if not isinstance(source_message_payload, dict):
                raise ValueError(
                    f"source request[{request_index}] message[{message_index}] must be an object"
                )
            role = source_message_payload.get("role")
            if not isinstance(role, str):
                raise ValueError(
                    f"source request[{request_index}] message[{message_index}] is missing role"
                )
            source_text = extract_message_text(source_message_payload)
            token_count = (
                _token_count(model_name, source_text) if source_text is not None else None
            )
            if token_count is not None:
                prompt_token_count += token_count

            is_initial_message = (
                message_index < len(source_initial_messages)
                and source_message_payload == source_initial_messages[message_index]
            )
            if is_initial_message:
                message_specs.append(
                    {
                        "kind": "initial",
                        "role": role,
                        "source_message": source_message_payload,
                        "token_count": token_count,
                    }
                )
                continue

            if role == "assistant":
                message_specs.append(
                    {
                        "kind": "assistant",
                        "role": role,
                        "assistant_ordinal": assistant_ordinal,
                        "token_count": token_count,
                    }
                )
                assistant_ordinal += 1
                continue

            message_specs.append(
                {
                    "kind": "literal",
                    "role": role,
                    "source_message": source_message_payload,
                    "token_count": token_count,
                }
            )

        source_request_specs.append(
            {
                "model_name": model_name,
                "generation_token_count": len(source_forced_token_ids),
                "prompt_token_count": prompt_token_count,
                "message_specs": message_specs,
            }
        )

    request_count_mismatch_workers: list[str] = []
    generation_mismatch_workers: list[str] = []
    prompt_mismatch_workers: list[str] = []

    for worker_index, worker_payload in enumerate(derived_workers):
        worker_label = build_worker_label(worker_payload, worker_index)
        if not isinstance(worker_payload, dict):
            request_count_mismatch_workers.append(worker_label)
            generation_mismatch_workers.append(worker_label)
            prompt_mismatch_workers.append(worker_label)
            continue
        derived_requests = worker_payload.get("requests")
        if not isinstance(derived_requests, list) or len(derived_requests) != len(source_requests):
            request_count_mismatch_workers.append(worker_label)
            if not isinstance(derived_requests, list):
                generation_mismatch_workers.append(worker_label)
                prompt_mismatch_workers.append(worker_label)
            continue

        derived_expected_response_texts = [
            extract_expected_response_text(request_payload)
            if isinstance(request_payload, dict)
            else None
            for request_payload in derived_requests
        ]

        worker_generation_ok = True
        worker_prompt_ok = True
        for request_index, (derived_request_payload, request_spec) in enumerate(
            zip(derived_requests, source_request_specs, strict=False)
        ):
            if not isinstance(derived_request_payload, dict):
                worker_generation_ok = False
                worker_prompt_ok = False
                continue

            derived_forced_token_ids = extract_forced_token_ids(derived_request_payload)
            if derived_forced_token_ids is None:
                worker_generation_ok = False
            elif len(derived_forced_token_ids) != request_spec["generation_token_count"]:
                worker_generation_ok = False

            derived_messages = extract_request_messages(derived_request_payload)
            if derived_messages is None or len(derived_messages) != len(request_spec["message_specs"]):
                worker_prompt_ok = False
                continue

            derived_prompt_token_count = 0
            for message_index, message_spec in enumerate(request_spec["message_specs"]):
                derived_message_payload = derived_messages[message_index]
                if not isinstance(derived_message_payload, dict):
                    worker_prompt_ok = False
                    continue

                if derived_message_payload.get("role") != message_spec["role"]:
                    worker_prompt_ok = False

                kind = message_spec["kind"]
                if kind == "literal":
                    if derived_message_payload != message_spec["source_message"]:
                        worker_prompt_ok = False
                    token_count = message_spec["token_count"]
                    if token_count is not None:
                        derived_prompt_token_count += int(token_count)
                    continue

                derived_text = extract_message_text(derived_message_payload)
                if kind == "initial":
                    if derived_text is None:
                        if derived_message_payload != message_spec["source_message"]:
                            worker_prompt_ok = False
                    else:
                        token_count = _token_count(request_spec["model_name"], derived_text)
                        source_token_count = message_spec["token_count"]
                        if source_token_count is None or token_count != source_token_count:
                            worker_prompt_ok = False
                        derived_prompt_token_count += token_count
                    continue

                if kind == "assistant":
                    assistant_ordinal = int(message_spec["assistant_ordinal"])
                    expected_text = (
                        derived_expected_response_texts[assistant_ordinal]
                        if assistant_ordinal < len(derived_expected_response_texts)
                        else None
                    )
                    if expected_text is None or derived_text != expected_text:
                        worker_prompt_ok = False
                    if derived_text is not None:
                        token_count = _token_count(request_spec["model_name"], derived_text)
                        source_token_count = message_spec["token_count"]
                        if source_token_count is None or token_count != source_token_count:
                            worker_prompt_ok = False
                        derived_prompt_token_count += token_count
                    continue

                worker_prompt_ok = False

            if derived_prompt_token_count != request_spec["prompt_token_count"]:
                worker_prompt_ok = False

        if not worker_generation_ok:
            generation_mismatch_workers.append(worker_label)
        if not worker_prompt_ok:
            prompt_mismatch_workers.append(worker_label)

    unique_history_counts: list[int] = []
    duplicate_history_request_indices: list[int] = []
    for request_index in range(len(source_requests)):
        message_signatures: list[str] = []
        for worker_payload in derived_workers:
            if not isinstance(worker_payload, dict):
                continue
            derived_requests = worker_payload.get("requests")
            if not isinstance(derived_requests, list) or request_index >= len(derived_requests):
                continue
            request_payload = derived_requests[request_index]
            if not isinstance(request_payload, dict):
                continue
            messages = extract_request_messages(request_payload)
            if messages is None:
                continue
            message_signatures.append(sha256_hex(canonicalize_messages(messages)))
        unique_count = len(set(message_signatures))
        unique_history_counts.append(unique_count)
        if unique_count != len(message_signatures):
            duplicate_history_request_indices.append(request_index)

    verification = {
        "status": "ok",
        "verified_at": now_iso8601_utc(),
        "source_request_count": len(source_requests),
        "derived_worker_count": len(derived_workers),
        "request_count_per_worker_matches": not request_count_mismatch_workers,
        "generation_length_distribution_matches": not generation_mismatch_workers,
        "prompt_length_distribution_matches": not prompt_mismatch_workers,
        "cross_worker_history_unique": not duplicate_history_request_indices,
        "request_count_mismatch_worker_count": len(request_count_mismatch_workers),
        "generation_mismatch_worker_count": len(generation_mismatch_workers),
        "prompt_mismatch_worker_count": len(prompt_mismatch_workers),
        "request_count_mismatch_workers": request_count_mismatch_workers[:10],
        "generation_mismatch_workers": generation_mismatch_workers[:10],
        "prompt_mismatch_workers": prompt_mismatch_workers[:10],
        "min_unique_worker_history_count": (
            min(unique_history_counts) if unique_history_counts else 0
        ),
        "max_duplicate_worker_history_count": (
            max(len(derived_workers) - unique_count for unique_count in unique_history_counts)
            if unique_history_counts
            else 0
        ),
        "duplicate_history_request_indices": duplicate_history_request_indices[:10],
    }
    if not (
        verification["request_count_per_worker_matches"]
        and verification["generation_length_distribution_matches"]
        and verification["prompt_length_distribution_matches"]
        and verification["cross_worker_history_unique"]
    ):
        verification["status"] = "failed"
    return verification


def estimate_plan_work_units(plan_payload: Any, *, size: int) -> int:
    if size <= 0:
        return 0
    if not isinstance(plan_payload, dict):
        return 0
    workers = plan_payload.get("workers")
    if not isinstance(workers, list) or len(workers) != 1:
        return 0
    worker = workers[0]
    if not isinstance(worker, dict):
        return 0
    requests = worker.get("requests")
    if not isinstance(requests, list):
        return 0
    return len(requests) * size


def is_source_plan_candidate(plan_payload: Any) -> bool:
    if not isinstance(plan_payload, dict):
        return False
    if plan_payload.get("schema_version") != REPLAY_PLAN_SCHEMA_VERSION:
        return False
    if plan_payload.get("is_derived") is True:
        return False
    if extract_compile_single_trail(plan_payload) is None:
        return False
    workers = plan_payload.get("workers")
    if not isinstance(workers, list) or len(workers) != 1:
        return False
    worker = workers[0]
    if not isinstance(worker, dict):
        return False
    requests = worker.get("requests")
    if not isinstance(requests, list) or not requests:
        return False
    return True


def find_candidate_source_plan_paths(root: Path) -> tuple[list[Path], int]:
    candidates: list[Path] = []
    scanned_count = 0
    for path in sorted(root.rglob("replay-plan*.json")):
        if not path.is_file():
            continue
        scanned_count += 1
        try:
            payload = read_json(path)
        except Exception:
            continue
        if is_source_plan_candidate(payload):
            candidates.append(path.resolve())
    return candidates, scanned_count


def read_replacement_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text == "":
        raise ValueError("replacement text file must not be empty")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_derived_single_plan.py",
        description=(
            "Take a single-worker replay plan and derive a multi-worker plan by "
            "cloning the source worker and replacing each request's forced_token_ids "
            "with contiguous token windows sampled from a user-provided text file."
        ),
    )
    parser.add_argument(
        "--plan",
        required=True,
        help=(
            "Path to a replay-plan JSON file containing exactly one worker, or a "
            "directory to recursively scan for such plans in batch mode."
        ),
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Random seed for deterministic forced-token sampling.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Number of workers to generate in the derived plan (default: {DEFAULT_SIZE}).",
    )
    parser.add_argument(
        "--port-profile",
        required=True,
        type=int,
        help="Port profile numeric ID from configs/port_profiles.toml.",
    )
    parser.add_argument(
        "--replacement-text-file",
        required=True,
        help=(
            "Path to a UTF-8 text file. The helper tokenizes this text and samples "
            "contiguous token windows from it for request replacement."
        ),
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help=(
            "Timeout in seconds for each /detokenize or /tokenize call "
            f"(default: {DEFAULT_REQUEST_TIMEOUT_S})."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output plan path. Default: write next to --plan as "
            "seed-<seed>-size-<size>.<original-name>."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    plan_path = Path(args.plan).expanduser().resolve()
    if not plan_path.exists():
        print(f"error: invalid --plan: {plan_path}", file=sys.stderr)
        return 2
    replacement_text_path = Path(args.replacement_text_file).expanduser().resolve()
    if not replacement_text_path.is_file():
        print(
            f"error: invalid --replacement-text-file: {replacement_text_path}",
            file=sys.stderr,
        )
        return 2
    if args.size <= 0:
        print("error: --size must be > 0", file=sys.stderr)
        return 2
    if args.request_timeout_s <= 0:
        print("error: --request-timeout-s must be > 0", file=sys.stderr)
        return 2

    try:
        replacement_text = read_replacement_text(replacement_text_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    detokenize_endpoint = resolve_detokenize_endpoint(args.port_profile)
    tokenize_endpoint = resolve_tokenize_endpoint(args.port_profile)

    def _detokenize(model_name: str, forced_token_ids: list[int]) -> str:
        return detokenize_tokens(
            detokenize_endpoint=detokenize_endpoint,
            model_name=model_name,
            forced_token_ids=forced_token_ids,
            timeout_s=float(args.request_timeout_s),
        )

    def _tokenize(model_name: str, text: str) -> list[int]:
        return tokenize_text(
            tokenize_endpoint=tokenize_endpoint,
            model_name=model_name,
            text=text,
            timeout_s=float(args.request_timeout_s),
        )

    replacement_tokens_by_model: dict[str, list[int]] = {}

    def _replacement_tokens(model_name: str) -> list[int]:
        cached = replacement_tokens_by_model.get(model_name)
        if cached is not None:
            return cached
        tokens = _tokenize(model_name, replacement_text)
        if not tokens:
            raise ValueError(
                "replacement text file tokenized to an empty token sequence "
                f"for model {model_name!r}"
            )
        replacement_tokens_by_model[model_name] = tokens
        return tokens

    if plan_path.is_dir():
        if args.output is not None:
            print("error: --output is only supported when --plan is a file", file=sys.stderr)
            return 2

        candidate_paths, scanned_count = find_candidate_source_plan_paths(plan_path)
        if not candidate_paths:
            print(
                f"error: no non-derived single-worker replay plans found under {plan_path}",
                file=sys.stderr,
            )
            return 2

        total_work_units = 0
        for candidate_path in candidate_paths:
            total_work_units += estimate_plan_work_units(read_json(candidate_path), size=args.size)
        progress = ProgressBar(total=total_work_units, label="Deriving plans")

        derived_summaries: list[dict[str, Any]] = []
        failed_summaries: list[dict[str, Any]] = []
        try:
            for plan_index, candidate_path in enumerate(candidate_paths):
                plan_label = (
                    f"plan {plan_index + 1}/{len(candidate_paths)} {candidate_path.name}"
                )
                progress.set_message(plan_label)
                try:
                    source_plan_payload = read_json(candidate_path)
                    derived_payload, summary = derive_plan_payload(
                        plan_payload=source_plan_payload,
                        seed=args.seed,
                        size=args.size,
                        detokenize_fn=_detokenize,
                        tokenize_fn=_tokenize,
                        replacement_tokens_provider=_replacement_tokens,
                        source_plan_path=candidate_path,
                        replacement_text_path=replacement_text_path,
                        progress_callback=lambda message, progress=progress: progress.advance(
                            message=message
                        ),
                        progress_label=plan_label,
                    )
                    verification = verify_derived_plan_payload(
                        source_plan_payload=source_plan_payload,
                        derived_plan_payload=derived_payload,
                        tokenize_fn=_tokenize,
                    )
                    summary["verification"] = verification
                    if verification["status"] != "ok":
                        raise ValueError(
                            "derived plan verification failed: "
                            f"{json.dumps(verification, ensure_ascii=True)}"
                        )
                    output_path = build_output_plan_path(
                        plan_path=candidate_path,
                        seed=args.seed,
                        size=args.size,
                    )
                    write_json(output_path, derived_payload)
                    summary["output_plan_path"] = str(output_path)
                    derived_summaries.append(summary)
                except Exception as exc:  # noqa: BLE001
                    failed_summaries.append(
                        {
                            "source_plan_path": str(candidate_path),
                            "error": str(exc),
                        }
                    )
        finally:
            progress.close()

        batch_summary = {
            "status": "ok" if not failed_summaries else "partial_failure",
            "mode": "batch",
            "seed": args.seed,
            "size": args.size,
            "plan_root": str(plan_path),
            "port_profile_id": int(args.port_profile),
            "replacement_text_file": str(replacement_text_path),
            "detokenize_endpoint": detokenize_endpoint,
            "tokenize_endpoint": tokenize_endpoint,
            "scanned_replay_plan_count": scanned_count,
            "candidate_plan_count": len(candidate_paths),
            "derived_plan_count": len(derived_summaries),
            "failed_plan_count": len(failed_summaries),
            "derived_plans": derived_summaries,
            "failed_plans": failed_summaries,
        }
        print(json.dumps(batch_summary, indent=2, ensure_ascii=True))
        return 0 if not failed_summaries else 1

    if not plan_path.is_file():
        print(f"error: invalid --plan: {plan_path}", file=sys.stderr)
        return 2

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else build_output_plan_path(plan_path=plan_path, seed=args.seed, size=args.size)
    )

    plan_payload = read_json(plan_path)
    progress = ProgressBar(
        total=estimate_plan_work_units(plan_payload, size=args.size),
        label="Deriving plan",
    )
    try:
        progress.set_message(plan_path.name)
        derived_payload, summary = derive_plan_payload(
            plan_payload=plan_payload,
            seed=args.seed,
            size=args.size,
            detokenize_fn=_detokenize,
            tokenize_fn=_tokenize,
            replacement_tokens_provider=_replacement_tokens,
            source_plan_path=plan_path,
            replacement_text_path=replacement_text_path,
            progress_callback=lambda message, progress=progress: progress.advance(message=message),
            progress_label=plan_path.name,
        )
        verification = verify_derived_plan_payload(
            source_plan_payload=plan_payload,
            derived_plan_payload=derived_payload,
            tokenize_fn=_tokenize,
        )
        summary["verification"] = verification
        if verification["status"] != "ok":
            raise ValueError(
                "derived plan verification failed: "
                f"{json.dumps(verification, ensure_ascii=True)}"
            )
    except Exception as exc:  # noqa: BLE001
        progress.close()
        print(f"error: {exc}", file=sys.stderr)
        return 2
    progress.close()

    write_json(output_path, derived_payload)
    summary["mode"] = "single"
    summary["output_plan_path"] = str(output_path)
    summary["port_profile_id"] = int(args.port_profile)
    summary["replacement_text_file"] = str(replacement_text_path)
    summary["detokenize_endpoint"] = detokenize_endpoint
    summary["tokenize_endpoint"] = tokenize_endpoint
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
