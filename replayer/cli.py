"""CLI for compiling and replaying gateway-profiled jobs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from contextlib import AbstractContextManager
from random import Random
import sys
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib import error as url_error
from urllib.parse import urlparse
from urllib import request as url_request

from replayer.port_profiles import build_replay_target_from_port_profile

try:
    from rich.progress import (
        BarColumn,
        Progress as RichProgress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
except ImportError:  # pragma: no cover
    RichProgress = None


def now_iso8601_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def to_iso8601_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def safe_name(value: str) -> str:
    chars: list[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars) or "worker"


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


class _NullProgress(AbstractContextManager["_NullProgress"]):
    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def add_task(self, description: str, *, total: int, **fields: Any) -> int:
        return 0

    def update(self, task_id: int, *, advance: int = 0, **fields: Any) -> None:
        return None


def create_replay_progress() -> Any:
    if RichProgress is None:
        return _NullProgress()
    return RichProgress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn(
            "launched={task.fields[launched]} "
            "active={task.fields[active]} "
            "failed={task.fields[failed]}"
        ),
        TimeElapsedColumn(),
        transient=False,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object in {path}, got: {type(parsed)!r}")
        records.append(parsed)
    return records


def parse_toml(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"TOML root must be table: {path}")
    return data


def join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def normalize_request_path_for_api_base(api_base: str, path: str) -> str:
    """Normalize request path so api_base + path does not duplicate /v1 segments.

    Example:
    - api_base = http://host:11457/v1
    - path = v1/chat/completions
    => chat/completions
    """
    normalized_path = path.strip().lstrip("/")
    if not normalized_path:
        return normalized_path

    base_path = urlparse(api_base).path.rstrip("/")
    if base_path.endswith("/v1") and normalized_path.startswith("v1/"):
        return normalized_path[len("v1/") :]
    if base_path.endswith("/v1") and normalized_path == "v1":
        return ""
    return normalized_path


def http_json(
    *,
    method: str,
    url: str,
    payload: Any,
    timeout_s: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    req_headers = {"content-type": "application/json"}
    if headers:
        req_headers.update(headers)
    body_bytes = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = url_request.Request(
        url=url,
        data=body_bytes,
        headers=req_headers,
        method=method.upper(),
    )
    try:
        with url_request.urlopen(req, timeout=timeout_s) as resp:
            status = int(resp.getcode())
            raw = resp.read().decode("utf-8", errors="replace")
    except url_error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")
    except url_error.URLError as exc:
        raise RuntimeError(f"HTTP request failed: {method} {url}: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"HTTP request timed out: {method} {url}") from exc

    if not raw:
        return status, {}
    try:
        return status, json.loads(raw)
    except Exception:
        return status, {"raw_body": raw}


def require_file(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Missing required file: {path}")
    if not path.is_file():
        raise ValueError(f"Expected file path: {path}")


def extract_api_token_from_command(command: list[Any]) -> str | None:
    tokens = [str(item) for item in command]
    for idx, token in enumerate(tokens):
        if token == "--api-token" and idx + 1 < len(tokens):
            return tokens[idx + 1]
    return None


def parse_model_and_api_base_from_tokens(tokens: list[str]) -> tuple[str | None, str | None]:
    model_name: str | None = None
    api_base: str | None = None
    for idx, token in enumerate(tokens):
        if token == "--model" and idx + 1 < len(tokens):
            model_name = tokens[idx + 1]
        if token == "--agent-kwarg" and idx + 1 < len(tokens):
            kv_value = tokens[idx + 1]
            if kv_value.startswith("api_base="):
                api_base = kv_value.split("=", 1)[1]
    return model_name, api_base


def detect_backend(config: dict[str, Any], backend_override: str | None) -> str:
    if backend_override:
        return backend_override.strip().lower()

    backend_section = config.get("backend")
    if isinstance(backend_section, dict):
        backend_name = backend_section.get("name")
        if isinstance(backend_name, str) and backend_name.strip():
            return backend_name.strip().lower()

    run_section = config.get("run")
    if isinstance(run_section, dict):
        backend_name = run_section.get("driver_backend")
        if isinstance(backend_name, str) and backend_name.strip():
            return backend_name.strip().lower()

    raise ValueError("Unable to detect backend from meta/config.toml")


def _to_int_or_default(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except Exception:
        return default


def parse_pattern_args_tokens(tokens: Any) -> dict[str, str]:
    if not isinstance(tokens, list):
        return {}
    items = [str(item).strip() for item in tokens if str(item).strip()]
    args: dict[str, str] = {}
    index = 0
    while index < len(items):
        token = items[index]
        if not token.startswith("--"):
            index += 1
            continue
        key_value = token[2:]
        if "=" in key_value:
            key, value = key_value.split("=", 1)
            args[key] = value
            index += 1
            continue
        key = key_value
        value = "true"
        if index + 1 < len(items) and not items[index + 1].startswith("--"):
            value = items[index + 1]
            index += 1
        args[key] = value
        index += 1
    return args


def build_launch_policy_from_config(config: dict[str, Any]) -> dict[str, Any]:
    run_section = config.get("run")
    if not isinstance(run_section, dict):
        raise ValueError("Missing [run] section in meta/config.toml")

    pattern_raw = run_section.get("pattern")
    if isinstance(pattern_raw, str) and pattern_raw.strip():
        pattern_name = pattern_raw.strip().lower()
    else:
        pattern_name = "eager"
    pattern_tokens = run_section.get("pattern_args")
    pattern_args = parse_pattern_args_tokens(pattern_tokens)

    max_concurrent = _to_int_or_default(run_section.get("max_concurrent"), default=1)
    if max_concurrent <= 0:
        max_concurrent = 1

    seed_value = run_section.get("seed")
    seed: int | None = None
    if isinstance(seed_value, int) and not isinstance(seed_value, bool):
        seed = seed_value

    if pattern_name == "eager":
        pattern_payload: dict[str, Any] = {"name": "eager"}
    elif pattern_name in {"poisson", "possion"}:
        rate_value = (
            pattern_args.get("rate")
            or pattern_args.get("arrival-rate")
            or pattern_args.get("arrival_rate")
            or pattern_args.get("lambda")
        )
        if rate_value is None:
            mean_value = (
                pattern_args.get("mean-interval-s")
                or pattern_args.get("mean_interval_s")
                or pattern_args.get("interval-s")
                or pattern_args.get("interval_s")
            )
            if mean_value is None:
                raise ValueError(
                    "Poisson pattern requires one of: "
                    "--rate=<arrivals_per_second> or --mean-interval-s=<seconds>"
                )
            mean_interval = float(mean_value)
            if mean_interval <= 0:
                raise ValueError("Poisson mean interval must be > 0")
            rate_per_second = 1.0 / mean_interval
        else:
            rate_per_second = float(rate_value)
            if rate_per_second <= 0:
                raise ValueError("Poisson rate must be > 0")
        pattern_payload = {
            "name": "poisson",
            "rate_per_second": rate_per_second,
            "mean_interval_s": 1.0 / rate_per_second,
        }
    else:
        raise ValueError(
            f"Unsupported run.pattern={pattern_name!r}; supported: eager, poisson"
        )

    return {
        "strategy": "config_ordered",
        "source": "meta/config.toml[run]",
        "max_concurrent": max_concurrent,
        "seed": seed,
        "pattern": pattern_payload,
        "pattern_args": pattern_args,
    }


def extract_harbor_target_config(
    config: dict[str, Any],
    results_entries: list[dict[str, Any]],
) -> tuple[str, str, str]:
    gateway_section = config.get("gateway")
    if not isinstance(gateway_section, dict):
        raise ValueError("Missing [gateway] section in meta/config.toml")
    gateway_url = gateway_section.get("url")
    if not isinstance(gateway_url, str) or not gateway_url.strip():
        raise ValueError("Missing [gateway].url in meta/config.toml")
    gateway_url = gateway_url.strip()

    model_name: str | None = None
    api_base: str | None = None
    backend_section = config.get("backend")
    if isinstance(backend_section, dict):
        forwarded_args = backend_section.get("forwarded_args")
        if isinstance(forwarded_args, list):
            tokens = [str(item) for item in forwarded_args]
            model_name, api_base = parse_model_and_api_base_from_tokens(tokens)

    if not model_name or not api_base:
        for entry in results_entries:
            command = entry.get("command")
            if not isinstance(command, list):
                continue
            tokens = [str(item) for item in command]
            command_model, command_api_base = parse_model_and_api_base_from_tokens(tokens)
            if not model_name and command_model:
                model_name = command_model
            if not api_base and command_api_base:
                api_base = command_api_base
            if model_name and api_base:
                break

    if not model_name:
        raise ValueError("Unable to extract --model for harbor backend")
    if not api_base:
        raise ValueError("Unable to extract api_base for harbor backend")

    return gateway_url, api_base, model_name


def parse_port_profile_id(value: Any, *, field_name: str = "port_profile_id") -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    raise ValueError(f"{field_name} must be an integer, got {value!r}")


def extract_runtime_port_profile_id(config: dict[str, Any]) -> int | None:
    runtime_section = config.get("runtime")
    if not isinstance(runtime_section, dict):
        return None
    return parse_port_profile_id(
        runtime_section.get("port_profile_id"),
        field_name="runtime.port_profile_id",
    )


def extract_gateway_enabled(config: dict[str, Any]) -> bool:
    gateway_section = config.get("gateway")
    if not isinstance(gateway_section, dict):
        return False
    raw_enabled = gateway_section.get("enabled")
    if isinstance(raw_enabled, bool):
        return raw_enabled
    gateway_url = gateway_section.get("url")
    return isinstance(gateway_url, str) and bool(gateway_url.strip())


def resolve_compile_target(
    *,
    config: dict[str, Any],
    results_entries: list[dict[str, Any]],
    port_profile_id_override: int | None,
    tokenize_endpoint_override: str | None,
) -> tuple[str, str, str, str, int | None]:
    gateway_url, api_base, configured_model = extract_harbor_target_config(
        config,
        results_entries,
    )
    source_port_profile_id = extract_runtime_port_profile_id(config)
    effective_port_profile_id = (
        port_profile_id_override
        if port_profile_id_override is not None
        else source_port_profile_id
    )

    if effective_port_profile_id is not None:
        gateway_enabled = extract_gateway_enabled(config)
        resolved_gateway_url, resolved_api_base, resolved_tokenize_endpoint = (
            build_replay_target_from_port_profile(
                effective_port_profile_id,
                gateway_enabled=gateway_enabled,
            )
        )
        gateway_url = resolved_gateway_url
        api_base = resolved_api_base
        tokenize_endpoint = tokenize_endpoint_override or resolved_tokenize_endpoint
    else:
        tokenize_endpoint = tokenize_endpoint_override or "http://127.0.0.1:11451/tokenize"

    return (
        gateway_url,
        api_base,
        configured_model,
        tokenize_endpoint,
        effective_port_profile_id,
    )


def resolve_replay_target(
    *,
    replay_target: dict[str, Any],
    port_profile_id_override: int | None,
) -> tuple[str, str | None, str | None, str | None]:
    api_base = replay_target.get("api_base")
    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError("Missing replay_target.api_base in plan")
    api_base = api_base.strip()

    gateway_url = replay_target.get("gateway_url")
    if isinstance(gateway_url, str):
        gateway_url = gateway_url.strip() or None
    else:
        gateway_url = None

    tokenize_endpoint = replay_target.get("tokenize_endpoint")
    if isinstance(tokenize_endpoint, str):
        tokenize_endpoint = tokenize_endpoint.strip() or None
    else:
        tokenize_endpoint = None

    effective_port_profile_id = (
        port_profile_id_override
        if port_profile_id_override is not None
        else parse_port_profile_id(
            replay_target.get("port_profile_id"),
            field_name="replay_target.port_profile_id",
        )
    )

    if effective_port_profile_id is not None:
        gateway_enabled = resolve_gateway_lifecycle_mode("auto", gateway_url, api_base)
        (
            resolved_gateway_url,
            api_base,
            resolved_tokenize_endpoint,
        ) = build_replay_target_from_port_profile(
            effective_port_profile_id,
            gateway_enabled=gateway_enabled,
        )
        gateway_url = resolved_gateway_url
        tokenize_endpoint = resolved_tokenize_endpoint

    return api_base, gateway_url, tokenize_endpoint, (
        str(effective_port_profile_id) if effective_port_profile_id is not None else None
    )


def resolve_t0(
    run_manifest: dict[str, Any],
    events_records: list[dict[str, Any]],
) -> tuple[datetime, str]:
    started_at = run_manifest.get("started_at")
    if isinstance(started_at, str) and started_at.strip():
        dt = parse_iso8601(started_at)
        return dt, "meta/run_manifest.json.started_at"

    for event in events_records:
        if event.get("event") != "gateway_job_start":
            continue
        value = event.get("time")
        if isinstance(value, str) and value.strip():
            dt = parse_iso8601(value)
            return dt, "meta/events.jsonl.gateway_job_start.time"

    raise ValueError("Unable to resolve T0 from run_manifest/events")


def extract_agent_start_time(lifecycle_records: list[dict[str, Any]]) -> datetime:
    for record in lifecycle_records:
        if record.get("event_type") != "agent_start":
            continue
        ts = record.get("timestamp")
        if isinstance(ts, str) and ts.strip():
            return parse_iso8601(ts)
    raise ValueError("Missing agent_start event in lifecycle.jsonl")


def extract_agent_end_time(lifecycle_records: list[dict[str, Any]]) -> datetime:
    for record in reversed(lifecycle_records):
        if record.get("event_type") != "agent_end":
            continue
        ts = record.get("timestamp")
        if isinstance(ts, str) and ts.strip():
            return parse_iso8601(ts)
    raise ValueError("Missing agent_end event in lifecycle.jsonl")


def extract_response_text(response_payload: Any) -> str:
    if isinstance(response_payload, str):
        return response_payload

    if isinstance(response_payload, dict):
        choices = response_payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                text_value = first.get("text")
                if isinstance(text_value, str):
                    return text_value
        output_text = response_payload.get("output_text")
        if isinstance(output_text, str):
            return output_text

    raise ValueError("Unable to extract response text for deterministic tokenization")


def tokenize_response_text(
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
            f"/tokenize failed for model={model_name!r}: HTTP {status}, payload={payload}"
        )
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected /tokenize payload type: {type(payload)!r}")
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(tok, int) for tok in tokens):
        raise ValueError(f"Invalid /tokenize tokens payload: {payload}")
    return tokens


def parse_agent_action_durations(jaeger_payload: dict[str, Any]) -> list[float]:
    data = jaeger_payload.get("data")
    if not isinstance(data, list) or not data:
        return []
    trace_payload = data[0]
    if not isinstance(trace_payload, dict):
        return []
    spans = trace_payload.get("spans")
    if not isinstance(spans, list):
        return []

    rows: list[tuple[int, int]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        if span.get("operationName") != "agent_action":
            continue
        start_time = span.get("startTime")
        duration = span.get("duration")
        if isinstance(start_time, int) and isinstance(duration, int):
            rows.append((start_time, duration))
    rows.sort(key=lambda item: item[0])
    return [max(0.0, duration / 1_000_000.0) for _, duration in rows]


def parse_request_start(record: dict[str, Any]) -> datetime:
    value = record.get("request_start_time") or record.get("span_start_time")
    if isinstance(value, str) and value.strip():
        return parse_iso8601(value)
    raise ValueError("Missing request_start_time/span_start_time in request record")


def parse_request_end(record: dict[str, Any]) -> datetime:
    value = record.get("request_end_time") or record.get("span_end_time")
    if isinstance(value, str) and value.strip():
        return parse_iso8601(value)
    raise ValueError("Missing request_end_time/span_end_time in request record")


def compute_agent_action_deltas(
    request_records: list[dict[str, Any]],
    agent_action_durations: list[float],
    final_agent_tail_s: float | None = None,
) -> list[float]:
    if not request_records:
        return []
    start_times = [parse_request_start(item) for item in request_records]
    end_times = [parse_request_end(item) for item in request_records]

    deltas: list[float] = []
    for index in range(len(request_records)):
        if index == len(request_records) - 1:
            if final_agent_tail_s is None:
                deltas.append(0.0)
            else:
                deltas.append(round(max(0.0, final_agent_tail_s), 6))
            continue
        if index < len(agent_action_durations):
            delta = max(0.0, agent_action_durations[index])
        else:
            delta = max(0.0, (start_times[index + 1] - end_times[index]).total_seconds())
        deltas.append(round(delta, 6))
    return deltas


def cmd_compile(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir).expanduser().resolve()
    if not job_dir.exists() or not job_dir.is_dir():
        raise ValueError(f"Invalid --job-dir: {job_dir}")

    config_path = job_dir / "meta" / "config.toml"
    run_manifest_path = job_dir / "meta" / "run_manifest.json"
    results_path = job_dir / "meta" / "results.json"
    events_path = job_dir / "meta" / "events.jsonl"
    for path in [config_path, run_manifest_path, results_path, events_path]:
        require_file(path)

    config = parse_toml(config_path)
    run_manifest = read_json(run_manifest_path)
    results_entries = read_json(results_path)
    if not isinstance(results_entries, list):
        raise ValueError(f"Expected list in {results_path}")
    events_records = read_jsonl(events_path)

    backend_name = detect_backend(config, args.backend)
    if backend_name != "harbor":
        raise ValueError(f"unsupported backend: {backend_name!r}")

    (
        gateway_url,
        api_base,
        configured_model,
        tokenize_endpoint,
        replay_port_profile_id,
    ) = resolve_compile_target(
        config=config,
        results_entries=results_entries,
        port_profile_id_override=args.port_profile_id,
        tokenize_endpoint_override=args.tokenize_endpoint,
    )
    launch_policy = build_launch_policy_from_config(config)

    t0_dt, t0_source = resolve_t0(run_manifest, events_records)
    t0_iso = to_iso8601_utc(t0_dt)

    trial_to_token: dict[str, str] = {}
    hash_to_trial: dict[str, str] = {}
    for entry in results_entries:
        if not isinstance(entry, dict):
            continue
        trial_id = entry.get("trial_id")
        command = entry.get("command")
        if not isinstance(trial_id, str) or not trial_id.strip():
            continue
        if not isinstance(command, list):
            continue
        api_token = extract_api_token_from_command(command)
        if not api_token:
            continue
        trial_to_token[trial_id] = api_token
        hash_to_trial[sha256_hex(api_token)] = trial_id

    gateway_output_dir = job_dir / "gateway-output"
    if not gateway_output_dir.exists() or not gateway_output_dir.is_dir():
        raise ValueError(f"Missing gateway-output directory: {gateway_output_dir}")

    worker_plans: list[dict[str, Any]] = []
    for run_dir in sorted(gateway_output_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        lifecycle_path = run_dir / "events" / "lifecycle.jsonl"
        requests_path = run_dir / "requests" / "model_inference.jsonl"
        jaeger_path = run_dir / "trace" / "jaeger_trace.json"
        for path in [manifest_path, lifecycle_path, requests_path, jaeger_path]:
            require_file(path)

        run_manifest_payload = read_json(manifest_path)
        if not isinstance(run_manifest_payload, dict):
            raise ValueError(f"Invalid JSON object: {manifest_path}")

        token_hash = run_manifest_payload.get("api_token_hash")
        run_start_raw = run_manifest_payload.get("run_start_time")
        if not isinstance(token_hash, str) or not token_hash:
            raise ValueError(f"Missing api_token_hash in {manifest_path}")
        if not isinstance(run_start_raw, str) or not run_start_raw:
            raise ValueError(f"Missing run_start_time in {manifest_path}")

        trial_id = hash_to_trial.get(token_hash)
        if not trial_id:
            raise ValueError(
                f"No trial mapping for api_token_hash={token_hash} in {manifest_path}"
            )
        api_token = trial_to_token.get(trial_id)
        if not api_token:
            raise ValueError(f"Missing api token for trial={trial_id}")

        run_start_dt = parse_iso8601(run_start_raw)
        run_offset_s = round((run_start_dt - t0_dt).total_seconds(), 6)
        if run_offset_s < 0:
            raise ValueError(f"run_offset_s is negative for {run_dir.name}: {run_offset_s}")

        lifecycle_records = read_jsonl(lifecycle_path)
        agent_start_dt = extract_agent_start_time(lifecycle_records)
        agent_end_dt = extract_agent_end_time(lifecycle_records)
        delta_agent_start_s = round(
            max(0.0, (agent_start_dt - run_start_dt).total_seconds()),
            6,
        )

        request_records = read_jsonl(requests_path)
        request_records.sort(key=parse_request_start)
        if request_records:
            delta_first_request_s = round(
                max(0.0, (parse_request_start(request_records[0]) - agent_start_dt).total_seconds()),
                6,
            )
        else:
            delta_first_request_s = 0.0

        jaeger_payload = read_json(jaeger_path)
        if not isinstance(jaeger_payload, dict):
            raise ValueError(f"Invalid jaeger payload in {jaeger_path}")
        agent_action_durations = parse_agent_action_durations(jaeger_payload)
        if request_records:
            final_agent_tail_s = max(
                0.0,
                (agent_end_dt - parse_request_end(request_records[-1])).total_seconds(),
            )
        else:
            final_agent_tail_s = max(0.0, (agent_end_dt - agent_start_dt).total_seconds())
        delta_agent_action_after = compute_agent_action_deltas(
            request_records,
            agent_action_durations,
            final_agent_tail_s=final_agent_tail_s,
        )

        planned_requests: list[dict[str, Any]] = []
        for index, record in enumerate(request_records):
            request_body = record.get("request")
            if not isinstance(request_body, dict):
                raise ValueError(
                    f"Request body must be object in {requests_path} at index={index}"
                )
            request_body = copy.deepcopy(request_body)

            record_model = record.get("model")
            model_for_tokenize: str | None = None
            if isinstance(record_model, str) and record_model.strip():
                model_for_tokenize = record_model.strip()
            else:
                payload_model = request_body.get("model")
                if isinstance(payload_model, str) and payload_model.strip():
                    model_for_tokenize = payload_model.strip()
            if not model_for_tokenize:
                model_for_tokenize = configured_model

            response_text = extract_response_text(record.get("response"))
            forced_token_ids = tokenize_response_text(
                tokenize_endpoint=tokenize_endpoint,
                model_name=model_for_tokenize,
                text=response_text,
                timeout_s=args.request_timeout_s,
            )

            vllm_xargs = request_body.get("vllm_xargs")
            if not isinstance(vllm_xargs, dict):
                vllm_xargs = {}
            vllm_xargs["forced_token_ids"] = forced_token_ids
            vllm_xargs["force_eos_after_sequence"] = True
            request_body["vllm_xargs"] = vllm_xargs

            max_tokens_required = len(forced_token_ids) + 1
            current_max_tokens = request_body.get("max_tokens")
            if not isinstance(current_max_tokens, int) or current_max_tokens < max_tokens_required:
                request_body["max_tokens"] = max_tokens_required

            method = record.get("http_method")
            path = record.get("http_path")
            if not isinstance(method, str) or not method:
                method = "POST"
            if not isinstance(path, str) or not path:
                path = "v1/chat/completions"

            planned_requests.append(
                {
                    "index": index,
                    "request_id": record.get("request_id"),
                    "method": method.upper(),
                    "path": path,
                    "body": request_body,
                    "model_for_tokenize": model_for_tokenize,
                    "delta_agent_action_after_s": delta_agent_action_after[index],
                    "expected_response_text": response_text,
                    "forced_token_ids": forced_token_ids,
                    "force_eos_after_sequence": True,
                }
            )

        worker_plans.append(
            {
                "worker_id": trial_id,
                "trial_id": trial_id,
                "api_token": api_token,
                "api_token_hash": token_hash,
                "trace_id": run_manifest_payload.get("trace_id"),
                "run_start_time": to_iso8601_utc(run_start_dt),
                "run_offset_s": run_offset_s,
                "delta_agent_start_s": delta_agent_start_s,
                "delta_first_request_s": delta_first_request_s,
                "requests": planned_requests,
            }
        )

    worker_plans.sort(key=lambda item: (float(item["run_offset_s"]), str(item["worker_id"])))
    for launch_priority, worker in enumerate(worker_plans):
        worker["launch_priority"] = launch_priority

    if args.plan_out:
        plan_path = Path(args.plan_out).expanduser().resolve()
    else:
        plan_path = (job_dir / "replay-plan.json").resolve()

    plan_payload = {
        "schema_version": "replay-plan.v1",
        "compiled_at": now_iso8601_utc(),
        "source_job_dir": str(job_dir),
        "backend": backend_name,
        "t0": t0_iso,
        "t0_source": t0_source,
        "replay_target": {
            "port_profile_id": replay_port_profile_id,
            "gateway_url": gateway_url,
            "api_base": api_base,
            "model": configured_model,
            "tokenize_endpoint": tokenize_endpoint,
            "deterministic_required": True,
        },
        "launch_policy": launch_policy,
        "workers": worker_plans,
    }
    write_json(plan_path, plan_payload)

    summary = {
        "status": "ok",
        "backend": backend_name,
        "plan_path": str(plan_path),
        "launch_strategy": launch_policy.get("strategy"),
        "worker_count": len(worker_plans),
        "request_count": sum(len(worker["requests"]) for worker in worker_plans),
        "port_profile_id": replay_port_profile_id,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


def sleep_with_stop(stop_event: threading.Event, seconds: float) -> bool:
    if seconds <= 0:
        return not stop_event.is_set()
    deadline = time.monotonic() + seconds
    while not stop_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        stop_event.wait(timeout=min(0.2, remaining))
    return False


def resolve_gateway_lifecycle_mode(
    mode: str,
    gateway_url: str | None,
    api_base: str,
) -> bool:
    mode = mode.lower()
    if mode == "off":
        return False
    if mode == "on":
        if not gateway_url:
            raise ValueError("gateway lifecycle mode 'on' requires gateway_url")
        return True
    if mode != "auto":
        raise ValueError(f"Invalid gateway lifecycle mode: {mode}")
    if not gateway_url:
        return False
    expected_prefix = f"{gateway_url.rstrip('/')}/v1"
    return api_base.rstrip("/").startswith(expected_prefix)


def _normalize_launch_pattern_name(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return "eager"


def _coerce_pattern_arg_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _extract_poisson_rate_per_second(
    pattern_payload: dict[str, Any],
    pattern_args_payload: Any,
) -> float | None:
    direct_rate = _coerce_pattern_arg_float(pattern_payload.get("rate_per_second"))
    if direct_rate is not None:
        return direct_rate

    if not isinstance(pattern_args_payload, dict):
        return None

    rate_keys = ("rate", "arrival-rate", "arrival_rate", "lambda")
    for key in rate_keys:
        rate_value = _coerce_pattern_arg_float(pattern_args_payload.get(key))
        if rate_value is not None:
            return rate_value

    mean_keys = ("mean-interval-s", "mean_interval_s", "interval-s", "interval_s")
    for key in mean_keys:
        mean_value = _coerce_pattern_arg_float(pattern_args_payload.get(key))
        if mean_value is not None:
            if mean_value <= 0:
                raise ValueError("Replay launch pattern mean interval must be > 0")
            return 1.0 / mean_value
    return None


def merge_dict_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = merge_dict_overlay(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_launch_policy_override_json(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --launch-policy-override-json payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--launch-policy-override-json must decode to a JSON object")
    launch_policy_payload = parsed.get("launch_policy")
    if launch_policy_payload is not None:
        if not isinstance(launch_policy_payload, dict):
            raise ValueError(
                "--launch-policy-override-json field launch_policy must be a JSON object"
            )
        return launch_policy_payload
    return parsed


def resolve_replay_launch_policy(
    *,
    launch_policy_payload: dict[str, Any],
    launch_policy_override: dict[str, Any] | None,
) -> tuple[str, int, int | None, str, float | None, Callable[[], float], dict[str, Any], dict[str, Any] | None]:
    effective_launch_policy = (
        merge_dict_overlay(launch_policy_payload, launch_policy_override)
        if launch_policy_override is not None
        else copy.deepcopy(launch_policy_payload)
    )
    strategy_raw = launch_policy_payload.get("strategy")
    if not isinstance(strategy_raw, str) or not strategy_raw.strip():
        raise ValueError("launch_policy.strategy is required")
    launch_strategy = strategy_raw.strip().lower()
    if launch_strategy != "config_ordered":
        raise ValueError(
            f"Unsupported launch strategy {launch_strategy!r}; only config_ordered is supported"
        )

    max_concurrent_raw = effective_launch_policy.get("max_concurrent")
    max_concurrent = _to_int_or_default(max_concurrent_raw, default=1)
    if max_concurrent <= 0:
        raise ValueError("Replay launch max_concurrent must be > 0")
    launch_max_concurrent = max_concurrent

    seed_raw = effective_launch_policy.get("seed")
    launch_seed: int | None = None
    if isinstance(seed_raw, int) and not isinstance(seed_raw, bool):
        launch_seed = seed_raw
        rng = Random(seed_raw)
    else:
        rng = Random()

    pattern_payload = effective_launch_policy.get("pattern")
    if not isinstance(pattern_payload, dict):
        raise ValueError(
            "launch_policy.strategy=config_ordered requires launch_policy.pattern"
        )
    pattern_name = _normalize_launch_pattern_name(pattern_payload.get("name"))
    launch_pattern_name = pattern_name

    launch_pattern_rate_per_second: float | None = None
    if pattern_name == "eager":
        next_launch_delay_s = lambda: 0.0
    elif pattern_name in {"poisson", "possion"}:
        rate_value = _extract_poisson_rate_per_second(
            pattern_payload,
            effective_launch_policy.get("pattern_args"),
        )
        if rate_value is None:
            raise ValueError(
                "Poisson replay launch pattern requires rate_per_second; "
                "set it in launch_policy.pattern.rate_per_second or launch_policy.pattern_args"
            )
        rate_per_second = rate_value
        if rate_per_second <= 0:
            raise ValueError("Replay launch pattern rate_per_second must be > 0")
        launch_pattern_rate_per_second = rate_per_second

        def _next_poisson_delay() -> float:
            return rng.expovariate(rate_per_second)

        next_launch_delay_s = _next_poisson_delay
    else:
        raise ValueError(
            f"Unsupported replay launch pattern {pattern_name!r}; supported: eager, poisson"
        )

    overrides = launch_policy_override
    return (
        launch_strategy,
        launch_max_concurrent,
        launch_seed,
        launch_pattern_name,
        launch_pattern_rate_per_second,
        next_launch_delay_s,
        overrides,
        effective_launch_policy,
    )


def cmd_replay(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).expanduser().resolve()
    require_file(plan_path)
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError(f"Invalid replay plan payload: {plan_path}")

    replay_target = plan.get("replay_target")
    if not isinstance(replay_target, dict):
        raise ValueError("Missing replay_target in plan")
    api_base, gateway_url, tokenize_endpoint, resolved_port_profile_id = resolve_replay_target(
        replay_target=replay_target,
        port_profile_id_override=args.port_profile_id,
    )

    workers = plan.get("workers")
    if not isinstance(workers, list):
        raise ValueError("Missing workers list in plan")
    for worker in workers:
        if not isinstance(worker, dict):
            raise ValueError("Worker item must be an object")

    launch_policy_payload = plan.get("launch_policy")
    if not isinstance(launch_policy_payload, dict):
        raise ValueError(
            "Missing launch_policy in plan. Old plans without launch_policy are not supported."
        )

    (
        launch_strategy,
        launch_max_concurrent,
        launch_seed,
        launch_pattern_name,
        launch_pattern_rate_per_second,
        next_launch_delay_s,
        launch_policy_overrides,
        effective_launch_policy,
    ) = resolve_replay_launch_policy(
        launch_policy_payload=launch_policy_payload,
        launch_policy_override=parse_launch_policy_override_json(
            getattr(args, "launch_policy_override_json", None)
        ),
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        source_job_dir = str(plan.get("source_job_dir") or "")
        source_name = Path(source_job_dir).name if source_job_dir else plan_path.stem
        replay_name = f"{safe_name(source_name)}.replayed-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        output_dir = (plan_path.parent / replay_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workers_dir = output_dir / "replay" / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    replay_plan_copy_path = output_dir / "replay" / "replay-plan.json"
    write_json(replay_plan_copy_path, plan)

    use_gateway_lifecycle = resolve_gateway_lifecycle_mode(
        args.gateway_lifecycle,
        gateway_url,
        api_base,
    )

    stop_event = threading.Event()
    lock = threading.Lock()
    worker_results: dict[str, dict[str, Any]] = {}
    summary = {
        "source_plan": str(plan_path),
        "output_dir": str(output_dir),
        "started_at": now_iso8601_utc(),
        "port_profile_id": resolved_port_profile_id,
        "gateway_lifecycle_enabled": use_gateway_lifecycle,
        "launch_strategy": launch_strategy,
        "launch_pattern": launch_pattern_name,
        "launch_max_concurrent": launch_max_concurrent,
        "launch_seed": launch_seed,
        "launch_pattern_rate_per_second": launch_pattern_rate_per_second,
        "launch_policy_overrides": launch_policy_overrides,
        "effective_launch_policy": effective_launch_policy,
        "workers_total": len(workers),
        "workers_completed": 0,
        "workers_failed": 0,
        "requests_sent": 0,
        "requests_failed": 0,
    }
    progress_nonlocal: list[Any] = [_NullProgress()]
    progress_task_id_nonlocal = [0]
    progress_state = {"launched": 0, "active": 0}

    def _update_progress(*, advance: int = 0) -> None:
        progress_nonlocal[0].update(
            progress_task_id_nonlocal[0],
            advance=advance,
            launched=progress_state["launched"],
            active=progress_state["active"],
            failed=summary["workers_failed"],
        )

    def call_gateway(path: str, payload: dict[str, Any]) -> Any:
        if not gateway_url:
            raise RuntimeError("Gateway URL is required for lifecycle calls")
        status, response_payload = http_json(
            method="POST",
            url=join_url(gateway_url, path),
            payload=payload,
            timeout_s=args.request_timeout_s,
        )
        if status >= 400:
            raise RuntimeError(
                f"Gateway lifecycle call failed: {path} HTTP {status} payload={response_payload}"
            )
        return response_payload

    if use_gateway_lifecycle:
        gateway_output_dir = output_dir / "gateway-output"
        gateway_output_dir.mkdir(parents=True, exist_ok=True)
        call_gateway(
            "/job/start",
            {"output_location": str(gateway_output_dir)},
        )

    def worker_fn(worker: dict[str, Any]) -> None:
        worker_id = str(worker.get("worker_id") or worker.get("trial_id") or "worker")
        worker_log_path = workers_dir / f"{safe_name(worker_id)}.json"
        record: dict[str, Any] = {
            "worker_id": worker_id,
            "started_at": now_iso8601_utc(),
            "status": "running",
            "requests_total": 0,
            "requests_succeeded": 0,
            "requests_failed": 0,
            "error": None,
        }

        api_token = worker.get("api_token")
        if not isinstance(api_token, str) or not api_token:
            api_token = None

        try:
            delta_agent_start_s = float(worker.get("delta_agent_start_s", 0.0))
            if not sleep_with_stop(stop_event, max(0.0, delta_agent_start_s)):
                record["status"] = "cancelled"
                return

            if use_gateway_lifecycle:
                if not api_token:
                    raise RuntimeError(
                        f"Missing worker api_token for gateway lifecycle: {worker_id}"
                    )
                call_gateway("/agent/start", {"api_token": api_token})

            delta_first_request_s = float(worker.get("delta_first_request_s", 0.0))
            if not sleep_with_stop(stop_event, max(0.0, delta_first_request_s)):
                record["status"] = "cancelled"
                return

            planned_requests = worker.get("requests")
            if not isinstance(planned_requests, list):
                raise RuntimeError(f"Worker requests must be list: {worker_id}")
            record["requests_total"] = len(planned_requests)

            for req in planned_requests:
                if stop_event.is_set():
                    record["status"] = "cancelled"
                    return
                if not isinstance(req, dict):
                    raise RuntimeError(f"Invalid request object in worker={worker_id}")
                method = str(req.get("method", "POST")).upper()
                path = str(req.get("path", "v1/chat/completions"))
                path = normalize_request_path_for_api_base(api_base, path)
                body = req.get("body")
                if not isinstance(body, dict):
                    raise RuntimeError(
                        f"Request body must be object in worker={worker_id}, path={path}"
                    )
                headers: dict[str, str] = {}
                if api_token:
                    headers["x-api-key"] = api_token

                status, response_payload = http_json(
                    method=method,
                    url=join_url(api_base, path),
                    payload=body,
                    timeout_s=args.request_timeout_s,
                    headers=headers,
                )
                with lock:
                    summary["requests_sent"] += 1
                if status >= 400:
                    record["requests_failed"] += 1
                    with lock:
                        summary["requests_failed"] += 1
                    raise RuntimeError(
                        f"Replay request failed: worker={worker_id}, "
                        f"path={path}, status={status}, response={response_payload}"
                    )

                expected_response_text = req.get("expected_response_text")
                if not isinstance(expected_response_text, str):
                    record["requests_failed"] += 1
                    with lock:
                        summary["requests_failed"] += 1
                    raise RuntimeError(
                        "Replay request is missing expected_response_text in plan: "
                        f"worker={worker_id}, path={path}"
                    )
                try:
                    actual_response_text = extract_response_text(response_payload)
                except Exception as exc:  # noqa: BLE001
                    record["requests_failed"] += 1
                    with lock:
                        summary["requests_failed"] += 1
                    raise RuntimeError(
                        "Failed to extract response text from replay response: "
                        f"worker={worker_id}, path={path}, error={exc}"
                    ) from exc
                if actual_response_text != expected_response_text:
                    record["requests_failed"] += 1
                    with lock:
                        summary["requests_failed"] += 1
                    raise RuntimeError(
                        "Replay response text mismatch: "
                        f"worker={worker_id}, path={path}, "
                        f"expected={expected_response_text!r}, "
                        f"actual={actual_response_text!r}"
                    )
                record["requests_succeeded"] += 1

                delay_after_s = float(req.get("delta_agent_action_after_s", 0.0))
                if not sleep_with_stop(stop_event, max(0.0, delay_after_s)):
                    record["status"] = "cancelled"
                    return

            record["status"] = "completed"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = str(exc)
            with lock:
                summary["workers_failed"] += 1
            stop_event.set()
        finally:
            if use_gateway_lifecycle and api_token:
                try:
                    rc = 0 if record["status"] == "completed" else 1
                    call_gateway(
                        "/agent/end",
                        {"api_token": api_token, "return_code": rc},
                    )
                except Exception as exc:  # noqa: BLE001
                    if record["status"] == "completed":
                        record["status"] = "failed"
                        record["error"] = f"agent/end failed: {exc}"
                        with lock:
                            summary["workers_failed"] += 1
                    stop_event.set()

            record["finished_at"] = now_iso8601_utc()
            with lock:
                worker_results[worker_id] = record
                if record["status"] == "completed":
                    summary["workers_completed"] += 1
                progress_state["active"] = max(progress_state["active"] - 1, 0)
                _update_progress(advance=1)
            write_json(worker_log_path, record)

    def _launch_priority(worker: dict[str, Any]) -> int:
        value = worker.get("launch_priority")
        if isinstance(value, bool):
            return 1_000_000_000
        if isinstance(value, int):
            return value
        return 1_000_000_000

    ordered_workers = sorted(
        workers,
        key=lambda worker: (
            _launch_priority(worker),
            float(worker.get("run_offset_s", 0.0)),
            str(worker.get("worker_id") or worker.get("trial_id") or "worker"),
        ),
    )

    threads: list[threading.Thread] = []
    for worker in ordered_workers:
        worker_id = str(worker.get("worker_id") or worker.get("trial_id") or "worker")
        thread = threading.Thread(target=worker_fn, args=(worker,), name=f"replay-{worker_id}")
        thread.daemon = True
        threads.append(thread)

    started_threads: list[threading.Thread] = []

    with create_replay_progress() as progress:
        progress_nonlocal[0] = progress
        progress_task_id_nonlocal[0] = progress.add_task(
            "replaying",
            total=len(threads),
            launched=0,
            active=0,
            failed=0,
        )
        try:
            next_launch_at = time.monotonic()
            for thread in threads:
                while True:
                    active_threads = [item for item in started_threads if item.is_alive()]
                    if len(active_threads) < launch_max_concurrent:
                        break
                    if stop_event.is_set():
                        break
                    for active_thread in active_threads:
                        active_thread.join(timeout=0.2)
                if stop_event.is_set():
                    break

                launch_delay = next_launch_at - time.monotonic()
                if launch_delay > 0 and not sleep_with_stop(stop_event, launch_delay):
                    break

                with lock:
                    progress_state["launched"] += 1
                    progress_state["active"] += 1
                    _update_progress()
                try:
                    thread.start()
                except Exception:
                    with lock:
                        progress_state["launched"] = max(progress_state["launched"] - 1, 0)
                        progress_state["active"] = max(progress_state["active"] - 1, 0)
                        _update_progress()
                    raise
                started_threads.append(thread)
                next_launch_at = time.monotonic() + max(0.0, next_launch_delay_s())

            while any(thread.is_alive() for thread in started_threads):
                for thread in started_threads:
                    thread.join(timeout=0.2)
        except KeyboardInterrupt:
            stop_event.set()
            for thread in started_threads:
                thread.join(timeout=2)

    if use_gateway_lifecycle:
        try:
            status = "completed" if summary["workers_failed"] == 0 else "failed"
            call_gateway("/job/end", {"status": status})
        except Exception as exc:  # noqa: BLE001
            summary["gateway_job_end_error"] = str(exc)
            summary["workers_failed"] += 1

    summary["finished_at"] = now_iso8601_utc()
    summary["worker_results"] = worker_results
    summary_path = output_dir / "replay" / "summary.json"
    write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "status": "ok" if summary["workers_failed"] == 0 else "failed",
                "output_dir": str(output_dir),
                "summary_path": str(summary_path),
                "workers_total": summary["workers_total"],
                "workers_completed": summary["workers_completed"],
                "workers_failed": summary["workers_failed"],
                "requests_sent": summary["requests_sent"],
                "requests_failed": summary["requests_failed"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0 if summary["workers_failed"] == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replayer",
        description="Compile replay plans and execute replay from plans.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile a profiled job directory into replay-plan.json",
    )
    compile_parser.add_argument(
        "--job-dir",
        required=True,
        help="Path to profiled con-driver job directory.",
    )
    compile_parser.add_argument(
        "--plan-out",
        default=None,
        help="Output path for replay plan. Default: <job-dir>/replay-plan.json",
    )
    compile_parser.add_argument(
        "--backend",
        default=None,
        help="Override backend name detection from meta/config.toml.",
    )
    compile_parser.add_argument(
        "--port-profile-id",
        type=int,
        default=None,
        help=(
            "Resolve replay target URLs from configs/port_profiles.toml. "
            "Defaults to meta/config.toml [runtime].port_profile_id when present."
        ),
    )
    compile_parser.add_argument(
        "--tokenize-endpoint",
        default=None,
        help=(
            "vLLM tokenize endpoint used to build forced_token_ids. "
            "Defaults to the selected port profile's vLLM /tokenize when available."
        ),
    )
    compile_parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=30.0,
        help="HTTP timeout seconds for tokenize/validation requests.",
    )
    compile_parser.set_defaults(func=cmd_compile)

    replay_parser = subparsers.add_parser(
        "replay",
        help="Run replay from a compiled replay-plan.json",
    )
    replay_parser.add_argument(
        "--plan",
        required=True,
        help="Path to compiled replay-plan.json.",
    )
    replay_parser.add_argument(
        "--output-dir",
        default=None,
        help="Replay output directory. Default: <plan-dir>/<job>.replayed-<ts>",
    )
    replay_parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=120.0,
        help="HTTP timeout seconds for replay requests.",
    )
    replay_parser.add_argument(
        "--port-profile-id",
        type=int,
        default=None,
        help=(
            "Resolve replay target URLs from configs/port_profiles.toml. "
            "Defaults to replay_target.port_profile_id from the plan when present."
        ),
    )
    replay_parser.add_argument(
        "--gateway-lifecycle",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Control /job/* and /agent/* lifecycle calls. "
            "auto enables lifecycle when api_base looks like gateway_url + /v1."
        ),
    )
    replay_parser.add_argument(
        "--launch-policy-override-json",
        default=None,
        help=(
            "JSON object used to overlay replay-plan launch_policy during replay. "
            "Accepts either the launch_policy object itself or an object with a "
            "top-level launch_policy field."
        ),
    )
    replay_parser.set_defaults(func=cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
