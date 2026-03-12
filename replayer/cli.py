"""CLI for compiling and replaying gateway-profiled jobs."""

from __future__ import annotations

import argparse
import copy
import http.client
import hashlib
import json
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from random import Random
import os
import signal
import socket
import subprocess
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


DEFAULT_VLLM_LOG_INTERVAL_S = 1.0
DEFAULT_VLLM_LOG_TIMEOUT_S = 3600.0
DEFAULT_COMPILE_TOKENIZE_TIMEOUT_S = 3600.0
_MONITOR_INTERRUPT_GRACE_SEC = 3600.0
_MONITOR_TERMINATE_GRACE_SEC = 3600.0


@dataclass(frozen=True)
class ReplayVLLMLogConfig:
    enabled: bool
    endpoint: str
    interval_s: float
    timeout_s: float


@dataclass
class ReplayVLLMMonitorProcess:
    process: subprocess.Popen[str]
    stdout_handle: Any
    stderr_handle: Any
    stdout_log: Path
    stderr_log: Path


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


def create_compile_progress() -> Any:
    if RichProgress is None:
        return _NullProgress()
    return RichProgress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} requests"),
        TextColumn("workers={task.fields[workers_completed]}/{task.fields[workers_total]}"),
        TimeElapsedColumn(),
        transient=False,
    )


def resolve_replay_vllm_log_config(
    *,
    port_profile_id: int | None,
    interval_s: float,
    timeout_s: float,
) -> ReplayVLLMLogConfig:
    if port_profile_id is None:
        raise ValueError(
            "vLLM metrics logging requires --port-profile-id so the endpoint "
            "can be resolved from configs/port_profiles.toml."
        )
    from gateway.port_profiles import load_port_profile

    profile = load_port_profile(port_profile_id)
    endpoint = f"http://127.0.0.1:{profile.vllm_port}/metrics"
    if interval_s <= 0:
        raise ValueError("--vllm-log-interval-s must be > 0")
    if timeout_s <= 0:
        raise ValueError("--vllm-log-timeout-s must be > 0")
    return ReplayVLLMLogConfig(
        enabled=True,
        endpoint=endpoint,
        interval_s=interval_s,
        timeout_s=timeout_s,
    )


def _build_monitor_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    con_driver_src = repo_root / "con-driver" / "src"
    pythonpath_parts = [str(con_driver_src)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def start_replay_vllm_monitor(
    *,
    output_dir: Path,
    config: ReplayVLLMLogConfig,
) -> ReplayVLLMMonitorProcess:
    vllm_log_dir = output_dir / "vllm-log"
    vllm_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = vllm_log_dir / "monitor.stdout.log"
    stderr_log = vllm_log_dir / "monitor.stderr.log"
    stdout_handle = stdout_log.open("w", encoding="utf-8")
    stderr_handle = stderr_log.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "con_driver.vllm_metrics_monitor",
        "--endpoint",
        config.endpoint,
        "--output-dir",
        str(vllm_log_dir),
        "--interval-s",
        str(config.interval_s),
        "--timeout-s",
        str(config.timeout_s),
        "--block-size",
        "100",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=_build_monitor_env(),
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise
    return ReplayVLLMMonitorProcess(
        process=process,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )


def stop_replay_vllm_monitor(monitor: ReplayVLLMMonitorProcess) -> int:
    process = monitor.process
    try:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=_MONITOR_INTERRUPT_GRACE_SEC)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=_MONITOR_TERMINATE_GRACE_SEC)
                except subprocess.TimeoutExpired:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
        else:
            process.wait()
    finally:
        monitor.stdout_handle.close()
        monitor.stderr_handle.close()
    return int(process.returncode if process.returncode is not None else 1)


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


def parse_optional_path(value: Any, *, field_name: str) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return Path(stripped)
    raise ValueError(f"{field_name} must be a path")


def parse_optional_str(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return stripped
    raise ValueError(f"{field_name} must be a string")


def parse_optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def parse_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
    raise ValueError(f"{field_name} must be an integer")


def parse_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a number") from exc
    raise ValueError(f"{field_name} must be a number")


def resolve_required_option(value: Any, *, option_name: str, config_key: str) -> Any:
    if value is None:
        raise ValueError(
            f"Missing required option '{option_name}' (or set '{config_key}' in --config)."
        )
    return value


def load_replayer_subcommand_config(
    *,
    config_path: Path | None,
    section_name: str,
) -> dict[str, Any]:
    if config_path is None:
        return {}

    resolved_config_path = config_path.expanduser().resolve()
    if not resolved_config_path.exists():
        raise ValueError(f"Config file does not exist: {resolved_config_path}")
    if not resolved_config_path.is_file():
        raise ValueError(f"Config path is not a file: {resolved_config_path}")

    config_payload = parse_toml(resolved_config_path)
    root_payload: Any = config_payload
    if "replayer" in config_payload:
        root_payload = config_payload["replayer"]
        if not isinstance(root_payload, dict):
            raise ValueError("Config key 'replayer' must be a table")
    elif not isinstance(root_payload, dict):
        raise ValueError("Config root must be a table")

    resolved: dict[str, Any] = {}
    for key, value in root_payload.items():
        if key in {"compile", "replay"} and isinstance(value, dict):
            continue
        resolved[key] = value

    section_payload = root_payload.get(section_name)
    if section_payload is not None:
        if not isinstance(section_payload, dict):
            raise ValueError(f"Config key '{section_name}' must be a table")
        resolved.update(section_payload)
    return resolved


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
    timeout_s: float | None,
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
        if timeout_s is None:
            resp_cm = url_request.urlopen(req)
        else:
            resp_cm = url_request.urlopen(req, timeout=timeout_s)
        with resp_cm as resp:
            status = int(resp.getcode())
            raw = resp.read().decode("utf-8", errors="replace")
    except url_error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")
    except url_error.URLError as exc:
        raise RuntimeError(f"HTTP request failed: {method} {url}: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"HTTP request timed out: {method} {url}") from exc

    return status, decode_json_response_body(raw)


def decode_json_response_body(raw: str) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {"raw_body": raw}


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


def parse_model_from_tokens(tokens: list[str]) -> str | None:
    for idx, token in enumerate(tokens):
        if token == "--model" and idx + 1 < len(tokens):
            model_name = tokens[idx + 1]
            if isinstance(model_name, str) and model_name.strip():
                return model_name.strip()
    return None


def detect_backend(config: dict[str, Any]) -> str:
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


def extract_harbor_compile_model(
    config: dict[str, Any],
    results_entries: list[dict[str, Any]],
) -> str:
    model_name: str | None = None
    backend_section = config.get("backend")
    if isinstance(backend_section, dict):
        forwarded_args = backend_section.get("forwarded_args")
        if isinstance(forwarded_args, list):
            tokens = [str(item) for item in forwarded_args]
            model_name = parse_model_from_tokens(tokens)

    if not model_name:
        for entry in results_entries:
            command = entry.get("command")
            if not isinstance(command, list):
                continue
            tokens = [str(item) for item in command]
            command_model = parse_model_from_tokens(tokens)
            if command_model:
                model_name = command_model
                break

    if not model_name:
        raise ValueError("Unable to extract --model for harbor backend")
    return model_name


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


def resolve_compile_target(
    *,
    config: dict[str, Any],
    results_entries: list[dict[str, Any]],
    port_profile_id: int,
) -> tuple[str, str]:
    configured_model = extract_harbor_compile_model(config, results_entries)
    _, _, tokenize_endpoint = build_replay_target_from_port_profile(
        port_profile_id,
        gateway_enabled=True,
    )

    return configured_model, tokenize_endpoint


def resolve_replay_target(
    *,
    replay_target: dict[str, Any],
    port_profile_id_override: int | None,
) -> tuple[str, str | None, str | None, str | None]:
    del replay_target
    effective_port_profile_id = parse_port_profile_id(
        port_profile_id_override,
        field_name="--port-profile-id",
    )
    if effective_port_profile_id is None:
        raise ValueError("Replay requires --port-profile-id.")

    (
        gateway_url,
        api_base,
        tokenize_endpoint,
    ) = build_replay_target_from_port_profile(
        effective_port_profile_id,
        gateway_enabled=True,
    )
    return api_base, gateway_url, tokenize_endpoint, str(effective_port_profile_id)


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


def _parse_gateway_profile_dir_name(name: str) -> int | None:
    prefix = "profile-"
    if not name.startswith(prefix):
        return None
    raw = name[len(prefix) :]
    if not raw:
        return None
    try:
        return parse_port_profile_id(raw, field_name="gateway-output profile dir")
    except ValueError:
        return None


def discover_gateway_run_dirs(
    gateway_output_dir: Path,
) -> list[tuple[Path, int | None]]:
    run_dirs: list[tuple[Path, int | None]] = []

    for run_dir in sorted(gateway_output_dir.glob("run_*")):
        if run_dir.is_dir():
            run_dirs.append((run_dir, None))

    for child in sorted(gateway_output_dir.iterdir()):
        if not child.is_dir():
            continue
        profile_id = _parse_gateway_profile_dir_name(child.name)
        if profile_id is None:
            continue
        for run_dir in sorted(child.glob("run_*")):
            if run_dir.is_dir():
                run_dirs.append((run_dir, profile_id))

    return run_dirs


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


def is_client_disconnected_record(record: dict[str, Any]) -> bool:
    status_code = record.get("status_code")
    if isinstance(status_code, bool):
        status_code = None
    response_payload = record.get("response")
    if not isinstance(response_payload, dict):
        return False
    error_name = response_payload.get("error")
    if error_name != "client_disconnected":
        return False
    return status_code in {None, 499}


def request_duration_seconds(record: dict[str, Any]) -> float:
    duration_ms = record.get("request_duration_ms")
    if not isinstance(duration_ms, (int, float)) or isinstance(duration_ms, bool):
        duration_ms = record.get("duration_ms")
    if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
        return round(max(0.0, float(duration_ms) / 1000.0), 6)
    return round(
        max(0.0, (parse_request_end(record) - parse_request_start(record)).total_seconds()),
        6,
    )


def parse_required_positive_timeout_s(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive number")
    if isinstance(value, (int, float)):
        timeout_s = float(value)
    elif isinstance(value, str) and value.strip():
        timeout_s = float(value.strip())
    else:
        raise ValueError(f"{field_name} is required")
    if timeout_s <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return timeout_s


def parse_optional_positive_timeout_s(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return parse_required_positive_timeout_s(value, field_name=field_name)


def parse_optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be a positive integer, got {value!r}"
            ) from exc
    else:
        raise ValueError(f"{field_name} must be a positive integer")
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return parsed


def build_replay_worker_schedule(
    *,
    workers: list[dict[str, Any]],
    num_tasks: int | None,
) -> tuple[list[dict[str, Any]], bool]:
    if num_tasks is None:
        return workers, False
    if num_tasks <= len(workers):
        return workers[:num_tasks], False
    if not workers:
        raise ValueError("--num-tasks requires at least one worker in the replay plan")

    scheduled_workers: list[dict[str, Any]] = []
    worker_count = len(workers)
    for index in range(num_tasks):
        source_index = index % worker_count
        wrap_round = index // worker_count
        source_worker = workers[source_index]
        if wrap_round == 0:
            scheduled_workers.append(source_worker)
            continue
        source_worker_id = str(
            source_worker.get("worker_id") or source_worker.get("trial_id") or "worker"
        )
        worker_copy = copy.deepcopy(source_worker)
        worker_copy["worker_id"] = (
            f"{source_worker_id}__wrap{wrap_round + 1}__task{index + 1}"
        )
        worker_copy["wrapped_from_worker_id"] = source_worker_id
        worker_copy["wrapped_from_task_index"] = source_index + 1
        source_api_token = source_worker.get("api_token")
        if isinstance(source_api_token, str) and source_api_token:
            worker_copy["api_token"] = (
                f"{source_api_token}__wrap{wrap_round + 1}__task{index + 1}"
            )
        scheduled_workers.append(worker_copy)
    return scheduled_workers, True


CLIENT_DISCONNECT_ACCEPTABLE_GATEWAY_ERROR_MIN_S = 60.0


def build_client_disconnect_request_body(request_body: dict[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(request_body)
    body.pop("stop", None)
    body.pop("stop_token_ids", None)
    body.pop("max_tokens", None)
    body.pop("max_completion_tokens", None)
    body["ignore_eos"] = True

    vllm_xargs = body.get("vllm_xargs")
    if isinstance(vllm_xargs, dict):
        cleaned_vllm_xargs = copy.deepcopy(vllm_xargs)
        cleaned_vllm_xargs.pop("forced_token_ids", None)
        cleaned_vllm_xargs.pop("force_eos_after_sequence", None)
        if cleaned_vllm_xargs:
            body["vllm_xargs"] = cleaned_vllm_xargs
        else:
            body.pop("vllm_xargs", None)
    return body


def build_planned_request(
    *,
    record: dict[str, Any],
    index: int,
    configured_model: str,
    tokenize_endpoint: str,
    request_timeout_s: float,
    delta_agent_action_after_s: float,
) -> dict[str, Any]:
    request_body = record.get("request")
    if not isinstance(request_body, dict):
        raise ValueError(f"Request body must be object at index={index}")
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

    method = record.get("http_method")
    path = record.get("http_path")
    if not isinstance(method, str) or not method:
        method = "POST"
    if not isinstance(path, str) or not path:
        path = "v1/chat/completions"

    if is_client_disconnected_record(record):
        request_duration_s = request_duration_seconds(record)
        request_body = build_client_disconnect_request_body(request_body)
        return {
            "index": index,
            "request_id": record.get("request_id"),
            "method": method.upper(),
            "path": path,
            "body": request_body,
            "model_for_tokenize": model_for_tokenize,
            "delta_agent_action_after_s": delta_agent_action_after_s,
            "replay_mode": "client_disconnect_after_duration",
            "cancel_after_s": request_duration_s,
            "expected_status_code": 499,
            "expected_error": "client_disconnected",
            "expected_response_text": None,
            "forced_token_ids": None,
            "force_eos_after_sequence": False,
        }

    response_text = extract_response_text(record.get("response"))
    forced_token_ids = tokenize_response_text(
        tokenize_endpoint=tokenize_endpoint,
        model_name=model_for_tokenize,
        text=response_text,
        timeout_s=request_timeout_s,
    )

    vllm_xargs = request_body.get("vllm_xargs")
    if not isinstance(vllm_xargs, dict):
        vllm_xargs = {}
    vllm_xargs["forced_token_ids"] = forced_token_ids
    vllm_xargs["force_eos_after_sequence"] = True
    request_body["vllm_xargs"] = vllm_xargs

    max_tokens_required = len(forced_token_ids) + 1
    current_max_tokens = request_body.get("max_tokens")
    if not isinstance(current_max_tokens, int) or isinstance(current_max_tokens, bool):
        current_max_tokens = 0
    if current_max_tokens < max_tokens_required:
        request_body["max_tokens"] = max_tokens_required

    return {
        "index": index,
        "request_id": record.get("request_id"),
        "method": method.upper(),
        "path": path,
        "body": request_body,
        "model_for_tokenize": model_for_tokenize,
        "delta_agent_action_after_s": delta_agent_action_after_s,
        "replay_mode": "deterministic_forced_tokens",
        "cancel_after_s": None,
        "expected_status_code": int(record.get("status_code", 200) or 200),
        "expected_error": None,
        "expected_response_text": response_text,
        "forced_token_ids": forced_token_ids,
        "force_eos_after_sequence": True,
    }


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
    config_file_path = parse_optional_path(
        getattr(args, "config", None),
        field_name="--config",
    )
    compile_config = load_replayer_subcommand_config(
        config_path=config_file_path,
        section_name="compile",
    )

    job_dir_value = parse_optional_path(
        (
            args.job_dir
            if getattr(args, "job_dir", None) is not None
            else compile_config.get("job_dir")
        ),
        field_name="job_dir",
    )
    job_dir = resolve_required_option(
        job_dir_value,
        option_name="--job-dir",
        config_key="job_dir",
    ).expanduser().resolve()
    if not job_dir.exists() or not job_dir.is_dir():
        raise ValueError(f"Invalid --job-dir: {job_dir}")

    backend_override = parse_optional_str(
        compile_config.get("backend"),
        field_name="backend",
    )
    if backend_override is not None:
        raise ValueError(
            "Compile backend override is no longer supported. "
            "Backend is detected from meta/config.toml."
        )
    compile_port_profile_id_value = parse_optional_int(
        (
            args.port_profile_id
            if getattr(args, "port_profile_id", None) is not None
            else compile_config.get("port_profile_id")
        ),
        field_name="port_profile_id",
    )
    compile_port_profile_id = parse_port_profile_id(
        resolve_required_option(
            compile_port_profile_id_value,
            option_name="--port-profile-id",
            config_key="port_profile_id",
        ),
        field_name="--port-profile-id",
    )
    request_timeout_s_override = parse_optional_positive_timeout_s(
        (
            args.request_timeout_s
            if getattr(args, "request_timeout_s", None) is not None
            else compile_config.get("request_timeout_s")
        ),
        field_name="--request-timeout-s",
    )
    plan_out_override = parse_optional_path(
        (
            args.plan_out
            if getattr(args, "plan_out", None) is not None
            else compile_config.get("plan_out")
        ),
        field_name="plan_out",
    )

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

    backend_name = detect_backend(config)
    if backend_name != "harbor":
        raise ValueError(f"unsupported backend: {backend_name!r}")

    configured_model, tokenize_endpoint = resolve_compile_target(
        config=config,
        results_entries=results_entries,
        port_profile_id=compile_port_profile_id,
    )
    request_timeout_s = (
        request_timeout_s_override
        if request_timeout_s_override is not None
        else DEFAULT_COMPILE_TOKENIZE_TIMEOUT_S
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

    discovered_run_dirs = discover_gateway_run_dirs(gateway_output_dir)
    if not discovered_run_dirs:
        raise ValueError(
            "No run_* artifacts found under gateway-output. "
            "Expected either gateway-output/run_* or gateway-output/profile-*/run_*."
        )

    run_infos: list[tuple[Path, dict[str, Any], int, int | None]] = []
    total_requests = 0
    for run_dir, source_port_profile_id in discovered_run_dirs:
        manifest_path = run_dir / "manifest.json"
        require_file(manifest_path)
        run_manifest_payload = read_json(manifest_path)
        if not isinstance(run_manifest_payload, dict):
            raise ValueError(f"Invalid JSON object: {manifest_path}")
        request_count_raw = run_manifest_payload.get("request_count")
        request_count = 0
        if isinstance(request_count_raw, int) and not isinstance(request_count_raw, bool):
            request_count = max(0, request_count_raw)
        run_infos.append((run_dir, run_manifest_payload, request_count, source_port_profile_id))
        total_requests += request_count

    worker_plans: list[dict[str, Any]] = []
    compile_progress_state = {
        "workers_completed": 0,
        "workers_total": len(run_infos),
        "requests_total": total_requests,
    }
    with create_compile_progress() as progress:
        progress_task_id = progress.add_task(
            "compiling replay plan",
            total=compile_progress_state["requests_total"],
            workers_completed=compile_progress_state["workers_completed"],
            workers_total=compile_progress_state["workers_total"],
        )

        def _update_compile_progress(*, advance: int = 0, request_total_delta: int = 0) -> None:
            if request_total_delta:
                compile_progress_state["requests_total"] = max(
                    0,
                    compile_progress_state["requests_total"] + request_total_delta,
                )
            progress.update(
                progress_task_id,
                advance=advance,
                total=compile_progress_state["requests_total"],
                workers_completed=compile_progress_state["workers_completed"],
                workers_total=compile_progress_state["workers_total"],
            )

        for run_dir, run_manifest_payload, expected_request_count, _source_port_profile_id in run_infos:
            manifest_path = run_dir / "manifest.json"
            lifecycle_path = run_dir / "events" / "lifecycle.jsonl"
            requests_path = run_dir / "requests" / "model_inference.jsonl"
            jaeger_path = run_dir / "trace" / "jaeger_trace.json"
            for path in [manifest_path, lifecycle_path, requests_path, jaeger_path]:
                require_file(path)

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
                raise ValueError(
                    f"run_offset_s is negative for {run_dir.name}: {run_offset_s}"
                )

            lifecycle_records = read_jsonl(lifecycle_path)
            agent_start_dt = extract_agent_start_time(lifecycle_records)
            agent_end_dt = extract_agent_end_time(lifecycle_records)
            delta_agent_start_s = round(
                max(0.0, (agent_start_dt - run_start_dt).total_seconds()),
                6,
            )

            request_records = read_jsonl(requests_path)
            request_records.sort(key=parse_request_start)
            actual_request_count = len(request_records)
            if actual_request_count != expected_request_count:
                _update_compile_progress(
                    request_total_delta=actual_request_count - expected_request_count
                )
            if request_records:
                delta_first_request_s = round(
                    max(
                        0.0,
                        (parse_request_start(request_records[0]) - agent_start_dt).total_seconds(),
                    ),
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
                final_agent_tail_s = max(
                    0.0,
                    (agent_end_dt - agent_start_dt).total_seconds(),
                )
            delta_agent_action_after = compute_agent_action_deltas(
                request_records,
                agent_action_durations,
                final_agent_tail_s=final_agent_tail_s,
            )

            planned_requests: list[dict[str, Any]] = []
            for index, record in enumerate(request_records):
                planned_requests.append(
                    build_planned_request(
                        record=record,
                        index=index,
                        configured_model=configured_model,
                        tokenize_endpoint=tokenize_endpoint,
                        request_timeout_s=request_timeout_s,
                        delta_agent_action_after_s=delta_agent_action_after[index],
                    )
                )
                _update_compile_progress(advance=1)

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
            compile_progress_state["workers_completed"] += 1
            _update_compile_progress()

    worker_plans.sort(key=lambda item: (float(item["run_offset_s"]), str(item["worker_id"])))
    for launch_priority, worker in enumerate(worker_plans):
        worker["launch_priority"] = launch_priority

    if plan_out_override is not None:
        plan_path = plan_out_override.expanduser().resolve()
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
            "model": configured_model,
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
        "port_profile_id": compile_port_profile_id,
    }
    if config_file_path is not None:
        summary["source_config"] = str(config_file_path.expanduser().resolve())
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


def timed_cancel_http_json(
    *,
    method: str,
    url: str,
    payload: Any,
    cancel_after_s: float,
    connect_timeout_s: float | None,
    headers: dict[str, str] | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    start_time = time.monotonic()
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise RuntimeError(f"Unsupported URL scheme for timed cancel replay: {url}")

    host = parsed_url.hostname
    if not host:
        raise RuntimeError(f"Missing hostname in replay URL: {url}")
    port = parsed_url.port
    if port is None:
        port = 443 if parsed_url.scheme == "https" else 80
    path = parsed_url.path or "/"
    if parsed_url.query:
        path = f"{path}?{parsed_url.query}"

    req_headers = {"content-type": "application/json"}
    if headers:
        req_headers.update(headers)
    body_bytes = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    if parsed_url.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            host,
            port,
            timeout=connect_timeout_s,
        )
    else:
        connection = http.client.HTTPConnection(
            host,
            port,
            timeout=connect_timeout_s,
        )

    try:
        connection.connect()
        if connection.sock is not None:
            connection.sock.settimeout(None)
    except Exception as exc:
        connection.close()
        raise RuntimeError(f"Timed cancel replay failed to connect: {method} {url}: {exc}") from exc

    state: dict[str, Any] = {
        "completed": False,
        "response_status": None,
        "response_started": False,
        "response_payload": None,
        "error": None,
    }
    finished = threading.Event()

    def _request_thread() -> None:
        try:
            connection.request(method.upper(), path, body=body_bytes, headers=req_headers)
            response = connection.getresponse()
            state["response_started"] = True
            state["response_status"] = int(response.status)
            raw_chunks: list[bytes] = []
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                raw_chunks.append(chunk)
            raw_body = b"".join(raw_chunks).decode("utf-8", errors="replace")
            state["response_payload"] = decode_json_response_body(raw_body)
            state["completed"] = True
        except Exception as exc:  # noqa: BLE001
            state["error"] = str(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=_request_thread, name="replay-timed-cancel-http")
    thread.daemon = True
    thread.start()

    if stop_event is None:
        interrupted = False
        completed_before_deadline = finished.wait(timeout=max(0.0, cancel_after_s))
    else:
        completed_before_deadline = False
        interrupted = False
        deadline = time.monotonic() + max(0.0, cancel_after_s)
        while not finished.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait_slice = min(0.2, remaining)
            if stop_event.wait(timeout=wait_slice):
                interrupted = True
                break
        completed_before_deadline = finished.is_set()

    if completed_before_deadline:
        connection.close()
        thread.join(timeout=1.0)
        return {
            "outcome": "completed_early",
            "elapsed_s": max(0.0, time.monotonic() - start_time),
            "response_status": state["response_status"],
            "response_payload": state["response_payload"],
            "error": state["error"],
        }

    try:
        if connection.sock is not None:
            with suppress(Exception):
                connection.sock.shutdown(socket.SHUT_RDWR)
    finally:
        connection.close()
    thread.join(timeout=max(1.0, min(10.0, max(1.0, cancel_after_s * 0.05))))

    if thread.is_alive():
        return {
            "outcome": "cancel_failed",
            "elapsed_s": max(0.0, time.monotonic() - start_time),
            "response_status": state["response_status"],
            "response_payload": state["response_payload"],
            "error": state["error"],
        }
    if interrupted:
        return {
            "outcome": "stopped",
            "elapsed_s": max(0.0, time.monotonic() - start_time),
            "response_status": state["response_status"],
            "response_payload": state["response_payload"],
            "error": state["error"],
        }
    return {
        "outcome": "cancelled",
        "elapsed_s": max(0.0, time.monotonic() - start_time),
        "response_status": state["response_status"],
        "response_payload": state["response_payload"],
        "error": state["error"],
    }


def is_acceptable_client_disconnect_early_error(timed_result: dict[str, Any]) -> bool:
    response_status = timed_result.get("response_status")
    elapsed_s = timed_result.get("elapsed_s")
    if not isinstance(response_status, int) or response_status < 400:
        return False
    if not isinstance(elapsed_s, (int, float)) or isinstance(elapsed_s, bool):
        return False
    return float(elapsed_s) >= CLIENT_DISCONNECT_ACCEPTABLE_GATEWAY_ERROR_MIN_S


def sleep_with_stop_or_deadline(
    stop_event: threading.Event,
    *,
    seconds: float,
    deadline_at: float | None,
) -> str:
    if deadline_at is not None and time.monotonic() >= deadline_at:
        return "deadline"
    if seconds <= 0:
        return "ok" if not stop_event.is_set() else "stopped"

    effective_deadline = time.monotonic() + seconds
    if deadline_at is not None:
        effective_deadline = min(effective_deadline, deadline_at)

    while not stop_event.is_set():
        remaining = effective_deadline - time.monotonic()
        if remaining <= 0:
            if deadline_at is not None and time.monotonic() >= deadline_at:
                return "deadline"
            return "ok"
        if stop_event.wait(timeout=min(0.2, remaining)):
            return "stopped"

    return "stopped"


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


def parse_launch_policy_override_payload(
    raw: Any,
    *,
    field_name: str = "--launch-policy-override-json",
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        try:
            parsed: Any = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {field_name} payload: {exc}") from exc
    elif isinstance(raw, dict):
        parsed = raw
    else:
        raise ValueError(f"{field_name} must be a JSON object or JSON string")
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to a JSON object")
    launch_policy_payload = parsed.get("launch_policy")
    if launch_policy_payload is not None:
        if not isinstance(launch_policy_payload, dict):
            raise ValueError(f"{field_name} field launch_policy must be a JSON object")
        return launch_policy_payload
    return parsed


def parse_launch_policy_override_json(raw: str | None) -> dict[str, Any] | None:
    return parse_launch_policy_override_payload(
        raw,
        field_name="--launch-policy-override-json",
    )


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
    config_file_path = parse_optional_path(
        getattr(args, "config", None),
        field_name="--config",
    )
    replay_config = load_replayer_subcommand_config(
        config_path=config_file_path,
        section_name="replay",
    )

    plan_value = parse_optional_path(
        (
            args.plan
            if getattr(args, "plan", None) is not None
            else replay_config.get("plan")
        ),
        field_name="plan",
    )
    plan_path = resolve_required_option(
        plan_value,
        option_name="--plan",
        config_key="plan",
    ).expanduser().resolve()
    require_file(plan_path)
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError(f"Invalid replay plan payload: {plan_path}")

    output_dir_override = parse_optional_path(
        (
            args.output_dir
            if getattr(args, "output_dir", None) is not None
            else replay_config.get("output_dir")
        ),
        field_name="output_dir",
    )
    requested_num_tasks = parse_optional_positive_int(
        (
            args.num_tasks
            if getattr(args, "num_tasks", None) is not None
            else replay_config.get("num_tasks")
        ),
        field_name="--num-tasks",
    )
    randomize_seed = parse_optional_int(
        (
            args.randomize_seed
            if getattr(args, "randomize_seed", None) is not None
            else replay_config.get("randomize_seed")
        ),
        field_name="--randomize-seed",
    )
    time_constraint_s = parse_optional_positive_timeout_s(
        (
            args.time_constraint_s
            if getattr(args, "time_constraint_s", None) is not None
            else replay_config.get("time_constraint_s")
        ),
        field_name="--time-constraint-s",
    )
    if time_constraint_s is not None and requested_num_tasks is not None:
        raise ValueError(
            "--time-constraint-s cannot be combined with --num-tasks. "
            "Time-constrained replay is unbounded by task count."
        )
    port_profile_id_override = parse_port_profile_id(
        parse_optional_int(
            getattr(args, "port_profile_id", None),
            field_name="--port-profile-id",
        ),
        field_name="--port-profile-id",
    )
    if port_profile_id_override is None:
        raise ValueError("Missing required option '--port-profile-id'.")
    launch_policy_override = parse_launch_policy_override_payload(
        (
            args.launch_policy_override_json
            if getattr(args, "launch_policy_override_json", None) is not None
            else replay_config.get(
                "launch_policy_override_json",
                replay_config.get("launch_policy_override"),
            )
        ),
        field_name="launch_policy_override_json",
    )
    configured_vllm_log = parse_optional_bool(
        replay_config.get("vllm_log"),
        field_name="vllm_log",
    )
    if bool(getattr(args, "vllm_log_explicit_on", False)):
        raise ValueError(
            "--vllm-log is not needed. Replay always enables vLLM logging."
        )
    if bool(getattr(args, "vllm_log_explicit_off", False)):
        raise ValueError(
            "--no-vllm-log is no longer supported. Replay always enables vLLM logging."
        )
    if configured_vllm_log is False:
        raise ValueError(
            "Replay no longer supports disabling vLLM logging. "
            "Remove replay.vllm_log = false."
        )
    vllm_log_interval_s = parse_optional_float(
        (
            args.vllm_log_interval_s
            if getattr(args, "vllm_log_interval_s", None) is not None
            else replay_config.get("vllm_log_interval_s")
        ),
        field_name="vllm_log_interval_s",
    )
    vllm_log_interval_s = float(
        vllm_log_interval_s
        if vllm_log_interval_s is not None
        else DEFAULT_VLLM_LOG_INTERVAL_S
    )
    vllm_log_timeout_s = parse_optional_float(
        (
            args.vllm_log_timeout_s
            if getattr(args, "vllm_log_timeout_s", None) is not None
            else replay_config.get("vllm_log_timeout_s")
        ),
        field_name="vllm_log_timeout_s",
    )
    vllm_log_timeout_s = float(
        vllm_log_timeout_s
        if vllm_log_timeout_s is not None
        else DEFAULT_VLLM_LOG_TIMEOUT_S
    )
    agent_timeout_s = parse_optional_positive_timeout_s(
        (
            args.agent_timeout_s
            if getattr(args, "agent_timeout_s", None) is not None
            else replay_config.get("agent_timeout_s")
        ),
        field_name="--agent-timeout-s",
    )

    replay_target = plan.get("replay_target")
    if not isinstance(replay_target, dict):
        raise ValueError("Missing replay_target in plan")
    api_base, gateway_url, tokenize_endpoint, resolved_port_profile_id = resolve_replay_target(
        replay_target=replay_target,
        port_profile_id_override=port_profile_id_override,
    )
    resolved_port_profile_id_int = parse_port_profile_id(
        resolved_port_profile_id,
        field_name="resolved replay port_profile_id",
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
        launch_policy_override=launch_policy_override,
    )

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
    if randomize_seed is not None:
        randomize_rng = Random(randomize_seed)
        ordered_workers = list(ordered_workers)
        randomize_rng.shuffle(ordered_workers)

    if time_constraint_s is None:
        scheduled_workers, tasks_wrapped = build_replay_worker_schedule(
            workers=ordered_workers,
            num_tasks=requested_num_tasks,
        )
    else:
        if not ordered_workers:
            raise ValueError("--time-constraint-s requires at least one worker in the replay plan")
        scheduled_workers = []
        tasks_wrapped = False

    if output_dir_override is not None:
        output_dir = output_dir_override.expanduser().resolve()
    else:
        source_job_dir = str(plan.get("source_job_dir") or "")
        source_name = Path(source_job_dir).name if source_job_dir else plan_path.stem
        replay_name = f"{safe_name(source_name)}.replayed-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        output_dir = (plan_path.parent / replay_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_http_timeout_s: float | None = None

    workers_dir = output_dir / "replay" / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    replay_plan_copy_path = output_dir / "replay" / "replay-plan.json"
    write_json(replay_plan_copy_path, plan)

    use_gateway_lifecycle = True
    vllm_log_config = resolve_replay_vllm_log_config(
        port_profile_id=resolved_port_profile_id_int,
        interval_s=vllm_log_interval_s,
        timeout_s=vllm_log_timeout_s,
    )

    stop_event = threading.Event()
    lock = threading.Lock()
    stop_lock = threading.Lock()
    worker_results: dict[str, dict[str, Any]] = {}
    stop_reason: dict[str, str | None] = {"value": None}

    def set_stop_reason(reason: str) -> None:
        with stop_lock:
            if stop_reason["value"] is None:
                stop_reason["value"] = reason
        stop_event.set()

    def current_stop_reason() -> str | None:
        with stop_lock:
            return stop_reason["value"]

    replay_deadline_at = (
        time.monotonic() + time_constraint_s
        if time_constraint_s is not None
        else None
    )
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
        "vllm_log_enabled": vllm_log_config.enabled,
        "vllm_log_endpoint": vllm_log_config.endpoint,
        "vllm_log_interval_s": vllm_log_config.interval_s,
        "vllm_log_timeout_s": vllm_log_config.timeout_s,
        "vllm_log_dir": str(output_dir / "vllm-log") if vllm_log_config.enabled else None,
        "workers_total": len(scheduled_workers),
        "workers_completed": 0,
        "workers_failed": 0,
        "workers_timed_out": 0,
        "workers_time_bound_finished": 0,
        "requests_sent": 0,
        "requests_failed": 0,
        "time_constraint_reached": False,
    }
    if randomize_seed is not None:
        summary["randomize_seed"] = randomize_seed
    if time_constraint_s is not None:
        summary["time_constraint_s"] = time_constraint_s
        summary["workers_total"] = 0
    if config_file_path is not None:
        summary["source_config"] = str(config_file_path.expanduser().resolve())
    if agent_timeout_s is not None:
        summary["agent_timeout_s"] = agent_timeout_s
    if requested_num_tasks is not None:
        summary["num_tasks_requested"] = requested_num_tasks
        summary["workers_in_plan"] = len(workers)
        summary["tasks_wrapped"] = tasks_wrapped
    progress_nonlocal: list[Any] = [_NullProgress()]
    progress_task_id_nonlocal = [0]
    progress_state = {"launched": 0, "active": 0}
    summary_path = output_dir / "replay" / "summary.json"

    def _update_progress(*, advance: int = 0, total: int | None = None) -> None:
        kwargs: dict[str, Any] = {
            "advance": advance,
            "launched": progress_state["launched"],
            "active": progress_state["active"],
            "failed": summary["workers_failed"],
        }
        if total is not None:
            kwargs["total"] = total
        progress_nonlocal[0].update(progress_task_id_nonlocal[0], **kwargs)

    def call_gateway(path: str, payload: dict[str, Any]) -> Any:
        if not gateway_url:
            raise RuntimeError("Gateway URL is required for lifecycle calls")
        status, response_payload = http_json(
            method="POST",
            url=join_url(gateway_url, path),
            payload=payload,
            timeout_s=replay_http_timeout_s,
        )
        if status >= 400:
            raise RuntimeError(
                f"Gateway lifecycle call failed: {path} HTTP {status} payload={response_payload}"
            )
        return response_payload

    monitor: ReplayVLLMMonitorProcess | None = None
    if vllm_log_config.enabled:
        monitor = start_replay_vllm_monitor(output_dir=output_dir, config=vllm_log_config)
        summary["vllm_log_stdout"] = str(monitor.stdout_log)
        summary["vllm_log_stderr"] = str(monitor.stderr_log)

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
        if agent_timeout_s is not None:
            record["agent_timeout_s"] = agent_timeout_s
        if time_constraint_s is not None:
            record["time_constraint_s"] = time_constraint_s

        api_token = worker.get("api_token")
        if not isinstance(api_token, str) or not api_token:
            api_token = None
        agent_started = False

        def _status_from_stop_reason() -> str:
            return (
                "time_bound_finished"
                if current_stop_reason() == "time_constraint"
                else "cancelled"
            )

        def _deadline_kind(agent_deadline: float | None) -> str:
            if replay_deadline_at is None:
                return "agent_timeout"
            if agent_deadline is None:
                return "time_constraint"
            return "time_constraint" if replay_deadline_at <= agent_deadline else "agent_timeout"

        def _remaining(deadline_at: float | None) -> float | None:
            if deadline_at is None:
                return None
            return max(0.0, deadline_at - time.monotonic())

        try:
            delta_agent_start_s = float(worker.get("delta_agent_start_s", 0.0))
            sleep_result = sleep_with_stop_or_deadline(
                stop_event,
                seconds=max(0.0, delta_agent_start_s),
                deadline_at=replay_deadline_at,
            )
            if sleep_result == "stopped":
                record["status"] = _status_from_stop_reason()
                return
            if sleep_result == "deadline":
                record["status"] = "time_bound_finished"
                summary["time_constraint_reached"] = True
                set_stop_reason("time_constraint")
                return

            if use_gateway_lifecycle:
                if not api_token:
                    raise RuntimeError(
                        f"Missing worker api_token for gateway lifecycle: {worker_id}"
                    )
                call_gateway("/agent/start", {"api_token": api_token})
                agent_started = True

            agent_deadline_at = (
                time.monotonic() + agent_timeout_s
                if agent_timeout_s is not None
                else None
            )
            runtime_deadline_at = agent_deadline_at
            if replay_deadline_at is not None and (
                runtime_deadline_at is None
                or replay_deadline_at < runtime_deadline_at
            ):
                runtime_deadline_at = replay_deadline_at

            delta_first_request_s = float(worker.get("delta_first_request_s", 0.0))
            sleep_result = sleep_with_stop_or_deadline(
                stop_event,
                seconds=max(0.0, delta_first_request_s),
                deadline_at=runtime_deadline_at,
            )
            if sleep_result == "stopped":
                record["status"] = _status_from_stop_reason()
                return
            if sleep_result == "deadline":
                deadline_kind = _deadline_kind(agent_deadline_at)
                if deadline_kind == "time_constraint":
                    record["status"] = "time_bound_finished"
                    summary["time_constraint_reached"] = True
                    set_stop_reason("time_constraint")
                else:
                    record["status"] = "timed_out"
                    record["error"] = f"agent timeout exceeded after {agent_timeout_s:.3f}s"
                return

            planned_requests = worker.get("requests")
            if not isinstance(planned_requests, list):
                raise RuntimeError(f"Worker requests must be list: {worker_id}")
            record["requests_total"] = len(planned_requests)

            for req in planned_requests:
                if stop_event.is_set():
                    record["status"] = _status_from_stop_reason()
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

                remaining_agent_s: float | None
                if agent_deadline_at is None:
                    remaining_agent_s = None
                else:
                    remaining_agent_s = _remaining(agent_deadline_at)
                    if remaining_agent_s <= 0:
                        record["status"] = "timed_out"
                        record["error"] = f"agent timeout exceeded after {agent_timeout_s:.3f}s"
                        return
                remaining_replay_s = _remaining(replay_deadline_at)
                if remaining_replay_s is not None and remaining_replay_s <= 0:
                    record["status"] = "time_bound_finished"
                    summary["time_constraint_reached"] = True
                    set_stop_reason("time_constraint")
                    return

                replay_mode = str(req.get("replay_mode") or "deterministic_forced_tokens")
                if replay_mode == "client_disconnect_after_duration":
                    cancel_after_s = float(req.get("cancel_after_s", 0.0))
                    timeout_candidates: list[tuple[str, float]] = [("client_disconnect", cancel_after_s)]
                    if remaining_agent_s is not None:
                        timeout_candidates.append(("agent_timeout", remaining_agent_s))
                    if remaining_replay_s is not None:
                        timeout_candidates.append(("time_constraint", remaining_replay_s))
                    timeout_reason, effective_cancel_after_s = min(
                        timeout_candidates,
                        key=lambda item: item[1],
                    )
                    timed_result = timed_cancel_http_json(
                        method=method,
                        url=join_url(api_base, path),
                        payload=body,
                        cancel_after_s=effective_cancel_after_s,
                        connect_timeout_s=replay_http_timeout_s,
                        headers=headers,
                        stop_event=stop_event,
                    )
                    with lock:
                        summary["requests_sent"] += 1
                    outcome = timed_result.get("outcome")
                    if outcome == "stopped":
                        record["status"] = _status_from_stop_reason()
                        return
                    if outcome == "cancelled":
                        if timeout_reason == "client_disconnect":
                            pass
                        elif timeout_reason == "agent_timeout":
                            record["status"] = "timed_out"
                            record["error"] = (
                                f"agent timeout exceeded after {agent_timeout_s:.3f}s"
                            )
                            record["requests_failed"] += 1
                            with lock:
                                summary["requests_failed"] += 1
                            return
                        elif timeout_reason == "time_constraint":
                            record["status"] = "time_bound_finished"
                            summary["time_constraint_reached"] = True
                            set_stop_reason("time_constraint")
                            return
                        else:
                            raise RuntimeError(
                                f"Unexpected timeout reason for client-disconnect replay: {timeout_reason!r}"
                            )
                    if (
                        outcome == "completed_early"
                        and is_acceptable_client_disconnect_early_error(timed_result)
                    ):
                        record["requests_succeeded"] += 1
                        sleep_result = sleep_with_stop_or_deadline(
                            stop_event,
                            seconds=float(req.get("delta_agent_action_after_s", 0.0)),
                            deadline_at=runtime_deadline_at,
                        )
                        if sleep_result == "stopped":
                            record["status"] = _status_from_stop_reason()
                            return
                        if sleep_result == "deadline":
                            deadline_kind = _deadline_kind(agent_deadline_at)
                            if deadline_kind == "time_constraint":
                                record["status"] = "time_bound_finished"
                                summary["time_constraint_reached"] = True
                                set_stop_reason("time_constraint")
                            else:
                                record["status"] = "timed_out"
                                record["error"] = (
                                    f"agent timeout exceeded after {agent_timeout_s:.3f}s"
                                )
                            return
                        continue
                    if outcome != "cancelled":
                        record["requests_failed"] += 1
                        with lock:
                            summary["requests_failed"] += 1
                        raise RuntimeError(
                            "Timed disconnect replay did not cancel cleanly: "
                            f"worker={worker_id}, path={path}, outcome={outcome}, "
                            f"response_status={timed_result.get('response_status')}, "
                            f"error={timed_result.get('error')}"
                        )
                else:
                    request_timeout_candidates: list[tuple[str, float]] = []
                    if remaining_agent_s is not None:
                        request_timeout_candidates.append(("agent_timeout", remaining_agent_s))
                    if remaining_replay_s is not None:
                        request_timeout_candidates.append(("time_constraint", remaining_replay_s))

                    if not request_timeout_candidates:
                        status, response_payload = http_json(
                            method=method,
                            url=join_url(api_base, path),
                            payload=body,
                            timeout_s=replay_http_timeout_s,
                            headers=headers,
                        )
                        with lock:
                            summary["requests_sent"] += 1
                    else:
                        request_timeout_reason, request_cancel_after_s = min(
                            request_timeout_candidates,
                            key=lambda item: item[1],
                        )
                        timed_result = timed_cancel_http_json(
                            method=method,
                            url=join_url(api_base, path),
                            payload=body,
                            cancel_after_s=request_cancel_after_s,
                            connect_timeout_s=replay_http_timeout_s,
                            headers=headers,
                            stop_event=stop_event,
                        )
                        outcome = timed_result.get("outcome")
                        with lock:
                            summary["requests_sent"] += 1
                        if outcome == "stopped":
                            record["status"] = _status_from_stop_reason()
                            return
                        if outcome == "cancelled":
                            if request_timeout_reason == "agent_timeout":
                                record["status"] = "timed_out"
                                record["error"] = (
                                    f"agent timeout exceeded after {agent_timeout_s:.3f}s"
                                )
                                record["requests_failed"] += 1
                                with lock:
                                    summary["requests_failed"] += 1
                            elif request_timeout_reason == "time_constraint":
                                record["status"] = "time_bound_finished"
                                summary["time_constraint_reached"] = True
                                set_stop_reason("time_constraint")
                            else:
                                raise RuntimeError(
                                    f"Unexpected request timeout reason: {request_timeout_reason!r}"
                                )
                            return
                        if outcome != "completed_early":
                            record["requests_failed"] += 1
                            with lock:
                                summary["requests_failed"] += 1
                            raise RuntimeError(
                                "Replay request ended unexpectedly: "
                                f"worker={worker_id}, path={path}, outcome={outcome}, "
                                f"response_status={timed_result.get('response_status')}, "
                                f"error={timed_result.get('error')}"
                            )
                        status = int(timed_result.get("response_status") or 0)
                        response_payload = timed_result.get("response_payload")
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
                sleep_result = sleep_with_stop_or_deadline(
                    stop_event,
                    seconds=max(0.0, delay_after_s),
                    deadline_at=runtime_deadline_at,
                )
                if sleep_result == "stopped":
                    record["status"] = _status_from_stop_reason()
                    return
                if sleep_result == "deadline":
                    deadline_kind = _deadline_kind(agent_deadline_at)
                    if deadline_kind == "time_constraint":
                        record["status"] = "time_bound_finished"
                        summary["time_constraint_reached"] = True
                        set_stop_reason("time_constraint")
                    else:
                        record["status"] = "timed_out"
                        record["error"] = f"agent timeout exceeded after {agent_timeout_s:.3f}s"
                    return

            record["status"] = "completed"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = str(exc)
            set_stop_reason("failure")
        finally:
            if use_gateway_lifecycle and api_token and agent_started:
                try:
                    rc = (
                        0
                        if record["status"] in {"completed", "timed_out", "time_bound_finished"}
                        else 1
                    )
                    call_gateway(
                        "/agent/end",
                        {"api_token": api_token, "return_code": rc},
                    )
                except Exception as exc:  # noqa: BLE001
                    if record["status"] == "completed":
                        record["status"] = "failed"
                        record["error"] = f"agent/end failed: {exc}"
                    set_stop_reason("failure")

            record["finished_at"] = now_iso8601_utc()
            with lock:
                worker_results[worker_id] = record
                if record["status"] == "completed":
                    summary["workers_completed"] += 1
                elif record["status"] == "timed_out":
                    summary["workers_timed_out"] += 1
                elif record["status"] == "time_bound_finished":
                    summary["workers_time_bound_finished"] += 1
                elif record["status"] == "failed":
                    summary["workers_failed"] += 1
                progress_state["active"] = max(progress_state["active"] - 1, 0)
                _update_progress(advance=1)
            write_json(worker_log_path, record)

    try:
        if use_gateway_lifecycle:
            gateway_output_dir = output_dir / "gateway-output"
            gateway_output_dir.mkdir(parents=True, exist_ok=True)
            call_gateway(
                "/job/start",
                {"output_location": str(gateway_output_dir)},
            )

        threads: list[threading.Thread] = []
        if time_constraint_s is None:
            for worker in scheduled_workers:
                worker_id = str(worker.get("worker_id") or worker.get("trial_id") or "worker")
                thread = threading.Thread(
                    target=worker_fn,
                    args=(worker,),
                    name=f"replay-{worker_id}",
                )
                thread.daemon = True
                threads.append(thread)

        started_threads: list[threading.Thread] = []

        with create_replay_progress() as progress:
            progress_nonlocal[0] = progress
            progress_task_id_nonlocal[0] = progress.add_task(
                "replaying",
                total=len(threads) if time_constraint_s is None else 1,
                launched=0,
                active=0,
                failed=0,
            )
            try:
                next_launch_at = time.monotonic()
                if time_constraint_s is None:
                    for thread in threads:
                        while True:
                            active_threads = [item for item in started_threads if item.is_alive()]
                            if len(active_threads) != len(started_threads):
                                started_threads = active_threads
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
                else:
                    if replay_deadline_at is None:
                        raise RuntimeError("internal error: missing replay_deadline_at")

                    source_cycle: list[int] = list(range(len(ordered_workers)))
                    source_cursor = 0
                    source_repeat_count: dict[str, int] = {}
                    source_rng = (
                        Random(randomize_seed)
                        if randomize_seed is not None
                        else None
                    )
                    if source_rng is not None:
                        source_rng.shuffle(source_cycle)
                    launch_count = 0

                    def next_worker_payload() -> dict[str, Any]:
                        nonlocal source_cursor, launch_count, source_cycle
                        if source_cursor >= len(source_cycle):
                            source_cursor = 0
                            source_cycle = list(range(len(ordered_workers)))
                            if source_rng is not None:
                                source_rng.shuffle(source_cycle)
                        source_index = source_cycle[source_cursor]
                        source_cursor += 1
                        source_worker = ordered_workers[source_index]
                        launch_count += 1

                        source_worker_id = str(
                            source_worker.get("worker_id")
                            or source_worker.get("trial_id")
                            or "worker"
                        )
                        next_repeat = source_repeat_count.get(source_worker_id, 0) + 1
                        source_repeat_count[source_worker_id] = next_repeat

                        if next_repeat == 1:
                            return source_worker

                        worker_copy = copy.deepcopy(source_worker)
                        worker_copy["worker_id"] = (
                            f"{source_worker_id}__wrap{next_repeat}__task{launch_count}"
                        )
                        worker_copy["wrapped_from_worker_id"] = source_worker_id
                        worker_copy["wrapped_from_task_index"] = source_index + 1
                        source_api_token = source_worker.get("api_token")
                        if isinstance(source_api_token, str) and source_api_token:
                            worker_copy["api_token"] = (
                                f"{source_api_token}__wrap{next_repeat}__task{launch_count}"
                            )
                        return worker_copy

                    while True:
                        if stop_event.is_set():
                            break
                        if time.monotonic() >= replay_deadline_at:
                            summary["time_constraint_reached"] = True
                            set_stop_reason("time_constraint")
                            break

                        while True:
                            active_threads = [item for item in started_threads if item.is_alive()]
                            if len(active_threads) != len(started_threads):
                                started_threads = active_threads
                            if len(active_threads) < launch_max_concurrent:
                                break
                            if stop_event.is_set():
                                break
                            if time.monotonic() >= replay_deadline_at:
                                summary["time_constraint_reached"] = True
                                set_stop_reason("time_constraint")
                                break
                            for active_thread in active_threads:
                                active_thread.join(timeout=0.2)
                        if stop_event.is_set():
                            break

                        launch_delay = next_launch_at - time.monotonic()
                        if launch_delay > 0:
                            sleep_result = sleep_with_stop_or_deadline(
                                stop_event,
                                seconds=launch_delay,
                                deadline_at=replay_deadline_at,
                            )
                            if sleep_result == "stopped":
                                break
                            if sleep_result == "deadline":
                                summary["time_constraint_reached"] = True
                                set_stop_reason("time_constraint")
                                break
                        if stop_event.is_set():
                            break

                        worker = next_worker_payload()
                        worker_id = str(worker.get("worker_id") or worker.get("trial_id") or "worker")
                        thread = threading.Thread(
                            target=worker_fn,
                            args=(worker,),
                            name=f"replay-{worker_id}",
                        )
                        thread.daemon = True
                        with lock:
                            summary["workers_total"] += 1
                            progress_state["launched"] += 1
                            progress_state["active"] += 1
                            _update_progress(total=max(1, summary["workers_total"]))
                        try:
                            thread.start()
                        except Exception:
                            with lock:
                                summary["workers_total"] = max(summary["workers_total"] - 1, 0)
                                progress_state["launched"] = max(progress_state["launched"] - 1, 0)
                                progress_state["active"] = max(progress_state["active"] - 1, 0)
                                _update_progress(total=max(1, summary["workers_total"]))
                            raise
                        started_threads.append(thread)
                        next_launch_at = time.monotonic() + max(0.0, next_launch_delay_s())

                while any(thread.is_alive() for thread in started_threads):
                    started_threads = [thread for thread in started_threads if thread.is_alive()]
                    if not started_threads:
                        break
                    if (
                        replay_deadline_at is not None
                        and not stop_event.is_set()
                        and time.monotonic() >= replay_deadline_at
                    ):
                        summary["time_constraint_reached"] = True
                        set_stop_reason("time_constraint")
                    for thread in started_threads:
                        thread.join(timeout=0.2)
            except KeyboardInterrupt:
                set_stop_reason("keyboard_interrupt")
                for thread in started_threads:
                    thread.join(timeout=2)

        if use_gateway_lifecycle:
            try:
                status = "completed" if summary["workers_failed"] == 0 else "failed"
                call_gateway("/job/end", {"status": status})
            except Exception as exc:  # noqa: BLE001
                summary["gateway_job_end_error"] = str(exc)
                summary["workers_failed"] += 1
    finally:
        if monitor is not None:
            summary["vllm_log_monitor_return_code"] = stop_replay_vllm_monitor(monitor)
        summary["finished_at"] = now_iso8601_utc()
        summary["worker_results"] = worker_results
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
                "workers_timed_out": summary["workers_timed_out"],
                "workers_time_bound_finished": summary["workers_time_bound_finished"],
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
        "--config",
        default=None,
        help=(
            "Path to TOML config file. CLI options override config values. "
            "Supports [compile] or [replayer.compile] sections."
        ),
    )
    compile_parser.add_argument(
        "--job-dir",
        default=None,
        help="Path to profiled con-driver job directory.",
    )
    compile_parser.add_argument(
        "--plan-out",
        default=None,
        help="Output path for replay plan. Default: <job-dir>/replay-plan.json",
    )
    compile_parser.add_argument(
        "--port-profile-id",
        type=int,
        default=None,
        help=(
            "Port profile used to resolve compile-time tokenize endpoint from "
            "configs/port_profiles.toml."
        ),
    )
    compile_parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=None,
        help=(
            "Optional HTTP timeout seconds for compile-time tokenizer requests. "
            f"Default: {DEFAULT_COMPILE_TOKENIZE_TIMEOUT_S:.0f}s."
        ),
    )
    compile_parser.set_defaults(func=cmd_compile)

    replay_parser = subparsers.add_parser(
        "replay",
        help="Run replay from a compiled replay-plan.json",
    )
    replay_parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to TOML config file. CLI options override config values. "
            "Supports [replay] or [replayer.replay] sections."
        ),
    )
    replay_parser.add_argument(
        "--plan",
        default=None,
        help="Path to compiled replay-plan.json.",
    )
    replay_parser.add_argument(
        "--output-dir",
        default=None,
        help="Replay output directory. Default: <plan-dir>/<job>.replayed-<ts>",
    )
    replay_parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help=(
            "Replay exactly this many tasks. If smaller than the plan worker count, "
            "replay the first tasks in launch order. If larger, wrap and repeat from "
            "the beginning of launch order."
        ),
    )
    replay_parser.add_argument(
        "--randomize-seed",
        type=int,
        default=None,
        help=(
            "Optional worker-order randomization seed. "
            "When set, replay does not follow plan launch order."
        ),
    )
    replay_parser.add_argument(
        "--time-constraint-s",
        type=float,
        default=None,
        help=(
            "Optional replay wall-time limit in seconds. "
            "When set, replay launches unbounded tasks until this deadline."
        ),
    )
    replay_parser.add_argument(
        "--port-profile-id",
        type=int,
        required=True,
        help=(
            "Required. Resolve replay target URLs from configs/port_profiles.toml."
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
    replay_parser.add_argument(
        "--agent-timeout-s",
        type=float,
        default=None,
        help=(
            "Optional per-worker runtime limit in seconds. "
            "When set, replay terminates a worker that exceeds this duration."
        ),
    )
    replay_parser.add_argument(
        "--vllm-log",
        dest="vllm_log_explicit_on",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    replay_parser.add_argument(
        "--no-vllm-log",
        dest="vllm_log_explicit_off",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    replay_parser.add_argument(
        "--vllm-log-interval-s",
        type=float,
        default=None,
        help="Optional replay vLLM metrics sampling interval in seconds.",
    )
    replay_parser.add_argument(
        "--vllm-log-timeout-s",
        type=float,
        default=None,
        help="Optional replay vLLM metrics scrape timeout in seconds.",
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
