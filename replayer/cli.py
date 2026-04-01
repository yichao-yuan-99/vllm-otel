"""CLI for compiling and replaying gateway-profiled jobs."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import copy
import http.client
import hashlib
import io
import json
from contextlib import AbstractContextManager, redirect_stdout, suppress
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


REPLAY_PLAN_SCHEMA_VERSION = "replay-plan.v1"
# NOTE FOR MAINTAINERS:
# `cmd_compile` reuses an existing replay plan when its `compile_version` matches
# this value. Missing/empty `compile_version` is interpreted as v1 for backward
# compatibility.
# If compile-time semantics change in a way that requires rebuilding plans,
# increment this version.
REPLAY_PLAN_COMPILE_VERSION = "1"
REPLAY_PLAN_COMPILE_VERSION_V1 = "1"
SPLIT_TWO_GROUP_METRICS = {"token_usage", "context_usage"}
DEFAULT_SPLIT_TWO_GROUP_METRIC = "token_usage"
DEFAULT_SPLIT_TWO_GROUP_METRICS = ("token_usage", "context_usage")
SPLIT_TWO_GROUP_PLAN_METRIC_ALIASES = {
    "token_usage": "token",
    "context_usage": "context",
}
COMPILE_CLEAN_PLAN_SUFFIX = "clean"
CLEANABLE_REQUEST_STATUS_CODE = 499

DEFAULT_VLLM_LOG_INTERVAL_S = 1.0
DEFAULT_VLLM_LOG_TIMEOUT_S = 3600.0
DEFAULT_LMCACHE_LOG_PROBE_TIMEOUT_S = 2.0
DEFAULT_COMPILE_TOKENIZE_TIMEOUT_S = 3600.0
_MONITOR_INTERRUPT_GRACE_SEC = 3600.0
_MONITOR_TERMINATE_GRACE_SEC = 3600.0


@dataclass(frozen=True)
class ReplayVLLMLogConfig:
    enabled: bool
    endpoint: str
    interval_s: float
    timeout_s: float


@dataclass(frozen=True)
class ReplayLMCacheLogConfig:
    configured: bool
    endpoint: str | None
    interval_s: float
    timeout_s: float
    probe_timeout_s: float


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


class _TeeTextIO(io.TextIOBase):
    def __init__(self, *streams: Any) -> None:
        super().__init__()
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            with suppress(Exception):
                stream.flush()


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
    if RichProgress is None or os.environ.get("REPLAYER_NO_PROGRESS") == "1":
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


def create_batch_compile_progress() -> Any:
    if RichProgress is None:
        return _NullProgress()
    return RichProgress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} jobs"),
        TextColumn("ok={task.fields[succeeded]} failed={task.fields[failed]}"),
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


def resolve_replay_lmcache_log_config(
    *,
    port_profile_id: int | None,
    interval_s: float,
    timeout_s: float,
    probe_timeout_s: float = DEFAULT_LMCACHE_LOG_PROBE_TIMEOUT_S,
) -> ReplayLMCacheLogConfig:
    if interval_s <= 0:
        raise ValueError("--vllm-log-interval-s must be > 0")
    if timeout_s <= 0:
        raise ValueError("--vllm-log-timeout-s must be > 0")
    if probe_timeout_s <= 0:
        raise ValueError("LMCache log probe timeout must be > 0")
    if port_profile_id is None:
        return ReplayLMCacheLogConfig(
            configured=False,
            endpoint=None,
            interval_s=interval_s,
            timeout_s=timeout_s,
            probe_timeout_s=probe_timeout_s,
        )
    from gateway.port_profiles import load_port_profile

    profile = load_port_profile(port_profile_id)
    if profile.lmcache_port is None:
        return ReplayLMCacheLogConfig(
            configured=False,
            endpoint=None,
            interval_s=interval_s,
            timeout_s=timeout_s,
            probe_timeout_s=probe_timeout_s,
        )
    endpoint = f"http://127.0.0.1:{profile.lmcache_port}/metrics"
    return ReplayLMCacheLogConfig(
        configured=True,
        endpoint=endpoint,
        interval_s=interval_s,
        timeout_s=timeout_s,
        probe_timeout_s=probe_timeout_s,
    )


def probe_metrics_endpoint(*, endpoint: str, timeout_s: float) -> tuple[bool, str | None]:
    request = url_request.Request(url=endpoint, method="GET")
    try:
        with url_request.urlopen(request, timeout=timeout_s) as response:
            status = int(response.getcode())
        if status >= 400:
            return False, f"HTTP {status}"
        return True, None
    except url_error.HTTPError as exc:
        return False, f"HTTP {int(exc.code)}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _build_monitor_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    con_driver_src = repo_root / "con-driver" / "src"
    pythonpath_parts = [str(con_driver_src)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _start_replay_metrics_monitor(
    *,
    output_dir: Path,
    endpoint: str,
    interval_s: float,
    timeout_s: float,
    log_dir_name: str,
) -> ReplayVLLMMonitorProcess:
    metrics_log_dir = output_dir / log_dir_name
    metrics_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = metrics_log_dir / "monitor.stdout.log"
    stderr_log = metrics_log_dir / "monitor.stderr.log"
    stdout_handle = stdout_log.open("w", encoding="utf-8")
    stderr_handle = stderr_log.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "con_driver.vllm_metrics_monitor",
        "--endpoint",
        endpoint,
        "--output-dir",
        str(metrics_log_dir),
        "--interval-s",
        str(interval_s),
        "--timeout-s",
        str(timeout_s),
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


def start_replay_vllm_monitor(
    *,
    output_dir: Path,
    config: ReplayVLLMLogConfig,
) -> ReplayVLLMMonitorProcess:
    return _start_replay_metrics_monitor(
        output_dir=output_dir,
        endpoint=config.endpoint,
        interval_s=config.interval_s,
        timeout_s=config.timeout_s,
        log_dir_name="vllm-log",
    )


def start_replay_lmcache_monitor(
    *,
    output_dir: Path,
    config: ReplayLMCacheLogConfig,
) -> ReplayVLLMMonitorProcess:
    if config.endpoint is None:
        raise ValueError("LMCache metrics endpoint is required to start lmcache monitor")
    return _start_replay_metrics_monitor(
        output_dir=output_dir,
        endpoint=config.endpoint,
        interval_s=config.interval_s,
        timeout_s=config.timeout_s,
        log_dir_name="lmcache-log",
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


def resolve_configured_compile_model_override(
    model_override: str,
) -> tuple[str, str]:
    try:
        from gateway.model_configs import load_model_registry
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "failed to import gateway model registry for --model validation"
        ) from exc

    registry = load_model_registry()
    spec = registry.resolve(model_override)
    if spec is None:
        allowed_keys = ", ".join(sorted(registry.models.keys()))
        raise ValueError(
            "--model must match a name configured in configs/model_config.toml "
            f"(model key, served_model_name, or vllm_model_name). Got {model_override!r}. "
            f"Allowed model keys: {allowed_keys}"
        )
    return spec.key, spec.served_model_name


def is_replay_plan_compile_version_current(plan_payload: dict[str, Any]) -> bool:
    raw_version = plan_payload.get("compile_version")
    if raw_version is None:
        return REPLAY_PLAN_COMPILE_VERSION_V1 == REPLAY_PLAN_COMPILE_VERSION
    if isinstance(raw_version, str):
        stripped_version = raw_version.strip()
        if not stripped_version:
            return REPLAY_PLAN_COMPILE_VERSION_V1 == REPLAY_PLAN_COMPILE_VERSION
        return stripped_version == REPLAY_PLAN_COMPILE_VERSION
    return False


def count_plan_workers_and_requests(plan_payload: dict[str, Any]) -> tuple[int, int]:
    workers_payload = plan_payload.get("workers")
    if not isinstance(workers_payload, list):
        return 0, 0

    request_count = 0
    for worker_payload in workers_payload:
        if not isinstance(worker_payload, dict):
            continue
        requests_payload = worker_payload.get("requests")
        if isinstance(requests_payload, list):
            request_count += len(requests_payload)
    return len(workers_payload), request_count


def extract_plan_compile_model_override(plan_payload: dict[str, Any]) -> str | None:
    compile_options = plan_payload.get("compile_options")
    if not isinstance(compile_options, dict):
        return None
    model_override = compile_options.get("model_override")
    if not isinstance(model_override, str):
        return None
    stripped = model_override.strip()
    return stripped or None


def extract_plan_compile_clean(plan_payload: dict[str, Any]) -> bool:
    compile_options = plan_payload.get("compile_options")
    if not isinstance(compile_options, dict):
        return False
    return bool(compile_options.get("clean"))


def extract_plan_compile_single_trail(plan_payload: dict[str, Any]) -> str | None:
    compile_options = plan_payload.get("compile_options")
    if not isinstance(compile_options, dict):
        return None
    single_trail = compile_options.get("single_trail")
    if not isinstance(single_trail, str):
        return None
    stripped = single_trail.strip()
    return stripped or None


def extract_plan_replay_model(plan_payload: dict[str, Any]) -> str | None:
    replay_target = plan_payload.get("replay_target")
    if not isinstance(replay_target, dict):
        return None
    replay_model = replay_target.get("model")
    if not isinstance(replay_model, str):
        return None
    stripped = replay_model.strip()
    return stripped or None


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


def _extract_rate_per_second_from_pattern_args(
    pattern_args_payload: Any,
    *,
    pattern_label: str,
) -> float | None:
    if not isinstance(pattern_args_payload, dict):
        return None

    rate_keys = ("rate", "arrival-rate", "arrival_rate", "lambda")
    for key in rate_keys:
        rate_value = _coerce_pattern_arg_float(pattern_args_payload.get(key))
        if rate_value is not None:
            if rate_value <= 0:
                raise ValueError(f"{pattern_label} rate must be > 0")
            return rate_value

    mean_keys = ("mean-interval-s", "mean_interval_s", "interval-s", "interval_s")
    for key in mean_keys:
        mean_value = _coerce_pattern_arg_float(pattern_args_payload.get(key))
        if mean_value is not None:
            if mean_value <= 0:
                raise ValueError(f"{pattern_label} mean interval must be > 0")
            return 1.0 / mean_value

    return None


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
    elif pattern_name in {"poisson", "possion", "uniform"}:
        normalized_pattern_name = (
            "poisson" if pattern_name in {"poisson", "possion"} else "uniform"
        )
        pattern_label = (
            "Poisson" if normalized_pattern_name == "poisson" else "Uniform"
        )
        rate_per_second = _extract_rate_per_second_from_pattern_args(
            pattern_args,
            pattern_label=pattern_label,
        )
        if rate_per_second is None:
            raise ValueError(
                f"{pattern_label} pattern requires one of: "
                "--rate=<arrivals_per_second> or --mean-interval-s=<seconds>"
            )
        pattern_payload = {
            "name": normalized_pattern_name,
            "rate_per_second": rate_per_second,
            "mean_interval_s": 1.0 / rate_per_second,
        }
    else:
        raise ValueError(
            f"Unsupported run.pattern={pattern_name!r}; supported: eager, poisson, uniform"
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


def _is_profiled_job_dir(path: Path) -> bool:
    required_files = [
        path / "meta" / "config.toml",
        path / "meta" / "run_manifest.json",
        path / "meta" / "results.json",
        path / "meta" / "events.jsonl",
    ]
    if not all(file_path.is_file() for file_path in required_files):
        return False
    return (path / "gateway-output").is_dir()


def discover_profiled_job_dirs(job_root: Path) -> list[Path]:
    resolved_root = job_root.expanduser().resolve()
    discovered: set[Path] = set()

    if _is_profiled_job_dir(resolved_root):
        discovered.add(resolved_root)

    for config_path in resolved_root.rglob("meta/config.toml"):
        candidate_job_dir = config_path.parent.parent.resolve()
        if _is_profiled_job_dir(candidate_job_dir):
            discovered.add(candidate_job_dir)

    return sorted(discovered)


def _job_root_default_max_workers(job_count: int) -> int:
    if job_count <= 0:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(job_count, cpu_count, 8))


def _build_job_root_child_compile_command(
    *,
    job_dir: Path,
    port_profile_id: int,
    request_timeout_s: float | None,
    model_override: str | None,
    single_trail: str | None,
    split_two_group_plans: bool,
    split_two_group_metric: str,
    exclude_unranked_trails: bool,
    clean: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "replayer",
        "compile",
        "--job-dir",
        str(job_dir),
        "--port-profile-id",
        str(port_profile_id),
    ]
    if request_timeout_s is not None:
        command.extend(["--request-timeout-s", str(request_timeout_s)])
    if model_override is not None:
        command.extend(["--model", model_override])
    if single_trail is not None:
        command.extend(["--single-trail", single_trail])
    if split_two_group_plans:
        command.append("--split-two-group-plans")
        command.extend(["--split-two-group-metric", split_two_group_metric])
    if exclude_unranked_trails:
        command.append("--exclude-unranked-trails")
    if clean:
        command.append("--clean")
    return command


def _run_job_root_child_compile(
    *,
    job_dir: Path,
    port_profile_id: int,
    request_timeout_s: float | None,
    model_override: str | None,
    single_trail: str | None,
    split_two_group_plans: bool,
    split_two_group_metric: str,
    exclude_unranked_trails: bool,
    clean: bool,
) -> dict[str, Any]:
    command = _build_job_root_child_compile_command(
        job_dir=job_dir,
        port_profile_id=port_profile_id,
        request_timeout_s=request_timeout_s,
        model_override=model_override,
        single_trail=single_trail,
        split_two_group_plans=split_two_group_plans,
        split_two_group_metric=split_two_group_metric,
        exclude_unranked_trails=exclude_unranked_trails,
        clean=clean,
    )
    env = os.environ.copy()
    env["REPLAYER_NO_PROGRESS"] = "1"
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        error_text = completed.stderr.strip()
        if not error_text:
            error_text = completed.stdout.strip() or f"compile exited with {completed.returncode}"
        return {
            "job_dir": str(job_dir),
            "status": "failed",
            "error": error_text,
            "exit_code": completed.returncode,
        }
    stdout = completed.stdout.strip()
    parsed_summary: dict[str, Any] | None = None
    if stdout:
        with suppress(Exception):
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                parsed_summary = parsed
    result: dict[str, Any] = {
        "job_dir": str(job_dir),
        "status": "ok",
    }
    if parsed_summary is not None:
        result["summary"] = parsed_summary
    return result


def build_gateway_trail_name(
    *,
    run_dir: Path,
    source_port_profile_id: int | None,
) -> str:
    run_id = run_dir.name
    if isinstance(source_port_profile_id, int):
        return f"profile-{source_port_profile_id}/{run_id}"
    return run_id


def resolve_split_two_group_file_path(
    *,
    job_dir: Path,
    metric: str,
) -> Path:
    if metric == "token_usage":
        file_name = "top-p-token-usage-two-groups.json"
    elif metric == "context_usage":
        file_name = "top-p-context-usage-two-groups.json"
    else:
        raise ValueError(
            f"Unsupported split two-group metric {metric!r}; "
            f"supported metrics: {sorted(SPLIT_TWO_GROUP_METRICS)}"
        )
    return (job_dir / "original-analysis" / "split" / file_name).resolve()


def resolve_split_summary_file_path(split_payload_path: Path) -> Path:
    return (split_payload_path.parent / "top-p-usage-ratio-summary.json").resolve()


def _build_split_summary_trail_name(item: dict[str, Any]) -> str | None:
    trail_name_raw = item.get("trail_name")
    if isinstance(trail_name_raw, str):
        trail_name = trail_name_raw.strip()
        if trail_name:
            return trail_name

    run_id_raw = item.get("gateway_run_id")
    if not isinstance(run_id_raw, str):
        return None
    run_id = run_id_raw.strip()
    if not run_id:
        return None

    profile_id_raw = item.get("gateway_profile_id")
    if profile_id_raw is None:
        return run_id
    if isinstance(profile_id_raw, bool):
        return None
    if isinstance(profile_id_raw, int):
        return f"profile-{profile_id_raw}/{run_id}"
    if isinstance(profile_id_raw, str) and profile_id_raw.strip():
        with suppress(ValueError):
            parsed_profile_id = parse_port_profile_id(
                profile_id_raw,
                field_name="split summary gateway_profile_id",
            )
            if isinstance(parsed_profile_id, int):
                return f"profile-{parsed_profile_id}/{run_id}"
    return None


def load_split_unranked_trail_names(split_payload_path: Path) -> set[str]:
    summary_path = resolve_split_summary_file_path(split_payload_path)
    if not summary_path.is_file():
        return set()

    with suppress(Exception):
        payload = read_json(summary_path)
        if not isinstance(payload, dict):
            return set()
        unranked_payload = payload.get("unranked_trails")
        if not isinstance(unranked_payload, list):
            return set()

        parsed_names: set[str] = set()
        for item in unranked_payload:
            if not isinstance(item, dict):
                continue
            parsed_name = _build_split_summary_trail_name(item)
            if parsed_name:
                parsed_names.add(parsed_name)
        return parsed_names

    return set()


def _parse_group_trail_names(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    parsed: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string")
        name = item.strip()
        if not name:
            continue
        parsed.append(name)
    return parsed


def load_split_two_group_trail_names(
    split_payload_path: Path,
) -> tuple[set[str], set[str], dict[str, Any]]:
    require_file(split_payload_path)
    payload = read_json(split_payload_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid split payload (expected object): {split_payload_path}")

    group_top = payload.get("group_top")
    group_rest = payload.get("group_rest")
    if not isinstance(group_top, dict) or not isinstance(group_rest, dict):
        raise ValueError(
            "Split payload must contain group_top and group_rest objects: "
            f"{split_payload_path}"
        )

    top_names = set(
        _parse_group_trail_names(
            group_top.get("trail_names"),
            field_name=f"{split_payload_path}.group_top.trail_names",
        )
    )
    rest_names = set(
        _parse_group_trail_names(
            group_rest.get("trail_names"),
            field_name=f"{split_payload_path}.group_rest.trail_names",
        )
    )
    overlap = sorted(top_names.intersection(rest_names))
    if overlap:
        raise ValueError(
            "Split payload has overlapping trail names between top/rest groups: "
            f"{overlap}"
        )

    return top_names, rest_names, payload


def with_plan_name_suffix(plan_path: Path, suffix: str) -> Path:
    if not suffix:
        raise ValueError("Plan suffix cannot be empty")
    file_name = plan_path.name
    dot_index = file_name.rfind(".")
    if dot_index <= 0:
        suffixed_name = f"{file_name}.{suffix}"
    else:
        suffixed_name = f"{file_name[:dot_index]}.{suffix}{file_name[dot_index:]}"
    return (plan_path.parent / suffixed_name).resolve()


def apply_additional_suffix(plan_path: Path, additional_suffix: str | None) -> Path:
    """Apply an additional suffix before the final file extension.

    Examples:
        replay-plan.json + "v2" -> replay-plan.v2.json
        replay-plan.exclude-unranked.json + "v2" -> replay-plan.exclude-unranked.v2.json
    """
    if not additional_suffix:
        return plan_path
    file_name = plan_path.name
    dot_index = file_name.rfind(".")
    if dot_index <= 0:
        new_name = f"{file_name}.{additional_suffix}"
    else:
        new_name = f"{file_name[:dot_index]}.{additional_suffix}{file_name[dot_index:]}"
    return (plan_path.parent / new_name).resolve()


def split_two_group_plan_paths(
    *,
    plan_path: Path,
    metric: str,
) -> tuple[Path, Path, Path]:
    metric_alias = SPLIT_TWO_GROUP_PLAN_METRIC_ALIASES.get(metric)
    if metric_alias is None:
        raise ValueError(
            f"Unsupported split two-group metric {metric!r}; "
            f"supported metrics: {sorted(SPLIT_TWO_GROUP_METRICS)}"
        )
    top_path = with_plan_name_suffix(plan_path, f"{metric_alias}.top")
    rest_path = with_plan_name_suffix(plan_path, f"{metric_alias}.rest")
    exclude_unranked_path = with_plan_name_suffix(
        plan_path,
        f"{metric_alias}.exclude-unranked",
    )
    return top_path, rest_path, exclude_unranked_path


def split_two_group_clean_report_paths(
    *,
    plan_path: Path,
    metric: str,
) -> tuple[Path, Path]:
    metric_alias = SPLIT_TWO_GROUP_PLAN_METRIC_ALIASES.get(metric)
    if metric_alias is None:
        raise ValueError(
            f"Unsupported split two-group metric {metric!r}; "
            f"supported metrics: {sorted(SPLIT_TWO_GROUP_METRICS)}"
        )
    stats_path = with_plan_name_suffix(plan_path, f"{metric_alias}.removal-stats")
    details_path = with_plan_name_suffix(plan_path, f"{metric_alias}.removal-details")
    return stats_path, details_path


def clone_workers_with_reindexed_launch_priority(
    workers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cloned = copy.deepcopy(workers)
    for launch_priority, worker in enumerate(cloned):
        worker["launch_priority"] = launch_priority
    return cloned


def split_plan_matches_requested_metric(
    *,
    plan_payload: dict[str, Any],
    metric: str,
    group: str,
) -> bool:
    split_payload = plan_payload.get("split_two_group")
    if not isinstance(split_payload, dict):
        return False
    payload_metric = split_payload.get("metric")
    payload_group = split_payload.get("group")
    return payload_metric == metric and payload_group == group


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
    return status_code in {None, CLEANABLE_REQUEST_STATUS_CODE}


def extract_record_status_code(record: dict[str, Any]) -> int | None:
    status_code = record.get("status_code")
    if isinstance(status_code, bool):
        return None
    if isinstance(status_code, int):
        return status_code
    if isinstance(status_code, str) and status_code.strip():
        with suppress(ValueError):
            return int(status_code.strip())
    return None


def is_cleanable_499_record(record: dict[str, Any]) -> bool:
    return extract_record_status_code(record) == CLEANABLE_REQUEST_STATUS_CODE


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
    model_override: str | None = None,
    tokenize_endpoint: str,
    request_timeout_s: float,
    delta_agent_action_after_s: float,
) -> dict[str, Any]:
    request_body = record.get("request")
    if not isinstance(request_body, dict):
        raise ValueError(f"Request body must be object at index={index}")
    request_body = copy.deepcopy(request_body)

    model_for_tokenize: str | None = None
    if model_override is not None:
        model_for_tokenize = model_override
        request_body["model"] = model_override
    else:
        record_model = record.get("model")
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


def _parse_trailing_json_object(text: str) -> dict[str, Any] | None:
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index].lstrip()
        if not line.startswith("{"):
            continue
        candidate = "\n".join(lines[index:]).strip()
        if not candidate:
            continue
        with suppress(Exception):
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
    return None


def _run_compile_for_default_split_two_group_metrics(args: argparse.Namespace) -> int:
    metric_results: dict[str, dict[str, Any]] = {}
    failed_metrics: list[str] = []
    total_metrics = len(DEFAULT_SPLIT_TWO_GROUP_METRICS)

    for metric_index, metric in enumerate(DEFAULT_SPLIT_TWO_GROUP_METRICS, start=1):
        print(
            f"[compile] split-two-group metric {metric_index}/{total_metrics}: {metric}",
            file=sys.stderr,
        )
        metric_args = argparse.Namespace(**vars(args))
        setattr(metric_args, "split_two_group_metric", metric)

        captured_stdout = io.StringIO()
        tee_stdout = _TeeTextIO(sys.stdout, captured_stdout)
        try:
            with redirect_stdout(tee_stdout):
                exit_code = cmd_compile(metric_args)
        except Exception as exc:  # noqa: BLE001
            metric_results[metric] = {
                "status": "failed",
                "exit_code": 2,
                "error": str(exc),
            }
            failed_metrics.append(metric)
            continue

        summary_payload: dict[str, Any] | None = None
        summary_stdout = captured_stdout.getvalue().strip()
        if summary_stdout:
            summary_payload = _parse_trailing_json_object(summary_stdout)
        metric_result: dict[str, Any] = {
            "status": "ok" if exit_code == 0 else "failed",
            "exit_code": exit_code,
        }
        if summary_payload is not None:
            metric_result["summary"] = summary_payload
        elif summary_stdout:
            metric_result["stdout"] = summary_stdout
        metric_results[metric] = metric_result
        if exit_code != 0:
            failed_metrics.append(metric)

    status = "ok" if not failed_metrics else "failed"
    output_payload = {
        "status": status,
        "split_two_group_plans": True,
        "split_two_group_metrics": list(DEFAULT_SPLIT_TWO_GROUP_METRICS),
        "metric_results": metric_results,
    }
    if failed_metrics:
        output_payload["failed_metrics"] = failed_metrics
    print(json.dumps(output_payload, indent=2, ensure_ascii=True))
    return 0 if status == "ok" else 2


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
    job_root_value = parse_optional_path(
        (
            args.job_root
            if getattr(args, "job_root", None) is not None
            else compile_config.get("job_root")
        ),
        field_name="job_root",
    )

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
    compile_model_override_value = parse_optional_str(
        (
            args.model
            if getattr(args, "model", None) is not None
            else compile_config.get("model")
        ),
        field_name="model",
    )
    compile_model_override = (
        compile_model_override_value.strip()
        if compile_model_override_value is not None
        else None
    )
    single_trail_value = parse_optional_str(
        (
            args.single_trail
            if getattr(args, "single_trail", None) is not None
            else compile_config.get("single_trail")
        ),
        field_name="single_trail",
    )
    single_trail = single_trail_value.strip() if single_trail_value is not None else None
    compile_model_override_key: str | None = None
    compile_model_override_resolved: str | None = None
    if compile_model_override is not None:
        (
            compile_model_override_key,
            compile_model_override_resolved,
        ) = resolve_configured_compile_model_override(compile_model_override)
    split_two_group_plans = parse_optional_bool(
        (
            args.split_two_group_plans
            if getattr(args, "split_two_group_plans", None) is not None
            else compile_config.get("split_two_group_plans")
        ),
        field_name="split_two_group_plans",
    )
    split_two_group_plans = bool(split_two_group_plans) if split_two_group_plans is not None else False
    split_two_group_metric_value = parse_optional_str(
        (
            args.split_two_group_metric
            if getattr(args, "split_two_group_metric", None) is not None
            else compile_config.get("split_two_group_metric")
        ),
        field_name="split_two_group_metric",
    )
    split_two_group_metric_explicit = split_two_group_metric_value is not None
    split_two_group_metric = (
        split_two_group_metric_value.strip().lower()
        if split_two_group_metric_value is not None
        else DEFAULT_SPLIT_TWO_GROUP_METRIC
    )
    if split_two_group_metric not in SPLIT_TWO_GROUP_METRICS:
        supported_values = ", ".join(sorted(SPLIT_TWO_GROUP_METRICS))
        raise ValueError(
            "split_two_group_metric must be one of: "
            f"{supported_values}. Got {split_two_group_metric!r}"
        )
    additional_suffix_value = parse_optional_str(
        (
            args.additional_suffix
            if getattr(args, "additional_suffix", None) is not None
            else compile_config.get("additional_suffix")
        ),
        field_name="additional_suffix",
    )
    additional_suffix = (
        additional_suffix_value.strip()
        if additional_suffix_value is not None
        else None
    )
    exclude_unranked_trails_value = parse_optional_bool(
        (
            args.exclude_unranked_trails
            if getattr(args, "exclude_unranked_trails", None) is not None
            else compile_config.get("exclude_unranked_trails")
        ),
        field_name="exclude_unranked_trails",
    )
    exclude_unranked_trails = (
        bool(exclude_unranked_trails_value)
        if exclude_unranked_trails_value is not None
        else False
    )
    compile_clean_value = parse_optional_bool(
        (
            args.clean
            if getattr(args, "clean", None) is not None
            else compile_config.get("clean")
        ),
        field_name="clean",
    )
    compile_clean = bool(compile_clean_value) if compile_clean_value is not None else False
    if exclude_unranked_trails and split_two_group_plans:
        raise ValueError(
            "--exclude-unranked-trails cannot be combined with "
            "--split-two-group-plans"
        )
    if single_trail is not None and split_two_group_plans:
        raise ValueError("--single-trail cannot be combined with --split-two-group-plans")
    if compile_clean and not split_two_group_plans:
        raise ValueError("--clean can only be combined with --split-two-group-plans")
    if split_two_group_plans and not split_two_group_metric_explicit:
        return _run_compile_for_default_split_two_group_metrics(args)
    plan_out_override = parse_optional_path(
        (
            args.plan_out
            if getattr(args, "plan_out", None) is not None
            else compile_config.get("plan_out")
        ),
        field_name="plan_out",
    )

    if job_root_value is not None:
        if job_dir_value is not None:
            raise ValueError("--job-root cannot be combined with --job-dir")
        if plan_out_override is not None:
            raise ValueError("--job-root cannot be combined with --plan-out")
        job_root = job_root_value.expanduser().resolve()
        if not job_root.exists() or not job_root.is_dir():
            raise ValueError(f"Invalid --job-root: {job_root}")
        discovered_job_dirs = discover_profiled_job_dirs(job_root)
        if not discovered_job_dirs:
            raise ValueError(
                "No profiled job directories found under --job-root: "
                f"{job_root}"
            )

        max_workers = _job_root_default_max_workers(len(discovered_job_dirs))
        job_summaries: list[dict[str, Any]] = []
        failed_count = 0
        succeeded_count = 0
        show_plain_progress = RichProgress is None
        with create_batch_compile_progress() as progress:
            task_id = progress.add_task(
                "compiling jobs under root",
                total=len(discovered_job_dirs),
                succeeded=succeeded_count,
                failed=failed_count,
            )
            future_to_job_dir: dict[Future[dict[str, Any]], Path] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for discovered_job_dir in discovered_job_dirs:
                    future = executor.submit(
                        _run_job_root_child_compile,
                        job_dir=discovered_job_dir,
                        port_profile_id=compile_port_profile_id,
                        request_timeout_s=request_timeout_s_override,
                        model_override=compile_model_override,
                        single_trail=single_trail,
                        split_two_group_plans=split_two_group_plans,
                        split_two_group_metric=split_two_group_metric,
                        exclude_unranked_trails=exclude_unranked_trails,
                        clean=compile_clean,
                    )
                    future_to_job_dir[future] = discovered_job_dir

                for future in as_completed(future_to_job_dir):
                    discovered_job_dir = future_to_job_dir[future]
                    try:
                        job_summary = future.result()
                    except Exception as exc:  # noqa: BLE001
                        job_summary = {
                            "job_dir": str(discovered_job_dir),
                            "status": "failed",
                            "error": str(exc),
                        }
                    if job_summary.get("status") == "ok":
                        succeeded_count += 1
                    else:
                        failed_count += 1
                    job_summaries.append(job_summary)
                    progress.update(
                        task_id,
                        advance=1,
                        succeeded=succeeded_count,
                        failed=failed_count,
                    )
                    if show_plain_progress:
                        completed = succeeded_count + failed_count
                        print(
                            (
                                f"[compile {completed}/{len(discovered_job_dirs)}] "
                                f"ok={succeeded_count} failed={failed_count} "
                                f"job={job_summary.get('job_dir')}"
                            ),
                            file=sys.stderr,
                        )

        job_summaries.sort(key=lambda item: str(item.get("job_dir", "")))

        summary: dict[str, Any] = {
            "status": "ok" if failed_count == 0 else "failed",
            "mode": "job_root",
            "job_root": str(job_root),
            "job_count_total": len(discovered_job_dirs),
            "job_count_succeeded": succeeded_count,
            "job_count_failed": failed_count,
            "parallel_max_workers": max_workers,
            "port_profile_id": compile_port_profile_id,
            "model_override": compile_model_override,
            "model_override_key": compile_model_override_key,
            "model_override_resolved": compile_model_override_resolved,
            "single_trail": single_trail,
            "split_two_group_plans": split_two_group_plans,
            "exclude_unranked_trails": exclude_unranked_trails,
            "clean": compile_clean,
            "jobs": job_summaries,
        }
        if split_two_group_plans:
            summary["split_two_group_metric"] = split_two_group_metric
        if config_file_path is not None:
            summary["source_config"] = str(config_file_path.expanduser().resolve())
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0 if failed_count == 0 else 2

    job_dir = resolve_required_option(
        job_dir_value,
        option_name="--job-dir",
        config_key="job_dir",
    ).expanduser().resolve()
    if not job_dir.exists() or not job_dir.is_dir():
        raise ValueError(f"Invalid --job-dir: {job_dir}")

    if plan_out_override is not None:
        plan_path = plan_out_override.expanduser().resolve()
    else:
        default_plan_path = (job_dir / "replay-plan.json").resolve()
        if exclude_unranked_trails:
            plan_path = with_plan_name_suffix(
                default_plan_path,
                "exclude-unranked",
            )
        else:
            plan_path = default_plan_path
        if single_trail is not None:
            plan_path = with_plan_name_suffix(
                plan_path,
                f"trail-{safe_name(single_trail)}",
            )

    if compile_clean:
        plan_path = with_plan_name_suffix(plan_path, COMPILE_CLEAN_PLAN_SUFFIX)

    (
        split_two_group_top_path,
        split_two_group_rest_path,
        split_two_group_exclude_unranked_path,
    ) = split_two_group_plan_paths(
        plan_path=plan_path,
        metric=split_two_group_metric,
    )
    split_two_group_clean_stats_path: Path | None = None
    split_two_group_clean_details_path: Path | None = None
    if split_two_group_plans and compile_clean:
        (
            split_two_group_clean_stats_path,
            split_two_group_clean_details_path,
        ) = split_two_group_clean_report_paths(
            plan_path=plan_path,
            metric=split_two_group_metric,
        )

    # Apply additional suffix if specified
    plan_path = apply_additional_suffix(plan_path, additional_suffix)
    split_two_group_top_path = apply_additional_suffix(
        split_two_group_top_path, additional_suffix
    )
    split_two_group_rest_path = apply_additional_suffix(
        split_two_group_rest_path, additional_suffix
    )
    split_two_group_exclude_unranked_path = apply_additional_suffix(
        split_two_group_exclude_unranked_path, additional_suffix
    )
    if split_two_group_clean_stats_path is not None:
        split_two_group_clean_stats_path = apply_additional_suffix(
            split_two_group_clean_stats_path, additional_suffix
        )
    if split_two_group_clean_details_path is not None:
        split_two_group_clean_details_path = apply_additional_suffix(
            split_two_group_clean_details_path, additional_suffix
        )

    existing_plan_payload: dict[str, Any] | None = None
    if not split_two_group_plans:
        if plan_path.exists():
            if not plan_path.is_file():
                raise ValueError(f"Invalid replay plan path (not a file): {plan_path}")
            with suppress(Exception):
                loaded_payload = read_json(plan_path)
                if isinstance(loaded_payload, dict):
                    existing_plan_payload = loaded_payload
        existing_exclude_unranked = False
        existing_model_override: str | None = None
        existing_clean = False
        existing_single_trail: str | None = None
        if existing_plan_payload is not None:
            existing_compile_options = existing_plan_payload.get("compile_options")
            if isinstance(existing_compile_options, dict):
                existing_exclude_unranked = bool(
                    existing_compile_options.get("exclude_unranked_trails")
                )
            existing_model_override = extract_plan_compile_model_override(
                existing_plan_payload
            )
            existing_clean = extract_plan_compile_clean(existing_plan_payload)
            existing_single_trail = extract_plan_compile_single_trail(
                existing_plan_payload
            )
        if (
            existing_plan_payload is not None
            and is_replay_plan_compile_version_current(existing_plan_payload)
            and existing_exclude_unranked == exclude_unranked_trails
            and existing_clean == compile_clean
            and existing_single_trail == single_trail
            and existing_model_override == compile_model_override_resolved
            and (
                compile_model_override_resolved is None
                or extract_plan_replay_model(existing_plan_payload)
                == compile_model_override_resolved
            )
        ):
            launch_policy_payload = existing_plan_payload.get("launch_policy")
            launch_strategy = (
                launch_policy_payload.get("strategy")
                if isinstance(launch_policy_payload, dict)
                else None
            )
            worker_count, request_count = count_plan_workers_and_requests(existing_plan_payload)
            summary = {
                "status": "ok",
                "backend": existing_plan_payload.get("backend"),
                "plan_path": str(plan_path),
                "launch_strategy": launch_strategy,
                "worker_count": worker_count,
                "request_count": request_count,
                "port_profile_id": compile_port_profile_id,
                "compile_version": REPLAY_PLAN_COMPILE_VERSION,
                "reused_existing_plan": True,
                "exclude_unranked_trails": exclude_unranked_trails,
                "clean": compile_clean,
            }
            if single_trail is not None:
                summary["single_trail"] = single_trail
            if compile_model_override is not None:
                summary["model_override"] = compile_model_override
                summary["model_override_key"] = compile_model_override_key
                summary["model_override_resolved"] = compile_model_override_resolved
            if config_file_path is not None:
                summary["source_config"] = str(config_file_path.expanduser().resolve())
            print(json.dumps(summary, indent=2, ensure_ascii=True))
            return 0
    else:
        existing_top_plan: dict[str, Any] | None = None
        existing_rest_plan: dict[str, Any] | None = None
        existing_exclude_unranked_plan: dict[str, Any] | None = None
        if split_two_group_top_path.exists():
            if not split_two_group_top_path.is_file():
                raise ValueError(
                    "Invalid replay plan path (not a file): "
                    f"{split_two_group_top_path}"
                )
            with suppress(Exception):
                loaded_top_payload = read_json(split_two_group_top_path)
                if isinstance(loaded_top_payload, dict):
                    existing_top_plan = loaded_top_payload
        if split_two_group_rest_path.exists():
            if not split_two_group_rest_path.is_file():
                raise ValueError(
                    "Invalid replay plan path (not a file): "
                    f"{split_two_group_rest_path}"
                )
            with suppress(Exception):
                loaded_rest_payload = read_json(split_two_group_rest_path)
                if isinstance(loaded_rest_payload, dict):
                    existing_rest_plan = loaded_rest_payload
        if split_two_group_exclude_unranked_path.exists():
            if not split_two_group_exclude_unranked_path.is_file():
                raise ValueError(
                    "Invalid replay plan path (not a file): "
                    f"{split_two_group_exclude_unranked_path}"
                )
            with suppress(Exception):
                loaded_exclude_unranked_payload = read_json(
                    split_two_group_exclude_unranked_path
                )
                if isinstance(loaded_exclude_unranked_payload, dict):
                    existing_exclude_unranked_plan = loaded_exclude_unranked_payload
        existing_clean_stats_payload: dict[str, Any] | None = None
        clean_report_files_ready = True
        if compile_clean:
            clean_report_files_ready = bool(
                split_two_group_clean_stats_path is not None
                and split_two_group_clean_details_path is not None
                and split_two_group_clean_stats_path.is_file()
                and split_two_group_clean_details_path.is_file()
            )
            if (
                split_two_group_clean_stats_path is not None
                and split_two_group_clean_stats_path.is_file()
            ):
                with suppress(Exception):
                    loaded_clean_stats_payload = read_json(split_two_group_clean_stats_path)
                    if isinstance(loaded_clean_stats_payload, dict):
                        existing_clean_stats_payload = loaded_clean_stats_payload

        if (
            existing_top_plan is not None
            and existing_rest_plan is not None
            and existing_exclude_unranked_plan is not None
            and clean_report_files_ready
            and is_replay_plan_compile_version_current(existing_top_plan)
            and is_replay_plan_compile_version_current(existing_rest_plan)
            and is_replay_plan_compile_version_current(existing_exclude_unranked_plan)
            and extract_plan_compile_clean(existing_top_plan) == compile_clean
            and extract_plan_compile_clean(existing_rest_plan) == compile_clean
            and extract_plan_compile_clean(existing_exclude_unranked_plan)
            == compile_clean
            and extract_plan_compile_model_override(existing_top_plan)
            == compile_model_override_resolved
            and extract_plan_compile_model_override(existing_rest_plan)
            == compile_model_override_resolved
            and extract_plan_compile_model_override(existing_exclude_unranked_plan)
            == compile_model_override_resolved
            and (
                compile_model_override_resolved is None
                or (
                    extract_plan_replay_model(existing_top_plan)
                    == compile_model_override_resolved
                    and extract_plan_replay_model(existing_rest_plan)
                    == compile_model_override_resolved
                    and extract_plan_replay_model(existing_exclude_unranked_plan)
                    == compile_model_override_resolved
                )
            )
            and split_plan_matches_requested_metric(
                plan_payload=existing_top_plan,
                metric=split_two_group_metric,
                group="top",
            )
            and split_plan_matches_requested_metric(
                plan_payload=existing_rest_plan,
                metric=split_two_group_metric,
                group="rest",
            )
            and split_plan_matches_requested_metric(
                plan_payload=existing_exclude_unranked_plan,
                metric=split_two_group_metric,
                group="exclude-unranked",
            )
        ):
            top_launch_policy = existing_top_plan.get("launch_policy")
            top_launch_strategy = (
                top_launch_policy.get("strategy")
                if isinstance(top_launch_policy, dict)
                else None
            )
            top_worker_count, top_request_count = count_plan_workers_and_requests(
                existing_top_plan
            )
            rest_worker_count, rest_request_count = count_plan_workers_and_requests(
                existing_rest_plan
            )
            (
                exclude_unranked_worker_count,
                exclude_unranked_request_count,
            ) = count_plan_workers_and_requests(existing_exclude_unranked_plan)
            summary = {
                "status": "ok",
                "backend": existing_top_plan.get("backend"),
                "plan_paths": {
                    "top": str(split_two_group_top_path),
                    "rest": str(split_two_group_rest_path),
                    "exclude_unranked": str(split_two_group_exclude_unranked_path),
                },
                "launch_strategy": top_launch_strategy,
                "worker_count_top": top_worker_count,
                "request_count_top": top_request_count,
                "worker_count_rest": rest_worker_count,
                "request_count_rest": rest_request_count,
                "worker_count_exclude_unranked": exclude_unranked_worker_count,
                "request_count_exclude_unranked": exclude_unranked_request_count,
                "port_profile_id": compile_port_profile_id,
                "compile_version": REPLAY_PLAN_COMPILE_VERSION,
                "reused_existing_plan": True,
                "split_two_group_plans": True,
                "split_two_group_metric": split_two_group_metric,
                "clean": compile_clean,
            }
            if compile_clean:
                summary["clean_removal_stats_path"] = (
                    str(split_two_group_clean_stats_path)
                    if split_two_group_clean_stats_path is not None
                    else None
                )
                summary["clean_removal_details_path"] = (
                    str(split_two_group_clean_details_path)
                    if split_two_group_clean_details_path is not None
                    else None
                )
                if isinstance(existing_clean_stats_payload, dict):
                    summary["clean_removed_499_request_count"] = _to_int_or_default(
                        existing_clean_stats_payload.get("removed_request_count"),
                        default=0,
                    )
            if compile_model_override is not None:
                summary["model_override"] = compile_model_override
                summary["model_override_key"] = compile_model_override_key
                summary["model_override_resolved"] = compile_model_override_resolved
            if config_file_path is not None:
                summary["source_config"] = str(config_file_path.expanduser().resolve())
            print(json.dumps(summary, indent=2, ensure_ascii=True))
            return 0

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

    source_configured_model, tokenize_endpoint = resolve_compile_target(
        config=config,
        results_entries=results_entries,
        port_profile_id=compile_port_profile_id,
    )
    configured_model = compile_model_override_resolved or source_configured_model
    request_timeout_s = (
        request_timeout_s_override
        if request_timeout_s_override is not None
        else DEFAULT_COMPILE_TOKENIZE_TIMEOUT_S
    )
    launch_policy = build_launch_policy_from_config(config)

    split_two_group_source_path: Path | None = None
    split_two_group_unranked_source_path: Path | None = None
    top_group_trail_names: set[str] = set()
    rest_group_trail_names: set[str] = set()
    unranked_trail_names: set[str] = set()
    split_two_group_source_payload: dict[str, Any] | None = None
    if split_two_group_plans:
        split_two_group_source_path = resolve_split_two_group_file_path(
            job_dir=job_dir,
            metric=split_two_group_metric,
        )
        (
            top_group_trail_names,
            rest_group_trail_names,
            split_two_group_source_payload,
        ) = load_split_two_group_trail_names(split_two_group_source_path)
        split_two_group_unranked_source_path = resolve_split_summary_file_path(
            split_two_group_source_path
        )
        unranked_trail_names = load_split_unranked_trail_names(split_two_group_source_path)
        if not top_group_trail_names:
            raise ValueError(
                "Split two-group payload has no top-group trail names: "
                f"{split_two_group_source_path}"
            )
        if not rest_group_trail_names:
            raise ValueError(
                "Split two-group payload has no rest-group trail names: "
                f"{split_two_group_source_path}"
            )

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

    run_infos: list[tuple[Path, dict[str, Any], int, int | None, str]] = []
    total_requests = 0
    exclude_unranked_source_path: Path | None = None
    excluded_unranked_trails: list[str] = []
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
        source_trail_name = build_gateway_trail_name(
            run_dir=run_dir,
            source_port_profile_id=source_port_profile_id,
        )
        run_infos.append(
            (
                run_dir,
                run_manifest_payload,
                request_count,
                source_port_profile_id,
                source_trail_name,
            )
        )
        total_requests += request_count

    if exclude_unranked_trails:
        split_payload_probe_path = resolve_split_two_group_file_path(
            job_dir=job_dir,
            metric=DEFAULT_SPLIT_TWO_GROUP_METRIC,
        )
        exclude_unranked_source_path = resolve_split_summary_file_path(
            split_payload_probe_path
        )
        if not exclude_unranked_source_path.is_file():
            raise ValueError(
                "--exclude-unranked-trails requires split summary file: "
                f"{exclude_unranked_source_path}"
            )
        compile_unranked_trail_names = load_split_unranked_trail_names(
            split_payload_probe_path
        )
        filtered_run_infos: list[tuple[Path, dict[str, Any], int, int | None, str]] = []
        for run_info in run_infos:
            source_trail_name = run_info[4]
            if source_trail_name in compile_unranked_trail_names:
                excluded_unranked_trails.append(source_trail_name)
                continue
            filtered_run_infos.append(run_info)
        run_infos = filtered_run_infos
        excluded_unranked_trails = sorted(set(excluded_unranked_trails))
        total_requests = sum(run_info[2] for run_info in run_infos)
        if excluded_unranked_trails:
            print(
                "[compile] note: excluding trails listed in unranked_trails "
                f"({len(excluded_unranked_trails)}): {exclude_unranked_source_path}",
                file=sys.stderr,
            )

    if single_trail is not None:
        selected_run_infos = [
            run_info for run_info in run_infos if run_info[4] == single_trail
        ]
        if not selected_run_infos:
            if single_trail in excluded_unranked_trails:
                raise ValueError(
                    f"--single-trail {single_trail!r} was excluded by "
                    "--exclude-unranked-trails"
                )
            available_trails = sorted({run_info[4] for run_info in run_infos})
            available_preview = ", ".join(available_trails[:20]) if available_trails else "<none>"
            if len(available_trails) > 20:
                available_preview += f", ... ({len(available_trails)} total)"
            raise ValueError(
                f"--single-trail {single_trail!r} did not match any discovered trail. "
                f"Available trails: {available_preview}"
            )
        run_infos = selected_run_infos
        total_requests = sum(run_info[2] for run_info in run_infos)

    worker_plans: list[dict[str, Any]] = []
    cleaned_499_request_count = 0
    cleaned_499_count_by_trail: dict[str, int] = {}
    cleaned_499_count_by_worker: dict[str, int] = {}
    cleaned_499_request_details: list[dict[str, Any]] = []
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

        for (
            run_dir,
            run_manifest_payload,
            expected_request_count,
            source_port_profile_id,
            source_trail_name,
        ) in run_infos:
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

            request_keep_mask = [True] * len(request_records)
            if compile_clean and request_records:
                request_keep_mask = [
                    not is_cleanable_499_record(record) for record in request_records
                ]
                removed_count = len(request_records) - sum(request_keep_mask)
                if removed_count > 0:
                    cleaned_499_request_count += removed_count
                    _update_compile_progress(request_total_delta=-removed_count)
                    cleaned_499_count_by_trail[source_trail_name] = (
                        cleaned_499_count_by_trail.get(source_trail_name, 0) + removed_count
                    )
                    cleaned_499_count_by_worker[trial_id] = (
                        cleaned_499_count_by_worker.get(trial_id, 0) + removed_count
                    )
                    for index, should_keep in enumerate(request_keep_mask):
                        if should_keep:
                            continue
                        removed_record = request_records[index]
                        response_payload = removed_record.get("response")
                        response_error = (
                            response_payload.get("error")
                            if isinstance(response_payload, dict)
                            else None
                        )
                        removed_duration_s: float | None = None
                        with suppress(Exception):
                            removed_duration_s = request_duration_seconds(removed_record)
                        cleaned_499_request_details.append(
                            {
                                "source_trail_name": source_trail_name,
                                "worker_id": trial_id,
                                "source_gateway_run_id": run_dir.name,
                                "source_gateway_profile_id": source_port_profile_id,
                                "request_index": index,
                                "request_id": removed_record.get("request_id"),
                                "status_code": extract_record_status_code(removed_record),
                                "response_error": response_error,
                                "request_start_time": (
                                    removed_record.get("request_start_time")
                                    or removed_record.get("span_start_time")
                                ),
                                "request_end_time": (
                                    removed_record.get("request_end_time")
                                    or removed_record.get("span_end_time")
                                ),
                                "request_duration_s": removed_duration_s,
                            }
                        )

            planned_requests: list[dict[str, Any]] = []
            for index, record in enumerate(request_records):
                if not request_keep_mask[index]:
                    continue
                planned_requests.append(
                    build_planned_request(
                        record=record,
                        index=index,
                        configured_model=configured_model,
                        model_override=compile_model_override_resolved,
                        tokenize_endpoint=tokenize_endpoint,
                        request_timeout_s=request_timeout_s,
                        delta_agent_action_after_s=delta_agent_action_after[index],
                    )
                )
                _update_compile_progress(advance=1)

            if compile_clean and not planned_requests:
                delta_first_request_s = 0.0

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
                    "source_gateway_run_id": run_dir.name,
                    "source_gateway_profile_id": source_port_profile_id,
                    "source_trail_name": source_trail_name,
                    "requests": planned_requests,
                }
            )
            compile_progress_state["workers_completed"] += 1
            _update_compile_progress()

    if compile_clean and cleaned_499_request_count > 0:
        print(
            "[compile] clean mode: removed "
            f"{cleaned_499_request_count} request(s) with status_code=499",
            file=sys.stderr,
        )

    worker_plans.sort(key=lambda item: (float(item["run_offset_s"]), str(item["worker_id"])))
    for launch_priority, worker in enumerate(worker_plans):
        worker["launch_priority"] = launch_priority

    compiled_at = now_iso8601_utc()

    def _build_plan_payload_for_workers(
        workers_payload: list[dict[str, Any]],
        *,
        split_group: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": REPLAY_PLAN_SCHEMA_VERSION,
            "compile_version": REPLAY_PLAN_COMPILE_VERSION,
            "compiled_at": compiled_at,
            "source_job_dir": str(job_dir),
            "backend": backend_name,
            "t0": t0_iso,
            "t0_source": t0_source,
            "replay_target": {
                "model": configured_model,
                "deterministic_required": True,
            },
            "launch_policy": launch_policy,
            "workers": workers_payload,
            "compile_options": {
                "exclude_unranked_trails": exclude_unranked_trails,
                "clean": compile_clean,
            },
        }
        if single_trail is not None:
            payload["compile_options"]["single_trail"] = single_trail
        if compile_clean:
            payload["compile_options"]["clean_removed_499_request_count"] = (
                cleaned_499_request_count
            )
        if compile_model_override_resolved is not None:
            payload["compile_options"]["model_override"] = compile_model_override_resolved
            payload["compile_options"]["model_override_key"] = compile_model_override_key
        if exclude_unranked_trails:
            payload["compile_options"]["exclude_unranked_source_path"] = (
                str(exclude_unranked_source_path)
                if exclude_unranked_source_path is not None
                else None
            )
            payload["compile_options"]["excluded_unranked_trails_count"] = len(
                excluded_unranked_trails
            )
        if split_two_group_plans:
            payload["split_two_group"] = {
                "enabled": True,
                "metric": split_two_group_metric,
                "source_path": (
                    str(split_two_group_source_path)
                    if split_two_group_source_path is not None
                    else None
                ),
                "source_selected_p": (
                    split_two_group_source_payload.get("selection", {}).get("selected_p")
                    if isinstance(split_two_group_source_payload, dict)
                    else None
                ),
                "group": split_group,
            }
        return payload

    if not split_two_group_plans:
        plan_payload = _build_plan_payload_for_workers(worker_plans)
        write_json(plan_path, plan_payload)

        summary = {
            "status": "ok",
            "backend": backend_name,
            "plan_path": str(plan_path),
            "launch_strategy": launch_policy.get("strategy"),
            "worker_count": len(worker_plans),
            "request_count": sum(len(worker["requests"]) for worker in worker_plans),
            "port_profile_id": compile_port_profile_id,
            "compile_version": REPLAY_PLAN_COMPILE_VERSION,
            "reused_existing_plan": False,
            "exclude_unranked_trails": exclude_unranked_trails,
            "clean": compile_clean,
            "model": configured_model,
        }
        if single_trail is not None:
            summary["single_trail"] = single_trail
        if compile_clean:
            summary["clean_removed_499_request_count"] = cleaned_499_request_count
        if compile_model_override is not None:
            summary["model_override"] = compile_model_override
            summary["model_override_key"] = compile_model_override_key
            summary["model_override_resolved"] = compile_model_override_resolved
            summary["source_model"] = source_configured_model
        if exclude_unranked_trails:
            summary["exclude_unranked_source_path"] = (
                str(exclude_unranked_source_path)
                if exclude_unranked_source_path is not None
                else None
            )
            summary["excluded_unranked_trails_count"] = len(excluded_unranked_trails)
            summary["excluded_unranked_trails"] = excluded_unranked_trails
        if config_file_path is not None:
            summary["source_config"] = str(config_file_path.expanduser().resolve())
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0

    all_compiled_trail_names = {
        str(worker.get("source_trail_name") or "")
        for worker in worker_plans
        if isinstance(worker.get("source_trail_name"), str)
    }
    unmatched_trails = sorted(
        trail_name
        for trail_name in all_compiled_trail_names
        if trail_name not in top_group_trail_names and trail_name not in rest_group_trail_names
    )
    ignored_unranked_unmatched_trails: list[str] = []
    if unmatched_trails:
        unmatched_not_unranked = sorted(
            trail_name
            for trail_name in unmatched_trails
            if trail_name not in unranked_trail_names
        )
        if unmatched_not_unranked:
            raise ValueError(
                "Split two-group payload does not partition all compiled trails. "
                f"Unmatched trails: {unmatched_not_unranked}"
            )
        ignored_unranked_unmatched_trails = list(unmatched_trails)
        note_path = (
            str(split_two_group_unranked_source_path)
            if split_two_group_unranked_source_path is not None
            else "split summary"
        )
        print(
            "[compile] split two-group note: ignoring unmatched trails because they are "
            f"listed in unranked_trails ({len(ignored_unranked_unmatched_trails)}): {note_path}",
            file=sys.stderr,
        )
    missing_from_compiled = sorted(
        trail_name
        for trail_name in top_group_trail_names.union(rest_group_trail_names)
        if trail_name not in all_compiled_trail_names
    )
    if missing_from_compiled:
        raise ValueError(
            "Split two-group payload references trails that are not present in compiled artifacts. "
            f"Missing trails: {missing_from_compiled}"
        )

    top_group_workers = [
        worker for worker in worker_plans if worker.get("source_trail_name") in top_group_trail_names
    ]
    rest_group_workers = [
        worker for worker in worker_plans if worker.get("source_trail_name") in rest_group_trail_names
    ]
    exclude_unranked_group_workers = [
        worker
        for worker in worker_plans
        if worker.get("source_trail_name") in top_group_trail_names
        or worker.get("source_trail_name") in rest_group_trail_names
    ]
    if not top_group_workers:
        raise ValueError("Split two-group top plan would be empty")
    if not rest_group_workers:
        raise ValueError("Split two-group rest plan would be empty")
    if not exclude_unranked_group_workers:
        raise ValueError("Split two-group exclude-unranked plan would be empty")

    top_group_workers = clone_workers_with_reindexed_launch_priority(top_group_workers)
    rest_group_workers = clone_workers_with_reindexed_launch_priority(rest_group_workers)
    exclude_unranked_group_workers = clone_workers_with_reindexed_launch_priority(
        exclude_unranked_group_workers
    )

    top_plan_payload = _build_plan_payload_for_workers(
        top_group_workers,
        split_group="top",
    )
    rest_plan_payload = _build_plan_payload_for_workers(
        rest_group_workers,
        split_group="rest",
    )
    exclude_unranked_plan_payload = _build_plan_payload_for_workers(
        exclude_unranked_group_workers,
        split_group="exclude-unranked",
    )
    write_json(split_two_group_top_path, top_plan_payload)
    write_json(split_two_group_rest_path, rest_plan_payload)
    write_json(split_two_group_exclude_unranked_path, exclude_unranked_plan_payload)

    if compile_clean:
        if (
            split_two_group_clean_stats_path is None
            or split_two_group_clean_details_path is None
        ):
            raise ValueError("Missing clean report output path while --clean is enabled")

        removed_by_trail_rows = [
            {
                "source_trail_name": trail_name,
                "removed_request_count": cleaned_499_count_by_trail[trail_name],
            }
            for trail_name in sorted(cleaned_499_count_by_trail)
        ]
        removed_by_worker_rows = [
            {
                "worker_id": worker_id,
                "removed_request_count": cleaned_499_count_by_worker[worker_id],
            }
            for worker_id in sorted(cleaned_499_count_by_worker)
        ]
        clean_stats_payload = {
            "status": "ok",
            "schema": "clean-499-removal-stats.v1",
            "compiled_at": compiled_at,
            "source_job_dir": str(job_dir),
            "split_two_group_metric": split_two_group_metric,
            "clean_enabled": True,
            "removed_request_count": cleaned_499_request_count,
            "removed_trail_count": len(removed_by_trail_rows),
            "removed_worker_count": len(removed_by_worker_rows),
            "removed_count_by_trail": removed_by_trail_rows,
            "removed_count_by_worker": removed_by_worker_rows,
        }
        clean_details_payload = {
            "status": "ok",
            "schema": "clean-499-removal-details.v1",
            "compiled_at": compiled_at,
            "source_job_dir": str(job_dir),
            "split_two_group_metric": split_two_group_metric,
            "clean_enabled": True,
            "removed_request_count": cleaned_499_request_count,
            "removed_requests": cleaned_499_request_details,
        }
        write_json(split_two_group_clean_stats_path, clean_stats_payload)
        write_json(split_two_group_clean_details_path, clean_details_payload)

    summary = {
        "status": "ok",
        "backend": backend_name,
        "plan_paths": {
            "top": str(split_two_group_top_path),
            "rest": str(split_two_group_rest_path),
            "exclude_unranked": str(split_two_group_exclude_unranked_path),
        },
        "launch_strategy": launch_policy.get("strategy"),
        "worker_count_top": len(top_group_workers),
        "request_count_top": sum(len(worker["requests"]) for worker in top_group_workers),
        "worker_count_rest": len(rest_group_workers),
        "request_count_rest": sum(len(worker["requests"]) for worker in rest_group_workers),
        "worker_count_exclude_unranked": len(exclude_unranked_group_workers),
        "request_count_exclude_unranked": sum(
            len(worker["requests"]) for worker in exclude_unranked_group_workers
        ),
        "port_profile_id": compile_port_profile_id,
        "compile_version": REPLAY_PLAN_COMPILE_VERSION,
        "reused_existing_plan": False,
        "split_two_group_plans": True,
        "split_two_group_metric": split_two_group_metric,
        "clean": compile_clean,
        "split_two_group_source_path": (
            str(split_two_group_source_path) if split_two_group_source_path is not None else None
        ),
        "model": configured_model,
    }
    if compile_clean:
        summary["clean_removed_499_request_count"] = cleaned_499_request_count
        summary["clean_removal_stats_path"] = (
            str(split_two_group_clean_stats_path)
            if split_two_group_clean_stats_path is not None
            else None
        )
        summary["clean_removal_details_path"] = (
            str(split_two_group_clean_details_path)
            if split_two_group_clean_details_path is not None
            else None
        )
    if compile_model_override is not None:
        summary["model_override"] = compile_model_override
        summary["model_override_key"] = compile_model_override_key
        summary["model_override_resolved"] = compile_model_override_resolved
        summary["source_model"] = source_configured_model
    if ignored_unranked_unmatched_trails:
        summary["split_two_group_note"] = (
            "ignored unmatched compiled trails that were listed in split summary "
            "unranked_trails"
        )
        summary["split_two_group_ignored_unranked_trails_count"] = len(
            ignored_unranked_unmatched_trails
        )
        summary["split_two_group_ignored_unranked_trails"] = (
            ignored_unranked_unmatched_trails
        )
        summary["split_two_group_unranked_source_path"] = (
            str(split_two_group_unranked_source_path)
            if split_two_group_unranked_source_path is not None
            else None
        )
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


def _launch_policy_override_selects_uncapped_arrival_pattern(
    launch_policy_override: dict[str, Any] | None,
) -> bool:
    if not isinstance(launch_policy_override, dict):
        return False
    pattern_payload = launch_policy_override.get("pattern")
    if not isinstance(pattern_payload, dict):
        return False
    return _normalize_launch_pattern_name(pattern_payload.get("name")) in {
        "poisson",
        "possion",
        "uniform",
    }


def _launch_capacity_available(
    *,
    active_count: int,
    launch_max_concurrent: int | None,
) -> bool:
    return launch_max_concurrent is None or active_count < launch_max_concurrent


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


def _extract_launch_pattern_rate_per_second(
    pattern_payload: dict[str, Any],
    pattern_args_payload: Any,
    *,
    pattern_label: str,
) -> float | None:
    direct_rate = _coerce_pattern_arg_float(pattern_payload.get("rate_per_second"))
    if direct_rate is not None:
        if direct_rate <= 0:
            raise ValueError(
                f"{pattern_label} replay launch pattern rate_per_second must be > 0"
            )
        return direct_rate

    direct_mean_interval = _coerce_pattern_arg_float(
        pattern_payload.get("mean_interval_s")
    )
    if direct_mean_interval is not None:
        if direct_mean_interval <= 0:
            raise ValueError(
                f"{pattern_label} replay launch pattern mean interval must be > 0"
            )
        return 1.0 / direct_mean_interval

    return _extract_rate_per_second_from_pattern_args(
        pattern_args_payload,
        pattern_label=pattern_label,
    )


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
) -> tuple[
    str,
    int | None,
    int | None,
    str,
    float | None,
    Callable[[], float],
    dict[str, Any] | None,
    dict[str, Any],
]:
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

    override_selects_uncapped_arrival_pattern = (
        _launch_policy_override_selects_uncapped_arrival_pattern(
            launch_policy_override,
        )
    )
    override_declares_max_concurrent = (
        isinstance(launch_policy_override, dict)
        and "max_concurrent" in launch_policy_override
    )
    if (
        override_selects_uncapped_arrival_pattern
        and not override_declares_max_concurrent
    ):
        launch_max_concurrent: int | None = None
        effective_launch_policy.pop("max_concurrent", None)
    else:
        max_concurrent_raw = effective_launch_policy.get("max_concurrent")
        max_concurrent = _to_int_or_default(max_concurrent_raw, default=1)
        if max_concurrent <= 0:
            raise ValueError("Replay launch max_concurrent must be > 0")
        launch_max_concurrent = max_concurrent

    launch_pattern_rate_per_second: float | None = None
    if pattern_name == "eager":
        next_launch_delay_s = lambda: 0.0
    elif pattern_name in {"poisson", "possion", "uniform"}:
        normalized_pattern_name = (
            "poisson" if pattern_name in {"poisson", "possion"} else "uniform"
        )
        pattern_label = (
            "Poisson" if normalized_pattern_name == "poisson" else "Uniform"
        )
        rate_value = _extract_launch_pattern_rate_per_second(
            pattern_payload,
            effective_launch_policy.get("pattern_args"),
            pattern_label=pattern_label,
        )
        if rate_value is None:
            raise ValueError(
                f"{pattern_label} replay launch pattern requires rate_per_second; "
                "set it in launch_policy.pattern.rate_per_second or launch_policy.pattern_args"
            )
        rate_per_second = rate_value
        if rate_per_second <= 0:
            raise ValueError("Replay launch pattern rate_per_second must be > 0")
        launch_pattern_rate_per_second = rate_per_second
        launch_pattern_name = normalized_pattern_name
        if normalized_pattern_name == "poisson":
            next_launch_delay_s = lambda: rng.expovariate(rate_per_second)
        else:
            fixed_interval_s = 1.0 / rate_per_second
            next_launch_delay_s = lambda: fixed_interval_s
    else:
        raise ValueError(
            f"Unsupported replay launch pattern {pattern_name!r}; supported: eager, poisson, uniform"
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
    lmcache_log_config = resolve_replay_lmcache_log_config(
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
        "lmcache_log_configured": lmcache_log_config.configured,
        "lmcache_log_enabled": False,
        "lmcache_log_endpoint": lmcache_log_config.endpoint,
        "lmcache_log_interval_s": lmcache_log_config.interval_s,
        "lmcache_log_timeout_s": lmcache_log_config.timeout_s,
        "lmcache_log_probe_timeout_s": lmcache_log_config.probe_timeout_s,
        "lmcache_log_probe_success": None,
        "lmcache_log_probe_error": None,
        "lmcache_log_dir": None,
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
    lmcache_monitor: ReplayVLLMMonitorProcess | None = None
    if lmcache_log_config.configured and lmcache_log_config.endpoint is not None:
        probe_success, probe_error = probe_metrics_endpoint(
            endpoint=lmcache_log_config.endpoint,
            timeout_s=lmcache_log_config.probe_timeout_s,
        )
        summary["lmcache_log_probe_success"] = probe_success
        summary["lmcache_log_probe_error"] = probe_error
        if probe_success:
            lmcache_monitor = start_replay_lmcache_monitor(
                output_dir=output_dir,
                config=lmcache_log_config,
            )
            summary["lmcache_log_enabled"] = True
            summary["lmcache_log_dir"] = str(output_dir / "lmcache-log")
            summary["lmcache_log_stdout"] = str(lmcache_monitor.stdout_log)
            summary["lmcache_log_stderr"] = str(lmcache_monitor.stderr_log)

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
        record["api_token"] = api_token
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
                            if _launch_capacity_available(
                                active_count=len(active_threads),
                                launch_max_concurrent=launch_max_concurrent,
                            ):
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
                            if _launch_capacity_available(
                                active_count=len(active_threads),
                                launch_max_concurrent=launch_max_concurrent,
                            ):
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
        if lmcache_monitor is not None:
            summary["lmcache_log_monitor_return_code"] = stop_replay_vllm_monitor(
                lmcache_monitor
            )
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
        help="Compile one profiled job dir or all jobs under a root into replay plans",
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
        "--job-root",
        default=None,
        help=(
            "Path root to recursively discover and compile all profiled job directories. "
            "Cannot be combined with --job-dir or --plan-out."
        ),
    )
    compile_parser.add_argument(
        "--plan-out",
        default=None,
        help=(
            "Output path for replay plan. Default: <job-dir>/replay-plan.json "
            "(or <job-dir>/replay-plan.exclude-unranked.json when "
            "--exclude-unranked-trails is set, or "
            "<job-dir>/replay-plan.trail-<safe-trail>.json when "
            "--single-trail is set)."
        ),
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
    compile_parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model override for compiled replay plans. "
            "Must match a name configured in configs/model_config.toml "
            "(model key, served_model_name, or vllm_model_name). "
            "When set, compile rewrites replay_target.model and each request body model."
        ),
    )
    compile_parser.add_argument(
        "--single-trail",
        default=None,
        help=(
            "Non-split mode only. Compile only the specified source trail name "
            "(for example run_alpha or profile-2/run_beta). Unless --plan-out is "
            "set, the default output becomes "
            "<job-dir>/replay-plan.trail-<safe-trail>.json."
        ),
    )
    compile_parser.add_argument(
        "--split-two-group-plans",
        action="store_true",
        default=None,
        help=(
            "When set, compile writes split plans instead of one: "
            "<plan>.<metric>.top.<ext>, <plan>.<metric>.rest.<ext>, and "
            "<plan>.<metric>.exclude-unranked.<ext>, based on "
            "<job-dir>/original-analysis/split top/rest grouping."
        ),
    )
    compile_parser.add_argument(
        "--split-two-group-metric",
        choices=sorted(SPLIT_TWO_GROUP_METRICS),
        default=None,
        help=(
            "Grouping metric used with --split-two-group-plans. "
            "When omitted, compile generates both token_usage and context_usage plans."
        ),
    )
    compile_parser.add_argument(
        "--exclude-unranked-trails",
        action="store_true",
        default=None,
        help=(
            "Non-split mode only. Exclude trails listed under "
            "<job-dir>/original-analysis/split/top-p-usage-ratio-summary.json "
            "unranked_trails."
        ),
    )
    compile_parser.add_argument(
        "--clean",
        action="store_true",
        default=None,
        help=(
            "Split mode only. Drop source requests with status_code=499 and collapse "
            "their request-time windows. Output plans use a .clean filename suffix."
        ),
    )
    compile_parser.add_argument(
        "--additional-suffix",
        default=None,
        help=(
            "Optional suffix to append before the final .json extension. "
            "For example, with --additional-suffix v2, replay-plan.json "
            "becomes replay-plan.v2.json. Applied after other suffixes."
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
