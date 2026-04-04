#!/usr/bin/env python3
"""GPU frequency controller driven by gateway context usage."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime, timezone
import http.client
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time
from typing import Any
from typing import Callable
from typing import Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_CONTROL_INTERVAL_S = 5.0
DEFAULT_CONTEXT_QUERY_HZ = 5.0
DEFAULT_FREQUENCY_MHZ_LEVELS = (810, 1005, 1200, 1395)
DEFAULT_SHARED_CONFIG_FILE_NAME = "script-shared.toml"
DEFAULT_GATEWAY_PORT_PROFILE_ID = 0
DEFAULT_GATEWAY_IPC_SOCKET_DIR = Path("/tmp")
DEFAULT_ZEUSD_SOCKET_PATH = "/var/run/zeusd.sock"
DEFAULT_GPU_INDEX = 0
DEFAULT_GATEWAY_TIMEOUT_S = 5.0
DEFAULT_LOG_FILE_PREFIX = "freq-controller"


def now_iso8601_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def iso8601_to_compact(iso_value: str) -> str:
    dt = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _parse_float(value: object, key: str, *, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"{key} is required")
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite")
    return parsed


def _parse_int(value: object, key: str, *, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{key} is required")
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _parse_optional_str(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    stripped = value.strip()
    return stripped or None


def _parse_frequency_levels(value: object, key: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list of integers")
    parsed: list[int] = []
    for index, item in enumerate(value):
        item_key = f"{key}[{index}]"
        item_value = _parse_int(item, item_key)
        if item_value <= 0:
            raise ValueError(f"{item_key} must be > 0")
        parsed.append(item_value)
    unique_sorted = sorted(set(parsed))
    if len(unique_sorted) != len(parsed):
        raise ValueError(f"{key} must not contain duplicate values")
    return tuple(unique_sorted)


def _default_gateway_ipc_socket_path(port_profile_id: int | str | None) -> Path:
    resolved_profile_id = (
        str(DEFAULT_GATEWAY_PORT_PROFILE_ID)
        if port_profile_id is None
        else str(port_profile_id)
    )
    return (
        DEFAULT_GATEWAY_IPC_SOCKET_DIR
        / f"vllm-gateway-profile-{resolved_profile_id}.sock"
    )


def _lookup_first(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _lookup_with_fallback(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    keys: Sequence[str],
) -> Any:
    value = _lookup_first(primary, keys)
    if value is not None:
        return value
    return _lookup_first(fallback, keys)


def _load_toml_payload(path: Path) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"missing config file: {resolved_path}")
    payload = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a TOML table: {resolved_path}")
    return payload


def _default_shared_config_path() -> Path:
    return Path(__file__).resolve().with_name(DEFAULT_SHARED_CONFIG_FILE_NAME)


def _load_shared_controller_table(path: Path) -> dict[str, Any]:
    payload = _load_toml_payload(path)
    controller_table = payload.get("controller")
    if controller_table is None:
        return payload
    if isinstance(controller_table, dict):
        return controller_table
    raise ValueError(f"shared controller must be a TOML table: {path}")


@dataclass(frozen=True)
class GatewayIPCConfig:
    ipc_socket_path: str | None = None
    timeout_s: float = DEFAULT_GATEWAY_TIMEOUT_S

    def resolved_socket_path(self, port_profile_id: int) -> Path:
        if self.ipc_socket_path:
            return Path(self.ipc_socket_path).expanduser().resolve()
        return _default_gateway_ipc_socket_path(port_profile_id).resolve()

    def to_log_payload(self, *, port_profile_id: int) -> dict[str, Any]:
        return {
            "port_profile_id": port_profile_id,
            "ipc_socket_path": str(self.resolved_socket_path(port_profile_id)),
            "timeout_s": self.timeout_s,
        }


@dataclass(frozen=True)
class GatewayContextSnapshot:
    job_active: bool
    job_started_at: str | None
    agent_count: int
    total_context_tokens: float


@dataclass(frozen=True)
class FrequencyControllerLogPaths:
    query_path: Path
    decision_path: Path


@dataclass(frozen=True)
class ZeusdConfig:
    socket_path: str = DEFAULT_ZEUSD_SOCKET_PATH

    def to_log_payload(self, *, gpu_index: int) -> dict[str, Any]:
        return {
            "socket_path": self.socket_path,
            "gpu_index": gpu_index,
        }


@dataclass(frozen=True)
class FrequencyControllerConfig:
    frequency_mhz_levels: tuple[int, ...]
    target_context_usage_lower_bound: float
    target_context_usage_upper_bound: float
    control_interval_s: float = DEFAULT_CONTROL_INTERVAL_S
    context_query_hz: float = DEFAULT_CONTEXT_QUERY_HZ
    gateway: GatewayIPCConfig = field(default_factory=GatewayIPCConfig)
    zeusd: ZeusdConfig = field(default_factory=ZeusdConfig)

    def __post_init__(self) -> None:
        if not self.frequency_mhz_levels:
            raise ValueError("frequency_mhz_levels must be non-empty")
        if self.control_interval_s <= 0:
            raise ValueError("control_interval_s must be > 0")
        if self.context_query_hz <= 0:
            raise ValueError("context_query_hz must be > 0")
        if (
            self.target_context_usage_lower_bound
            > self.target_context_usage_upper_bound
        ):
            raise ValueError(
                "target_context_usage_lower_bound must be <= "
                "target_context_usage_upper_bound"
            )

    @property
    def query_interval_s(self) -> float:
        return 1.0 / self.context_query_hz

    @property
    def initial_frequency_index(self) -> int:
        return len(self.frequency_mhz_levels) // 2

    @property
    def initial_frequency_mhz(self) -> int:
        return self.frequency_mhz_levels[self.initial_frequency_index]

    def to_log_payload(
        self,
        *,
        port_profile_id: int,
        gpu_index: int,
    ) -> dict[str, Any]:
        return {
            "frequency_mhz_levels": list(self.frequency_mhz_levels),
            "target_context_usage_lower_bound": self.target_context_usage_lower_bound,
            "target_context_usage_upper_bound": self.target_context_usage_upper_bound,
            "control_interval_s": self.control_interval_s,
            "context_query_hz": self.context_query_hz,
            "gateway": self.gateway.to_log_payload(
                port_profile_id=port_profile_id,
            ),
            "zeusd": self.zeusd.to_log_payload(gpu_index=gpu_index),
        }


def load_controller_config(
    config_path: Path | None,
    *,
    target_context_usage_lower_bound: float | None = None,
    target_context_usage_upper_bound: float | None = None,
) -> FrequencyControllerConfig:
    default_shared_config_table = _load_shared_controller_table(
        _default_shared_config_path(),
    )
    if config_path is None:
        resolved_config_path: Path | None = None
        payload: dict[str, Any] = {}
    else:
        resolved_config_path = config_path.expanduser().resolve()
        payload = _load_toml_payload(resolved_config_path)

    controller_table = payload.get("controller")
    if controller_table is None:
        config_table = payload
    elif isinstance(controller_table, dict):
        config_table = controller_table
    else:
        raise ValueError("controller must be a TOML table")

    shared_table = payload.get("shared")
    if shared_table is None:
        shared_table = {}
    elif not isinstance(shared_table, dict):
        raise ValueError("shared must be a TOML table")

    shared_config_ref = _parse_optional_str(
        _lookup_first(shared_table, ("config_path", "shared_config_path")),
        "shared.config_path",
    )
    if shared_config_ref:
        shared_config_path = Path(shared_config_ref)
        if not shared_config_path.is_absolute():
            shared_config_path = (
                resolved_config_path.parent / shared_config_path
            ).resolve()
        shared_config_table = {
            **default_shared_config_table,
            **_load_shared_controller_table(shared_config_path),
        }
    else:
        shared_config_table = default_shared_config_table

    gateway_table = payload.get("gateway")
    if gateway_table is None:
        gateway_table = {}
    elif not isinstance(gateway_table, dict):
        raise ValueError("gateway must be a TOML table")
    if "port_profile_id" in gateway_table:
        raise ValueError(
            "gateway.port_profile_id is no longer configured in TOML; "
            "use --port-profile-id instead"
        )

    zeusd_table = payload.get("zeusd")
    if zeusd_table is None:
        zeusd_table = {}
    elif not isinstance(zeusd_table, dict):
        raise ValueError("zeusd must be a TOML table")
    if "gpu_index" in zeusd_table:
        raise ValueError(
            "zeusd.gpu_index is no longer configured in TOML; "
            "use --gpu-index instead"
        )

    frequency_levels = _parse_frequency_levels(
        _lookup_with_fallback(
            config_table,
            shared_config_table,
            ("frequency_mhz_levels", "frequencies_mhz", "frequency_mhz"),
        ),
        "frequency_mhz_levels",
    )
    lower_bound_value = (
        target_context_usage_lower_bound
        if target_context_usage_lower_bound is not None
        else _lookup_first(
            config_table,
            (
                "target_context_usage_lower_bound",
                "target_context_tokens_lower_bound",
            ),
        )
    )
    lower_bound = _parse_float(
        lower_bound_value,
        "target_context_usage_lower_bound",
    )
    upper_bound_value = (
        target_context_usage_upper_bound
        if target_context_usage_upper_bound is not None
        else _lookup_first(
            config_table,
            (
                "target_context_usage_upper_bound",
                "target_context_tokens_upper_bound",
            ),
        )
    )
    upper_bound = _parse_float(
        upper_bound_value,
        "target_context_usage_upper_bound",
    )
    control_interval_s = _parse_float(
        _lookup_with_fallback(
            config_table,
            shared_config_table,
            ("control_interval_s", "control_interval"),
        ),
        "control_interval_s",
    )
    context_query_hz = _parse_float(
        _lookup_with_fallback(
            config_table,
            shared_config_table,
            ("context_query_hz", "context_query_frequency_hz"),
        ),
        "context_query_hz",
    )

    gateway = GatewayIPCConfig(
        ipc_socket_path=_parse_optional_str(
            gateway_table.get("ipc_socket_path"),
            "gateway.ipc_socket_path",
        ),
        timeout_s=_parse_float(
            gateway_table.get("timeout_s"),
            "gateway.timeout_s",
            default=DEFAULT_GATEWAY_TIMEOUT_S,
        ),
    )
    zeusd = ZeusdConfig(
        socket_path=_parse_optional_str(
            zeusd_table.get("socket_path"),
            "zeusd.socket_path",
        )
        or DEFAULT_ZEUSD_SOCKET_PATH,
    )
    return FrequencyControllerConfig(
        frequency_mhz_levels=frequency_levels,
        target_context_usage_lower_bound=lower_bound,
        target_context_usage_upper_bound=upper_bound,
        control_interval_s=control_interval_s,
        context_query_hz=context_query_hz,
        gateway=gateway,
        zeusd=zeusd,
    )


class UnixSocketHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._socket_path)


class GatewayContextClient:
    def __init__(self, socket_path: Path, timeout_s: float = DEFAULT_GATEWAY_TIMEOUT_S) -> None:
        self.socket_path = socket_path.expanduser().resolve()
        self.timeout_s = timeout_s

    def read_context_snapshot(self) -> GatewayContextSnapshot:
        connection = UnixSocketHTTPConnection(str(self.socket_path), timeout=self.timeout_s)
        try:
            connection.request("GET", "/ipc/context", headers={"Host": "localhost"})
            response = connection.getresponse()
            raw_payload = response.read()
        finally:
            connection.close()

        if response.status != 200:
            raise RuntimeError(
                f"gateway IPC request failed with status={response.status}: "
                f"{raw_payload.decode('utf-8', errors='replace')}"
            )
        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("gateway IPC response was not valid JSON") from exc

        job_active = payload.get("job_active")
        if not isinstance(job_active, bool):
            raise RuntimeError("gateway IPC response is missing job_active")
        job_started_at = payload.get("job_started_at")
        if job_started_at is not None and not isinstance(job_started_at, str):
            raise RuntimeError("gateway IPC response job_started_at must be a string")
        agent_count = payload.get("agent_count")
        if isinstance(agent_count, bool) or not isinstance(agent_count, int):
            raise RuntimeError("gateway IPC response is missing agent_count")
        if agent_count < 0:
            raise RuntimeError("gateway IPC response agent_count must be >= 0")
        total_context_tokens = payload.get("total_context_tokens")
        if isinstance(total_context_tokens, bool) or not isinstance(
            total_context_tokens, (int, float)
        ):
            raise RuntimeError("gateway IPC response is missing total_context_tokens")
        value = float(total_context_tokens)
        if not math.isfinite(value):
            raise RuntimeError("gateway IPC response total_context_tokens is not finite")
        return GatewayContextSnapshot(
            job_active=job_active,
            job_started_at=job_started_at,
            agent_count=agent_count,
            total_context_tokens=value,
        )

    def read_total_context_usage(self) -> float:
        return self.read_context_snapshot().total_context_tokens


class ZeusdGPUFrequencyController:
    def __init__(self, *, socket_path: str, gpu_index: int) -> None:
        self.socket_path = socket_path
        self.gpu_index = gpu_index
        self._gpus: Any | None = None

    def _get_gpus(self) -> Any:
        if self._gpus is None:
            os.environ["ZEUSD_SOCK_PATH"] = self.socket_path
            from zeus.device import get_gpus  # pylint: disable=import-outside-toplevel

            self._gpus = get_gpus()
        return self._gpus

    def set_frequency(self, frequency_mhz: int) -> None:
        gpus = self._get_gpus()
        gpus.set_gpu_locked_clocks(
            gpu_index=self.gpu_index,
            min_clock_mhz=frequency_mhz,
            max_clock_mhz=frequency_mhz,
        )

    def reset_frequency(self) -> None:
        gpus = self._get_gpus()
        gpus.reset_gpu_locked_clocks(gpu_index=self.gpu_index)


class JsonlLogFile:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, **fields: Any) -> None:
        payload = {"timestamp": now_iso8601_utc()}
        payload.update(fields)
        self._handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class FrequencyControllerLogs:
    def __init__(self, log_dir: Path) -> None:
        resolved_log_dir = log_dir.expanduser().resolve()
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        started_at = iso8601_to_compact(now_iso8601_utc())
        self.query = JsonlLogFile(
            resolved_log_dir / f"{DEFAULT_LOG_FILE_PREFIX}.query.{started_at}.jsonl"
        )
        self.decision = JsonlLogFile(
            resolved_log_dir / f"{DEFAULT_LOG_FILE_PREFIX}.decision.{started_at}.jsonl"
        )
        self.paths = FrequencyControllerLogPaths(
            query_path=self.query.path,
            decision_path=self.decision.path,
        )

    def close(self) -> None:
        self.query.close()
        self.decision.close()


class MovingAverageWindow:
    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._samples: deque[tuple[float, float]] = deque()

    def add(self, timestamp_s: float, value: float) -> None:
        self._samples.append((timestamp_s, value))
        self.prune(timestamp_s)

    def prune(self, now_s: float) -> None:
        cutoff_s = now_s - self.window_s
        while self._samples and self._samples[0][0] < cutoff_s:
            self._samples.popleft()

    def average(self, now_s: float) -> float | None:
        self.prune(now_s)
        if not self._samples:
            return None
        return sum(value for _, value in self._samples) / len(self._samples)

    @property
    def sample_count(self) -> int:
        return len(self._samples)


def choose_next_frequency_index(
    *,
    current_index: int,
    moving_average_context_usage: float,
    lower_bound: float,
    upper_bound: float,
    max_index: int,
) -> tuple[int, str]:
    if moving_average_context_usage < lower_bound:
        return max(0, current_index - 1), "decrease"
    if moving_average_context_usage > upper_bound:
        return min(max_index, current_index + 1), "increase"
    return current_index, "hold"


class FrequencyController:
    def __init__(
        self,
        config: FrequencyControllerConfig,
        log_dir: Path,
        *,
        port_profile_id: int = DEFAULT_GATEWAY_PORT_PROFILE_ID,
        gpu_index: int = DEFAULT_GPU_INDEX,
        gateway_client: GatewayContextClient | None = None,
        gpu_controller: ZeusdGPUFrequencyController | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep_func: Callable[[float], None] | None = None,
    ) -> None:
        if port_profile_id < 0:
            raise ValueError("port_profile_id must be >= 0")
        if gpu_index < 0:
            raise ValueError("gpu_index must be >= 0")
        self.config = config
        self.port_profile_id = port_profile_id
        self.gpu_index = gpu_index
        self.gateway_client = gateway_client or GatewayContextClient(
            config.gateway.resolved_socket_path(port_profile_id),
            timeout_s=config.gateway.timeout_s,
        )
        self.gpu_controller = gpu_controller or ZeusdGPUFrequencyController(
            socket_path=config.zeusd.socket_path,
            gpu_index=gpu_index,
        )
        self.monotonic = monotonic or time.monotonic
        self.sleep_func = sleep_func or time.sleep
        self.logs = FrequencyControllerLogs(log_dir)
        self.window = MovingAverageWindow(config.control_interval_s)
        self.current_frequency_index = config.initial_frequency_index
        self.last_snapshot: GatewayContextSnapshot | None = None

    @property
    def current_frequency_mhz(self) -> int:
        return self.config.frequency_mhz_levels[self.current_frequency_index]

    def _apply_frequency_index(
        self,
        index: int,
        *,
        reason: str,
        action: str,
        moving_average_context_usage: float | None = None,
        sample_count: int | None = None,
    ) -> None:
        previous_index = self.current_frequency_index
        next_frequency_mhz = self.config.frequency_mhz_levels[index]
        self.gpu_controller.set_frequency(next_frequency_mhz)
        self.current_frequency_index = index

    def _read_context_snapshot_with_fallback(
        self,
    ) -> tuple[GatewayContextSnapshot, str | None]:
        try:
            snapshot = self.gateway_client.read_context_snapshot()
        except Exception as exc:
            if self.last_snapshot is None:
                raise
            return self.last_snapshot, str(exc)
        self.last_snapshot = snapshot
        return snapshot, None

    def _sample_context_usage(self, now_s: float) -> None:
        snapshot, read_error = self._read_context_snapshot_with_fallback()
        context_usage = snapshot.total_context_tokens
        self.window.add(now_s, context_usage)
        query_fields: dict[str, Any] = {
            "phase": "active",
            "context_usage": context_usage,
            "job_active": snapshot.job_active,
            "agent_count": snapshot.agent_count,
            "sample_count_window": self.window.sample_count,
        }
        if read_error is not None:
            query_fields["error"] = read_error
        self.logs.query.write(
            **query_fields,
        )

    def _wait_for_job_start(
        self,
        *,
        should_stop: Callable[[], bool],
    ) -> bool:
        while True:
            if should_stop():
                return False

            try:
                snapshot, read_error = self._read_context_snapshot_with_fallback()
            except Exception as exc:
                self.logs.query.write(
                    phase="pending",
                    error=str(exc),
                )
            else:
                query_fields: dict[str, Any] = {
                    "phase": "pending",
                    "context_usage": snapshot.total_context_tokens,
                    "job_active": snapshot.job_active,
                    "agent_count": snapshot.agent_count,
                }
                if read_error is not None:
                    query_fields["error"] = read_error
                self.logs.query.write(**query_fields)
                if read_error is None and snapshot.job_active:
                    return True

            self.sleep_func(max(0.0, self.config.query_interval_s))

    def _control_once(self, now_s: float) -> None:
        moving_average = self.window.average(now_s)
        if moving_average is None:
            return

        current_frequency_mhz = self.current_frequency_mhz
        next_index, action = choose_next_frequency_index(
            current_index=self.current_frequency_index,
            moving_average_context_usage=moving_average,
            lower_bound=self.config.target_context_usage_lower_bound,
            upper_bound=self.config.target_context_usage_upper_bound,
            max_index=len(self.config.frequency_mhz_levels) - 1,
        )
        target_frequency_mhz = self.config.frequency_mhz_levels[next_index]
        changed = next_index != self.current_frequency_index
        self.logs.decision.write(
            action=action,
            changed=changed,
            current_frequency_mhz=current_frequency_mhz,
            target_frequency_mhz=target_frequency_mhz,
            window_context_usage=moving_average,
            sample_count=self.window.sample_count,
            lower_bound=self.config.target_context_usage_lower_bound,
            upper_bound=self.config.target_context_usage_upper_bound,
        )
        if not changed:
            return

        try:
            self._apply_frequency_index(
                next_index,
                reason="control_decision",
                action=action,
                moving_average_context_usage=moving_average,
                sample_count=self.window.sample_count,
            )
        except Exception:
            raise

    def run(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
        max_control_decisions: int | None = None,
    ) -> FrequencyControllerLogPaths:
        should_stop = stop_requested or (lambda: False)
        decision_count = 0

        try:
            if not self._wait_for_job_start(should_stop=should_stop):
                return self.logs.paths

            start_monotonic = self.monotonic()
            next_sample_s = start_monotonic
            next_control_s = start_monotonic + self.config.control_interval_s
            self._apply_frequency_index(
                self.current_frequency_index,
                reason="job_started",
                action="initialize",
            )
            while True:
                if should_stop():
                    break
                if (
                    max_control_decisions is not None
                    and decision_count >= max_control_decisions
                ):
                    break

                now_s = self.monotonic()
                if now_s >= next_sample_s:
                    try:
                        self._sample_context_usage(now_s)
                    except Exception as exc:
                        self.logs.query.write(
                            phase="active",
                            error=str(exc),
                        )
                    next_sample_s += self.config.query_interval_s
                    while next_sample_s <= now_s:
                        next_sample_s += self.config.query_interval_s
                    continue

                if now_s >= next_control_s:
                    self._control_once(now_s)
                    decision_count += 1
                    next_control_s += self.config.control_interval_s
                    while next_control_s <= now_s:
                        next_control_s += self.config.control_interval_s
                    continue

                sleep_s = min(next_sample_s, next_control_s) - now_s
                self.sleep_func(max(0.0, min(sleep_s, 0.5)))
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.gpu_controller.reset_frequency()
            except Exception:
                pass
            self.logs.close()
        return self.logs.paths


def run_controller(
    config_path: Path | None,
    log_dir: Path,
    *,
    target_context_usage_lower_bound: float | None = None,
    target_context_usage_upper_bound: float | None = None,
    port_profile_id: int = DEFAULT_GATEWAY_PORT_PROFILE_ID,
    gpu_index: int = DEFAULT_GPU_INDEX,
) -> FrequencyControllerLogPaths:
    config = load_controller_config(
        config_path,
        target_context_usage_lower_bound=target_context_usage_lower_bound,
        target_context_usage_upper_bound=target_context_usage_upper_bound,
    )
    stop_flag = {"requested": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_flag["requested"] = True

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        controller = FrequencyController(
            config,
            log_dir,
            port_profile_id=port_profile_id,
            gpu_index=gpu_index,
        )
        return controller.run(stop_requested=lambda: stop_flag["requested"])
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="freq-controller",
        description=(
            "Control GPU core frequency from gateway context usage using zeusd."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help=(
            "Optional path to a controller TOML config file used to override "
            "the default shared settings."
        ),
    )
    parser.add_argument(
        "--log-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory where the controller log files will be written.",
    )
    parser.add_argument(
        "--target-context-usage-lower-bound",
        "--lower-bound",
        dest="target_context_usage_lower_bound",
        type=float,
        help=(
            "Lower moving-average context usage bound. Required unless "
            "configured in TOML."
        ),
    )
    parser.add_argument(
        "--target-context-usage-upper-bound",
        "--upper-bound",
        dest="target_context_usage_upper_bound",
        type=float,
        help=(
            "Upper moving-average context usage bound. Required unless "
            "configured in TOML."
        ),
    )
    parser.add_argument(
        "--port-profile-id",
        type=int,
        default=DEFAULT_GATEWAY_PORT_PROFILE_ID,
        help=(
            "Gateway port profile id used to derive the default IPC socket path. "
            f"Default: {DEFAULT_GATEWAY_PORT_PROFILE_ID}."
        ),
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=DEFAULT_GPU_INDEX,
        help=(
            "GPU index to control through zeusd. "
            f"Default: {DEFAULT_GPU_INDEX}."
        ),
    )
    args = parser.parse_args(argv)

    try:
        log_paths = run_controller(
            args.config,
            args.log_dir,
            target_context_usage_lower_bound=args.target_context_usage_lower_bound,
            target_context_usage_upper_bound=args.target_context_usage_upper_bound,
            port_profile_id=args.port_profile_id,
            gpu_index=args.gpu_index,
        )
        print(
            (
                "freq-controller logs: "
                f"query={log_paths.query_path} "
                f"decision={log_paths.decision_path}"
            ),
            file=sys.stderr,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
