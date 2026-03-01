from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "gateway" / "config.toml"


@dataclass(frozen=True)
class GatewayRuntimeSettings:
    port_profile_id: int | None = None
    output_root: str | None = None
    service_name: str = "vllm-gateway"
    otlp_traces_insecure: bool = True
    artifact_compression: str = "none"
    job_end_trace_wait_seconds: float = 10.0


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _require_table(value: object, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a TOML table")
    return value


def _parse_optional_int(value: object, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _parse_optional_str(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    stripped = value.strip()
    return stripped or None


def _parse_bool(value: object, key: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _parse_float(value: object, key: str, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def load_runtime_settings(
    config_path: Path | None = None,
    *,
    allow_missing: bool = True,
) -> GatewayRuntimeSettings:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        if allow_missing:
            return GatewayRuntimeSettings()
        raise FileNotFoundError(f"missing config file: {path}")

    payload = _load_toml(path)
    run_table = _require_table(payload.get("run"), "run")
    gateway_table = _require_table(payload.get("gateway"), "gateway")
    telemetry_table = _require_table(payload.get("telemetry"), "telemetry")

    return GatewayRuntimeSettings(
        port_profile_id=_parse_optional_int(run_table.get("port_profile_id"), "run.port_profile_id"),
        output_root=_parse_optional_str(run_table.get("output_root"), "run.output_root"),
        service_name=_parse_optional_str(telemetry_table.get("service_name"), "telemetry.service_name") or "vllm-gateway",
        otlp_traces_insecure=_parse_bool(
            telemetry_table.get("otlp_traces_insecure"),
            "telemetry.otlp_traces_insecure",
            default=True,
        ),
        artifact_compression=_parse_optional_str(
            gateway_table.get("artifact_compression"),
            "gateway.artifact_compression",
        ) or "none",
        job_end_trace_wait_seconds=_parse_float(
            gateway_table.get("job_end_trace_wait_seconds"),
            "gateway.job_end_trace_wait_seconds",
            default=10.0,
        ),
    )
