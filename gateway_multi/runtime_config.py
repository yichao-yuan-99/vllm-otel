from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "gateway_multi" / "config.toml"
DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS = 263_856
SUPPORTED_ASSIGNMENT_POLICIES = {
    "balanced",
    "round_robin",
    "lowest_usage",
    "lowest_profile_without_pending",
}


@dataclass(frozen=True)
class GatewayMultiRuntimeSettings:
    port_profile_ids: tuple[str, ...] = ("0",)
    assignment_policy: str = "round_robin"
    balanced_usage_threshold_tokens: int = DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS
    output_root: str | None = None
    service_name: str = "vllm-gateway-multi"
    otlp_traces_insecure: bool = True
    artifact_compression: str = "none"
    job_end_trace_wait_seconds: float = 10.0
    ipc_enabled: bool = True
    ipc_socket_path_template: str | None = None
    ipc_socket_permissions: int = 0o660
    ipc_socket_uid: int | None = None
    ipc_socket_gid: int | None = None


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


def _parse_positive_int(value: object, key: str, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _parse_octal_permissions(value: object, key: str, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an octal permission string")
    if isinstance(value, int):
        raw = str(value)
    elif isinstance(value, str):
        raw = value.strip()
    else:
        raise ValueError(f"{key} must be an octal permission string")
    if not raw:
        raise ValueError(f"{key} cannot be empty")
    try:
        return int(raw, 8)
    except ValueError as exc:
        raise ValueError(
            f"{key} must be an octal permission string like '660'"
        ) from exc


def _normalize_assignment_policy(value: str, key: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{key} cannot be empty")
    if normalized not in SUPPORTED_ASSIGNMENT_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_ASSIGNMENT_POLICIES))
        raise ValueError(f"{key} must be one of: {supported}")
    return normalized


def _parse_port_profile_ids(value: object, key: str) -> tuple[str, ...]:
    if value is None:
        return ("0",)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    if not value:
        raise ValueError(f"{key} must contain at least one profile id")

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{key} must contain integers or strings")
        if isinstance(item, int):
            normalized = str(item)
        elif isinstance(item, str):
            normalized = item.strip()
        else:
            raise ValueError(f"{key} must contain integers or strings")
        if not normalized:
            raise ValueError(f"{key} cannot contain empty profile ids")
        if normalized in seen:
            raise ValueError(f"{key} cannot contain duplicate profile ids")
        result.append(normalized)
        seen.add(normalized)
    return tuple(result)


def load_runtime_settings(
    config_path: Path | None = None,
    *,
    allow_missing: bool = True,
) -> GatewayMultiRuntimeSettings:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        if allow_missing:
            return GatewayMultiRuntimeSettings()
        raise FileNotFoundError(f"missing config file: {path}")

    payload = _load_toml(path)
    run_table = _require_table(payload.get("run"), "run")
    gateway_table = _require_table(payload.get("gateway"), "gateway")
    telemetry_table = _require_table(payload.get("telemetry"), "telemetry")
    ipc_table = _require_table(payload.get("ipc"), "ipc")

    return GatewayMultiRuntimeSettings(
        port_profile_ids=_parse_port_profile_ids(
            run_table.get("port_profile_ids"),
            "run.port_profile_ids",
        ),
        assignment_policy=_normalize_assignment_policy(
            _parse_optional_str(
                run_table.get("assignment_policy"),
                "run.assignment_policy",
            )
            or "round_robin",
            "run.assignment_policy",
        ),
        balanced_usage_threshold_tokens=_parse_positive_int(
            run_table.get("balanced_usage_threshold_tokens"),
            "run.balanced_usage_threshold_tokens",
            default=DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS,
        ),
        output_root=_parse_optional_str(run_table.get("output_root"), "run.output_root"),
        service_name=_parse_optional_str(
            telemetry_table.get("service_name"),
            "telemetry.service_name",
        )
        or "vllm-gateway-multi",
        otlp_traces_insecure=_parse_bool(
            telemetry_table.get("otlp_traces_insecure"),
            "telemetry.otlp_traces_insecure",
            default=True,
        ),
        artifact_compression=_parse_optional_str(
            gateway_table.get("artifact_compression"),
            "gateway.artifact_compression",
        )
        or "none",
        job_end_trace_wait_seconds=_parse_float(
            gateway_table.get("job_end_trace_wait_seconds"),
            "gateway.job_end_trace_wait_seconds",
            default=10.0,
        ),
        ipc_enabled=_parse_bool(
            ipc_table.get("enabled"),
            "ipc.enabled",
            default=True,
        ),
        ipc_socket_path_template=_parse_optional_str(
            ipc_table.get("socket_path_template"),
            "ipc.socket_path_template",
        ),
        ipc_socket_permissions=_parse_octal_permissions(
            ipc_table.get("socket_permissions"),
            "ipc.socket_permissions",
            default=0o660,
        ),
        ipc_socket_uid=_parse_optional_int(ipc_table.get("socket_uid"), "ipc.socket_uid"),
        ipc_socket_gid=_parse_optional_int(ipc_table.get("socket_gid"), "ipc.socket_gid"),
    )
