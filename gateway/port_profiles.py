from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
PORT_PROFILES_PATH = REPO_ROOT / "configs" / "port_profiles.toml"


@dataclass(frozen=True)
class PortProfile:
    profile_id: str
    label: str | None
    vllm_port: int
    jaeger_api_port: int
    jaeger_otlp_port: int
    gateway_port: int
    gateway_parse_port: int
    lmcache_port: int | None


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _parse_port(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < 1 or value > 65535:
        raise ValueError(f"{key} must be in range 1..65535")
    return value


def _parse_optional_port(value: object, key: str) -> int | None:
    if value is None:
        return None
    return _parse_port(value, key)


def load_port_profile(
    profile_id: int | str | None = None,
    *,
    config_path: Path | None = None,
) -> PortProfile:
    path = config_path or PORT_PROFILES_PATH
    payload = _load_toml(path)
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise ValueError("configs/port_profiles.toml must include [profiles]")

    if profile_id is None:
        default_profile = payload.get("default_profile")
        if isinstance(default_profile, str) and default_profile in raw_profiles:
            key = default_profile
        elif raw_profiles:
            key = sorted(raw_profiles.keys())[0]
        else:
            raise ValueError("configs/port_profiles.toml has no profiles")
    else:
        key = str(profile_id)

    raw = raw_profiles.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"unknown port profile id: {key}")

    return PortProfile(
        profile_id=key,
        label=raw.get("label") if isinstance(raw.get("label"), str) else None,
        vllm_port=_parse_port(raw.get("vllm_port"), f"profiles.{key}.vllm_port"),
        jaeger_api_port=_parse_port(raw.get("jaeger_api_port"), f"profiles.{key}.jaeger_api_port"),
        jaeger_otlp_port=_parse_port(raw.get("jaeger_otlp_port"), f"profiles.{key}.jaeger_otlp_port"),
        gateway_port=_parse_port(raw.get("gateway_port"), f"profiles.{key}.gateway_port"),
        gateway_parse_port=_parse_port(
            raw.get("gateway_parse_port"),
            f"profiles.{key}.gateway_parse_port",
        ),
        lmcache_port=_parse_optional_port(
            raw.get("lmcache_port"),
            f"profiles.{key}.lmcache_port",
        ),
    )
