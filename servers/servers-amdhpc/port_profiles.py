#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared AMD HPC port-profile helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
PORT_PROFILES_PATH = REPO_ROOT / "configs" / "port_profiles.toml"
DEFAULT_REMOTE_SERVER_PORT = 23971


@dataclass(frozen=True)
class PortProfile:
    profile_id: int
    label: str
    vllm_port: int
    jaeger_api_port: int
    jaeger_otlp_port: int


def _parse_port(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < 1 or value > 65535:
        raise ValueError(f"{key} must be in range 1..65535")
    return value


def _load_profiles_payload(config_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    path = (config_path or PORT_PROFILES_PATH).resolve()
    if not path.exists():
        raise FileNotFoundError(f"port profiles file not found: {path}")

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"port profiles file is missing [profiles]: {path}")
    return path, profiles


def load_port_profile(profile_id: int | str, *, config_path: Path | None = None) -> PortProfile:
    path, profiles = _load_profiles_payload(config_path)
    profile_key = str(profile_id)
    profile = profiles.get(profile_key)
    if not isinstance(profile, dict):
        raise KeyError(f"unknown port profile id: {profile_id}")

    label = profile.get("label")
    if label is not None and not isinstance(label, str):
        raise ValueError(f"profiles.{profile_key}.label must be a string in {path}")

    return PortProfile(
        profile_id=int(profile_key),
        label=label or profile_key,
        vllm_port=_parse_port(profile.get("vllm_port"), f"profiles.{profile_key}.vllm_port"),
        jaeger_api_port=_parse_port(profile.get("jaeger_api_port"), f"profiles.{profile_key}.jaeger_api_port"),
        jaeger_otlp_port=_parse_port(profile.get("jaeger_otlp_port"), f"profiles.{profile_key}.jaeger_otlp_port"),
    )


def load_port_profiles(*, config_path: Path | None = None) -> dict[int, PortProfile]:
    _, profiles = _load_profiles_payload(config_path)
    out: dict[int, PortProfile] = {}
    for profile_key in sorted(profiles.keys(), key=int):
        out[int(profile_key)] = load_port_profile(profile_key, config_path=config_path)
    return out


def default_local_server_port(
    profile_id: int | str,
    *,
    remote_server_port: int = DEFAULT_REMOTE_SERVER_PORT,
) -> int:
    resolved_profile_id = int(profile_id)
    local_server_port = remote_server_port + resolved_profile_id
    return _parse_port(local_server_port, "local_server_port")

