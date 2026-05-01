#!/usr/bin/env python3
"""Resolve a vLLM port from configs/port_profiles.toml."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def resolve_vllm_port(*, config_path: Path, profile_id: str) -> int:
    normalized_profile_id = profile_id.strip()
    if not normalized_profile_id.isdigit():
        raise ValueError(f"invalid PORT_PROFILE_ID: {profile_id!r}")

    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("configs/port_profiles.toml must define [profiles]")

    profile = profiles.get(normalized_profile_id)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown port profile id: {normalized_profile_id}")

    vllm_port = profile.get("vllm_port")
    if isinstance(vllm_port, bool) or not isinstance(vllm_port, int):
        raise ValueError(f"invalid vllm_port for profile {normalized_profile_id}")

    return vllm_port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve the vLLM port for a port profile id."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configs/port_profiles.toml.",
    )
    parser.add_argument(
        "--profile-id",
        required=True,
        help="Port profile id to resolve.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved_port = resolve_vllm_port(
        config_path=args.config.expanduser().resolve(),
        profile_id=args.profile_id,
    )
    print(resolved_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
