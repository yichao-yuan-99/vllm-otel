from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_PROFILE_ID_KEYS: tuple[str, ...] = (
    "backend_port_profile_id",
    "port_profile_id",
    "profile_id",
    "gateway_profile_id",
)
_PROFILE_ID_LIST_KEYS: tuple[str, ...] = (
    "port_profile_ids",
    "port_profile_id_list",
    "vllm_log_port_profile_ids",
)


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def profile_label(profile_id: int) -> str:
    return f"profile-{profile_id}"


def profile_id_from_payload(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in _PROFILE_ID_KEYS:
        profile_id = int_or_none(payload.get(key))
        if profile_id is not None:
            return profile_id
    return None


def profile_ids_from_payload(payload: Any) -> list[int]:
    if not isinstance(payload, dict):
        return []
    profile_ids: set[int] = set()
    for key in _PROFILE_ID_LIST_KEYS:
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            profile_id = int_or_none(value)
            if profile_id is not None:
                profile_ids.add(profile_id)
    for key in _PROFILE_ID_KEYS:
        profile_id = int_or_none(payload.get(key))
        if profile_id is not None:
            profile_ids.add(profile_id)
    return sorted(profile_ids)


def api_token_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def gateway_run_profile_id_from_manifest(gateway_run_dir: Path) -> int | None:
    manifest_path = gateway_run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return profile_id_from_payload(payload)


def build_gateway_run_profile_id_by_api_token_hash(run_dir: Path) -> dict[str, int]:
    gateway_output_dir = run_dir / "gateway-output"
    if not gateway_output_dir.is_dir():
        return {}

    profile_id_by_api_token_hash: dict[str, int] = {}
    for manifest_path in sorted(gateway_output_dir.rglob("manifest.json")):
        gateway_run_dir = manifest_path.parent
        if not gateway_run_dir.is_dir() or not gateway_run_dir.name.startswith("run_"):
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        api_token_hash = payload.get("api_token_hash")
        profile_id = profile_id_from_payload(payload)
        if not isinstance(api_token_hash, str) or not api_token_hash or profile_id is None:
            continue
        profile_id_by_api_token_hash[api_token_hash] = profile_id
    return profile_id_by_api_token_hash
