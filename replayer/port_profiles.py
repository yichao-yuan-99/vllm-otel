"""Port profile helpers for replay tooling."""

from __future__ import annotations

from gateway.port_profiles import PortProfile, load_port_profile


def build_replay_target_from_port_profile(
    profile_id: int | str,
    *,
    gateway_enabled: bool,
) -> tuple[str, str, str]:
    profile = load_port_profile(profile_id)
    gateway_url = f"http://127.0.0.1:{profile.gateway_port}"
    api_base = (
        f"{gateway_url}/v1"
        if gateway_enabled
        else f"http://127.0.0.1:{profile.vllm_port}/v1"
    )
    tokenize_endpoint = f"http://127.0.0.1:{profile.vllm_port}/tokenize"
    return gateway_url, api_base, tokenize_endpoint


__all__ = ["PortProfile", "build_replay_target_from_port_profile", "load_port_profile"]
