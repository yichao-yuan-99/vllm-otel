"""Harbor-specific runtime resolution and forwarded-arg synthesis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit

from con_driver.runtime_resolution import (
    build_endpoint_env,
    build_model_info_json,
    load_agent_defaults,
    load_model_catalog,
    load_port_profile,
    probe_single_served_model_name,
    stringify_agent_kwarg_value,
)

_AGENT_OPTION_NAMES = {"--agent"}
_MODEL_OPTION_NAMES = {"--model"}
_AGENT_KWARG_OPTION_NAMES = {"--agent-kwarg", "--ak"}
_AUTO_MANAGED_AGENT_KWARGS = {"api_base", "base_url", "model_info"}
_CONTAINER_HOST_REWRITE_AGENTS = {"mini-swe-agent"}
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DEFAULT_CONTAINER_HOST = "192.168.5.1"
_CONTAINER_HOST_ENV = "CON_DRIVER_CONTAINER_HOST"


@dataclass(frozen=True)
class HarborRuntimeResolution:
    forwarded_args: list[str]
    trial_env: dict[str, str]
    gateway_url: str
    vllm_log_enabled: bool
    vllm_log_endpoint: str
    port_profile_id: int | None
    resolved_agent_name: str | None
    resolved_model_name: str | None
    resolved_model_context_window: int | None
    agent_base_url: str | None


def _extract_forwarded_option_values(
    tokens: Sequence[str],
    *,
    option_names: set[str],
) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        option_name = token.split("=", 1)[0]
        if option_name not in option_names:
            index += 1
            continue
        if "=" in token:
            values.append(token.split("=", 1)[1])
            index += 1
            continue
        if index + 1 >= len(tokens):
            raise ValueError(f"Forwarded Harbor arg '{token}' is missing its value")
        values.append(tokens[index + 1])
        index += 2
    return values


def _extract_single_forwarded_option_value(
    tokens: Sequence[str],
    *,
    option_names: set[str],
    display_name: str,
) -> str | None:
    values = _extract_forwarded_option_values(tokens, option_names=option_names)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"Forwarded Harbor args contain multiple {display_name} values")
    value = values[0].strip()
    if not value:
        raise ValueError(f"Forwarded Harbor arg {display_name} cannot be empty")
    return value


def _extract_forwarded_agent_kwarg_keys(tokens: Sequence[str]) -> set[str]:
    keys: set[str] = set()
    for raw_value in _extract_forwarded_option_values(tokens, option_names=_AGENT_KWARG_OPTION_NAMES):
        key, separator, _ = raw_value.partition("=")
        parsed_key = key.strip()
        if not separator or not parsed_key:
            raise ValueError(
                "Forwarded Harbor args contain an invalid --agent-kwarg entry. "
                "Expected key=value."
            )
        keys.add(parsed_key)
    return keys


def _append_agent_kwarg(tokens: list[str], *, key: str, value: str) -> None:
    tokens.extend(["--agent-kwarg", f"{key}={value}"])


def _resolve_forwarded_model_name(*, agent_name: str, served_model_name: str) -> str:
    _ = agent_name
    return f"hosted_vllm/{served_model_name}"


def _container_reachable_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    parsed = urlsplit(base_url)
    host = parsed.hostname
    if host not in _LOOPBACK_HOSTS:
        return base_url
    container_host = os.environ.get(_CONTAINER_HOST_ENV, _DEFAULT_CONTAINER_HOST).strip()
    if not container_host:
        container_host = _DEFAULT_CONTAINER_HOST
    netloc = container_host if parsed.port is None else f"{container_host}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def resolve_harbor_runtime(
    *,
    forwarded_args_from_config: Sequence[str],
    forwarded_args_from_cli: Sequence[str],
    port_profile_id: int | None,
    agent_name: str | None,
    gateway_enabled: bool,
    configured_gateway_url: str | None,
    gateway_timeout_s: float,
    configured_vllm_log: bool | None,
) -> HarborRuntimeResolution:
    resolved_port_profile = load_port_profile(port_profile_id) if port_profile_id is not None else None
    derived_vllm_base_url = (
        f"http://127.0.0.1:{resolved_port_profile.vllm_port}"
        if resolved_port_profile is not None
        else None
    )
    derived_gateway_base_url = (
        f"http://127.0.0.1:{resolved_port_profile.gateway_parse_port}"
        if resolved_port_profile is not None
        else None
    )
    service_base_url = None
    agent_base_url = None
    if resolved_port_profile is not None:
        service_base_url = derived_gateway_base_url if gateway_enabled else derived_vllm_base_url
        agent_base_url = f"{service_base_url}/v1" if service_base_url is not None else None

    gateway_url = configured_gateway_url or derived_gateway_base_url or "http://127.0.0.1:11457"
    vllm_log_enabled = (
        configured_vllm_log
        if configured_vllm_log is not None
        else (True if port_profile_id is not None else False)
    )
    if vllm_log_enabled and resolved_port_profile is None:
        raise ValueError(
            "vLLM metrics logging requires 'port_profile_id' so the endpoint can be "
            "resolved from configs/port_profiles.toml."
        )
    vllm_log_endpoint = (
        f"{derived_vllm_base_url}/metrics" if derived_vllm_base_url is not None else ""
    )

    combined_forwarded_args = list(forwarded_args_from_config) + list(forwarded_args_from_cli)
    forwarded_agent_name = _extract_single_forwarded_option_value(
        combined_forwarded_args,
        option_names=_AGENT_OPTION_NAMES,
        display_name="--agent",
    )
    if agent_name is not None and forwarded_agent_name is not None:
        raise ValueError(
            "Do not pass '--agent' in forwarded Harbor args when using top-level "
            "'agent' or '--agent-name'."
        )

    resolved_agent_name = agent_name or forwarded_agent_name
    if agent_name is not None and port_profile_id is None:
        raise ValueError(
            "Top-level 'agent'/'agent_name' requires 'port_profile_id' so con-driver "
            "can resolve the serving endpoint and model."
        )
    if (
        resolved_agent_name in _CONTAINER_HOST_REWRITE_AGENTS
        and service_base_url is not None
    ):
        reachable_base_url = _container_reachable_base_url(service_base_url)
        agent_base_url = (
            f"{reachable_base_url}/v1" if reachable_base_url is not None else None
        )

    forwarded_model_name = _extract_single_forwarded_option_value(
        combined_forwarded_args,
        option_names=_MODEL_OPTION_NAMES,
        display_name="--model",
    )
    forwarded_agent_kwarg_keys = _extract_forwarded_agent_kwarg_keys(combined_forwarded_args)

    resolved_model_name: str | None = None
    resolved_model_context_window: int | None = None
    trial_env: dict[str, str] = {}
    synthesized_forwarded_args = list(combined_forwarded_args)

    if port_profile_id is not None:
        if resolved_agent_name is None:
            raise ValueError(
                "Missing Harbor agent. Set top-level 'agent'/'agent_name' (recommended) "
                "or pass '--agent' in forwarded Harbor args."
            )
        if forwarded_model_name is not None:
            raise ValueError(
                "Do not pass '--model' in forwarded Harbor args when 'port_profile_id' "
                "is set. con-driver probes the served model and sets '--model' automatically."
            )

        blocked_agent_kwargs = sorted(_AUTO_MANAGED_AGENT_KWARGS & forwarded_agent_kwarg_keys)
        if blocked_agent_kwargs:
            blocked = ", ".join(blocked_agent_kwargs)
            raise ValueError(
                "Do not pass auto-managed '--agent-kwarg' values when 'port_profile_id' "
                f"is set: {blocked}."
            )

        resolved_model_name = probe_single_served_model_name(
            base_url=derived_vllm_base_url or gateway_url,
            timeout_s=gateway_timeout_s,
        )
        resolved_model_spec = load_model_catalog().resolve(resolved_model_name)
        if resolved_model_spec is None:
            raise ValueError(
                "The served model "
                f"'{resolved_model_name}' is not configured in configs/model_config.toml."
            )
        resolved_model_context_window = resolved_model_spec.context_window

        if forwarded_agent_name is None:
            synthesized_forwarded_args.extend(["--agent", resolved_agent_name])
        synthesized_forwarded_args.extend(
            [
                "--model",
                _resolve_forwarded_model_name(
                    agent_name=resolved_agent_name,
                    served_model_name=resolved_model_name,
                ),
            ]
        )
        _append_agent_kwarg(
            synthesized_forwarded_args,
            key="api_base",
            value=agent_base_url or "",
        )
        _append_agent_kwarg(
            synthesized_forwarded_args,
            key="base_url",
            value=agent_base_url or "",
        )
        _append_agent_kwarg(
            synthesized_forwarded_args,
            key="model_info",
            value=build_model_info_json(context_window=resolved_model_context_window),
        )

        agent_defaults = load_agent_defaults(resolved_agent_name)
        existing_agent_kwarg_keys = _extract_forwarded_agent_kwarg_keys(synthesized_forwarded_args)
        for key, raw_value in agent_defaults.agent_kwargs.items():
            if key in existing_agent_kwarg_keys:
                continue
            _append_agent_kwarg(
                synthesized_forwarded_args,
                key=key,
                value=stringify_agent_kwarg_value(raw_value),
            )
        synthesized_forwarded_args.extend(agent_defaults.forwarded_args)
        trial_env = build_endpoint_env(api_base_url=agent_base_url or "")

    return HarborRuntimeResolution(
        forwarded_args=synthesized_forwarded_args,
        trial_env=trial_env,
        gateway_url=gateway_url,
        vllm_log_enabled=vllm_log_enabled,
        vllm_log_endpoint=vllm_log_endpoint,
        port_profile_id=port_profile_id,
        resolved_agent_name=resolved_agent_name,
        resolved_model_name=resolved_model_name,
        resolved_model_context_window=resolved_model_context_window,
        agent_base_url=agent_base_url,
    )
