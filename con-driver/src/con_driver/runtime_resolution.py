"""Helpers for resolving port-profile-backed runtime settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
PORT_PROFILES_PATH = REPO_ROOT / "configs" / "port_profiles.toml"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"
AGENT_DEFAULTS_PATH = REPO_ROOT / "con-driver" / "configs" / "agent_defaults.toml"
DEFAULT_PROBE_TIMEOUT_S = 5.0

_ENDPOINT_ENV_VARS = (
    "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL",
    "LLM_BASE_URL",
    "BASE_URL",
    "OPENAI_API_BASE",
    "HOSTED_VLLM_API_BASE",
    "VLLM_API_BASE",
)


@dataclass(frozen=True)
class PortProfile:
    profile_id: str
    label: str | None
    vllm_port: int
    jaeger_api_port: int
    jaeger_otlp_port: int
    gateway_port: int
    gateway_parse_port: int


@dataclass(frozen=True)
class ModelSpec:
    key: str
    vllm_model_name: str
    served_model_name: str
    context_window: int


@dataclass(frozen=True)
class AgentDefaults:
    forwarded_args: list[str]
    agent_kwargs: dict[str, object]


class ModelCatalog:
    def __init__(self, models: dict[str, ModelSpec]) -> None:
        self.models = models
        self._aliases: dict[str, ModelSpec] = {}
        for spec in models.values():
            for alias in {spec.key, spec.vllm_model_name, spec.served_model_name}:
                cleaned = alias.strip()
                if cleaned:
                    self._aliases[cleaned] = spec

    def resolve(self, model_name: str) -> ModelSpec | None:
        return self._aliases.get(model_name.strip())


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


def _parse_required_str(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _parse_positive_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _parse_string_list(value: object, key: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of strings")
    parsed: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{key}[{index}] must be a string")
        parsed.append(item)
    return parsed


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
    )


def load_model_catalog(config_path: Path | None = None) -> ModelCatalog:
    path = config_path or MODEL_CONFIG_PATH
    payload = _load_toml(path)
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError("configs/model_config.toml must include [models]")

    models: dict[str, ModelSpec] = {}
    for key, raw in raw_models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"models.{key} must be a table")
        models[key] = ModelSpec(
            key=key,
            vllm_model_name=_parse_required_str(
                raw.get("vllm_model_name"),
                f"models.{key}.vllm_model_name",
            ),
            served_model_name=_parse_required_str(
                raw.get("served_model_name"),
                f"models.{key}.served_model_name",
            ),
            context_window=_parse_positive_int(
                raw.get("context_window"),
                f"models.{key}.context_window",
            ),
        )

    return ModelCatalog(models)


def load_agent_defaults(
    agent_name: str,
    *,
    config_path: Path | None = None,
) -> AgentDefaults:
    path = config_path or AGENT_DEFAULTS_PATH
    if not path.exists():
        return AgentDefaults(forwarded_args=[], agent_kwargs={})

    payload = _load_toml(path)
    raw_agents = payload.get("agents")
    if not isinstance(raw_agents, dict):
        return AgentDefaults(forwarded_args=[], agent_kwargs={})

    raw_agent = raw_agents.get(agent_name)
    if not isinstance(raw_agent, dict):
        return AgentDefaults(forwarded_args=[], agent_kwargs={})

    raw_agent_kwargs = raw_agent.get("agent_kwargs", {})
    if raw_agent_kwargs is None:
        parsed_agent_kwargs: dict[str, object] = {}
    elif isinstance(raw_agent_kwargs, dict):
        parsed_agent_kwargs = dict(raw_agent_kwargs)
    else:
        raise ValueError(f"agents.{agent_name}.agent_kwargs must be a table")

    return AgentDefaults(
        forwarded_args=_parse_string_list(
            raw_agent.get("forwarded_args"),
            f"agents.{agent_name}.forwarded_args",
        ),
        agent_kwargs=parsed_agent_kwargs,
    )


def probe_single_served_model_name(
    *,
    base_url: str,
    timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
) -> str:
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        response = requests.get(url, timeout=timeout_s)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to query serving endpoint {url}: {exc}") from exc

    if response.status_code >= 300:
        raise RuntimeError(
            "Serving endpoint probe failed with non-success "
            f"{response.status_code}: {response.text}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Serving endpoint {url} did not return JSON") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Serving endpoint {url} returned invalid JSON payload")
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise RuntimeError(f"Serving endpoint {url} did not return a model list")

    model_ids: list[str] = []
    for index, item in enumerate(raw_models):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Serving endpoint {url} returned invalid model record at index {index}"
            )
        raw_id = item.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise RuntimeError(
                f"Serving endpoint {url} returned invalid model id at index {index}"
            )
        model_ids.append(raw_id.strip())

    if not model_ids:
        raise RuntimeError(f"Serving endpoint {url} returned no models")
    if len(model_ids) != 1:
        raise RuntimeError(
            "Expected exactly one served model from "
            f"{url}, found {len(model_ids)}: {model_ids}"
        )
    return model_ids[0]


def stringify_agent_kwarg_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def build_model_info_json(*, context_window: int) -> str:
    return json.dumps(
        {
            "max_input_tokens": context_window,
            "max_output_tokens": context_window,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )


def build_endpoint_env(*, api_base_url: str) -> dict[str, str]:
    return {name: api_base_url for name in _ENDPOINT_ENV_VARS}
