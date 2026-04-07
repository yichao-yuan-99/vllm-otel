from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"


@dataclass(frozen=True)
class GatewayModelSpec:
    key: str
    vllm_model_name: str
    served_model_name: str
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None


class ModelRegistry:
    def __init__(
        self,
        *,
        default_model: str | None,
        models: dict[str, GatewayModelSpec],
    ) -> None:
        self.default_model = default_model
        self.models = models
        self._aliases: dict[str, GatewayModelSpec] = {}
        for spec in models.values():
            for alias in {spec.key, spec.vllm_model_name, spec.served_model_name}:
                cleaned = alias.strip()
                if cleaned:
                    self._aliases[cleaned] = spec

    def resolve(self, model_name: str | None) -> GatewayModelSpec | None:
        if model_name is None:
            return None
        return self._aliases.get(model_name.strip())

    def resolve_reasoning_parser(self, model_name: str | None) -> str | None:
        spec = self.resolve(model_name)
        if spec is None:
            return None
        return spec.reasoning_parser


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _parse_optional_str(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    stripped = value.strip()
    return stripped or None


def load_model_registry(config_path: Path | None = None) -> ModelRegistry:
    path = config_path or MODEL_CONFIG_PATH
    payload = _load_toml(path)

    raw_models = payload.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError("configs/model_config.toml must include [models]")

    models: dict[str, GatewayModelSpec] = {}
    for key, raw in raw_models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"models.{key} must be a table")

        vllm_model_name = _parse_optional_str(
            raw.get("vllm_model_name"),
            f"models.{key}.vllm_model_name",
        )
        served_model_name = _parse_optional_str(
            raw.get("served_model_name"),
            f"models.{key}.served_model_name",
        )
        if not vllm_model_name:
            raise ValueError(f"models.{key}.vllm_model_name must be a non-empty string")
        if not served_model_name:
            raise ValueError(f"models.{key}.served_model_name must be a non-empty string")

        models[key] = GatewayModelSpec(
            key=key,
            vllm_model_name=vllm_model_name,
            served_model_name=served_model_name,
            reasoning_parser=_parse_optional_str(
                raw.get("reasoning_parser"),
                f"models.{key}.reasoning_parser",
            ),
            tool_call_parser=_parse_optional_str(
                raw.get("tool_call_parser"),
                f"models.{key}.tool_call_parser",
            ),
        )

    default_model = _parse_optional_str(payload.get("default_model"), "default_model")
    if default_model and default_model not in models:
        raise ValueError(f"default_model references unknown model key: {default_model}")

    return ModelRegistry(default_model=default_model, models=models)
