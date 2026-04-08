#!/usr/bin/env python3
"""Resolve mi3008x embedded launcher model settings from configs/model_config.toml."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_TP1_EXTRA_ARGS = ["--trust-remote-code"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    vllm_model_name: str
    served_model_name: str
    extra_args: list[str]


@dataclass(frozen=True)
class ResolvedModelLaunchConfig:
    resolved_model_key: str
    vllm_model_name: str
    served_model_name: str
    model_extra_args_b64: str


def _clean_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _encode_model_extra_args(extra_args: list[str]) -> str:
    payload = json.dumps(extra_args, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _normalize_tp1_extra_args(extra_args: list[str]) -> list[str]:
    normalized_args: list[str] = []
    skip_next = False

    for index, arg in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue

        normalized = arg.strip().lower()
        if normalized == "--trust-remote-code":
            continue
        if normalized.startswith("--trust-remote-code="):
            continue
        if normalized == "--distributed_executor_backend":
            if index + 1 < len(extra_args):
                skip_next = True
            continue
        if normalized.startswith("--distributed_executor_backend="):
            continue

        normalized_args.append(arg)

    normalized_args.append("--trust-remote-code")
    return normalized_args


def load_model_specs(config_path: Path) -> dict[str, ModelSpec]:
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config file: {config_path}")

    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError(f"model config must define [models.<name>] tables: {config_path}")

    models: dict[str, ModelSpec] = {}
    for key, raw_spec in raw_models.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"models.{key} must be a table in {config_path}")

        vllm_model_name = raw_spec.get("vllm_model_name")
        served_model_name = raw_spec.get("served_model_name")
        extra_args = raw_spec.get("extra_args", [])
        if not isinstance(vllm_model_name, str) or not vllm_model_name.strip():
            raise ValueError(f"models.{key}.vllm_model_name must be a non-empty string")
        if not isinstance(served_model_name, str) or not served_model_name.strip():
            raise ValueError(f"models.{key}.served_model_name must be a non-empty string")
        if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
            raise ValueError(f"models.{key}.extra_args must be a string array")

        models[key] = ModelSpec(
            key=key,
            vllm_model_name=vllm_model_name.strip(),
            served_model_name=served_model_name.strip(),
            extra_args=list(extra_args),
        )

    return models


def resolve_model_spec(specs: dict[str, ModelSpec], selector: str) -> ModelSpec | None:
    selected = selector.strip()
    if not selected:
        raise ValueError("model selector must be non-empty")
    if selected in specs:
        return specs[selected]

    for spec in specs.values():
        aliases = {spec.key, spec.vllm_model_name, spec.served_model_name}
        if selected in aliases:
            return spec
    return None


def resolve_model_launch_config(
    *,
    config_path: Path,
    selector: str,
    served_model_name: str | None = None,
    extra_args_b64: str | None = None,
) -> ResolvedModelLaunchConfig:
    cleaned_selector = selector.strip()
    if not cleaned_selector:
        raise ValueError("model selector must be non-empty")

    cleaned_served_model_name = _clean_optional_str(served_model_name)
    cleaned_extra_args_b64 = _clean_optional_str(extra_args_b64)
    specs = load_model_specs(config_path)
    spec = resolve_model_spec(specs, cleaned_selector)

    if spec is None:
        if cleaned_served_model_name is None:
            raise ValueError(
                "model selector "
                f"{cleaned_selector!r} was not found in {config_path}; "
                "set VLLM_MODEL_KEY to a config key or also set VLLM_SERVED_MODEL_NAME"
            )

        return ResolvedModelLaunchConfig(
            resolved_model_key="",
            vllm_model_name=cleaned_selector,
            served_model_name=cleaned_served_model_name,
            model_extra_args_b64=cleaned_extra_args_b64
            or _encode_model_extra_args(DEFAULT_TP1_EXTRA_ARGS),
        )

    return ResolvedModelLaunchConfig(
        resolved_model_key=spec.key,
        vllm_model_name=spec.vllm_model_name,
        served_model_name=cleaned_served_model_name or spec.served_model_name,
        model_extra_args_b64=cleaned_extra_args_b64
        or _encode_model_extra_args(_normalize_tp1_extra_args(spec.extra_args)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a model key, alias, or raw model name into the launch values "
            "used by the mi3008x embedded TP1 launcher."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/model_config.toml.")
    parser.add_argument("--selector", required=True, help="Model key, alias, or raw model name.")
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Optional explicit served model name override.",
    )
    parser.add_argument(
        "--extra-args-b64",
        default=None,
        help="Optional explicit VLLM_MODEL_EXTRA_ARGS_B64 override.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    resolved = resolve_model_launch_config(
        config_path=args.config.expanduser().resolve(),
        selector=args.selector,
        served_model_name=args.served_model_name,
        extra_args_b64=args.extra_args_b64,
    )
    print(resolved.resolved_model_key)
    print(resolved.vllm_model_name)
    print(resolved.served_model_name)
    print(resolved.model_extra_args_b64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
