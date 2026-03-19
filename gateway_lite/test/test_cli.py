"""Tests for gateway_lite CLI behavior."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import pytest

from gateway_lite.port_profiles import PortProfile
from gateway_lite.runtime_config import GatewayRuntimeSettings


def _run_cmd_run_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    upstream_base_url: str | None,
    upstream_api_key: str | None = None,
    env_upstream_base_url: str | None = None,
    env_upstream_api_key: str | None = None,
) -> dict[str, Any]:
    from gateway_lite import cli

    captured: dict[str, Any] = {}
    settings = GatewayRuntimeSettings(
        port_profile_id=3,
    )
    profile = PortProfile(
        profile_id="3",
        label="test-profile",
        vllm_port=11451,
        jaeger_api_port=16686,
        jaeger_otlp_port=4317,
        gateway_port=11457,
        gateway_parse_port=18171,
        lmcache_port=29411,
    )

    monkeypatch.setattr(cli, "setup_logging", lambda verbose: None)
    monkeypatch.setattr(cli, "load_runtime_settings", lambda: settings)
    monkeypatch.setattr(cli, "load_port_profile", lambda profile_id: profile)

    def fake_create_gateway_service(*, config):
        captured["config"] = config
        return object()

    monkeypatch.setattr(cli, "create_gateway_service", fake_create_gateway_service)
    monkeypatch.setattr(cli, "create_app", lambda service, gateway_parse_port: object())

    async def fake_serve_gateway(app_instance, host: str, ports: list[int]) -> None:
        captured["host"] = host
        captured["ports"] = list(ports)

    monkeypatch.setattr(cli, "_serve_gateway", fake_serve_gateway)
    def fake_asyncio_run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(cli.asyncio, "run", fake_asyncio_run)

    if env_upstream_base_url is None:
        monkeypatch.delenv("GATEWAY_LITE_VLLM_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("GATEWAY_LITE_VLLM_BASE_URL", env_upstream_base_url)
    if env_upstream_api_key is None:
        monkeypatch.delenv("GATEWAY_LITE_UPSTREAM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GATEWAY_LITE_UPSTREAM_API_KEY", env_upstream_api_key)

    args = argparse.Namespace(
        verbose=False,
        host="127.0.0.1",
        port_profile_id=None,
        upstream_base_url=upstream_base_url,
        upstream_api_key=upstream_api_key,
    )
    exit_code = cli.cmd_run(args)
    assert exit_code == 0
    return captured


def test_cmd_run_prefers_cli_upstream_url_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _run_cmd_run_with_fakes(
        monkeypatch,
        upstream_base_url="https://api.openai.com/v1",
        env_upstream_base_url="https://example.invalid/v1",
    )
    config = captured["config"]
    assert config.vllm_base_url == "https://api.openai.com/v1"


def test_cmd_run_prefers_cli_upstream_api_key_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _run_cmd_run_with_fakes(
        monkeypatch,
        upstream_base_url=None,
        upstream_api_key="sk-cli",
        env_upstream_api_key="sk-env",
    )
    config = captured["config"]
    assert config.upstream_api_key == "sk-cli"
