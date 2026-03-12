from __future__ import annotations

from types import SimpleNamespace

from con_driver import gateway_wrapper


def test_run_with_gateway_preserves_existing_resolved_api_base_for_mini_swe_agent(monkeypatch) -> None:
    post_calls: list[tuple[str, dict[str, object]]] = []
    captured_env: dict[str, str] = {}

    def fake_post_json(*, endpoint: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
        _ = timeout_s
        post_calls.append((endpoint, payload))
        return {"ok": True}

    def fake_run(command: list[str], check: bool, env: dict[str, str]) -> SimpleNamespace:
        _ = check
        assert command == ["harbor", "run"]
        captured_env.update(env)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gateway_wrapper, "_post_json", fake_post_json)
    monkeypatch.setattr(gateway_wrapper.subprocess, "run", fake_run)

    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "http://192.168.5.1:28171/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "http://192.168.5.1:28171/v1")
    monkeypatch.setenv("MSWEA_API_KEY", "dummy")

    exit_code = gateway_wrapper.run_with_gateway(
        gateway_url="http://127.0.0.1:28171",
        api_token="token-123",
        timeout_s=30.0,
        agent_name="mini-swe-agent",
        command=["harbor", "run"],
    )

    assert exit_code == 0
    assert captured_env["HOSTED_VLLM_API_BASE"] == "http://192.168.5.1:28171/v1"
    assert captured_env["OPENAI_API_BASE"] == "http://192.168.5.1:28171/v1"
    assert captured_env["OPENAI_API_KEY"] == "token-123"
    assert captured_env["LITELLM_API_KEY"] == "token-123"
    assert captured_env["HOSTED_VLLM_API_KEY"] == "token-123"
    assert "MSWEA_API_KEY" not in captured_env
    assert post_calls[0][0] == "http://127.0.0.1:28171/agent/start"
    assert post_calls[1][0] == "http://127.0.0.1:28171/agent/end"


def test_run_with_gateway_sets_api_base_when_missing_for_mini_swe_agent(monkeypatch) -> None:
    captured_env: dict[str, str] = {}

    def fake_post_json(*, endpoint: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
        _ = endpoint, payload, timeout_s
        return {"ok": True}

    def fake_run(command: list[str], check: bool, env: dict[str, str]) -> SimpleNamespace:
        _ = command, check
        captured_env.update(env)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gateway_wrapper, "_post_json", fake_post_json)
    monkeypatch.setattr(gateway_wrapper.subprocess, "run", fake_run)

    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("MSWEA_API_KEY", raising=False)

    exit_code = gateway_wrapper.run_with_gateway(
        gateway_url="http://127.0.0.1:18171",
        api_token="token-abc",
        timeout_s=30.0,
        agent_name="mini-swe-agent",
        command=["harbor", "run"],
    )

    assert exit_code == 0
    assert captured_env["HOSTED_VLLM_API_BASE"] == "http://127.0.0.1:18171/v1"
    assert captured_env["OPENAI_API_BASE"] == "http://127.0.0.1:18171/v1"
    assert captured_env["OPENAI_API_KEY"] == "token-abc"
    assert captured_env["LITELLM_API_KEY"] == "token-abc"
    assert captured_env["HOSTED_VLLM_API_KEY"] == "token-abc"


def test_run_with_gateway_for_non_mini_agent_preserves_existing_bases(monkeypatch) -> None:
    captured_env: dict[str, str] = {}

    def fake_post_json(*, endpoint: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
        _ = endpoint, payload, timeout_s
        return {"ok": True}

    def fake_run(command: list[str], check: bool, env: dict[str, str]) -> SimpleNamespace:
        _ = command, check
        captured_env.update(env)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gateway_wrapper, "_post_json", fake_post_json)
    monkeypatch.setattr(gateway_wrapper.subprocess, "run", fake_run)

    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "http://192.168.5.1:28171/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "http://192.168.5.1:28171/v1")
    monkeypatch.setenv("MSWEA_API_KEY", "dummy")

    exit_code = gateway_wrapper.run_with_gateway(
        gateway_url="http://127.0.0.1:28171",
        api_token="token-123",
        timeout_s=30.0,
        agent_name="terminus-2",
        command=["harbor", "run"],
    )

    assert exit_code == 0
    assert captured_env["HOSTED_VLLM_API_BASE"] == "http://192.168.5.1:28171/v1"
    assert captured_env["OPENAI_API_BASE"] == "http://192.168.5.1:28171/v1"
    assert captured_env["MSWEA_API_KEY"] == "dummy"
