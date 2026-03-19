"""Tests for gateway_lite."""
from __future__ import annotations

import asyncio
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway_lite.app import (
    ForwardResult,
    GatewayConfig,
    GatewayService,
    create_app,
    create_gateway_service,
)
from gateway_lite.port_profiles import load_port_profile
from gateway_lite.runtime_config import GatewayRuntimeSettings


def build_client(
    *,
    artifact_compression: str = "none",
    base_url: str = "http://testserver",
    response_json_factory: Any | None = None,
) -> TestClient:
    async def fake_forward(
        method: str, path: str, headers: dict[str, str], body: bytes
    ) -> ForwardResult:
        request_json = json.loads(body.decode("utf-8")) if body else {}
        if callable(response_json_factory):
            response_json = response_json_factory(request_json)
        else:
            response_json = {
                "id": "cmpl-test",
                "object": "text_completion",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "choices": [{"text": "hello"}],
                "echo_model": request_json.get("model"),
            }
        return ForwardResult(
            status_code=200,
            content=json.dumps(response_json).encode("utf-8"),
            content_type="application/json",
            response_json=response_json,
        )

    service = create_gateway_service(
        config=GatewayConfig(
            vllm_base_url="http://vllm:11451",
            artifact_compression=artifact_compression,
        ),
        forwarder=fake_forward,
    )
    app = create_app(
        service=service,
        gateway_parse_port=18171,
    )
    return TestClient(app, base_url=base_url)


def load_artifact_json(artifact_path: Path, member_name: str) -> dict:
    if artifact_path.is_dir():
        return json.loads((artifact_path / member_name).read_text(encoding="utf-8"))
    with tarfile.open(artifact_path, "r:gz") as archive:
        member = archive.extractfile(member_name)
        assert member is not None
        return json.loads(member.read().decode("utf-8"))


def load_artifact_jsonl(artifact_path: Path, member_name: str) -> list[dict]:
    if artifact_path.is_dir():
        lines = (artifact_path / member_name).read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    with tarfile.open(artifact_path, "r:gz") as archive:
        member = archive.extractfile(member_name)
        assert member is not None
        lines = member.read().decode("utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_gateway_writes_artifact_on_job_end(tmp_path: Path) -> None:
    client = build_client()
    token = "agent-token-1"
    output_dir = tmp_path / "artifacts"

    response = client.post("/job/start", json={"output_location": str(output_dir)})
    assert response.status_code == 200

    response = client.post("/agent/start", json={"api_token": token})
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token},
        json={"model": "Qwen3-Coder-30B-A3B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["echo_model"] == "Qwen3-Coder-30B-A3B-Instruct"

    response = client.post("/agent/end", json={"api_token": token, "return_code": 0})
    assert response.status_code == 200

    response = client.post("/job/end", json={"status": "completed"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_count"] == 1
    artifact_path = Path(payload["artifacts"][0]["path"])
    assert artifact_path.exists()
    assert payload["artifacts"][0]["artifact_format"] == "none"
    assert artifact_path.is_dir()

    manifest = load_artifact_json(artifact_path, "manifest.json")
    assert manifest["run_id"] == run_id
    assert manifest["request_count"] == 1
    assert manifest["return_code"] == 0
    assert manifest["schema_version"] == "v1-lite"

    request_records = load_artifact_jsonl(artifact_path, "requests/model_inference.jsonl")
    assert len(request_records) == 1
    assert request_records[0]["request"]["model"] == "Qwen3-Coder-30B-A3B-Instruct"
    assert request_records[0]["response"]["id"] == "cmpl-test"
    assert request_records[0]["response"]["choices"][0]["text"] == "hello"
    assert request_records[0]["request_start_time"].endswith("Z")
    assert request_records[0]["request_end_time"].endswith("Z")
    assert request_records[0]["request_duration_ms"] >= 0

    lifecycle_records = load_artifact_jsonl(artifact_path, "events/lifecycle.jsonl")
    event_types = [record["event_type"] for record in lifecycle_records]
    assert event_types == ["job_start", "agent_start", "agent_end", "job_end"]


def test_gateway_writes_compressed_artifact_when_enabled(tmp_path: Path) -> None:
    client = build_client(artifact_compression="tar.gz")
    token = "agent-token-zip"
    output_dir = tmp_path / "artifacts"

    response = client.post("/job/start", json={"output_location": str(output_dir)})
    assert response.status_code == 200

    response = client.post("/agent/start", json={"api_token": token})
    assert response.status_code == 200

    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token},
        json={"model": "Qwen3-Coder-30B-A3B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200

    response = client.post("/agent/end", json={"api_token": token, "return_code": 0})
    assert response.status_code == 200

    response = client.post("/job/end", json={"status": "completed"})
    assert response.status_code == 200
    payload = response.json()
    artifact_path = Path(payload["artifacts"][0]["path"])
    assert artifact_path.exists()
    assert payload["artifacts"][0]["artifact_format"] == "tar.gz"
    assert artifact_path.suffixes[-2:] == [".tar", ".gz"]

    manifest = load_artifact_json(artifact_path, "manifest.json")
    assert manifest["return_code"] == 0


def test_agent_start_requires_active_job() -> None:
    client = build_client()
    response = client.post("/agent/start", json={"api_token": "token"})
    assert response.status_code == 400
    assert "job is not active" in response.json()["detail"]


def test_job_end_rejects_active_agents(tmp_path: Path) -> None:
    client = build_client()
    token = "agent-token-2"

    response = client.post("/job/start", json={"output_location": str(tmp_path / "jobout")})
    assert response.status_code == 200

    response = client.post("/agent/start", json={"api_token": token})
    assert response.status_code == 200

    response = client.post("/job/end", json={"status": "completed"})
    assert response.status_code == 409
    assert "active" in response.json()["detail"]


def test_job_start_rejects_when_job_active(tmp_path: Path) -> None:
    client = build_client()
    first_output = tmp_path / "job-a"
    second_output = tmp_path / "job-b"

    response = client.post("/job/start", json={"output_location": str(first_output)})
    assert response.status_code == 200

    response = client.post("/job/start", json={"output_location": str(second_output)})
    assert response.status_code == 409
    assert "job already active" in response.json()["detail"]


def test_gateway_rejects_token_mismatch_with_single_active_agent(tmp_path: Path) -> None:
    client = build_client()
    token = "agent-token-1"
    output_dir = tmp_path / "artifacts"

    response = client.post("/job/start", json={"output_location": str(output_dir)})
    assert response.status_code == 200

    response = client.post("/agent/start", json={"api_token": token})
    assert response.status_code == 200

    response = client.post(
        "/v1/chat/completions",
        headers={"authorization": "Bearer different-token"},
        json={"model": "Qwen3-Coder-30B-A3B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert "agent not started" in response.json()["detail"]


def test_gateway_rejects_token_mismatch_with_multiple_active_agents(tmp_path: Path) -> None:
    client = build_client()
    output_dir = tmp_path / "artifacts"

    response = client.post("/job/start", json={"output_location": str(output_dir)})
    assert response.status_code == 200

    response = client.post("/agent/start", json={"api_token": "agent-token-1"})
    assert response.status_code == 200
    response = client.post("/agent/start", json={"api_token": "agent-token-2"})
    assert response.status_code == 200

    response = client.post(
        "/v1/chat/completions",
        headers={"authorization": "Bearer different-token"},
        json={"model": "Qwen3-Coder-30B-A3B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert "agent not started" in response.json()["detail"]


def test_gateway_parse_port_preserves_raw_port_response(tmp_path: Path) -> None:
    def response_json_factory(request_json: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": request_json["model"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>draft plan</think>final answer",
                        "refusal": None,
                        "annotations": None,
                        "audio": None,
                        "function_call": None,
                        "tool_calls": [],
                        "reasoning": None,
                        "reasoning_content": None,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    client = build_client(response_json_factory=response_json_factory)
    token = "agent-token-raw"

    assert client.post("/job/start", json={"output_location": str(tmp_path / "rawout")}).status_code == 200
    assert client.post("/agent/start", json={"api_token": token}).status_code == 200

    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token},
        json={"model": "Kimi-K2.5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    message = payload["choices"][0]["message"]
    assert message["content"] == "<think>draft plan</think>final answer"
    assert message["reasoning"] is None
    assert message["reasoning_content"] is None


def test_gateway_parse_port_same_as_regular_port(tmp_path: Path) -> None:
    """Both ports should behave identically - no reasoning extraction."""
    def response_json_factory(request_json: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": request_json["model"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>draft plan</think>final answer",
                        "refusal": None,
                        "annotations": None,
                        "audio": None,
                        "function_call": None,
                        "tool_calls": [],
                        "reasoning": None,
                        "reasoning_content": None,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    client = build_client(
        base_url="http://testserver:18171",
        response_json_factory=response_json_factory,
    )
    token = "agent-token-parse"

    assert client.post("/job/start", json={"output_location": str(tmp_path / "parseout")}).status_code == 200
    assert client.post("/agent/start", json={"api_token": token}).status_code == 200

    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token},
        json={"model": "Kimi-K2.5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    message = payload["choices"][0]["message"]
    # Content should remain unchanged - no transformation
    assert message["content"] == "<think>draft plan</think>final answer"
    assert message["reasoning"] is None
    assert message["reasoning_content"] is None


def test_gateway_forward_to_vllm_uses_no_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200
        content = b'{"ok":true}'
        headers = {"content-type": "application/json"}

        @staticmethod
        def json() -> dict[str, bool]:
            return {"ok": True}

    class FakeAsyncClient:
        def __init__(self, *, timeout: Any) -> None:
            observed["timeout"] = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def request(
            self,
            *,
            method: str,
            url: str,
            headers: dict[str, str],
            content: bytes,
        ) -> FakeResponse:
            observed["method"] = method
            observed["url"] = url
            observed["headers"] = headers
            observed["content"] = content
            return FakeResponse()

    monkeypatch.setattr("gateway_lite.app.httpx.AsyncClient", FakeAsyncClient)

    service = GatewayService(
        config=GatewayConfig(
            vllm_base_url="http://vllm:11451",
        ),
    )

    result = asyncio.run(
        service._forward_to_vllm(
            "POST",
            "v1/chat/completions",
            {"content-type": "application/json"},
            b'{"hello":"world"}',
        )
    )

    assert observed["timeout"] is None
    assert observed["method"] == "POST"
    assert observed["url"] == "http://vllm:11451/v1/chat/completions"
    assert result.status_code == 200
    assert result.response_json == {"ok": True}


def test_gateway_forward_to_vllm_avoids_duplicate_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200
        content = b"{}"
        headers = {"content-type": "application/json"}

        @staticmethod
        def json() -> dict[str, Any]:
            return {}

    class FakeAsyncClient:
        def __init__(self, *, timeout: Any) -> None:
            observed["timeout"] = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def request(
            self,
            *,
            method: str,
            url: str,
            headers: dict[str, str],
            content: bytes,
        ) -> FakeResponse:
            observed["method"] = method
            observed["url"] = url
            observed["headers"] = headers
            observed["content"] = content
            return FakeResponse()

    monkeypatch.setattr("gateway_lite.app.httpx.AsyncClient", FakeAsyncClient)

    service = GatewayService(
        config=GatewayConfig(
            vllm_base_url="https://api.openai.com/v1",
        ),
    )

    result = asyncio.run(
        service._forward_to_vllm(
            "POST",
            "v1/chat/completions?foo=bar",
            {"content-type": "application/json"},
            b'{"hello":"world"}',
        )
    )

    assert observed["timeout"] is None
    assert observed["method"] == "POST"
    assert observed["url"] == "https://api.openai.com/v1/chat/completions?foo=bar"
    assert result.status_code == 200
    assert result.response_json == {}


def test_proxy_request_cancels_upstream_on_client_disconnect(tmp_path: Path) -> None:
    observed: dict[str, Any] = {"cancelled": False}
    forward_started = asyncio.Event()

    async def fake_forward(
        method: str, path: str, headers: dict[str, str], body: bytes
    ) -> ForwardResult:
        forward_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            observed["cancelled"] = True
            raise

    async def disconnect_waiter() -> None:
        await forward_started.wait()

    service = GatewayService(
        config=GatewayConfig(
            vllm_base_url="http://vllm:11451",
        ),
        forwarder=fake_forward,
    )
    token = "agent-token-disconnect"
    service.start_job(str(tmp_path / "jobout"))
    service.start_agent(token)

    result = asyncio.run(
        service.proxy_request(
            api_token=token,
            method="POST",
            path="v1/chat/completions",
            headers={"content-type": "application/json"},
            body=b'{"model":"Qwen3-Coder-30B-A3B-Instruct"}',
            disconnect_waiter=disconnect_waiter,
        )
    )

    assert observed["cancelled"] is True
    assert result.status_code == 499
    assert result.response_json == {
        "error": "client_disconnected",
        "detail": "downstream client disconnected before the upstream response completed",
    }
    run = service.active_runs[token]
    assert len(run.request_records) == 1
    assert run.request_records[0]["status_code"] == 499
    assert run.request_records[0]["response_summary"]["error"] == "client_disconnected"


def test_proxy_request_cancels_upstream_when_handler_is_cancelled(tmp_path: Path) -> None:
    observed: dict[str, Any] = {"cancelled": False}
    forward_started = asyncio.Event()

    async def fake_forward(
        method: str, path: str, headers: dict[str, str], body: bytes
    ) -> ForwardResult:
        forward_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            observed["cancelled"] = True
            raise

    async def run_case() -> None:
        service = GatewayService(
            config=GatewayConfig(
                vllm_base_url="http://vllm:11451",
            ),
            forwarder=fake_forward,
        )
        token = "agent-token-cancel"
        service.start_job(str(tmp_path / "jobout"))
        service.start_agent(token)

        proxy_task = asyncio.create_task(
            service.proxy_request(
                api_token=token,
                method="POST",
                path="v1/chat/completions",
                headers={"content-type": "application/json"},
                body=b'{"model":"Qwen3-Coder-30B-A3B-Instruct"}',
            )
        )
        await forward_started.wait()
        proxy_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await proxy_task

    asyncio.run(run_case())
    assert observed["cancelled"] is True


def test_gateway_config_uses_profile_ports() -> None:
    cfg = GatewayConfig.from_port_profile(1)
    assert cfg.vllm_base_url == "http://localhost:24123"


def test_gateway_config_uses_default_profile_when_unspecified() -> None:
    cfg = GatewayConfig.from_port_profile()
    profile = load_port_profile()
    assert cfg.vllm_base_url == f"http://localhost:{profile.vllm_port}"


def test_port_profile_exposes_gateway_parse_port() -> None:
    profile = load_port_profile(1)
    assert profile.gateway_port == 24157
    assert profile.gateway_parse_port == 28171


def test_gateway_config_reads_runtime_settings() -> None:
    settings = GatewayRuntimeSettings(
        artifact_compression="tar.gz",
        service_name="gateway-custom",
        upstream_api_key="sk-runtime",
    )

    cfg = GatewayConfig.from_port_profile(0, runtime_settings=settings)
    assert cfg.artifact_compression == "tar.gz"
    assert cfg.service_name == "gateway-custom"
    assert cfg.upstream_api_key == "sk-runtime"


def test_gateway_config_rejects_unknown_port_profile() -> None:
    with pytest.raises(ValueError, match="unknown port profile id"):
        GatewayConfig.from_port_profile(999)


def test_runtime_settings_load_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "gateway_lite.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[run]",
                "port_profile_id = 2",
                'output_root = "./artifacts"',
                "",
                "[gateway]",
                'artifact_compression = "tar.gz"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings = GatewayRuntimeSettings(
        port_profile_id=2,
        output_root="./artifacts",
        artifact_compression="tar.gz",
    )
    assert settings.port_profile_id == 2
    assert settings.output_root == "./artifacts"
    assert settings.artifact_compression == "tar.gz"


# ============================================================================
# Upstream Key Injection Tests
# ============================================================================

def build_client_with_upstream_key(
    *,
    upstream_api_key: str,
    base_url: str = "http://testserver",
) -> TestClient:
    captured_headers: dict[str, str] = {}

    async def fake_forward(
        method: str, path: str, headers: dict[str, str], body: bytes
    ) -> ForwardResult:
        captured_headers.update(headers)
        response_json = {"ok": True}
        return ForwardResult(
            status_code=200,
            content=json.dumps(response_json).encode("utf-8"),
            content_type="application/json",
            response_json=response_json,
        )

    service = create_gateway_service(
        config=GatewayConfig(
            vllm_base_url="http://vllm:11451",
            upstream_api_key=upstream_api_key,
        ),
        forwarder=fake_forward,
    )
    app = create_app(
        service=service,
        gateway_parse_port=18171,
    )
    app.state.captured_headers = captured_headers
    return TestClient(app, base_url=base_url)


def test_extract_api_token_parsing() -> None:
    """Test the extract_api_token function directly."""
    from gateway_lite.app import extract_api_token

    assert extract_api_token({"Authorization": "Bearer my-key"}) == "my-key"
    assert extract_api_token({"x-api-key": "my-key"}) == "my-key"
    assert extract_api_token({"api-key": "my-key"}) == "my-key"
    assert extract_api_token({"openai-api-key": "my-key"}) == "my-key"


def test_upstream_api_key_override_rewrites_plain_agent_token(tmp_path: Path) -> None:
    client = build_client_with_upstream_key(
        upstream_api_key="sk-real-openai-key",
    )
    output_dir = tmp_path / "artifacts"
    agent_id = "condrv-agent-1"

    assert client.post("/job/start", json={"output_location": str(output_dir)}).status_code == 200
    assert client.post("/agent/start", json={"api_token": agent_id}).status_code == 200

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {agent_id}"},
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert client.app.state.captured_headers.get("authorization") == "Bearer sk-real-openai-key"
