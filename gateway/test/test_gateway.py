from __future__ import annotations

import asyncio
import json
import socket
import stat
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.app import (
    ForwardResult,
    GatewayConfig,
    GatewayService,
    create_app,
    create_gateway_service,
    sha256_hex,
)
from gateway.cli import (
    _create_unix_listen_socket,
    _default_ipc_socket_path,
    _resolve_ipc_socket_path,
    resolve_listener_ports,
)
from gateway.model_configs import load_model_registry
from gateway.port_profiles import load_port_profile
from gateway.runtime_config import GatewayRuntimeSettings, load_runtime_settings


def build_client(
    *,
    artifact_compression: str = "none",
    job_end_trace_wait_seconds: float = 0.0,
    base_url: str = "http://testserver",
    response_json_factory: Any | None = None,
    forwarder: Any | None = None,
) -> TestClient:
    captured: dict[str, str | None] = {"traceparent": None}

    async def fake_forward(
        method: str, path: str, headers: dict[str, str], body: bytes
    ) -> ForwardResult:
        captured["traceparent"] = headers.get("traceparent") or headers.get("Traceparent")
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

    def fake_jaeger_fetch(trace_id: str) -> dict:
        return {"data": [{"traceID": trace_id}]}

    service = create_gateway_service(
        config=GatewayConfig(
            vllm_base_url="http://vllm:11451",
            jaeger_api_base_url="http://jaeger:16686/api/traces",
            otlp_traces_endpoint="grpc://jaeger:4317",
            artifact_compression=artifact_compression,
            job_end_trace_wait_seconds=job_end_trace_wait_seconds,
        ),
        forwarder=forwarder or fake_forward,
        jaeger_fetcher=fake_jaeger_fetch,
        service_name="gateway-test",
    )
    app = create_app(
        service=service,
        gateway_parse_port=18171,
        model_registry=load_model_registry(),
    )
    app.state.captured_headers = captured
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
    trace_id = response.json()["trace_id"]

    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token},
        json={"model": "Qwen3-Coder-30B-A3B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["echo_model"] == "Qwen3-Coder-30B-A3B-Instruct"
    assert client.app.state.captured_headers["traceparent"]

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
    assert manifest["trace_id"] == trace_id
    assert manifest["request_count"] == 1
    assert manifest["model_inference_span_count"] == 1
    assert manifest["return_code"] == 0

    request_records = load_artifact_jsonl(artifact_path, "requests/model_inference.jsonl")
    assert len(request_records) == 1
    assert request_records[0]["request"]["model"] == "Qwen3-Coder-30B-A3B-Instruct"
    assert request_records[0]["response"]["id"] == "cmpl-test"
    assert request_records[0]["response"]["choices"][0]["text"] == "hello"
    assert request_records[0]["model_inference_span_id"]
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


def test_job_end_blocks_for_configured_wait(monkeypatch, tmp_path: Path) -> None:
    observed_sleep: list[float] = []

    def fake_sleep(seconds: float) -> None:
        observed_sleep.append(seconds)

    monkeypatch.setattr("gateway.app.time.sleep", fake_sleep)

    client = build_client(job_end_trace_wait_seconds=2.5)
    response = client.post("/job/start", json={"output_location": str(tmp_path / "jobout")})
    assert response.status_code == 200

    response = client.post("/job/end", json={"status": "completed"})
    assert response.status_code == 200
    assert observed_sleep == [2.5]


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


def test_gateway_parse_port_rewrites_reasoning_response(tmp_path: Path) -> None:
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
    assert message["content"] == "final answer"
    assert message["reasoning"] == "draft plan"
    assert message["reasoning_content"] == "draft plan"

    assert client.post("/agent/end", json={"api_token": token, "return_code": 0}).status_code == 200
    artifact_response = client.post("/job/end", json={"status": "completed"})
    assert artifact_response.status_code == 200
    artifact_path = Path(artifact_response.json()["artifacts"][0]["path"])
    request_records = load_artifact_jsonl(artifact_path, "requests/model_inference.jsonl")
    raw_message = request_records[0]["response"]["choices"][0]["message"]
    assert raw_message["content"] == "<think>draft plan</think>final answer"
    assert raw_message["reasoning"] is None
    assert raw_message["reasoning_content"] is None


def test_gateway_parse_port_is_noop_without_configured_reasoning_parser(tmp_path: Path) -> None:
    def response_json_factory(request_json: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": request_json["model"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "plain answer",
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
    token = "agent-token-noop"

    assert client.post("/job/start", json={"output_location": str(tmp_path / "noopout")}).status_code == 200
    assert client.post("/agent/start", json={"api_token": token}).status_code == 200

    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token},
        json={"model": "Qwen3-Coder-30B-A3B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "plain answer"
    assert message["reasoning"] is None
    assert message["reasoning_content"] is None


def test_ipc_context_reports_active_agent_context_usage(tmp_path: Path) -> None:
    def response_json_factory(request_json: dict[str, Any]) -> dict[str, Any]:
        usage = request_json["test_usage"]
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": request_json["model"],
            "usage": usage,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        }

    client = build_client(response_json_factory=response_json_factory)
    token_a = "agent-token-a"
    token_b = "agent-token-b"

    assert client.post("/job/start", json={"output_location": str(tmp_path / "ipc-out")}).status_code == 200
    assert client.post("/agent/start", json={"api_token": token_a}).status_code == 200
    assert client.post("/agent/start", json={"api_token": token_b}).status_code == 200

    response = client.get("/ipc/context")
    assert response.status_code == 200
    assert response.json()["agent_count"] == 2
    assert response.json()["total_context_tokens"] == 0

    assert client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token_a},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "test_usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        },
    ).status_code == 200
    assert client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token_b},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "test_usage": {"prompt_tokens": 400, "completion_tokens": 100},
        },
    ).status_code == 200

    response = client.get("/ipc/context")
    assert response.status_code == 200
    payload = response.json()
    active_runs = client.app.state.gateway_service.active_runs
    assert payload["agent_count"] == 2
    assert payload["total_context_tokens"] == 1700
    assert payload["agents"] == [
        {
            "api_token_hash": sha256_hex(token_a),
            "trace_id": active_runs[token_a].trace_id,
            "run_start_time": active_runs[token_a].run_start_time,
            "context_tokens": 1200,
        },
        {
            "api_token_hash": sha256_hex(token_b),
            "trace_id": active_runs[token_b].trace_id,
            "run_start_time": active_runs[token_b].run_start_time,
            "context_tokens": 500,
        },
    ]

    assert client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token_a},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "again"}],
            "test_usage": {"prompt_tokens": 1300, "completion_tokens": 300},
        },
    ).status_code == 200

    response = client.get("/ipc/context")
    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_count"] == 2
    assert payload["total_context_tokens"] == 2100

    assert client.post("/agent/end", json={"api_token": token_b, "return_code": 0}).status_code == 200
    response = client.get("/ipc/context")
    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_count"] == 1
    assert payload["total_context_tokens"] == 1600
    assert payload["agents"] == [
        {
            "api_token_hash": sha256_hex(token_a),
            "trace_id": active_runs[token_a].trace_id,
            "run_start_time": active_runs[token_a].run_start_time,
            "context_tokens": 1600,
        }
    ]


def test_ipc_context_keeps_last_known_value_when_usage_is_missing(tmp_path: Path) -> None:
    def response_json_factory(request_json: dict[str, Any]) -> dict[str, Any]:
        if request_json.get("omit_usage"):
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": request_json["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            }
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": request_json["model"],
            "usage": request_json["test_usage"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        }

    client = build_client(response_json_factory=response_json_factory)
    token = "agent-token-ipc-missing"

    assert client.post("/job/start", json={"output_location": str(tmp_path / "ipc-missing")}).status_code == 200
    assert client.post("/agent/start", json={"api_token": token}).status_code == 200

    assert client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "test_usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        },
    ).status_code == 200
    assert client.get("/ipc/context").json()["total_context_tokens"] == 1200

    assert client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi again"}],
            "omit_usage": True,
        },
    ).status_code == 200
    assert client.get("/ipc/context").json()["total_context_tokens"] == 1200


def test_ipc_output_throughput_reports_summary_and_details(tmp_path: Path) -> None:
    async def fake_forward(
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> ForwardResult:
        request_json = json.loads(body.decode("utf-8")) if body else {}
        await asyncio.sleep(float(request_json.get("test_delay_s", 0.0)))
        response_json = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": request_json["model"],
            "usage": request_json["test_usage"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        }
        return ForwardResult(
            status_code=200,
            content=json.dumps(response_json).encode("utf-8"),
            content_type="application/json",
            response_json=response_json,
        )

    client = build_client(forwarder=fake_forward)
    token_a = "agent-token-throughput-a"
    token_b = "agent-token-throughput-b"

    assert client.post("/job/start", json={"output_location": str(tmp_path / "ipc-throughput")}).status_code == 200
    assert client.post("/agent/start", json={"api_token": token_a}).status_code == 200
    assert client.post("/agent/start", json={"api_token": token_b}).status_code == 200

    response = client.get("/ipc/output-throughput")
    assert response.status_code == 200
    assert response.json()["agent_count"] == 2
    assert response.json()["throughput_agent_count"] == 0
    assert response.json()["min_output_tokens_per_s"] is None
    assert response.json()["max_output_tokens_per_s"] is None
    assert response.json()["avg_output_tokens_per_s"] is None

    response = client.get("/ipc/output-throughput/agents")
    assert response.status_code == 200
    payload = response.json()
    active_runs = client.app.state.gateway_service.active_runs
    expected_agents = [
        {
            "api_token_hash": run.api_token_hash,
            "output_tokens_per_s": None,
        }
        for run in sorted(
            active_runs.values(),
            key=lambda run: (run.run_start_time, run.api_token_hash, run.trace_id),
        )
    ]
    assert payload["agent_count"] == 2
    assert payload["agents"] == expected_agents

    assert client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token_a},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "test_delay_s": 0.04,
            "test_usage": {"prompt_tokens": 1000, "completion_tokens": 120},
        },
    ).status_code == 200
    assert client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token_b},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "test_delay_s": 0.01,
            "test_usage": {"prompt_tokens": 400, "completion_tokens": 80},
        },
    ).status_code == 200

    active_runs = client.app.state.gateway_service.active_runs
    expected_agents = [
        {
            "api_token_hash": run.api_token_hash,
            "output_tokens_per_s": (
                round(run.current_output_tokens_per_s, 6)
                if run.current_output_tokens_per_s is not None
                else None
            ),
        }
        for run in sorted(
            active_runs.values(),
            key=lambda run: (run.run_start_time, run.api_token_hash, run.trace_id),
        )
    ]
    expected_values = [
        agent["output_tokens_per_s"]
        for agent in expected_agents
        if agent["output_tokens_per_s"] is not None
    ]

    response = client.get("/ipc/output-throughput")
    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_count"] == 2
    assert payload["throughput_agent_count"] == 2
    assert payload["min_output_tokens_per_s"] == min(expected_values)
    assert payload["max_output_tokens_per_s"] == max(expected_values)
    assert payload["avg_output_tokens_per_s"] == round(
        sum(expected_values) / len(expected_values),
        6,
    )

    response = client.get("/ipc/output-throughput/agents")
    assert response.status_code == 200
    assert response.json()["agents"] == expected_agents

    assert client.post("/agent/end", json={"api_token": token_b, "return_code": 0}).status_code == 200

    response = client.get("/ipc/output-throughput")
    assert response.status_code == 200
    payload = response.json()
    remaining_run = client.app.state.gateway_service.active_runs[token_a]
    remaining_throughput = round(remaining_run.current_output_tokens_per_s, 6)
    assert payload["agent_count"] == 1
    assert payload["throughput_agent_count"] == 1
    assert payload["min_output_tokens_per_s"] == remaining_throughput
    assert payload["max_output_tokens_per_s"] == remaining_throughput
    assert payload["avg_output_tokens_per_s"] == remaining_throughput

    response = client.get("/ipc/output-throughput/agents")
    assert response.status_code == 200
    assert response.json()["agents"] == [
        {
            "api_token_hash": remaining_run.api_token_hash,
            "output_tokens_per_s": remaining_throughput,
        }
    ]


def test_ipc_output_throughput_keeps_last_known_value_when_usage_is_missing(
    tmp_path: Path,
) -> None:
    async def fake_forward(
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> ForwardResult:
        request_json = json.loads(body.decode("utf-8")) if body else {}
        await asyncio.sleep(float(request_json.get("test_delay_s", 0.0)))
        response_json = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": request_json["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        }
        if not request_json.get("omit_usage"):
            response_json["usage"] = request_json["test_usage"]
        return ForwardResult(
            status_code=200,
            content=json.dumps(response_json).encode("utf-8"),
            content_type="application/json",
            response_json=response_json,
        )

    client = build_client(forwarder=fake_forward)
    token = "agent-token-throughput-missing"

    assert client.post(
        "/job/start",
        json={"output_location": str(tmp_path / "ipc-throughput-missing")},
    ).status_code == 200
    assert client.post("/agent/start", json={"api_token": token}).status_code == 200

    assert client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "test_delay_s": 0.02,
            "test_usage": {"prompt_tokens": 1000, "completion_tokens": 120},
        },
    ).status_code == 200

    first_detail = client.get("/ipc/output-throughput/agents").json()
    assert first_detail["agent_count"] == 1
    first_value = first_detail["agents"][0]["output_tokens_per_s"]
    assert first_value is not None

    assert client.post(
        "/v1/chat/completions",
        headers={"x-api-key": token},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi again"}],
            "test_delay_s": 0.02,
            "omit_usage": True,
        },
    ).status_code == 200

    second_detail = client.get("/ipc/output-throughput/agents").json()
    assert second_detail["agents"] == first_detail["agents"]

    summary = client.get("/ipc/output-throughput").json()
    assert summary["agent_count"] == 1
    assert summary["throughput_agent_count"] == 1
    assert summary["min_output_tokens_per_s"] == first_value
    assert summary["max_output_tokens_per_s"] == first_value
    assert summary["avg_output_tokens_per_s"] == first_value


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

    monkeypatch.setattr("gateway.app.httpx.AsyncClient", FakeAsyncClient)

    service = GatewayService(
        config=GatewayConfig(
            vllm_base_url="http://vllm:11451",
            jaeger_api_base_url="http://jaeger:16686/api/traces",
            otlp_traces_endpoint="grpc://jaeger:4317",
        ),
        tracer=trace.get_tracer("gateway-test"),
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
            jaeger_api_base_url="http://jaeger:16686/api/traces",
            otlp_traces_endpoint="grpc://jaeger:4317",
        ),
        tracer=trace.get_tracer("gateway-test"),
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
                jaeger_api_base_url="http://jaeger:16686/api/traces",
                otlp_traces_endpoint="grpc://jaeger:4317",
            ),
            tracer=trace.get_tracer("gateway-test"),
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
    assert cfg.jaeger_api_base_url == "http://localhost:27544/api/traces"
    assert cfg.otlp_traces_endpoint == "grpc://localhost:24831"


def test_gateway_config_uses_default_profile_when_unspecified() -> None:
    cfg = GatewayConfig.from_port_profile()
    profile = load_port_profile()
    assert cfg.vllm_base_url == f"http://localhost:{profile.vllm_port}"
    assert cfg.jaeger_api_base_url == f"http://localhost:{profile.jaeger_api_port}/api/traces"
    assert cfg.otlp_traces_endpoint == f"grpc://localhost:{profile.jaeger_otlp_port}"


def test_gateway_config_respects_jaeger_endpoint_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "GATEWAY_JAEGER_API_BASE_URL_OVERRIDE",
        "http://localhost:16686/api/traces",
    )
    monkeypatch.setenv(
        "GATEWAY_OTLP_TRACES_ENDPOINT_OVERRIDE",
        "grpc://localhost:4317",
    )

    cfg = GatewayConfig.from_port_profile(2)
    assert cfg.vllm_base_url == "http://localhost:31987"
    assert cfg.jaeger_api_base_url == "http://localhost:16686/api/traces"
    assert cfg.otlp_traces_endpoint == "grpc://localhost:4317"


def test_port_profile_exposes_gateway_parse_port() -> None:
    profile = load_port_profile(1)
    assert profile.gateway_port == 24157
    assert profile.gateway_parse_port == 28171


def test_listener_ports_still_expose_parse_port_when_model_has_no_reasoning_parser() -> None:
    profile = load_port_profile(1)
    config = GatewayConfig.from_port_profile(1)
    raw_port, parse_port = resolve_listener_ports(
        gateway_config=config,
        profile=profile,
        model_registry=load_model_registry(),
        served_model_name="Qwen3-Coder-30B-A3B-Instruct",
    )

    assert raw_port == 24157
    assert parse_port == 28171


def test_listener_ports_preserve_parse_port_when_model_has_reasoning_parser() -> None:
    profile = load_port_profile(1)
    config = GatewayConfig.from_port_profile(1)
    raw_port, parse_port = resolve_listener_ports(
        gateway_config=config,
        profile=profile,
        model_registry=load_model_registry(),
        served_model_name="MiniMax-M2.5",
    )

    assert raw_port == 24157
    assert parse_port == 28171


def test_default_ipc_socket_path_uses_port_profile_id() -> None:
    assert _default_ipc_socket_path(0) == Path("/tmp/vllm-gateway-profile-0.sock")
    assert _default_ipc_socket_path("3") == Path("/tmp/vllm-gateway-profile-3.sock")


def test_resolve_ipc_socket_path_defaults_to_profile_specific_path() -> None:
    assert _resolve_ipc_socket_path(
        ipc_enabled=True,
        configured_socket_path=None,
        profile_id=2,
    ) == Path("/tmp/vllm-gateway-profile-2.sock")


def test_resolve_ipc_socket_path_respects_disable_and_override(tmp_path: Path) -> None:
    custom_socket_path = tmp_path / "custom.sock"

    assert _resolve_ipc_socket_path(
        ipc_enabled=False,
        configured_socket_path=None,
        profile_id=2,
    ) is None
    assert _resolve_ipc_socket_path(
        ipc_enabled=True,
        configured_socket_path=str(custom_socket_path),
        profile_id=2,
    ) == custom_socket_path.resolve()


def test_gateway_config_reads_runtime_settings() -> None:
    settings = GatewayRuntimeSettings(
        artifact_compression="tar.gz",
        job_end_trace_wait_seconds=2.5,
        service_name="gateway-custom",
        otlp_traces_insecure=False,
        ipc_enabled=True,
        ipc_socket_path="/tmp/vllm-gateway.sock",
        ipc_socket_permissions=0o666,
        ipc_socket_uid=1234,
        ipc_socket_gid=5678,
    )

    cfg = GatewayConfig.from_port_profile(0, runtime_settings=settings)
    assert cfg.artifact_compression == "tar.gz"
    assert cfg.job_end_trace_wait_seconds == 2.5
    assert cfg.service_name == "gateway-custom"
    assert cfg.otlp_traces_insecure is False


def test_gateway_config_rejects_unknown_port_profile() -> None:
    with pytest.raises(ValueError, match="unknown port profile id"):
        GatewayConfig.from_port_profile(999)


def test_runtime_settings_load_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "gateway.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[run]",
                "port_profile_id = 2",
                'output_root = "./artifacts"',
                "",
                "[telemetry]",
                'service_name = "gateway-from-toml"',
                "otlp_traces_insecure = false",
                "",
                "[gateway]",
                'artifact_compression = "tar.gz"',
                "job_end_trace_wait_seconds = 3.5",
                "",
                "[ipc]",
                "enabled = false",
                'socket_path = "/tmp/vllm-gateway.sock"',
                'socket_permissions = "666"',
                "socket_uid = 1234",
                "socket_gid = 5678",
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(config_path, allow_missing=False)
    assert settings.port_profile_id == 2
    assert settings.output_root == "./artifacts"
    assert settings.service_name == "gateway-from-toml"
    assert settings.otlp_traces_insecure is False
    assert settings.artifact_compression == "tar.gz"
    assert settings.job_end_trace_wait_seconds == 3.5
    assert settings.ipc_enabled is False
    assert settings.ipc_socket_path == "/tmp/vllm-gateway.sock"
    assert settings.ipc_socket_permissions == 0o666
    assert settings.ipc_socket_uid == 1234
    assert settings.ipc_socket_gid == 5678


def test_runtime_settings_default_ipc_is_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "gateway-default-ipc.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[run]",
                "port_profile_id = 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(config_path, allow_missing=False)
    assert settings.ipc_enabled is True
    assert settings.ipc_socket_path is None


def test_create_unix_listen_socket_creates_socket_with_permissions(tmp_path: Path) -> None:
    socket_path = tmp_path / "gateway.sock"

    listen_socket = _create_unix_listen_socket(
        socket_path,
        permissions=0o660,
        uid=None,
        gid=None,
    )
    try:
        assert socket_path.exists()
        assert listen_socket.family == socket.AF_UNIX
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o660
    finally:
        listen_socket.close()
        if socket_path.exists():
            socket_path.unlink()


def test_create_unix_listen_socket_reuses_stale_socket_path(tmp_path: Path) -> None:
    socket_path = tmp_path / "gateway.sock"
    stale_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_socket.bind(str(socket_path))
    stale_socket.close()

    listen_socket = _create_unix_listen_socket(
        socket_path,
        permissions=0o660,
        uid=None,
        gid=None,
    )
    try:
        assert socket_path.exists()
        assert listen_socket.family == socket.AF_UNIX
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o660
    finally:
        listen_socket.close()
        if socket_path.exists():
            socket_path.unlink()


def test_create_unix_listen_socket_rejects_active_socket(tmp_path: Path) -> None:
    socket_path = tmp_path / "gateway.sock"
    active_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        active_socket.bind(str(socket_path))
        active_socket.listen(1)

        with pytest.raises(RuntimeError, match="IPC socket path already exists and is active"):
            _create_unix_listen_socket(
                socket_path,
                permissions=0o660,
                uid=None,
                gid=None,
            )
    finally:
        active_socket.close()
        if socket_path.exists():
            socket_path.unlink()
