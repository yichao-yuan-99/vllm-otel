from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.app import ForwardResult, GatewayConfig, create_app, create_gateway_service


def build_client(
    *,
    artifact_compression: str = "none",
    job_end_trace_wait_seconds: float = 0.0,
) -> TestClient:
    captured: dict[str, str | None] = {"traceparent": None}

    async def fake_forward(
        method: str, path: str, headers: dict[str, str], body: bytes
    ) -> ForwardResult:
        captured["traceparent"] = headers.get("traceparent") or headers.get("Traceparent")
        request_json = json.loads(body.decode("utf-8")) if body else {}
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
            artifact_compression=artifact_compression,
            job_end_trace_wait_seconds=job_end_trace_wait_seconds,
        ),
        forwarder=fake_forward,
        jaeger_fetcher=fake_jaeger_fetch,
        service_name="gateway-test",
    )
    app = create_app(service=service)
    app.state.captured_headers = captured
    return TestClient(app)


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
