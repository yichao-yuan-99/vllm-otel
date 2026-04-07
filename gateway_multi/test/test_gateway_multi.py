from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.app import ForwardResult
from gateway.port_profiles import load_port_profile
from gateway_multi.app import (
    BackendBinding,
    BackendGatewayService,
    build_backend_config,
    create_control_app,
    create_ipc_app,
    GatewayMultiService,
    normalize_assignment_policy,
)
from gateway_multi.cli import (
    _default_ipc_socket_path,
    _resolve_ipc_socket_path,
)
from gateway_multi.runtime_config import GatewayMultiRuntimeSettings, load_runtime_settings


def load_artifact_json(artifact_path: Path, member_name: str) -> dict[str, Any]:
    return json.loads((artifact_path / member_name).read_text(encoding="utf-8"))


def load_artifact_jsonl(artifact_path: Path, member_name: str) -> list[dict[str, Any]]:
    lines = (artifact_path / member_name).read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_multi_clients(
    *,
    profile_ids: tuple[str, ...] = ("0", "1"),
    assignment_policy: str = "round_robin",
) -> tuple[TestClient, dict[str, TestClient], GatewayMultiService]:
    runtime_settings = GatewayMultiRuntimeSettings(
        port_profile_ids=profile_ids,
        assignment_policy=assignment_policy,
        artifact_compression="none",
        job_end_trace_wait_seconds=0.0,
        service_name="gateway-multi-test",
    )

    bindings: list[BackendBinding] = []
    for profile_id in profile_ids:
        profile = load_port_profile(profile_id)
        config = build_backend_config(profile, runtime_settings)
        tracer_provider = TracerProvider()
        tracer = tracer_provider.get_tracer("gateway-multi-test")

        async def fake_forward(
            method: str,
            path: str,
            headers: dict[str, str],
            body: bytes,
            *,
            resolved_profile_id: str = profile.profile_id,
        ) -> ForwardResult:
            request_json = json.loads(body.decode("utf-8")) if body else {}
            response_json = {
                "id": f"cmpl-{resolved_profile_id}",
                "object": "chat.completion",
                "model": request_json.get("model"),
                "backend_port_profile_id": resolved_profile_id,
                "usage": request_json.get(
                    "test_usage",
                    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                ),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"backend-{resolved_profile_id}",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            return ForwardResult(
                status_code=200,
                content=json.dumps(response_json).encode("utf-8"),
                content_type="application/json",
                response_json=response_json,
            )

        def fake_jaeger_fetch(trace_id: str, *, resolved_profile_id: str = profile.profile_id) -> dict[str, Any]:
            return {
                "data": [
                    {
                        "traceID": trace_id,
                        "backend_port_profile_id": resolved_profile_id,
                    }
                ]
            }

        backend_service = BackendGatewayService(
            profile=profile,
            config=config,
            tracer_provider=tracer_provider,
            tracer=tracer,
            forwarder=fake_forward,
            jaeger_fetcher=fake_jaeger_fetch,
        )
        bindings.append(BackendBinding(profile=profile, service=backend_service))

    multi_service = GatewayMultiService(bindings, assignment_policy=assignment_policy)
    control_profile = bindings[0].profile
    control_client = TestClient(
        create_control_app(
            multi_service,
            gateway_parse_port=control_profile.gateway_parse_port,
        ),
        base_url=f"http://testserver:{control_profile.gateway_port}",
    )
    ipc_clients = {
        binding.profile.profile_id: TestClient(create_ipc_app(binding.service))
        for binding in bindings
    }
    return control_client, ipc_clients, multi_service


def test_round_robin_assignment_and_ipc_isolation(tmp_path: Path) -> None:
    control_client, ipc_clients, _ = build_multi_clients()
    output_dir = tmp_path / "multi-artifacts"

    response = control_client.post("/job/start", json={"output_location": str(output_dir)})
    assert response.status_code == 200
    assert response.json()["backend_port_profile_ids"] == ["0", "1"]
    assert response.json()["control_port_profile_id"] == "0"
    assert set(response.json()["ctx_aware_job_log_paths"]) == {"0", "1"}

    first = control_client.post("/agent/start", json={"api_token": "token-a"})
    second = control_client.post("/agent/start", json={"api_token": "token-b"})
    third = control_client.post("/agent/start", json={"api_token": "token-c"})
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert first.json()["backend_port_profile_id"] == "0"
    assert second.json()["backend_port_profile_id"] == "1"
    assert third.json()["backend_port_profile_id"] == "0"

    response = control_client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "token-a"},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "test_usage": {"prompt_tokens": 100, "completion_tokens": 10},
        },
    )
    assert response.status_code == 200
    assert response.json()["backend_port_profile_id"] == "0"

    response = control_client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "token-b"},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "test_usage": {"prompt_tokens": 40, "completion_tokens": 5},
        },
    )
    assert response.status_code == 200
    assert response.json()["backend_port_profile_id"] == "1"

    ipc_first = ipc_clients["0"].get("/ipc/context")
    assert ipc_first.status_code == 200
    assert ipc_first.json()["backend_port_profile_id"] == "0"
    assert ipc_first.json()["agent_count"] == 2
    assert ipc_first.json()["total_context_tokens"] == 110
    assert ipc_first.json()["ctx_aware_enabled"] is False
    assert ipc_first.json()["effective_total_context_tokens"] == 3110
    assert ipc_first.json()["ongoing_agent_count"] == 2
    assert ipc_first.json()["pending_agent_count"] == 0
    assert [agent["backend_port_profile_id"] for agent in ipc_first.json()["agents"]] == ["0", "0"]
    assert [agent["schedule_state"] for agent in ipc_first.json()["agents"]] == ["ongoing", "ongoing"]
    assert [agent["effective_context_tokens"] for agent in ipc_first.json()["agents"]] == [110, 3000]
    assert [agent["has_usable_context_usage"] for agent in ipc_first.json()["agents"]] == [True, False]

    ipc_second = ipc_clients["1"].get("/ipc/context")
    assert ipc_second.status_code == 200
    assert ipc_second.json()["backend_port_profile_id"] == "1"
    assert ipc_second.json()["agent_count"] == 1
    assert ipc_second.json()["total_context_tokens"] == 45
    assert ipc_second.json()["ctx_aware_enabled"] is False
    assert ipc_second.json()["effective_total_context_tokens"] == 45
    assert ipc_second.json()["agents"][0]["backend_port_profile_id"] == "1"
    assert ipc_second.json()["agents"][0]["schedule_state"] == "ongoing"
    assert ipc_second.json()["agents"][0]["effective_context_tokens"] == 45
    assert ipc_second.json()["agents"][0]["has_usable_context_usage"] is True

    assert ipc_clients["1"].post("/job/start", json={"output_location": str(output_dir)}).status_code == 404
    assert ipc_clients["1"].post("/v1/chat/completions", json={}).status_code == 404


def test_lowest_usage_assignment_uses_ongoing_context_usage(tmp_path: Path) -> None:
    control_client, ipc_clients, _ = build_multi_clients(assignment_policy="lowest_usage")
    output_dir = tmp_path / "lowest-usage-artifacts"

    response = control_client.post("/job/start", json={"output_location": str(output_dir)})
    assert response.status_code == 200
    assert response.json()["assignment_policy"] == "lowest_usage"

    first = control_client.post("/agent/start", json={"api_token": "token-a"})
    second = control_client.post("/agent/start", json={"api_token": "token-b"})
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["backend_port_profile_id"] == "0"
    assert second.json()["backend_port_profile_id"] == "1"

    assert control_client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "token-a"},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "test_usage": {"prompt_tokens": 100, "completion_tokens": 10},
        },
    ).status_code == 200
    assert control_client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "token-b"},
        json={
            "model": "Qwen3-Coder-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "test_usage": {"prompt_tokens": 200, "completion_tokens": 5},
        },
    ).status_code == 200

    assert ipc_clients["0"].get("/ipc/context").json()["ongoing_effective_context_tokens"] == 110
    assert ipc_clients["1"].get("/ipc/context").json()["ongoing_effective_context_tokens"] == 205

    third = control_client.post("/agent/start", json={"api_token": "token-c"})
    assert third.status_code == 200
    assert third.json()["assignment_policy"] == "lowest_usage"
    assert third.json()["backend_port_profile_id"] == "0"


def test_lowest_profile_without_pending_prefers_lowest_profile_and_pending_fallback(
    tmp_path: Path,
) -> None:
    control_client, ipc_clients, _ = build_multi_clients(
        profile_ids=("13", "2"),
        assignment_policy="lowest_profile_without_pending",
    )
    output_dir = tmp_path / "lowest-profile-without-pending"

    assert control_client.post(
        "/ctx-aware/start",
        json={
            "usage_threshold_tokens": 9000,
            "scheduling_threshold_tokens": 6000,
        },
    ).status_code == 200
    response = control_client.post("/job/start", json={"output_location": str(output_dir)})
    assert response.status_code == 200
    assert response.json()["assignment_policy"] == "lowest_profile_without_pending"
    assert response.json()["control_port_profile_id"] == "13"

    first = control_client.post("/agent/start", json={"api_token": "token-a"})
    second = control_client.post("/agent/start", json={"api_token": "token-b"})
    third = control_client.post("/agent/start", json={"api_token": "token-c"})
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert first.json()["backend_port_profile_id"] == "2"
    assert second.json()["backend_port_profile_id"] == "2"
    assert third.json()["backend_port_profile_id"] == "2"

    assert ipc_clients["2"].get("/ipc/context").json()["pending_agent_count"] == 1
    assert ipc_clients["13"].get("/ipc/context").json()["pending_agent_count"] == 0

    fourth = control_client.post("/agent/start", json={"api_token": "token-d"})
    fifth = control_client.post("/agent/start", json={"api_token": "token-e"})
    sixth = control_client.post("/agent/start", json={"api_token": "token-f"})
    assert fourth.status_code == 200
    assert fifth.status_code == 200
    assert sixth.status_code == 200
    assert fourth.json()["backend_port_profile_id"] == "13"
    assert fifth.json()["backend_port_profile_id"] == "13"
    assert sixth.json()["backend_port_profile_id"] == "13"

    assert ipc_clients["2"].get("/ipc/context").json()["pending_agent_count"] == 1
    assert ipc_clients["13"].get("/ipc/context").json()["pending_agent_count"] == 1

    seventh = control_client.post("/agent/start", json={"api_token": "token-g"})
    assert seventh.status_code == 200
    assert seventh.json()["backend_port_profile_id"] == "2"

    pending_second = ipc_clients["2"].get("/ipc/context").json()
    pending_thirteenth = ipc_clients["13"].get("/ipc/context").json()
    assert pending_second["pending_agent_count"] == 2
    assert pending_second["pending_effective_context_tokens"] == 6000
    assert pending_thirteenth["pending_agent_count"] == 1
    assert pending_thirteenth["pending_effective_context_tokens"] == 3000

    eighth = control_client.post("/agent/start", json={"api_token": "token-h"})
    assert eighth.status_code == 200
    assert eighth.json()["backend_port_profile_id"] == "13"


def test_job_end_aggregates_artifacts_from_all_backends(tmp_path: Path) -> None:
    control_client, _, _ = build_multi_clients()
    output_dir = tmp_path / "multi-artifacts"

    assert control_client.post("/job/start", json={"output_location": str(output_dir)}).status_code == 200
    assert control_client.post("/agent/start", json={"api_token": "token-a"}).status_code == 200
    assert control_client.post("/agent/start", json={"api_token": "token-b"}).status_code == 200

    assert control_client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "token-a"},
        json={"model": "Qwen3-Coder-30B-A3B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
    ).status_code == 200
    assert control_client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "token-b"},
        json={"model": "Qwen3-Coder-30B-A3B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
    ).status_code == 200

    assert control_client.post("/agent/end", json={"api_token": "token-a", "return_code": 0}).status_code == 200
    assert control_client.post("/agent/end", json={"api_token": "token-b", "return_code": 0}).status_code == 200

    response = control_client.post("/job/end", json={"status": "completed"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_count"] == 2
    assert payload["backend_port_profile_ids"] == ["0", "1"]
    assert set(payload["ctx_aware_job_log_paths"]) == {"0", "1"}
    for job_log_path in payload["ctx_aware_job_log_paths"].values():
        assert Path(job_log_path).exists()

    backend_ids = sorted(
        artifact["backend_port_profile_id"] for artifact in payload["artifacts"]
    )
    assert backend_ids == ["0", "1"]

    for artifact in payload["artifacts"]:
        artifact_path = Path(artifact["path"])
        assert artifact_path.exists()
        assert artifact["artifact_format"] == "none"
        manifest = load_artifact_json(artifact_path, "manifest.json")
        assert manifest["backend_port_profile_id"] == artifact["backend_port_profile_id"]
        assert manifest["request_count"] == 1
        request_records = load_artifact_jsonl(
            artifact_path,
            "requests/model_inference.jsonl",
        )
        assert len(request_records) == 1
        assert request_records[0]["backend_port_profile_id"] == artifact["backend_port_profile_id"]
        assert request_records[0]["port_profile_id"] == artifact["backend_port_profile_id"]
        assert request_records[0]["forward_start_time"] is not None
        assert request_records[0]["pending_duration_ms"] >= 0
        assert request_records[0]["span_duration_ms"] is not None
        assert request_records[0]["request_duration_ms"] >= request_records[0]["span_duration_ms"]
        lifecycle_records = load_artifact_jsonl(
            artifact_path,
            "events/lifecycle.jsonl",
        )
        assert [record["event_type"] for record in lifecycle_records] == [
            "job_start",
            "agent_start",
            "port_profile_assignment",
            "agent_end",
            "job_end",
        ]
        assignment_event = lifecycle_records[2]
        assert assignment_event["metadata"] == {
            "assignment_policy": "round_robin",
            "reason": "initial_assignment",
            "port_profile_id": artifact["backend_port_profile_id"],
            "backend_port_profile_id": artifact["backend_port_profile_id"],
            "previous_port_profile_id": None,
        }


def test_ctx_aware_control_propagates_to_all_backends(tmp_path: Path) -> None:
    control_client, ipc_clients, _ = build_multi_clients()
    output_dir = tmp_path / "ctx-aware-multi"

    response = control_client.get("/ctx-aware")
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert response.json()["policy_mode"] == "age"
    assert response.json()["backend_port_profile_ids"] == ["0", "1"]

    response = control_client.post(
        "/ctx-aware/start",
        json={
            "usage_threshold_tokens": 9000,
            "scheduling_threshold_tokens": 6000,
            "policy_mode": "throughput",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["policy_mode"] == "throughput"
    assert payload["ongoing_agent_count"] == 0
    assert payload["pending_agent_count"] == 0
    assert len(payload["backends"]) == 2
    assert {backend["backend_port_profile_id"] for backend in payload["backends"]} == {"0", "1"}
    assert all(backend["enabled"] is True for backend in payload["backends"])

    response = control_client.post("/job/start", json={"output_location": str(output_dir)})
    assert response.status_code == 200
    assert set(response.json()["ctx_aware_job_log_paths"]) == {"0", "1"}

    assert control_client.post(
        "/ctx-aware/start",
        json={
            "usage_threshold_tokens": 12000,
            "scheduling_threshold_tokens": 9000,
        },
    ).status_code == 409
    assert control_client.post("/ctx-aware/end").status_code == 409

    for token in ["agent-a", "agent-b", "agent-c", "agent-d", "agent-e"]:
        assert control_client.post("/agent/start", json={"api_token": token}).status_code == 200

    status = control_client.get("/ctx-aware")
    assert status.status_code == 200
    payload = status.json()
    assert payload["enabled"] is True
    assert payload["policy_mode"] == "throughput"
    assert payload["ongoing_agent_count"] == 4
    assert payload["pending_agent_count"] == 1
    assert payload["ongoing_effective_context_tokens"] == 12000
    assert payload["pending_effective_context_tokens"] == 3000

    backend_summaries = {
        backend["backend_port_profile_id"]: backend for backend in payload["backends"]
    }
    assert backend_summaries["0"]["ongoing_agent_count"] == 2
    assert backend_summaries["0"]["pending_agent_count"] == 1
    assert backend_summaries["1"]["ongoing_agent_count"] == 2
    assert backend_summaries["1"]["pending_agent_count"] == 0

    ipc_first = ipc_clients["0"].get("/ipc/context")
    assert ipc_first.status_code == 200
    assert ipc_first.json()["ctx_aware_enabled"] is True
    assert ipc_first.json()["ctx_aware_policy_mode"] == "throughput"
    assert ipc_first.json()["ongoing_agent_count"] == 2
    assert ipc_first.json()["pending_agent_count"] == 1

    for token in ["agent-a", "agent-b", "agent-c", "agent-d", "agent-e"]:
        assert control_client.post("/agent/end", json={"api_token": token, "return_code": 0}).status_code == 200

    response = control_client.post("/job/end", json={"status": "completed"})
    assert response.status_code == 200
    assert set(response.json()["ctx_aware_job_log_paths"]) == {"0", "1"}

    persisted = control_client.get("/ctx-aware")
    assert persisted.status_code == 200
    assert persisted.json()["enabled"] is True
    assert persisted.json()["policy_mode"] == "throughput"

    disabled = control_client.post("/ctx-aware/end")
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False
    assert disabled.json()["policy_mode"] == "age"


def test_policy_endpoint_only_allows_changes_when_no_job_is_running(tmp_path: Path) -> None:
    control_client, _, _ = build_multi_clients()
    output_dir = tmp_path / "policy-multi"

    status = control_client.get("/policy")
    assert status.status_code == 200
    assert status.json() == {
        "status": "ok",
        "assignment_policy": "round_robin",
        "supported_assignment_policies": [
            "lowest_profile_without_pending",
            "lowest_usage",
            "round_robin",
        ],
        "job_active": False,
        "backend_port_profile_ids": ["0", "1"],
        "control_port_profile_id": "0",
    }

    updated = control_client.post(
        "/policy",
        json={"assignment_policy": "lowest_usage"},
    )
    assert updated.status_code == 200
    assert updated.json()["assignment_policy"] == "lowest_usage"
    assert updated.json()["job_active"] is False

    invalid = control_client.post(
        "/policy",
        json={"assignment_policy": "least_loaded"},
    )
    assert invalid.status_code == 400
    assert "assignment policy must be one of" in invalid.json()["detail"]

    assert control_client.post("/job/start", json={"output_location": str(output_dir)}).status_code == 200

    while_running = control_client.post(
        "/policy",
        json={"assignment_policy": "round_robin"},
    )
    assert while_running.status_code == 409
    assert "cannot be changed while a job is active" in while_running.json()["detail"]

    assert control_client.post("/job/end", json={"status": "completed"}).status_code == 200

    after_end = control_client.post(
        "/policy",
        json={"assignment_policy": "round_robin"},
    )
    assert after_end.status_code == 200
    assert after_end.json()["assignment_policy"] == "round_robin"
    assert after_end.json()["job_active"] is False


def test_runtime_settings_accept_lowest_usage_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "gateway-multi-lowest-usage.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[run]",
                'port_profile_ids = [0, 1]',
                'assignment_policy = "lowest_usage"',
            ]
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(config_path, allow_missing=False)
    assert settings.assignment_policy == "lowest_usage"


def test_runtime_settings_accept_lowest_profile_without_pending_policy(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "gateway-multi-lowest-profile-without-pending.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[run]",
                'port_profile_ids = [0, 1]',
                'assignment_policy = "lowest_profile_without_pending"',
            ]
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(config_path, allow_missing=False)
    assert settings.assignment_policy == "lowest_profile_without_pending"


def test_lowest_profile_without_pending_requires_ctx_aware_before_job_start(
    tmp_path: Path,
) -> None:
    control_client, _, _ = build_multi_clients(
        assignment_policy="lowest_profile_without_pending",
    )
    output_dir = tmp_path / "lowest-profile-without-pending-no-ctx-aware"

    response = control_client.post("/job/start", json={"output_location": str(output_dir)})
    assert response.status_code == 409
    assert "requires ctx-aware mode to be enabled" in response.json()["detail"]


def test_ctx_aware_pending_request_waits_on_assigned_backend(tmp_path: Path) -> None:
    async def run_case() -> None:
        forward_calls: list[float] = []
        runtime_settings = GatewayMultiRuntimeSettings(
            port_profile_ids=("0",),
            assignment_policy="round_robin",
            artifact_compression="none",
            job_end_trace_wait_seconds=0.0,
            service_name="gateway-multi-test",
        )
        profile = load_port_profile("0")
        config = build_backend_config(profile, runtime_settings)
        tracer_provider = TracerProvider()
        tracer = tracer_provider.get_tracer("gateway-multi-test")

        async def fake_forward(
            method: str,
            path: str,
            headers: dict[str, str],
            body: bytes,
        ) -> ForwardResult:
            forward_calls.append(asyncio.get_running_loop().time())
            await asyncio.sleep(0.02)
            response_json = {
                "id": "cmpl-0",
                "object": "chat.completion",
                "model": "Qwen3-Coder-30B-A3B-Instruct",
                "backend_port_profile_id": "0",
                "usage": {"prompt_tokens": 100, "completion_tokens": 12},
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "backend-0"},
                        "finish_reason": "stop",
                    }
                ],
            }
            return ForwardResult(
                status_code=200,
                content=json.dumps(response_json).encode("utf-8"),
                content_type="application/json",
                response_json=response_json,
            )

        backend_service = BackendGatewayService(
            profile=profile,
            config=config,
            tracer_provider=tracer_provider,
            tracer=tracer,
            forwarder=fake_forward,
            jaeger_fetcher=lambda trace_id: {"data": [{"traceID": trace_id}]},
        )
        service = GatewayMultiService(
            [BackendBinding(profile=profile, service=backend_service)],
            assignment_policy="round_robin",
        )

        await service.start_background_tasks()
        try:
            service.start_ctx_aware(
                usage_threshold_tokens=9000,
                scheduling_threshold_tokens=6000,
            )
            service.start_job(str(tmp_path / "ctx-pending"))
            service.start_agent("agent-a")
            service.start_agent("agent-b")
            service.start_agent("agent-c")

            assert backend_service.active_runs["agent-c"].schedule_state == "pending"

            proxy_task = asyncio.create_task(
                service.proxy_request(
                    api_token="agent-c",
                    method="POST",
                    path="v1/chat/completions",
                    headers={"content-type": "application/json"},
                    body=json.dumps(
                        {
                            "model": "Qwen3-Coder-30B-A3B-Instruct",
                            "messages": [{"role": "user", "content": "wait"}],
                        }
                    ).encode("utf-8"),
                )
            )

            await asyncio.sleep(0.05)
            assert forward_calls == []

            await service.end_agent("agent-b", 0)
            result = await proxy_task
            assert result.status_code == 200
            assert len(forward_calls) == 1

            record = backend_service.active_runs["agent-c"].request_records[0]
            assert record["backend_port_profile_id"] == "0"
            assert record["port_profile_id"] == "0"
            assert record["forward_start_time"] is not None
            assert record["pending_duration_ms"] > 0
            assert record["span_duration_ms"] is not None

            await service.end_agent("agent-a", 0)
            await service.end_agent("agent-c", 0)
            end_payload = service.end_job("completed")
            assert end_payload["artifact_count"] == 3
        finally:
            await service.stop_background_tasks()

    asyncio.run(run_case())


def test_runtime_settings_and_ipc_path_resolution(tmp_path: Path) -> None:
    config_path = tmp_path / "gateway-multi.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[run]",
                'port_profile_ids = [2, "3"]',
                'assignment_policy = "round_robin"',
                'output_root = "./artifacts"',
                "",
                "[telemetry]",
                'service_name = "gateway-multi-from-toml"',
                "otlp_traces_insecure = false",
                "",
                "[gateway]",
                'artifact_compression = "tar.gz"',
                "job_end_trace_wait_seconds = 3.5",
                "",
                "[ipc]",
                'socket_path_template = "/tmp/gateway-multi-{profile_id}.sock"',
                'socket_permissions = "666"',
                "socket_uid = 1234",
                "socket_gid = 5678",
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(config_path, allow_missing=False)
    assert settings.port_profile_ids == ("2", "3")
    assert settings.assignment_policy == "round_robin"
    assert settings.output_root == "./artifacts"
    assert settings.service_name == "gateway-multi-from-toml"
    assert settings.otlp_traces_insecure is False
    assert settings.artifact_compression == "tar.gz"
    assert settings.job_end_trace_wait_seconds == 3.5
    assert settings.ipc_socket_path_template == "/tmp/gateway-multi-{profile_id}.sock"
    assert settings.ipc_socket_permissions == 0o666
    assert settings.ipc_socket_uid == 1234
    assert settings.ipc_socket_gid == 5678

    assert _default_ipc_socket_path(2) == Path("/tmp/vllm-gateway-profile-2.sock")
    assert _resolve_ipc_socket_path(
        ipc_enabled=True,
        configured_socket_path_template="/tmp/gateway-multi-{profile_id}.sock",
        profile_id=3,
    ) == Path("/tmp/gateway-multi-3.sock")
    assert _resolve_ipc_socket_path(
        ipc_enabled=False,
        configured_socket_path_template=None,
        profile_id=3,
    ) is None


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="assignment policy must be one of"):
        normalize_assignment_policy("least_loaded")
