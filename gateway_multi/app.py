from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field

from gateway_ctx.app import (
    CTX_AWARE_JOB_LOG_DIRNAME,
    CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS,
    CTX_AWARE_SCHEDULER_INTERVAL_HZ,
    AgentEndRequest,
    AgentStartRequest,
    CtxAwareStartRequest,
    ForwardResult,
    Forwarder,
    GatewayConfig,
    GatewayService,
    JaegerFetcher,
    JobEndRequest,
    JobStartRequest,
    ResponseTransformer,
    SloAwareStartRequest,
    extract_api_token,
    file_sha256_and_size,
    iso8601_to_compact,
    listener_port_from_request,
    now_iso8601_utc,
    parse_json_if_possible,
    parse_output_location,
    write_json_file,
    write_jsonl_file,
)
from gateway.model_configs import ModelRegistry, load_model_registry
from gateway.port_profiles import PortProfile, load_port_profile
from gateway.reasoning_response_parser import ReasoningResponseParser
from gateway_multi.runtime_config import (
    DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS,
    GatewayMultiRuntimeSettings,
    SUPPORTED_ASSIGNMENT_POLICIES as RUNTIME_SUPPORTED_ASSIGNMENT_POLICIES,
)

SUPPORTED_ASSIGNMENT_POLICIES = set(RUNTIME_SUPPORTED_ASSIGNMENT_POLICIES)


def normalize_assignment_policy(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_ASSIGNMENT_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_ASSIGNMENT_POLICIES))
        raise ValueError(f"assignment policy must be one of: {supported}")
    return normalized


def normalize_balanced_usage_threshold_tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("balanced usage threshold tokens must be an integer")
    if value <= 0:
        raise ValueError("balanced usage threshold tokens must be > 0")
    return value


def _port_profile_sort_key(profile_id: str) -> tuple[int, int | str, str]:
    try:
        return (0, int(profile_id), profile_id)
    except ValueError:
        return (1, profile_id, profile_id)


def configure_backend_tracer(
    service_name: str,
    *,
    endpoint: str,
    insecure: bool,
) -> tuple[TracerProvider, trace.Tracer]:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure))
        )
    return provider, provider.get_tracer("vllm-gateway-multi")


def build_backend_config(
    profile: PortProfile,
    runtime_settings: GatewayMultiRuntimeSettings,
) -> GatewayConfig:
    return GatewayConfig(
        vllm_base_url=f"http://localhost:{profile.vllm_port}",
        jaeger_api_base_url=f"http://localhost:{profile.jaeger_api_port}/api/traces",
        otlp_traces_endpoint=f"grpc://localhost:{profile.jaeger_otlp_port}",
        service_name=f"{runtime_settings.service_name}-profile-{profile.profile_id}",
        otlp_traces_insecure=runtime_settings.otlp_traces_insecure,
        artifact_compression=runtime_settings.artifact_compression,
        job_end_trace_wait_seconds=runtime_settings.job_end_trace_wait_seconds,
        output_root=runtime_settings.output_root,
    )


class BackendGatewayService(GatewayService):
    def __init__(
        self,
        *,
        profile: PortProfile,
        config: GatewayConfig,
        tracer_provider: TracerProvider,
        tracer: trace.Tracer,
        forwarder: Forwarder | None = None,
        jaeger_fetcher: JaegerFetcher | None = None,
    ) -> None:
        super().__init__(
            config=config,
            tracer=tracer,
            forwarder=forwarder,
            jaeger_fetcher=jaeger_fetcher,
        )
        self.profile = profile
        self.backend_port_profile_id = profile.profile_id
        self._tracer_provider = tracer_provider

    def force_flush_traces(self) -> None:
        force_flush = getattr(self._tracer_provider, "force_flush", None)
        if callable(force_flush):
            try:
                force_flush(timeout_millis=5000)
            except Exception:
                pass

    def _annotate_active_run_metadata(self, api_token: str) -> None:
        run = self.active_runs.get(api_token)
        if run is None:
            return
        setattr(run, "backend_port_profile_id", self.backend_port_profile_id)
        for event in run.lifecycle_events:
            metadata = event.get("metadata")
            if isinstance(metadata, dict):
                metadata.setdefault("backend_port_profile_id", self.backend_port_profile_id)

    def _annotate_request_records(self, request_records: list[dict[str, Any]]) -> None:
        for record in request_records:
            record.setdefault("backend_port_profile_id", self.backend_port_profile_id)
            record.setdefault("port_profile_id", self.backend_port_profile_id)

    def record_port_profile_assignment(
        self,
        api_token: str,
        *,
        assignment_policy: str,
        reason: str,
        previous_port_profile_id: str | None = None,
    ) -> None:
        with self._lock:
            run = self.active_runs.get(api_token)
            if run is None:
                raise KeyError("agent not started")
            run.lifecycle_events.append(
                {
                    "event_type": "port_profile_assignment",
                    "timestamp": now_iso8601_utc(),
                    "trace_id": run.trace_id,
                    "api_token_hash": run.api_token_hash,
                    "metadata": {
                        "assignment_policy": assignment_policy,
                        "reason": reason,
                        "port_profile_id": self.backend_port_profile_id,
                        "backend_port_profile_id": self.backend_port_profile_id,
                        "previous_port_profile_id": previous_port_profile_id,
                    },
                }
            )

    def start_agent(self, api_token: str) -> dict[str, Any]:
        payload = super().start_agent(api_token)
        with self._lock:
            self._annotate_active_run_metadata(api_token)
        payload["backend_port_profile_id"] = self.backend_port_profile_id
        return payload

    async def end_agent(self, api_token: str, return_code: int) -> dict[str, Any]:
        with self._lock:
            run = self.active_runs.get(api_token)
        payload = await super().end_agent(api_token, return_code)
        if run is not None and run.lifecycle_events:
            metadata = run.lifecycle_events[-1].get("metadata")
            if isinstance(metadata, dict):
                metadata["backend_port_profile_id"] = self.backend_port_profile_id
        payload["backend_port_profile_id"] = self.backend_port_profile_id
        return payload

    async def proxy_request(
        self,
        api_token: str,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        *,
        request_payload: Any | None = None,
        response_transformer: ResponseTransformer | None = None,
        disconnect_waiter: Any | None = None,
    ) -> ForwardResult:
        result = await super().proxy_request(
            api_token=api_token,
            method=method,
            path=path,
            headers=headers,
            body=body,
            request_payload=request_payload,
            response_transformer=response_transformer,
            disconnect_waiter=disconnect_waiter,
        )
        with self._lock:
            run = self.active_runs.get(api_token)
            if run is not None:
                self._annotate_request_records(run.request_records)
        return result

    def get_context_usage_summary(self) -> dict[str, Any]:
        payload = super().get_context_usage_summary()
        payload["backend_port_profile_id"] = self.backend_port_profile_id
        for agent in payload["agents"]:
            agent["backend_port_profile_id"] = self.backend_port_profile_id
        return payload

    def get_ongoing_context_tokens(self) -> int:
        with self._lock:
            return sum(
                run.current_context_tokens
                for run in self.active_runs.values()
                if run.schedule_state == "ongoing"
            )

    def get_pending_assignment_state(self) -> tuple[int, int]:
        with self._lock:
            pending_runs = [
                run for run in self.active_runs.values() if run.schedule_state == "pending"
            ]
            return (
                len(pending_runs),
                sum(self._effective_context_tokens(run) for run in pending_runs),
            )

    def is_ctx_aware_enabled(self) -> bool:
        with self._lock:
            return self.ctx_aware_enabled

    def get_ctx_aware_summary(self) -> dict[str, Any]:
        payload = super().get_ctx_aware_summary()
        payload["backend_port_profile_id"] = self.backend_port_profile_id
        for agent in payload["agents"]:
            agent["backend_port_profile_id"] = self.backend_port_profile_id
        return payload

    def get_slo_aware_summary(self) -> dict[str, Any]:
        payload = super().get_slo_aware_summary()
        payload["backend_port_profile_id"] = self.backend_port_profile_id
        for agent in payload["agents"]:
            agent["backend_port_profile_id"] = self.backend_port_profile_id
        return payload

    def get_output_throughput_summary(self) -> dict[str, Any]:
        payload = super().get_output_throughput_summary()
        payload["backend_port_profile_id"] = self.backend_port_profile_id
        return payload

    def get_output_throughput_details(self) -> dict[str, Any]:
        payload = super().get_output_throughput_details()
        payload["backend_port_profile_id"] = self.backend_port_profile_id
        for agent in payload["agents"]:
            agent["backend_port_profile_id"] = self.backend_port_profile_id
        return payload

    def _job_ctx_aware_log_path_for_output(
        self,
        output_path: Path,
        *,
        job_started_at: str,
    ) -> Path:
        job_dir = output_path / CTX_AWARE_JOB_LOG_DIRNAME
        file_name = (
            f"ctx_aware_{iso8601_to_compact(job_started_at)}"
            f"_profile-{self.backend_port_profile_id}.jsonl"
        )
        return job_dir / file_name

    def _job_slo_aware_decision_log_path_for_output(
        self,
        output_path: Path,
        *,
        job_started_at: str,
    ) -> Path:
        job_dir = output_path / CTX_AWARE_JOB_LOG_DIRNAME
        file_name = (
            f"slo_aware_decisions_{iso8601_to_compact(job_started_at)}"
            f"_profile-{self.backend_port_profile_id}.jsonl"
        )
        return job_dir / file_name

    def end_job(
        self,
        status: str,
        *,
        flush_traces: bool = True,
        wait_for_traces: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            if not self.job_active:
                raise ValueError("job is not active")
            if self.active_runs:
                raise RuntimeError("cannot end job while agents are active")
            if not self.job_output_location:
                raise ValueError("job output location missing")

            ended_runs = list(self.completed_runs)
            job_ended_at = now_iso8601_utc()
            output_location = self.job_output_location
            job_log_path = (
                str(self.ctx_aware_job_log_path)
                if self.ctx_aware_job_log_path is not None
                else None
            )
            slo_decision_log_path = (
                str(self.slo_aware_decision_log_path)
                if self.slo_aware_decision_log_path is not None
                else None
            )

            for run in ended_runs:
                run.lifecycle_events.append(
                    {
                        "event_type": "job_end",
                        "timestamp": job_ended_at,
                        "trace_id": run.trace_id,
                        "api_token_hash": run.api_token_hash,
                        "metadata": {
                            "status": status,
                            "backend_port_profile_id": self.backend_port_profile_id,
                        },
                    }
                )

            self._write_ctx_aware_job_log_sample_locked(sample_time=job_ended_at)
            if flush_traces:
                self.force_flush_traces()
            if wait_for_traces and self.config.job_end_trace_wait_seconds > 0:
                time.sleep(self.config.job_end_trace_wait_seconds)

            artifacts: list[dict[str, Any]] = []
            for run in ended_runs:
                jaeger_payload = self.jaeger_fetcher(run.trace_id)
                artifact_summary = self._write_artifact(output_location, run, jaeger_payload)
                artifacts.append(artifact_summary)

            self.job_active = False
            self.job_started_at = None
            self.job_output_location = None
            self.completed_runs = []
            self._rebalance_ctx_aware_locked()
            self._close_ctx_aware_job_log_locked()

            return {
                "status": "ok",
                "job_end_status": status,
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
                "ctx_aware_job_log_path": job_log_path,
                "slo_aware_decision_log_path": slo_decision_log_path,
            }

    def _write_artifact(
        self,
        output_location: str,
        run: Any,
        jaeger_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not run.run_end_time:
            raise ValueError("run_end_time missing")
        if run.return_code is None:
            raise ValueError("return_code missing")

        output_dir = parse_output_location(output_location)
        output_dir.mkdir(parents=True, exist_ok=True)

        compact_start = iso8601_to_compact(run.run_start_time)
        artifact_base_name = (
            f"run_{compact_start}_{run.api_token_hash[:12]}_{run.trace_id}"
        )

        with tempfile.TemporaryDirectory(prefix="gateway_multi_artifact_") as temp_dir:
            root = Path(temp_dir)
            trace_dir = root / "trace"
            events_dir = root / "events"
            requests_dir = root / "requests"
            trace_dir.mkdir(parents=True, exist_ok=True)
            events_dir.mkdir(parents=True, exist_ok=True)
            requests_dir.mkdir(parents=True, exist_ok=True)

            jaeger_path = trace_dir / "jaeger_trace.json"
            lifecycle_path = events_dir / "lifecycle.jsonl"
            requests_path = requests_dir / "model_inference.jsonl"

            request_records = list(run.request_records)
            self._annotate_request_records(request_records)

            write_json_file(jaeger_path, jaeger_payload)
            write_jsonl_file(lifecycle_path, run.lifecycle_events)
            write_jsonl_file(requests_path, request_records)

            artifact_files: list[dict[str, Any]] = []
            for relative_path, file_path in [
                ("trace/jaeger_trace.json", jaeger_path),
                ("events/lifecycle.jsonl", lifecycle_path),
                ("requests/model_inference.jsonl", requests_path),
            ]:
                digest, size_bytes = file_sha256_and_size(file_path)
                artifact_files.append(
                    {"path": relative_path, "sha256": digest, "size_bytes": size_bytes}
                )

            manifest = {
                "schema_version": "v1",
                "generated_at": now_iso8601_utc(),
                "trace_id": run.trace_id,
                "api_token_hash": run.api_token_hash,
                "backend_port_profile_id": self.backend_port_profile_id,
                "run_start_time": run.run_start_time,
                "run_end_time": run.run_end_time,
                "return_code": run.return_code,
                "request_count": len(request_records),
                "model_inference_span_count": len(request_records),
                "artifact_files": artifact_files,
            }
            manifest_path = root / "manifest.json"
            write_json_file(manifest_path, manifest)

            artifact_format = self.config.artifact_compression
            if artifact_format == "tar.gz":
                artifact_path = self._make_unique_path(
                    output_dir / f"{artifact_base_name}.tar.gz"
                )
                with tarfile.open(artifact_path, "w:gz") as archive:
                    archive.add(manifest_path, arcname="manifest.json")
                    archive.add(jaeger_path, arcname="trace/jaeger_trace.json")
                    archive.add(lifecycle_path, arcname="events/lifecycle.jsonl")
                    archive.add(requests_path, arcname="requests/model_inference.jsonl")
            else:
                artifact_path = self._make_unique_path(output_dir / artifact_base_name)
                artifact_path.mkdir(parents=True, exist_ok=True)
                (artifact_path / "trace").mkdir(parents=True, exist_ok=True)
                (artifact_path / "events").mkdir(parents=True, exist_ok=True)
                (artifact_path / "requests").mkdir(parents=True, exist_ok=True)

                shutil.copy2(manifest_path, artifact_path / "manifest.json")
                shutil.copy2(jaeger_path, artifact_path / "trace" / "jaeger_trace.json")
                shutil.copy2(lifecycle_path, artifact_path / "events" / "lifecycle.jsonl")
                shutil.copy2(
                    requests_path,
                    artifact_path / "requests" / "model_inference.jsonl",
                )

        return {
            "trace_id": run.trace_id,
            "api_token_hash": run.api_token_hash,
            "backend_port_profile_id": self.backend_port_profile_id,
            "path": str(artifact_path),
            "artifact_format": artifact_format,
            "request_count": len(request_records),
        }


@dataclass(frozen=True)
class BackendBinding:
    profile: PortProfile
    service: BackendGatewayService


class AssignmentPolicyRequest(BaseModel):
    assignment_policy: str = Field(min_length=1)
    balanced_usage_threshold_tokens: int | None = None


class GatewayMultiService:
    def __init__(
        self,
        backends: Sequence[BackendBinding],
        *,
        assignment_policy: str,
        balanced_usage_threshold_tokens: int = DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS,
    ) -> None:
        if not backends:
            raise ValueError("gateway_multi requires at least one backend profile")
        self.backends = list(backends)
        self.assignment_policy = normalize_assignment_policy(assignment_policy)
        self.balanced_usage_threshold_tokens = normalize_balanced_usage_threshold_tokens(
            balanced_usage_threshold_tokens
        )
        self._lock = Lock()
        self._next_backend_index = 0
        self._active_backend_by_token: dict[str, BackendBinding] = {}

    @property
    def control_backend(self) -> BackendBinding:
        return self.backends[0]

    def get_backend(self, profile_id: int | str) -> BackendBinding:
        normalized = str(profile_id)
        for backend in self.backends:
            if backend.profile.profile_id == normalized:
                return backend
        raise KeyError(f"unknown backend port profile id: {normalized}")

    def _job_active(self) -> bool:
        return any(backend.service.job_active for backend in self.backends)

    def _active_agent_count(self) -> int:
        return sum(len(backend.service.active_runs) for backend in self.backends)

    def _assignment_policy_requires_ctx_aware(self) -> bool:
        return self.assignment_policy == "lowest_profile_without_pending"

    def _ctx_aware_enabled_for_all_backends(self) -> bool:
        return all(backend.service.is_ctx_aware_enabled() for backend in self.backends)

    def _backend_ongoing_context_usages(self) -> list[tuple[int, int, BackendBinding]]:
        return [
            (backend.service.get_ongoing_context_tokens(), index, backend)
            for index, backend in enumerate(self.backends)
        ]

    def _select_lowest_usage_backend(
        self,
        backend_usages: Sequence[tuple[int, int, BackendBinding]],
    ) -> BackendBinding:
        if not backend_usages:
            raise ValueError("backend usage candidates must not be empty")
        lowest_usage = min(usage for usage, _, _ in backend_usages)
        candidate_indices = {
            index for usage, index, _ in backend_usages if usage == lowest_usage
        }
        start_index = self._next_backend_index % len(self.backends)
        for offset in range(len(self.backends)):
            candidate_index = (start_index + offset) % len(self.backends)
            if candidate_index in candidate_indices:
                return self.backends[candidate_index]
        raise RuntimeError("failed to choose a backend from the lowest-usage candidates")

    def _select_backend_for_new_agent(self) -> BackendBinding:
        if self.assignment_policy == "round_robin":
            return self.backends[self._next_backend_index % len(self.backends)]
        if self.assignment_policy == "lowest_usage":
            return self._select_lowest_usage_backend(
                self._backend_ongoing_context_usages()
            )
        if self.assignment_policy == "balanced":
            backend_usages = self._backend_ongoing_context_usages()
            active_under_threshold = [
                item
                for item in backend_usages
                if 0 < item[0] < self.balanced_usage_threshold_tokens
            ]
            if active_under_threshold:
                return self._select_lowest_usage_backend(active_under_threshold)
            return self._select_lowest_usage_backend(backend_usages)
        if self.assignment_policy == "lowest_profile_without_pending":
            ordered_backends = sorted(
                self.backends,
                key=lambda backend: _port_profile_sort_key(backend.profile.profile_id),
            )
            pending_states = [
                (
                    backend.service.get_pending_assignment_state(),
                    backend,
                )
                for backend in ordered_backends
            ]
            for (pending_agent_count, _pending_context_tokens), backend in pending_states:
                if pending_agent_count == 0:
                    return backend
            return min(
                pending_states,
                key=lambda item: (
                    item[0][1],
                    _port_profile_sort_key(item[1].profile.profile_id),
                ),
            )[1]
        raise ValueError(f"unsupported assignment policy: {self.assignment_policy}")

    def _assign_backend_to_agent(
        self,
        api_token: str,
        backend: BackendBinding,
        *,
        reason: str,
        previous_port_profile_id: str | None = None,
    ) -> None:
        self._active_backend_by_token[api_token] = backend
        backend.service.record_port_profile_assignment(
            api_token,
            assignment_policy=self.assignment_policy,
            reason=reason,
            previous_port_profile_id=previous_port_profile_id,
        )

    async def start_background_tasks(self) -> None:
        await asyncio.gather(
            *(backend.service.start_background_tasks() for backend in self.backends)
        )

    async def stop_background_tasks(self) -> None:
        await asyncio.gather(
            *(backend.service.stop_background_tasks() for backend in self.backends)
        )

    def get_ctx_aware_summary(self) -> dict[str, Any]:
        summaries = [backend.service.get_ctx_aware_summary() for backend in self.backends]
        combined_agents = sorted(
            [agent for summary in summaries for agent in summary["agents"]],
            key=lambda agent: (
                agent.get("run_start_time") or "",
                str(agent.get("backend_port_profile_id") or ""),
                agent.get("api_token_hash") or "",
                agent.get("trace_id") or "",
            ),
        )
        first = summaries[0]
        return {
            "status": "ok",
            "enabled": first["enabled"],
            "usage_threshold_tokens": first["usage_threshold_tokens"],
            "scheduling_threshold_tokens": first["scheduling_threshold_tokens"],
            "policy_mode": first["policy_mode"],
            "new_agent_pseudo_tokens": CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS,
            "scheduler_interval_hz": CTX_AWARE_SCHEDULER_INTERVAL_HZ,
            "ongoing_agent_count": sum(
                summary["ongoing_agent_count"] for summary in summaries
            ),
            "pending_agent_count": sum(
                summary["pending_agent_count"] for summary in summaries
            ),
            "ongoing_effective_context_tokens": sum(
                summary["ongoing_effective_context_tokens"] for summary in summaries
            ),
            "pending_effective_context_tokens": sum(
                summary["pending_effective_context_tokens"] for summary in summaries
            ),
            "backend_port_profile_ids": [
                backend.profile.profile_id for backend in self.backends
            ],
            "control_port_profile_id": self.control_backend.profile.profile_id,
            "backends": summaries,
            "agents": combined_agents,
        }

    @staticmethod
    def _round_metric(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 6)

    def get_slo_aware_summary(self) -> dict[str, Any]:
        summaries = [backend.service.get_slo_aware_summary() for backend in self.backends]
        combined_agents = sorted(
            [agent for summary in summaries for agent in summary["agents"]],
            key=lambda agent: (
                agent.get("run_start_time") or "",
                str(agent.get("backend_port_profile_id") or ""),
                agent.get("api_token_hash") or "",
                agent.get("trace_id") or "",
            ),
        )
        throughput_values = [
            float(agent["output_tokens_per_s"])
            for agent in combined_agents
            if isinstance(agent.get("output_tokens_per_s"), (int, float))
        ]
        min_output_tokens_per_s: float | None = None
        avg_output_tokens_per_s: float | None = None
        if throughput_values:
            min_output_tokens_per_s = min(throughput_values)
            avg_output_tokens_per_s = sum(throughput_values) / len(throughput_values)
        first = summaries[0]
        return {
            "status": "ok",
            "enabled": first["enabled"],
            "requires_ctx_aware": True,
            "ctx_aware_enabled": first["ctx_aware_enabled"],
            "target_tokens_per_s": first["target_tokens_per_s"],
            "policy_mode": first["policy_mode"],
            "ralexation_agent_count": sum(
                summary["ralexation_agent_count"] for summary in summaries
            ),
            "ralexation_effective_context_tokens": sum(
                summary["ralexation_effective_context_tokens"]
                for summary in summaries
            ),
            "min_output_tokens_per_s": self._round_metric(min_output_tokens_per_s),
            "avg_output_tokens_per_s": self._round_metric(avg_output_tokens_per_s),
            "backend_port_profile_ids": [
                backend.profile.profile_id for backend in self.backends
            ],
            "control_port_profile_id": self.control_backend.profile.profile_id,
            "backends": summaries,
            "agents": combined_agents,
        }

    def get_policy_summary(self) -> dict[str, Any]:
        with self._lock:
            assignment_policy = self.assignment_policy
            balanced_usage_threshold_tokens = self.balanced_usage_threshold_tokens
            job_active = self._job_active()
        return {
            "status": "ok",
            "assignment_policy": assignment_policy,
            "balanced_usage_threshold_tokens": balanced_usage_threshold_tokens,
            "supported_assignment_policies": sorted(SUPPORTED_ASSIGNMENT_POLICIES),
            "job_active": job_active,
            "backend_port_profile_ids": [
                backend.profile.profile_id for backend in self.backends
            ],
            "control_port_profile_id": self.control_backend.profile.profile_id,
        }

    def set_assignment_policy(
        self,
        assignment_policy: str,
        *,
        balanced_usage_threshold_tokens: int | None = None,
    ) -> dict[str, Any]:
        normalized_policy = normalize_assignment_policy(assignment_policy)
        normalized_threshold = (
            normalize_balanced_usage_threshold_tokens(balanced_usage_threshold_tokens)
            if balanced_usage_threshold_tokens is not None
            else None
        )
        with self._lock:
            if self._job_active():
                raise RuntimeError("assignment policy cannot be changed while a job is active")
            self.assignment_policy = normalized_policy
            if normalized_threshold is not None:
                self.balanced_usage_threshold_tokens = normalized_threshold
            self._next_backend_index = 0
        return self.get_policy_summary()

    def start_ctx_aware(
        self,
        *,
        usage_threshold_tokens: int,
        scheduling_threshold_tokens: int,
        policy_mode: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._job_active():
                raise RuntimeError("ctx-aware mode cannot be changed while a job is active")
            backends = list(self.backends)

        for backend in backends:
            backend.service.start_ctx_aware(
                usage_threshold_tokens=usage_threshold_tokens,
                scheduling_threshold_tokens=scheduling_threshold_tokens,
                policy_mode=policy_mode,
            )
        return self.get_ctx_aware_summary()

    def end_ctx_aware(self) -> dict[str, Any]:
        with self._lock:
            if self._job_active():
                raise RuntimeError("ctx-aware mode cannot be changed while a job is active")
            backends = list(self.backends)

        for backend in backends:
            backend.service.end_ctx_aware()
        return self.get_ctx_aware_summary()

    def start_slo_aware(
        self,
        *,
        target_tokens_per_s: float,
        policy_mode: str,
    ) -> dict[str, Any]:
        with self._lock:
            if self._job_active():
                raise RuntimeError("slo-aware mode cannot be changed while a job is active")
            backends = list(self.backends)

        for backend in backends:
            backend.service.start_slo_aware(
                target_tokens_per_s=target_tokens_per_s,
                policy_mode=policy_mode,
            )
        return self.get_slo_aware_summary()

    def end_slo_aware(self) -> dict[str, Any]:
        with self._lock:
            if self._job_active():
                raise RuntimeError("slo-aware mode cannot be changed while a job is active")
            backends = list(self.backends)

        for backend in backends:
            backend.service.end_slo_aware()
        return self.get_slo_aware_summary()

    def start_job(self, output_location: str) -> dict[str, Any]:
        normalized_output = str(parse_output_location(output_location))
        ctx_aware_job_log_paths: dict[str, str] = {}
        slo_aware_decision_log_paths: dict[str, str] = {}
        with self._lock:
            if self._job_active():
                raise RuntimeError("job already active")
            if (
                self._assignment_policy_requires_ctx_aware()
                and not self._ctx_aware_enabled_for_all_backends()
            ):
                raise RuntimeError(
                    "assignment policy lowest_profile_without_pending requires ctx-aware "
                    "mode to be enabled before starting a job"
                )
            self._next_backend_index = 0
            self._active_backend_by_token = {}
            for backend in self.backends:
                payload = backend.service.start_job(normalized_output)
                job_log_path = payload.get("ctx_aware_job_log_path")
                if isinstance(job_log_path, str):
                    ctx_aware_job_log_paths[backend.profile.profile_id] = job_log_path
                slo_log_path = payload.get("slo_aware_decision_log_path")
                if isinstance(slo_log_path, str):
                    slo_aware_decision_log_paths[backend.profile.profile_id] = slo_log_path

        return {
            "status": "ok",
            "job_started_at": self.control_backend.service.job_started_at,
            "output_location": normalized_output,
            "assignment_policy": self.assignment_policy,
            "backend_port_profile_ids": [
                backend.profile.profile_id for backend in self.backends
            ],
            "control_port_profile_id": self.control_backend.profile.profile_id,
            "ctx_aware_job_log_paths": ctx_aware_job_log_paths,
            "slo_aware_decision_log_paths": slo_aware_decision_log_paths,
        }

    def start_agent(self, api_token: str) -> dict[str, Any]:
        with self._lock:
            if api_token in self._active_backend_by_token:
                raise RuntimeError("agent already started")
            backend = self._select_backend_for_new_agent()
            payload = backend.service.start_agent(api_token)
            self._assign_backend_to_agent(
                api_token,
                backend,
                reason="initial_assignment",
            )
            self._next_backend_index += 1
            payload["assignment_policy"] = self.assignment_policy
            return payload

    async def end_agent(self, api_token: str, return_code: int) -> dict[str, Any]:
        with self._lock:
            backend = self._active_backend_by_token.get(api_token)
        if backend is None:
            raise KeyError("agent not started")
        payload = await backend.service.end_agent(api_token, return_code)
        with self._lock:
            self._active_backend_by_token.pop(api_token, None)
        payload["assignment_policy"] = self.assignment_policy
        return payload

    async def proxy_request(
        self,
        api_token: str,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        *,
        request_payload: Any | None = None,
        response_transformer: ResponseTransformer | None = None,
        disconnect_waiter: Any | None = None,
    ) -> ForwardResult:
        with self._lock:
            backend = self._active_backend_by_token.get(api_token)
        if backend is None:
            raise KeyError("agent not started")
        return await backend.service.proxy_request(
            api_token=api_token,
            method=method,
            path=path,
            headers=headers,
            body=body,
            request_payload=request_payload,
            response_transformer=response_transformer,
            disconnect_waiter=disconnect_waiter,
        )

    def end_job(self, status: str) -> dict[str, Any]:
        with self._lock:
            if not self._job_active():
                raise ValueError("job is not active")
            if self._active_agent_count() > 0:
                raise RuntimeError("cannot end job while agents are active")
            backends = list(self.backends)

        for backend in backends:
            backend.service.force_flush_traces()

        wait_seconds = max(
            backend.service.config.job_end_trace_wait_seconds for backend in backends
        )
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        artifacts: list[dict[str, Any]] = []
        ctx_aware_job_log_paths: dict[str, str] = {}
        slo_aware_decision_log_paths: dict[str, str] = {}
        for backend in backends:
            result = backend.service.end_job(
                status,
                flush_traces=False,
                wait_for_traces=False,
            )
            artifacts.extend(result["artifacts"])
            job_log_path = result.get("ctx_aware_job_log_path")
            if isinstance(job_log_path, str):
                ctx_aware_job_log_paths[backend.profile.profile_id] = job_log_path
            slo_log_path = result.get("slo_aware_decision_log_path")
            if isinstance(slo_log_path, str):
                slo_aware_decision_log_paths[backend.profile.profile_id] = slo_log_path

        with self._lock:
            self._active_backend_by_token = {}

        return {
            "status": "ok",
            "job_end_status": status,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "assignment_policy": self.assignment_policy,
            "backend_port_profile_ids": [
                backend.profile.profile_id for backend in self.backends
            ],
            "control_port_profile_id": self.control_backend.profile.profile_id,
            "ctx_aware_job_log_paths": ctx_aware_job_log_paths,
            "slo_aware_decision_log_paths": slo_aware_decision_log_paths,
        }


def create_gateway_service(
    *,
    runtime_settings: GatewayMultiRuntimeSettings,
    profiles: Sequence[PortProfile] | None = None,
    forwarders_by_profile: dict[str, Forwarder] | None = None,
    jaeger_fetchers_by_profile: dict[str, JaegerFetcher] | None = None,
) -> GatewayMultiService:
    resolved_profiles = list(
        profiles
        if profiles is not None
        else [
            load_port_profile(profile_id)
            for profile_id in runtime_settings.port_profile_ids
        ]
    )
    bindings: list[BackendBinding] = []
    for profile in resolved_profiles:
        config = build_backend_config(profile, runtime_settings)
        tracer_provider, tracer = configure_backend_tracer(
            config.service_name,
            endpoint=config.otlp_traces_endpoint,
            insecure=config.otlp_traces_insecure,
        )
        service = BackendGatewayService(
            profile=profile,
            config=config,
            tracer_provider=tracer_provider,
            tracer=tracer,
            forwarder=(forwarders_by_profile or {}).get(profile.profile_id),
            jaeger_fetcher=(jaeger_fetchers_by_profile or {}).get(profile.profile_id),
        )
        bindings.append(BackendBinding(profile=profile, service=service))
    return GatewayMultiService(
        bindings,
        assignment_policy=runtime_settings.assignment_policy,
        balanced_usage_threshold_tokens=runtime_settings.balanced_usage_threshold_tokens,
    )


def _build_reasoning_transformer(
    *,
    model_registry: ModelRegistry | None,
) -> ResponseTransformer:
    registry = model_registry or load_model_registry()
    parser = ReasoningResponseParser(registry)

    def transform(path: str, request_payload: Any, result: ForwardResult) -> ForwardResult:
        if result.response_json is None:
            return result
        transformed_payload = parser.transform(path, request_payload, result.response_json)
        if transformed_payload is result.response_json:
            return result
        transformed_content = json.dumps(transformed_payload, ensure_ascii=True).encode("utf-8")
        return ForwardResult(
            status_code=result.status_code,
            content=transformed_content,
            content_type="application/json",
            response_json=transformed_payload,
            error_message=result.error_message,
        )

    return transform


def create_control_app(
    service: GatewayMultiService,
    *,
    gateway_parse_port: int | None = None,
    model_registry: ModelRegistry | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        await service.start_background_tasks()
        try:
            yield
        finally:
            await service.stop_background_tasks()

    app = FastAPI(title="vLLM Gateway Multi", version="0.1.0", lifespan=lifespan)
    app.state.gateway_multi_service = service
    app.state.gateway_parse_port = gateway_parse_port
    app.state.response_transformer = (
        _build_reasoning_transformer(model_registry=model_registry)
        if gateway_parse_port is not None
        else None
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ctx-aware")
    async def ctx_aware_status() -> dict[str, Any]:
        return service.get_ctx_aware_summary()

    @app.get("/policy")
    async def policy_status() -> dict[str, Any]:
        return service.get_policy_summary()

    @app.post("/policy")
    async def policy_update(payload: AssignmentPolicyRequest) -> dict[str, Any]:
        try:
            return service.set_assignment_policy(
                payload.assignment_policy,
                balanced_usage_threshold_tokens=payload.balanced_usage_threshold_tokens,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/ctx-aware/start")
    async def ctx_aware_start(payload: CtxAwareStartRequest) -> dict[str, Any]:
        try:
            return service.start_ctx_aware(
                usage_threshold_tokens=payload.usage_threshold_tokens,
                scheduling_threshold_tokens=payload.scheduling_threshold_tokens,
                policy_mode=payload.policy_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/ctx-aware/end")
    async def ctx_aware_end() -> dict[str, Any]:
        try:
            return service.end_ctx_aware()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/slo-aware")
    async def slo_aware_status() -> dict[str, Any]:
        return service.get_slo_aware_summary()

    @app.post("/slo-aware/start")
    async def slo_aware_start(payload: SloAwareStartRequest) -> dict[str, Any]:
        try:
            return service.start_slo_aware(
                target_tokens_per_s=payload.target_tokens_per_s,
                policy_mode=payload.policy_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/slo-aware/end")
    async def slo_aware_end() -> dict[str, Any]:
        try:
            return service.end_slo_aware()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/job/start")
    async def job_start(payload: JobStartRequest) -> dict[str, Any]:
        try:
            return service.start_job(payload.output_location)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/job/end")
    async def job_end(payload: JobEndRequest) -> dict[str, Any]:
        try:
            return service.end_job(payload.status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/agent/start")
    async def agent_start(payload: AgentStartRequest) -> dict[str, Any]:
        try:
            return service.start_agent(payload.api_token)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/agent/end")
    async def agent_end(payload: AgentEndRequest) -> dict[str, Any]:
        try:
            return await service.end_agent(
                api_token=payload.api_token,
                return_code=payload.return_code,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    async def proxy_v1(path: str, request: Request) -> Response:
        token = extract_api_token(dict(request.headers))
        if not token:
            raise HTTPException(
                status_code=401,
                detail="missing api token (Authorization: Bearer ... or x-api-key)",
            )

        body = await request.body()
        request_payload = parse_json_if_possible(body)
        query = request.url.query
        target_path = f"v1/{path}"
        if query:
            target_path = f"{target_path}?{query}"

        listener_port = listener_port_from_request(request)
        response_transformer: ResponseTransformer | None = None
        if listener_port == app.state.gateway_parse_port:
            response_transformer = app.state.response_transformer

        async def wait_for_disconnect() -> None:
            while True:
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.1)

        try:
            result = await service.proxy_request(
                api_token=token,
                method=request.method,
                path=target_path,
                headers=dict(request.headers),
                body=body,
                request_payload=request_payload,
                response_transformer=response_transformer,
                disconnect_waiter=wait_for_disconnect,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        response_headers: dict[str, str] = {}
        if result.content_type:
            response_headers["content-type"] = result.content_type
        return Response(
            status_code=result.status_code,
            content=result.content,
            headers=response_headers,
        )

    return app


def create_ipc_app(service: BackendGatewayService) -> FastAPI:
    app = FastAPI(title="vLLM Gateway Multi IPC", version="0.1.0")
    app.state.gateway_backend_service = service

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ipc/context")
    async def ipc_context() -> dict[str, Any]:
        return service.get_context_usage_summary()

    @app.get("/ipc/output-throughput")
    async def ipc_output_throughput() -> dict[str, Any]:
        return service.get_output_throughput_summary()

    @app.get("/ipc/output-throughput/agents")
    async def ipc_output_throughput_agents() -> dict[str, Any]:
        return service.get_output_throughput_details()

    return app
