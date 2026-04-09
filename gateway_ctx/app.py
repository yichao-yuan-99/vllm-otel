from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Awaitable, Callable, Literal, TextIO
from urllib.parse import unquote, urlparse

import httpx
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel, Field

from gateway_ctx.model_configs import ModelRegistry, load_model_registry
from gateway_ctx.port_profiles import load_port_profile
from gateway_ctx.reasoning_response_parser import ReasoningResponseParser
from gateway_ctx.runtime_config import GatewayRuntimeSettings, load_runtime_settings


LOGGER = logging.getLogger(__name__)

CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS = 3000
CTX_AWARE_SCHEDULER_INTERVAL_S = 0.2
CTX_AWARE_SCHEDULER_INTERVAL_HZ = int(1 / CTX_AWARE_SCHEDULER_INTERVAL_S)
CTX_AWARE_JOB_LOG_DIRNAME = "job"
CTX_AWARE_POLICY_MODE_AGE = "age"
CTX_AWARE_POLICY_MODE_THROUGHPUT = "throughput"
SLO_AWARE_POLICY_MODE_PUSH_BACK_HALF_SLACK = "push-back-half-slack"

CtxAwarePolicyMode = Literal["age", "throughput"]
SloAwarePolicyMode = Literal["push-back-half-slack"]
ScheduleState = Literal["ongoing", "pending", "ralexation"]


def now_iso8601_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def datetime_to_iso8601_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def iso8601_to_compact(iso_value: str) -> str:
    dt = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%dT%H%M%SZ")


def parse_output_location(output_location: str) -> Path:
    if output_location.startswith("file://"):
        parsed = urlparse(output_location)
        if parsed.scheme != "file":
            raise ValueError("Only file:// URI is supported for output_location")
        path_value = unquote(parsed.path)
    elif "://" in output_location:
        raise ValueError("Only local path or file:// URI is supported for output_location")
    else:
        path_value = output_location

    if not path_value:
        raise ValueError("output_location cannot be empty")
    return Path(path_value).expanduser().resolve()


def normalize_artifact_compression(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return "none"
    if normalized in {"tar.gz", "tgz", "gz", "gzip", "on", "true", "1"}:
        return "tar.gz"
    raise ValueError(
        "Invalid artifact compression mode. "
        "Use one of: none, tar.gz."
    )


def extract_api_token(headers: dict[str, str]) -> str | None:
    lowered_headers = {key.lower(): value for key, value in headers.items()}
    api_key = (
        lowered_headers.get("x-api-key")
        or lowered_headers.get("api-key")
        or lowered_headers.get("openai-api-key")
    )
    if api_key:
        return api_key.strip()
    authorization = lowered_headers.get("authorization")
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return None


def parse_json_if_possible(raw_body: bytes) -> Any:
    if not raw_body:
        return {}
    try:
        return json.loads(raw_body.decode("utf-8"))
    except Exception:
        return {"raw_body": raw_body.decode("utf-8", errors="replace")}


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def listener_port_from_request(request: Request) -> int | None:
    scope_server = request.scope.get("server")
    if isinstance(scope_server, tuple) and len(scope_server) >= 2:
        port = scope_server[1]
        if isinstance(port, int):
            return port

    host_header = request.headers.get("host", "")
    if ":" not in host_header:
        return None
    _, _, raw_port = host_header.rpartition(":")
    try:
        return int(raw_port)
    except ValueError:
        return None


def write_json_file(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_jsonl_file(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def file_sha256_and_size(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def configure_tracer(service_name: str, *, endpoint: str, insecure: bool) -> trace.Tracer:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure))
        )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("vllm-gateway-ctx")


@dataclass
class GatewayConfig:
    vllm_base_url: str
    jaeger_api_base_url: str
    otlp_traces_endpoint: str
    service_name: str = "vllm-gateway-ctx"
    otlp_traces_insecure: bool = True
    artifact_compression: str = "none"
    job_end_trace_wait_seconds: float = 10.0
    output_root: str | None = None

    def __post_init__(self) -> None:
        self.artifact_compression = normalize_artifact_compression(
            self.artifact_compression
        )
        if self.job_end_trace_wait_seconds < 0:
            raise ValueError("job_end_trace_wait_seconds must be >= 0")

    @classmethod
    def from_port_profile(
        cls,
        profile_id: int | str | None = None,
        *,
        runtime_settings: GatewayRuntimeSettings | None = None,
    ) -> "GatewayConfig":
        settings = runtime_settings or load_runtime_settings()
        selected_profile_id = profile_id if profile_id is not None else settings.port_profile_id
        profile = load_port_profile(selected_profile_id)
        default_jaeger_api_base_url = f"http://localhost:{profile.jaeger_api_port}/api/traces"
        default_otlp_traces_endpoint = f"grpc://localhost:{profile.jaeger_otlp_port}"
        jaeger_api_base_url_override = (
            os.environ.get("GATEWAY_JAEGER_API_BASE_URL_OVERRIDE", "").strip()
        )
        otlp_traces_endpoint_override = (
            os.environ.get("GATEWAY_OTLP_TRACES_ENDPOINT_OVERRIDE", "").strip()
        )
        return cls(
            vllm_base_url=f"http://localhost:{profile.vllm_port}",
            jaeger_api_base_url=jaeger_api_base_url_override or default_jaeger_api_base_url,
            otlp_traces_endpoint=otlp_traces_endpoint_override or default_otlp_traces_endpoint,
            service_name=settings.service_name,
            otlp_traces_insecure=settings.otlp_traces_insecure,
            artifact_compression=settings.artifact_compression,
            job_end_trace_wait_seconds=settings.job_end_trace_wait_seconds,
            output_root=settings.output_root,
        )


@dataclass
class ForwardResult:
    status_code: int
    content: bytes
    content_type: str | None
    response_json: Any
    error_message: str | None = None


Forwarder = Callable[
    [str, str, dict[str, str], bytes],
    Awaitable[ForwardResult],
]
JaegerFetcher = Callable[[str], dict[str, Any]]
ResponseTransformer = Callable[[str, Any, ForwardResult], ForwardResult]
DisconnectWaiter = Callable[[], Awaitable[None]]


class ClientDisconnectedError(RuntimeError):
    pass


@dataclass
class RunState:
    api_token: str
    api_token_hash: str
    trace_id: str
    run_start_time: str
    root_span: Any
    root_context: Any
    lifecycle_events: list[dict[str, Any]] = field(default_factory=list)
    request_records: list[dict[str, Any]] = field(default_factory=list)
    active_agent_action_span: Any | None = None
    run_end_time: str | None = None
    return_code: int | None = None
    current_context_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_request_duration_s: float = 0.0
    total_ctx_aware_request_duration_s: float = 0.0
    current_output_tokens_per_s: float | None = None
    schedule_state: ScheduleState = "ongoing"
    has_usable_context_usage: bool = False
    pending_since_iso: str | None = None
    pending_since_monotonic: float | None = None
    ralexation_since_iso: str | None = None
    ralexation_until_iso: str | None = None
    ralexation_until_monotonic: float | None = None
    queued_request_started_monotonic: float | None = None
    ready_event: asyncio.Event | None = None
    requests_drained_event: asyncio.Event | None = None
    active_request_count: int = 0
    queued_request_count: int = 0
    cancel_pending_requests: bool = False


@dataclass
class CtxAwareJobLogCounters:
    agents_turned_pending_due_to_context_threshold: int = 0
    agents_turned_ongoing: int = 0
    new_agents_added_as_pending: int = 0
    new_agents_added_as_ongoing: int = 0
    agents_turned_ralexation: int = 0
    agents_left_ralexation_to_pending: int = 0
    agents_left_ralexation_to_ongoing: int = 0

    def reset(self) -> None:
        self.agents_turned_pending_due_to_context_threshold = 0
        self.agents_turned_ongoing = 0
        self.new_agents_added_as_pending = 0
        self.new_agents_added_as_ongoing = 0
        self.agents_turned_ralexation = 0
        self.agents_left_ralexation_to_pending = 0
        self.agents_left_ralexation_to_ongoing = 0


class GatewayService:
    def __init__(
        self,
        config: GatewayConfig,
        tracer: trace.Tracer,
        forwarder: Forwarder | None = None,
        jaeger_fetcher: JaegerFetcher | None = None,
    ) -> None:
        self.config = config
        self.tracer = tracer
        self.forwarder = forwarder or self._forward_to_vllm
        self.jaeger_fetcher = jaeger_fetcher or self._fetch_jaeger_trace

        self._lock = Lock()
        self.job_active = False
        self.job_started_at: str | None = None
        self.job_output_location: str | None = None
        self.active_runs: dict[str, RunState] = {}
        self.completed_runs: list[RunState] = []
        self.ctx_aware_enabled = False
        self.ctx_aware_usage_threshold_tokens: int | None = None
        self.ctx_aware_scheduling_threshold_tokens: int | None = None
        self.ctx_aware_policy_mode: CtxAwarePolicyMode = CTX_AWARE_POLICY_MODE_AGE
        self.ctx_aware_ongoing_agent_count = 0
        self.ctx_aware_pending_agent_count = 0
        self.ctx_aware_ralexation_agent_count = 0
        self.ctx_aware_ongoing_effective_context_tokens = 0
        self.ctx_aware_pending_effective_context_tokens = 0
        self.ctx_aware_ralexation_effective_context_tokens = 0
        self.slo_aware_enabled = False
        self.slo_target_tokens_per_s: float | None = None
        self.slo_policy_mode: SloAwarePolicyMode | None = None
        self.ctx_aware_scheduler_task: asyncio.Task[None] | None = None
        self.ctx_aware_scheduler_wakeup: asyncio.Event | None = None
        self.ctx_aware_job_log_task: asyncio.Task[None] | None = None
        self.ctx_aware_job_log_path: Path | None = None
        self.ctx_aware_job_log_handle: TextIO | None = None
        self.slo_aware_decision_log_path: Path | None = None
        self.slo_aware_decision_log_handle: TextIO | None = None
        self.ctx_aware_job_log_counters = CtxAwareJobLogCounters()

    @staticmethod
    def _make_unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        for index in range(1, 1000):
            candidate = path.parent / f"{path.name}_{index}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not allocate unique artifact path under {path.parent}")

    async def _forward_to_vllm(
        self, method: str, path: str, headers: dict[str, str], body: bytes
    ) -> ForwardResult:
        forward_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        target_url = f"{self.config.vllm_base_url.rstrip('/')}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.request(
                method=method,
                url=target_url,
                headers=forward_headers,
                content=body,
            )
        response_json: Any
        try:
            response_json = response.json()
        except Exception:
            response_json = None
        return ForwardResult(
            status_code=response.status_code,
            content=response.content,
            content_type=response.headers.get("content-type"),
            response_json=response_json,
        )

    async def _forward_with_disconnect_support(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        *,
        disconnect_waiter: DisconnectWaiter | None = None,
    ) -> ForwardResult:
        forward_task = asyncio.create_task(self.forwarder(method, path, headers, body))
        disconnect_task: asyncio.Task[None] | None = None
        if disconnect_waiter is not None:
            disconnect_task = asyncio.create_task(disconnect_waiter())

        try:
            if disconnect_task is None:
                return await forward_task

            done, _ = await asyncio.wait(
                {forward_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done and not forward_task.done():
                forward_task.cancel()
                with suppress(asyncio.CancelledError):
                    await forward_task
                raise ClientDisconnectedError(
                    "downstream client disconnected before the upstream response completed"
                )
            return await forward_task
        except asyncio.CancelledError:
            forward_task.cancel()
            with suppress(asyncio.CancelledError):
                await forward_task
            raise
        finally:
            if disconnect_task is not None and not disconnect_task.done():
                disconnect_task.cancel()

    def _fetch_jaeger_trace(self, trace_id: str) -> dict[str, Any]:
        trace_url = f"{self.config.jaeger_api_base_url.rstrip('/')}/{trace_id}"
        try:
            response = requests.get(trace_url, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            return {
                "error": "jaeger_fetch_failed",
                "trace_id": trace_id,
                "detail": str(exc),
            }

    @staticmethod
    def _force_flush_traces() -> None:
        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            try:
                force_flush(timeout_millis=5000)
            except Exception:
                # Best effort flush; artifact generation should continue even if this fails.
                pass

    @staticmethod
    def _context_tokens_from_response(response_payload: Any) -> int | None:
        if not isinstance(response_payload, dict):
            return None
        usage = response_payload.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt_tokens = int_or_none(usage.get("prompt_tokens"))
        if prompt_tokens is None:
            return None
        completion_tokens = int_or_none(usage.get("completion_tokens"))
        if completion_tokens is None:
            completion_tokens = 0
        return prompt_tokens + completion_tokens

    @staticmethod
    def _completion_tokens_from_response(response_payload: Any) -> int | None:
        if not isinstance(response_payload, dict):
            return None
        usage = response_payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return int_or_none(usage.get("completion_tokens"))

    @staticmethod
    def _output_throughput_from_totals(
        completion_tokens: int,
        llm_request_duration_s: float,
    ) -> float | None:
        if llm_request_duration_s <= 0.0:
            return None
        return completion_tokens / llm_request_duration_s

    @staticmethod
    def _ctx_aware_throughput_from_totals(
        completion_tokens: int,
        llm_request_duration_s: float,
    ) -> float:
        if completion_tokens <= 0 or llm_request_duration_s <= 0.0:
            return 0.0
        return completion_tokens / llm_request_duration_s

    @staticmethod
    def _round_metric(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 6)

    @staticmethod
    def _effective_context_tokens(run: RunState) -> int:
        if run.has_usable_context_usage:
            return run.current_context_tokens
        return CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS

    def _sorted_active_runs(self) -> list[RunState]:
        return sorted(
            self.active_runs.values(),
            key=lambda run: (run.run_start_time, run.api_token_hash, run.trace_id),
        )

    @staticmethod
    def _normalize_ctx_aware_policy_mode(policy_mode: str | None) -> CtxAwarePolicyMode:
        if policy_mode is None:
            return CTX_AWARE_POLICY_MODE_AGE
        normalized = policy_mode.strip().lower()
        if normalized == CTX_AWARE_POLICY_MODE_AGE:
            return CTX_AWARE_POLICY_MODE_AGE
        if normalized == CTX_AWARE_POLICY_MODE_THROUGHPUT:
            return CTX_AWARE_POLICY_MODE_THROUGHPUT
        raise ValueError(
            "policy_mode must be one of: age, throughput"
        )

    @staticmethod
    def _normalize_slo_aware_policy_mode(policy_mode: str | None) -> SloAwarePolicyMode:
        if policy_mode is None:
            raise ValueError("policy_mode is required")
        normalized = policy_mode.strip().lower()
        if normalized == SLO_AWARE_POLICY_MODE_PUSH_BACK_HALF_SLACK:
            return SLO_AWARE_POLICY_MODE_PUSH_BACK_HALF_SLACK
        raise ValueError("policy_mode must be one of: push-back-half-slack")

    def _stored_throughput_runs_locked(self) -> list[RunState]:
        return [
            run
            for run in self.active_runs.values()
            if run.current_output_tokens_per_s is not None
        ]

    def _min_stored_output_tokens_per_s_locked(self) -> float | None:
        values = [
            run.current_output_tokens_per_s
            for run in self._stored_throughput_runs_locked()
            if run.current_output_tokens_per_s is not None
        ]
        if not values:
            return None
        return min(values)

    def _avg_stored_output_tokens_per_s_locked(self) -> float | None:
        values = [
            run.current_output_tokens_per_s
            for run in self._stored_throughput_runs_locked()
            if run.current_output_tokens_per_s is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def _slo_policy_is_active_locked(self) -> bool:
        if (
            not self.slo_aware_enabled
            or self.slo_target_tokens_per_s is None
            or self.slo_policy_mode != SLO_AWARE_POLICY_MODE_PUSH_BACK_HALF_SLACK
        ):
            return False
        min_throughput = self._min_stored_output_tokens_per_s_locked()
        if min_throughput is None:
            return False
        return min_throughput < self.slo_target_tokens_per_s

    def _slo_slack_s_locked(self, run: RunState) -> float | None:
        if not self.slo_aware_enabled or self.slo_target_tokens_per_s is None:
            return None
        if run.current_output_tokens_per_s is None:
            return None
        if run.total_llm_request_duration_s <= 0.0:
            return None
        return (
            run.total_completion_tokens / self.slo_target_tokens_per_s
            - run.total_llm_request_duration_s
        )

    def _build_agent_context_entry(self, run: RunState) -> dict[str, Any]:
        return {
            "api_token_hash": run.api_token_hash,
            "trace_id": run.trace_id,
            "run_start_time": run.run_start_time,
            "schedule_state": run.schedule_state,
            "context_tokens": run.current_context_tokens,
            "effective_context_tokens": self._effective_context_tokens(run),
            "has_usable_context_usage": run.has_usable_context_usage,
            "pending_since": run.pending_since_iso,
            "ralexation_until": run.ralexation_until_iso,
            "output_tokens_per_s": self._round_metric(run.current_output_tokens_per_s),
            "slo_slack_s": self._round_metric(self._slo_slack_s_locked(run)),
        }

    def _ctx_aware_queue_duration_s_locked(
        self,
        run: RunState,
        *,
        now_monotonic: float | None = None,
    ) -> float:
        if run.queued_request_started_monotonic is None:
            return 0.0
        current_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        return max(current_monotonic - run.queued_request_started_monotonic, 0.0)

    def _ctx_aware_decode_throughput_tokens_per_s_locked(
        self,
        run: RunState,
        *,
        include_current_queue_time: bool,
        now_monotonic: float | None = None,
    ) -> float:
        total_duration_s = run.total_ctx_aware_request_duration_s
        if include_current_queue_time:
            total_duration_s += self._ctx_aware_queue_duration_s_locked(
                run,
                now_monotonic=now_monotonic,
            )
        return self._ctx_aware_throughput_from_totals(
            run.total_completion_tokens,
            total_duration_s,
        )

    def _refresh_ctx_aware_totals_locked(self) -> None:
        ongoing_runs = [
            run for run in self.active_runs.values() if run.schedule_state == "ongoing"
        ]
        pending_runs = [
            run for run in self.active_runs.values() if run.schedule_state == "pending"
        ]
        ralexation_runs = [
            run for run in self.active_runs.values() if run.schedule_state == "ralexation"
        ]
        self.ctx_aware_ongoing_agent_count = len(ongoing_runs)
        self.ctx_aware_pending_agent_count = len(pending_runs)
        self.ctx_aware_ralexation_agent_count = len(ralexation_runs)
        self.ctx_aware_ongoing_effective_context_tokens = sum(
            self._effective_context_tokens(run) for run in ongoing_runs
        )
        self.ctx_aware_pending_effective_context_tokens = sum(
            self._effective_context_tokens(run) for run in pending_runs
        )
        self.ctx_aware_ralexation_effective_context_tokens = sum(
            self._effective_context_tokens(run) for run in ralexation_runs
        )

    def _mark_run_pending_locked(self, run: RunState) -> None:
        if run.schedule_state != "pending":
            run.schedule_state = "pending"
            run.pending_since_iso = now_iso8601_utc()
            run.pending_since_monotonic = time.monotonic()
        run.ralexation_since_iso = None
        run.ralexation_until_iso = None
        run.ralexation_until_monotonic = None
        if run.ready_event is not None:
            run.ready_event.clear()

    def _mark_run_ongoing_locked(self, run: RunState) -> None:
        run.schedule_state = "ongoing"
        run.pending_since_iso = None
        run.pending_since_monotonic = None
        run.ralexation_since_iso = None
        run.ralexation_until_iso = None
        run.ralexation_until_monotonic = None
        if run.ready_event is not None:
            run.ready_event.set()

    def _mark_run_ralexation_locked(self, run: RunState, *, duration_s: float) -> None:
        duration_s = max(duration_s, 0.0)
        run.schedule_state = "ralexation"
        run.pending_since_iso = None
        run.pending_since_monotonic = None
        run.ralexation_since_iso = now_iso8601_utc()
        run.ralexation_until_monotonic = time.monotonic() + duration_s
        run.ralexation_until_iso = datetime_to_iso8601_utc(
            datetime.now(timezone.utc) + timedelta(seconds=duration_s)
        )
        if run.ready_event is not None:
            run.ready_event.clear()

    @staticmethod
    def _increment_queued_request_locked(run: RunState) -> None:
        if run.queued_request_count == 0:
            run.queued_request_started_monotonic = time.monotonic()
        run.queued_request_count += 1

    @staticmethod
    def _decrement_queued_request_locked(run: RunState) -> None:
        run.queued_request_count = max(run.queued_request_count - 1, 0)
        if run.queued_request_count == 0:
            run.queued_request_started_monotonic = None

    def _youngest_ongoing_run_locked(self) -> RunState | None:
        ongoing_runs = [
            run for run in self.active_runs.values() if run.schedule_state == "ongoing"
        ]
        if not ongoing_runs:
            return None
        return max(
            ongoing_runs,
            key=lambda run: (run.run_start_time, run.api_token_hash, run.trace_id),
        )

    def _oldest_pending_run_locked(self) -> RunState | None:
        pending_runs = [
            run for run in self.active_runs.values() if run.schedule_state == "pending"
        ]
        if not pending_runs:
            return None
        return min(
            pending_runs,
            key=lambda run: (
                float("inf")
                if run.pending_since_monotonic is None
                else run.pending_since_monotonic,
                run.run_start_time,
                run.api_token_hash,
                run.trace_id,
            ),
        )

    def _highest_throughput_ongoing_run_locked(self) -> RunState | None:
        ongoing_runs = [
            run for run in self.active_runs.values() if run.schedule_state == "ongoing"
        ]
        if not ongoing_runs:
            return None
        return max(
            ongoing_runs,
            key=lambda run: (
                self._ctx_aware_decode_throughput_tokens_per_s_locked(
                    run,
                    include_current_queue_time=False,
                ),
                run.run_start_time,
                run.api_token_hash,
                run.trace_id,
            ),
        )

    def _lowest_throughput_pending_run_locked(self) -> RunState | None:
        pending_runs = [
            run for run in self.active_runs.values() if run.schedule_state == "pending"
        ]
        if not pending_runs:
            return None
        now_monotonic = time.monotonic()
        return min(
            pending_runs,
            key=lambda run: (
                self._ctx_aware_decode_throughput_tokens_per_s_locked(
                    run,
                    include_current_queue_time=True,
                    now_monotonic=now_monotonic,
                ),
                float("inf")
                if run.pending_since_monotonic is None
                else run.pending_since_monotonic,
                run.run_start_time,
                run.api_token_hash,
                run.trace_id,
            ),
        )

    def _demotion_candidate_locked(self) -> RunState | None:
        if self.ctx_aware_policy_mode == CTX_AWARE_POLICY_MODE_THROUGHPUT:
            return self._highest_throughput_ongoing_run_locked()
        return self._youngest_ongoing_run_locked()

    def _promotion_candidate_locked(self) -> RunState | None:
        if self.ctx_aware_policy_mode == CTX_AWARE_POLICY_MODE_THROUGHPUT:
            return self._lowest_throughput_pending_run_locked()
        return self._oldest_pending_run_locked()

    def _ctx_aware_admits_run_locked(self, run: RunState) -> bool:
        if (
            not self.ctx_aware_enabled
            or self.ctx_aware_scheduling_threshold_tokens is None
        ):
            return True
        candidate_tokens = self._effective_context_tokens(run)
        return (
            self.ctx_aware_ongoing_effective_context_tokens + candidate_tokens
            <= self.ctx_aware_scheduling_threshold_tokens
        )

    def _ready_ralexation_runs_locked(self) -> list[tuple[RunState, str]]:
        now_monotonic = time.monotonic()
        release_all = False
        if self.slo_aware_enabled and self.slo_target_tokens_per_s is not None:
            min_throughput = self._min_stored_output_tokens_per_s_locked()
            release_all = (
                min_throughput is not None
                and min_throughput >= self.slo_target_tokens_per_s
            )
        ready_runs = []
        for run in self.active_runs.values():
            if run.schedule_state != "ralexation":
                continue
            if release_all:
                ready_runs.append((run, "slo_recovered"))
                continue
            if (
                run.ralexation_until_monotonic is not None
                and run.ralexation_until_monotonic <= now_monotonic
            ):
                ready_runs.append((run, "timer_expired"))
        return sorted(
            ready_runs,
            key=lambda item: (
                float("inf")
                if item[0].ralexation_until_monotonic is None
                else item[0].ralexation_until_monotonic,
                item[0].run_start_time,
                item[0].api_token_hash,
                item[0].trace_id,
            ),
        )

    def _should_enter_ralexation_locked(self, run: RunState) -> bool:
        if run.schedule_state != "ongoing":
            return False
        if not self._slo_policy_is_active_locked():
            return False
        run_throughput = run.current_output_tokens_per_s
        if run_throughput is None:
            return False
        avg_throughput = self._avg_stored_output_tokens_per_s_locked()
        if avg_throughput is None or self.slo_target_tokens_per_s is None:
            return False
        if run_throughput <= avg_throughput:
            return False
        if run_throughput <= self.slo_target_tokens_per_s:
            return False
        slack_s = self._slo_slack_s_locked(run)
        if slack_s is None or slack_s <= 0.0:
            return False
        return True

    def _rebalance_ctx_aware_locked(self) -> None:
        if (
            not self.ctx_aware_enabled
            or self.ctx_aware_usage_threshold_tokens is None
            or self.ctx_aware_scheduling_threshold_tokens is None
        ):
            for run in self.active_runs.values():
                self._mark_run_ongoing_locked(run)
            self._refresh_ctx_aware_totals_locked()
            return

        self._refresh_ctx_aware_totals_locked()
        while (
            self.ctx_aware_ongoing_effective_context_tokens
            > self.ctx_aware_usage_threshold_tokens
        ):
            demoted_run = self._demotion_candidate_locked()
            if demoted_run is None:
                break
            self._mark_run_pending_locked(demoted_run)
            self.ctx_aware_job_log_counters.agents_turned_pending_due_to_context_threshold += 1
            self._refresh_ctx_aware_totals_locked()

        for ready_run, wake_reason in self._ready_ralexation_runs_locked():
            had_pending = self.ctx_aware_pending_agent_count > 0
            if had_pending:
                self._mark_run_pending_locked(ready_run)
                self.ctx_aware_job_log_counters.agents_left_ralexation_to_pending += 1
                self._write_slo_aware_decision_log_locked(
                    event_type="agent_left_ralexation",
                    run=ready_run,
                    details={
                        "wake_reason": wake_reason,
                        "from_schedule_state": "ralexation",
                        "to_schedule_state": "pending",
                        "resume_disposition": "existing_pending_agent",
                        "ralexation_until": None,
                    },
                )
                self._refresh_ctx_aware_totals_locked()
                continue
            if self._ctx_aware_admits_run_locked(ready_run):
                self._mark_run_ongoing_locked(ready_run)
                self.ctx_aware_job_log_counters.agents_left_ralexation_to_ongoing += 1
                self._write_slo_aware_decision_log_locked(
                    event_type="agent_left_ralexation",
                    run=ready_run,
                    details={
                        "wake_reason": wake_reason,
                        "from_schedule_state": "ralexation",
                        "to_schedule_state": "ongoing",
                        "resume_disposition": "ctx_aware_admitted",
                        "ralexation_until": None,
                    },
                )
            else:
                self._mark_run_pending_locked(ready_run)
                self.ctx_aware_job_log_counters.agents_left_ralexation_to_pending += 1
                self._write_slo_aware_decision_log_locked(
                    event_type="agent_left_ralexation",
                    run=ready_run,
                    details={
                        "wake_reason": wake_reason,
                        "from_schedule_state": "ralexation",
                        "to_schedule_state": "pending",
                        "resume_disposition": "ctx_aware_rejected",
                        "ralexation_until": None,
                    },
                )
            self._refresh_ctx_aware_totals_locked()

        while True:
            pending_run = self._promotion_candidate_locked()
            if pending_run is None:
                break
            candidate_tokens = self._effective_context_tokens(pending_run)
            if (
                self.ctx_aware_ongoing_effective_context_tokens + candidate_tokens
                > self.ctx_aware_scheduling_threshold_tokens
            ):
                break
            self._mark_run_ongoing_locked(pending_run)
            self.ctx_aware_job_log_counters.agents_turned_ongoing += 1
            self._refresh_ctx_aware_totals_locked()

        self._refresh_ctx_aware_totals_locked()

    def _wake_ctx_aware_scheduler(self) -> None:
        if self.ctx_aware_scheduler_wakeup is not None:
            self.ctx_aware_scheduler_wakeup.set()

    def _ctx_aware_status_locked(self) -> dict[str, Any]:
        active_runs = self._sorted_active_runs()
        agents = [self._build_agent_context_entry(run) for run in active_runs]
        return {
            "status": "ok",
            "enabled": self.ctx_aware_enabled,
            "usage_threshold_tokens": self.ctx_aware_usage_threshold_tokens,
            "scheduling_threshold_tokens": self.ctx_aware_scheduling_threshold_tokens,
            "policy_mode": self.ctx_aware_policy_mode,
            "slo_aware_enabled": self.slo_aware_enabled,
            "slo_target_tokens_per_s": self._round_metric(self.slo_target_tokens_per_s),
            "slo_policy_mode": self.slo_policy_mode,
            "new_agent_pseudo_tokens": CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS,
            "scheduler_interval_hz": CTX_AWARE_SCHEDULER_INTERVAL_HZ,
            "ongoing_agent_count": self.ctx_aware_ongoing_agent_count,
            "pending_agent_count": self.ctx_aware_pending_agent_count,
            "ralexation_agent_count": self.ctx_aware_ralexation_agent_count,
            "ongoing_effective_context_tokens": self.ctx_aware_ongoing_effective_context_tokens,
            "pending_effective_context_tokens": self.ctx_aware_pending_effective_context_tokens,
            "ralexation_effective_context_tokens": (
                self.ctx_aware_ralexation_effective_context_tokens
            ),
            "agents": agents,
        }

    def _slo_aware_status_locked(self) -> dict[str, Any]:
        active_runs = self._sorted_active_runs()
        agents = [self._build_agent_context_entry(run) for run in active_runs]
        return {
            "status": "ok",
            "enabled": self.slo_aware_enabled,
            "requires_ctx_aware": True,
            "ctx_aware_enabled": self.ctx_aware_enabled,
            "target_tokens_per_s": self._round_metric(self.slo_target_tokens_per_s),
            "policy_mode": self.slo_policy_mode,
            "ralexation_agent_count": self.ctx_aware_ralexation_agent_count,
            "ralexation_effective_context_tokens": (
                self.ctx_aware_ralexation_effective_context_tokens
            ),
            "min_output_tokens_per_s": self._round_metric(
                self._min_stored_output_tokens_per_s_locked()
            ),
            "avg_output_tokens_per_s": self._round_metric(
                self._avg_stored_output_tokens_per_s_locked()
            ),
            "agents": agents,
        }

    @staticmethod
    def _job_ctx_aware_log_path_for_output(
        output_path: Path,
        *,
        job_started_at: str,
    ) -> Path:
        job_dir = output_path / CTX_AWARE_JOB_LOG_DIRNAME
        file_name = f"ctx_aware_{iso8601_to_compact(job_started_at)}.jsonl"
        return job_dir / file_name

    @staticmethod
    def _job_slo_aware_decision_log_path_for_output(
        output_path: Path,
        *,
        job_started_at: str,
    ) -> Path:
        job_dir = output_path / CTX_AWARE_JOB_LOG_DIRNAME
        file_name = f"slo_aware_decisions_{iso8601_to_compact(job_started_at)}.jsonl"
        return job_dir / file_name

    def _close_ctx_aware_job_log_locked(self) -> None:
        handle = self.ctx_aware_job_log_handle
        self.ctx_aware_job_log_handle = None
        self.ctx_aware_job_log_path = None
        decision_handle = self.slo_aware_decision_log_handle
        self.slo_aware_decision_log_handle = None
        self.slo_aware_decision_log_path = None
        self.ctx_aware_job_log_counters.reset()
        if handle is None:
            pass
        else:
            with suppress(Exception):
                handle.close()
        if decision_handle is not None:
            with suppress(Exception):
                decision_handle.close()

    def _write_slo_aware_decision_log_locked(
        self,
        *,
        event_type: str,
        run: RunState,
        details: dict[str, Any] | None = None,
        sample_time: str | None = None,
    ) -> None:
        handle = self.slo_aware_decision_log_handle
        if handle is None or not self.job_active:
            return
        payload = {
            "timestamp": sample_time or now_iso8601_utc(),
            "event_type": event_type,
            "api_token_hash": run.api_token_hash,
            "trace_id": run.trace_id,
            "schedule_state": run.schedule_state,
            "context_tokens": run.current_context_tokens,
            "effective_context_tokens": self._effective_context_tokens(run),
            "output_tokens_per_s": self._round_metric(run.current_output_tokens_per_s),
            "slo_slack_s": self._round_metric(self._slo_slack_s_locked(run)),
            "slo_target_tokens_per_s": self._round_metric(self.slo_target_tokens_per_s),
            "min_output_tokens_per_s": self._round_metric(
                self._min_stored_output_tokens_per_s_locked()
            ),
            "avg_output_tokens_per_s": self._round_metric(
                self._avg_stored_output_tokens_per_s_locked()
            ),
        }
        if details:
            payload.update(details)
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")
        handle.flush()

    def _write_ctx_aware_job_log_sample_locked(
        self,
        *,
        sample_time: str | None = None,
    ) -> None:
        handle = self.ctx_aware_job_log_handle
        if handle is None or not self.job_active:
            self.ctx_aware_job_log_counters.reset()
            return
        payload = {
            "timestamp": sample_time or now_iso8601_utc(),
            "ongoing_agent_count": self.ctx_aware_ongoing_agent_count,
            "pending_agent_count": self.ctx_aware_pending_agent_count,
            "ralexation_agent_count": self.ctx_aware_ralexation_agent_count,
            "ongoing_effective_context_tokens": self.ctx_aware_ongoing_effective_context_tokens,
            "pending_effective_context_tokens": self.ctx_aware_pending_effective_context_tokens,
            "ralexation_effective_context_tokens": (
                self.ctx_aware_ralexation_effective_context_tokens
            ),
            "slo_aware_enabled": self.slo_aware_enabled,
            "slo_target_tokens_per_s": self._round_metric(self.slo_target_tokens_per_s),
            "slo_policy_mode": self.slo_policy_mode,
            "agents_turned_pending_due_to_context_threshold": (
                self.ctx_aware_job_log_counters.agents_turned_pending_due_to_context_threshold
            ),
            "agents_turned_ongoing": self.ctx_aware_job_log_counters.agents_turned_ongoing,
            "agents_turned_ralexation": self.ctx_aware_job_log_counters.agents_turned_ralexation,
            "agents_left_ralexation_to_pending": (
                self.ctx_aware_job_log_counters.agents_left_ralexation_to_pending
            ),
            "agents_left_ralexation_to_ongoing": (
                self.ctx_aware_job_log_counters.agents_left_ralexation_to_ongoing
            ),
            "new_agents_added_as_pending": (
                self.ctx_aware_job_log_counters.new_agents_added_as_pending
            ),
            "new_agents_added_as_ongoing": (
                self.ctx_aware_job_log_counters.new_agents_added_as_ongoing
            ),
        }
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")
        handle.flush()
        self.ctx_aware_job_log_counters.reset()

    async def _ctx_aware_job_log_loop(self) -> None:
        while True:
            await asyncio.sleep(CTX_AWARE_SCHEDULER_INTERVAL_S)
            try:
                with self._lock:
                    self._write_ctx_aware_job_log_sample_locked()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("failed to write ctx-aware job log sample")

    async def start_background_tasks(self) -> None:
        if self.ctx_aware_scheduler_task is not None and not self.ctx_aware_scheduler_task.done():
            if self.ctx_aware_job_log_task is not None and not self.ctx_aware_job_log_task.done():
                return
        if self.ctx_aware_scheduler_task is None or self.ctx_aware_scheduler_task.done():
            self.ctx_aware_scheduler_wakeup = asyncio.Event()
            self.ctx_aware_scheduler_task = asyncio.create_task(
                self._ctx_aware_scheduler_loop()
            )
        if self.ctx_aware_job_log_task is None or self.ctx_aware_job_log_task.done():
            self.ctx_aware_job_log_task = asyncio.create_task(
                self._ctx_aware_job_log_loop()
            )

    async def stop_background_tasks(self) -> None:
        scheduler_task = self.ctx_aware_scheduler_task
        self.ctx_aware_scheduler_task = None
        self.ctx_aware_scheduler_wakeup = None
        job_log_task = self.ctx_aware_job_log_task
        self.ctx_aware_job_log_task = None
        tasks = [task for task in [scheduler_task, job_log_task] if task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        with self._lock:
            self._close_ctx_aware_job_log_locked()

    async def _ctx_aware_scheduler_loop(self) -> None:
        while True:
            wakeup = self.ctx_aware_scheduler_wakeup
            if wakeup is None:
                return
            try:
                await asyncio.wait_for(
                    wakeup.wait(),
                    timeout=CTX_AWARE_SCHEDULER_INTERVAL_S,
                )
                wakeup.clear()
            except TimeoutError:
                pass
            with self._lock:
                self._rebalance_ctx_aware_locked()

    async def _wait_for_agent_ready(
        self,
        *,
        ready_event: asyncio.Event,
        disconnect_waiter: DisconnectWaiter | None,
    ) -> bool:
        if ready_event.is_set():
            return True
        ready_task = asyncio.create_task(ready_event.wait())
        disconnect_task: asyncio.Task[None] | None = None
        if disconnect_waiter is not None:
            disconnect_task = asyncio.create_task(disconnect_waiter())
        try:
            if disconnect_task is None:
                await ready_task
                return True
            done, _ = await asyncio.wait(
                {ready_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done and not ready_task.done():
                ready_task.cancel()
                with suppress(asyncio.CancelledError):
                    await ready_task
                return False
            await ready_task
            return True
        except asyncio.CancelledError:
            ready_task.cancel()
            with suppress(asyncio.CancelledError):
                await ready_task
            raise
        finally:
            if disconnect_task is not None and not disconnect_task.done():
                disconnect_task.cancel()

    def _start_agent_action_span_locked(self, run: RunState) -> None:
        run.active_agent_action_span = self.tracer.start_span(
            "agent_action",
            context=run.root_context,
            kind=SpanKind.INTERNAL,
        )

    def start_job(self, output_location: str) -> dict[str, Any]:
        output_path = parse_output_location(output_location)
        output_path.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self.job_active:
                raise RuntimeError("job already active")
            self.job_active = True
            self.job_started_at = now_iso8601_utc()
            self.job_output_location = str(output_path)
            self.active_runs = {}
            self.completed_runs = []
            self._rebalance_ctx_aware_locked()
            self._close_ctx_aware_job_log_locked()
            job_log_path = self._job_ctx_aware_log_path_for_output(
                output_path,
                job_started_at=self.job_started_at,
            )
            slo_decision_log_path = self._job_slo_aware_decision_log_path_for_output(
                output_path,
                job_started_at=self.job_started_at,
            )
            job_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.ctx_aware_job_log_path = job_log_path
            self.ctx_aware_job_log_handle = job_log_path.open("w", encoding="utf-8")
            self.slo_aware_decision_log_path = slo_decision_log_path
            self.slo_aware_decision_log_handle = slo_decision_log_path.open(
                "w",
                encoding="utf-8",
            )
            self._write_ctx_aware_job_log_sample_locked(sample_time=self.job_started_at)

        return {
            "status": "ok",
            "job_started_at": self.job_started_at,
            "output_location": str(output_path),
            "ctx_aware_job_log_path": str(job_log_path),
            "slo_aware_decision_log_path": str(slo_decision_log_path),
        }

    @staticmethod
    def _increment_active_request_locked(run: RunState) -> None:
        run.active_request_count += 1
        if run.active_request_count == 1 and run.requests_drained_event is not None:
            run.requests_drained_event.clear()

    @staticmethod
    def _decrement_active_request_locked(run: RunState) -> None:
        run.active_request_count = max(run.active_request_count - 1, 0)
        if run.active_request_count == 0 and run.requests_drained_event is not None:
            run.requests_drained_event.set()

    def _finalize_end_agent_locked(
        self,
        *,
        api_token: str,
        run: RunState,
        return_code: int,
    ) -> dict[str, Any]:
        if run.active_agent_action_span is not None:
            run.active_agent_action_span.end()
            run.active_agent_action_span = None

        run.return_code = return_code
        run.run_end_time = now_iso8601_utc()
        run.lifecycle_events.append(
            {
                "event_type": "agent_end",
                "timestamp": run.run_end_time,
                "trace_id": run.trace_id,
                "api_token_hash": run.api_token_hash,
                "metadata": {"return_code": return_code},
            }
        )
        run.root_span.end()

        del self.active_runs[api_token]
        self.completed_runs.append(run)
        self._rebalance_ctx_aware_locked()
        return {"status": "ok"}

    def start_agent(self, api_token: str) -> dict[str, Any]:
        with self._lock:
            if not self.job_active or not self.job_started_at or not self.job_output_location:
                raise ValueError("job is not active")
            if api_token in self.active_runs:
                raise RuntimeError("agent already started")

            run_started_at = now_iso8601_utc()
            api_token_hash = sha256_hex(api_token)
            root_span = self.tracer.start_span("agent_run", kind=SpanKind.INTERNAL)
            root_context = trace.set_span_in_context(root_span)
            trace_id = f"{root_span.get_span_context().trace_id:032x}"
            requests_drained_event = asyncio.Event()
            requests_drained_event.set()

            run_state = RunState(
                api_token=api_token,
                api_token_hash=api_token_hash,
                trace_id=trace_id,
                run_start_time=run_started_at,
                root_span=root_span,
                root_context=root_context,
                ready_event=asyncio.Event(),
                requests_drained_event=requests_drained_event,
            )
            run_state.lifecycle_events.append(
                {
                    "event_type": "job_start",
                    "timestamp": self.job_started_at,
                    "trace_id": trace_id,
                    "api_token_hash": api_token_hash,
                    "metadata": {"output_location": self.job_output_location},
                }
            )
            run_state.lifecycle_events.append(
                {
                    "event_type": "agent_start",
                    "timestamp": run_started_at,
                    "trace_id": trace_id,
                    "api_token_hash": api_token_hash,
                    "metadata": {},
                }
            )
            if not self._ctx_aware_admits_run_locked(run_state):
                self._mark_run_pending_locked(run_state)
                self.ctx_aware_job_log_counters.new_agents_added_as_pending += 1
            else:
                self._mark_run_ongoing_locked(run_state)
                self.ctx_aware_job_log_counters.new_agents_added_as_ongoing += 1
            self.active_runs[api_token] = run_state
            self._rebalance_ctx_aware_locked()

        self._wake_ctx_aware_scheduler()
        return {"status": "ok", "trace_id": trace_id}

    async def end_agent(self, api_token: str, return_code: int) -> dict[str, Any]:
        while True:
            with self._lock:
                run = self.active_runs.get(api_token)
                if run is None:
                    raise KeyError("agent not started")
                if run.active_request_count == 0:
                    result = self._finalize_end_agent_locked(
                        api_token=api_token,
                        run=run,
                        return_code=return_code,
                    )
                    self._wake_ctx_aware_scheduler()
                    return result
                if (
                    run.queued_request_count > 0
                    and run.active_request_count == run.queued_request_count
                ):
                    run.cancel_pending_requests = True
                    if run.ready_event is not None:
                        run.ready_event.set()
                    requests_drained_event = run.requests_drained_event
                else:
                    raise RuntimeError("cannot end agent while requests are active")

            if requests_drained_event is None:
                raise RuntimeError("agent missing requests drained event")
            await requests_drained_event.wait()

    def get_context_usage_summary(self) -> dict[str, Any]:
        with self._lock:
            active_runs = self._sorted_active_runs()
            agents = [self._build_agent_context_entry(run) for run in active_runs]
            total_context_tokens = sum(
                run.current_context_tokens for run in active_runs
            )
            return {
                "status": "ok",
                "job_active": self.job_active,
                "job_started_at": self.job_started_at,
                "agent_count": len(agents),
                "total_context_tokens": total_context_tokens,
                "ctx_aware_enabled": self.ctx_aware_enabled,
                "ctx_aware_policy_mode": self.ctx_aware_policy_mode,
                "slo_aware_enabled": self.slo_aware_enabled,
                "slo_target_tokens_per_s": self._round_metric(self.slo_target_tokens_per_s),
                "slo_policy_mode": self.slo_policy_mode,
                "usage_threshold_tokens": self.ctx_aware_usage_threshold_tokens,
                "scheduling_threshold_tokens": self.ctx_aware_scheduling_threshold_tokens,
                "effective_total_context_tokens": (
                    self.ctx_aware_ongoing_effective_context_tokens
                    + self.ctx_aware_pending_effective_context_tokens
                    + self.ctx_aware_ralexation_effective_context_tokens
                ),
                "ongoing_agent_count": self.ctx_aware_ongoing_agent_count,
                "pending_agent_count": self.ctx_aware_pending_agent_count,
                "ralexation_agent_count": self.ctx_aware_ralexation_agent_count,
                "ongoing_effective_context_tokens": self.ctx_aware_ongoing_effective_context_tokens,
                "pending_effective_context_tokens": self.ctx_aware_pending_effective_context_tokens,
                "ralexation_effective_context_tokens": (
                    self.ctx_aware_ralexation_effective_context_tokens
                ),
                "agents": agents,
            }

    def get_ctx_aware_summary(self) -> dict[str, Any]:
        with self._lock:
            return self._ctx_aware_status_locked()

    def get_slo_aware_summary(self) -> dict[str, Any]:
        with self._lock:
            return self._slo_aware_status_locked()

    def start_ctx_aware(
        self,
        *,
        usage_threshold_tokens: int,
        scheduling_threshold_tokens: int,
        policy_mode: str | None = None,
    ) -> dict[str, Any]:
        if usage_threshold_tokens <= 0:
            raise ValueError("usage_threshold_tokens must be > 0")
        if scheduling_threshold_tokens <= 0:
            raise ValueError("scheduling_threshold_tokens must be > 0")
        if scheduling_threshold_tokens >= usage_threshold_tokens:
            raise ValueError(
                "scheduling_threshold_tokens must be smaller than usage_threshold_tokens"
            )
        if scheduling_threshold_tokens < CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS:
            raise ValueError(
                "scheduling_threshold_tokens must be >= "
                f"{CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS}"
            )
        normalized_policy_mode = self._normalize_ctx_aware_policy_mode(policy_mode)

        with self._lock:
            if self.job_active:
                raise RuntimeError("ctx-aware mode cannot be changed while a job is active")
            self.ctx_aware_enabled = True
            self.ctx_aware_usage_threshold_tokens = usage_threshold_tokens
            self.ctx_aware_scheduling_threshold_tokens = scheduling_threshold_tokens
            self.ctx_aware_policy_mode = normalized_policy_mode
            self._rebalance_ctx_aware_locked()
            payload = self._ctx_aware_status_locked()

        self._wake_ctx_aware_scheduler()
        return payload

    def end_ctx_aware(self) -> dict[str, Any]:
        with self._lock:
            if self.job_active:
                raise RuntimeError("ctx-aware mode cannot be changed while a job is active")
            self.slo_aware_enabled = False
            self.slo_target_tokens_per_s = None
            self.slo_policy_mode = None
            self.ctx_aware_enabled = False
            self.ctx_aware_usage_threshold_tokens = None
            self.ctx_aware_scheduling_threshold_tokens = None
            self.ctx_aware_policy_mode = CTX_AWARE_POLICY_MODE_AGE
            self._rebalance_ctx_aware_locked()
            payload = self._ctx_aware_status_locked()

        self._wake_ctx_aware_scheduler()
        return payload

    def start_slo_aware(
        self,
        *,
        target_tokens_per_s: float,
        policy_mode: str,
    ) -> dict[str, Any]:
        if target_tokens_per_s <= 0.0:
            raise ValueError("target_tokens_per_s must be > 0")
        normalized_policy_mode = self._normalize_slo_aware_policy_mode(policy_mode)

        with self._lock:
            if self.job_active:
                raise RuntimeError("slo-aware mode cannot be changed while a job is active")
            if not self.ctx_aware_enabled:
                raise RuntimeError("slo-aware mode requires ctx-aware mode to be enabled")
            self.slo_aware_enabled = True
            self.slo_target_tokens_per_s = target_tokens_per_s
            self.slo_policy_mode = normalized_policy_mode
            self._rebalance_ctx_aware_locked()
            payload = self._slo_aware_status_locked()

        self._wake_ctx_aware_scheduler()
        return payload

    def end_slo_aware(self) -> dict[str, Any]:
        with self._lock:
            if self.job_active:
                raise RuntimeError("slo-aware mode cannot be changed while a job is active")
            self.slo_aware_enabled = False
            self.slo_target_tokens_per_s = None
            self.slo_policy_mode = None
            self._rebalance_ctx_aware_locked()
            payload = self._slo_aware_status_locked()

        self._wake_ctx_aware_scheduler()
        return payload

    def get_output_throughput_summary(self) -> dict[str, Any]:
        with self._lock:
            active_runs = self._sorted_active_runs()
            throughput_values = [
                run.current_output_tokens_per_s
                for run in active_runs
                if run.current_output_tokens_per_s is not None
            ]
            avg_output_tokens_per_s: float | None = None
            min_output_tokens_per_s: float | None = None
            max_output_tokens_per_s: float | None = None
            if throughput_values:
                avg_output_tokens_per_s = sum(throughput_values) / len(throughput_values)
                min_output_tokens_per_s = min(throughput_values)
                max_output_tokens_per_s = max(throughput_values)
            return {
                "status": "ok",
                "job_active": self.job_active,
                "job_started_at": self.job_started_at,
                "agent_count": len(active_runs),
                "throughput_agent_count": len(throughput_values),
                "min_output_tokens_per_s": self._round_metric(min_output_tokens_per_s),
                "max_output_tokens_per_s": self._round_metric(max_output_tokens_per_s),
                "avg_output_tokens_per_s": self._round_metric(avg_output_tokens_per_s),
            }

    def get_output_throughput_details(self) -> dict[str, Any]:
        with self._lock:
            active_runs = self._sorted_active_runs()
            agents = [
                {
                    "api_token_hash": run.api_token_hash,
                    "output_tokens_per_s": self._round_metric(
                        run.current_output_tokens_per_s
                    ),
                }
                for run in active_runs
            ]
            return {
                "status": "ok",
                "job_active": self.job_active,
                "job_started_at": self.job_started_at,
                "agent_count": len(agents),
                "agents": agents,
            }

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
        disconnect_waiter: DisconnectWaiter | None = None,
    ) -> ForwardResult:
        parsed_request_payload = (
            request_payload if request_payload is not None else parse_json_if_possible(body)
        )
        request_id = str(uuid.uuid4())
        if isinstance(parsed_request_payload, dict):
            model_name = str(parsed_request_payload.get("model", ""))
        else:
            model_name = ""
        request_arrived_at = datetime.now(timezone.utc)
        request_arrived_iso = datetime_to_iso8601_utc(request_arrived_at)
        request_arrived_monotonic = time.monotonic()
        api_token_hash: str | None = None
        active_request_registered = False

        try:
            with self._lock:
                if not self.job_active:
                    raise ValueError("job is not active")
                run = self.active_runs.get(api_token)
                if run is None:
                    raise KeyError("agent not started")
                if run.active_agent_action_span is not None:
                    run.active_agent_action_span.end()
                    run.active_agent_action_span = None
                self._increment_active_request_locked(run)
                active_request_registered = True
                api_token_hash = run.api_token_hash

            while True:
                queued_request_registered = False
                with self._lock:
                    if not self.job_active:
                        raise ValueError("job is not active")
                    run = self.active_runs.get(api_token)
                    if run is None:
                        raise KeyError("agent not started")
                    root_context = run.root_context
                    ready_event = run.ready_event
                    cancel_pending = run.cancel_pending_requests
                    should_wait = (
                        self.ctx_aware_enabled
                        and run.schedule_state in {"pending", "ralexation"}
                        and not cancel_pending
                    )
                    if should_wait:
                        self._increment_queued_request_locked(run)
                        queued_request_registered = True
                if cancel_pending:
                    request_ended_at = datetime.now(timezone.utc)
                    request_ended_iso = datetime_to_iso8601_utc(request_ended_at)
                    request_duration_s = max(
                        time.monotonic() - request_arrived_monotonic,
                        0.0,
                    )
                    request_duration_ms = round(request_duration_s * 1000, 3)
                    error_payload = {
                        "error": "agent_ended_while_pending",
                        "detail": "agent ended while the request was still queued",
                    }
                    raw_forward_result = ForwardResult(
                        status_code=409,
                        content=json.dumps(error_payload).encode("utf-8"),
                        content_type="application/json",
                        response_json=error_payload,
                        error_message=error_payload["detail"],
                    )
                    record = {
                        "request_id": request_id,
                        "trace_id": None,
                        "model_inference_span_id": None,
                        "model_inference_parent_span_id": None,
                        "request_start_time": request_arrived_iso,
                        "forward_start_time": None,
                        "request_end_time": request_ended_iso,
                        "request_duration_ms": request_duration_ms,
                        "pending_duration_ms": request_duration_ms,
                        "span_start_time": None,
                        "span_end_time": None,
                        "span_duration_ms": None,
                        "duration_ms": request_duration_ms,
                        "api_token_hash": api_token_hash,
                        "http_method": method.upper(),
                        "http_path": path,
                        "model": model_name,
                        "status_code": raw_forward_result.status_code,
                        "request": parsed_request_payload,
                        "response": error_payload,
                        "response_summary": {
                            "status_code": raw_forward_result.status_code,
                            "error": error_payload["error"],
                            "forward_error": error_payload["detail"],
                        },
                    }
                    with self._lock:
                        active = self.active_runs.get(api_token)
                        if active is not None:
                            active.request_records.append(record)
                            self._decrement_active_request_locked(active)
                            self._start_agent_action_span_locked(active)
                            self._rebalance_ctx_aware_locked()
                    active_request_registered = False
                    self._wake_ctx_aware_scheduler()
                    return raw_forward_result
                if not should_wait:
                    break
                if ready_event is None:
                    raise RuntimeError("blocked agent missing ready event")
                try:
                    became_ready = await self._wait_for_agent_ready(
                        ready_event=ready_event,
                        disconnect_waiter=disconnect_waiter,
                    )
                finally:
                    if queued_request_registered:
                        with self._lock:
                            active = self.active_runs.get(api_token)
                            if active is not None:
                                self._decrement_queued_request_locked(active)
                if became_ready:
                    continue

                request_ended_at = datetime.now(timezone.utc)
                request_ended_iso = datetime_to_iso8601_utc(request_ended_at)
                request_duration_s = max(
                    time.monotonic() - request_arrived_monotonic,
                    0.0,
                )
                request_duration_ms = round(request_duration_s * 1000, 3)
                error_payload = {
                    "error": "client_disconnected",
                    "detail": "downstream client disconnected before the upstream response completed",
                }
                raw_forward_result = ForwardResult(
                    status_code=499,
                    content=json.dumps(error_payload).encode("utf-8"),
                    content_type="application/json",
                    response_json=error_payload,
                    error_message=error_payload["detail"],
                )
                record = {
                    "request_id": request_id,
                    "trace_id": None,
                    "model_inference_span_id": None,
                    "model_inference_parent_span_id": None,
                    "request_start_time": request_arrived_iso,
                    "forward_start_time": None,
                    "request_end_time": request_ended_iso,
                    "request_duration_ms": request_duration_ms,
                    "pending_duration_ms": request_duration_ms,
                    "span_start_time": None,
                    "span_end_time": None,
                    "span_duration_ms": None,
                    "duration_ms": request_duration_ms,
                    "api_token_hash": api_token_hash,
                    "http_method": method.upper(),
                    "http_path": path,
                    "model": model_name,
                    "status_code": raw_forward_result.status_code,
                    "request": parsed_request_payload,
                    "response": error_payload,
                    "response_summary": {
                        "status_code": raw_forward_result.status_code,
                        "error": error_payload["error"],
                        "forward_error": error_payload["detail"],
                    },
                }
                with self._lock:
                    active = self.active_runs.get(api_token)
                    if active is not None:
                        active.request_records.append(record)
                        self._decrement_active_request_locked(active)
                        self._start_agent_action_span_locked(active)
                        self._rebalance_ctx_aware_locked()
                active_request_registered = False
                self._wake_ctx_aware_scheduler()
                return raw_forward_result

            forward_started_at = datetime.now(timezone.utc)
            forward_started_iso = datetime_to_iso8601_utc(forward_started_at)
            forward_started_monotonic = time.monotonic()
            pending_duration_s = max(
                forward_started_monotonic - request_arrived_monotonic,
                0.0,
            )
            pending_duration_ms = round(pending_duration_s * 1000, 3)

            span = self.tracer.start_span(
                "model_inference",
                context=root_context,
                kind=SpanKind.CLIENT,
            )
            span.set_attribute("http.method", method.upper())
            span.set_attribute("http.target", path)

            raw_forward_result: ForwardResult
            client_result: ForwardResult
            forward_headers = dict(headers)
            try:
                with trace.use_span(span, end_on_exit=False):
                    TraceContextTextMapPropagator().inject(forward_headers)
                    try:
                        raw_forward_result = await self._forward_with_disconnect_support(
                            method,
                            path,
                            forward_headers,
                            body,
                            disconnect_waiter=disconnect_waiter,
                        )
                    except ClientDisconnectedError as exc:
                        error_payload = {
                            "error": "client_disconnected",
                            "detail": str(exc),
                        }
                        raw_forward_result = ForwardResult(
                            status_code=499,
                            content=json.dumps(error_payload).encode("utf-8"),
                            content_type="application/json",
                            response_json=error_payload,
                            error_message=str(exc),
                        )
                    except Exception as exc:
                        error_payload = {"error": "vllm_forward_failed", "detail": str(exc)}
                        raw_forward_result = ForwardResult(
                            status_code=502,
                            content=json.dumps(error_payload).encode("utf-8"),
                            content_type="application/json",
                            response_json=error_payload,
                            error_message=str(exc),
                        )
                    client_result = raw_forward_result
                    if response_transformer is not None:
                        try:
                            client_result = response_transformer(
                                path,
                                parsed_request_payload,
                                raw_forward_result,
                            )
                        except Exception:
                            LOGGER.exception("response transform failed for path=%s", path)
            finally:
                span.end()

            request_ended_at = datetime.now(timezone.utc)
            request_ended_iso = datetime_to_iso8601_utc(request_ended_at)
            request_duration_s = max(
                time.monotonic() - request_arrived_monotonic,
                0.0,
            )
            request_duration_ms = round(request_duration_s * 1000, 3)
            span_duration_s = max(time.monotonic() - forward_started_monotonic, 0.0)
            span_duration_ms = round(span_duration_s * 1000, 3)

            span_context = span.get_span_context()
            trace_id = f"{span_context.trace_id:032x}"
            span_id = f"{span_context.span_id:016x}"
            parent_span_id: str | None = None
            if getattr(span, "parent", None):
                parent_span_id = f"{span.parent.span_id:016x}"

            response_summary: dict[str, Any] = {"status_code": raw_forward_result.status_code}
            if isinstance(raw_forward_result.response_json, dict):
                usage = raw_forward_result.response_json.get("usage")
                if usage is not None:
                    response_summary["usage"] = usage
                error = raw_forward_result.response_json.get("error")
                if error is not None:
                    response_summary["error"] = error
            if raw_forward_result.error_message:
                response_summary["forward_error"] = raw_forward_result.error_message

            response_payload = (
                raw_forward_result.response_json
                if raw_forward_result.response_json is not None
                else parse_json_if_possible(raw_forward_result.content)
            )
            record = {
                "request_id": request_id,
                "trace_id": trace_id,
                "model_inference_span_id": span_id,
                "model_inference_parent_span_id": parent_span_id,
                "request_start_time": request_arrived_iso,
                "forward_start_time": forward_started_iso,
                "request_end_time": request_ended_iso,
                "request_duration_ms": request_duration_ms,
                "pending_duration_ms": pending_duration_ms,
                "span_start_time": forward_started_iso,
                "span_end_time": request_ended_iso,
                "span_duration_ms": span_duration_ms,
                "duration_ms": request_duration_ms,
                "api_token_hash": api_token_hash,
                "http_method": method.upper(),
                "http_path": path,
                "model": model_name,
                "status_code": raw_forward_result.status_code,
                "request": parsed_request_payload,
                "response": response_payload,
                "response_summary": response_summary,
            }

            updated_context_tokens = self._context_tokens_from_response(response_payload)
            completion_tokens = self._completion_tokens_from_response(response_payload)

            with self._lock:
                active = self.active_runs.get(api_token)
                if active is not None:
                    active.request_records.append(record)
                    if updated_context_tokens is not None:
                        active.current_context_tokens = updated_context_tokens
                        active.has_usable_context_usage = True
                    if completion_tokens is not None:
                        active.total_completion_tokens += completion_tokens
                        active.total_llm_request_duration_s += request_duration_s
                        active.total_ctx_aware_request_duration_s += request_duration_s
                        active.current_output_tokens_per_s = self._output_throughput_from_totals(
                            active.total_completion_tokens,
                            active.total_llm_request_duration_s,
                        )
                    self._decrement_active_request_locked(active)
                    self._start_agent_action_span_locked(active)
                    self._rebalance_ctx_aware_locked()
                    if self._should_enter_ralexation_locked(active):
                        slack_s = self._slo_slack_s_locked(active)
                        if slack_s is not None and slack_s > 0.0:
                            ralexation_duration_s = slack_s / 2.0
                            self._mark_run_ralexation_locked(
                                active,
                                duration_s=ralexation_duration_s,
                            )
                            self.ctx_aware_job_log_counters.agents_turned_ralexation += 1
                            self._write_slo_aware_decision_log_locked(
                                event_type="agent_entered_ralexation",
                                run=active,
                                details={
                                    "from_schedule_state": "ongoing",
                                    "to_schedule_state": "ralexation",
                                    "policy_mode": self.slo_policy_mode,
                                    "ralexation_duration_s": self._round_metric(
                                        ralexation_duration_s
                                    ),
                                    "ralexation_until": active.ralexation_until_iso,
                                },
                            )
                            self._refresh_ctx_aware_totals_locked()
            active_request_registered = False
            self._wake_ctx_aware_scheduler()
            return client_result
        finally:
            if active_request_registered:
                with self._lock:
                    active = self.active_runs.get(api_token)
                    if active is not None:
                        self._decrement_active_request_locked(active)
                        self._rebalance_ctx_aware_locked()
                self._wake_ctx_aware_scheduler()

    def end_job(self, status: str) -> dict[str, Any]:
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
                        "metadata": {"status": status},
                    }
                )

            self._write_ctx_aware_job_log_sample_locked(sample_time=job_ended_at)
            self._force_flush_traces()
            if self.config.job_end_trace_wait_seconds > 0:
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
        self, output_location: str, run: RunState, jaeger_payload: dict[str, Any]
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

        with tempfile.TemporaryDirectory(prefix="gateway_artifact_") as temp_dir:
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

            write_json_file(jaeger_path, jaeger_payload)
            write_jsonl_file(lifecycle_path, run.lifecycle_events)
            write_jsonl_file(requests_path, run.request_records)

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
                "run_start_time": run.run_start_time,
                "run_end_time": run.run_end_time,
                "return_code": run.return_code,
                "request_count": len(run.request_records),
                "model_inference_span_count": len(run.request_records),
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
            "path": str(artifact_path),
            "artifact_format": artifact_format,
            "request_count": len(run.request_records),
        }


class JobStartRequest(BaseModel):
    output_location: str = Field(min_length=1)


class JobEndRequest(BaseModel):
    status: str = "completed"


class AgentStartRequest(BaseModel):
    api_token: str = Field(min_length=1)


class AgentEndRequest(BaseModel):
    api_token: str = Field(min_length=1)
    return_code: int


class CtxAwareStartRequest(BaseModel):
    usage_threshold_tokens: int = Field(gt=0)
    scheduling_threshold_tokens: int = Field(gt=0)
    policy_mode: str | None = None


class SloAwareStartRequest(BaseModel):
    target_tokens_per_s: float = Field(gt=0)
    policy_mode: str


def create_gateway_service(
    config: GatewayConfig | None = None,
    forwarder: Forwarder | None = None,
    jaeger_fetcher: JaegerFetcher | None = None,
    service_name: str | None = None,
) -> GatewayService:
    gateway_config = config or GatewayConfig.from_port_profile()
    tracer = configure_tracer(
        service_name or gateway_config.service_name,
        endpoint=gateway_config.otlp_traces_endpoint,
        insecure=gateway_config.otlp_traces_insecure,
    )
    return GatewayService(
        config=gateway_config,
        tracer=tracer,
        forwarder=forwarder,
        jaeger_fetcher=jaeger_fetcher,
    )


def _build_reasoning_transformer(
    *,
    model_registry: ModelRegistry | None,
) -> ResponseTransformer | None:
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


def create_app(
    service: GatewayService | None = None,
    *,
    gateway_parse_port: int | None = None,
    model_registry: ModelRegistry | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        gateway_service = app_instance.state.gateway_service
        if gateway_service is None:
            gateway_service = create_gateway_service()
            app_instance.state.gateway_service = gateway_service
        await gateway_service.start_background_tasks()
        try:
            yield
        finally:
            await gateway_service.stop_background_tasks()

    app = FastAPI(title="vLLM Gateway Ctx", version="0.1.0", lifespan=lifespan)
    app.state.gateway_service = service
    app.state.gateway_parse_port = gateway_parse_port
    app.state.response_transformer = (
        _build_reasoning_transformer(model_registry=model_registry)
        if gateway_parse_port is not None
        else None
    )

    def get_gateway_service() -> GatewayService:
        existing_service = app.state.gateway_service
        if existing_service is None:
            existing_service = create_gateway_service()
            app.state.gateway_service = existing_service
        return existing_service

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ctx-aware")
    async def ctx_aware_status() -> dict[str, Any]:
        gateway_service = get_gateway_service()
        return gateway_service.get_ctx_aware_summary()

    @app.post("/ctx-aware/start")
    async def ctx_aware_start(payload: CtxAwareStartRequest) -> dict[str, Any]:
        gateway_service = get_gateway_service()
        try:
            return gateway_service.start_ctx_aware(
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
        gateway_service = get_gateway_service()
        try:
            return gateway_service.end_ctx_aware()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/slo-aware")
    async def slo_aware_status() -> dict[str, Any]:
        gateway_service = get_gateway_service()
        return gateway_service.get_slo_aware_summary()

    @app.post("/slo-aware/start")
    async def slo_aware_start(payload: SloAwareStartRequest) -> dict[str, Any]:
        gateway_service = get_gateway_service()
        try:
            return gateway_service.start_slo_aware(
                target_tokens_per_s=payload.target_tokens_per_s,
                policy_mode=payload.policy_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/slo-aware/end")
    async def slo_aware_end() -> dict[str, Any]:
        gateway_service = get_gateway_service()
        try:
            return gateway_service.end_slo_aware()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/ipc/context")
    async def ipc_context() -> dict[str, Any]:
        gateway_service = get_gateway_service()
        return gateway_service.get_context_usage_summary()

    @app.get("/ipc/output-throughput")
    async def ipc_output_throughput() -> dict[str, Any]:
        gateway_service = get_gateway_service()
        return gateway_service.get_output_throughput_summary()

    @app.get("/ipc/output-throughput/agents")
    async def ipc_output_throughput_agents() -> dict[str, Any]:
        gateway_service = get_gateway_service()
        return gateway_service.get_output_throughput_details()

    @app.post("/job/start")
    async def job_start(payload: JobStartRequest) -> dict[str, Any]:
        gateway_service = get_gateway_service()
        try:
            return gateway_service.start_job(payload.output_location)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/job/end")
    async def job_end(payload: JobEndRequest) -> dict[str, Any]:
        gateway_service = get_gateway_service()
        try:
            return gateway_service.end_job(payload.status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/agent/start")
    async def agent_start(payload: AgentStartRequest) -> dict[str, Any]:
        gateway_service = get_gateway_service()
        try:
            return gateway_service.start_agent(payload.api_token)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/agent/end")
    async def agent_end(payload: AgentEndRequest) -> dict[str, Any]:
        gateway_service = get_gateway_service()
        try:
            return await gateway_service.end_agent(
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
        gateway_service = get_gateway_service()
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
            result = await gateway_service.proxy_request(
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


app = create_app()
