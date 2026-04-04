from __future__ import annotations

import asyncio
from contextlib import suppress
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
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Awaitable, Callable
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

from gateway.model_configs import ModelRegistry, load_model_registry
from gateway.port_profiles import load_port_profile
from gateway.reasoning_response_parser import ReasoningResponseParser
from gateway.runtime_config import GatewayRuntimeSettings, load_runtime_settings


LOGGER = logging.getLogger(__name__)


def now_iso8601_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
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
    return trace.get_tracer("vllm-gateway")


@dataclass
class GatewayConfig:
    vllm_base_url: str
    jaeger_api_base_url: str
    otlp_traces_endpoint: str
    service_name: str = "vllm-gateway"
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
    current_output_tokens_per_s: float | None = None


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
    def _round_metric(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 6)

    def _sorted_active_runs(self) -> list[RunState]:
        return sorted(
            self.active_runs.values(),
            key=lambda run: (run.run_start_time, run.api_token_hash, run.trace_id),
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

        return {
            "status": "ok",
            "job_started_at": self.job_started_at,
            "output_location": str(output_path),
        }

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

            run_state = RunState(
                api_token=api_token,
                api_token_hash=api_token_hash,
                trace_id=trace_id,
                run_start_time=run_started_at,
                root_span=root_span,
                root_context=root_context,
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
            self.active_runs[api_token] = run_state

        return {"status": "ok", "trace_id": trace_id}

    def end_agent(self, api_token: str, return_code: int) -> dict[str, Any]:
        with self._lock:
            run = self.active_runs.get(api_token)
            if run is None:
                raise KeyError("agent not started")

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

        return {"status": "ok"}

    def get_context_usage_summary(self) -> dict[str, Any]:
        with self._lock:
            active_runs = self._sorted_active_runs()
            agents = [
                {
                    "api_token_hash": run.api_token_hash,
                    "trace_id": run.trace_id,
                    "run_start_time": run.run_start_time,
                    "context_tokens": run.current_context_tokens,
                }
                for run in active_runs
            ]
            total_context_tokens = sum(
                run.current_context_tokens for run in active_runs
            )
            return {
                "status": "ok",
                "job_active": self.job_active,
                "job_started_at": self.job_started_at,
                "agent_count": len(agents),
                "total_context_tokens": total_context_tokens,
                "agents": agents,
            }

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
        with self._lock:
            if not self.job_active:
                raise ValueError("job is not active")
            run = self.active_runs.get(api_token)
            if run is None:
                raise KeyError("agent not started")
            if run.active_agent_action_span is not None:
                run.active_agent_action_span.end()
                run.active_agent_action_span = None
            root_context = run.root_context

        parsed_request_payload = (
            request_payload if request_payload is not None else parse_json_if_possible(body)
        )
        request_id = str(uuid.uuid4())
        request_started = datetime.now(timezone.utc)
        request_started_iso = request_started.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )

        span = self.tracer.start_span(
            "model_inference",
            context=root_context,
            kind=SpanKind.CLIENT,
        )
        span.set_attribute("http.method", method.upper())
        span.set_attribute("http.target", path)

        raw_forward_result: ForwardResult
        forward_headers = dict(headers)
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

        request_ended = datetime.now(timezone.utc)
        request_ended_iso = request_ended.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
        duration_s = max((request_ended - request_started).total_seconds(), 0.0)
        duration_ms = round(duration_s * 1000, 3)

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

        if isinstance(parsed_request_payload, dict):
            model_name = str(parsed_request_payload.get("model", ""))
        else:
            model_name = ""
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
            "request_start_time": request_started_iso,
            "request_end_time": request_ended_iso,
            "request_duration_ms": duration_ms,
            "span_start_time": request_started_iso,
            "span_end_time": request_ended_iso,
            "duration_ms": duration_ms,
            "api_token_hash": run.api_token_hash,
            "http_method": method.upper(),
            "http_path": path,
            "model": model_name,
            "status_code": raw_forward_result.status_code,
            "request": parsed_request_payload,
            "response": response_payload,
            "response_summary": response_summary,
        }

        span.end()
        updated_context_tokens = self._context_tokens_from_response(response_payload)
        completion_tokens = self._completion_tokens_from_response(response_payload)

        with self._lock:
            active = self.active_runs.get(api_token)
            if active is not None:
                active.request_records.append(record)
                if updated_context_tokens is not None:
                    active.current_context_tokens = updated_context_tokens
                if completion_tokens is not None:
                    active.total_completion_tokens += completion_tokens
                    active.total_llm_request_duration_s += duration_s
                    active.current_output_tokens_per_s = self._output_throughput_from_totals(
                        active.total_completion_tokens,
                        active.total_llm_request_duration_s,
                    )
                active.active_agent_action_span = self.tracer.start_span(
                    "agent_action",
                    context=active.root_context,
                    kind=SpanKind.INTERNAL,
                )

        return client_result

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

            return {
                "status": "ok",
                "job_end_status": status,
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
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
    app = FastAPI(title="vLLM Gateway", version="0.1.0")
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
            return gateway_service.end_agent(
                api_token=payload.api_token,
                return_code=payload.return_code,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
