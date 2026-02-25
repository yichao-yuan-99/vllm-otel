from __future__ import annotations

import hashlib
import json
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


def configure_tracer(service_name: str) -> trace.Tracer:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "")
    insecure = os.getenv("OTEL_EXPORTER_OTLP_TRACES_INSECURE", "true").lower() == "true"
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
    request_timeout_seconds: float = 120.0
    artifact_compression: str = "none"
    job_end_trace_wait_seconds: float = 10.0

    def __post_init__(self) -> None:
        self.artifact_compression = normalize_artifact_compression(
            self.artifact_compression
        )
        if self.job_end_trace_wait_seconds < 0:
            raise ValueError("job_end_trace_wait_seconds must be >= 0")

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        return cls(
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:11451"),
            jaeger_api_base_url=os.getenv(
                "JAEGER_API_BASE_URL", "http://localhost:16686/api/traces"
            ),
            request_timeout_seconds=float(
                os.getenv("GATEWAY_REQUEST_TIMEOUT_SECONDS", "120")
            ),
            artifact_compression=os.getenv("GATEWAY_ARTIFACT_COMPRESSION", "none"),
            job_end_trace_wait_seconds=float(
                os.getenv("GATEWAY_JOB_END_TRACE_WAIT_SECONDS", "10")
            ),
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
        async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
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

    async def proxy_request(
        self,
        api_token: str,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
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

        request_payload = parse_json_if_possible(body)
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

        forward_result: ForwardResult
        forward_headers = dict(headers)
        with trace.use_span(span, end_on_exit=False):
            TraceContextTextMapPropagator().inject(forward_headers)
            try:
                forward_result = await self.forwarder(
                    method, path, forward_headers, body
                )
            except Exception as exc:
                error_payload = {"error": "vllm_forward_failed", "detail": str(exc)}
                forward_result = ForwardResult(
                    status_code=502,
                    content=json.dumps(error_payload).encode("utf-8"),
                    content_type="application/json",
                    response_json=error_payload,
                    error_message=str(exc),
                )

        request_ended = datetime.now(timezone.utc)
        request_ended_iso = request_ended.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
        duration_ms = round((request_ended - request_started).total_seconds() * 1000, 3)

        span_context = span.get_span_context()
        trace_id = f"{span_context.trace_id:032x}"
        span_id = f"{span_context.span_id:016x}"
        parent_span_id: str | None = None
        if getattr(span, "parent", None):
            parent_span_id = f"{span.parent.span_id:016x}"

        response_summary: dict[str, Any] = {"status_code": forward_result.status_code}
        if isinstance(forward_result.response_json, dict):
            usage = forward_result.response_json.get("usage")
            if usage is not None:
                response_summary["usage"] = usage
            error = forward_result.response_json.get("error")
            if error is not None:
                response_summary["error"] = error
        if forward_result.error_message:
            response_summary["forward_error"] = forward_result.error_message

        if isinstance(request_payload, dict):
            model_name = str(request_payload.get("model", ""))
        else:
            model_name = ""
        response_payload = (
            forward_result.response_json
            if forward_result.response_json is not None
            else parse_json_if_possible(forward_result.content)
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
            "status_code": forward_result.status_code,
            "request": request_payload,
            "response": response_payload,
            "response_summary": response_summary,
        }

        span.end()

        with self._lock:
            active = self.active_runs.get(api_token)
            if active is not None:
                active.request_records.append(record)
                active.active_agent_action_span = self.tracer.start_span(
                    "agent_action",
                    context=active.root_context,
                    kind=SpanKind.INTERNAL,
                )

        return forward_result

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
    gateway_config = config or GatewayConfig.from_env()
    tracer = configure_tracer(service_name or os.getenv("OTEL_SERVICE_NAME", "vllm-gateway"))
    return GatewayService(
        config=gateway_config,
        tracer=tracer,
        forwarder=forwarder,
        jaeger_fetcher=jaeger_fetcher,
    )


def create_app(service: GatewayService | None = None) -> FastAPI:
    app = FastAPI(title="vLLM Gateway", version="0.1.0")
    gateway_service = service or create_gateway_service()
    app.state.gateway_service = gateway_service

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/job/start")
    async def job_start(payload: JobStartRequest) -> dict[str, Any]:
        try:
            return gateway_service.start_job(payload.output_location)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/job/end")
    async def job_end(payload: JobEndRequest) -> dict[str, Any]:
        try:
            return gateway_service.end_job(payload.status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/agent/start")
    async def agent_start(payload: AgentStartRequest) -> dict[str, Any]:
        try:
            return gateway_service.start_agent(payload.api_token)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/agent/end")
    async def agent_end(payload: AgentEndRequest) -> dict[str, Any]:
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
        token = extract_api_token(dict(request.headers))
        if not token:
            raise HTTPException(
                status_code=401,
                detail="missing api token (Authorization: Bearer ... or x-api-key)",
            )

        body = await request.body()
        query = request.url.query
        target_path = f"v1/{path}"
        if query:
            target_path = f"{target_path}?{query}"

        try:
            result = await gateway_service.proxy_request(
                api_token=token,
                method=request.method,
                path=target_path,
                headers=dict(request.headers),
                body=body,
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
