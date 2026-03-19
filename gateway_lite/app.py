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
from urllib.parse import unquote, urlparse, urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from gateway_lite.port_profiles import load_port_profile
from gateway_lite.runtime_config import GatewayRuntimeSettings, load_runtime_settings


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
    return dt.strftime("%Y%m%dT%H%MZ")


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


def build_upstream_url(base_url: str, path: str) -> str:
    """
    Join base URL and request path while avoiding duplicate '/v1' segments.

    Examples:
    - base='https://api.openai.com', path='v1/chat/completions'
      -> 'https://api.openai.com/v1/chat/completions'
    - base='https://api.openai.com/v1', path='v1/chat/completions'
      -> 'https://api.openai.com/v1/chat/completions'
    """
    parsed = urlsplit(base_url)

    request_path, separator, request_query = path.partition("?")
    relative_path = "/" + request_path.lstrip("/")
    base_path = parsed.path.rstrip("/")

    if base_path.endswith("/v1") and relative_path.startswith("/v1/"):
        joined_path = f"{base_path[:-3]}{relative_path}"
    else:
        joined_path = f"{base_path}{relative_path}" if base_path else relative_path

    if not joined_path:
        joined_path = "/"

    base_query = parsed.query
    if base_query and request_query:
        joined_query = f"{base_query}&{request_query}"
    elif request_query:
        joined_query = request_query
    else:
        joined_query = base_query

    if separator and not request_query:
        joined_query = ""

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            joined_path,
            joined_query,
            parsed.fragment,
        )
    )


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
    """Extract plain API token from request headers."""
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


def _replace_api_key_in_headers(headers: dict[str, str], new_api_key: str) -> dict[str, str]:
    """
    Replace the API key in headers with the new one.
    Handles Authorization (Bearer) and x-api-key/api-key headers.
    """
    new_headers = dict(headers)
    lowered_headers = {key.lower(): key for key in headers.keys()}
    
    # Check for x-api-key variations
    for key in ["x-api-key", "api-key", "openai-api-key"]:
        if key in lowered_headers:
            original_key = lowered_headers[key]
            new_headers[original_key] = new_api_key
            return new_headers
    
    # Check for Authorization header
    if "authorization" in lowered_headers:
        original_key = lowered_headers["authorization"]
        new_headers[original_key] = f"Bearer {new_api_key}"
        return new_headers
    
    return new_headers


def parse_json_if_possible(raw_body: bytes) -> Any:
    if not raw_body:
        return {}
    try:
        return json.loads(raw_body.decode("utf-8"))
    except Exception:
        return {"raw_body": raw_body.decode("utf-8", errors="replace")}


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


@dataclass
class GatewayConfig:
    vllm_base_url: str
    service_name: str = "vllm-gateway-lite"
    artifact_compression: str = "none"
    output_root: str | None = None
    upstream_api_key: str | None = None  # Optional fixed key injected for upstream forwarding.

    def __post_init__(self) -> None:
        self.artifact_compression = normalize_artifact_compression(
            self.artifact_compression
        )

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
        return cls(
            vllm_base_url=f"http://localhost:{profile.vllm_port}",
            service_name=settings.service_name,
            artifact_compression=settings.artifact_compression,
            output_root=settings.output_root,
            upstream_api_key=settings.upstream_api_key,
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
ResponseTransformer = Callable[[str, Any, ForwardResult], ForwardResult]
DisconnectWaiter = Callable[[], Awaitable[None]]


class ClientDisconnectedError(RuntimeError):
    pass


@dataclass
class RunState:
    api_token: str
    api_token_hash: str
    run_id: str
    run_start_time: str
    lifecycle_events: list[dict[str, Any]] = field(default_factory=list)
    request_records: list[dict[str, Any]] = field(default_factory=list)
    run_end_time: str | None = None
    return_code: int | None = None


class GatewayService:
    def __init__(
        self,
        config: GatewayConfig,
        forwarder: Forwarder | None = None,
    ) -> None:
        self.config = config
        self.forwarder = forwarder or self._forward_to_vllm

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
        target_url = build_upstream_url(self.config.vllm_base_url, path)
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
            run_id = str(uuid.uuid4())

            run_state = RunState(
                api_token=api_token,
                api_token_hash=api_token_hash,
                run_id=run_id,
                run_start_time=run_started_at,
            )
            run_state.lifecycle_events.append(
                {
                    "event_type": "job_start",
                    "timestamp": self.job_started_at,
                    "run_id": run_id,
                    "api_token_hash": api_token_hash,
                    "metadata": {"output_location": self.job_output_location},
                }
            )
            run_state.lifecycle_events.append(
                {
                    "event_type": "agent_start",
                    "timestamp": run_started_at,
                    "run_id": run_id,
                    "api_token_hash": api_token_hash,
                    "metadata": {},
                }
            )
            self.active_runs[api_token] = run_state

        return {"status": "ok", "run_id": run_id}

    def end_agent(self, api_token: str, return_code: int) -> dict[str, Any]:
        with self._lock:
            run = self.active_runs.get(api_token)
            if run is None:
                raise KeyError("agent not started")

            run.return_code = return_code
            run.run_end_time = now_iso8601_utc()
            run.lifecycle_events.append(
                {
                    "event_type": "agent_end",
                    "timestamp": run.run_end_time,
                    "run_id": run.run_id,
                    "api_token_hash": run.api_token_hash,
                    "metadata": {"return_code": return_code},
                }
            )

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

        parsed_request_payload = (
            request_payload if request_payload is not None else parse_json_if_possible(body)
        )
        request_id = str(uuid.uuid4())
        request_started = datetime.now(timezone.utc)
        request_started_iso = request_started.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )

        raw_forward_result: ForwardResult
        try:
            raw_forward_result = await self._forward_with_disconnect_support(
                method,
                path,
                dict(headers),
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
            error_payload = {"error": "forward_failed", "detail": str(exc)}
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
        duration_ms = round((request_ended - request_started).total_seconds() * 1000, 3)

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
            "run_id": run.run_id,
            "request_start_time": request_started_iso,
            "request_end_time": request_ended_iso,
            "request_duration_ms": duration_ms,
            "api_token_hash": run.api_token_hash,
            "http_method": method.upper(),
            "http_path": path,
            "model": model_name,
            "status_code": raw_forward_result.status_code,
            "request": parsed_request_payload,
            "response": response_payload,
            "response_summary": response_summary,
        }

        with self._lock:
            active = self.active_runs.get(api_token)
            if active is not None:
                active.request_records.append(record)

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
                        "run_id": run.run_id,
                        "api_token_hash": run.api_token_hash,
                        "metadata": {"status": status},
                    }
                )

            artifacts: list[dict[str, Any]] = []
            for run in ended_runs:
                artifact_summary = self._write_artifact(output_location, run)
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
        self, output_location: str, run: RunState
    ) -> dict[str, Any]:
        if not run.run_end_time:
            raise ValueError("run_end_time missing")
        if run.return_code is None:
            raise ValueError("return_code missing")

        output_dir = parse_output_location(output_location)
        output_dir.mkdir(parents=True, exist_ok=True)

        compact_start = iso8601_to_compact(run.run_start_time)
        artifact_base_name = (
            f"run_{compact_start}_{run.api_token_hash[:12]}_{run.run_id}"
        )

        with tempfile.TemporaryDirectory(prefix="gateway_lite_artifact_") as temp_dir:
            root = Path(temp_dir)
            events_dir = root / "events"
            requests_dir = root / "requests"
            events_dir.mkdir(parents=True, exist_ok=True)
            requests_dir.mkdir(parents=True, exist_ok=True)

            lifecycle_path = events_dir / "lifecycle.jsonl"
            requests_path = requests_dir / "model_inference.jsonl"

            write_jsonl_file(lifecycle_path, run.lifecycle_events)
            write_jsonl_file(requests_path, run.request_records)

            artifact_files: list[dict[str, Any]] = []
            for relative_path, file_path in [
                ("events/lifecycle.jsonl", lifecycle_path),
                ("requests/model_inference.jsonl", requests_path),
            ]:
                digest, size_bytes = file_sha256_and_size(file_path)
                artifact_files.append(
                    {"path": relative_path, "sha256": digest, "size_bytes": size_bytes}
                )

            manifest = {
                "schema_version": "v1-lite",
                "generated_at": now_iso8601_utc(),
                "run_id": run.run_id,
                "api_token_hash": run.api_token_hash,
                "run_start_time": run.run_start_time,
                "run_end_time": run.run_end_time,
                "return_code": run.return_code,
                "request_count": len(run.request_records),
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
                    archive.add(lifecycle_path, arcname="events/lifecycle.jsonl")
                    archive.add(requests_path, arcname="requests/model_inference.jsonl")
            else:
                artifact_path = self._make_unique_path(output_dir / artifact_base_name)
                artifact_path.mkdir(parents=True, exist_ok=True)
                (artifact_path / "events").mkdir(parents=True, exist_ok=True)
                (artifact_path / "requests").mkdir(parents=True, exist_ok=True)

                shutil.copy2(manifest_path, artifact_path / "manifest.json")
                shutil.copy2(lifecycle_path, artifact_path / "events" / "lifecycle.jsonl")
                shutil.copy2(
                    requests_path,
                    artifact_path / "requests" / "model_inference.jsonl",
                )

        return {
            "run_id": run.run_id,
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
) -> GatewayService:
    gateway_config = config or GatewayConfig.from_port_profile()
    return GatewayService(
        config=gateway_config,
        forwarder=forwarder,
    )


def create_app(
    service: GatewayService | None = None,
    *,
    gateway_parse_port: int | None = None,
) -> FastAPI:
    app = FastAPI(title="vLLM Gateway Lite", version="0.1.0")
    app.state.gateway_service = service
    # Both ports behave identically - no response transformation
    app.state.gateway_parse_port = gateway_parse_port

    def get_gateway_service() -> GatewayService:
        existing_service = app.state.gateway_service
        if existing_service is None:
            existing_service = create_gateway_service()
            app.state.gateway_service = existing_service
        return existing_service

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

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
        config = gateway_service.config

        api_token = extract_api_token(dict(request.headers))
        if not api_token:
            raise HTTPException(
                status_code=401,
                detail="missing api token (Authorization: Bearer ... or x-api-key)",
            )
        forward_headers = dict(request.headers)

        # Optional fixed key override for upstream requests.
        if config.upstream_api_key:
            forward_headers = _replace_api_key_in_headers(
                forward_headers,
                config.upstream_api_key,
            )

        body = await request.body()
        request_payload = parse_json_if_possible(body)
        query = request.url.query
        target_path = f"v1/{path}"
        if query:
            target_path = f"{target_path}?{query}"

        async def wait_for_disconnect() -> None:
            while True:
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.1)

        try:
            result = await gateway_service.proxy_request(
                api_token=api_token,
                method=request.method,
                path=target_path,
                headers=forward_headers,
                body=body,
                request_payload=request_payload,
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
