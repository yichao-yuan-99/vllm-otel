# vLLM Gateway (Concise Design)

## Goal

Build a gateway in front of vLLM to analyze multi-agent behavior with trace-level attribution.

## Assumptions

- Each agent has a unique API token.
- Agents call the gateway, not vLLM directly.
- vLLM and the gateway both emit OpenTelemetry spans to Jaeger.
- Job lifecycle calls (`/job/start`, `/job/end`) are made by an external orchestrator process.

## Components

- **Agent wrapper**
  - Starts/stops an agent subprocess.
  - Calls gateway lifecycle endpoints (`/agent/start`, `/agent/end`).
- **Gateway**
  - OpenAI-compatible proxy endpoint for model requests.
  - Maps API token -> agent identity/session.
  - Maintains active job context and output location.
  - Creates or reuses a trace for that agent session.
  - Forwards request to vLLM with trace context propagation.
  - Records request metadata for later analysis.
- **vLLM**
  - Serves model requests and emits OTEL spans.
- **Jaeger**
  - Stores distributed traces.

## Flow

1. External orchestrator process calls `POST /job/start` with a required output location for job artifacts.
2. Wrapper calls `POST /agent/start` with API token.
3. Gateway opens a new root trace (or run context) for that agent run.
4. Agent sends normal OpenAI-style requests to the gateway.
5. For each gateway -> vLLM call, gateway starts a `model_inference` span, propagates context, and forwards the request.
6. When the vLLM response returns, gateway ends that `model_inference` span, returns the response to the agent, and starts an `agent_action` span.
7. The `agent_action` span ends when the next request is made; then a new `model_inference` span starts for that next round.
8. Wrapper detects agent process exit and calls `POST /agent/end`.
9. When the job is complete, external orchestrator process calls `POST /job/end`.
10. Gateway fetches trace data from Jaeger, attaches captured request metadata, and writes artifacts to the job output location.

## Minimal API Sketch

- `POST /job/start`
  - Input: `output_location` (required path/URI for job output).
  - Output: status.
- `POST /job/end`
  - Input: status.
  - Output: artifact summary + status.
- `POST /agent/start`
  - Input: `api_token`.
  - Output: `trace_id`, status.
- `POST /agent/end`
  - Input: `api_token`, `return_code` (agent process exit code).
  - Output: status.
- `POST /v1/*`
  - Standard OpenAI-compatible proxy path to vLLM.

## Timestamp Format

All timestamps must be normalized to ISO 8601 (UTC), for example: `2026-02-24T09:54:12.345Z`.
This includes job/agent lifecycle events, request/response timestamps, span start/end times, and artifact metadata.

## Output Artifact Format

Write one artifact per completed agent run under `output_location` from `POST /job/start`.
Default output is an uncompressed directory. Optional compression mode (`tar.gz`) is supported.

### Artifact Type and Name

- Default format: directory named `run_<run_start_iso_compact>_<api_token_hash12>_<trace_id>`
- Optional compressed format: `.tar.gz` with same base name
- Example (default): `run_20260224T095412Z_a13f0c91c2ab_4fd1...9e2a/`
- Example (compressed): `run_20260224T095412Z_a13f0c91c2ab_4fd1...9e2a.tar.gz`
- `api_token_hash12` is the first 12 hex chars of `sha256(api_token)` (never store raw token in file name).

### Archive Layout

```text
run_<...>/
|- manifest.json
|- trace/
|  `- jaeger_trace.json
|- events/
|  `- lifecycle.jsonl
`- requests/
   `- model_inference.jsonl
```

### File Schemas

`manifest.json` (single JSON object):

- `schema_version`: string (e.g. `v1`)
- `generated_at`: ISO 8601 UTC
- `trace_id`: hex string
- `api_token_hash`: hex string (`sha256(api_token)`)
- `run_start_time`: ISO 8601 UTC
- `run_end_time`: ISO 8601 UTC
- `return_code`: integer
- `request_count`: integer
- `model_inference_span_count`: integer
- `artifact_files`: array of `{path, sha256, size_bytes}`

`trace/jaeger_trace.json`:

- Raw trace export payload fetched from Jaeger for this run/trace.

`events/lifecycle.jsonl` (JSON Lines, one object per line):

- `event_type`: `job_start` | `agent_start` | `agent_end` | `job_end`
- `timestamp`: ISO 8601 UTC
- `trace_id`: hex string (if available for event)
- `api_token_hash`: hex string
- `metadata`: object (small event-specific fields)

`requests/model_inference.jsonl` (JSON Lines, one object per gateway->vLLM call):

- `request_id`: stable UUID/string
- `trace_id`: hex string
- `model_inference_span_id`: hex string
- `model_inference_parent_span_id`: hex string or null
- `request_start_time`: ISO 8601 UTC
- `request_end_time`: ISO 8601 UTC
- `request_duration_ms`: number
- `span_start_time`: ISO 8601 UTC
- `span_end_time`: ISO 8601 UTC
- `duration_ms`: number
- `api_token_hash`: hex string
- `http_method`: string
- `http_path`: string (e.g. `/v1/chat/completions`)
- `model`: string
- `status_code`: integer
- `request`: object containing the original forwarded request body/envelope
- `response`: object containing the full returned LLM response payload for this call
- `response_summary`: object with non-sensitive response metadata (status, token usage if available, error message if any)

### Span-to-Request Binding Rule

Each record in `requests/model_inference.jsonl` must include `model_inference_span_id`, the exact forwarded `request`, and the full `response` payload.  
This is the canonical binding between a `model_inference` span and its request/response pair.
