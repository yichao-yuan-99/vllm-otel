# Logging with OpenAI Endpoint in Gateway Lite

This document describes what gateway-lite records when forwarding to OpenAI (or
any OpenAI-compatible endpoint).

## Overview

Gateway-lite is backend-agnostic. It forwards `/v1/*` requests and records:

- lifecycle events (`events/lifecycle.jsonl`)
- request/response records (`requests/model_inference.jsonl`)
- run manifest (`manifest.json`)

## Recommended Key Flow

- Agents send per-agent API tokens (for identity tracking).
- Gateway-lite is configured with one real upstream provider key.
- Gateway-lite rewrites forwarded auth headers to the configured upstream key.

Example:

```bash
GATEWAY_LITE_UPSTREAM_API_KEY=sk-real-openai-key \
gateway-lite run --upstream-base-url https://api.openai.com/v1
```

## What Gets Recorded

### Request/Response Records

`requests/model_inference.jsonl` includes:

- request timing (`request_start_time`, `request_end_time`, `request_duration_ms`)
- `api_token_hash` (SHA256 of the agent token)
- method/path/status
- request payload
- response payload
- response summary

### Lifecycle Events

`events/lifecycle.jsonl` includes:

- `job_start`
- `agent_start`
- `agent_end`
- `job_end`

### Manifest

`manifest.json` includes:

- run metadata (`run_id`, start/end time, return code)
- request count
- artifact checksums and sizes

## Privacy Notes

- Real API keys are not written to artifacts.
- Stored key material is hashed agent token identity (`api_token_hash`).
- Full prompts/completions are recorded in request artifacts, so avoid
  sensitive payloads unless intended.
