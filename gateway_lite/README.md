# Gateway Lite

A lightweight version of the vLLM Gateway without Jaeger/OpenTelemetry
dependencies.

## Differences from Full Gateway

| Feature | Gateway | Gateway Lite |
|---------|---------|--------------|
| OpenTelemetry tracing | ✅ | ❌ |
| Jaeger trace export | ✅ | ❌ |
| Request/response recording | ✅ | ✅ |
| Lifecycle events | ✅ | ✅ |
| Response transformation (parsed port) | ✅ | ❌ (both ports identical) |
| Multiple backend support | ✅ | ✅ |

## Installation

```bash
pip install -e ./gateway_lite
```

## Usage

```bash
# Run the gateway-lite server
gateway-lite run

# Forward to OpenAI (or any OpenAI-compatible upstream)
GATEWAY_LITE_UPSTREAM_API_KEY=sk-real-openai-key \
gateway-lite run --upstream-base-url https://api.openai.com/v1

# Check health
gateway-lite health
```

## Configuration

Copy `config.example.toml` to `config.toml` and customize:

```toml
schema_version = 1

[run]
# Optional. If omitted, gateway-lite uses configs/port_profiles.toml default_profile.
port_profile_id = 0
output_root = "./tests/output/gateway-lite-artifacts"

[gateway]
artifact_compression = "none"
# Optional fixed key injected into forwarded upstream requests.
# upstream_api_key = "sk-real-openai-key"
```

## API Key Behavior

- Incoming API keys are used as internal agent IDs for gateway lifecycle
  tracking (`/agent/start`, `/agent/end`, `/v1/*`).
- If `upstream_api_key` is configured (or passed via
  `--upstream-api-key` / `GATEWAY_LITE_UPSTREAM_API_KEY`), gateway-lite rewrites
  forwarded auth headers to that key before sending upstream.
- No `agent_id@@real_api_key` parsing mode is supported.

## Output Artifacts

When a job ends, gateway-lite produces:

- `events/lifecycle.jsonl` - Job/agent lifecycle events
- `requests/model_inference.jsonl` - All request/response pairs
- `manifest.json` - Run summary

Note: No `trace/jaeger_trace.json` since Jaeger is not used.
