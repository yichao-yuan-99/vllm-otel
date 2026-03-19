# Con-Driver with Gateway-Lite Upstream Key Injection

This document describes the recommended API-key flow for `con-driver` +
`gateway-lite`.

## Recommended Design

- `con-driver` keeps using unique per-trial agent IDs as API tokens.
- `gateway-lite` stores the real upstream provider key (for example OpenAI).
- For each `/v1/*` request, `gateway-lite` replaces the incoming key with the
  configured upstream key before forwarding.

This avoids passing real provider keys through `con-driver` or trial subprocess
arguments.

## Setup

Start gateway-lite with a fixed upstream key:

```bash
GATEWAY_LITE_UPSTREAM_API_KEY=sk-your-real-openai-api-key \
gateway-lite run \
  --upstream-base-url https://api.openai.com/v1
```

Then run `con-driver` normally in gateway mode (no real API key needed):

```bash
con-driver \
  --pool swebench-verified \
  --max-concurrent 5 \
  --n-task 10 \
  --results-dir ./output \
  --gateway \
  --gateway-url http://127.0.0.1:18171 \
  -- --agent mini-swe-agent --model openai/gpt-4.1-mini
```

## Behavior

1. `con-driver` generates `api_token` like `condrv_<run>_<idx>_<trial>`.
2. It uses that token for:
   - gateway `/agent/start` and `/agent/end`
   - trial-side API authentication header values
3. `gateway-lite` uses the token as the internal agent identity.
4. `gateway-lite` replaces forwarded auth headers with the configured real
   upstream key.

## Removed Option

`con-driver --gateway-api-key` is removed.
Configure the real key in `gateway-lite` instead (`--upstream-api-key` or
`GATEWAY_LITE_UPSTREAM_API_KEY`).
