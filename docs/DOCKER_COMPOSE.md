# Docker Compose Setup (Current)

Compose runs only:

- `vllm`
- `jaeger`
- optional `otel-smoke-test`

Gateway is launched on the host, not in compose.

## Configuration

- `docker/.env` for compose + host gateway settings
- current shell env for `HF_HOME`, `HF_HUB_CACHE`, `HF_TOKEN`

## Quick Start

1. Copy template:

```bash
cp docker/.env.example docker/.env
```

Default `GATEWAY_OUTPUT_ROOT` is project-root relative: `./tests/output/gateway-artifacts`.
Default `GATEWAY_ARTIFACT_COMPRESSION` is `none` (uncompressed directories). Set `tar.gz` to emit compressed archives.
Gateway blocks `GATEWAY_JOB_END_TRACE_WAIT_SECONDS` (default: `10`) at `job/end` before collecting trace payloads.
vLLM tracing is enabled with `--otlp-traces-endpoint` and `--collect-detailed-traces` (`VLLM_COLLECT_DETAILED_TRACES=all` by default).
Prompt token details are enabled with `--enable-prompt-tokens-details`.
Custom logits processor loading is enabled with `VLLM_LOGITS_PROCESSORS` (default: `forceSeq.force_sequence_logits_processor:ForceSequenceAdapter`).
The processor is request-gated and only applies when `vllm_xargs.forced_token_ids` is present.
At vLLM startup, `docker/vllm_entrypoint.sh` infers `eos_token_id` from `VLLM_MODEL_NAME` and exports `VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID` automatically.
If tokenizer loading needs remote code, set `VLLM_FORCE_SEQ_TRUST_REMOTE_CODE=true` in `docker/.env`.

2. Export HF env vars:

```bash
export HF_HOME=/path/to/hf-home
export HF_HUB_CACHE=/path/to/hf-hub-cache
export HF_TOKEN=your_hf_token
```

3. Start compose services:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d --build
```

4. Start gateway on host (new terminal):

```bash
bash docker/start_gateway.sh
```

`docker/start_gateway.sh` always creates/uses shared `./.venv`.

## Verify

```bash
curl http://localhost:11451/v1/models
curl http://localhost:11457/healthz
```

Jaeger UI: `http://localhost:16686`

## Logs

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env logs -f vllm
docker compose -f docker/docker-compose.yml --env-file docker/.env logs -f jaeger
```

Gateway logs are in the terminal running `uvicorn`.

## Optional Gateway Flow

```bash
set -a
source docker/.env
set +a

curl -X POST "http://localhost:${GATEWAY_PORT}/job/start" \
  -H 'content-type: application/json' \
  -d '{"output_location":"'"${GATEWAY_OUTPUT_ROOT}"'/job-1"}'

curl -X POST "http://localhost:${GATEWAY_PORT}/agent/start" \
  -H 'content-type: application/json' \
  -d '{"api_token":"agent-token-1"}'

curl -X POST "http://localhost:${GATEWAY_PORT}/v1/completions" \
  -H 'content-type: application/json' \
  -H 'x-api-key: agent-token-1' \
  -d '{"model":"'"${VLLM_SERVED_MODEL_NAME}"'","prompt":"hello","max_tokens":16}'

curl -X POST "http://localhost:${GATEWAY_PORT}/agent/end" \
  -H 'content-type: application/json' \
  -d '{"api_token":"agent-token-1","return_code":0}'

curl -X POST "http://localhost:${GATEWAY_PORT}/job/end" \
  -H 'content-type: application/json' \
  -d '{"status":"completed"}'
```

Artifacts are written to host path `GATEWAY_OUTPUT_ROOT`.

## Stop

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env down
```

Stop host gateway with `Ctrl+C`.

## Optional Force-Sequence Test

Run:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env --profile test run --rm force-seq-smoke-test
```
