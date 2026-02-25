# vLLM + Jaeger (Docker Compose) + Host Gateway

This directory runs:

- `jaeger` (trace backend + UI) in Docker
- `vllm` (OpenAI-compatible server with OTEL packages) in Docker
- `gateway` on the host (not in Docker)

## 1) Create env file

From repo root:

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env` as needed (`VLLM_MODEL_NAME`, `VLLM_SERVED_MODEL_NAME`, `VLLM_SERVICE_PORT`, `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_VISIBLE_DEVICES`, `GATEWAY_PORT`, `GATEWAY_OUTPUT_ROOT`, `GATEWAY_ARTIFACT_COMPRESSION`, `GATEWAY_JOB_END_TRACE_WAIT_SECONDS`).
`GATEWAY_OUTPUT_ROOT` in the example is project-root relative (`./tests/output/gateway-artifacts`).
`GATEWAY_ARTIFACT_COMPRESSION` defaults to `none` (uncompressed directory output). Set to `tar.gz` for compressed artifacts.
`GATEWAY_JOB_END_TRACE_WAIT_SECONDS` defaults to `10` and blocks `/job/end` before pulling Jaeger traces.
`VLLM_COLLECT_DETAILED_TRACES` defaults to `all` so vLLM emits spans into Jaeger.
Prompt token details are enabled in the vLLM container (`--enable-prompt-tokens-details`).
Custom logits processor loading is enabled via `VLLM_LOGITS_PROCESSORS` (default: `forceSeq.force_sequence_logits_processor:ForceSequenceAdapter`).
The force-sequence processor only activates for requests that include `vllm_xargs.forced_token_ids`.
At vLLM startup, `docker/vllm_entrypoint.sh` resolves `eos_token_id` from `VLLM_MODEL_NAME` and exports `VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID` automatically for the processor.
If tokenizer loading requires remote code, set `VLLM_FORCE_SEQ_TRUST_REMOTE_CODE=true`.

## 2) Export Hugging Face env vars in your current shell

```bash
export HF_HOME=/path/to/hf-home
export HF_HUB_CACHE=/path/to/hf-hub-cache
export HF_TOKEN=your_hf_token
```

`HF_TOKEN` can be unset for public models.

## 3) Start Docker services (jaeger + vllm)

Detached:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d --build
```

Foreground:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env up --build
```

## 4) Start gateway on host

In a separate terminal:

```bash
bash docker/start_gateway.sh
```

The script always creates/uses shared `./.venv` and runs gateway from it.

Optional flags:

```bash
bash docker/start_gateway.sh --skip-install
bash docker/start_gateway.sh --env-file docker/.env
```

## 5) Verify

```bash
curl http://localhost:11451/v1/models
curl http://localhost:11457/healthz
```

Use ports from `docker/.env` if changed.

Jaeger UI: `http://localhost:16686`

## 6) Logs

Docker services:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env logs -f vllm
docker compose -f docker/docker-compose.yml --env-file docker/.env logs -f jaeger
```

Gateway logs: from the terminal where `uvicorn` is running.

## 7) Gateway quick flow (optional)

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

Artifacts are written under `GATEWAY_OUTPUT_ROOT` on the host.

Force-sequence request example (directly to vLLM):

```bash
curl -s "http://localhost:${VLLM_SERVICE_PORT}/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model":"'"${VLLM_SERVED_MODEL_NAME}"'",
    "messages":[{"role":"user","content":"ignored"}],
    "max_tokens":16,
    "temperature":0,
    "vllm_xargs":{
      "forced_token_ids":[42,53,99],
      "force_eos_after_sequence":true
    }
  }'
```

## 8) OTEL smoke test (optional)

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env --profile test run --rm otel-smoke-test
```

## 9) Force-Sequence Smoke Test (optional)

This test tokenizes `VLLM_FORCE_SEQUENCE_TEST_TEXT` via `/tokenize`,
then sends `/v1/chat/completions` with `vllm_xargs.forced_token_ids`
and verifies the generated text exactly matches detokenized forced tokens.

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env --profile test run --rm force-seq-smoke-test
```

## 10) Stop

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env down
```

Also stop host gateway (`Ctrl+C` in the gateway terminal).
