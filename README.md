# vllm-otel (Single-Venv Workflow)

This repo assumes one shared Python virtual environment at `./.venv` for local scripts/tests.

## Ports

- vLLM OpenAI API: `11451`
- Gateway API: `11457`
- Jaeger UI/API: `16686` (UI) and `4317` (OTLP gRPC ingest)

## 1) Setup `.venv` once

```bash
cd /scratch/yichaoy2/work/vllm-otel
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r gateway/requirements.txt
python -m pip install -e ./con-driver
```

## 2) Start Docker services (vLLM + Jaeger)

```bash
cp -n docker/.env.example docker/.env
docker compose -f docker/docker-compose.yml --env-file docker/.env up --build -d
```

## 3) Start gateway on host (inside `.venv`)

```bash
set -a
source docker/.env
set +a

export OTEL_SERVICE_NAME="${GATEWAY_OTEL_SERVICE_NAME:-vllm-gateway}"
export OTEL_EXPORTER_OTLP_TRACES_INSECURE="${GATEWAY_OTEL_EXPORTER_OTLP_TRACES_INSECURE:-true}"
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${GATEWAY_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-grpc://127.0.0.1:4317}"
export VLLM_BASE_URL="http://127.0.0.1:${VLLM_SERVICE_PORT:-11451}"
export JAEGER_API_BASE_URL="${GATEWAY_JAEGER_API_BASE_URL:-http://127.0.0.1:16686/api/traces}"
export GATEWAY_REQUEST_TIMEOUT_SECONDS="${GATEWAY_REQUEST_TIMEOUT_SECONDS:-120}"
export GATEWAY_ARTIFACT_COMPRESSION="${GATEWAY_ARTIFACT_COMPRESSION:-none}"
export GATEWAY_JOB_END_TRACE_WAIT_SECONDS="${GATEWAY_JOB_END_TRACE_WAIT_SECONDS:-10}"

python -m uvicorn gateway.app:app --host 0.0.0.0 --port "${GATEWAY_PORT:-11457}"
```

## 4) Run con-driver (inside `.venv`)

```bash
mkdir -p tests/output
con-driver --config con-driver/tests/config.gateway.toml
```

## 5) Expected output

- con-driver run dir: `tests/output/con-driver/job-<timestamp>/`
- gateway artifacts for that run: `tests/output/con-driver/job-<timestamp>/gateway-output/`
- run manifest: `tests/output/con-driver/job-<timestamp>/meta/run_manifest.json`

## Notes

- This README intentionally uses only `./.venv`.
- Wrapper scripts that create dedicated envs are optional and not required for this workflow.
