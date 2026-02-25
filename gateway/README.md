# Gateway Service

FastAPI gateway in front of vLLM with:

- `POST /job/start`
- `POST /job/end`
- `POST /agent/start`
- `POST /agent/end`
- `POST /v1/*` (proxy to vLLM with span and request capture)

## Local run

```bash
python3 -m pip install -r gateway/requirements.txt
uvicorn gateway.app:app --host 0.0.0.0 --port 11457
```

Required runtime env vars:

- `VLLM_BASE_URL` (default: `http://localhost:11451`)
- `JAEGER_API_BASE_URL` (default: `http://localhost:16686/api/traces`)
- `OTEL_SERVICE_NAME` (default: `vllm-gateway`)
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (optional)
- `OTEL_EXPORTER_OTLP_TRACES_INSECURE` (default: `true`)
- `GATEWAY_ARTIFACT_COMPRESSION` (default: `none`; options: `none`, `tar.gz`)
- `GATEWAY_JOB_END_TRACE_WAIT_SECONDS` (default: `10`)

With `none`, each run artifact is written as an uncompressed directory.

## Tests

```bash
python3 -m pip install -r gateway/requirements-dev.txt
pytest gateway/test -q
```
