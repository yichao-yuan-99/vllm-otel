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
python -m pip install -e ./gateway
python -m pip install -e ./con-driver
```

## 2) Start Docker runtime (vLLM + Jaeger)

```bash
cp -n gateway/config.example.toml gateway/config.toml
python3 servers/servers-docker/client.py start -m qwen3_coder_30b -p 0 -l h100_nvl_gpu23 -b
```

## 3) Start gateway on host (inside `.venv`)

```bash
python -m gateway start --config gateway/config.toml --port-profile-id 0
```

This starts both:

- raw gateway on `gateway_port`
- parsed gateway on `gateway_parse_port`

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
