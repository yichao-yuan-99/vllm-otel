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

## 2) Start runtime (vLLM + Jaeger + gateway)

```bash
cp -n gateway/config.example.toml gateway/config.toml
python3 servers/servers-docker/client.py start -m qwen3_coder_30b -p 0 -l h100_nvl_gpu23 -b
```

`servers/servers-docker/client.py start` also starts the local gateway daemon for the selected port profile.
Gateway serves both:

- raw gateway on `gateway_port`
- parsed gateway on `gateway_parse_port`

## 3) Run con-driver (inside `.venv`)

```bash
mkdir -p tests/output
con-driver --config con-driver/tests/config.gateway.toml
```

## 4) Expected output

- con-driver run dir: `tests/output/con-driver/job-<timestamp>/`
- gateway artifacts for that run: `tests/output/con-driver/job-<timestamp>/gateway-output/`
- run manifest: `tests/output/con-driver/job-<timestamp>/meta/run_manifest.json`

## Figures

- Figure-specific conventions live in `figures/README.md`.
- Every figure subdirectory under `figures/` should include both a detailed `README.md` and a concise `description.txt`.
- Each `description.txt` should include a paper-ready figure title followed by a single 2-3 sentence paragraph describing what the figure shows.

## Notes

- This README intentionally uses only `./.venv`.
- Wrapper scripts that create dedicated envs are optional and not required for this workflow.
