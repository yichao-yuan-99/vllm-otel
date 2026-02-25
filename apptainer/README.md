# Apptainer on HPC (AMD MI325X, no Docker)

This directory provides an Apptainer-only workflow that mirrors `docker/docker-compose.yml` for:

- Jaeger all-in-one (`jaegertracing/all-in-one:1.57`)
- vLLM ROCm + OTEL image (`yichaoyuan/vllm-openai-otel:v0.14.1-otel-lp-rocm`)

It also runs both existing smoke tests:

- `docker/tests/dummy_otel_client.py`
- `docker/tests/force_sequence_smoke_test.py`

## 1) Prepare env file

```bash
cp apptainer/.env.example apptainer/.env
```

Edit `apptainer/.env` and set at least:

- `VLLM_MODEL_NAME`
- `VLLM_SERVED_MODEL_NAME`

Assumption: `HF_HOME`, `HF_HUB_CACHE`, and optional `HF_TOKEN` are already exported in your shell (cluster module/profile env). You can still override them in `apptainer/.env` if needed.

Load variables from `apptainer/.env` into the current shell when you want to use `${VLLM_SERVICE_PORT}` (and other vars) directly:

```bash
set -a
source apptainer/.env
set +a
```

For MI325X x8, default is already:

- `VLLM_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- `VLLM_TENSOR_PARALLEL_SIZE=8`

## 2) Pull/convert OCI images to SIF

```bash
bash apptainer/run_services.sh --env-file apptainer/.env pull
```

By default SIF files are stored under `${APPTAINER_IMGS}`.

## 3) Start Jaeger + vLLM

```bash
bash apptainer/run_services.sh --env-file apptainer/.env start
```

Health checks used by the script:

- Jaeger UI: `http://127.0.0.1:16686`
- vLLM models API: `http://127.0.0.1:${VLLM_SERVICE_PORT}/v1/models`

## 4) Run smoke tests

```bash
bash apptainer/run_services.sh --env-file apptainer/.env test
```

This executes both tests inside the vLLM SIF with host networking:

1. OTEL client smoke test (expects spans exported to Jaeger OTLP gRPC `127.0.0.1:4317`)
2. Force-sequence logits processor smoke test

## 5) Inspect status/logs

```bash
bash apptainer/run_services.sh --env-file apptainer/.env status
bash apptainer/run_services.sh --env-file apptainer/.env logs
```

Runtime files are written to:

- `apptainer/run/` (pid files)
- `apptainer/logs/` (service logs)

## 6) Stop services

```bash
bash apptainer/run_services.sh --env-file apptainer/.env stop
```

## Notes

- `pull` and `start` are intentionally separate; run `pull` once before first `start` (or whenever image tags change).
- vLLM is launched with `--rocm` and exports both `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`.
- vLLM/aiter compile caches are redirected to `${TMPDIR:-/tmp}` by default (via `AITER_JIT_DIR`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`) to avoid filling `~/.cache`.
- The same custom logits processor as Docker Compose is enabled:
  - `forceSeq.force_sequence_logits_processor:ForceSequenceAdapter`
