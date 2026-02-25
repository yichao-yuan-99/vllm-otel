# End-to-End Workflow: con-driver + host gateway Trace Collection

This runs a full job where:

- `vllm` + `jaeger` run in Docker
- `gateway` runs on host
- `con-driver` runs on host

Because gateway and con-driver are on host, `output_location` in `/job/start` is a normal host path.
con-driver sets it to a subdirectory inside each run output directory.

## 1) Start vLLM + Jaeger

From repo root:

```bash
cp docker/.env.example docker/.env
```

Set model/GPU values in `docker/.env`, export HF env vars in current shell, then start:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d --build
```

Check vLLM:

```bash
curl -s http://localhost:11451/v1/models
```

## 2) Start gateway on host

In a separate terminal:

```bash
bash docker/start_gateway.sh
```

`docker/start_gateway.sh` always runs gateway from shared `./.venv`.

Check gateway:

```bash
curl -s http://localhost:11457/healthz
```

## 3) Run con-driver

Use checked-in config: `con-driver/tests/config.gateway.toml`.

```bash
mkdir -p tests/output
bash con-driver/run_con_driver.sh --config con-driver/tests/config.gateway.toml
```

`con-driver/run_con_driver.sh` always runs con-driver from shared `./.venv` and forwards all args.
Do not add `api_key` manually; con-driver appends a unique token per launch.

## 4) Verify con-driver lifecycle + wrapper behavior

```bash
RUN_DIR="$(ls -dt tests/output/con-driver/* | head -n1)"
echo "$RUN_DIR"
cat "$RUN_DIR/meta/run_manifest.json"
cat "$RUN_DIR/meta/events.jsonl"
cat "$RUN_DIR/meta/results.json"
```

What to check:

- `run_manifest.json` has `gateway_enabled: true`, non-empty `gateway_output_location`, non-empty `gateway_output_relative_to_results_dir`, and populated `gateway_job_start_status` / `gateway_job_end_status`.
- `events.jsonl` includes `gateway_job_start` and `gateway_job_end`.
- `results.json` command entries include wrapper launch via `python -m con_driver.gateway_wrapper`.
- `results.json` command entries include appended `--agent-kwarg api_key=condrv_...` (unique token per launch).

## 5) Verify gateway artifact output

Gateway artifacts are written under the con-driver run directory:

```bash
ART_DIR="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["gateway_output_location"])' "$RUN_DIR/meta/run_manifest.json")"
echo "$ART_DIR"
ls -lh "$ART_DIR"
find "$ART_DIR" -maxdepth 2 -type f | sort
```

Inspect one artifact directory:

```bash
RUN_ART_DIR="$(find "$ART_DIR" -maxdepth 1 -type d -name 'run_*' | head -n1)"
echo "$RUN_ART_DIR"
```

Inspect files in that artifact:

```bash
cat "$RUN_ART_DIR/manifest.json"
cat "$RUN_ART_DIR/events/lifecycle.jsonl"
cat "$RUN_ART_DIR/requests/model_inference.jsonl"
```

## 6) Validate spans in Jaeger

Open `http://localhost:16686` and query `vllm-gateway` / `vllm-server`.

Use trace ID from artifact:

```bash
TRACE_ID="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[\"trace_id\"])' "$RUN_ART_DIR/manifest.json")"
echo "$TRACE_ID"
```

If you set `GATEWAY_ARTIFACT_COMPRESSION=tar.gz`, use `tar -tzf` / `tar -xzf` instead.

Expected shape:

- `agent_run` root span
- repeated `model_inference` spans for gateway -> vLLM calls
- `agent_action` spans between inference calls
- vLLM-side spans attached to propagated trace context

## 7) Logs and cleanup

Docker logs:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env logs -f vllm
docker compose -f docker/docker-compose.yml --env-file docker/.env logs -f jaeger
```

Gateway logs: from the terminal running `uvicorn`.

Stop docker stack:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env down
```
