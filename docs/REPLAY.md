# Profile -> Replay Workflow (Complete Cycle)

This document describes the complete cycle from gateway profiling to replay validation.

References:

- plan schema: `docs/REPLAY-PLAN.md`
- replayer behavior/details: `docs/REPLAYER.md`
- docker + gateway startup: `docs/DOCKER_COMPOSE.md`

## 1) Start Runtime Services

From repo root:

```bash
cp docker/.env.example docker/.env
```

Export HF env vars in the current shell:

```bash
export HF_HOME=/path/to/hf-home
export HF_HUB_CACHE=/path/to/hf-hub-cache
export HF_TOKEN=your_hf_token
```

Start Docker services (`vllm` + `jaeger`):

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d --build
```

Start gateway on host (separate terminal):

```bash
bash docker/start_gateway.sh
```

Quick checks:

```bash
curl -s http://localhost:11451/v1/models
curl -s http://localhost:11457/healthz
```

## 2) Run a Profiling Job (con-driver + gateway)

Run con-driver with gateway-enabled config:

```bash
mkdir -p tests/output
bash con-driver/run_con_driver.sh --config con-driver/tests/config.gateway.toml
```

Get the latest profiled run:

```bash
RUN_DIR="$(ls -dt tests/output/con-driver/job-* | head -n1)"
echo "$RUN_DIR"
```

Expected profile contents:

- `meta/config.toml`
- `meta/run_manifest.json`
- `meta/results.json`
- `gateway-output/run_*/...`

## 3) Compile Profile -> Replay Plan

Compile from full profiled job directory:

```bash
python3 -m replayer compile \
  --job-dir "$RUN_DIR" \
  --plan-out "$RUN_DIR/replay-plan.json"
```

Important:

- input is the full job directory, not only `gateway-output`
- compiled plan includes required `launch_policy` (`config_ordered`) extracted from `meta/config.toml[run]`

## 4) Replay From Plan

Run replay:

```bash
REPLAY_DIR="$RUN_DIR/replay-run-$(date -u +%Y%m%dT%H%M%SZ)"

python3 -m replayer replay \
  --plan "$RUN_DIR/replay-plan.json" \
  --output-dir "$REPLAY_DIR" \
  --gateway-lifecycle auto
```

Replay launch semantics:

- preserves original launch ordering (`launch_priority`)
- launch timing is driven by recorded con-driver scheduling config (`max_concurrent`, `pattern`, `seed`)
- does not replay exact original absolute launch offsets

## 5) Validate Replay vs Source Profile

Run validation:

```bash
python3 -m replayer.validate \
  --source-job-dir "$RUN_DIR" \
  --replay-run-dir "$REPLAY_DIR" \
  --report-out "$REPLAY_DIR/replay/validation-report.json"
```

Validation checks:

- exact: worker mapping, request count, prompt content, response content, usage fields
- similarity: job/agent/request duration relative errors

Pass/fail:

- any exact mismatch -> failure
- similarity metrics are diagnostic

## 6) Common Issues

`Missing launch_policy in plan`:

- replay now requires new plans with `launch_policy`
- fix: re-run compile on the profile with current replayer

Only `cached_tokens` exact mismatches:

- typically KV cache state drift between source and replay
- fix: restart vLLM, then replay+validate again

Gateway `/job/end` timeout from con-driver:

- gateway waits before trace collection (`GATEWAY_JOB_END_TRACE_WAIT_SECONDS`)
- increase `gateway_timeout_s` in con-driver config if needed

## 7) Stop Services

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env down
```

Stop gateway with `Ctrl+C` in its terminal.
