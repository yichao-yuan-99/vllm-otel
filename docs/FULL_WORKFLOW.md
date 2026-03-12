# Full Workflow Runbook (Profile -> Replay -> Validate)

This runbook describes the full lifecycle for this project:

1. start runtime services (`vllm`, `jaeger`, host `gateway`)
2. run profiling (`con-driver` + gateway)
3. compile replay plan
4. replay workload
5. validate replay against the original profile

Use this as the operational checklist.

## 0) What You Get

Inputs:

- model + runtime config from CLI profile selection (`start -m/-p/-l`)
- con-driver config from `con-driver/tests/config.gateway.toml`

Outputs:

- profiled job directory: `tests/output/con-driver/job-<timestamp>/...`
- replay plan: `<job-dir>/replay-plan.json`
- replay output: `<job-dir>/replay-run-<timestamp>/...`
- validation report: `<replay-dir>/replay/validation-report.json`

## 1) Prerequisites

From repo root:

```bash
cp gateway/config.example.toml gateway/config.toml
```

Export Hugging Face env vars in your current shell:

```bash
export HF_HOME=/path/to/hf-home
export HF_HUB_CACHE=/path/to/hf-hub-cache
export HF_TOKEN=your_hf_token
```

Notes:

- `HF_TOKEN` can be unset for public models.
- default service ports used below:
  - vLLM: `11451`
  - gateway: `11457`
  - Jaeger UI: `16686`

## 2) Start Runtime

Start Docker runtime (`vllm` + `jaeger`) through the CLI package:

```bash
python3 servers/servers-docker/client.py start -m qwen3_coder_30b -p 0 -l h100_nvl_gpu23 -b
```

Start gateway on host (new terminal):

```bash
python3 -m gateway start --config gateway/config.toml --port-profile-id 0
```

Quick health checks:

```bash
curl -s http://localhost:11451/v1/models
curl -s http://localhost:11457/healthz
```

## 3) Run Profiling Job

From repo root:

```bash
mkdir -p tests/output
bash con-driver/run_con_driver.sh --config con-driver/tests/config.gateway.toml
```

Resolve latest run directory:

```bash
RUN_DIR="$(ls -dt tests/output/con-driver/job-* | head -n1)"
echo "$RUN_DIR"
```

Expected profile artifacts:

- `$RUN_DIR/meta/config.toml`
- `$RUN_DIR/meta/run_manifest.json`
- `$RUN_DIR/meta/results.json`
- `$RUN_DIR/gateway-output/run_*/...` (single profile) or
- `$RUN_DIR/gateway-output/profile-*/run_*/...` (cluster mode)

## 4) Compile Replay Plan

Compile from full profile directory:

```bash
python3 -m replayer compile \
  --job-dir "$RUN_DIR" \
  --port-profile-id 0 \
  --plan-out "$RUN_DIR/replay-plan.json"
```

Sanity check:

```bash
jq '.launch_policy,.workers|length' "$RUN_DIR/replay-plan.json"
```

Important:

- replay now requires `launch_policy` in plan (`config_ordered`).
- replay accepts optional runtime `--agent-timeout-s`.
- old plans without `launch_policy` are rejected by replay.

## 5) Execute Replay

Create replay output directory and run:

```bash
REPLAY_DIR="$RUN_DIR/replay-run-$(date -u +%Y%m%dT%H%M%SZ)"

python3 -m replayer replay \
  --plan "$RUN_DIR/replay-plan.json" \
  --port-profile-id 0 \
  --output-dir "$REPLAY_DIR"
```

Check replay summary:

```bash
cat "$REPLAY_DIR/replay/summary.json"
```

What replay launch timing does:

- preserves original worker launch order (`launch_priority`)
- uses scheduling policy from recorded con-driver config:
  - `max_concurrent`
  - `pattern`
  - `pattern_args`
  - `seed`
- does not replay exact absolute original launch offsets

## 6) Validate Replay

Run validator:

```bash
python3 -m replayer.validate \
  --source-job-dir "$RUN_DIR" \
  --replay-run-dir "$REPLAY_DIR" \
  --report-out "$REPLAY_DIR/replay/validation-report.json"
```

Inspect report:

```bash
cat "$REPLAY_DIR/replay/validation-report.json"
```

Interpretation:

- `exact.passed=true` is required for strict correctness.
- similarity metrics are diagnostic (timing drift visibility).

## 7) Troubleshooting

`Missing launch_policy in plan`:

- you are using an old compiled plan.
- re-run `python3 -m replayer compile ...` with current code.

Only `cached_tokens` exact mismatches:

- usually KV cache state drift between source and replay.
- restart vLLM, then replay + validate again.

Gateway `/job/end` timeout from con-driver:

- gateway blocks for trace collection (`gateway.job_end_trace_wait_seconds` in `gateway/config.toml`).
- increase `gateway_timeout_s` in con-driver config if needed.

## 8) Stop Services

Stop Docker runtime:

```bash
python3 servers/servers-docker/client.py stop -b
python3 servers/servers-docker/client.py daemon-stop
```

Stop gateway:

- press `Ctrl+C` in the terminal running `python3 -m gateway start ...`.
