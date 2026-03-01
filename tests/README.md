# End-to-End Test Workflow

This document reflects the current repo behavior for:

- `servers/servers-docker`
- host `gateway`
- host `con-driver`
- `replayer`

All examples below use `port_profile_id = 3`.

## Port Profile 3

From `configs/port_profiles.toml`:

- `vllm_port = 40823`
- `gateway_port = 40857`
- `gateway_parse_port = 48171`
- `jaeger_api_port = 44612`
- `jaeger_otlp_port = 41735`

Use these checks:

```bash
curl -s http://localhost:40823/v1/models
curl -s http://localhost:40857/healthz
curl -s http://localhost:48171/healthz
```

Do not use `http://localhost:40823/healthz` for vLLM; in this repo the vLLM
readiness check is `/v1/models`, not `/healthz`.

## Current Behavior Summary

- Gateway always binds both `gateway_port` and `gateway_parse_port`.
- If the served model has no configured `reasoning_parser`, the parse-port
  listener is still present but behaves the same as the raw port.
- Gateway artifacts always record the raw vLLM response in
  `requests/model_inference.jsonl`, even if the client talked to
  `gateway_parse_port`.
- `con-driver` uses `gateway_parse_port` by default when `port_profile_id` is
  set.
- `replayer` compiles and replays against the raw `gateway_port`, not the parse
  port.
- Harbor dataset downloads are cached in
  `con-driver/.cache/harbor-datasets/`, not inside `results_dir`.
- Gateway does not impose a client-side timeout on forwarded vLLM requests.
- If the downstream client disconnects, gateway cancels the upstream vLLM
  request as well.

## 1. Start vLLM + Jaeger

From repo root:

```bash
cp gateway/config.example.toml gateway/config.toml
```

Export any required Hugging Face environment variables in the current shell,
then start the Docker runtime on profile `3`:

```bash
python3 servers/servers-docker/client.py start -m qwen3_coder_30b -p 3 -l h100_nvl_gpu23 -b
```

Check vLLM:

```bash
curl -s http://localhost:40823/v1/models
```

## 2. Start Gateway On Host

In a separate terminal:

```bash
python3 -m gateway start --config gateway/config.toml --port-profile-id 3
```

Check both listeners:

```bash
curl -s http://localhost:40857/healthz
curl -s http://localhost:48171/healthz
```

Listener semantics:

- `40857`: raw gateway listener
- `48171`: parsed gateway listener

If the served model has no configured reasoning parser, both ports still exist
but their client-visible behavior is the same.

## 3. Run con-driver

Use the checked-in gateway config and override the profile on the CLI:

```bash
mkdir -p tests/output
bash con-driver/run_con_driver.sh \
  --config con-driver/tests/config.gateway.toml \
  --port-profile-id 3
```

Notes:

- `con-driver` runs from the shared `./.venv`.
- Do not pass `api_key` manually; `con-driver` injects a unique token per run.
- Harbor datasets are downloaded into the shared cache:

```text
con-driver/.cache/harbor-datasets/
```

## 4. Inspect The Latest con-driver Run

```bash
RUN_DIR="$(ls -dt tests/output/con-driver/job-* tests/output/con-driver/*-* 2>/dev/null | head -n1)"
echo "$RUN_DIR"
cat "$RUN_DIR/meta/run_manifest.json"
cat "$RUN_DIR/meta/events.jsonl"
cat "$RUN_DIR/meta/results.json"
```

What to check:

- `run_manifest.json` has `gateway_enabled: true`
- `run_manifest.json` has non-empty `gateway_output_location`
- `events.jsonl` includes `gateway_job_start` and `gateway_job_end`
- `results.json` launch commands go through `python -m con_driver.gateway_wrapper`
- launch commands include an appended `api_key=condrv_...`

## 5. Inspect Gateway Artifacts

Gateway artifacts live inside each con-driver run:

```bash
ART_DIR="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[\"gateway_output_location\"])' "$RUN_DIR/meta/run_manifest.json")"
echo "$ART_DIR"
find "$ART_DIR" -maxdepth 2 -type f | sort
```

Pick one artifact directory:

```bash
RUN_ART_DIR="$(find "$ART_DIR" -maxdepth 1 -type d -name 'run_*' | head -n1)"
echo "$RUN_ART_DIR"
```

Inspect the files:

```bash
cat "$RUN_ART_DIR/manifest.json"
cat "$RUN_ART_DIR/events/lifecycle.jsonl"
cat "$RUN_ART_DIR/requests/model_inference.jsonl"
```

Important artifact contract:

- `requests/model_inference.jsonl` stores the exact raw vLLM `response`
- this is true even when the client used `gateway_parse_port`
- if the parse listener rewrote the client-visible payload, that rewrite is not
  what gets stored in the artifact

## 6. Check Jaeger

Use the trace ID from the artifact:

```bash
TRACE_ID="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[\"trace_id\"])' "$RUN_ART_DIR/manifest.json")"
echo "$TRACE_ID"
```

Open Jaeger at:

```text
http://localhost:44612
```

Expected shape:

- `agent_run` root span
- repeated `model_inference` spans for gateway -> vLLM calls
- `agent_action` spans between inference calls
- vLLM-side spans attached to the propagated trace context

## 7. Compile Replay Plan

Compile from the full profiled con-driver job:

```bash
python -m replayer compile \
  --job-dir "$RUN_DIR" \
  --port-profile-id 3 \
  --plan-out "$RUN_DIR/replay-plan.json"
```

Check the compiled target:

```bash
cat "$RUN_DIR/replay-plan.json"
```

Current replay contract:

- replay targets raw gateway `40857`
- replay does not target parsed gateway `48171`
- replay compares against the raw artifact payloads recorded by gateway

## 8. Run Replay

```bash
REPLAY_OUT="tests/output/replayer/$(basename "$RUN_DIR").replayed-profile3"
python -m replayer replay \
  --plan "$RUN_DIR/replay-plan.json" \
  --port-profile-id 3 \
  --output-dir "$REPLAY_OUT" \
  --gateway-lifecycle auto
```

Inspect replay output:

```bash
cat "$REPLAY_OUT/replay/summary.json"
find "$REPLAY_OUT/replay/workers" -maxdepth 1 -type f | sort
```

## 9. Validate Replay

```bash
python -m replayer.validate \
  --source-job-dir "$RUN_DIR" \
  --replay-run-dir "$REPLAY_OUT" \
  --report-out "$REPLAY_OUT/replay/validation-report.json"
```

Inspect:

```bash
cat "$REPLAY_OUT/replay/validation-report.json"
```

## 10. Logs And Cleanup

Docker logs:

```bash
python3 servers/servers-docker/client.py logs -n 200
```

Direct vLLM container logs:

```bash
docker logs --tail 200 vllm-openai-otel-lp
```

Stop the Docker runtime:

```bash
python3 servers/servers-docker/client.py stop -b
python3 servers/servers-docker/client.py daemon-stop
```
