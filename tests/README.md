# End-to-End Test Workflow (AMDHPC, Single Node)

This runbook is aligned with the current codebase and uses:

- `servers/servers-amdhpc` control plane
- single profile (`-P <id>`)
- partition `mi2104x` (single-node flow; use `start`, not `start-group`)
- host `con-driver`
- `replayer`

## 0) Prerequisites

From repo root:

```bash
cp gateway/config.example.toml gateway/config.toml
python3 -m pip install -e ./servers/servers-amdhpc
```

Set required environment variables in your shell before startup (for example
`HF_HOME`, `HF_HUB_CACHE`, optional `HF_TOKEN`, and any image override vars
used by your cluster setup).

## 1) Choose Port Profile And Export Ports

Pick one profile ID (example uses `0`):

```bash
export PORT_PROFILE_ID=0
```

Load profile ports from `configs/port_profiles.toml`:

```bash
eval "$(python3 - <<'PY'
import os
import pathlib
import tomllib

profile_id = str(int(os.environ["PORT_PROFILE_ID"]))
cfg = tomllib.loads(pathlib.Path("configs/port_profiles.toml").read_text(encoding="utf-8"))
profile = cfg["profiles"][profile_id]
print(f'export VLLM_PORT={profile["vllm_port"]}')
print(f'export GATEWAY_PORT={profile["gateway_port"]}')
print(f'export GATEWAY_PARSE_PORT={profile["gateway_parse_port"]}')
print(f'export JAEGER_API_PORT={profile["jaeger_api_port"]}')
print(f'export JAEGER_OTLP_PORT={profile["jaeger_otlp_port"]}')
PY
)"
```

Optional quick print:

```bash
echo "profile=$PORT_PROFILE_ID vllm=$VLLM_PORT gw=$GATEWAY_PORT parse=$GATEWAY_PARSE_PORT jaeger_ui=$JAEGER_API_PORT jaeger_otlp=$JAEGER_OTLP_PORT"
```

## 2) Start AMDHPC Control Server (Login Node)

Run this on the AMD HPC login node (keep it running):

```bash
python3 servers/servers-amdhpc/server.py --config servers/servers-amdhpc/server_config.toml
```

If you launch from your local machine, run it through your SSH target in a
persistent shell/session (for example `tmux`).

## 3) Start Single-Node Runtime For The Selected Profile

From repo root on your local machine:

```bash
python3 servers/servers-amdhpc/client.py start \
  -P "$PORT_PROFILE_ID" \
  -p mi2104x \
  -m qwen3_coder_30b \
  -b
```

Notes:

- This is single-profile/single-node flow (no `start-group`).
- Client startup also launches a local gateway daemon for the selected profile.
- Default SSH target is `amd-hpc`; override with `-t <ssh-target>` if needed.

Health checks:

```bash
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models"
curl -s "http://127.0.0.1:${GATEWAY_PORT}/healthz"
curl -s "http://127.0.0.1:${GATEWAY_PARSE_PORT}/healthz"
```

## 4) Run con-driver Profile Job

```bash
mkdir -p tests/output
bash con-driver/run_con_driver.sh \
  --config con-driver/tests/config.gateway.toml \
  --port-profile-id "$PORT_PROFILE_ID"
```

Resolve the latest run directory:

```bash
RUN_DIR="$(ls -dt tests/output/con-driver/job-* tests/output/con-driver/*-* 2>/dev/null | head -n1)"
echo "$RUN_DIR"
```

Sanity checks:

```bash
cat "$RUN_DIR/meta/run_manifest.json"
cat "$RUN_DIR/meta/events.jsonl"
cat "$RUN_DIR/meta/results.json"
```

## 5) Compile Replay Plan

```bash
python -m replayer compile \
  --job-dir "$RUN_DIR" \
  --port-profile-id "$PORT_PROFILE_ID" \
  --plan-out "$RUN_DIR/replay-plan.json"
```

Compile now derives tokenizer endpoint from `--port-profile-id`; there is no
backend override and no compile-time `--agent-timeout-s`.

## 6) Reallocate Runtime For Clean vLLM Cache

Before replay, force a fresh node allocation so replay does not reuse warm KV
cache state from profiling.

Deallocate current runtime:

```bash
python3 servers/servers-amdhpc/client.py stop -P "$PORT_PROFILE_ID"
```

Allocate again on the same profile/partition/model:

```bash
python3 servers/servers-amdhpc/client.py start \
  -P "$PORT_PROFILE_ID" \
  -p mi2104x \
  -m qwen3_coder_30b \
  -b
```

Quick readiness checks:

```bash
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models"
curl -s "http://127.0.0.1:${GATEWAY_PORT}/healthz"
```

## 7) Replay And Validate

Replay:

```bash
REPLAY_OUT="tests/output/replayer/$(basename "$RUN_DIR").replayed-profile${PORT_PROFILE_ID}"
python -m replayer replay \
  --plan "$RUN_DIR/replay-plan.json" \
  --port-profile-id "$PORT_PROFILE_ID" \
  --output-dir "$REPLAY_OUT"
```

Inspect replay outputs:

```bash
cat "$REPLAY_OUT/replay/summary.json"
find "$REPLAY_OUT/replay/workers" -maxdepth 1 -type f | sort
```

Validate:

```bash
python -m replayer.validate \
  --source-job-dir "$RUN_DIR" \
  --replay-run-dir "$REPLAY_OUT" \
  --report-out "$REPLAY_OUT/replay/validation-report.json"
cat "$REPLAY_OUT/replay/validation-report.json"
```

## 8) Logs And Cleanup

Inspect current profile status/logs:

```bash
python3 servers/servers-amdhpc/client.py status -P "$PORT_PROFILE_ID"
python3 servers/servers-amdhpc/client.py logs -P "$PORT_PROFILE_ID" -n 200
```

Stop services and local daemons for this profile:

```bash
python3 servers/servers-amdhpc/client.py stop -P "$PORT_PROFILE_ID"
```

The `stop` flow blocks on remote shutdown, then tears down local gateway and
client-d for that profile.
