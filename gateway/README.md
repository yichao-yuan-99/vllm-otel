# Gateway Service

FastAPI gateway in front of vLLM with:

- `POST /job/start`
- `POST /job/end`
- `POST /agent/start`
- `POST /agent/end`
- `GET /ipc/context` on an auto-enabled Unix domain socket
- `GET /ipc/output-throughput` on the same Unix domain socket
- `GET /ipc/output-throughput/agents` for per-agent output-throughput details
- `POST /v1/*` on `gateway_port` (raw proxy to vLLM)
- `POST /v1/*` on `gateway_parse_port` (same proxy, but chat-completion responses are reasoning-parsed in gateway)

Project assumption:

- vLLM itself is launched without server-side reasoning parsing.
- tool call parsing is intentionally not handled in gateway yet.

## Local run

```bash
cp gateway/config.example.toml gateway/config.toml
python3 -m pip install -e ./gateway
vllm-otel-gateway start --config gateway/config.toml
```

From repo root, you can also use the module entrypoint directly:

```bash
python3 -m gateway start --config gateway/config.toml
```

Optional flags:

```bash
python3 -m gateway start --config gateway/config.toml --skip-install
python3 -m gateway start --config gateway/config.toml --port-profile-id 0
```

`--skip-install` means the launcher will not run `pip install -e ./gateway` into the
shared `./.venv` before starting. Use it when the gateway package is already installed
in that venv and you only want a fast restart after config changes. Do not use it for
the first run in a fresh environment; if `./.venv` or the gateway package is missing,
startup will fail.

Gateway resolves `gateway_port`, `gateway_parse_port`, `vllm_port`, `jaeger_api_port`, and `jaeger_otlp_port`
from `configs/port_profiles.toml` using `run.port_profile_id` from `gateway/config.toml`
or an explicit `--port-profile-id` override.

Each selected port profile now exposes two gateway listeners:

- `gateway_port`: raw passthrough response from vLLM
- `gateway_parse_port`: gateway-parsed chat completions using the model's `reasoning_parser` from `configs/model_config.toml`

If the live served model does not configure a `reasoning_parser`, gateway still
binds both configured ports. In that case, `gateway_parse_port` behaves the
same as `gateway_port` because the response transformer is a no-op for that
model.

`python3 -m gateway start ...` can bootstrap the shared `./.venv` automatically when
the gateway package is not installed yet. Once installed into `./.venv`, the
`vllm-otel-gateway` console script is available there as well.

Gateway runtime settings now live in `gateway/config.toml`:

- `[telemetry].service_name`
- `[telemetry].otlp_traces_insecure`
- `[gateway].artifact_compression`
- `[gateway].job_end_trace_wait_seconds`
- `[run].output_root`
- `[ipc].enabled`
- `[ipc].socket_path`
- `[ipc].socket_permissions`
- `[ipc].socket_uid`
- `[ipc].socket_gid`

IPC is enabled by default. Unless overridden, gateway binds a profile-specific
Unix socket and exposes these local stats endpoints:

- `GET /ipc/context`
- `GET /ipc/output-throughput`
- `GET /ipc/output-throughput/agents`

- profile `0`: `/tmp/vllm-gateway-profile-0.sock`
- profile `3`: `/tmp/vllm-gateway-profile-3.sock`

This keeps different `port_profile_id` runs from colliding even when no IPC
settings are provided. The payload reports the number of active agents, the sum
of their current context sizes, and the per-agent context size keyed by
`api_token_hash`. Each agent's context size is set from the last completed
request with usage data:

- `context_tokens = prompt_tokens + completion_tokens`
- before an agent has any completed request with usage, its context is `0`
- if a later response has no usage block, gateway keeps the last known value

Gateway also tracks each active agent's average output throughput:

- `output_tokens_per_s = sum(completion_tokens) / sum(request_duration_s)`
- counters are updated when a request returns with `usage.completion_tokens`
- before an agent has any completed request with `completion_tokens`, its
  throughput is `null` and it is excluded from the summary min/max/avg
- if a later response has no `completion_tokens`, gateway keeps the last known
  throughput

`GET /ipc/output-throughput` reports:

- active `agent_count`
- `throughput_agent_count` used in the summary
- `min_output_tokens_per_s`
- `max_output_tokens_per_s`
- `avg_output_tokens_per_s`

`GET /ipc/output-throughput/agents` reports all active agents with:

- `api_token_hash`
- `output_tokens_per_s`

To disable IPC entirely:

```toml
[ipc]
enabled = false
```

Example:

```bash
curl --unix-socket /tmp/vllm-gateway-profile-0.sock http://localhost/ipc/context
curl --unix-socket /tmp/vllm-gateway-profile-0.sock http://localhost/ipc/output-throughput
```

With `none`, each run artifact is written as an uncompressed directory.

Gateway does not impose a client-side timeout on forwarded vLLM inference
requests. If vLLM keeps the connection open, gateway will continue waiting for
the upstream response.

If the downstream client disconnects before vLLM responds, gateway cancels the
in-flight upstream request as well. When the handler can still finalize a
result, gateway records and returns a `499` response with
`error = "client_disconnected"`.

## Tests

```bash
python3 -m pip install -e './gateway[dev]'
pytest gateway/test -q
```
