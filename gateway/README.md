# Gateway Service

FastAPI gateway in front of vLLM with:

- `POST /job/start`
- `POST /job/end`
- `POST /agent/start`
- `POST /agent/end`
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
