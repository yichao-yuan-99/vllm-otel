# Gateway Multi

`gateway_multi` is a standalone FastAPI package built from the ctx-aware gateway
stack that fronts multiple independent vLLM/Jaeger backend profiles with one
shared control-plane gateway.

It keeps the same lifecycle, proxy, context tracking, ctx-aware scheduling, and
SLO-aware scheduling
behavior as `gateway_ctx`, but changes the topology:

- only the first configured port profile exposes the public HTTP listeners
  - `GET /policy`
  - `POST /policy`
  - `GET /ctx-aware`
  - `POST /ctx-aware/start`
  - `POST /ctx-aware/end`
  - `GET /slo-aware`
  - `POST /slo-aware/start`
  - `POST /slo-aware/end`
  - `POST /job/start`
  - `POST /job/end`
  - `POST /agent/start`
  - `POST /agent/end`
  - `POST /v1/*` on that profile's `gateway_port`
  - `POST /v1/*` on that profile's `gateway_parse_port`
- every configured backend profile still gets its own IPC Unix socket
  - `GET /ipc/context`
  - `GET /ipc/output-throughput`
  - `GET /ipc/output-throughput/agents`
- each agent is assigned to one backend profile and stays pinned to it for the
  lifetime of the run

## Assignment Policy

`gateway_multi` takes a list of backend `port_profile_ids` and an
`assignment_policy`.

Currently the supported policies are:

- `balanced`: if any backend already has ongoing context usage strictly between
  `0` and `balanced_usage_threshold_tokens`, assign the new agent to the one
  with the lowest usage inside that subset; otherwise fall back to
  `lowest_usage`
- `round_robin`: assign agents evenly across backend profiles in order
- `lowest_usage`: assign each incoming agent to the backend with the lowest
  total `context_tokens` across ongoing agents; ties fall back to round-robin
  order across the tied backends
- `lowest_profile_without_pending`: scan backend profiles in ascending
  `port_profile_id` order and pick the first backend with no pending agents; if
  every backend already has pending agents, pick the backend with the smallest
  pending effective context usage, breaking ties by the lowest `port_profile_id`
  - this policy requires ctx-aware mode to be enabled before `POST /job/start`

Example with profiles `[2, 3]`:

- agent 1 -> profile `2`
- agent 2 -> profile `3`
- agent 3 -> profile `2`

`POST /agent/start` returns the selected `backend_port_profile_id`, and IPC
payloads also include that backend profile for each active agent.
Saved gateway request records in `requests/model_inference.jsonl` also include
`port_profile_id` for the backend that handled each request, along with
ctx-aware timing fields such as `forward_start_time`, `pending_duration_ms`,
and `span_duration_ms`.
Each run's `events/lifecycle.jsonl` also records a `port_profile_assignment`
event for the initial backend assignment, and for any future reassignment.

You can inspect and update the active assignment policy over HTTP:

- `GET /policy`: returns the current `assignment_policy`
- `POST /policy`: updates `assignment_policy`

`POST /policy` is only allowed when no job is running, meaning before
`POST /job/start` and after `POST /job/end`. The request body is:

```json
{
  "assignment_policy": "round_robin",
  "balanced_usage_threshold_tokens": 263856
}
```

`balanced_usage_threshold_tokens` is optional on `POST /policy` and only affects
the `balanced` policy. If omitted, the gateway keeps the existing threshold.

## Ctx-Aware Mode

The public control profile exposes the same ctx-aware control endpoints as
`gateway_ctx`:

- `GET /ctx-aware`
- `POST /ctx-aware/start`
- `POST /ctx-aware/end`

`gateway_multi` fans the selected ctx-aware configuration out to every backend
profile. Scheduling still happens independently inside each backend after an
agent has been assigned there, and `GET /ctx-aware` returns both aggregate
totals and per-backend summaries.

Each backend also writes its own ctx-aware sampler log under the shared job
output directory:

- `job/ctx_aware_<job_started_at>_profile-<port_profile_id>.jsonl`

## SLO-Aware Mode

The public control profile also exposes the same SLO-aware control endpoints as
`gateway_ctx`:

- `GET /slo-aware`
- `POST /slo-aware/start`
- `POST /slo-aware/end`

`gateway_multi` fans the selected SLO-aware configuration out to every backend
profile. Like ctx-aware mode, `ralexation` decisions still happen
independently inside each backend after an agent has been assigned there, and
`GET /slo-aware` returns both aggregate totals and per-backend summaries.

Each backend also writes its own SLO-aware decision log under the shared job
output directory:

- `job/slo_aware_decisions_<job_started_at>_profile-<port_profile_id>.jsonl`

Replay result note:

- in time-constrained replay runs, many workers can end near the global replay
  deadline even when ctx-aware scheduling is behaving as expected
- workers cut cleanly by the replay deadline are reported as
  `time_bound_finished`
- some workers that are still mid-request during teardown may instead surface
  as `failed` with a `cancel_failed` or remote-connection-closed error
- when those failures cluster tightly at the end of the run, treat them as a
  replay shutdown artifact first, not automatically as a gateway assignment or
  ctx-aware scheduling bug
- the per-profile ctx-aware job logs are still useful here: if
  `pending_agent_count` stays at `0`, the failures are usually not caused by
  the context-threshold pending mechanism

## Local Run

```bash
cp gateway_multi/config.example.toml gateway_multi/config.toml
python3 -m pip install -e ./gateway_multi
vllm-otel-gateway-multi start --config gateway_multi/config.toml
```

Module entrypoint:

```bash
python3 -m gateway_multi start --config gateway_multi/config.toml
```

Optional flags:

```bash
python3 -m gateway_multi start --config gateway_multi/config.toml --skip-install
python3 -m gateway_multi start --config gateway_multi/config.toml --port-profile-id 2 --port-profile-id 3
python3 -m gateway_multi start --config gateway_multi/config.toml --policy balanced
python3 -m gateway_multi start --config gateway_multi/config.toml --policy balanced --balanced-usage-threshold-tokens 263856
python3 -m gateway_multi start --config gateway_multi/config.toml --policy round_robin
python3 -m gateway_multi start --config gateway_multi/config.toml --policy lowest_usage
python3 -m gateway_multi start --config gateway_multi/config.toml --policy lowest_profile_without_pending
```

`--port-profile-id` may be repeated to override `run.port_profile_ids` from the
config file. The first selected profile becomes the control/public profile.

## Config

Runtime settings live in `gateway_multi/config.toml`:

- `[run].port_profile_ids`
- `[run].assignment_policy`
- `[run].balanced_usage_threshold_tokens`
- `[run].output_root`
- `[telemetry].service_name`
- `[telemetry].otlp_traces_insecure`
- `[gateway].artifact_compression`
- `[gateway].job_end_trace_wait_seconds`
- `[ipc].enabled`
- `[ipc].socket_path_template`
- `[ipc].socket_permissions`
- `[ipc].socket_uid`
- `[ipc].socket_gid`

Example:

```toml
[run]
port_profile_ids = [2, 3]
assignment_policy = "round_robin"
balanced_usage_threshold_tokens = 263856
```

Each backend profile resolves its own `vllm_port`, `jaeger_api_port`, and
`jaeger_otlp_port` from `configs/port_profiles.toml`.

The first profile in `port_profile_ids` also contributes the public
`gateway_port` and `gateway_parse_port`.

## IPC Sockets

IPC is enabled by default. Unless overridden, `gateway_multi` preserves the same
profile-specific socket naming used by `gateway`:

- profile `2`: `/tmp/vllm-gateway-profile-2.sock`
- profile `3`: `/tmp/vllm-gateway-profile-3.sock`

This makes `gateway_multi` a drop-in replacement for running one separate
`gateway` process per backend profile.

You can query them independently:

```bash
curl --unix-socket /tmp/vllm-gateway-profile-2.sock http://localhost/ipc/context
curl --unix-socket /tmp/vllm-gateway-profile-3.sock http://localhost/ipc/output-throughput
```

## Tests

```bash
python3 -m pip install -e './gateway_multi[dev]'
.venv/bin/python -m pytest gateway_multi/test -q
```
