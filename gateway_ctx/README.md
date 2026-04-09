# Gateway Ctx

`gateway_ctx` is a standalone FastAPI package in front of vLLM with:

- raw and parsed `POST /v1/*` proxy listeners
- run lifecycle endpoints: `POST /job/start`, `POST /job/end`, `POST /agent/start`, `POST /agent/end`
- ctx-aware scheduling controls: `GET /ctx-aware`, `POST /ctx-aware/start`, `POST /ctx-aware/end`
- SLO-aware scheduling controls: `GET /slo-aware`, `POST /slo-aware/start`, `POST /slo-aware/end`
- IPC stats on a Unix socket: `GET /ipc/context`, `GET /ipc/output-throughput`, `GET /ipc/output-throughput/agents`

This package keeps the original gateway behavior and adds:

- a ctx-aware scheduler that can hold agents in `pending` until they fit inside configured context-token thresholds
- an SLO-aware extension that can temporarily move fast agents into `ralexation` so slower agents have a chance to recover toward a configured throughput target

## Local Run

```bash
cp gateway_ctx/config.example.toml gateway_ctx/config.toml
python3 -m pip install -e ./gateway_ctx
vllm-otel-gateway-ctx start --config gateway_ctx/config.toml
```

Module entrypoint:

```bash
python3 -m gateway_ctx start --config gateway_ctx/config.toml
```

Optional flags:

```bash
python3 -m gateway_ctx start --config gateway_ctx/config.toml --skip-install
python3 -m gateway_ctx start --config gateway_ctx/config.toml --port-profile-id 0
```

`--skip-install` skips the editable install into the shared `./.venv`. Use it only after the package is already installed there.

`gateway_ctx` resolves `gateway_port`, `gateway_parse_port`, `vllm_port`, `jaeger_api_port`, and `jaeger_otlp_port` from `configs/port_profiles.toml` using `run.port_profile_id` from `gateway_ctx/config.toml` or an explicit `--port-profile-id`.

## Ctx-Aware Mode

Ctx-aware mode is controlled at runtime over the main HTTP app:

- `GET /ctx-aware`
- `POST /ctx-aware/start`
- `POST /ctx-aware/end`

Enable payload:

```json
{
  "usage_threshold_tokens": 501280,
  "scheduling_threshold_tokens": 474897,
  "policy_mode": "throughput"
}
```

`policy_mode` is optional. Supported values are `age` and `throughput`. If omitted,
gateway keeps the original `age` policy.

Behavior:

- scheduling uses each agent's `effective_context_tokens`
- an agent with no usable `usage` yet counts as `3000` tokens
- new agents start `ongoing` only if they fit under `scheduling_threshold_tokens`
- age mode demotes the youngest ongoing agent first and promotes the oldest pending agent first
- throughput mode demotes the ongoing agent with the highest decode throughput first
- throughput mode promotes the pending agent with the lowest decode throughput first
- throughput mode uses `completion_tokens / request_time`; completed request time includes the full forwarded request duration, and pending agents also include their current queued wait
- pending requests wait inside gateway, and that wait time is included in request duration
- `POST /ctx-aware/start` and `POST /ctx-aware/end` return `409` while a job is active

## SLO-Aware Mode

SLO-aware mode is controlled at runtime over the main HTTP app:

- `GET /slo-aware`
- `POST /slo-aware/start`
- `POST /slo-aware/end`

Enable payload:

```json
{
  "target_tokens_per_s": 25.0,
  "policy_mode": "push-back-half-slack"
}
```

Behavior:

- ctx-aware mode must already be enabled before SLO-aware mode can be enabled
- `POST /slo-aware/start` and `POST /slo-aware/end` return `409` while a job is active
- SLO-aware configuration persists across jobs until `POST /slo-aware/end` or `POST /ctx-aware/end`
- the current throughput source is each agent's stored `current_output_tokens_per_s`
- `current_output_tokens_per_s` is computed from `total_completion_tokens / total_llm_request_duration_s`
- in `gateway_ctx`, `total_llm_request_duration_s` accumulates full end-to-end request
  time, including any gateway-side `pending` or `ralexation` wait before
  forwarding
- v1 supports one policy: `push-back-half-slack`
- the policy is only active while the minimum stored throughput across active agents is below the configured SLO target
- when an ongoing agent finishes a request and its stored throughput is above both the active-agent average and the SLO target, it can enter `ralexation` for half of its computed SLO slack
- `ralexation` blocks new requests from that agent just like `pending`, but it wakes up under SLO-aware rules instead of normal ctx-aware promotion
- if a `ralexation` agent wakes while any normal `pending` agent already exists, it becomes normal `pending`
- otherwise, a waking `ralexation` agent is re-admitted as if it were a new incoming agent under ctx-aware admission

Detailed semantics and examples live in:

- `gateway_ctx/README.slo-aware-model.md`

## IPC And Request Logs

IPC is enabled by default. Unless overridden, the default socket path is:

- profile `0`: `/tmp/vllm-gateway-ctx-profile-0.sock`
- profile `3`: `/tmp/vllm-gateway-ctx-profile-3.sock`

`GET /ipc/context` now includes:

- raw `total_context_tokens`
- additive ctx-aware fields such as `ctx_aware_policy_mode`, `effective_total_context_tokens`, `ongoing_agent_count`, `pending_agent_count`, `ongoing_effective_context_tokens`, and `pending_effective_context_tokens`
- additive SLO-aware fields such as `slo_aware_enabled`, `slo_target_tokens_per_s`, `slo_policy_mode`, `ralexation_agent_count`, and `ralexation_effective_context_tokens`
- per-agent `schedule_state`, `effective_context_tokens`, `has_usable_context_usage`, `pending_since`, `ralexation_until`, `output_tokens_per_s`, and `slo_slack_s`

Request records in `requests/model_inference.jsonl` now also include:

- `forward_start_time`
- `pending_duration_ms`
- `span_duration_ms`

Each job also writes a top-level ctx-aware sampler log at:

- `job/ctx_aware_<job_started_at>.jsonl`

Each job also writes a top-level SLO-aware decision log at:

- `job/slo_aware_decisions_<job_started_at>.jsonl`

This file is emitted at `5 Hz` and each JSON line contains:

- `ongoing_agent_count`
- `pending_agent_count`
- `ralexation_agent_count`
- `ongoing_effective_context_tokens`
- `pending_effective_context_tokens`
- `ralexation_effective_context_tokens`
- `slo_aware_enabled`
- `slo_target_tokens_per_s`
- `slo_policy_mode`
- `agents_turned_pending_due_to_context_threshold`
- `agents_turned_ongoing`
- `agents_turned_ralexation`
- `agents_left_ralexation_to_pending`
- `agents_left_ralexation_to_ongoing`
- `new_agents_added_as_pending`
- `new_agents_added_as_ongoing`

The SLO-aware decision log is event-driven rather than periodic. Each JSON line
records a `ralexation` decision such as:

- an agent entering `ralexation`
- an agent leaving `ralexation`
- the wake reason, resulting schedule state, and the agent's throughput/slack at that moment

Example:

```bash
curl --unix-socket /tmp/vllm-gateway-ctx-profile-0.sock http://localhost/ipc/context
curl --unix-socket /tmp/vllm-gateway-ctx-profile-0.sock http://localhost/ipc/output-throughput
```

Gateway still does not impose a client-side timeout on forwarded vLLM inference requests. If the downstream client disconnects, the gateway cancels the in-flight upstream request, or returns `499` immediately if the request was still waiting in a gateway-side blocked state such as `pending` or `ralexation`.

## Tests

```bash
python3 -m pip install -e './gateway_ctx[dev]'
pytest gateway_ctx/test -q
```
