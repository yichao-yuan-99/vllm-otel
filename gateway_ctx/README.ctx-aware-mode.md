# Ctx-Aware Mode

This document describes the ctx-aware scheduling mode implemented in `gateway_ctx`.

## Goal

Ctx-aware mode keeps only a bounded set of agents eligible to send new vLLM
requests at any given time.

The bound is expressed in tokens, using two thresholds:

- `usage_threshold_tokens`: hard cap for the sum of `effective_context_tokens`
  across `ongoing` agents
- `scheduling_threshold_tokens`: lower cap used for new-agent admission and
  pending-agent promotion, so the scheduler does not immediately schedule
  agents back in after a demotion

This is an agent-level policy, not a request-level policy. Every active agent is
always in one of two states:

- `ongoing`: new requests from this agent may be forwarded immediately
- `pending`: new requests from this agent wait inside gateway until the agent is
  promoted back to `ongoing`

## Effective Context Size

Scheduling uses `effective_context_tokens`, not raw `context_tokens`.

For each agent:

- if the gateway has seen a completed response with usable `usage` data, then
  `effective_context_tokens = prompt_tokens + completion_tokens`
- otherwise the agent is treated as a new agent with pseudo size `3000`

Additional rules:

- a response without usable `usage` does not clear an older known context size
- before the first usable `usage`, `context_tokens` remains `0` but
  `effective_context_tokens` is still `3000`

## Control API

Ctx-aware mode is controlled through the main HTTP app.

### `GET /ctx-aware`

Returns current mode status and scheduler state, including:

- `enabled`
- `usage_threshold_tokens`
- `scheduling_threshold_tokens`
- `policy_mode`
- `new_agent_pseudo_tokens`
- `scheduler_interval_hz`
- `ongoing_agent_count`
- `pending_agent_count`
- `ongoing_effective_context_tokens`
- `pending_effective_context_tokens`
- per-agent scheduling details

### `POST /ctx-aware/start`

Request body:

```json
{
  "usage_threshold_tokens": 501280,
  "scheduling_threshold_tokens": 474897,
  "policy_mode": "throughput"
}
```

Validation:

- both fields are required integers
- both must be `> 0`
- `scheduling_threshold_tokens < usage_threshold_tokens`
- `scheduling_threshold_tokens >= 3000`
- `policy_mode` is optional
- if provided, `policy_mode` must be `age` or `throughput`
- if omitted, `policy_mode` defaults to `age`

Behavior:

- returns `409` if a job is active
- enables ctx-aware mode
- stores the thresholds in gateway runtime state
- stores the selected policy mode in gateway runtime state
- keeps the configuration across later `POST /job/end` and `POST /job/start`
  calls until `POST /ctx-aware/end` is called

### `POST /ctx-aware/end`

Behavior:

- returns `409` if a job is active
- disables ctx-aware mode
- clears both thresholds back to `null`

## Scheduling Behavior

The scheduler runs at `5 Hz` and also wakes immediately on relevant state
changes such as:

- `POST /ctx-aware/start`
- `POST /ctx-aware/end`
- `POST /agent/start`
- `POST /agent/end`
- completion of a request that updates context usage
- cleanup of a queued request that disconnects before forwarding

### New-agent admission

When a new agent starts:

- if ctx-aware mode is disabled, it starts as `ongoing`
- if ctx-aware mode is enabled, it starts as `ongoing` only if:

```text
ongoing_effective_context_tokens + 3000 <= scheduling_threshold_tokens
```

- otherwise it starts as `pending`

### Demotion

If total ongoing effective context grows above `usage_threshold_tokens`, the
scheduler demotes agents until the ongoing total fits.

Age-mode demotion order is:

- youngest ongoing agent first
- ordering key: `(run_start_time, api_token_hash, trace_id)` descending

Throughput-mode demotion order is:

- highest decode throughput first
- throughput = `total_completion_tokens / total_ctx_aware_request_time`
- `total_ctx_aware_request_time` is the sum of completed request durations for
  that agent
- demotion does not add current queue time because only `ongoing` agents are
  eligible

Demotion does not cancel an already forwarded upstream request. It only affects
later requests from that agent.

### Promotion

When capacity is available under `scheduling_threshold_tokens`, the scheduler
promotes pending agents in order:

Age-mode promotion order is:

- oldest pending agent first
- ordering key:
  `(pending_since_monotonic, run_start_time, api_token_hash, trace_id)`
  ascending

Throughput-mode promotion order is:

- lowest decode throughput first
- throughput uses the same completed-request totals as demotion
- if the agent currently has queued requests waiting behind the pending gate,
  the scheduler also adds that live queue wait into the throughput denominator
- ties fall back to the same pending-age ordering as age mode

Promotion continues until the next pending agent would exceed
`scheduling_threshold_tokens`.

### Oversized agents

There is no special escape hatch for an agent whose effective context no longer
fits by itself. Once demoted, that agent may remain pending indefinitely until
thresholds or agent mix change.

## Request Behavior

Ctx-aware mode changes request handling for pending agents.

### Pending wait

If an agent is `pending` when a request arrives:

- the request is held inside gateway
- nothing is forwarded upstream yet
- the request waits until the agent becomes `ongoing` or the client disconnects

### Timing semantics

Request timing and model-forward timing are tracked separately:

- `request_start_time`: when gateway received the request
- `forward_start_time`: when gateway actually began forwarding to vLLM
- `request_end_time`: when gateway finished handling the request
- `request_duration_ms`: full end-to-end duration
- `pending_duration_ms`: time spent queued because the agent was pending
- `span_duration_ms`: actual `model_inference` span duration

Important consequence:

- pending wait time counts toward overall request duration
- pending wait time does not count toward the `model_inference` span duration

### Disconnects

If the client disconnects while the request is still waiting in the pending
queue:

- gateway returns `499`
- no upstream request is sent to vLLM

If the client disconnects after forwarding has started, gateway cancels the
in-flight upstream request and records the same `499` style error as before.

### Agent shutdown

`POST /agent/end` does not drain in-flight work.

If the agent still has an upstream request in flight, the endpoint returns
`409`.

If the agent only has requests still queued behind ctx-aware pending gating,
gateway cancels those queued requests locally, records them with
`error = "agent_ended_while_pending"`, and does not forward them to vLLM
before completing the agent shutdown.

## Observability

Ctx-aware mode is visible in both the control API and IPC stats.

### `GET /ipc/context`

This endpoint keeps all existing fields and adds:

- `ctx_aware_enabled`
- `ctx_aware_policy_mode`
- `usage_threshold_tokens`
- `scheduling_threshold_tokens`
- `effective_total_context_tokens`
- `ongoing_agent_count`
- `pending_agent_count`
- `ongoing_effective_context_tokens`
- `pending_effective_context_tokens`

Each agent entry also includes:

- `schedule_state`
- `effective_context_tokens`
- `has_usable_context_usage`
- `pending_since`

### Request artifact logs

`requests/model_inference.jsonl` now also contains:

- `forward_start_time`
- `pending_duration_ms`
- `span_duration_ms`

These are additive fields; older fields such as `duration_ms` remain present for
compatibility.

### Job-level ctx-aware sampler log

Each active job also writes:

- `job/ctx_aware_<job_started_at>.jsonl`

The sampler writes one JSON line at `5 Hz` with:

- `ongoing_agent_count`
- `pending_agent_count`
- `ongoing_effective_context_tokens`
- `pending_effective_context_tokens`
- `agents_turned_pending_due_to_context_threshold`
- `agents_turned_ongoing`
- `new_agents_added_as_pending`
- `new_agents_added_as_ongoing`

Transition counters are counted since the previous sample.

## Example

Suppose:

- `usage_threshold_tokens = 501280`
- `scheduling_threshold_tokens = 474897`
- there are 100 active agents with total effective context `600000`

Then the scheduler may keep, for example:

- 90 agents in `ongoing`
- 10 agents in `pending`

Requests from those 10 pending agents stay queued in gateway until older
pending agents can be promoted without pushing the ongoing total above
`scheduling_threshold_tokens`.
