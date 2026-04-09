# SLO-Aware Mode

This document is the detailed README for the SLO-aware scheduling mode in
`gateway_ctx`.

SLO-aware mode is built on top of ctx-aware mode. It does not replace ctx-aware
admission control. Instead, it adds a second layer that can temporarily hold
back agents with spare throughput headroom so that slower agents have a better
chance to recover toward a target throughput.

## Prerequisite

SLO-aware mode requires ctx-aware mode to already be enabled.

That means:

- `POST /ctx-aware/start` must succeed before `POST /slo-aware/start`
- `POST /slo-aware/start` returns `409` if ctx-aware mode is not enabled
- turning off ctx-aware mode also clears SLO-aware configuration

## Control API

Runtime endpoints:

- `GET /slo-aware`
- `POST /slo-aware/start`
- `POST /slo-aware/end`

Enable request body:

```json
{
  "target_tokens_per_s": 25.0,
  "policy_mode": "push-back-80p-slack"
}
```

Rules:

- `target_tokens_per_s` must be positive
- `policy_mode` is required
- supported values are `push-back-half-slack` and `push-back-80p-slack`
- SLO-aware control changes are only allowed while no job is active
- settings persist across jobs until `POST /slo-aware/end` or `POST /ctx-aware/end`

## Throughput Source

SLO-aware mode reuses the throughput already stored per agent:

- `current_output_tokens_per_s`

That value is derived from:

- `total_completion_tokens`
- `total_llm_request_duration_s`, which in `gateway_ctx` accumulates full
  end-to-end request time including any gateway-side `pending` or `ralexation`
  wait before forwarding

Current throughput definition:

```text
current_output_tokens_per_s =
    total_completion_tokens / total_llm_request_duration_s
```

SLO-aware mode does not introduce a second instantaneous throughput metric.

## SLO Slack

Let:

- `x` be the configured SLO target in tokens per second
- `y` be the agent's stored `current_output_tokens_per_s`
- `t` be the matching `total_llm_request_duration_s`

Then the SLO slack is:

```text
slack_s = t' - t = (y * t) / x - t = t * (y / x - 1)
```

Equivalent implementation form:

```text
slack_s =
    total_completion_tokens / target_tokens_per_s
    - total_llm_request_duration_s
```

Interpretation:

- positive slack means the agent has spare delay budget
- zero or negative slack means the agent has no delay budget to spend

## States

Each active agent is in one of these scheduling states:

- `ongoing`
- `pending`
- `ralexation`

Meaning:

- `ongoing`: new requests may be forwarded immediately
- `pending`: ctx-aware mode is currently blocking new requests
- `ralexation`: SLO-aware mode is currently blocking new requests

Both `pending` and `ralexation` keep queued requests inside gateway. Neither
state cancels an already forwarded upstream request.

## Policy

Supported SLO-aware policies are:

- `push-back-half-slack`
- `push-back-80p-slack`

The policy is globally active only when the minimum stored throughput among
active agents with usable throughput is below the configured SLO target.

Agents without stored throughput yet:

- do not participate in the minimum-throughput check
- do not participate in the average-throughput check
- cannot be selected for `ralexation`

## Entering `ralexation`

The gateway evaluates `ralexation` after an ongoing agent finishes a request and
its stored throughput has been updated.

An agent enters `ralexation` only if all of these are true:

- ctx-aware rebalance has already run, and the agent is still `ongoing`
- SLO-aware mode is enabled
- the global minimum stored throughput is below the SLO target
- the agent's stored throughput is greater than the average stored throughput of
  active agents with usable throughput
- the agent's stored throughput is greater than the SLO target
- the agent's computed SLO slack is positive

Duration:

```text
push-back-half-slack: ralexation_duration_s = slo_slack_s * 0.5
push-back-80p-slack: ralexation_duration_s = slo_slack_s * 0.8
```

## Leaving `ralexation`

A `ralexation` agent becomes ready to leave that state when either of these is
true:

- its `ralexation` timer has expired
- the global minimum stored throughput has risen back to or above the SLO target

Wake-up behavior:

- if any normal `pending` agent already exists, the waking `ralexation` agent
  becomes normal `pending`
- otherwise, the waking agent is treated as if it were a new incoming agent
  under ctx-aware admission
- if that ctx-aware admission check succeeds, it becomes `ongoing`
- otherwise, it becomes `pending`

When a `ralexation` agent turns into normal `pending`, it joins the pending
queue as newly pending. Its `pending_since` timestamp starts at that transition,
not at the earlier `ralexation` start time.

## Priority Rules

Ctx-aware blocking takes priority over SLO-aware blocking.

Practical consequences:

- an agent that should already be `pending` is not moved into `ralexation`
- request completion runs normal ctx-aware rebalance before any SLO-aware
  `ralexation` decision
- normal pending agents always keep priority over waking `ralexation` agents

## Request Path Behavior

Requests from agents in `pending` or `ralexation` wait inside gateway before any
upstream forwarding happens.

That wait:

- counts toward `request_duration_ms`
- counts toward `pending_duration_ms`
- counts toward `total_llm_request_duration_s` and therefore reduces the stored
  `current_output_tokens_per_s` used by SLO-aware mode and `/ipc/output-throughput`
- does not count toward the forwarded `model_inference` span duration

If the downstream client disconnects before the agent becomes runnable again,
gateway returns `499` without forwarding anything upstream.

## Observability

### `GET /slo-aware`

This endpoint exposes:

- whether SLO-aware mode is enabled
- whether ctx-aware mode is enabled
- the configured target throughput
- the policy name
- current min and average stored throughput
- current `ralexation` counts
- per-agent state including `schedule_state`, `output_tokens_per_s`,
  `slo_slack_s`, and `ralexation_until`

### `GET /ipc/context`

Additive SLO-aware fields include:

- `slo_aware_enabled`
- `slo_target_tokens_per_s`
- `slo_policy_mode`
- `ralexation_agent_count`
- `ralexation_effective_context_tokens`

Per-agent additive fields include:

- `output_tokens_per_s`
- `slo_slack_s`
- `ralexation_until`

### Job sampler log

The existing ctx-aware sampler log also records SLO-aware fields when present,
including:

- `ralexation_agent_count`
- `ralexation_effective_context_tokens`
- `slo_aware_enabled`
- `slo_target_tokens_per_s`
- `slo_policy_mode`
- `agents_turned_ralexation`
- `agents_left_ralexation_to_pending`
- `agents_left_ralexation_to_ongoing`

### SLO-aware decision log

Each job also writes an event-driven decision log at:

- `job/slo_aware_decisions_<job_started_at>.jsonl`

This file records discrete SLO-aware state transitions rather than periodic
samples. Typical events include:

- `agent_entered_ralexation`
- `agent_left_ralexation`

Each event includes the agent identity and decision metadata such as:

- `from_schedule_state`
- `to_schedule_state`
- `wake_reason`
- `resume_disposition`
- `output_tokens_per_s`
- `slo_slack_s`
- `ralexation_duration_s`
- `ralexation_until`
- current min and average stored throughput across active agents

## Example Flow

1. Ctx-aware mode is enabled.
2. SLO-aware mode is enabled with target `25 tokens/s`.
3. One active agent falls to `10 tokens/s`.
4. Another agent completes a request and is now running at `60 tokens/s`.
5. If that `60 tokens/s` agent is also above the active-agent average and has
   positive slack, it enters `ralexation`.
6. While in `ralexation`, new requests from that agent wait in gateway.
7. When the `ralexation` timer expires, or when the minimum throughput rises
   back to at least `25 tokens/s`, the agent is reconsidered.
8. If normal `pending` agents already exist, it joins that queue.
9. Otherwise, it is re-admitted using the same ctx-aware rule as a new agent.
