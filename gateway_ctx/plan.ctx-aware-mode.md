# Context-Aware Mode Implementation Plan

## 1. Goal

Implement a gateway-side scheduling mode that keeps only a bounded set of agents
eligible to send new vLLM requests.

The bound is expressed in tokens:

- `usage_threshold_tokens`: hard cap for the sum of ongoing agents' effective
  context tokens.
- `scheduling_threshold_tokens`: lower cap used for re-admission, so we do not
  immediately schedule agents back in after demotion.

This mode is agent-level, not request-level:

- every active agent is always either `ongoing` or `pending`
- `ongoing` means a new request may be forwarded immediately
- `pending` means a new request waits inside gateway until the agent is
  promoted back to `ongoing`

## 2. Clarified Semantics

The original README leaves a few behaviors implicit. For implementation we
should make them explicit:

- Scheduling uses `effective_context_tokens`, not raw `current_context_tokens`.
- `effective_context_tokens` is:
  - the last known `prompt_tokens + completion_tokens` from a completed
    response with a usable `usage` block
  - otherwise `3000` for agents that have not yet produced any usable usage
    data
- A response without a usable `usage` block does not clear an older value.
- A newly started agent is admitted as `ongoing` only if:
  - `ongoing_effective_context_tokens + 3000 <= scheduling_threshold_tokens`
- If that check fails, the new agent starts as `pending`.
- Demotion order is "youngest ongoing first", using:
  - `(run_start_time, api_token_hash, trace_id)` descending
- Promotion order is "oldest pending first", using:
  - `(pending_since_monotonic, run_start_time, api_token_hash, trace_id)`
    ascending
- Demotion never cancels an already forwarded upstream request.
- Demotion only affects later requests from that agent.
- Pending wait time counts toward request end-to-end duration.
- Pending wait time does not count toward the `model_inference` span duration.
- The 5 Hz periodic scheduler is required, but state mutations should also wake
  it immediately so we do not wait a full 200 ms after every change.

## 3. Explicit Assumptions

To keep v1 implementable and predictable, we should lock in these assumptions:

- Thresholds are service-wide runtime state in gateway memory, not per-job
  config written into `gateway/config.toml`.
- Control happens through normal HTTP endpoints on the main gateway app, similar
  to `/job/start` and `/job/end`; no separate admin process is added.
- We may assume ctx-aware settings are only changed when no job is active.
  To keep the behavior explicit, `POST /ctx-aware/start` and
  `POST /ctx-aware/end` should return `409` if `job_active` is true.
- Once enabled, ctx-aware settings persist across `POST /job/end` and later
  `POST /job/start` calls until `POST /ctx-aware/end` is called explicitly.
- We keep the current context definition from [`gateway/README.md`](/srv/scratch/yichaoy2/work/vllm-otel/gateway/README.md):
  - context size is learned from the last completed response, not predicted from
    the next request payload
- We do not add request-body parsing heuristics to estimate future prompt growth.
- We assume the configured thresholds are large enough that a single agent can
  fit by itself.
- If an agent's effective size grows beyond the thresholds anyway, v1 does not
  invent a special escape hatch. That agent will simply stop fitting and remain
  pending once demoted. Status output must make that visible.

## 4. External API Changes

### 4.1 New control endpoints

Add these routes to the existing FastAPI app:

- `GET /ctx-aware`
- `POST /ctx-aware/start`
- `POST /ctx-aware/end`

These are regular HTTP routes on the main gateway listener, not IPC socket
routes.

`POST /ctx-aware/start` request body:

```json
{
  "usage_threshold_tokens": 501280,
  "scheduling_threshold_tokens": 474897
}
```

Validation rules:

- both fields are required integers
- both must be `> 0`
- `scheduling_threshold_tokens < usage_threshold_tokens`
- `scheduling_threshold_tokens >= 3000`

Enable behavior:

- if a job is active, return `409`
- set ctx-aware mode to enabled
- store both thresholds in gateway service state
- keep that configuration for subsequent jobs until `POST /ctx-aware/end`
- refresh cached scheduler totals; because this endpoint is only valid while no
  job is active, there are no active agents to rebalance
- return the same payload as `GET /ctx-aware`

`POST /ctx-aware/end` behavior:

- if a job is active, return `409`
- set ctx-aware mode to disabled
- clear both threshold fields back to `null`
- this is the only operation that clears ctx-aware configuration at runtime
- refresh cached scheduler totals; because this endpoint is only valid while no
  job is active, there are no active agents or pending requests to release
- return the same payload as `GET /ctx-aware`

### 4.2 `GET /ctx-aware` response

Return the scheduler control/status view:

```json
{
  "status": "ok",
  "enabled": true,
  "usage_threshold_tokens": 501280,
  "scheduling_threshold_tokens": 474897,
  "new_agent_pseudo_tokens": 3000,
  "scheduler_interval_hz": 5,
  "ongoing_agent_count": 90,
  "pending_agent_count": 10,
  "ongoing_effective_context_tokens": 470000,
  "pending_effective_context_tokens": 130000,
  "agents": [
    {
      "api_token_hash": "...",
      "trace_id": "...",
      "run_start_time": "2026-04-03T05:29:36.000Z",
      "schedule_state": "pending",
      "context_tokens": 0,
      "effective_context_tokens": 3000,
      "has_usable_context_usage": false,
      "pending_since": "2026-04-03T05:30:10.000Z"
    }
  ]
}
```

### 4.3 Additive `GET /ipc/context` changes

Keep all current fields unchanged, and add these fields:

- `ctx_aware_enabled`
- `usage_threshold_tokens`
- `scheduling_threshold_tokens`
- `effective_total_context_tokens`
- `ongoing_agent_count`
- `pending_agent_count`
- `ongoing_effective_context_tokens`
- `pending_effective_context_tokens`

Add these per-agent fields to the existing `agents` list:

- `schedule_state`
- `effective_context_tokens`
- `has_usable_context_usage`
- `pending_since`

This keeps existing consumers working while making the scheduler observable.

## 5. Data Model Changes

The current implementation stores per-agent state in `RunState` inside
[`gateway/app.py`](/srv/scratch/yichaoy2/work/vllm-otel/gateway/app.py). Extend
that structure rather than creating a second source of truth.

Add to `RunState`:

- `schedule_state: Literal["ongoing", "pending"] = "ongoing"`
- `has_usable_context_usage: bool = False`
- `pending_since_iso: str | None = None`
- `pending_since_monotonic: float | None = None`
- `ready_event: asyncio.Event | None = None`
- `active_request_count: int = 0`

Add a small gateway-global runtime structure:

- `ctx_aware_enabled: bool = False`
- `ctx_aware_usage_threshold_tokens: int | None = None`
- `ctx_aware_scheduling_threshold_tokens: int | None = None`
- `ctx_aware_ongoing_agent_count: int = 0`
- `ctx_aware_pending_agent_count: int = 0`
- `ctx_aware_ongoing_effective_context_tokens: int = 0`
- `ctx_aware_pending_effective_context_tokens: int = 0`
- `ctx_aware_scheduler_task: asyncio.Task | None = None`
- `ctx_aware_scheduler_wakeup: asyncio.Event | None = None`

Constants:

- `CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS = 3000`
- `CTX_AWARE_SCHEDULER_INTERVAL_S = 0.2`

## 6. Scheduler Rules

### 6.1 Effective context helper

Add a helper:

```python
def _effective_context_tokens(run: RunState) -> int:
    if run.has_usable_context_usage:
        return run.current_context_tokens
    return CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS
```

This helper is the only value used by the scheduler.

### 6.2 Rebalance algorithm

Implement one internal method that performs the full rebalance while holding the
gateway lock.

Algorithm:

1. Recompute ongoing and pending agent lists.
2. If mode is disabled:
   - mark every run `ongoing`
   - set every `ready_event`
   - clear pending timestamps
   - refresh cached counts/totals
   - return
3. While `ongoing_effective_context_tokens > usage_threshold_tokens`:
   - choose the youngest ongoing run
   - flip it to `pending`
   - set `pending_since_*` if it was not already pending
   - clear its `ready_event`
   - recompute totals
4. While there is at least one pending run:
   - choose the oldest pending run
   - let `candidate = effective_context_tokens(oldest_pending)`
   - if `ongoing_effective_context_tokens + candidate <= scheduling_threshold_tokens`:
     - flip that run to `ongoing`
     - clear `pending_since_*`
     - set its `ready_event`
     - recompute totals
   - otherwise stop promoting
5. Refresh cached counts/totals one final time.

This gives us the exact hysteresis described in the README:

- demotion uses the higher threshold
- promotion uses the lower threshold

### 6.3 Scheduler loop

Add one background coroutine started from FastAPI startup and cancelled on
shutdown.

Loop behavior:

- wait for either:
  - `ctx_aware_scheduler_wakeup` to be set
  - or `0.2 s` timeout
- when awakened, call rebalance under the service lock
- clear the wakeup event

Wake the scheduler immediately after:

- `POST /ctx-aware/start`
- `POST /ctx-aware/end`
- `POST /agent/start`
- `POST /agent/end`
- completion of any proxied request that updates context size
- disconnect or cancellation of a pending request

The periodic tick remains as a safety net and matches the requested 5 Hz
behavior.

Because `/ctx-aware/start` and `/ctx-aware/end` are only allowed with no active
job, their wakeups are effectively a no-op state refresh.

## 7. Request Path Changes

The current request path is implemented in `GatewayService.proxy_request`. That
method needs one structural change: request timing and span timing can no longer
share the same start timestamp.

### 7.1 On request arrival

At the start of `proxy_request`:

- capture `request_arrived_at`
- parse the request body as today
- under lock:
  - verify job is active
  - verify the agent exists
  - end `active_agent_action_span` as today
  - increment `active_request_count`
  - snapshot `root_context`
  - snapshot whether ctx-aware mode is enabled
  - snapshot the agent's current schedule state

### 7.2 Pending wait helper

If ctx-aware mode is enabled and the agent is `pending`, wait before creating
the `model_inference` span.

Add an async helper that waits on:

- the agent's `ready_event`
- or downstream disconnect

Behavior while waiting:

- do not forward anything upstream
- if the downstream client disconnects before promotion, return a `499`
  immediately
- after wakeup, re-check that the job is still active and the agent still
  exists

### 7.3 Forward timing

After the request is allowed to run:

- capture `forward_started_at`
- compute:
  - `pending_duration_ms = forward_started_at - request_arrived_at`
- start the `model_inference` span at `forward_started_at`
- forward to vLLM exactly as today

This preserves correct tracing semantics:

- `model_inference` span duration means gateway-to-vLLM forwarding time
- request duration means queue wait plus forward time

### 7.4 Response handling

When the response completes:

- capture `request_finished_at`
- compute:
  - `request_duration_ms = request_finished_at - request_arrived_at`
  - `span_duration_ms = request_finished_at - forward_started_at`
- keep `request_duration_ms` and `duration_ms` as end-to-end duration
- add `span_duration_ms` as the true forwarded-span duration
- set `span_start_time` from `forward_started_at`
- set `span_end_time` from `request_finished_at`
- update `current_context_tokens` and `has_usable_context_usage`
- append the request record
- decrement `active_request_count`
- wake the scheduler

If the client disconnects while the request is still waiting in pending state:

- do not call upstream
- emit the same `499 client_disconnected` style record we already use for
  forward-phase disconnects
- set `pending_duration_ms` from arrival until disconnect
- set `forward_start_time` to `null`
- decrement `active_request_count`

## 8. Logging Changes

The README asks for pending time in `model_inference.jsonl`. Make that explicit.

Add these fields to each request record:

- `forward_start_time`
- `pending_duration_ms`
- `span_duration_ms`

Interpretation:

- `request_start_time`: when gateway received the request
- `forward_start_time`: when gateway actually started forwarding to vLLM
- `request_end_time`: when gateway finished handling the request
- `request_duration_ms`: total wall-clock duration from arrival to completion
- `duration_ms`: keep as an alias of `request_duration_ms` for backward
  compatibility
- `pending_duration_ms`: time spent blocked because the agent was pending
- `span_start_time` / `span_end_time`: actual `model_inference` span timing
- `span_duration_ms`: actual forwarded `model_inference` span duration

No existing field should be renamed or removed.

## 9. Agent Lifecycle Changes

### 9.1 `POST /agent/start`

When a new run is created:

- initialize `ready_event`
- if ctx-aware mode is disabled:
  - set state to `ongoing`
  - set `ready_event`
- if ctx-aware mode is enabled:
  - if `ongoing_effective_context_tokens + 3000 <= scheduling_threshold_tokens`:
    - start as `ongoing`
    - set `ready_event`
  - otherwise:
    - start as `pending`
    - clear `ready_event`
    - set `pending_since_*`

After insertion, wake the scheduler so it can rebalance if needed.

### 9.2 `POST /agent/end`

Ctx-aware mode adds a new race: a request may be queued in gateway when the
controller tries to end the agent.

To make this deterministic, change `end_agent` to reject shutdown while the
agent still has queued or in-flight requests.

New rule:

- if `active_request_count > 0`, return `409`

This is stricter than current behavior, but it prevents silent record loss and
is especially important once pending requests can live inside gateway for a long
time.

After a successful end:

- remove the run
- wake the scheduler so older pending agents can be promoted

## 10. Code Touchpoints

### 10.1 `gateway/app.py`

Primary implementation file.

Changes:

- extend `RunState`
- add ctx-aware runtime state to `GatewayService`
- add helpers for effective context, ordering, rebalance, wakeup, and waiting
- modify `start_job`, `start_agent`, `end_agent`, `get_context_usage_summary`,
  and `proxy_request`
- add new HTTP routes for ctx-aware control/status
- add FastAPI startup/shutdown hooks for the scheduler task

`start_job` should clear all per-job cached counts and agent state but should
not implicitly disable ctx-aware mode if it was already enabled. A later job
should inherit the last ctx-aware configuration until `POST /ctx-aware/end`
clears it.

### 10.2 `gateway/test/test_gateway.py`

Add coverage for:

- enable endpoint validation
- `POST /ctx-aware/start` rejects while a job is active
- `POST /ctx-aware/end` rejects while a job is active
- ctx-aware configuration persists across `job/end` and the next `job/start`
  until explicitly ended
- new-agent admission using pseudo size
- demotion of youngest ongoing agent
- promotion of oldest pending agent
- response without usage keeping the pseudo-size behavior until first usable
  usage appears
- `pending_duration_ms` logging
- request timing versus span timing split
- disconnect while waiting in pending state returns `499` without upstream call
- `GET /ipc/context` additive fields
- `GET /ctx-aware` status payload
- `POST /agent/end` rejects while requests are still queued or in flight

### 10.3 Documentation

Update these docs after code lands:

- [`gateway/README.md`](/srv/scratch/yichaoy2/work/vllm-otel/gateway/README.md)
- [`docs/VLLM_GATEWAY.md`](/srv/scratch/yichaoy2/work/vllm-otel/docs/VLLM_GATEWAY.md)

Document:

- new control endpoints
- scheduler semantics
- new request-record fields
- additive IPC fields

## 11. Recommended Implementation Order

1. Extend `RunState` and gateway-global ctx-aware runtime state.
2. Add effective-context helpers and deterministic ordering helpers.
3. Add control/status endpoints without request gating first.
4. Add the rebalance engine and scheduler wakeup loop.
5. Wire `POST /agent/start` and `POST /agent/end` into the new state machine.
6. Change `proxy_request` to wait on pending state and record
   `pending_duration_ms`.
7. Extend IPC summaries.
8. Add tests for all edge cases.
9. Update README/docs after tests are green.

## 12. Non-Goals For This First Pass

To keep the change focused, v1 should not include:

- predictive context estimation from upcoming request payloads
- persistent ctx-aware settings in `gateway/config.toml`
- a separate admin server or authentication layer just for ctx-aware controls
- automatic recovery logic for permanently oversized agents
- fairness policies beyond:
  - youngest ongoing demotion
  - oldest pending promotion
