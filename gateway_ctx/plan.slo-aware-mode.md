# SLO-Aware Mode Implementation Plan

## 1. Goal

Implement an SLO-aware extension on top of ctx-aware scheduling in
`gateway_ctx`.

The purpose of SLO-aware mode is to temporarily hold back agents that are
comfortably above a configured decode-throughput target so that agents below the
target have a better chance to recover.

This mode remains agent-level, not request-level:

- every active agent is in one of three admission states:
  - `ongoing`
  - `pending`
  - `ralexation`
- `pending` is the existing ctx-aware blocked state
- `ralexation` is a new SLO-aware blocked state
- both `pending` and `ralexation` block new requests from being forwarded
- neither mode cancels an already forwarded upstream request

SLO-aware mode is not standalone:

- ctx-aware mode must already be enabled before SLO-aware mode can be enabled
- SLO-aware should reuse the same scheduler control path and runtime lock as
  ctx-aware

## 2. Clarified Semantics

The current design note in `SLO_AWARE.md` is close, but implementation needs a
few rules written down explicitly.

### 2.1 Throughput source

SLO-aware mode reuses the throughput already stored on each agent:

- `current_output_tokens_per_s`

That metric is already computed from:

- `total_completion_tokens`
- `total_llm_request_duration_s`

SLO-aware mode should not invent a second instantaneous throughput metric.

### 2.2 SLO slack

For an agent with:

- stored throughput `y`
- accumulated request duration `t`
- target throughput `x`

the SLO slack is:

```text
slack_s = t' - t = (y * t) / x - t = t * (y / x - 1)
```

Equivalent implementation form:

```text
slack_s = total_completion_tokens / target_tokens_per_s - total_llm_request_duration_s
```

Interpretation:

- positive slack means the agent has spare delay budget
- zero or negative slack means there is no delay budget to spend

### 2.3 Which agents participate in min/avg checks

The design note says "alive" or "active" agents. For v1 we should define that
precisely:

- start from all runs in `active_runs`
- only include runs whose `current_output_tokens_per_s` is not `None`

Agents without stored throughput yet:

- do not participate in the global minimum-throughput check
- do not participate in the average-throughput check
- cannot be selected for `ralexation`

### 2.4 Policy trigger

For v1 we support one policy:

- `push-back-half-slack`

The policy is globally active only when:

- SLO-aware mode is enabled
- there is at least one active agent with stored throughput
- the minimum stored throughput among those agents is below the configured SLO
  target

### 2.5 Entering `ralexation`

Evaluate `ralexation` only when an `ongoing` agent completes a request and its
stored throughput has just been refreshed.

That agent may enter `ralexation` only if all of the following are true:

- it is still `ongoing` after the normal ctx-aware rebalance step
- its stored throughput is greater than the average stored throughput of all
  throughput-bearing active agents
- its stored throughput is greater than the configured SLO target
- its computed slack is positive

Duration rule:

- `ralexation_duration_s = slo_slack_s / 2`

If the duration is `<= 0`, do not place the agent into `ralexation`.

### 2.6 Leaving `ralexation`

An agent becomes ready to leave `ralexation` when either condition is true:

- its `ralexation_duration_s` has expired
- the minimum stored throughput among throughput-bearing active agents is now at
  or above the SLO target

When a `ralexation` wake-up is processed:

- if there is already any normal `pending` agent, convert the waking agent to
  normal `pending`
- otherwise, treat it as if it were a new incoming agent under ctx-aware
  admission:
  - if it fits under the ctx-aware scheduling threshold, make it `ongoing`
  - otherwise, make it `pending`

If a `ralexation` agent becomes normal `pending`, it should join the pending
queue as newly pending:

- set `pending_since_*` at the transition time
- do not preserve an older `ralexation` timestamp as pending age

### 2.7 Priority between ctx-aware and SLO-aware blocking

Normal ctx-aware `pending` takes priority over `ralexation`.

Implementation rule:

- if normal ctx-aware logic would make a run `pending`, do that first
- do not place a run into `ralexation` when it should already be `pending`

This keeps ctx-aware as the primary admission control and SLO-aware as a
secondary fairness/efficiency layer on top.

## 3. Explicit Assumptions

To keep the first implementation predictable, we should lock in these
assumptions:

- SLO-aware configuration is gateway-global runtime state, not per-job config in
  `config.toml`
- like ctx-aware control, SLO-aware control changes are only allowed while no
  job is active
- `POST /slo-aware/start` and `POST /slo-aware/end` should return `409` if
  `job_active` is true
- `POST /slo-aware/start` should return `409` if ctx-aware mode is not enabled
- SLO-aware settings persist across `POST /job/end` and later `POST /job/start`
  calls until explicitly cleared or until ctx-aware mode is turned off
- turning off ctx-aware mode should also clear SLO-aware configuration, because
  SLO-aware is invalid without ctx-aware
- we keep the current meaning of throughput from the existing gateway code:
  completed output tokens divided by accumulated LLM request duration
- v1 keeps the only supported SLO policy as `push-back-half-slack`
- v1 does not attempt predictive throughput modeling from request payloads

## 4. External API Changes

### 4.1 New control endpoints

Add runtime control endpoints on the main HTTP app:

- `GET /slo-aware`
- `POST /slo-aware/start`
- `POST /slo-aware/end`

`POST /slo-aware/start` request body:

```json
{
  "target_tokens_per_s": 25.0,
  "policy_mode": "push-back-half-slack"
}
```

Validation rules:

- `target_tokens_per_s` is required
- it must be a positive number
- `policy_mode` is required
- for v1, `policy_mode` must equal `push-back-half-slack`
- ctx-aware mode must already be enabled
- no job may be active

Enable behavior:

- set SLO-aware enabled
- store `target_tokens_per_s`
- store `policy_mode`
- refresh cached status fields
- return the same payload as `GET /slo-aware`

`POST /slo-aware/end` behavior:

- reject with `409` if a job is active
- disable SLO-aware mode
- clear target and policy fields back to `null`
- clear any cached `ralexation` status because there should be no active agents
  while this endpoint is allowed
- return the same payload as `GET /slo-aware`

### 4.2 `GET /slo-aware` response

Return a status/control view similar to `GET /ctx-aware`:

```json
{
  "status": "ok",
  "enabled": true,
  "requires_ctx_aware": true,
  "target_tokens_per_s": 25.0,
  "policy_mode": "push-back-half-slack",
  "ralexation_agent_count": 3,
  "agents": [
    {
      "api_token_hash": "...",
      "trace_id": "...",
      "schedule_state": "ralexation",
      "output_tokens_per_s": 41.2,
      "slo_slack_s": 3.8,
      "ralexation_until": "2026-04-08T17:10:15.000Z"
    }
  ]
}
```

### 4.3 Additive status changes

Add SLO-aware fields to `GET /ipc/context` and, if convenient, `GET /ctx-aware`:

- `slo_aware_enabled`
- `slo_target_tokens_per_s`
- `slo_policy_mode`
- `ralexation_agent_count`
- `ralexation_effective_context_tokens`

Add these per-agent fields:

- `output_tokens_per_s`
- `schedule_state`
- `slo_slack_s`
- `ralexation_until`

This makes the new admission state observable without removing existing fields.

## 5. Data Model Changes

Extend `RunState` in `gateway_ctx/app.py` rather than creating a second source
of truth.

Add to `RunState`:

- `schedule_state: Literal["ongoing", "pending", "ralexation"] = "ongoing"`
- `ralexation_since_iso: str | None = None`
- `ralexation_until_iso: str | None = None`
- `ralexation_until_monotonic: float | None = None`

Keep existing throughput fields as the source for SLO evaluation:

- `total_completion_tokens`
- `total_llm_request_duration_s`
- `current_output_tokens_per_s`

Add gateway-global runtime state:

- `slo_aware_enabled: bool = False`
- `slo_target_tokens_per_s: float | None = None`
- `slo_policy_mode: Literal["push-back-half-slack"] | None = None`
- `slo_ralexation_agent_count: int = 0`
- `slo_ralexation_effective_context_tokens: int = 0`

Reuse the existing ctx-aware scheduler task and wakeup event instead of creating
a completely separate scheduler.

## 6. Helper Functions And Algorithms

### 6.1 Throughput-bearing active runs

Add a helper that returns all active runs with stored throughput:

```python
def _runs_with_stored_throughput_locked(self) -> list[RunState]:
    ...
```

Add helpers:

- `_min_stored_throughput_locked() -> float | None`
- `_avg_stored_throughput_locked() -> float | None`

### 6.2 Slack helper

Add:

```python
def _slo_slack_s_locked(self, run: RunState) -> float | None:
    ...
```

Behavior:

- return `None` if SLO-aware is disabled
- return `None` if target is missing
- return `None` if the run has no stored throughput
- return `None` if accumulated request duration is `<= 0`
- otherwise compute the slack from stored totals

### 6.3 State-marking helpers

Extend existing helpers:

- `_mark_run_ongoing_locked`
- `_mark_run_pending_locked`

Add:

- `_mark_run_ralexation_locked(run, *, duration_s: float)`

State semantics:

- `ongoing`:
  - `ready_event` is set
  - pending timestamps are cleared
  - `ralexation_*` timestamps are cleared
- `pending`:
  - `ready_event` is cleared
  - `pending_since_*` is set if entering from a non-pending state
  - `ralexation_*` timestamps are cleared
- `ralexation`:
  - `ready_event` is cleared
  - pending timestamps are cleared
  - `ralexation_until_*` is set

### 6.4 Candidate-selection helpers

Add:

- `_should_enter_ralexation_locked(run: RunState) -> bool`
- `_ready_ralexation_runs_locked(now_monotonic: float) -> list[RunState]`

`_ready_ralexation_runs_locked` should return ready runs in deterministic order:

- earliest `ralexation_until_monotonic` first
- then `run_start_time`
- then `api_token_hash`
- then `trace_id`

### 6.5 Combined rebalance algorithm

Extend the existing rebalance logic under the gateway lock.

Recommended order:

1. Refresh ongoing, pending, and `ralexation` totals.
2. If ctx-aware mode is disabled:
   - mark every run `ongoing`
   - clear `ralexation` timestamps
   - refresh totals
   - return
3. Run the existing ctx-aware demotion loop:
   - while ongoing effective context exceeds `usage_threshold_tokens`
   - demote normal ongoing agents using the current policy
4. Process ready `ralexation` runs:
   - if there is already any normal pending run, move the ready run to `pending`
   - otherwise, treat it as a new incoming agent under ctx-aware scheduling
     admission
5. Run the existing ctx-aware promotion loop over normal `pending` agents only.
6. Refresh totals one final time.

Key rule:

- normal promotion does not directly promote a `ralexation` agent
- a `ralexation` agent must first become ready, then either become `pending` or
  be re-admitted as `ongoing`

### 6.6 Request-completion policy hook

After a proxied request completes and the run's throughput fields are updated:

1. decrement active-request counters as today
2. run normal ctx-aware rebalance
3. if the run is still `ongoing` and `_should_enter_ralexation_locked(run)` is
   true:
   - compute `duration_s = slo_slack_s / 2`
   - mark the run `ralexation`
4. wake the scheduler

Doing ctx-aware rebalance first is what enforces "pending takes priority over
ralexation".

## 7. Request Path Changes

The request path already knows how to wait on `pending`. Extend that logic so
`ralexation` is treated as another blocked admission state.

On request arrival:

- if the agent is `ongoing`, forward as today
- if the agent is `pending` or `ralexation`, wait on its `ready_event`

While waiting:

- do not forward anything upstream
- if the downstream client disconnects first, return `499`
- after wakeup, re-check that the job is still active and the agent still exists

No change is needed to the meaning of forwarded span timing:

- waiting in `pending` or `ralexation` is gateway queue time
- forwarded request duration remains the actual upstream-forward time

## 8. Logging And Observability

Extend the current observability enough to debug SLO decisions.

### 8.1 Per-agent status

Expose on status endpoints:

- current `schedule_state`
- current stored throughput
- computed slack
- `ralexation_until`

### 8.2 Sampler/job logs

Extend the existing ctx-aware job sampler or add a separate SLO-aware job log
with fields such as:

- `ongoing_agent_count`
- `pending_agent_count`
- `ralexation_agent_count`
- `min_output_tokens_per_s`
- `avg_output_tokens_per_s`
- `agents_turned_ralexation`
- `agents_left_ralexation_to_pending`
- `agents_left_ralexation_to_ongoing`

### 8.3 Request records

No new request-record timing fields are required specifically for SLO-aware
mode, because ctx-aware already records queue wait. If useful, a small additive
field may be added later to distinguish whether wait time came from `pending`,
`ralexation`, or both.

## 9. Code Touchpoints

### 9.1 `gateway_ctx/app.py`

Primary implementation file.

Changes:

- extend `RunState`
- add SLO-aware runtime state to `GatewayService`
- add helpers for throughput-bearing runs, slack, `ralexation`, and wake-up
  decisions
- add `/slo-aware` HTTP routes
- extend the ctx-aware rebalance loop to handle `ralexation`
- extend request waiting logic to block on `ralexation`
- update status and IPC payload builders

### 9.2 `gateway_ctx/test/test_gateway.py`

Add tests for:

- `/slo-aware/start` validation
- `/slo-aware/start` rejects when ctx-aware is disabled
- `/slo-aware/start` rejects while a job is active
- `/slo-aware/end` rejects while a job is active
- SLO-aware config persists across jobs while ctx-aware remains enabled
- disabling ctx-aware also clears SLO-aware config
- agents with no stored throughput are excluded from min/avg checks
- `push-back-half-slack` triggers only when global minimum throughput is below
  target
- a run enters `ralexation` only when above both average throughput and target
- non-positive slack does not create `ralexation`
- ctx-aware `pending` wins over `ralexation`
- ready `ralexation` moves to `pending` if any normal pending agent already
  exists
- otherwise ready `ralexation` is re-admitted like a new incoming agent
- early release when global minimum throughput rises back above target
- requests blocked in `ralexation` wait correctly and return `499` on disconnect
- status and IPC payloads expose new SLO-aware fields

### 9.3 Documentation

Update these docs after code lands:

- `gateway_ctx/README.md`
- `gateway_ctx/SLO_AWARE.md`

Document:

- control endpoints
- activation dependency on ctx-aware
- the exact throughput source
- slack formula
- `ralexation` semantics
- wake-up and re-admission behavior
- observability fields

## 10. Recommended Implementation Order

1. Add SLO-aware runtime config and endpoint schemas.
2. Extend `RunState` with `ralexation` fields.
3. Add helpers for stored-throughput min/avg and slack calculation.
4. Add `/slo-aware` control/status endpoints.
5. Extend rebalance to track `ralexation` state and ready-to-leave logic.
6. Hook request-completion into `push-back-half-slack`.
7. Extend request waiting to treat `ralexation` like a blocked state.
8. Add observability/status fields.
9. Add tests for trigger, wake-up, and priority edge cases.
10. Update README/design docs after tests are green.

## 11. Non-Goals For This First Pass

To keep v1 focused, do not include:

- multiple SLO policies beyond `push-back-half-slack`
- predictive throughput models
- per-model or per-agent SLO targets
- persistent SLO-aware configuration in `config.toml`
- a separate scheduler task just for SLO-aware mode
- automatic tuning of the SLO target
