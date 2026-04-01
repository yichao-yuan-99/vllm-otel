# Replay Plan Compile Inputs and Replay Timing Drivers

This document explains two things in implementation detail:

1. what information is required to compile a replay plan from a `con-driver` run result
2. what information affects replay timing (launch timing + per-worker/request timing)

Terminology note: if you say "reply plan", in this repository that is `replay-plan.json`.

## Scope

This is based on current code paths in:

- `replayer/cli.py` (`cmd_compile`, `build_planned_request`, `cmd_replay`)
- `con-driver/src/con_driver/scheduler.py` (producer of `meta/*`)
- `gateway/app.py` (producer of `gateway-output/run_*/...` artifacts)

## Compile Input Contract (Hard Requirements)

`replayer compile` requires a full profiled job directory (`--job-dir`) with all of these paths:

- `meta/config.toml`
- `meta/run_manifest.json`
- `meta/results.json`
- `meta/events.jsonl`
- `gateway-output/run_*/...` and/or `gateway-output/profile-*/run_*/...`

For every discovered `run_*` directory under `gateway-output`, these are required:

- `manifest.json`
- `events/lifecycle.jsonl`
- `requests/model_inference.jsonl`
- `trace/jaeger_trace.json`

If any required file is missing, compile fails immediately.

## Compile Short-Circuit Behavior

Before reading source artifacts, compile checks the target output path:

- default: `<job-dir>/replay-plan.json`
- with `--exclude-unranked-trails`: `<job-dir>/replay-plan.exclude-unranked.json`
- with `--single-trail <trail>`: `<job-dir>/replay-plan.trail-<safe-trail>.json`

If that file already exists, compile reuses it only when all of these match:

- `compile_version` is current (`"1"`) or missing/empty (treated as v1)
- requested compile options that affect plan contents still match the file
  (`--model`, `--clean`, `--exclude-unranked-trails`, `--single-trail`)

If you expect fresh extraction from run artifacts, delete the existing plan or force a path with no existing plan file.

## Field-Level Dependency Matrix

### 1) `meta/config.toml`

Required fields:

- backend detection:
  - primary: `[backend].name`
  - fallback: `[run].driver_backend`
- supported value today: `harbor` only (others fail as unsupported)

Model extraction for tokenization target:

- primary: `[backend].forwarded_args` containing `--model <value>`
- fallback: parse `--model` from each `meta/results.json[].command`

Launch policy extraction:

- `[run].pattern` (default `"eager"` if missing)
- `[run].pattern_args` (token list parsed into key/value map)
- `[run].max_concurrent` (default `1`, clamped to at least `1`)
- `[run].seed` (optional int)

Supported pattern names:

- `"eager"`
- `"poisson"` (and typo alias `"possion"`)
- `"uniform"`

For Poisson and Uniform, rate must be derivable from `pattern_args` via one of:

- `rate`, `arrival-rate`, `arrival_rate`, `lambda`
- or inverse of `mean-interval-s`, `mean_interval_s`, `interval-s`, `interval_s`

### 2) `meta/run_manifest.json`

Used for compile T0 anchor:

- primary field: `started_at` (ISO-8601 string)

If missing/invalid, compile falls back to `meta/events.jsonl` (`gateway_job_start` event time).

### 3) `meta/events.jsonl`

Used only as T0 fallback:

- record with `event == "gateway_job_start"`
- field `time` as ISO-8601 string

If both `run_manifest.started_at` and fallback event time are unavailable, compile fails.

### 4) `meta/results.json`

Must be a JSON list.

Used for trial-to-gateway artifact mapping:

- each relevant entry needs:
  - `trial_id` (string)
  - `command` (list)
  - `--api-token <token>` inside `command`
- compiler computes `sha256(api_token)` and maps to gateway `manifest.api_token_hash`

Also used as model extraction fallback (`--model`) if config did not provide model.

### 5) `gateway-output/.../manifest.json` (per run)

Required fields:

- `api_token_hash` (string): joins run to `meta/results.json` trial via hashed token
- `run_start_time` (ISO-8601): used for `run_offset_s` and launch ordering

Optional/non-fatal:

- `request_count` (int): used for compile progress total; mismatches are tolerated
- `trace_id`: copied into compiled worker payload when present

### 6) `gateway-output/.../events/lifecycle.jsonl` (per run)

Required event timestamps:

- first `agent_start` timestamp
- last `agent_end` timestamp

These drive worker-level timing deltas.

### 7) `gateway-output/.../requests/model_inference.jsonl` (per run)

Request records are sorted by request start timestamp before compilation.

Required for every request record:

- `request` must be a JSON object (source request body)
- start/end timestamps:
  - `request_start_time` or `span_start_time`
  - `request_end_time` or `span_end_time`

Request mode detection:

- client-disconnect mode if:
  - `response.error == "client_disconnected"`
  - and `status_code` is `499` or missing
- otherwise deterministic forced-token mode

Deterministic mode additionally requires response text extraction from one of:

- response string
- `response.choices[0].message.content` (string)
- `response.choices[0].text` (string)
- `response.output_text` (string)

If no text can be extracted, compile fails.

### 8) `gateway-output/.../trace/jaeger_trace.json` (per run)

File must exist and parse as a JSON object.

Optional timing enrichment:

- spans with `operationName == "agent_action"`
- each needs integer `startTime` + `duration` (microseconds)

If absent/unusable, compiler falls back to timing inferred from request windows.

## How Compile Builds the Plan

### 1) Worker mapping and order

For each gateway run artifact:

1. map `manifest.api_token_hash` -> trial via hashed `--api-token` from `meta/results.json`
2. parse `manifest.run_start_time`
3. compute:
   - `run_offset_s = run_start_time - t0` (must be non-negative)

After all workers are built:

- sort workers by `(run_offset_s, worker_id)`
- assign `launch_priority` sequentially (`0..N-1`)

So launch order comes from observed gateway run start times, not directly from `launch_index` in con-driver events.

### 2) Worker timing deltas

Per worker:

- `delta_agent_start_s = agent_start - run_start_time`
- `delta_first_request_s = first_request_start - agent_start` (or `0.0` if no requests)

All deltas are clamped to non-negative and rounded to 6 decimals.

### 3) Inter-request / post-request delay (`delta_agent_action_after_s`)

For request index `i`:

- if `i < last_request_index`:
  - use Jaeger `agent_action` duration at same index when available
  - else fallback to `(next_request_start - current_request_end)`
- if `i == last_request_index`:
  - use `final_agent_tail_s = agent_end - last_request_end`

If worker has no requests:

- final tail is computed (`agent_end - agent_start`) but no request entries are emitted.

### 4) Request payload compilation

Each planned request includes:

- `method`, `path`, `body`
- `delta_agent_action_after_s`
- replay mode metadata

Deterministic mode:

1. extract `response_text` from recorded response
2. resolve `model_for_tokenize`:
   - `record.model` -> `request.body.model` -> configured model
3. call tokenize endpoint:
   - `POST /tokenize` with `{"model": ..., "prompt": response_text, "add_special_tokens": false}`
4. inject deterministic forcing:
   - `body.vllm_xargs.forced_token_ids`
   - `body.vllm_xargs.force_eos_after_sequence = true`
5. ensure `body.max_tokens >= len(forced_token_ids) + 1`

Client-disconnect mode:

- no tokenization
- body rewritten to allow long generation until client cancel:
  - removes `stop`, `stop_token_ids`, `max_tokens`, `max_completion_tokens`
  - removes forcing keys under `vllm_xargs`
  - sets `ignore_eos = true`
- `cancel_after_s` from recorded duration:
  - `request_duration_ms` or `duration_ms`
  - fallback: `(request_end_time - request_start_time)`

## What Directly Impacts Replay Timing

Replay timing is controlled at three layers: launch stream, worker timeline, and request runtime/deadlines.

### A) Launch stream timing

From plan `launch_policy` (plus optional replay override):

- `strategy` must be `config_ordered`
- `pattern.name`:
  - `eager`: next launch delay is always `0`
  - `poisson`: delay sampled from exponential distribution (`expovariate(rate_per_second)`)
  - `uniform`: next launch delay is always exactly `1 / rate_per_second`
- `seed`:
  - int seed => deterministic Poisson samples
  - missing seed => non-deterministic Poisson stream each replay run
  - `uniform` does not use randomness, so seed has no effect
- `max_concurrent`:
  - caps concurrently active workers
  - if Poisson/Uniform override is provided without explicit `max_concurrent`, replay removes cap (unbounded launch stream)

Replay launch loop is capacity-gated:

- no new launch while `active >= max_concurrent` (if cap exists)
- launch cadence therefore depends on worker completion rate when saturated

### B) Worker-local timeline

Per worker, replay applies:

1. sleep `delta_agent_start_s`
2. call gateway `/agent/start`
3. sleep `delta_first_request_s`
4. run requests sequentially
5. after each request, sleep `delta_agent_action_after_s`
6. call gateway `/agent/end`

So recorded deltas shape timing, but real wall-clock also includes gateway lifecycle call overhead and request runtime.

### C) Request runtime gating (major timing factor)

Replay is response-gated:

- request `i+1` never starts until request `i` is finished/cancelled and post-request delay is applied
- backend latency therefore directly shifts downstream timing

Mode-specific behavior:

- deterministic requests:
  - without deadlines: wait indefinitely for HTTP response (`timeout=None`)
  - with deadlines: request is run through timed-cancel path and can be cut early
- client-disconnect requests:
  - replay actively cancels client connection at `cancel_after_s` (or earlier if a tighter deadline applies)

### D) Deadline controls

- `--agent-timeout-s`:
  - per-worker wall-clock cap after agent starts
  - can terminate worker mid-request or between delays
- `--time-constraint-s`:
  - global replay wall-clock cap
  - replay switches to unbounded launch mode (cannot combine with `--num-tasks`)
  - launches repeat worker templates until deadline
  - workers cut by this deadline become `time_bound_finished`

When both per-worker and global limits exist, the earlier deadline wins for each operation.

### E) Task-shaping options that alter timing/load profile

- `--num-tasks`:
  - truncates workers if smaller than plan count
  - wraps/repeats workers if larger
- `--randomize-seed`:
  - shuffles worker order before scheduling
  - changes which worker timelines overlap, changing aggregate timing/load
- `--launch-policy-override-json`:
  - can alter pattern, rate, seed, concurrency at replay time

### F) Environment and endpoint selection

Replay always resolves target endpoints from `--port-profile-id` (not from plan URL fields):

- gateway URL / API base from selected port profile
- runtime characteristics of selected profile (server load, hardware, network path, model behavior) affect request latency and therefore overall replay timing

## What Does Not Directly Control Replay Timing

These are important metadata but not direct timing controls in replay:

- `t0` and `t0_source` (compile anchors only)
- `run_offset_s` (used for compile-time sorting/metadata; replay uses `launch_priority` order)
- compile-time tokenizer timeout (`--request-timeout-s`) once plan already exists
- `replay_target.model` itself as a timing knob (it affects correctness/tokenization assumptions; runtime latency impact is indirect via endpoint behavior)

## Practical Checklist

If compile fails:

1. check required files exist in `meta/` and each gateway `run_*`
2. check backend resolves to `harbor`
3. check `results.json` commands include `--api-token`
4. check gateway `manifest.api_token_hash` values match hashed tokens
5. check each request record has parseable start/end timestamps
6. check deterministic responses expose extractable text
7. check tokenize endpoint for selected `--port-profile-id` is reachable

If replay timing looks wrong:

1. inspect `launch_policy` and any replay override
2. verify whether Poisson is seeded or unseeded (`uniform` ignores seed)
3. check if `max_concurrent` was dropped by a Poisson/Uniform override
4. check whether `--randomize-seed`, `--num-tasks`, or `--time-constraint-s` changed worker stream shape
5. inspect request latencies and deadline-triggered cancellations/timeouts in replay worker logs
