# Replayer

`replayer` supports three steps:

1. compile a profiled con-driver + gateway job into a replay plan
2. run replay from that plan
3. validate replay output against the source profile

Detailed reference for compile input requirements and replay timing drivers:

- `replayer/README.replay-plan-inputs-and-timing.md`
- `replayer/README.split-two-group-plans.md`

## CLI

Show commands:

```bash
python -m replayer --help
```

### All Options

`python -m replayer compile` options:

- `--config <path>`: TOML config path (`[compile]` or `[replayer.compile]`).
- `--job-dir <path>`: profiled con-driver job directory.
- `--job-root <path>`: recursively discover all profiled job dirs under this root and compile them in parallel with live batch progress (cannot be combined with `--job-dir` or `--plan-out`).
- `--plan-out <path>`: output replay plan path (default: `<job-dir>/replay-plan.json`; with `--exclude-unranked-trails`, default: `<job-dir>/replay-plan.exclude-unranked.json`).
- `--port-profile-id <int>`: required; resolve compile-time tokenizer endpoint from `configs/port_profiles.toml`.
- `--request-timeout-s <float>`: optional HTTP timeout for compile-time tokenizer requests (default: `3600`).
- `--model <string>`: optional compile-time model override. Must match a name in `configs/model_config.toml` (model key, served model name, or vLLM model name). When set, compile rewrites `replay_target.model` and each request body `model`.
- `--split-two-group-plans`: optional; write two split plans based on precomputed grouping from `<job-dir>/original-analysis/split/`.
- `--split-two-group-metric <token_usage|context_usage>`: grouping metric file for `--split-two-group-plans` (default: `token_usage`).
  - `token_usage` reads `top-p-token-usage-two-groups.json`
  - `context_usage` reads `top-p-context-usage-two-groups.json`
- `--exclude-unranked-trails`: non-split compile only; exclude trails listed under `original-analysis/split/top-p-usage-ratio-summary.json` `unranked_trails`.
- `--additional-suffix <string>`: optional; append a suffix before the final `.json` extension. For example, `replay-plan.json` with `--additional-suffix v2` becomes `replay-plan.v2.json`. Applied after other suffixes like `--exclude-unranked-trails` or `--split-two-group-plans`.
- Naming/reuse details: `replayer/README.split-two-group-plans.md`

`python -m replayer replay` options:

- `--config <path>`: TOML config path (`[replay]` or `[replayer.replay]`).
- `--plan <path>`: compiled replay plan path.
- `--output-dir <path>`: replay output directory (default: `<plan-dir>/<job>.replayed-<ts>`).
- `--num-tasks <int>`: replay exactly this many tasks (truncate or wrap launch order).
- `--port-profile-id <int>`: required; resolve replay target endpoints from `configs/port_profiles.toml`.
- `--launch-policy-override-json <json>`: launch policy overlay JSON.
- `--agent-timeout-s <float>`: optional per-worker timeout enforced during replay runtime.
- `--vllm-log-interval-s <float>`: optional metrics sampling interval.
- `--vllm-log-timeout-s <float>`: optional metrics scrape timeout.

`python -m replayer.validate` options:

- `--source-job-dir <path>`: source con-driver profiled job directory.
- `--replay-run-dir <path>`: replay run output directory.
- `--report-out <path>`: optional validation report JSON output path.

Both `compile` and `replay` support `--config <path>` (TOML). CLI flags take
precedence over config values.
Replay still requires `--port-profile-id` on the command line; it is not read
from plan metadata or config.

Supported layouts:

- top-level command sections: `[compile]` and `[replay]`
- nested under `[replayer]`: `[replayer.compile]` and `[replayer.replay]`

Example:

```toml
[compile]
job_dir = "tests/output/con-driver/job-20260225T035758Z"
plan_out = "tests/output/tmp-replay-plan.json"
port_profile_id = 1
request_timeout_s = 3600.0

[replay]
plan = "tests/output/tmp-replay-plan.json"
output_dir = "tests/output/tmp-replay-run"
num_tasks = 120
agent_timeout_s = 3000.0
vllm_log_interval_s = 1.0
vllm_log_timeout_s = 3600.0

[replay.launch_policy_override]
max_concurrent = 10
```

Compile:

```bash
python -m replayer compile \
  --job-dir tests/output/con-driver/job-20260225T035758Z \
  --port-profile-id 1 \
  --plan-out tests/output/tmp-replay-plan-20260225T035758Z-v2.json
```

Compile with model override:

```bash
python -m replayer compile \
  --job-dir tests/output/con-driver/job-20260225T035758Z \
  --port-profile-id 1 \
  --model qwen3_coder_30b_fp8
```

Compile every profiled job under a root:

```bash
python -m replayer compile \
  --job-root tests/output/con-driver \
  --port-profile-id 1
```

Compile while excluding split-unranked trails (non-split mode):

```bash
python -m replayer compile \
  --job-dir tests/output/con-driver/job-20260225T035758Z \
  --port-profile-id 1 \
  --exclude-unranked-trails
```

Default output for that command is:
`<job-dir>/replay-plan.exclude-unranked.json`.

`replayer compile` shows a live progress bar while it builds worker plans and
deterministic request payloads. The bar advances by recorded request and also
shows completed workers.

Compile supports both gateway output layouts:

- single profile: `gateway-output/run_*/...`
- cluster mode: `gateway-output/profile-*/run_*/...`

Compile resolves tokenizer endpoint from the selected `--port-profile-id` and
uses it to build deterministic `forced_token_ids`.
`--request-timeout-s` controls tokenizer HTTP timeout during this compile step.
Compile backend is auto-detected from `meta/config.toml`; there is no override.

### Compile Versioning

Replay plans now include `compile_version`. During `replayer compile`, if the
target plan file already exists and its `compile_version` matches the current
code version, compile reuses that plan and skips recompilation.

For backward compatibility, missing or empty `compile_version` is treated as
`v1`.

Maintainer note: if you change compile-time plan semantics in
`replayer/cli.py`, you must increment `REPLAY_PLAN_COMPILE_VERSION`.

If the source profile contains recorded `499 client_disconnected` inference
requests, `compile` now keeps those requests in the plan instead of failing
deterministic tokenization. Those entries are compiled as synthetic
client-cancel requests: replay sends the original request with `ignore_eos =
true`, removes explicit completion-token caps, and then cancels the client
connection after the recorded request duration. If the gateway returns an error
after the request has already been running for at least `60s`, replay treats
that as acceptable equivalent behavior for this mode instead of failing the
worker.

If you want to drop split-unranked trails from a non-split plan, use
`--exclude-unranked-trails`.

Replay:

```bash
python -m replayer replay \
  --plan tests/output/tmp-replay-plan-20260225T035758Z-v2.json \
  --port-profile-id 1 \
  --agent-timeout-s 3000 \
  --output-dir tests/output/tmp-replay-run-20260225T035758Z-v2
```

`replayer replay` now shows a live progress bar with total workers, launched
workers, active workers, and failed workers.

If a worker hits `agent_timeout_s`, replay marks that worker `timed_out` and
tracks it separately from true replay failures. Timed-out workers do not cause
`replayer replay` itself to return a non-zero exit code.

Replay always routes requests through gateway and always writes
`<replay-output>/gateway-output/` artifacts.

Replay always resolves localhost endpoints from the selected port profile.
Older plans may still carry legacy URL fields under `replay_target`; replay
ignores them for routing.

If you pass `--agent-timeout-s`, replay enforces that wall-clock limit per
worker and marks timed-out workers as `timed_out`.

`replayer replay` always runs the same vLLM metrics monitor used by
`con-driver`. The current replay defaults are:

- sampling interval: `1.0s`
- scrape timeout: `3600.0s`

Replay uses the same defaults. Because `--port-profile-id` is required, replay
resolves the metrics endpoint from the selected profile's `vllm_port` and vLLM
logging defaults to enabled. There is no manual replay-side endpoint override.
Replay always enables vLLM logging; disabling it is not supported.

Replay vLLM metrics are written under:

- `<replay-output>/vllm-log/`
- `<replay-output>/vllm-log/monitor.stdout.log`
- `<replay-output>/vllm-log/monitor.stderr.log`

Each compressed block stores the raw `/metrics` response text per scrape, not a
parsed Prometheus JSON structure.

Replay also attempts LMCache metrics logging from the selected profile's
`lmcache_port` (same sampler and timeout settings as vLLM logging). LMCache
logging is started only when a first `/metrics` probe succeeds; otherwise replay
continues without LMCache logs.

When enabled, LMCache metrics are written under:

- `<replay-output>/lmcache-log/`
- `<replay-output>/lmcache-log/monitor.stdout.log`
- `<replay-output>/lmcache-log/monitor.stderr.log`

Replay can also override the compiled `launch_policy` while preserving the
compiled worker/request structure. Pass a JSON object with the fields you want
to overlay onto the plan's `launch_policy`.

If the override switches the launch pattern to `poisson` and does not also set
`max_concurrent`, replay treats the launch stream as unbounded instead of
inheriting the plan's recorded concurrency cap.

For example, to replay the same workers with `max_concurrent = 10` instead of
the value stored in the plan:

```bash
python -m replayer replay \
  --plan tests/output/tmp-replay-plan-20260225T035758Z-v2.json \
  --port-profile-id 1 \
  --launch-policy-override-json '{"max_concurrent": 10}'
```

This JSON may either be:

- the `launch_policy` object itself
- or a wrapper object with a top-level `launch_policy` field

Example shape:

```json
{
  "max_concurrent": 10,
  "seed": null,
  "pattern": {
    "name": "eager"
  },
  "pattern_args": {}
}
```

Replay can also control how many tasks to launch via `--num-tasks`:

- if `--num-tasks` is smaller than the plan's worker count, replay runs the
  first tasks in replay launch order
- if `--num-tasks` is larger, replay wraps from the beginning of launch order
  until it reaches the requested task count

Example:

```bash
python -m replayer replay \
  --plan tests/output/tmp-replay-plan-20260225T035758Z-v2.json \
  --port-profile-id 1 \
  --num-tasks 120
```

`replay` resolves gateway/vLLM endpoints from `configs/port_profiles.toml` and
requires `--port-profile-id`:

```bash
python -m replayer replay \
  --plan tests/output/tmp-replay-plan-20260225T035758Z-v2.json \
  --port-profile-id 1
```

Launch behavior:

- replay preserves original worker launch ordering (`launch_priority`)
- worker launch timing is driven by recorded con-driver scheduling config (`max_concurrent`, `pattern`, `pattern_args`, `seed`)
- a replay-side Poisson override does not inherit the plan's `max_concurrent` unless the override explicitly sets its own `max_concurrent`
- replay does not require exact original absolute launch offsets
- replay requires plans that include `launch_policy` (`config_ordered`)
- replay targets the raw gateway listener, not `gateway_parse_port`
- replay replays recorded `client_disconnected` requests as timed client-side cancellations rather than exact response-text comparisons

## Validation Script

Run validation:

```bash
python -m replayer.validate \
  --source-job-dir tests/output/con-driver/job-20260225T035758Z \
  --replay-run-dir tests/output/tmp-replay-run-20260225T035758Z-v2 \
  --report-out tests/output/tmp-replay-run-20260225T035758Z-v2/replay/validation-report.json
```

Validation behavior:

- exact checks (pass/fail): worker mapping, request counts, prompt content equality, response content equality, usage fields (`prompt_tokens`, `completion_tokens`, `cached_tokens` when present)
- similarity checks (diagnostic): job duration, per-agent duration, per-request duration relative error

Note:

- `usage.prompt_tokens_details.cached_tokens` is sensitive to vLLM KV-cache state. If source and replay run with different cache warmness/history, exact validation can fail only on `cached_tokens` while prompt/response content still matches.
- For strict exact-match validation, run replay from a clean vLLM cache state (for example, restart vLLM before replay).

Exit codes:

- `0`: exact checks passed
- `2`: exact checks failed
- `1`: runtime/argument error
