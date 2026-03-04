# Replayer

`replayer` supports three steps:

1. compile a profiled con-driver + gateway job into a replay plan
2. run replay from that plan
3. validate replay output against the source profile

## CLI

Show commands:

```bash
python -m replayer --help
```

Compile:

```bash
python -m replayer compile \
  --job-dir tests/output/con-driver/job-20260225T035758Z \
  --agent-timeout-s 3000 \
  --plan-out tests/output/tmp-replay-plan-20260225T035758Z-v2.json
```

`replayer compile` shows a live progress bar while it builds worker plans and
deterministic request payloads. The bar advances by recorded request and also
shows completed workers.

`--agent-timeout-s` is optional. If you pass it, `compile` writes that value
into the plan as `agent_timeout_s`, and `replay` enforces it as a per-worker
runtime limit. If a replayed worker exceeds that duration, `replayer` cancels
the in-flight request and marks the worker `timed_out`. If the field is absent,
`replay` does not apply an agent-level deadline.

If the source `meta/config.toml` was generated from a `port_profile_id`, `compile`
will reuse that convention automatically and resolve:

- `replay_target.gateway_url` using the raw `gateway_port`
- `replay_target.api_base` using the raw gateway listener
- `replay_target.tokenize_endpoint`

If the source profile contains recorded `499 client_disconnected` inference
requests, `compile` now keeps those requests in the plan instead of failing
deterministic tokenization. Those entries are compiled as synthetic
client-cancel requests: replay sends the original request with `ignore_eos =
true`, removes explicit completion-token caps, and then cancels the client
connection after the recorded request duration. If the gateway returns an error
after the request has already been running for at least `60s`, replay treats
that as acceptable equivalent behavior for this mode instead of failing the
worker.

You can also override it explicitly:

```bash
python -m replayer compile \
  --job-dir tests/output/con-driver/job-20260225T035758Z \
  --port-profile-id 1 \
  --agent-timeout-s 3000
```

Replay:

```bash
python -m replayer replay \
  --plan tests/output/tmp-replay-plan-20260225T035758Z-v2.json \
  --output-dir tests/output/tmp-replay-run-20260225T035758Z-v2 \
  --gateway-lifecycle auto
```

`replayer replay` now shows a live progress bar with total workers, launched
workers, active workers, and failed workers.

If a worker hits `agent_timeout_s`, replay marks that worker `timed_out` and
tracks it separately from true replay failures. Timed-out workers do not cause
`replayer replay` itself to return a non-zero exit code.

Replay-side HTTP calls do not use a request timeout. That applies to:

- replayed `POST /v1/chat/completions` requests
- gateway lifecycle calls (`/job/start`, `/agent/start`, `/agent/end`, `/job/end`)

If the plan includes `agent_timeout_s`, replay enforces that wall-clock limit
per worker. For example, a plan compiled with `--agent-timeout-s 3000` will
terminate a worker at about `3000s` even though the underlying HTTP request has
no request timeout.

The compile step still keeps its own tokenize timeout. The default is `3600.0s`.

`replayer replay` can also run the same vLLM metrics monitor used by
`con-driver`. The current replay defaults are:

- sampling interval: `1.0s`
- scrape timeout: `3600.0s`

Replay uses the same defaults. When `port_profile_id` is available, vLLM logging
defaults to enabled and resolves the metrics endpoint from the selected
profile's `vllm_port`. There is no manual replay-side endpoint override.

Example:

```bash
python -m replayer replay \
  --plan tests/output/tmp-replay-plan-20260225T035758Z-v2.json \
  --port-profile-id 1 \
  --vllm-log
```

Replay vLLM metrics are written under:

- `<replay-output>/vllm-log/`
- `<replay-output>/vllm-log/monitor.stdout.log`
- `<replay-output>/vllm-log/monitor.stderr.log`

Each compressed block stores the raw `/metrics` response text per scrape, not a
parsed Prometheus JSON structure.

Replay can also override the compiled `launch_policy` while preserving the
compiled worker/request structure. Pass a JSON object with the fields you want
to overlay onto the plan's `launch_policy`.

For example, to replay the same workers with `max_concurrent = 10` instead of
the value stored in the plan:

```bash
python -m replayer replay \
  --plan tests/output/tmp-replay-plan-20260225T035758Z-v2.json \
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

If the plan includes `replay_target.port_profile_id`, `replay` will resolve the
current host URLs from `configs/port_profiles.toml` automatically. You can
override the plan with:

```bash
python -m replayer replay \
  --plan tests/output/tmp-replay-plan-20260225T035758Z-v2.json \
  --port-profile-id 1
```

Launch behavior:

- replay preserves original worker launch ordering (`launch_priority`)
- worker launch timing is driven by recorded con-driver scheduling config (`max_concurrent`, `pattern`, `pattern_args`, `seed`)
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
