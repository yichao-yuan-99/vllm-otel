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
  --plan-out tests/output/tmp-replay-plan-20260225T035758Z-v2.json
```

Replay:

```bash
python -m replayer replay \
  --plan tests/output/tmp-replay-plan-20260225T035758Z-v2.json \
  --output-dir tests/output/tmp-replay-run-20260225T035758Z-v2 \
  --gateway-lifecycle auto
```

Launch behavior:

- replay preserves original worker launch ordering (`launch_priority`)
- worker launch timing is driven by recorded con-driver scheduling config (`max_concurrent`, `pattern`, `pattern_args`, `seed`)
- replay does not require exact original absolute launch offsets
- replay requires plans that include `launch_policy` (`config_ordered`)

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
