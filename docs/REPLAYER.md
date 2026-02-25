# Replayer (Design)

## Goal

Replay a profiled con-driver + gateway job to drive a workload pattern that is as similar as possible to the original run, while enforcing deterministic LLM output.

## Replayer Functions

The replayer must provide two functions:

1. `compile_profile_to_plan`
2. `replay_from_plan`

These can be run separately (two-step workflow) or chained in one command.

## CLI Interface

Invoke with:

```bash
python -m replayer --help
```

Compile profile to plan:

```bash
python -m replayer compile \
  --job-dir tests/output/con-driver/job-<timestamp> \
  --plan-out tests/output/replay-plan.json
```

Replay from compiled plan:

```bash
python -m replayer replay \
  --plan tests/output/replay-plan.json \
  --output-dir tests/output/replay-run \
  --gateway-lifecycle auto
```

## Function 1: `compile_profile_to_plan`

Input:

- full job profile directory from a previous con-driver + gateway run
- includes `meta/`, `trials/`, and `gateway-output/`

Output:

- one compiled `replay-plan.json` following `docs/REPLAY-PLAN.md`

Behavior:

- validate required profile files
- resolve backend-aware replay target fields
- for `harbor`, extract gateway/model/api_base from `meta/config.toml` and fallback to `meta/results.json` command parsing when needed
- resolve `T0` (job start) from `meta/run_manifest.json.started_at` with configured fallback
- map trial -> gateway run by hashing original `api_token` (`sha256`) and matching `manifest.api_token_hash`
- derive replay launch policy from recorded con-driver run config (`pattern`, `pattern_args`, `max_concurrent`, `seed`)
- convert raw timestamps/spans into replay deltas
- call target vLLM `/tokenize` to compute `forced_token_ids` from recorded response text
- fail fast on any deterministic/tokenization/mapping inconsistency

## Function 2: `replay_from_plan`

Input:

- compiled `replay-plan.json`

Output:

- replay run directory with replay logs and summary

Behavior:

- launch workers by `launch_priority` + recorded con-driver launch policy
- execute per-worker request sequence using plan deltas
- do not parse raw profile artifacts at runtime

## Replay Timing Logic (from Plan)

Replay is response-gated and delta-driven:

1. launch workers in original `launch_priority` order
   - use recorded con-driver launch policy (`max_concurrent`, `pattern`, `seed`)
   - do not replay exact absolute `run_offset_s` timing
2. sleep `delta_agent_start_s`
3. sleep `delta_first_request_s`
4. for each request in order:
   - send request
   - wait for response
   - verify returned output text exactly equals the recorded expected text
   - sleep `delta_agent_action_after_s` (for the last request, this is the recorded delay until `agent_end`)

Request latency is a runtime outcome. Replay control does not depend on measuring or fitting latency to original values.

## Deterministic Requirement

Deterministic replay is required.

Each replay request must carry force-sequence fields from the plan:

- `vllm_xargs.forced_token_ids`
- `vllm_xargs.force_eos_after_sequence=true`
- `expected_response_text` (used for strict response equality check)

No silent fallback to non-deterministic replay is allowed.

## Runtime Behavior

- show progress similar to con-driver (launched/active/completed workers, requests sent)
- handle `Ctrl+C` gracefully:
  - stop launching new workers
  - signal active workers to stop
  - wait with timeout
  - write partial replay summary

## Outputs

Replay output directory example:

- `<original-job-name>.replayed-<ISO8601-UTC>`

Expected contents:

- replay manifest
- orchestrator log
- per-worker logs
- replay summary (`scheduled/sent/succeeded/failed`)
- copied or linked replay plan used for execution

## Validation

Start with a small profiled job (few agents, few requests) before scaling up.

Exact-match checks (must pass):

- every replay request prompt content exactly matches the recorded prompt content (`messages`/`prompt`/`input`)
- every replay request returns text exactly equal to `expected_response_text`
- replayed request count per worker equals planned request count
- token usage fields match recorded values when present (`usage.prompt_tokens`)
- token usage fields match recorded values when present (`usage.completion_tokens`)
- token usage fields match recorded values when present (`usage.prompt_tokens_details.cached_tokens`, if reported by backend)

Similarity checks (for human inspection, report min/avg/max relative error):

- total job duration
- per-agent duration
- per-request latency

Pass/fail rule:

- any exact-match violation is a replay failure
- similarity metrics are diagnostic; they should be reviewed and tracked across runs

## Reference

- Plan schema and compile contract: `docs/REPLAY-PLAN.md`
