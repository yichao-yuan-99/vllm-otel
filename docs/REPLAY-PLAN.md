# Replay Plan Format

## Goal

Define a compiled replay plan that contains the minimum information needed to run deterministic replay.

The replay engine should consume this plan directly and should not need to parse raw con-driver/gateway artifacts at runtime.

## Why Compile First

1. Validate input integrity before replay starts.
2. Convert raw profile data into stable event deltas.
3. Precompute deterministic `forced_token_ids` for each request.
4. Keep replay runtime simple and robust.

## Compiler Input

A full prior job directory from con-driver + gateway profiling:

- `meta/config.toml`
- `meta/run_manifest.json`
- `meta/results.json`
- `trials/trial-*/result.json`
- single-profile gateway layout:
  - `gateway-output/run_*/manifest.json`
  - `gateway-output/run_*/events/lifecycle.jsonl`
  - `gateway-output/run_*/requests/model_inference.jsonl`
  - `gateway-output/run_*/trace/jaeger_trace.json`
- cluster-mode gateway layout:
  - `gateway-output/profile-*/run_*/manifest.json`
  - `gateway-output/profile-*/run_*/events/lifecycle.jsonl`
  - `gateway-output/profile-*/run_*/requests/model_inference.jsonl`
  - `gateway-output/profile-*/run_*/trace/jaeger_trace.json`

## Compiler Output

A single plan file, for example:

- `replay-plan.json`

The plan should be self-contained for replay scheduling and deterministic request construction.

## Deterministic Tokenization (Required)

For each recorded LLM response, compiler must compute `forced_token_ids` using the replay target vLLM tokenizer endpoint.

Rules:

- tokenization target must match replay model/tokenizer
- use vLLM `/tokenize` with `model=<replay_model>` and `add_special_tokens=false`
- source text is extracted from recorded response payload
- if tokens cannot be generated for a request, compilation fails for that request (no silent fallback)

## Minimal Plan Schema

```json
{
  "schema_version": "replay-plan.v1",
  "compiled_at": "2026-02-25T01:23:45.678Z",
  "source_job_dir": "/abs/path/to/job-...",
  "t0": "2026-02-24T19:50:34.521830Z",
  "launch_policy": {
    "strategy": "config_ordered",
    "source": "meta/config.toml[run]",
    "max_concurrent": 4,
    "seed": 7,
    "pattern": {
      "name": "poisson",
      "rate_per_second": 0.2,
      "mean_interval_s": 5.0
    }
  },
  "replay_target": {
    "model": "hosted_vllm/Qwen3-Coder-30B-A3B-Instruct",
    "deterministic_required": true
  },
  "workers": [
    {
      "worker_id": "trial-0000-...",
      "trial_id": "trial-0000-...",
      "launch_priority": 0,
      "api_token_hash": "sha256hex",
      "run_start_time": "2026-02-24T19:50:39.108149Z",
      "run_offset_s": 4.586319,
      "delta_agent_start_s": 1.000000,
      "delta_first_request_s": 1.500000,
      "requests": [
        {
          "index": 0,
          "method": "POST",
          "path": "v1/chat/completions",
          "body": { "model": "..." },
          "delta_agent_action_after_s": 0.500000,
          "expected_response_text": "expected assistant output",
          "forced_token_ids": [42, 53, 99],
          "force_eos_after_sequence": true
        }
      ]
    }
  ]
}
```

## Field Semantics

- `t0`: job start anchor.
- `launch_policy`: how replay launches workers.
  - `strategy=config_ordered` (required): preserve original worker launch ordering but compute launch timing from recorded con-driver scheduling config (`max_concurrent`, `pattern`, `seed`), not from exact original `run_offset_s`.
- `run_offset_s`: `worker.run_start_time - t0`.
- `launch_priority`: worker launch ordering index from the original run.
- `delta_agent_start_s`: `agent_start - run_start`.
- `delta_first_request_s`: `first_request_start - agent_start`.
- `delta_agent_action_after_s`: recorded agent-side delay after request `i`.
  For non-final requests, delay is before request `i+1`.
  For the final request, delay is from last response until `agent_end`.
- `expected_response_text`: canonical expected model output text for exact replay validation.
- `body`: original recorded request body, with deterministic fields already injected or ready to inject.

## Build Rules

1. Resolve `t0`:
   - primary: `meta/run_manifest.json.started_at`
   - fallback: `meta/events.jsonl` `gateway_job_start.time`
2. Map trial -> gateway run:
   - read `--api-token` from `meta/results.json` launch command
   - hash with `sha256`
   - match `manifest.api_token_hash` across:
     - `gateway-output/run_*/manifest.json`
     - `gateway-output/profile-*/run_*/manifest.json`
3. Detect backend from `meta/config.toml`:
   - primary: `[backend].name`
   - fallback: `[run].driver_backend`
4. Extract target config using backend-specific rules.
   - For `harbor` backend:
     - model: parse `[backend].forwarded_args` for `--model <value>`
     - if `forwarded_args` is missing/incomplete, fallback to per-launch command
       parsing in `meta/results.json[].command` with the same extraction logic.
   - For non-`harbor` backends:
     - use a dedicated extractor for that backend.
     - if extractor is not implemented, fail fast with `unsupported backend`.
5. Extract timing and request sequence from gateway request/lifecycle/trace files.
   - compute worker `launch_priority` from original run ordering.
6. Compute `forced_token_ids` from recorded response text via `/tokenize`.
7. Emit plan with workers ordered by `launch_priority` (original launch order) and requests by index.

## Replay Engine Contract

Given a compiled plan:

1. Launch workers by `launch_policy`:
   - keep `launch_priority` ordering
   - use `max_concurrent`, `pattern`, and `seed` from the plan
2. For each launched worker, sleep `delta_agent_start_s`.
3. Sleep `delta_first_request_s`.
4. For each request in order:
   - send request
   - wait until response returns
   - extract response text from the raw vLLM response payload and require exact match with `expected_response_text`
   - sleep `delta_agent_action_after_s`
5. Continue until all worker requests complete.

The engine does not need to compute tokenization or parse raw traces at runtime.

## Validation Requirements

Compiler must fail fast on:

- missing required input files
- missing/malformed `t0`
- trial-to-artifact mapping mismatch
- non-monotonic request order/timestamps
- missing request body/response text for deterministic tokenization
- tokenizer endpoint/model mismatch
- unsupported backend (no extractor)
