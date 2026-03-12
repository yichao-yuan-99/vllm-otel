This directory contains scripts to aggregate gateway token usage for one run.

## Script

- `post-process/gateway/usage/extract_run.py`

Input:

- `<run-dir>/gateway-output/`
- supports both:
  - `gateway-output/run_*/...`
  - `gateway-output/profile-*/run_*/...`

Command:

```bash
python post-process/gateway/usage/extract_run.py \
  --run-dir <run-dir>
```

Output file (default):

```text
<run-dir>/post-processed/gateway/usage/usage-summary.json
```

Optional output path:

```bash
python post-process/gateway/usage/extract_run.py \
  --run-dir <run-dir> \
  --output tmp/usage-summary.json
```

## Extract All Runs Under One Root Directory

When you have many replay/con-driver outputs under one root path, use recursive
discovery mode:

```bash
python post-process/gateway/usage/extract_run.py \
  --root-dir tests/output/con-driver
```

Discovery rule:

- any subdirectory that has a direct `gateway-output/` child is treated as a run directory

Optional worker count:

```bash
python post-process/gateway/usage/extract_run.py \
  --root-dir tests/output/con-driver \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/gateway/usage/extract_run.py \
  --root-dir tests/output/con-driver \
  --dry-run
```

## usage-summary.json

Contains both run-level and per-agent token usage derived from
`requests/model_inference.jsonl`.

Top-level fields:

- `source_run_dir`
- `source_gateway_output_dir`
- `agent_count`
- `request_count`
- `usage`
- `agents`

Usage fields:

- `prompt_tokens`: total prompt tokens
- `generation_tokens`: total generation tokens (same as `completion_tokens`)
- `completion_tokens`: total completion tokens
- `cached_prompt_tokens`: total cached prompt tokens
- `prefill_prompt_tokens`: `prompt_tokens - cached_prompt_tokens`
- `max_request_length`: max request length where request length is
  `prompt_tokens + generation_tokens`
- `avg_worker_max_request_length`: average of per-worker `max_request_length`
- `requests_with_prompt_tokens`
- `requests_with_generation_tokens`
- `requests_with_completion_tokens`
- `requests_with_cached_prompt_tokens`
- `requests_with_request_length`

Per-agent entries (`agents`) include:

- `gateway_run_id` (run directory name, for example `run_20260308T...`)
- `gateway_profile_id` (`null` for non-profile layout)
- `api_token_hash` (from `events/lifecycle.jsonl` when available)
- `request_count`
- `usage` (same usage fields listed above)
