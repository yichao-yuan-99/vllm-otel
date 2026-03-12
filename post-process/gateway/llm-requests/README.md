This directory contains scripts to post-process gateway request traces.

## Script

- `post-process/gateway/llm-requests/extract_run.py`

Input:

- `<run-dir>/gateway-output/`
- supports both:
  - `gateway-output/run_*/...`
  - `gateway-output/profile-*/run_*/...`

Command:

```bash
python post-process/gateway/llm-requests/extract_run.py \
  --run-dir <run-dir>
```

The extractor shows a live progress bar while request records are loaded.

Output directory (default):

```text
<run-dir>/post-processed/gateway/llm-requests/
```

Files written:

- `llm-requests.json`
- `llm-request-stats.json`
- `llm-requests-longest-10.json`
- `llm-requests-shortest-10.json`
- `llm-requests-stats.<status_code>.json` (for each observed return code, e.g. `200`, `499`)

## Extract All Runs Under One Root Directory

When you have many replay/con-driver outputs under one root path, use recursive
discovery mode:

```bash
python post-process/gateway/llm-requests/extract_run.py \
  --root-dir tests/output/con-driver
```

Discovery rule:

- any subdirectory that has a direct `gateway-output/` child is treated as a run directory

Optional worker count:

```bash
python post-process/gateway/llm-requests/extract_run.py \
  --root-dir tests/output/con-driver \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/gateway/llm-requests/extract_run.py \
  --root-dir tests/output/con-driver \
  --dry-run
```

## llm-requests.json

Contains one flattened dict per request, sorted by `request_start_time`.

Each request includes:

- offsets:
  - `request_start_offset_s` from `job_start`
  - `request_end_offset_s` from `job_start`
  - `request_end_to_run_end_s` from `job_end` when available
- request timing/duration from `requests/model_inference.jsonl`
- usage tokens from `response.usage`:
  - `prompt_tokens`
  - `completion_tokens`
  - `total_tokens`
  - `cached_tokens` (`prompt_tokens_details.cached_tokens`)
- identifiers and request metadata (`request_id`, `trace_id`, `model_inference_span_id`, etc.)
- all tags from the child `llm_request` span in `trace/jaeger_trace.json`, joined by `model_inference_span_id`

## llm-request-stats.json

Contains aggregated numeric stats across all flattened request records:

- `count`
- `min`
- `max`
- `avg`

Stats are generated for every numeric field found in `llm-requests.json`
(durations, token counts, latency tags, offsets, etc.).

## llm-requests-stats.<status_code>.json

Contains per-status-code numeric stats generated from only requests with that
return code.

Examples:

- `llm-requests-stats.200.json`
- `llm-requests-stats.499.json`

## llm-requests-longest-10.json

Contains up to 10 request records with the largest duration values.

- primary duration key: `request_duration_ms`
- fallback key: `duration_ms`
- records are sorted by duration descending

## llm-requests-shortest-10.json

Contains up to 10 request records with the smallest duration values.

- primary duration key: `request_duration_ms`
- fallback key: `duration_ms`
- records are sorted by duration ascending
