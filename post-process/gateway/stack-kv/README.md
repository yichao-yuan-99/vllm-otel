This directory contains scripts that recover stacked KV usage over time from
flattened request records produced by `post-process/gateway/llm-requests`.

## Script

- `post-process/gateway/stack-kv/extract_run.py`

Input (default):

- `<run-dir>/post-processed/gateway/llm-requests/llm-requests.json`

Command:

```bash
python post-process/gateway/stack-kv/extract_run.py \
  --run-dir <run-dir>
```

Optional custom input path:

```bash
python post-process/gateway/stack-kv/extract_run.py \
  --run-dir <run-dir> \
  --llm-requests tmp/llm-requests.json
```

Output directory (default):

```text
<run-dir>/post-processed/gateway/stack-kv/
```

## Extract All Runs Under One Root Directory

When you have many replay/con-driver outputs under one root path, use recursive
discovery mode:

```bash
python post-process/gateway/stack-kv/extract_run.py \
  --root-dir tests/output/con-driver
```

Discovery rule:

- any subdirectory that has
  `post-processed/gateway/llm-requests/llm-requests.json`

Optional worker count:

```bash
python post-process/gateway/stack-kv/extract_run.py \
  --root-dir tests/output/con-driver \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/gateway/stack-kv/extract_run.py \
  --root-dir tests/output/con-driver \
  --dry-run
```

## Method

This stage is similar to `gateway/stack`, but uses request-lifetime KV
occupancy:

- if request `i` has
  - `prompt_tokens = x`
  - `completion_tokens = y`
  - `request_start_offset_s = t0`
  - `request_end_offset_s = t1`
- then request `i` contributes `(x + y)` continuously on interval `[t0, t1)`

All request intervals are stacked into 1-second buckets to recover total
KV usage alive in the system over time.

Notes:

- if `completion_tokens` is missing for a request, this script treats it as `0`
- requests missing required fields (`request_start_offset_s`,
  `request_end_offset_s`, or `prompt_tokens`) are skipped

## Outputs

This script writes 2 JSON files.

Range file:

- `kv-usage-ranges.json`

Histogram file:

- `kv-usage-stacked-histogram.json`

Each range entry includes:

- request identifiers
- `range_start_s`, `range_end_s`, `range_duration_s`
- `kv_usage_tokens`
- `total_value`
- `avg_value_per_s`

Each histogram point includes:

- `second` (bucket start second)
- `accumulated_value` (stacked KV usage in that 1-second bucket)
