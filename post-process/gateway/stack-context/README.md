This directory contains scripts that recover stacked agent context usage over
time from flattened request records produced by
`post-process/gateway/llm-requests`.

## Script

- `post-process/gateway/stack-context/extract_run.py`

Input (default):

- `<run-dir>/post-processed/gateway/llm-requests/llm-requests.json`

Command:

```bash
python post-process/gateway/stack-context/extract_run.py \
  --run-dir <run-dir>
```

Optional custom input path:

```bash
python post-process/gateway/stack-context/extract_run.py \
  --run-dir <run-dir> \
  --llm-requests tmp/llm-requests.json
```

Output directory (default):

```text
<run-dir>/post-processed/gateway/stack-context/
```

## Extract All Runs Under One Root Directory

When you have many replay/con-driver outputs under one root path, use recursive
discovery mode:

```bash
python post-process/gateway/stack-context/extract_run.py \
  --root-dir tests/output/con-driver
```

Discovery rule:

- any subdirectory that has
  `post-processed/gateway/llm-requests/llm-requests.json`

Optional worker count:

```bash
python post-process/gateway/stack-context/extract_run.py \
  --root-dir tests/output/con-driver \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/gateway/stack-context/extract_run.py \
  --root-dir tests/output/con-driver \
  --dry-run
```

## Method

This stage is similar to `gateway/stack`, but instead of throughput ranges, it
models per-agent context usage over time.

For each agent:

- context starts at `0`
- each request refreshes context usage to `prompt_tokens + completion_tokens`
- that value remains active from this request's `request_start_offset_s` until
  the next request's `request_start_offset_s`
- after the final request, context is held until the lifecycle `agent_end`
  timestamp in that agent's `gateway-output/.../events/lifecycle.jsonl`
- before the first request, an idle `0` segment is emitted when the first start
  offset is greater than `0`

Fallback behavior:

- if lifecycle `agent_end` cannot be resolved for an agent, the script falls
  back to inferred end time from `request_end_offset_s` and
  `request_end_to_run_end_s`

Example for one agent:

- `t0 -> t1`: usage `0`
- `t1 -> t2`: usage `1100` (request 1: `1000 + 100`)
- `t2 -> t3`: usage `1450` (request 2: `1300 + 150`)

Then all agents' segments are stacked into per-second buckets like
`gateway/stack`.

## Outputs

This script writes 2 JSON files.

Range file:

- `context-usage-ranges.json`

Histogram file:

- `context-usage-stacked-histogram.json`

Each range entry includes:

- agent/request identifiers
- `segment_type` (`idle` or `active`)
- `range_start_s`, `range_end_s`, `range_duration_s`
- `context_usage_tokens`
- `total_value`
- `avg_value_per_s`

Each histogram point includes:

- `second` (bucket start second)
- `accumulated_value` (stacked context usage in that 1-second bucket)

## Visualization

You can render figures from stacked context histogram output with:

- `post-process/visualization/gateway-stack-context/generate_all_figures.py`

See:

- `post-process/visualization/gateway-stack-context/README.md`
