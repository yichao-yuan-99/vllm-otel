This directory contains scripts that recover gateway token throughput over time
from flattened request records produced by `post-process/gateway/llm-requests`.

## Script

- `post-process/gateway/stack/extract_run.py`

Input (default):

- `<run-dir>/post-processed/gateway/llm-requests/llm-requests.json`

Command:

```bash
python post-process/gateway/stack/extract_run.py \
  --run-dir <run-dir>
```

Optional custom input path:

```bash
python post-process/gateway/stack/extract_run.py \
  --run-dir <run-dir> \
  --llm-requests tmp/llm-requests.json
```

Output directory (default):

```text
<run-dir>/post-processed/gateway/stack/
```

## Extract All Runs Under One Root Directory

When you have many replay/con-driver outputs under one root path, use recursive
discovery mode:

```bash
python post-process/gateway/stack/extract_run.py \
  --root-dir tests/output/con-driver
```

Discovery rule:

- any subdirectory that has a direct `gateway-output/` child is treated as a run directory

Optional worker count:

```bash
python post-process/gateway/stack/extract_run.py \
  --root-dir tests/output/con-driver \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/gateway/stack/extract_run.py \
  --root-dir tests/output/con-driver \
  --dry-run
```

## Method

For each request in `llm-requests.json`, the script builds token ranges and
average rates (`total_value / range_duration_s`) for five metrics:

- `prompt_tokens`
- `cached_tokens`
- `compute_prompt_tokens = prompt_tokens - cached_tokens`
- `completion_tokens`
- `compute_prompt_plus_completion_tokens` (prefill `compute_prompt_tokens`
  ranges plus decode `completion_tokens` ranges)

Range definitions:

- prefill metrics (`prompt_tokens`, `cached_tokens`, `compute_prompt_tokens`)
  - range start = `request_start_offset_s + gen_ai.latency.time_in_queue`
  - range duration = `gen_ai.latency.time_in_model_prefill`
- decode metric (`completion_tokens`)
  - range start = `request_start_offset_s + gen_ai.latency.time_to_first_token`
  - range duration = `gen_ai.latency.time_in_model_decode`

Then, for every second bucket `[t, t + 1)` from run start, it stacks overlap
contributions from all request ranges to recover throughput over time.

Note:

- if `cached_tokens` is missing for a request, this script treats it as `0` for
  both `cached_tokens` and `compute_prompt_tokens` prefill ranges.

## Outputs

This script writes 10 JSON files.

Range files (per-request ranges + average rates):

- `prompt-tokens-ranges.json`
- `cached-tokens-ranges.json`
- `compute-prompt-tokens-ranges.json`
- `completion-tokens-ranges.json`
- `compute-prompt-plus-completion-tokens-ranges.json`

Histogram files (per-second stacked values):

- `prompt-tokens-stacked-histogram.json`
- `cached-tokens-stacked-histogram.json`
- `compute-prompt-tokens-stacked-histogram.json`
- `completion-tokens-stacked-histogram.json`
- `compute-prompt-plus-completion-tokens-stacked-histogram.json`

Each histogram point includes:

- `second` (bucket start second)
- `accumulated_value` (sum of overlap contributions in that 1-second bucket)

## Visualization

You can render figures from these stacked histogram outputs with:

- `post-process/visualization/gateway-stack/generate_all_figures.py`

See:

- `post-process/visualization/gateway-stack/README.md`
