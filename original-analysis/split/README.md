This directory computes top-p usage split statistics for replay runs.

## Scripts

- `original-analysis/split/extract_run.py`
- `original-analysis/split/aggregate_runs_csv.py`

## Per-Run Extraction

Input run layout:

- `<run-dir>/gateway-output/`
- per gateway job: `requests/model_inference.jsonl`

Command:

```bash
python original-analysis/split/extract_run.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/original-analysis/split/top-p-usage-ratio-summary.json
```

Batch mode:

```bash
python original-analysis/split/extract_run.py \
  --root-dir <root-dir>
```

Optional:

- `--max-procs <positive-int>`
- `--dry-run`
- `--output <json-path>` (only with `--run-dir`)

Definitions:

- trail context length = max request length in trail
- request length = `prompt_tokens + completion_tokens`
- request token usage = `prompt_tokens + completion_tokens - cached_tokens`
- trail token usage = sum of request token usage in that trail

For each run:

1. Discover all gateway trails (`gateway-output/run_*` and `gateway-output/profile-*/run_*`).
2. Rank trails by context length (descending).
3. For each `p` in `1..99`, split into top `p%` trails vs rest.
4. Compute:
   - top/rest token usage totals
   - top/rest context totals
   - top token usage ratio = `top_token_usage / rest_token_usage`
   - top context usage ratio = `top_context / rest_context`
   - top token/context share = `top / total` (also emitted for reference)

Output includes:

- `table_2x99.top_p_token_usage_ratio`
- `table_2x99.top_p_context_usage_ratio`
- `table_2x99.top_p_token_usage_share`
- `table_2x99.top_p_context_usage_share`
- `by_percentile` (per-p detailed totals and counts)
- `ranked_trails` / `unranked_trails`

## Root CSV Aggregation

After per-run extraction, aggregate summaries under one root:

```bash
python original-analysis/split/aggregate_runs_csv.py \
  --root-dir <root-dir>
```

Default output directory:

```text
<root-dir>/split-top-p-ratio-tables
```

Files written:

- `top_p_token_usage_ratio.csv`
- `top_p_context_usage_ratio.csv`
- `split-top-p-ratio-manifest.json`

Optional:

- `--skip-extract` (only aggregate existing JSON summaries)
- `--output-dir <dir-path>`
- `--max-procs <positive-int>`
- `--dry-run`

CSV format:

- one row per run
- row key: `run_path`
- columns: `p1` ... `p99` (or up to max available in input)
