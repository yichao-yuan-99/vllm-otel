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

Additional per-run outputs:

```text
<run-dir>/original-analysis/split/top-p-token-usage-two-groups.json
<run-dir>/original-analysis/split/top-p-context-usage-two-groups.json
<run-dir>/original-analysis/split/top-p-usage-ratio-summary.strict-499.json
<run-dir>/original-analysis/split/top-p-token-usage-two-groups.strict-499.json
<run-dir>/original-analysis/split/top-p-context-usage-two-groups.strict-499.json
```

Strict-mode details are documented in:

- `original-analysis/split/README.strict-499.md`

Batch mode:

```bash
python original-analysis/split/extract_run.py \
  --root-dir <root-dir>
```

When `--root-dir` is used, normal and strict top-level CSV tables are written:

```text
<root-dir>/split-top-p-token-usage-two-group-table.csv
<root-dir>/split-top-p-context-usage-two-group-table.csv
<root-dir>/split-top-p-token-usage-two-group-table.strict-499.csv
<root-dir>/split-top-p-context-usage-two-group-table.strict-499.csv
```

Columns:

- `run_path`
- `group_top_trail_count`
- `group_top_trail_percentage`
- `group_rest_trail_count`
- `group_rest_trail_percentage`

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
- `two_group_summaries.token_usage`
- `two_group_summaries.context_usage`
- `unranked_mode` (`normal` or `strict_499`)
- `trail_count_with_status_499`

## Two-Group Files

Two primary two-group files are emitted per run (plus strict `.strict-499`
counterparts listed above):

- `top-p-token-usage-two-groups.json`
- `top-p-context-usage-two-groups.json`

Each file chooses a split from the existing top-p ratio table:

1. find the first percentile where `top/rest > 1`
2. if none exists, fallback to the first percentile where `top/rest >= 1`
3. if neither exists (for example zero-usage edge cases), fallback to `p=1`

Then it emits two groups:

- `group_top`: the smallest top-p group that satisfies the criterion
- `group_rest`: all remaining trails

Each group includes:

- `trail_names`
- `trial_count`
- `trial_percentage` (percentage among ranked trails)

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
