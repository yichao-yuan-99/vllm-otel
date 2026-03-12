This directory computes run-level and root-level stats for con-driver runs.

## Scripts

- `original-analysis/run-stats/extract_run.py`
- `original-analysis/run-stats/aggregate_runs_csv.py`
- `original-analysis/run-stats/plot_job_context_cdf.py`

## Per-Run Extraction

Input run layout:

- `<run-dir>/meta/run_manifest.json`
- `<run-dir>/meta/results.json`
- `<run-dir>/gateway-output/`

Command:

```bash
python original-analysis/run-stats/extract_run.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/run-stats/run-stats-summary.json
```

Batch mode:

```bash
python original-analysis/run-stats/extract_run.py \
  --root-dir <root-dir>
```

Optional:

- `--max-procs <positive-int>`
- `--dry-run`

Per-run summary fields:

- `agent_type`
- `dataset`
- `score`
- `job_count`
- `jobs` (per-job stats)
- `job_max_request_lengths`
- `avg_job_max_request_length`
- `max_job_max_request_length`
- `job_avg_prompt_tokens_per_request`
- `job_avg_generation_tokens_per_request`
- `avg_turns_per_run`
- `max_turns_per_run`
- `avg_run_prompt_tokens_per_request`
- `avg_run_generation_tokens_per_request`

Definitions:

- request length = `prompt_tokens + generation_tokens`
- per-job max request length = max request length within one gateway job
- run max job max request length = max over per-job max request lengths
- per-job avg prompt/generation = average prompt/generation tokens per request for that job
- run average turns = average request count per job in the run
- run max turns = max request count across jobs in the run
- run avg prompt/generation = average prompt/generation tokens per request across all jobs in run

## Root CSV Aggregation

After per-run extraction, aggregate all run summaries under one root:

```bash
python original-analysis/run-stats/aggregate_runs_csv.py \
  --root-dir <root-dir>
```

Default output:

```text
<root-dir>/run-stats-summary.csv
```

Optional:

- `--output <csv-path>`

Rows and order:

- one row per run
- row key column: `run_path`
- rows sorted with the same path-ordering logic used by `post-process/global`

CSV columns:

- `agent_type`
- `dataset`
- `score`
- `job_count`
- `avg_job_max_request_length`
- `max_job_max_request_length`
- `avg_turns_per_run`
- `max_turns_per_run`
- `avg_run_prompt_tokens_per_request`
- `avg_run_generation_tokens_per_request`

## Per-Run CDF Visualization

Generate one CDF per run for job context usage:

- x-axis: per-job max request length (`max_request_length`)
- y-axis: cumulative fraction of jobs in `[0, 1]`

Single run:

```bash
python original-analysis/run-stats/plot_job_context_cdf.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/run-stats/job-max-request-length-cdf.png
```

Batch mode:

```bash
python original-analysis/run-stats/plot_job_context_cdf.py \
  --root-dir <root-dir>
```

Optional:

- `--format {png,pdf,svg}`
- `--dpi <positive-int>`
- `--max-procs <positive-int>`
- `--dry-run`

## Quick Start

```bash
python original-analysis/run-stats/extract_run.py --root-dir results/qwen3-coder-30b
python original-analysis/run-stats/aggregate_runs_csv.py --root-dir results/qwen3-coder-30b
python original-analysis/run-stats/plot_job_context_cdf.py --root-dir results/qwen3-coder-30b
```
