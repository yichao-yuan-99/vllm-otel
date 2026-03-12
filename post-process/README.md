# Post-Process Index

This directory contains post-process utilities for `con-driver` and `replay`
outputs.

## What This Covers

- `global`: run-level timing extraction and cross-run CSV aggregation
- `global-progress`: run-level replay completion milestone timing
- `split/duration`: context-usage percentile split tables for per-job duration/turn/token metrics
- `vllm-metrics`: parse/extract/summarize vLLM Prometheus metrics
- `gateway/llm-requests`: flatten and summarize gateway LLM request traces
- `gateway/usage`: aggregate gateway token usage per run and per agent

## Quick Navigation

- `post-process/global/README.md`
- `post-process/global-progress/README.md`
- `post-process/split/duration/README.md`
- `post-process/vllm-metrics/README.md`
- `post-process/gateway/llm-requests/README.md`
- `post-process/gateway/usage/README.md`
- `post-process/visualization/vllm-metrics/README.md`

## Script Index

### `run_all.py`

- `post-process/run_all.py`
- purpose: run all post-process stages in pipeline order
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers for batch-capable stages: `--max-procs`
- dry-run for batch-capable stages: `--dry-run`
- optional skips:
- `--skip-visualization`
- `--skip-aggregate-csv`
- smart fallback:
- if `--run-dir` does not match a direct run layout but contains nested run
  directories, it automatically runs the root-dir pipeline for that path

Pipeline order:

1. `global/extract_run.py`
2. `global-progress/extract_run.py`
3. `gateway/llm-requests/extract_run.py`
4. `gateway/usage/extract_run.py`
5. `split/duration/extract_run.py`
6. `vllm-metrics/extract_run.py`
7. `vllm-metrics/summarize_timeseries.py`
8. `visualization/vllm-metrics/generate_all_figures.py`
9. `global/aggregate_runs_csv.py` (root-dir mode only, skipped in dry-run)

### `global-progress`

- `post-process/global-progress/extract_run.py`
- purpose: compute replay completion milestone times since run start
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/global-progress/replay-progress-summary.json`

- `post-process/global-progress/aggregate_runs_csv.py`
- purpose: extract all runs under one root and aggregate milestones into one CSV
- supports:
- root aggregation: `--root-dir <root-dir>`
- extraction controls:
- `--max-procs`
- `--milestone-step`
- `--dry-run`
- optional output path: `--output <csv-path>`
- default output:
- `<root-dir>/replay-progress-summary.csv`

### `split/duration`

- `post-process/split/duration/extract_run.py`
- purpose: split jobs by `max_request_length` percentile rank and summarize
- per-bin averages for:
- `duration_s`
- `turn_count`
- `prompt_tokens`
- `decode_tokens`
- `cached_prompt_tokens`
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- optional `--split-count` (default `10`)
- default output:
- `<run-dir>/post-processed/split/duration/duration-split-summary.json`

- `post-process/split/duration/aggregate_runs_csv.py`
- purpose: extract all runs then write one CSV per metric table
- supports:
- root aggregation: `--root-dir <root-dir>`
- extraction controls:
- `--max-procs`
- `--split-count`
- `--dry-run`
- optional output directory: `--output-dir <dir>`
- optional `--skip-extract`
- default output directory:
- `<root-dir>/split-duration-tables/`

### `global`

- `post-process/global/extract_run.py`
- purpose: extract per-run global timing summary
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/global/trial-timing-summary.json`

- `post-process/global/aggregate_runs_csv.py`
- purpose: aggregate many run summaries into one CSV
- supports:
- root aggregation: `--root-dir <root-dir>`
- optional output path: `--output <csv-path>`
- default output:
- `<root-dir>/trial-timing-summary.csv`
- includes:
- global timing columns (`total_duration_s`, `trial_duration_*`)
- vLLM metric columns from vLLM stats (`vllm:*:{avg|min|max}`)

### `vllm-metrics`

- `post-process/vllm-metrics/parse_record.py`
- purpose: parse one raw metrics scrape record (`content`) into structured JSON

- `post-process/vllm-metrics/extract_run.py`
- purpose: extract vLLM gauge/counter timeseries from `vllm-log/`
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.json`

- `post-process/vllm-metrics/summarize_timeseries.py`
- purpose: summarize extracted timeseries with `min/max/avg/sample_count`
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.stats.json`

### `gateway/llm-requests`

- `post-process/gateway/llm-requests/extract_run.py`
- purpose: flatten and summarize gateway request traces
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output directory:
- `<run-dir>/post-processed/gateway/llm-requests/`
- primary outputs:
- `llm-requests.json`
- `llm-request-stats.json`
- `llm-requests-longest-10.json`
- `llm-requests-shortest-10.json`
- `llm-requests-stats.<status_code>.json`

### `gateway/usage`

- `post-process/gateway/usage/extract_run.py`
- purpose: aggregate gateway token usage for one run and per-agent breakdown
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/gateway/usage/usage-summary.json`

## Notes

- most scripts can run per-run (`--run-dir`) or recursively (`--root-dir`)
- batch mode discovers valid run directories by expected input layout
- batch mode can use multiprocessing via `--max-procs`
