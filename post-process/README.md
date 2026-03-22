# Post-Process Index

This directory contains post-process utilities for `con-driver` and `replay`
outputs.

## What This Covers

- `global`: run-level timing extraction and cross-run CSV aggregation
- `global-progress`: run-level replay completion milestone timing
- `job-throughput`: moving replay/job throughput over time for each run
- `job-concurrency`: per-second active job concurrency over time for each run
- `split/duration`: context-usage percentile split tables for per-job duration/turn/token metrics
- `vllm-metrics`: parse/extract/summarize vLLM Prometheus metrics
- `power`: summarize GPU power traces and estimate total run energy
- `power-sampling`: sample interpolated power at prefill-concurrency ticks
- `gateway/llm-requests`: flatten and summarize gateway LLM request traces
- `prefill-concurrency`: 10ms prefill-phase concurrency series from LLM request spans
- `gateway/stack`: recover stacked per-second gateway token throughput from request ranges
- `gateway/stack-context`: recover stacked per-second agent context usage from request history
- `gateway/stack-kv`: recover stacked per-second request-lifetime KV usage from request history
- `gateway/usage`: aggregate gateway token usage per run and per agent

## Quick Navigation

- `post-process/global/README.md`
- `post-process/global-progress/README.md`
- `post-process/job-throughput/README.md`
- `post-process/job-concurrency/README.md`
- `post-process/visualization/job-throughput/README.md`
- `post-process/visualization/job-concurrency/README.md`
- `post-process/split/duration/README.md`
- `post-process/vllm-metrics/README.md`
- `post-process/power/README.md`
- `post-process/power-sampling/README.md`
- `post-process/gateway/llm-requests/README.md`
- `post-process/prefill-concurrency/README.md`
- `post-process/gateway/stack/README.md`
- `post-process/gateway/stack-context/README.md`
- `post-process/gateway/stack-kv/README.md`
- `post-process/gateway/usage/README.md`
- `post-process/visualization/gateway-stack/README.md`
- `post-process/visualization/gateway-stack-context/README.md`
- `post-process/visualization/gateway-stack-kv/README.md`
- `post-process/visualization/vllm-metrics/README.md`
- `post-process/visualization/power/README.md`
- `post-process/visualization/prefill-concurrency/README.md`

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

1. `service-failure/extract_run.py`
2. `global/extract_run.py`
3. `global-progress/extract_run.py`
4. `job-throughput/extract_run.py`
5. `job-concurrency/extract_run.py`
6. `gateway/llm-requests/extract_run.py`
7. `prefill-concurrency/extract_run.py`
8. `gateway/stack/extract_run.py`
9. `gateway/stack-context/extract_run.py`
10. `gateway/stack-kv/extract_run.py`
11. `gateway/usage/extract_run.py`
12. `split/duration/extract_run.py`
13. `vllm-metrics/extract_run.py`
14. `vllm-metrics/summarize_timeseries.py`
15. `power/extract_run.py`
16. `power-sampling/extract_run.py`
17. `visualization/job-throughput/generate_all_figures.py`
18. `visualization/job-concurrency/generate_all_figures.py`
19. `visualization/prefill-concurrency/generate_all_figures.py`
20. `visualization/gateway-stack/generate_all_figures.py`
21. `visualization/gateway-stack-context/generate_all_figures.py`
22. `visualization/gateway-stack-kv/generate_all_figures.py`
23. `visualization/vllm-metrics/generate_all_figures.py`
24. `visualization/power/generate_all_figures.py`
25. `global/aggregate_runs_csv.py` (root-dir mode only, skipped in dry-run)

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

### `job-throughput`

- `post-process/job-throughput/extract_run.py`
- purpose: compute moving job/replay throughput over the run timeline
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- optional controls:
- `--timepoint-freq-hz`
- `--window-size-s`
- default output:
- `<run-dir>/post-processed/job-throughput/job-throughput-timeseries.json`

### `job-concurrency`

- `post-process/job-concurrency/extract_run.py`
- purpose: compute per-second active job concurrency from trial start/end ranges
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/job-concurrency/job-concurrency-timeseries.json`

- `post-process/visualization/job-concurrency/generate_all_figures.py`
- purpose: render one concurrency line chart per run from extracted job-concurrency timeseries
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/job-concurrency/`

- `post-process/visualization/job-throughput/generate_all_figures.py`
- purpose: render one throughput line chart per run from extracted job-throughput timeseries
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/job-throughput/`

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

### `power`

- `post-process/power/extract_run.py`
- purpose: summarize GPU power from `power/power-log.jsonl` into
- avg/min/max power
- total energy estimate
- time-offset power points (`time_offset_s`, `power_w`)
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/power/power-summary.json`
- behavior:
- if `power/power-log.jsonl` is missing, writes a normal output with `power_log_found=false`

- `post-process/visualization/power/generate_all_figures.py`
- purpose: render one GPU power-over-time chart per run from power summary points
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/power/`

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
- `llm-request-speed-stats.json`
- `llm-requests-longest-10.json`
- `llm-requests-shortest-10.json`
- `llm-requests-stats.<status_code>.json`

### `prefill-concurrency`

- `post-process/prefill-concurrency/extract_run.py`
- purpose: extract request-level prefill activity ranges and compute 10ms prefill
  concurrency over the full run duration
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- optional controls:
- `--tick-ms` (default `10`)
- optional input override: `--llm-requests <path>` (run-dir mode only)
- default output directory:
- `<run-dir>/post-processed/prefill-concurrency/`
- primary outputs:
- `prefill-activities.json`
- `prefill-concurrency-timeseries.json`
- `prefill-concurrency-stats.json`

- `post-process/visualization/prefill-concurrency/generate_all_figures.py`
- purpose: render one prefill-concurrency line chart per run from extracted
  prefill-concurrency timeseries
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/prefill-concurrency/`

### `gateway/stack`

- `post-process/gateway/stack/extract_run.py`
- purpose: recover per-request token ranges and stacked per-second throughput from `llm-requests.json`
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- optional input override: `--llm-requests <path>` (run-dir mode only)
- default output directory:
- `<run-dir>/post-processed/gateway/stack/`
- primary outputs:
- `prompt-tokens-ranges.json`
- `cached-tokens-ranges.json`
- `compute-prompt-tokens-ranges.json`
- `completion-tokens-ranges.json`
- `compute-prompt-plus-completion-tokens-ranges.json`
- `prompt-tokens-stacked-histogram.json`
- `cached-tokens-stacked-histogram.json`
- `compute-prompt-tokens-stacked-histogram.json`
- `completion-tokens-stacked-histogram.json`
- `compute-prompt-plus-completion-tokens-stacked-histogram.json`

- `post-process/visualization/gateway-stack/generate_all_figures.py`
- purpose: render raw + smoothed stacked-throughput line charts for the 5 gateway stack metrics
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/gateway-stack/`

### `gateway/stack-context`

- `post-process/gateway/stack-context/extract_run.py`
- purpose: recover per-agent context usage ranges and stacked per-second context
  usage from `llm-requests.json`
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- optional input override: `--llm-requests <path>` (run-dir mode only)
- default output directory:
- `<run-dir>/post-processed/gateway/stack-context/`
- primary outputs:
- `context-usage-ranges.json`
- `context-usage-stacked-histogram.json`
- note:
- final active segment per agent ends at lifecycle `agent_end`
  (`gateway-output/.../events/lifecycle.jsonl`) when available

- `post-process/visualization/gateway-stack-context/generate_all_figures.py`
- purpose: render raw + smoothed stacked context-usage line charts
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/gateway-stack-context/`

### `gateway/stack-kv`

- `post-process/gateway/stack-kv/extract_run.py`
- purpose: recover request-lifetime KV occupancy ranges and stacked per-second KV
  usage from `llm-requests.json`
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- optional input override: `--llm-requests <path>` (run-dir mode only)
- default output directory:
- `<run-dir>/post-processed/gateway/stack-kv/`
- primary outputs:
- `kv-usage-ranges.json`
- `kv-usage-stacked-histogram.json`

- `post-process/visualization/gateway-stack-kv/generate_all_figures.py`
- purpose: render raw + smoothed stacked KV-usage line charts
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/gateway-stack-kv/`

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
