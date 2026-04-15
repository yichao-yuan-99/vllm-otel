# Post-Process Index

This directory contains post-process utilities for `con-driver` and `replay`
outputs.

## What This Covers

- `global`: run-level timing extraction and cross-run CSV aggregation
- `global-progress`: run-level replay completion milestone timing
- `job-throughput`: moving replay/job throughput over time for each run
- `job-concurrency`: per-second active job concurrency over time for each run
- `agent-output-throughput`: per-agent output-token throughput from gateway request logs
- `slo-decision`: SLO-triggered controller decisions from the SLO linespace controller
- `split/duration`: context-usage percentile split tables for per-job duration/turn/token metrics
- `vllm-metrics`: parse/extract/summarize vLLM Prometheus metrics
- `power`: summarize GPU power traces and estimate total run energy
- `power-sampling`: sample interpolated power at prefill-concurrency ticks
- `freq-control`: summarize freq-controller query/decision logs and recover the control timeline
- `visualization/stacked-per-agent`: materialize and render a fixed-width per-agent context stack bar chart
- `gateway/llm-requests`: flatten and summarize gateway LLM request traces
- `gateway/slo-aware-log`: summarize event-driven gateway SLO-aware `ralexation` decisions
- `prefill-concurrency`: 10ms prefill-phase concurrency series from LLM request spans
- `gateway/stack`: recover stacked per-second gateway token throughput from request ranges
- `gateway/stack-context`: recover stacked per-second agent context usage from request history
- `gateway/stack-kv`: recover stacked per-second request-lifetime KV usage from request history
- `key-stats`: consolidate a small set of headline summary stats into one JSON
- `gateway/usage`: aggregate gateway token usage per run and per agent

## Quick Navigation

- `post-process/global/README.md`
- `post-process/global-progress/README.md`
- `post-process/job-throughput/README.md`
- `post-process/job-concurrency/README.md`
- `post-process/agent-output-throughput/README.md`
- `post-process/slo-decision/README.md`
- `post-process/visualization/job-throughput/README.md`
- `post-process/visualization/agent-output-throughput/README.md`
- `post-process/visualization/slo-decision/README.md`
- `post-process/visualization/job-concurrency/README.md`
- `post-process/split/duration/README.md`
- `post-process/vllm-metrics/README.md`
- `post-process/power/README.md`
- `post-process/power-sampling/README.md`
- `post-process/freq-control/README.md`
- `post-process/gateway/llm-requests/README.md`
- `post-process/gateway/slo-aware-log/README.md`
- `post-process/prefill-concurrency/README.md`
- `post-process/gateway/stack/README.md`
- `post-process/gateway/stack-context/README.md`
- `post-process/gateway/stack-kv/README.md`
- `post-process/key-stats/README.md`
- `post-process/gateway/usage/README.md`
- `post-process/visualization/gateway-stack/README.md`
- `post-process/visualization/gateway-stack-context/README.md`
- `post-process/visualization/stacked-per-agent/README.md`
- `post-process/visualization/gateway-stack-kv/README.md`
- `post-process/visualization/gateway-slo-aware/README.md`
- `post-process/visualization/vllm-metrics/README.md`
- `post-process/visualization/power/README.md`
- `post-process/visualization/prefill-concurrency/README.md`
- `post-process/visualization/freq-control/README.md`

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
7. `request-throughput/extract_run.py`
8. `agent-output-throughput/extract_run.py`
9. `prefill-concurrency/extract_run.py`
10. `gateway/stack/extract_run.py`
11. `gateway/stack-context/extract_run.py`
12. `gateway/stack-kv/extract_run.py`
13. `gateway/usage/extract_run.py`
14. `gateway/ctx-aware-log/extract_run.py`
15. `gateway/slo-aware-log/extract_run.py`
16. `split/duration/extract_run.py`
17. `vllm-metrics/extract_run.py`
18. `vllm-metrics/summarize_timeseries.py`
19. `power/extract_run.py`
20. `power-sampling/extract_run.py`
21. `freq-control/extract_run.py`
22. `slo-decision/extract_run.py`
23. `key-stats/extract_run.py`
24. `visualization/job-throughput/generate_all_figures.py`
25. `visualization/request-throughput/generate_all_figures.py`
26. `visualization/agent-output-throughput/generate_all_figures.py`
27. `visualization/job-concurrency/generate_all_figures.py`
28. `visualization/prefill-concurrency/generate_all_figures.py`
29. `visualization/gateway-stack/generate_all_figures.py`
30. `visualization/gateway-stack-context/generate_all_figures.py`
31. `visualization/stacked-per-agent/generate_all_figures.py`
32. `visualization/gateway-ctx-aware/generate_all_figures.py`
33. `visualization/gateway-slo-aware/generate_all_figures.py`
34. `visualization/gateway-stack-kv/generate_all_figures.py`
35. `visualization/vllm-metrics/generate_all_figures.py`
36. `visualization/power/generate_all_figures.py`
37. `visualization/freq-control/generate_all_figures.py`
38. `visualization/slo-decision/generate_all_figures.py`
39. `global/aggregate_runs_csv.py` (root-dir mode only, skipped in dry-run)

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

### `agent-output-throughput`

- `post-process/agent-output-throughput/extract_run.py`
- purpose: compute per-agent output-token throughput from gateway request logs
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/agent-output-throughput/agent-output-throughput.json`

- `post-process/visualization/agent-output-throughput/generate_all_figures.py`
- purpose: render an output-throughput histogram and an output-tokens-vs-throughput scatter
  plot for each run
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/agent-output-throughput/`

### `gateway/slo-aware-log`

- `post-process/gateway/slo-aware-log/extract_run.py`
- purpose: extract event-driven gateway SLO-aware `ralexation` decisions from
  `gateway-output/job/slo_aware_decisions_*.jsonl`
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/gateway/slo-aware-log/slo-aware-events.json`

- `post-process/visualization/gateway-slo-aware/generate_all_figures.py`
- purpose: render a gateway SLO-aware event timeline showing throughput, slack,
  and `ralexation` duration around entry and wake decisions
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/gateway-slo-aware/`

### `slo-decision`

- `post-process/slo-decision/extract_run.py`
- purpose: extract SLO-triggered controller decisions from
  `freq-controller-ls.slo-decision.*.jsonl` and
  `freq-controller-ls-amd.slo-decision.*.jsonl` and
  `freq-controller-ls-instance-slo.slo-decision.*.jsonl`
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/slo-decision/slo-decision-summary.json`

- `post-process/visualization/slo-decision/generate_all_figures.py`
- purpose: render an SLO-decision timeline showing throughput-triggered control
  points and resulting frequency targets
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/slo-decision/`

- `post-process/freq-control/extract_run.py`
- purpose: extract freq-controller query, decision, and control-error history
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- default output:
- `<run-dir>/post-processed/<freq-control|freq-control-seg|freq-control-linespace|freq-control-linespace-instance-slo|freq-control-linespace-instance|freq-control-linespace-amd|freq-control-linespace-multi>/freq-control-summary.json`

- `post-process/visualization/freq-control/generate_all_figures.py`
- purpose: render a freq-control timeline showing context/query history,
  applied frequencies, and both query-read and control-write failures
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/<freq-control|freq-control-seg|freq-control-linespace|freq-control-linespace-instance-slo|freq-control-linespace-instance|freq-control-linespace-amd|freq-control-linespace-multi>/`

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

- `post-process/visualization/stacked-per-agent/generate_all_figures.py`
- purpose: materialize fixed-width per-agent context windows and render the
  publication-style stacked-per-agent bar chart
- supports:
- single run: `--run-dir <run-dir>`
- batch discovery: `--root-dir <root-dir>`
- parallel workers: `--max-procs`
- dry-run: `--dry-run`
- rendering controls:
- `--window-size-s`
- `--start-s`
- `--end-s`
- `--agent-order`
- `--value-mode`
- `--legend`
- `--legend-max-agents`
- `--title`
- `--format`
- `--dpi`
- default output:
- `<run-dir>/post-processed/visualization/stacked-per-agent/`

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
