# Global Post-Process

This directory contains global post-process scripts for run-level summaries.

## Index

- `extract_run.py`: extract one run's global trial timing summary
- `aggregate_runs_csv.py`: aggregate many run summaries into one CSV

## `extract_run.py`

Path:

- `post-process/global/extract_run.py`

### Single Run Command

Input:

- one run directory from either:
  - `con-driver` output (`meta/results.json`, `meta/run_manifest.json`)
  - `replay` output (`replay/summary.json`)

Command:

```bash
python post-process/global/extract_run.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/post-processed/global/trial-timing-summary.json
```

Optional output path:

```bash
python post-process/global/extract_run.py \
  --run-dir <run-dir> \
  --output tmp/trial-timing-summary.json
```

### Batch Command

Command:

```bash
python post-process/global/extract_run.py \
  --root-dir results/replay
```

Discovery rule:

- any subdirectory that has either:
  - `replay/summary.json`
  - `meta/results.json` and `meta/run_manifest.json`

Optional worker count:

```bash
python post-process/global/extract_run.py \
  --root-dir results/replay \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/global/extract_run.py \
  --root-dir results/replay \
  --dry-run
```

Output includes:

- total duration of the experiment (`total_duration_s`)
- trial count (`trial_count`; alias: `trail_count`)
- avg/min/max of trial durations (`trial_duration_stats_s`)
- per-agent execution breakdown (`agent_time_breakdown_s`) with:
  - per-agent `llm_time_s`, `non_llm_time_s`, `agent_total_time_s`
  - aggregate sums and `avg/min/max/std` stats for each time bucket
- per-trial list with:
  - `start_offset_s`
  - `end_offset_s`
  - `duration_s`

## `aggregate_runs_csv.py`

Path:

- `post-process/global/aggregate_runs_csv.py`

### Command

Input:

- one root directory that contains many run result directories
- it scans for:
  - `<run-dir>/post-processed/global/trial-timing-summary.json`
  - `<run-dir>/post-processed/gateway/usage/usage-summary.json` (optional)
  - `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.stats.json` (optional)

Command:

```bash
python post-process/global/aggregate_runs_csv.py \
  --root-dir <root-dir>
```

Default output:

```text
<root-dir>/trial-timing-summary.csv
```

Optional output path:

```bash
python post-process/global/aggregate_runs_csv.py \
  --root-dir <root-dir> \
  --output tmp/trial-timing-summary.csv
```

CSV rows and columns:

- each row is one run (sorted by path relative to `--root-dir`)
- row key column: `run_path`
- metric columns:
  - `total_duration_s`
  - `trial_duration_avg_s`
  - `trial_duration_min_s`
  - `trial_duration_max_s`
  - `prompt_tokens`
  - `generation_tokens`
  - `cached_prompt_tokens`
  - `prefill_prompt_tokens`
  - `avg_worker_max_request_length`
  - `vllm:kv_cache_usage_perc:avg`
  - `vllm:kv_cache_usage_perc:min`
  - `vllm:kv_cache_usage_perc:max`
  - `vllm:num_requests_running:avg`
  - `vllm:num_requests_running:min`
  - `vllm:num_requests_running:max`
  - `vllm:num_requests_waiting:avg`
  - `vllm:num_requests_waiting:min`
  - `vllm:num_requests_waiting:max`

Notes:

- the `vllm:*:{avg|min|max}` columns are read from matching `metrics[*]` fields
  in `gauge-counter-timeseries.stats.json`
- token usage columns are read from `usage.*` fields in `usage-summary.json`
- `avg_worker_max_request_length` comes from
  `usage.avg_worker_max_request_length`
- if `usage.generation_tokens` is missing, `usage.completion_tokens` is used
  as fallback
- if `usage.prefill_prompt_tokens` is missing, it is computed from
  `prompt_tokens - cached_prompt_tokens` when both are available
- if multiple series share the same metric name (for example different labels),
  the `:avg` value is a sample-count-weighted average across those series
- when multiple series share the same metric name, `:min` is the minimum across
  matching series and `:max` is the maximum across matching series
- if the gateway usage file is missing for a run, token usage columns are left
  empty
- if the vLLM stats file is missing for a run, those columns are left empty
