# Power Stats Aggregation For Frequency Sweep Runs

This directory aggregates metrics for replay results that already have `post-process/run_all.py` completed.

It can also read percent-selected outputs produced by `post-process-select/select_post_processed.py`.

Input assumptions by mode:

- `--mode freq` (default):
  - `--root-dir` contains one or more run directories named like `core-<min>-<max>` (or `core-<min>-<max>-mem-<mhz>`).
  - The minimum core frequency is fixed across runs (default expectation is `345 MHz`).
  - The sweep variable is the maximum core frequency, which is used as `frequency_mhz`.
- `--mode non-freq`:
  - run directory names do not need to follow `core-*`.
  - run directories are discovered by `post-processed/` layout and summary files.
  - this is useful for layouts like `sweep-qps-docker-power-clean`, where run directories are timestamps.

Each CSV row corresponds to one frequency run directory.

## Script

- `power-stats/aggregate_runs_csv.py`

## Output

By default, writes:

- `--mode freq`: `<root-dir>/power-stats/frequency-sweep-summary.csv`
- `--mode non-freq`: `<root-dir>/power-stats/run-summary.csv`

It also writes an avg-only companion CSV that removes `min`, `max`, and `std`
stat columns:

- `--mode freq`: `<root-dir>/power-stats/frequency-sweep-summary-avg-only.csv`
- `--mode non-freq`: `<root-dir>/power-stats/run-summary-avg-only.csv`

You can override with `--output`.
When `--output` is provided, the avg-only companion is written next to it using
the same filename plus `-avg-only` before the `.csv` suffix.

With `--percent X`, the script reads from `post-processed-<X>/` instead of `post-processed/`, and the default output directory becomes `power-stats-<X>/`.

Examples:

- `--percent 50` -> input from `post-processed-50/`, output under `power-stats-50/`
- `--percent 12.5` -> input from `post-processed-12_5/`, output under `power-stats-12_5/`

## Columns

The script extracts/derives these columns from `post-processed/` by default, or from `post-processed-<X>/` when `--percent X` is used:

- `avg_power_w`: from `post-processed/power/power-summary.json` -> `power_stats_w.avg`
- `std_power_w`: from `post-processed/power/power-summary.json` -> `power_stats_w.std` when present, otherwise derived from `power_points[].power_w`
- `avg_job_throughput_jobs_per_s`: average of `throughput_points[].throughput_jobs_per_s` from `post-processed/job-throughput/job-throughput-timeseries.json`
  - fallback if no points: `finished_replay_count_excluding_cancelled / total_duration_s`
- `std_job_throughput_jobs_per_s`: derived from `throughput_points[].throughput_jobs_per_s` when points exist
- `agent_total_time_avg_s`: from `post-processed/global/trial-timing-summary.json` -> `agent_time_breakdown_s.agent_total_time_stats_s.avg`
- `agent_total_time_std_s`: from `post-processed/global/trial-timing-summary.json` -> `agent_time_breakdown_s.agent_total_time_stats_s.std`
- `llm_time_avg_s`: from `post-processed/global/trial-timing-summary.json` -> `agent_time_breakdown_s.llm_time_stats_s.avg`
- `llm_time_std_s`: from `post-processed/global/trial-timing-summary.json` -> `agent_time_breakdown_s.llm_time_stats_s.std`
- `non_llm_time_avg_s`: from `post-processed/global/trial-timing-summary.json` -> `agent_time_breakdown_s.non_llm_time_stats_s.avg`
- `non_llm_time_std_s`: from `post-processed/global/trial-timing-summary.json` -> `agent_time_breakdown_s.non_llm_time_stats_s.std`
- `prefill_avg_tokens_per_s`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `average_stage_speed_tokens_per_s.prefill.avg_tokens_per_s`
- `prefill_std_tokens_per_s`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `average_stage_speed_tokens_per_s.prefill.std_tokens_per_s` (or `.std` if present)
- `decode_avg_tokens_per_s`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `average_stage_speed_tokens_per_s.decode.avg_tokens_per_s`
- `decode_std_tokens_per_s`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `average_stage_speed_tokens_per_s.decode.std_tokens_per_s` (or `.std` if present)
- `completion_tokens_avg`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `metrics.completion_tokens.avg`
- `completion_tokens_std`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `metrics.completion_tokens.std` (if present)
- `cached_tokens_avg`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `metrics.cached_tokens.avg`
- `cached_tokens_std`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `metrics.cached_tokens.std` (if present)
- `duration_ms_avg`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `metrics.duration_ms.avg`
- `duration_ms_std`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `metrics.duration_ms.std` (if present)
- `prompt_tokens_avg`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `metrics.prompt_tokens.avg`
- `prompt_tokens_std`: from `post-processed/gateway/llm-requests/llm-request-stats.json` -> `metrics.prompt_tokens.std` (if present)

The full CSV now also flattens numeric values from:

- `post-processed/key-stats/key-stats.json`

Each numeric leaf becomes one CSV column with a `key_stats_` prefix.
Examples:

- `key_stats_job_concurrency_avg`
- `key_stats_job_concurrency_min`
- `key_stats_job_concurrency_max`
- `key_stats_job_concurrency_std`
- `key_stats_vllm_metrics_kv_cache_usage_perc_avg`
- `key_stats_gateway_stack_prompt_tokens_avg`
- `key_stats_gateway_llm_requests_metrics_gen_ai_latency_time_in_queue_avg`

The avg-only CSV keeps the same metadata and average/count-style columns, but
removes columns whose stat suffix is `min`, `max`, or `std`.

The CSV also includes:

- `run_path`
- `frequency_mhz`
- `core_min_mhz`
- `core_max_mhz`
- `mem_freq_mhz`

In `--mode non-freq`, frequency-related columns are blank unless the run directory name still matches `core-<min>-<max>(-mem-<mhz>)`.

## Usage

```bash
python3 power-stats/aggregate_runs_csv.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08
```

Non-frequency mode example (`sweep-qps-docker-power-clean` style):

```bash
python3 power-stats/aggregate_runs_csv.py \
  --mode non-freq \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/exclude-unranked
```

Selected-window example:

```bash
python3 power-stats/aggregate_runs_csv.py \
  --mode non-freq \
  --root-dir <root-dir> \
  --percent 50
```

If your sweep uses a different fixed minimum core clock:

```bash
python3 power-stats/aggregate_runs_csv.py \
  --root-dir <root-dir> \
  --expected-core-min-mhz 500
```

If you intentionally want repeated `frequency_mhz` values from multiple runs:

```bash
python3 power-stats/aggregate_runs_csv.py \
  --root-dir <root-dir> \
  --allow-duplicate-frequency
```
