This step collects the "headline" stats from the already-generated post-processed
JSON outputs and writes one consolidated summary JSON.

Default output:

```text
<run-dir>/post-processed/key-stats/key-stats.json
```

Commands:

```bash
python3 post-process/key-stats/extract_run.py \
  --run-dir <run-dir>
```

You can also point it at a derived directory such as `post-processed-50`:

```bash
python3 post-process/key-stats/extract_run.py \
  --post-processed-dir <run-dir>/post-processed-50
```

Batch mode:

```bash
python3 post-process/key-stats/extract_run.py \
  --root-dir <results-root> \
  --max-procs 8
```

It summarizes:

- vLLM KV-cache usage from `vllm-log/gauge-counter-timeseries.json`
  - specifically the `vllm:kv_cache_usage_perc` series
  - outputs `avg`, `min`, `max`, `std`
- `job-concurrency/job-concurrency-timeseries.json`
  - summarized from `concurrency_points[*].concurrency`
  - outputs `avg`, `min`, `max`, `std`
- `job-throughput/job-throughput-timeseries.json`
  - summarized from `throughput_points[*].throughput_jobs_per_s`
  - outputs `avg`, `min`, `max`, `std`
- `gateway/stack-kv/kv-usage-stacked-histogram.json`
  - summarized from `points[*].accumulated_value`
  - outputs `avg`, `min`, `max`, `std`
- `gateway/stack-context/context-usage-stacked-histogram.json`
  - summarized from `points[*].accumulated_value`
  - outputs `avg`, `min`, `max`, `std`
- all histogram outputs in `gateway/stack/`
  - `prompt_tokens`
  - `cached_tokens`
  - `compute_prompt_tokens`
  - `completion_tokens`
  - `compute_prompt_plus_completion_tokens`
  - each summarized from `points[*].accumulated_value`
  - outputs `avg`, `min`, `max`, `std`
- `gateway/llm-requests/llm-request-stats.json`
  - `average_stage_speed_tokens_per_s.prefill`
  - `average_stage_speed_tokens_per_s.decode`
  - these keep only `avg`, `min`, `max` plus the eligibility counts
- `gateway/llm-requests/llm-request-stats.json.metrics`
  - every metric keeps only `count`, `avg`, `min`, `max`

This step runs after the normal non-visual post-process calculations finish and
before visualization in `post-process/run_all.py`.
