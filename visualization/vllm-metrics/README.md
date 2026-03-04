This directory contains run-level PDF plots derived from:

- `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.json`

The directory is now flat: each plotter lives directly under
`visualization/vllm-metrics/`, and `common/` contains the shared PDF helpers.

## Plots

- `plot_kv_cache_usage.py`
  - metric: `vllm:kv_cache_usage_perc`
  - outputs: `kv_cache_usage/kv_cache_usage.pdf`, `kv_cache_usage/kv_cache_usage.json`
- `plot_num_requests_running.py`
  - metric: `vllm:num_requests_running`
  - outputs: `num_requests_running/num_requests_running.pdf`, `num_requests_running/num_requests_running.json`
- `plot_num_requests_waiting.py`
  - metric: `vllm:num_requests_waiting`
  - outputs: `num_requests_waiting/num_requests_waiting.pdf`, `num_requests_waiting/num_requests_waiting.json`
- `plot_generation_tokens.py`
  - metric: `vllm:generation_tokens`
  - outputs: `generation_tokens.raw.pdf`, `generation_tokens.raw.json`, `generation_tokens.avg5.pdf`, `generation_tokens.avg5.json`
- `plot_num_preemptions.py`
  - metric: `vllm:num_preemptions`
  - outputs: `num_preemptions.raw.pdf`, `num_preemptions.raw.json`, `num_preemptions.avg5.pdf`, `num_preemptions.avg5.json`
- `plot_prefill_computed_tokens.py`
  - metric: `vllm:prompt_tokens - vllm:prefix_cache_hits`
  - outputs: `prefill_computed_tokens.raw.pdf`, `prefill_computed_tokens.raw.json`, `prefill_computed_tokens.avg5.pdf`, `prefill_computed_tokens.avg5.json`

`plot_generation_tokens.py`, `plot_num_preemptions.py`, and
`plot_prefill_computed_tokens.py` each generate both:

- `raw`
  - one sample per point
- `avg5`
  - non-overlapping average of 5 samples per point

## Entry Script

Use `generate_all_figures.py` to produce the full figure set for one run:

```bash
python visualization/vllm-metrics/generate_all_figures.py \
  --run-dir tests/output/con-driver/job-20260301T014756Z
```

That runs all six plotters and writes:

- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/kv_cache_usage/kv_cache_usage.pdf`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/kv_cache_usage/kv_cache_usage.json`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/num_requests_running/num_requests_running.pdf`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/num_requests_running/num_requests_running.json`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/num_requests_waiting/num_requests_waiting.pdf`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/num_requests_waiting/num_requests_waiting.json`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/generation_tokens/generation_tokens.raw.pdf`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/generation_tokens/generation_tokens.raw.json`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/generation_tokens/generation_tokens.avg5.pdf`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/generation_tokens/generation_tokens.avg5.json`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/num_preemptions/num_preemptions.raw.pdf`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/num_preemptions/num_preemptions.raw.json`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/num_preemptions/num_preemptions.avg5.pdf`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/num_preemptions/num_preemptions.avg5.json`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/prefill_computed_tokens/prefill_computed_tokens.raw.pdf`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/prefill_computed_tokens/prefill_computed_tokens.raw.json`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/prefill_computed_tokens/prefill_computed_tokens.avg5.pdf`
- `tests/output/con-driver/job-20260301T014756Z/visualization/vllm-log/prefill_computed_tokens/prefill_computed_tokens.avg5.json`

You can also run an individual plotter directly. Example:

```bash
python visualization/vllm-metrics/plot_num_requests_running.py \
  --run-dir tests/output/con-driver/job-20260301T014756Z
```

## Notes

- All scripts read the already extracted gauge/counter timeseries rather than raw
  `.tar.gz` vLLM logs.
- Each figure also writes a curated JSON payload with the exact series used to
  render that figure.
- The plotting code writes PDF directly and does not depend on `matplotlib`.
- If multiple label-distinct series exist for a metric, they are plotted together
  with a legend.
