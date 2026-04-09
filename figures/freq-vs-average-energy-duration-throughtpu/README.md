This figure plots GPU core frequency against three windowed metrics using 2
subplots:

- Top subplot: average energy cost per finished replay.
- Bottom subplot left y-axis: average throughput.
- Bottom subplot right y-axis: average time spent in LLM.

The intended source data is:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08`

`start` and `end` are seconds from experiment start. For example, `--start-s 5000 --end-s 8000` means "only use data in the 5000s to 8000s window". If `--end-s` is omitted, each run uses its full post-processed analysis window.

This directory follows the top-level `figures/README.md` rule:

1. Step 1 materializes a figure-specific dataset into `data/`.
2. Step 2 renders the figure from that processed dataset only.

**Metric Definition**
This figure intentionally follows the non-integral energy formula from the
original requirement and adds a request-level LLM-time metric:

- `average_throughput = finished_replay_count_in_window / window_duration_s`
- `average_energy_cost = average_power_in_window * window_duration_s / finished_replay_count_in_window`
- `average_time_spent_in_llm = mean(gen_ai.latency.time_in_model_inference for requests completed in the window with a numeric value)`

where:

- `average_power_in_window` is the arithmetic mean of the sampled `power_w` values inside the selected window from `post-processed/power/power-summary.json`
- `finished_replay_count_in_window` is counted directly from `replay/summary.json` using worker `finished_at` timestamps that land inside the selected window and inside the run's analyzed duration
- `average_time_spent_in_llm` is computed from `post-processed/gateway/llm-requests/llm-requests.json` using requests whose `request_end_offset_s` lands inside the selected window

The materialized CSV also stores the trapezoidal integral energy for the same window as a reference column, but this plot uses the average-power-times-duration metric above. The sibling `freq-vs-average-energy-integral/` directory can use the integral-based metric instead.

**Step 1**
Materialize the processed data into `data/`:

```bash
python3 figures/freq-vs-average-energy-duration-throughtpu/materialize_windowed_metrics.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08
```

Example with an explicit window:

```bash
python3 figures/freq-vs-average-energy-duration-throughtpu/materialize_windowed_metrics.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08 \
  --start-s 5000 \
  --end-s 8000
```

By default this writes a gitignored CSV like:

- `figures/freq-vs-average-energy-duration-throughtpu/data/freq-vs-average-energy.start-0.end-full.csv`
- `figures/freq-vs-average-energy-duration-throughtpu/data/freq-vs-average-energy.start-5000.end-8000.csv`

Key columns in the CSV:

- `frequency_mhz`
- `window_avg_power_w`
- `window_energy_estimate_j`
- `window_energy_integral_j`
- `finished_replay_count_in_window`
- `average_throughput_jobs_per_s`
- `llm_request_count_in_window`
- `llm_request_with_model_inference_time_count_in_window`
- `average_request_time_in_llm_s`
- `average_energy_per_finished_replay_j`

**Step 2**
Render the 2-subplot figure from the materialized CSV:

```bash
python3 figures/freq-vs-average-energy-duration-throughtpu/plot_freq_vs_average_energy.py \
  --input figures/freq-vs-average-energy-duration-throughtpu/data/freq-vs-average-energy.start-0.end-full.csv
```

Example with a custom output path:

```bash
python3 figures/freq-vs-average-energy-duration-throughtpu/plot_freq_vs_average_energy.py \
  --input figures/freq-vs-average-energy-duration-throughtpu/data/freq-vs-average-energy.start-5000.end-8000.csv \
  --output figures/freq-vs-average-energy-duration-throughtpu/output/freq-vs-average-energy.start-5000.end-8000.pdf
```

The default rendered output goes into the gitignored `output/` directory.

The plotting step requires `matplotlib` to be installed in the active Python environment.
