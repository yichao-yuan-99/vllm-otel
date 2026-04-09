This figure overlays the KV-cache usage curves for 3 selected frequency runs in
one plot:

- `core-345-1680`
- `core-345-1185`
- `core-345-660`

The intended source sweep root is:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08`

This figure corresponds to the same underlying metric shown in per-run vLLM
metric figures such as:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260322T034746Z/core-345-810/post-processed/visualization/vllm-metrics/0008-vllm_kv_cache_usage_perc_engine_0_model_name_Qwen3-Coder-30B-A3B-Instruct-FP8.png`
- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260322T181625Z/core-345-660/post-processed/visualization/vllm-metrics/0008-vllm_kv_cache_usage_perc_engine_0_model_name_Qwen3-Coder-30B-A3B-Instruct-FP8.png`

This directory follows the top-level `figures/README.md` rule:

1. Step 1 materializes a figure-specific dataset into `data/`.
2. Step 2 renders the figure from that processed dataset only.

**Source Data**
For each selected run, the materialization step reads:

- `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.json`

and extracts the engine-0 series for:

- `vllm:kv_cache_usage_perc`

**Metric Definition**
The raw metric stored in the timeseries JSON is:

- `vllm:kv_cache_usage_perc`

Its raw values are fractions where:

- `1.0 = 100% KV-cache usage`
- `0.5 = 50% KV-cache usage`

This figure smooths each selected series with a centered 120-second window,
following the same convention used in the repo's other smoothed time-series
figures:

- `smoothed_value(t) = mean(raw samples with time in [t - 120s, t + 120s))`

The materialized dataset stores both:

- `raw_value`
- `value` / `smoothed_value`

The plotting step uses the smoothed values and renders them as percentages on
the y-axis.

When `--end-s` is omitted, this figure uses the latest time that is available in
all selected runs:

- `common_available_end_s = min(last_time_from_start_s across selected runs)`

That keeps the overlaid curves on a shared comparable time window.

**Step 1**
Materialize the selected runs into one processed JSON:

```bash
python3 figures/kv-usage-time/materialize_kv_usage_time.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08
```

Example with an explicit time window:

```bash
python3 figures/kv-usage-time/materialize_kv_usage_time.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08 \
  --start-s 0 \
  --end-s 8000
```

Example with an explicit smoothing half-window:

```bash
python3 figures/kv-usage-time/materialize_kv_usage_time.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08 \
  --smooth-window-s 120
```

Example with a different run order:

```bash
python3 figures/kv-usage-time/materialize_kv_usage_time.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/single-qps-sweep-freq-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08 \
  --run-slugs core-345-660 core-345-1185 core-345-1680
```

By default this writes a gitignored JSON like:

- `figures/kv-usage-time/data/kv-usage-time.smooth-120s.start-0.end-full.json`
- `figures/kv-usage-time/data/kv-usage-time.smooth-120s.start-0.end-8000.json`

Key fields in the materialized JSON:

- `analysis_window_start_s`
- `analysis_window_end_s`
- `common_available_end_s`
- `smooth_window_s`
- `metric_name`
- `series`
- `series[].series_label`
- `series[].stats`
- `series[].points`

**Step 2**
Render the combined KV-cache-usage figure from the materialized JSON:

```bash
python3 figures/kv-usage-time/plot_kv_usage_time.py \
  --input figures/kv-usage-time/data/kv-usage-time.smooth-120s.start-0.end-full.json
```

Example with a custom output path:

```bash
python3 figures/kv-usage-time/plot_kv_usage_time.py \
  --input figures/kv-usage-time/data/kv-usage-time.smooth-120s.start-0.end-8000.json \
  --output figures/kv-usage-time/output/kv-usage-time.smooth-120s.start-0.end-8000.pdf
```

The final figure:

- overlays the 3 selected runs as 3 smoothed line series
- uses time from start on the x-axis, rendered in minutes
- uses KV-cache usage percent on the y-axis

The default rendered output goes into the gitignored `output/` directory.

The plotting step requires `matplotlib` to be installed in the active Python
environment.
