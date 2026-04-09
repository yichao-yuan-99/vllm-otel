This figure turns the gateway stacked context-usage timeline into a stacked bar
chart with one contiguous bar every 120 seconds.

It is intended as a more structured version of:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260321T143624Z/post-processed/visualization/gateway-stack-context/context-usage-stacked-histogram-smoothed-120s.png`

The intended source run is:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260321T143624Z`

This directory follows the top-level `figures/README.md` rule:

1. Step 1 materializes a figure-specific dataset into `data/`.
2. Step 2 renders the figure from that processed dataset only.

**Source Data**
The materialization step reads:

- `<run-dir>/post-processed/gateway/stack-context/context-usage-ranges.json`

That file already contains per-agent active segments recovered from gateway
request history.

**Metric Definition**
For each active range entry in `context-usage-ranges.json`:

- `context_usage_tokens = prompt_tokens + completion_tokens`
- that value is held from `range_start_s` to `range_end_s`
- idle or finished periods contribute `0`, so they do not create a visible stack
  layer

For each bar window `[window_start_s, window_end_s)`:

- `overlap_duration_s = overlap(active_range, bar_window)`
- `agent_window_integral_value = sum(avg_value_per_s * overlap_duration_s)`
- `agent_window_average_value = agent_window_integral_value / window_duration_s`

This figure plots `agent_window_average_value` by default, so the y-axis stays on
the same scale as the original smoothed context-usage line chart.

Agent layers use:

- a fixed order across all bars
- default ordering: first active time, then `agent_key`
- a stable color assignment derived from that fixed order

**Step 1**
Materialize 120-second stacked-bar windows into `data/`:

```bash
python3 figures/stacked-per-agent/materialize_stacked_per_agent.py \
  --run-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260321T143624Z
```

Optional analysis window:

```bash
python3 figures/stacked-per-agent/materialize_stacked_per_agent.py \
  --run-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260321T143624Z \
  --start-s 1200 \
  --end-s 4800
```

Optional custom bar width:

```bash
python3 figures/stacked-per-agent/materialize_stacked_per_agent.py \
  --run-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260321T143624Z \
  --window-size-s 60
```

By default this writes a gitignored JSON like:

- `figures/stacked-per-agent/data/stacked-per-agent.window-120s.start-0.end-full.json`
- `figures/stacked-per-agent/data/stacked-per-agent.window-120s.start-1200.end-4800.json`

Important fields in the materialized JSON:

- `agents`
- `windows`
- `window_size_s`
- `analysis_window_start_s`
- `analysis_window_end_s`
- `windows[].contributions[].average_value`
- `windows[].contributions[].integral_value`

**Step 2**
Render the stacked bar chart from the materialized JSON:

```bash
python3 figures/stacked-per-agent/plot_stacked_per_agent.py \
  --input figures/stacked-per-agent/data/stacked-per-agent.window-120s.start-0.end-full.json
```

Example with a custom output path:

```bash
python3 figures/stacked-per-agent/plot_stacked_per_agent.py \
  --input figures/stacked-per-agent/data/stacked-per-agent.window-120s.start-1200.end-4800.json \
  --output figures/stacked-per-agent/output/stacked-per-agent.window-120s.start-1200.end-4800.png
```

Useful plotting options:

- `--value-mode average` to match the original smoothed line-chart scale
- `--value-mode integral` to plot token-seconds per bar instead
- `--legend auto|show|hide`

The default rendered output goes into the gitignored `output/` directory.

The plotting step requires `matplotlib` to be installed in the active Python
environment.
