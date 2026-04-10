This figure overlays the agent-output-throughput histograms for the two runs
originally referenced in this directory:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260321T143624Z/post-processed/visualization/agent-output-throughput/agent-output-throughput-histogram.png`
- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260403T004347Z/post-processed/visualization/agent-output-throughput/agent-output-throughput-histogram.png`

Instead of alpha-compositing the rendered PNGs, the implementation rebuilds a
shared-bin histogram from the underlying machine-readable throughput summaries
for those same runs. That guarantees the two series use the exact same x-axis
bin edges and y-axis scale, while still rendering each run with a distinct
transparent color.

This directory follows the top-level `figures/README.md` rule:

1. Step 1 materializes a figure-specific dataset into `data/`.
2. Step 2 renders the figure from that processed dataset only.

**Source Data**
The plotting target is the pair of visualization PNGs above, but the actual
materialization step reads the corresponding:

- `<run-dir>/post-processed/agent-output-throughput/agent-output-throughput.json`

For the default figure, those JSON inputs are:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260321T143624Z/post-processed/agent-output-throughput/agent-output-throughput.json`
- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260403T004347Z/post-processed/agent-output-throughput/agent-output-throughput.json`

`materialize_tmp_overlap.py` accepts any of the following for each source:

- a run directory
- an `agent-output-throughput.json` file
- the already-rendered histogram PNG path from `post-processed/visualization/agent-output-throughput/`

If you pass the PNG path, the script resolves it back to the sibling
`agent-output-throughput.json` automatically.

**Metric Definition**
For each agent in `agent-output-throughput.json`:

- `output_throughput_tokens_per_s = output_tokens / llm_request_duration_s`

If an agent already has `output_throughput_tokens_per_s`, that value is used
directly. Otherwise it is recomputed from `output_tokens` and
`llm_request_duration_s` when possible.

The overlay histogram uses one shared bin grid for both runs:

- `bin_index = floor(output_throughput_tokens_per_s / bin_size)`
- `bin_start = bin_index * bin_size`
- `bin_end = (bin_index + 1) * bin_size`
- `bin_count = count(agents whose throughput falls in [bin_start, bin_end))`

By default, `bin_size` is inferred from the source histogram metadata. If both
inputs provide a bin size and they disagree, pass `--bin-size` explicitly.

The rendered plot uses:

- one transparent bar series per run
- the same x-axis bin edges for both runs
- the same y-axis count scale for both runs
- one dashed mean-throughput line per run

**Step 1**
Materialize the default overlap dataset into `data/`:

```bash
python3 figures/tmp-overlap/materialize_tmp_overlap.py
```

Equivalent explicit input selection using the original PNG paths:

```bash
python3 figures/tmp-overlap/materialize_tmp_overlap.py \
  --source-a /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260321T143624Z/post-processed/visualization/agent-output-throughput/agent-output-throughput-histogram.png \
  --source-b /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_08/20260403T004347Z/post-processed/visualization/agent-output-throughput/agent-output-throughput-histogram.png
```

Optional label and bin-size override:

```bash
python3 figures/tmp-overlap/materialize_tmp_overlap.py \
  --label-a "docker-power-clean" \
  --label-b "freq-ctrl-linespace" \
  --bin-size 1.0
```

By default this writes a gitignored JSON like:

- `figures/tmp-overlap/data/tmp-overlap.docker-power-clean-vs-docker-power-clean-freq-ctrl-linespace.json`

Important fields in the materialized JSON:

- `metric_definition`
- `selection_policy`
- `bin_size`
- `shared_histogram`
- `datasets`
- `datasets[].resolved_input_path`
- `datasets[].summary`
- `datasets[].histogram`

**Step 2**
Render the overlaid histogram from the materialized JSON:

```bash
python3 figures/tmp-overlap/plot_tmp_overlap.py \
  --input figures/tmp-overlap/data/tmp-overlap.docker-power-clean-vs-docker-power-clean-freq-ctrl-linespace.json
```

Example with a custom output path and title:

```bash
python3 figures/tmp-overlap/plot_tmp_overlap.py \
  --input figures/tmp-overlap/data/tmp-overlap.docker-power-clean-vs-docker-power-clean-freq-ctrl-linespace.json \
  --output figures/tmp-overlap/output/tmp-overlap.qps0_08.png \
  --title "Agent Output Throughput Overlay at QPS 0.08"
```

Useful plotting options:

- `--title <string>`
- `--output <path>`
- `--figure-width <float>`
- `--figure-height <float>`
- `--alpha <float>`
- `--dpi <int>`

The default rendered output goes into the gitignored `output/` directory.

The plotting step requires `matplotlib` to be installed in the active Python
environment.
