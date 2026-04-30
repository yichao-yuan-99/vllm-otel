This figure combines the existing turn-count and agent-duration violin plots
into one transposed comparison for six benchmark-agent combinations from:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b`

The rendered chart is a single two-panel horizontal violin-plus-boxplot figure:

- left panel: per-job turn counts
- right panel: per-job agent durations in seconds
- shared rows: one row per benchmark-agent combination

By default the shared rows are:

- `dabstep` x `mini-swe-agent`
- `dabstep` x `terminus-2`
- `swebench-verified` x `mini-swe-agent`
- `swebench-verified` x `terminus-2`
- `terminal-bench-2.0` x `mini-swe-agent`
- `terminal-bench-2.0` x `terminus-2`

This directory follows the top-level `figures/README.md` rule:

1. Step 1 materializes a figure-specific dataset into `data/`.
2. Step 2 renders the figure from that processed dataset only.

**Goal**
This figure is the transposed combination of:

- `figures/turns-scatter-volin/output/turns-scatter-volin.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.pdf`
- `figures/turns-duration-violin/output/turns-duration-violin.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.pdf`

Here, "transposed" means the data runs in the horizontal direction: each
distribution is drawn as a horizontal violin and horizontal boxplot instead of a
vertical column.

**Source Data**
The materialization step chooses the latest available run for each selected
benchmark-agent pair using:

- `<root-dir>/<benchmark>/<agent>/<run-id>/run-stats/run-stats-summary.json`

For the default figure, `root-dir` is:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b`

After selecting a run, the pipeline uses the same underlying raw artifacts as
the existing source figures:

- turn counts from `run-stats/run-stats-summary.json`
- durations from the matching `gateway-output/...` job artifacts for the same run

If multiple run directories exist for the same benchmark-agent pair, the
materializer selects the latest one by lexicographic `run-id` order.

**Metric Definitions**
Turn metric:

- `turns_per_job = jobs[].request_count`
- each raw point is one job's `request_count`
- each violin and box summarize that same per-job turn-count distribution

Duration metric:

- `agent_duration_s = agent_end - agent_start` from `events/lifecycle.jsonl`
- fallback: `job_end - job_start` from the same lifecycle file
- fallback: `run_end_time - run_start_time` from `manifest.json`
- fallback: `max(request_end_time) - min(request_start_time)` from `requests/model_inference.jsonl`
- if a service-failure cutoff is detected for the run, end timestamps are
  clamped to that cutoff before duration calculation

For both metrics, the materialized JSON stores derived summary statistics:

- `sample_count = len(values)`
- `mean = sum(values) / sample_count`
- `median = quantile(values, 0.5)`
- `q1 = quantile(values, 0.25)`
- `q3 = quantile(values, 0.75)`
- `p90 = quantile(values, 0.90)`
- `p95 = quantile(values, 0.95)`
- `iqr = q3 - q1`

The plotting step renders, for both panels:

- filled violin bodies colored by agent type
- narrower white boxplots with dark outlines over the violins
- jittered raw points for every job
- a pink diamond at the panel mean
- log-scale x-axes by default

**Step 1**
Materialize the combined turns-and-duration dataset into `data/`:

```bash
python3 figures/turns-scatter-duration-violin/materialize_turns_scatter_duration_violin.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b
```

Optional explicit benchmark and agent selection:

```bash
python3 figures/turns-scatter-duration-violin/materialize_turns_scatter_duration_violin.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b \
  --benchmarks dabstep swebench-verified terminal-bench-2.0 \
  --agents mini-swe-agent terminus
```

By default this writes a gitignored JSON like:

- `figures/turns-scatter-duration-violin/data/turns-scatter-duration-violin.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json`

Important fields in the materialized JSON:

- `turn_metric_definition`
- `duration_metric_definition`
- `duration_source_priority`
- `benchmark_order`
- `agent_order`
- `panels`
- `panels[].turns`
- `panels[].turn_stats`
- `panels[].durations_s`
- `panels[].duration_stats`
- `panels[].duration_source_counts`
- `panels[].run_dir`
- `panels[].summary_path`
- `panels[].candidate_run_count`

**Step 2**
Render the figure from the materialized JSON:

```bash
python3 figures/turns-scatter-duration-violin/plot_turns_scatter_duration_violin.py \
  --input figures/turns-scatter-duration-violin/data/turns-scatter-duration-violin.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json
```

Example with a custom output path and linear axes:

```bash
python3 figures/turns-scatter-duration-violin/plot_turns_scatter_duration_violin.py \
  --input figures/turns-scatter-duration-violin/data/turns-scatter-duration-violin.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json \
  --output figures/turns-scatter-duration-violin/output/turns-scatter-duration-violin.qwen3-coder-30b.linear.png \
  --turn-scale linear \
  --duration-scale linear
```

Useful plotting options:

- `--turn-scale log|linear`
- `--duration-scale log|linear`
- `--jitter-height <float>`
- `--title <string>`
- `--output <path>`

The default rendered output goes into the gitignored `output/` directory.

The plotting step requires `matplotlib` to be installed in the active Python
environment.
