This figure compares the distribution of per-job turn counts across six
benchmark-agent combinations from:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b`

The rendered chart is a single six-column violin-plus-boxplot figure, with one
column for each requested combination:

- `dabstep` x `mini-swe-agent`
- `dabstep` x `terminus-2`
- `swebench-verified` x `mini-swe-agent`
- `swebench-verified` x `terminus-2`
- `terminal-bench-2.0` x `mini-swe-agent`
- `terminal-bench-2.0` x `terminus-2`

This directory follows the top-level `figures/README.md` rule:

1. Step 1 materializes a figure-specific dataset into `data/`.
2. Step 2 renders the figure from that processed dataset only.

**Source Data**
The materialization step scans the latest available run for each selected
benchmark-agent pair under:

- `<root-dir>/<benchmark>/<agent>/<run-id>/run-stats/run-stats-summary.json`

For the default figure, `root-dir` is:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b`

Example source file:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b/terminal-bench-2.0/terminus-2/terminal-bench@2.0-20260306T174037Z/run-stats/run-stats-summary.json`

If multiple run directories exist for the same benchmark-agent pair, the
materializer selects the latest one by lexicographic `run-id` order.

**Metric Definition**
For each selected `run-stats-summary.json`:

- `turns_per_job = jobs[].request_count`
- each raw point is one job's `request_count`
- each violin shows a kernel-density estimate over the same per-job values
- each overlaid box summarizes the same per-job turn-count distribution
- each diamond mark is the arithmetic mean of that panel's `turns`

Derived summary statistics stored in the materialized JSON:

- `sample_count = len(turns)`
- `mean = sum(turns) / sample_count`
- `median = quantile(turns, 0.5)`
- `q1 = quantile(turns, 0.25)`
- `q3 = quantile(turns, 0.75)`
- `p90 = quantile(turns, 0.90)`
- `p95 = quantile(turns, 0.95)`
- `iqr = q3 - q1`

The plotting step renders:

- filled violin bodies colored by agent type
- narrower white boxplots with dark outlines over the violins
- overlaid jittered raw points for every job
- a pink diamond at the panel mean
- a log-scale y-axis by default, because turn-count distributions are strongly
  heavy-tailed

This plot style follows the same basic Matplotlib violin pattern documented at:

- `https://matplotlib.org/stable/plot_types/stats/violin.html`

**Step 1**
Materialize the six benchmark-agent turn distributions into `data/`:

```bash
python3 figures/turns-scatter-volin/materialize_turns_scatter_volin.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b
```

Optional explicit benchmark and agent selection:

```bash
python3 figures/turns-scatter-volin/materialize_turns_scatter_volin.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b \
  --benchmarks dabstep swebench-verified terminal-bench-2.0 \
  --agents mini-swe-agent terminus
```

By default this writes a gitignored JSON like:

- `figures/turns-scatter-volin/data/turns-scatter-volin.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json`

Important fields in the materialized JSON:

- `metric_definition`
- `benchmark_order`
- `agent_order`
- `panels`
- `panels[].turns`
- `panels[].stats`
- `panels[].run_dir`
- `panels[].summary_path`
- `panels[].candidate_run_count`

**Step 2**
Render the figure from the materialized JSON:

```bash
python3 figures/turns-scatter-volin/plot_turns_scatter_volin.py \
  --input figures/turns-scatter-volin/data/turns-scatter-volin.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json
```

Example with a custom output path and linear y-axis:

```bash
python3 figures/turns-scatter-volin/plot_turns_scatter_volin.py \
  --input figures/turns-scatter-volin/data/turns-scatter-volin.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json \
  --output figures/turns-scatter-volin/output/turns-scatter-volin.qwen3-coder-30b.linear.png \
  --y-scale linear
```

Useful plotting options:

- `--y-scale log|linear`
- `--jitter-width <float>`
- `--title <string>`
- `--output <path>`

The default rendered output goes into the gitignored `output/` directory.

The plotting step requires `matplotlib` to be installed in the active Python
environment.
