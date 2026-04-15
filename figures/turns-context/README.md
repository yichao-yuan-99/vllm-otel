This figure compares the distribution of per-job maximum context usage across
six benchmark-agent combinations from:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b`

The rendered chart is a single six-column box-and-jitter plot, with one column
for each requested combination:

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

- `max_context_usage_per_job = job_max_request_lengths`
- fallback: `max_context_usage_per_job = jobs[].max_request_length`
- `max_request_length` is the max request length within one job
- `request_length = prompt_tokens + generation_tokens`
- each scatter point is one job's max request length
- each box summarizes the same per-job max-context distribution

Derived summary statistics stored in the materialized JSON:

- `sample_count = len(max_context_usages)`
- `mean = sum(max_context_usages) / sample_count`
- `median = quantile(max_context_usages, 0.5)`
- `q1 = quantile(max_context_usages, 0.25)`
- `q3 = quantile(max_context_usages, 0.75)`
- `p90 = quantile(max_context_usages, 0.90)`
- `p95 = quantile(max_context_usages, 0.95)`
- `iqr = q3 - q1`

The plotting step renders:

- white boxplots with dark outlines
- overlaid jittered points for every job
- one color per agent type
- a log-scale y-axis by default, because max-context distributions are
  strongly heavy-tailed

**Step 1**
Materialize the six benchmark-agent max-context distributions into `data/`:

```bash
python3 figures/turns-context/materialize_turns_context.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b
```

Optional explicit benchmark and agent selection:

```bash
python3 figures/turns-context/materialize_turns_context.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b \
  --benchmarks dabstep swebench-verified terminal-bench-2.0 \
  --agents mini-swe-agent terminus
```

By default this writes a gitignored JSON like:

- `figures/turns-context/data/turns-context.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json`

Important fields in the materialized JSON:

- `metric_definition`
- `benchmark_order`
- `agent_order`
- `panels`
- `panels[].max_context_usages`
- `panels[].stats`
- `panels[].run_dir`
- `panels[].summary_path`
- `panels[].candidate_run_count`

**Step 2**
Render the figure from the materialized JSON:

```bash
python3 figures/turns-context/plot_turns_context.py \
  --input figures/turns-context/data/turns-context.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json
```

Example with a custom output path and linear y-axis:

```bash
python3 figures/turns-context/plot_turns_context.py \
  --input figures/turns-context/data/turns-context.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json \
  --output figures/turns-context/output/turns-context.qwen3-coder-30b.linear.png \
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
