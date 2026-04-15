This figure compares the distribution of per-job agent durations across six
benchmark-agent combinations from:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b`

The rendered chart is a single six-column box-and-jitter plot in the style of
`figures/turns-scatter`, with one column for each requested combination:

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
The materialization step chooses the latest available run for each selected
benchmark-agent pair using:

- `<root-dir>/<benchmark>/<agent>/<run-id>/run-stats/run-stats-summary.json`

For the default figure, `root-dir` is:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b`

Example selection file:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b/terminal-bench-2.0/terminus-2/terminal-bench@2.0-20260306T174037Z/run-stats/run-stats-summary.json`

After selecting a run, the materializer resolves each `jobs[].gateway_run_id`
back to the corresponding gateway artifact directory under:

- `<run-dir>/gateway-output/profile-<id>/<gateway_run_id>/`
- fallback: `<run-dir>/gateway-output/<gateway_run_id>/`

If multiple run directories exist for the same benchmark-agent pair, the
materializer selects the latest one by lexicographic `run-id` order, matching
`figures/turns-scatter`.

**Metric Definition**
For each selected job, the materializer computes:

- `agent_duration_s = agent_end - agent_start` from `events/lifecycle.jsonl`
- fallback: `job_end - job_start` from the same lifecycle file
- fallback: `run_end_time - run_start_time` from `manifest.json`
- fallback: `max(request_end_time) - min(request_start_time)` from `requests/model_inference.jsonl`

Additional rule:

- if a service-failure cutoff is detected for the run, end timestamps are
  clamped to that cutoff before duration calculation

Each scatter point is one job's duration in seconds, and each box summarizes the
same per-job duration distribution.

Derived summary statistics stored in the materialized JSON:

- `sample_count = len(durations_s)`
- `mean = sum(durations_s) / sample_count`
- `median = quantile(durations_s, 0.5)`
- `q1 = quantile(durations_s, 0.25)`
- `q3 = quantile(durations_s, 0.75)`
- `p90 = quantile(durations_s, 0.90)`
- `p95 = quantile(durations_s, 0.95)`
- `iqr = q3 - q1`

The plotting step renders:

- white boxplots with dark outlines
- overlaid jittered points for every job
- one color per agent type
- a log-scale y-axis by default, because the duration distributions are also
  heavy-tailed in practice

**Step 1**
Materialize the six benchmark-agent duration distributions into `data/`:

```bash
python3 figures/turns-duration/materialize_turns_duration.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b
```

Optional explicit benchmark and agent selection:

```bash
python3 figures/turns-duration/materialize_turns_duration.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b \
  --benchmarks dabstep swebench-verified terminal-bench-2.0 \
  --agents mini-swe-agent terminus
```

By default this writes a gitignored JSON like:

- `figures/turns-duration/data/turns-duration.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json`

Important fields in the materialized JSON:

- `metric_definition`
- `duration_source_priority`
- `benchmark_order`
- `agent_order`
- `panels`
- `panels[].durations_s`
- `panels[].stats`
- `panels[].duration_source_counts`
- `panels[].run_dir`
- `panels[].summary_path`
- `panels[].candidate_run_count`

**Step 2**
Render the figure from the materialized JSON:

```bash
python3 figures/turns-duration/plot_turns_duration.py \
  --input figures/turns-duration/data/turns-duration.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json
```

Example with a custom output path and linear y-axis:

```bash
python3 figures/turns-duration/plot_turns_duration.py \
  --input figures/turns-duration/data/turns-duration.qwen3-coder-30b.benchmarks-dabstep_swebench-verified_terminal-bench-2-0.agents-mini-swe-agent_terminus-2.json \
  --output figures/turns-duration/output/turns-duration.qwen3-coder-30b.linear.png \
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
