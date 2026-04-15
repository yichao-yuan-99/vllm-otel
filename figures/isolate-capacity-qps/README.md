This figure renders a 1 row x 3 column bar-chart comparison for three explicit
`mini-swe-agent` source inputs.

The default source inputs, in plotting order, are:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/swebench-verified/mini-swe-agent/trail/profile-3_run_20260306T072549Z_e5e91775e142_3ac3451764525449b91ff296d1ada643/qps0_5`
- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/swebench-verified/mini-swe-agent/trail/profile-3_run_20260306T072549Z_e5e91775e142_3ac3451764525449b91ff296d1ada643/qps0_6/20260331T195832Z`
- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/swebench-verified/mini-swe-agent/trail-lmcache/profile-3_run_20260306T072549Z_e5e91775e142_3ac3451764525449b91ff296d1ada643/qps0_6/20260331T212051Z`

This directory follows the top-level `figures/README.md` rule:

1. Step 1 materializes a figure-specific dataset into `data/`.
2. Step 2 renders the figure from that processed dataset only.

**Figure Layout**
All 3 subplots are bar charts, with the same source ordering in every panel:

1. average throughput
2. completed-agent LLM time
3. average completion tokens

**Default Source Resolution**
Each source input can point either to:

- an exact run directory, or
- a QPS directory whose child run directories should be searched

If a source input is a QPS directory, the materialization step selects the
lexicographically latest child run directory that contains all required input
JSON files.

With the current defaults, that resolves to:

- `trail/.../qps0_5/20260331T185346Z`
- `trail/.../qps0_6/20260331T195832Z`
- `trail-lmcache/.../qps0_6/20260331T212051Z`

**Source Data**
For each resolved run, the materialization step reads:

- `<run-dir>/post-processed/job-throughput/job-throughput-timeseries.json`
- `<run-dir>/post-processed/agent-output-throughput/agent-output-throughput.json`
- `<run-dir>/post-processed/gateway/stack/completion-tokens-stacked-histogram.json`

These JSON files are the machine-readable inputs behind the job-throughput,
agent-output-throughput, and gateway-stack analyses.

**Metric Definition**
For throughput:

- `average_throughput_jobs_per_s = mean(throughput_points[].throughput_jobs_per_s)`

where `throughput_points` comes from
`job-throughput-timeseries.json`.

For the middle panel:

- `average_completed_agent_llm_time_s = mean(agents[replay_completed == true].llm_request_duration_s)`

This is the average total LLM request time for completed agents only.

The source `agent-output-throughput` metric is:

- `output_throughput_tokens_per_s = output_tokens / llm_request_duration_s`

For the completed-replay-only variant of that figure, taking the reciprocal of
throughput gives time per output token. In these runs the completed agents are
identical in output token count, so that reciprocal is directly proportional to
per-agent total LLM time. This figure uses the more interpretable seconds form:
the average `llm_request_duration_s` across completed agents.

For the right panel:

- `average_completion_tokens = mean(points[].accumulated_value)`

where `points` comes from
`completion-tokens-stacked-histogram.json`.

**Step 1**
Materialize the comparison dataset into `data/` using the default three source
inputs:

```bash
python3 figures/isolate-capacity-qps/materialize_isolate_capacity_qps.py
```

Optional explicit source inputs:

```bash
python3 figures/isolate-capacity-qps/materialize_isolate_capacity_qps.py \
  --source-dirs \
    /path/to/source-a \
    /path/to/source-b \
    /path/to/source-c
```

Optional custom labels:

```bash
python3 figures/isolate-capacity-qps/materialize_isolate_capacity_qps.py \
  --source-labels "TRAIL 0.5" "TRAIL 0.6" "TRAIL + LMCache 0.6"
```

By default this writes a gitignored JSON like:

- `figures/isolate-capacity-qps/data/isolate-capacity-qps.json`

Important fields in the materialized JSON:

- `source_count`
- `source_qps_slugs`
- `selection_policy`
- `metrics`
- `cases`
- `cases[].source_input_path`
- `cases[].run_dir`
- `cases[].metric_values`
- `cases[].metrics.<metric_key>.value`

**Step 2**
Render the 1x3 bar-chart figure from the materialized JSON:

```bash
python3 figures/isolate-capacity-qps/plot_isolate_capacity_qps.py \
  --input figures/isolate-capacity-qps/data/isolate-capacity-qps.json
```

Example with a custom output path:

```bash
python3 figures/isolate-capacity-qps/plot_isolate_capacity_qps.py \
  --input figures/isolate-capacity-qps/data/isolate-capacity-qps.json \
  --output figures/isolate-capacity-qps/output/isolate-capacity-qps.pdf
```

The default rendered output goes into the gitignored `output/` directory.

The plotting step requires `matplotlib` to be installed in the active Python
environment.
