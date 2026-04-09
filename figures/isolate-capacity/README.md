This figure renders a 1 row x 3 column bar-chart comparison for the
`mini-swe-agent` sweep root:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/swebench-verified/mini-swe-agent`

The compared case directories are:

- `trail`
- `trail-usage75`
- `trail-usage75-lmcache100`

The original note said "qps 0.05", but the referenced source paths all point to
the `qps0_5` directory slug. This implementation therefore compares the
`qps0_5` runs.

This directory follows the top-level `figures/README.md` rule:

1. Step 1 materializes a figure-specific dataset into `data/`.
2. Step 2 renders the figure from that processed dataset only.

**Figure Layout**
All 3 subplots are bar charts, with the same case ordering in every panel:

1. average throughput
2. completed-agent LLM time
3. average completion tokens

**Default Run Selection**
The materialization step scans each case directory for run directories under:

- `<case-dir>/*/<qps-slug>/*`

and selects the lexicographically latest run directory name that contains all
required input JSON files.

With the current sweep root and the default `--qps-slug qps0_5`, that resolves
to:

- `trail/.../qps0_5/20260331T185346Z`
- `trail-usage75/.../qps0_5/20260401T021317Z`
- `trail-usage75-lmcache100/.../qps0_5/20260401T035627Z`

**Source Data**
For each selected run, the materialization step reads:

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

The completion-tokens metric corresponds to figures such as:

- `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/swebench-verified/mini-swe-agent/trail/profile-3_run_20260306T072549Z_e5e91775e142_3ac3451764525449b91ff296d1ada643/qps0_5/20260331T185346Z/post-processed/visualization/gateway-stack/completion-tokens-stacked-histogram.png`

**Step 1**
Materialize the comparison dataset into `data/`:

```bash
python3 figures/isolate-capacity/materialize_isolate_capacity.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/swebench-verified/mini-swe-agent
```

Optional explicit QPS slug:

```bash
python3 figures/isolate-capacity/materialize_isolate_capacity.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/swebench-verified/mini-swe-agent \
  --qps-slug qps0_5
```

Optional profile filter:

```bash
python3 figures/isolate-capacity/materialize_isolate_capacity.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/swebench-verified/mini-swe-agent \
  --profile-dir-name profile-3_run_20260306T072549Z_e5e91775e142_3ac3451764525449b91ff296d1ada643
```

Optional custom labels:

```bash
python3 figures/isolate-capacity/materialize_isolate_capacity.py \
  --root-dir /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-same-agent-uniform/swebench-verified/mini-swe-agent \
  --case-labels TRAIL "TRAIL 75%" "TRAIL 75% + LMCache"
```

By default this writes a gitignored JSON like:

- `figures/isolate-capacity/data/isolate-capacity.qps0_5.json`

Important fields in the materialized JSON:

- `qps_slug`
- `selection_policy`
- `metrics`
- `cases`
- `cases[].run_path`
- `cases[].metric_values`
- `cases[].metrics.<metric_key>.value`

**Step 2**
Render the 1x3 bar-chart figure from the materialized JSON:

```bash
python3 figures/isolate-capacity/plot_isolate_capacity.py \
  --input figures/isolate-capacity/data/isolate-capacity.qps0_5.json
```

Example with a custom output path:

```bash
python3 figures/isolate-capacity/plot_isolate_capacity.py \
  --input figures/isolate-capacity/data/isolate-capacity.qps0_5.json \
  --output figures/isolate-capacity/output/isolate-capacity.qps0_5.png
```

The default rendered output goes into the gitignored `output/` directory.

The plotting step requires `matplotlib` to be installed in the active Python
environment.
