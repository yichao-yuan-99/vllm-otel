This is similar in style to /srv/scratch/yichaoy2/work/vllm-otel/figures/slo-compare
We compare 3 data
- no freq control /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/dabstep/mini-swe-agent/split/exclude-unranked/qps0_05
- KAIROS without thrashing avoidance /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance/dabstep/mini-swe-agent/split/exclude-unranked/qps0_05/20260413T135808Z
- KAIROS with thrashing avoidance /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware/dabstep/mini-swe-agent/split/exclude-unranked/qps0_05/20260413T235024Z

left: Per Agent P5 throughput
right: System Job Throughput

## Implemented As

- `figures/con-ctrl-compare/materialize_con_ctrl_compare.py`
- `figures/con-ctrl-compare/plot_con_ctrl_compare.py`

The materialization step reads the three runs above and writes a compact JSON
payload with two metrics per variant:

- `p5_output_throughput_tokens_per_s`
- `average_job_throughput_jobs_per_s`

Run selection rules:

- if the configured path is already a run directory, use it directly
- if the configured path is a parent directory containing timestamped runs, use
  the newest timestamp directory with `replay/summary.json`
- missing runs, files, or metric fields are logged and zero-filled in the JSON

Metric extraction rules:

- `p5_output_throughput_tokens_per_s`
  - primary source:
    `agent_output_throughput_tokens_per_s_summary.percentiles["5"]`
  - fallback:
    interpolated 5th percentile of `agents[].output_throughput_tokens_per_s`
- `average_job_throughput_jobs_per_s`
  - source: arithmetic mean of
    `throughput_points[].throughput_jobs_per_s` in
    `post-processed/job-throughput/job-throughput-timeseries.json`

## Outputs

The materialization step writes:

- `figures/con-ctrl-compare/data/con-ctrl-compare.json`
- `figures/con-ctrl-compare/data/con-ctrl-compare.missing.log`

The plotting step writes:

- `figures/con-ctrl-compare/output/con-ctrl-compare.pdf`

## Commands

Materialize the comparison dataset:

```bash
python3 figures/con-ctrl-compare/materialize_con_ctrl_compare.py
```

Render the two-panel bar chart:

```bash
python3 figures/con-ctrl-compare/plot_con_ctrl_compare.py \
  --input figures/con-ctrl-compare/data/con-ctrl-compare.json
```

Notes:

- the plotting step requires `matplotlib`
- the left subplot shows per-agent p5 throughput in tokens/s
- the right subplot shows average system job throughput in jobs/s
