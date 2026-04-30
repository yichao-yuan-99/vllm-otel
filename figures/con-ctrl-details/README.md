This is similar to /srv/scratch/yichaoy2/work/vllm-otel/figures/slo-details in style.

We build a four-panel time-varying figure for the concurrency-control example:

- job throughput for KAIROS without thrashing avoidance
- job throughput for KAIROS with thrashing avoidance
- context usage for KAIROS with thrashing avoidance
- pending agent count for KAIROS with thrashing avoidance

## Implemented As

- `figures/con-ctrl-details/materialize_con_ctrl_details.py`
- `figures/con-ctrl-details/plot_con_ctrl_details.py`

The materialization step reads these runs:

- KAIROS without thrashing avoidance:
  `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance/dabstep/mini-swe-agent/split/exclude-unranked/qps0_05/20260413T135808Z`
- KAIROS with thrashing avoidance:
  `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware/dabstep/mini-swe-agent/split/exclude-unranked/qps0_05/20260413T235024Z`

Run selection rules:

- if the configured path is already a run directory, use it directly
- if the configured path is a parent directory containing timestamped runs, use
  the newest timestamp directory with `replay/summary.json`

Series extraction rules:

- `job_throughput_no_thrashing_avoidance`
  - source:
    `post-processed/job-throughput/job-throughput-timeseries.json`
  - field: `throughput_points[].throughput_jobs_per_s`
- `job_throughput_with_thrashing_avoidance`
  - source:
    `post-processed/job-throughput/job-throughput-timeseries.json`
  - field: `throughput_points[].throughput_jobs_per_s`
- `context_usage_with_thrashing_avoidance`
  - source:
    `post-processed/gateway/stack-context/context-usage-stacked-histogram.json`
  - field: `points[].accumulated_value`
  - smoothing: centered rolling average, default `120s`
- `pending_agent_count_with_thrashing_avoidance`
  - source:
    `post-processed/gateway/ctx-aware-log/ctx-aware-timeseries.json`
  - field: `samples[].pending_agent_count`
  - resampling: keep the last value in each integer second

## Outputs

The materialization step writes:

- `figures/con-ctrl-details/data/con-ctrl-details.json`
- `figures/con-ctrl-details/data/con-ctrl-details.missing.log`

The plotting step writes:

- `figures/con-ctrl-details/output/con-ctrl-details.pdf`

## Commands

Materialize the dataset:

```bash
python3 figures/con-ctrl-details/materialize_con_ctrl_details.py
```

Render the figure:

```bash
python3 figures/con-ctrl-details/plot_con_ctrl_details.py \
  --input figures/con-ctrl-details/data/con-ctrl-details.json
```

Notes:

- the plotting step requires `matplotlib`
- the figure uses a shared x-axis in minutes from run start
- the top two panels show system job throughput in jobs/s
