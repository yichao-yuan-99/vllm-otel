In this directory, we want to do a bar-chart comparison across 3 sets of experiments for 3 different implementations.

3 set of experiments are
- A: swebench-verified and mini-swe-agent
- B: dabstep and mini-swe-agent
- C: terminal-bench and termius-2

3 implementations and their corresponding result directories are (can be missing, in that case just consider any data we need from that is 0)

1. The uncontroled data. the root is at `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean`.
For a certain, `dataset`, `agent`, and `qps`, the corresponding data is in `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/<dataset>/<agent>/split/exclude-unranked/<qps>/<the-newest-timestamp>`
2. the fix freq controlled data. the root is `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-fixed-freq`.
For a certain, `dataset`, `agent`, and `qps`, the corresponding data is in `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-fixed-freq/<dataset>/<agent>/split/exclude-unranked/<qps>/<the-newest-timestamp>`
3. the STEER controlled data. the root is `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance`
For a certain, `dataset`, `agent`, and `qps`, the corresponding data is in `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance/<dataset>/<agent>/split/exclude-unranked/<qps>/<the-newest-timestamp>`
Exception: for B (`dabstep` + `mini-swe-agent`) at `qps0_05`, use `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware/dabstep/mini-swe-agent/split/exclude-unranked/qps0_05/<the-newest-timestamp>`.


We want to draw five plots, each as a bar chart with 3 subplots.
Each subplot corresponds to one dataset-agent combination.
Each subplot contains 3 clusters of bars, each corresponding to one QPS (listed below).
Each cluster of bars contains 3 bars, each corresponding to one implementation.

The qps for A is 0.04,0.06,0.08, for B is 0.03,0.04,0.05, for C is 0.015,0.02,0.025.

The five plots correspond to different metrics.

The first one is the average energy consumption for each agent, which is calculated by 1. reading `workers_completed` from `replay/summary.json`; 2. multiplying the average power by the job duration to get the total energy cost; and 3. dividing the total energy cost by `workers_completed` to get the per-agent energy cost.

The second one is the average power during the analysis window, which is read from `post-processed/power/power-summary.json` as `power_stats_w.avg`, with a fallback to the arithmetic mean of `power_points[].power_w`.

The third one is the average context usage. This corresponds to the average value shown in figures like `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl/dabstep/mini-swe-agent/split/exclude-unranked/qps0_02/20260402T052552Z/post-processed/visualization/gateway-stack-context/context-usage-stacked-histogram.png`, divided by 527664 (the maximum number of tokens, so the result is a percentage value).

The fourth one is the 5th percentile tokens-per-second value in files like `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/20260410T142848Z/post-processed/agent-output-throughput/agent-output-throughput.json`.

The fifth one is the average job throughput value shown in figures like `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_02/20260321T143624Z/post-processed/visualization/job-throughput/job-throughput.png`, which corresponds to the arithmetic mean of `throughput_points[].throughput_jobs_per_s` in `post-processed/job-throughput/job-throughput-timeseries.json`.


Make a log for want ever data you cannot find

## Implemented As

- `figures/energy-context-latency/materialize_energy_context_latency.py`
- `figures/energy-context-latency/plot_energy_context_latency.py`

The implementation follows the layout above, with the real on-disk names used
for experiment C:

- dataset: `terminal-bench-2.0`
- agent: `terminus-2`

For the fixed-frequency root, the actual run data lives one level below each
timestamp directory in a fixed nested directory:

- `<timestamp>/core-345-810/`

When selecting a fixed-frequency run, use the newest timestamp that contains
that nested directory, even if a newer timestamp exists without it.

## Outputs

The materialization step writes:

- `figures/energy-context-latency/data/energy-context-latency.json`
- `figures/energy-context-latency/data/energy-context-latency.missing.log`

The plotting step writes one figure per metric to
`figures/energy-context-latency/output/`:

- `energy-context-latency.average_energy_per_finished_agent_kj.<format>`
- `energy-context-latency.average_power_w.<format>`
- `energy-context-latency.average_context_usage_pct.<format>`
- `energy-context-latency.p5_output_throughput_tokens_per_s.<format>`
- `energy-context-latency.average_job_throughput_jobs_per_s.<format>`

## Commands

Materialize the comparison dataset and missing-data log:

```bash
python3 figures/energy-context-latency/materialize_energy_context_latency.py
```

Render the five clustered bar charts:

```bash
python3 figures/energy-context-latency/plot_energy_context_latency.py \
  --input figures/energy-context-latency/data/energy-context-latency.json
```

Notes:

- missing runs, files, and missing metric fields are logged and plotted as `0`
- the newest timestamp directory is selected for each
  `(implementation, dataset, agent, qps)` tuple
- the plotting step requires `matplotlib`
