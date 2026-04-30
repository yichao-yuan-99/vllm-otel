This figure contains three panels built from three multi-instance replay runs:

- `No Freq Control`
- `Round Robin`
- `KAIROS`

The layout is:

- left panel: bar chart comparing aggregate average GPU power across the three runs
- middle panel: pie chart showing the Round Robin per-instance power split
- right panel: pie chart showing the KAIROS per-instance power split

The bar-chart style is intentionally close to `figures/slo-compare`: a simple publication-style bar chart with direct value labels and minimal decoration.

## Source Data

Default inputs:

- No Freq Control:
  `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-gateway_multi/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_15/20260414T183055Z`
- Round Robin:
  `/srv/local/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-gateway_multi-round_robin-freq-ctrl-linesapce-instance-ctx-aware/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_16/20260416T004134Z/`
- KAIROS:
  `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-gateway_multi-lowest_profile_leq_reloc_2-freq-ctrl-linesapce-instance-ctx-aware/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_16/20260415T054807Z`

The Round Robin and KAIROS defaults now point to different runs.

## Throughput Reference

From each run's `post-processed/agent-output-throughput/agent-output-throughput.json`:

- Round Robin p5 agent output throughput: `25.740539 tok/s`
- KAIROS p5 agent output throughput: `26.949981 tok/s`

Run selection rules:

- if an input path is already a run directory, use it directly
- if an input path is a parent directory containing timestamped runs, use the newest child directory that contains `replay/summary.json`

The materialization step reads:

- `<run-dir>/post-processed/power/power-summary.json`

## Metric Definition

For each run:

- `average_power_w = power_stats_w.avg`
- fallback: `average_power_w = mean(power_points[].power_w)`
- `total_energy_j = total_energy_j` from `power-summary.json`

For each GPU instance inside a run:

- `per_gpu_average_power_w = mean(per_gpu_power[i].power_points[].power_w)`
- `per_gpu_share_pct = 100 * per_gpu_average_power_w / sum(all per_gpu_average_power_w)`

Plot usage:

- the left bar chart uses `average_power_w`
- the pie charts use `per_gpu_average_power_w` for slice sizes
- `total_energy_j` is also stored in the materialized dataset as a reference field, but it is not plotted by default

## Outputs

The materialization step writes:

- `figures/mutli/data/mutli.json`
- `figures/mutli/data/mutli.missing.log`

The plotting step writes:

- `figures/mutli/output/mutli.pdf`

## Commands

Materialize the figure dataset with the default paths:

```bash
python3 figures/mutli/materialize_mutli.py
```

Render the figure:

```bash
python3 figures/mutli/plot_mutli.py \
  --input figures/mutli/data/mutli.json
```

Example overriding the default Round Robin path:

```bash
python3 figures/mutli/materialize_mutli.py \
  --round-robin-path /path/to/the/intended/round-robin/run \
  --output figures/mutli/data/mutli.custom.json
```

Then render that materialized dataset:

```bash
python3 figures/mutli/plot_mutli.py \
  --input figures/mutli/data/mutli.custom.json \
  --output figures/mutli/output/mutli.custom.pdf
```

Notes:

- the plotting step requires `matplotlib`
- the current implementation assumes the two pie-chart policies each expose four GPU instances in `per_gpu_power`
