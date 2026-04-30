A figure with two bar subplots.
The left subplot is p5 agent output throught, right is average power;
4 data: 
- no freq control /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03
- freq control no slo /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/20260412T014805Z
- freq control with 35 target, /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/35
- freq control with 45 target. /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/45

## Implemented As

- `figures/slo-compare/materialize_slo_compare.py`
- `figures/slo-compare/plot_slo_compare.py`

The materialization step reads the four runs above and writes a compact JSON
payload with two metrics per variant:

- `p5_output_throughput_tokens_per_s`
- `average_power_w`

Run selection rules:

- if the configured path is already a run directory, use it directly
- if the configured path is a parent directory containing timestamped runs, use
  the newest timestamp directory with `replay/summary.json`
- missing files or missing metric fields are logged and zero-filled in the JSON

Metric extraction rules:

- `p5_output_throughput_tokens_per_s`
  - primary source:
    `agent_output_throughput_tokens_per_s_summary.percentiles["5"]`
  - fallback:
    interpolated 5th percentile of `agents[].output_throughput_tokens_per_s`
- `average_power_w`
  - primary source: `power_stats_w.avg`
  - fallback: arithmetic mean of `power_points[].power_w`

## Outputs

The materialization step writes:

- `figures/slo-compare/data/slo-compare.json`
- `figures/slo-compare/data/slo-compare.missing.log`

The plotting step writes:

- `figures/slo-compare/output/slo-compare.pdf`

## Commands

Materialize the comparison dataset:

```bash
python3 figures/slo-compare/materialize_slo_compare.py
```

Render the two-panel bar chart:

```bash
python3 figures/slo-compare/plot_slo_compare.py \
  --input figures/slo-compare/data/slo-compare.json
```

Notes:

- the plotting step requires `matplotlib`
- the left subplot shows p5 output throughput in tokens/s
- the right subplot shows average power in watts
