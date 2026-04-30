An example of how slo mechanism works over time. 

x-axis: time
Time varying figure of /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/35/20260412T055419Z

subfigures share the x-axis:
- power change over time (smooth by 120s window)
- frequency decision (like the middle one in /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/35/20260412T055419Z/post-processed/visualization/freq-control-linespace-instance-slo/freq-control-timeline.png, remove the line make it scatter plot )
- context usage (like the first subplot in /srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/35/20260412T055419Z/post-processed/visualization/freq-control-linespace-instance-slo/freq-control-timeline.png)
include power, frequency control curve, context usage

## Implemented As

- `figures/slo-details/materialize_slo_details.py`
- `figures/slo-details/plot_slo_details.py`

The materialization step reads one replay run and writes a compact JSON payload
with aligned time series for the three rendered subplots, plus SLO decision data
that remains available in the JSON:

- smoothed power over time from `post-processed/power/power-summary.json`
- frequency-control decisions from
  `post-processed/freq-control-linespace-instance-slo/freq-control-summary.json`
- SLO decision samples from `post-processed/slo-decision/slo-decision-summary.json`
- smoothed context-usage totals from
  `post-processed/gateway/stack-context/context-usage-stacked-histogram.json`

Implementation details:

- if `--run-dir` already points at a run directory, it is used directly
- if `--run-dir` points at a parent directory with timestamped runs, the newest
  run with `replay/summary.json` is selected
- power is smoothed with a centered moving-average window, default `120s`
- context usage is smoothed with a centered moving-average window, default `120s`
- the plotting step renders three vertically stacked subplots with a shared x-axis
- the frequency panel uses scatter markers only
- the default figure width is 7.0 inches

## Outputs

The materialization step writes:

- `figures/slo-details/data/slo-details.json`
- `figures/slo-details/data/slo-details.missing.log`

The plotting step writes:

- `figures/slo-details/output/slo-details.pdf`

## Commands

Materialize the dataset for the default run from this README:

```bash
python3 figures/slo-details/materialize_slo_details.py
```

Render the three-panel timeline figure:

```bash
python3 figures/slo-details/plot_slo_details.py \
  --input figures/slo-details/data/slo-details.json
```

Useful overrides:

```bash
python3 figures/slo-details/materialize_slo_details.py \
  --run-dir /path/to/replay/run \
  --power-smooth-window-s 120 \
  --context-smooth-window-s 120
```
