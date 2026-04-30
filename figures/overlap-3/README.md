This figure overlays the per-agent output-throughput histogram for the
no-frequency-control baseline against each of the three variants used in
`figures/slo-compare`.

The result is a compact three-row figure:

- row 1: No Freq Control vs KAIROS SLO 20
- row 2: No Freq Control vs KAIROS SLO 35
- row 3: No Freq Control vs KAIROS SLO 45

The visual style is intentionally minimal, closer to `figures/slo-compare`
than `figures/tmp-overlap`: only the overlaid histograms, row titles, axis
labels, and a small legend are kept.

## Implemented As

- `figures/overlap-3/materialize_overlap_3.py`
- `figures/overlap-3/plot_overlap_3.py`

## Source Data

The figure uses the same four runs as `figures/slo-compare`:

- baseline:
  `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03`
- KAIROS SLO 20:
  `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/20260412T014805Z`
- KAIROS SLO 35:
  `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/35`
- KAIROS SLO 45:
  `/srv/scratch/yichaoy2/work/vllm-otel/results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/dabstep/mini-swe-agent/split/exclude-unranked/qps0_03/45`

Run selection rules:

- if the configured path is already a run directory, use it directly
- if the configured path is a parent directory containing timestamped runs, use
  the newest timestamp directory with `replay/summary.json`

The materialization step reads:

- `<run-dir>/post-processed/agent-output-throughput/agent-output-throughput.json`

## Metric Definition

For each agent:

- `output_throughput_tokens_per_s = output_tokens / llm_request_duration_s`

Extraction rules:

- use `agents[].output_throughput_tokens_per_s` directly when present
- otherwise recompute it from `output_tokens / llm_request_duration_s`
- agents missing both forms are skipped

Histogram construction rules:

- all four variants are projected onto one shared histogram grid
- `bin_index = floor(output_throughput_tokens_per_s / bin_size)`
- `bin_start = bin_index * bin_size`
- `bin_end = (bin_index + 1) * bin_size`
- `bin_count = count(agents whose throughput falls in [bin_start, bin_end))`

By default, `bin_size` is inferred from the source histogram metadata. If the
inputs advertise different bin sizes, pass `--bin-size` explicitly.

## Outputs

The materialization step writes:

- `figures/overlap-3/data/overlap-3.json`
- `figures/overlap-3/data/overlap-3.missing.log`

The plotting step writes:

- `figures/overlap-3/output/overlap-3.pdf`

## Commands

Materialize the dataset:

```bash
python3 figures/overlap-3/materialize_overlap_3.py
```

Render the three-row overlay figure:

```bash
python3 figures/overlap-3/plot_overlap_3.py \
  --input figures/overlap-3/data/overlap-3.json
```

Optional shared bin-size override:

```bash
python3 figures/overlap-3/materialize_overlap_3.py \
  --bin-size 1.0
```

Notes:

- the plotting step requires `matplotlib`
- every row shares the same x-axis bin edges
- every row shares the same y-axis count scale
