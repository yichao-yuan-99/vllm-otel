This directory builds the first main summary figure from the already
materialized `energy-context-latency` dataset.

The final figure has two panel groups laid out in one row:

1. Throughput and SLO attainment rate (20tokens/s)
   - Bars on the left y-axis: `p5_output_throughput_tokens_per_s`
   - Dotted line segments on the right y-axis, one per QPS cluster, with each
     circle marker colored to match its corresponding bar and outlined in black:
     `pct_agents_above_20_output_throughput_tokens_per_s`
2. Power and energy
   - Bars on the left y-axis: `average_power_w`
   - Dotted line segments on the right y-axis, one per QPS cluster, with each
     circle marker colored to match its corresponding bar and outlined in black:
     `average_energy_per_finished_agent_kj`

Each panel group contains the same three experiment subplots as
`figures/energy-context-latency/`:

- `A. swebench-verified + mini-swe-agent`
- `B. dabstep + mini-swe-agent`
- `C. terminal-bench-2.0 + terminus-2`

Within each experiment subplot:

- the x-axis is `Input Jobs/s`
- each QPS cluster contains three implementation-aligned bars:
  `Uncontrolled`, `Fixed Freq (810Mhz)`, and `STEER`
- below the x-axis label, each subplot shows a two-line caption with the
  dataset on the first line and the agent on the second line
- the secondary-axis overlay is broken at QPS boundaries, so only the three
  implementation points inside each cluster are connected

## Source Data

This figure does not read raw experiment directories directly. Instead it
reuses the processed comparison dataset from:

- `figures/energy-context-latency/data/energy-context-latency.json`

That upstream dataset already applies the raw-data selection rules, fallback
logic, and zero-filling for missing runs/files/fields. If upstream missing data
was logged, the main figure preserves the reference to the upstream missing log.

## Metrics

The four plotted metrics come directly from
`figures/energy-context-latency/README.md`:

- `p5_output_throughput_tokens_per_s`
  - 5th percentile of
    `agents[].output_throughput_tokens_per_s`
- `pct_agents_above_20_output_throughput_tokens_per_s`
  - `count(output_throughput_tokens_per_s > 20) / count(output_throughput_tokens_per_s) * 100`
- `average_power_w`
  - `power_stats_w.avg`
  - fallback: arithmetic mean of `power_points[].power_w`
- `average_energy_per_finished_agent_kj`
  - `(average_power_w * analysis_window_duration_s) / workers_completed / 1000`

## Average Power Saving of KAIROS vs. No Frequency Control

Assumption: in `main-fig-1`, KAIROS corresponds to the `STEER`
implementation, and "no frequency control" corresponds to `Uncontrolled`.

Using the figure source data in
`figures/energy-context-latency/data/energy-context-latency.json`, compute the
power saving for each matched experiment/QPS point as:

```text
power_saving_pct = (average_power_w_uncontrolled - average_power_w_kairos)
                   / average_power_w_uncontrolled * 100
```

Matched points from the three subplots in `main-fig-1`:

- A / 0.04: `(342.074019 - 205.885916) / 342.074019 * 100 = 39.81%`
- A / 0.06: `(367.933564 - 240.918117) / 367.933564 * 100 = 34.52%`
- A / 0.08: `(366.247271 - 284.611512) / 366.247271 * 100 = 22.29%`
- B / 0.03: `(333.181919 - 223.565637) / 333.181919 * 100 = 32.90%`
- B / 0.04: `(348.050546 - 248.964128) / 348.050546 * 100 = 28.47%`
- B / 0.05: `(368.205268 - 293.284310) / 368.205268 * 100 = 20.35%`
- C / 0.015: `(337.756259 - 232.867498) / 337.756259 * 100 = 31.05%`
- C / 0.02: `(365.200101 - 291.001912) / 365.200101 * 100 = 20.32%`
- C / 0.025: `(374.725961 - 323.755964) / 374.725961 * 100 = 13.60%`

Average across the 9 matched points:

```text
(39.81 + 34.52 + 22.29 + 32.90 + 28.47 + 20.35 + 31.05 + 20.32 + 13.60) / 9
= 27.03%
```

Result: KAIROS saves **27.03% average power** relative to no frequency control
in the workloads/QPS points shown by `main-fig-1`.

Equivalent mean-power view over the same 9 points:

- mean uncontrolled power = `355.93 W`
- mean KAIROS power = `260.54 W`
- mean absolute reduction = `95.39 W`
- reduction from mean powers = `(355.93 - 260.54) / 355.93 * 100 = 26.80%`

## Implemented As

- `figures/main-fig-1/materialize_main_fig_1.py`
- `figures/main-fig-1/plot_main_fig_1.py`

The materialization step reads the upstream JSON, keeps only the four metrics
used by the combined figure, and writes a figure-specific JSON payload.

The plotting step renders one combined figure with two panel groups. Each group
uses grouped bars with black edges on the primary axis and dotted circle-marker
segments on the secondary axis, with one segment per QPS cluster.

## Outputs

The materialization step writes:

- `figures/main-fig-1/data/main-fig-1.json`

The plotting step writes:

- `figures/main-fig-1/output/main-fig-1.pdf`

## Commands

Materialize the figure-specific dataset from the upstream comparison JSON:

```bash
python3 figures/main-fig-1/materialize_main_fig_1.py
```

Render the combined figure:

```bash
python3 figures/main-fig-1/plot_main_fig_1.py \
  --input figures/main-fig-1/data/main-fig-1.json
```

Notes:

- the main figure inherits upstream zero-filled values for missing raw data
- the plotting step requires `matplotlib`
