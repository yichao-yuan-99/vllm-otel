This directory contains a post-process wrapper for the publication-style
`figures/stacked-per-agent/` pipeline.

It reads per-agent context-usage ranges recovered by
`post-process/gateway/stack-context/extract_run.py`, materializes fixed-width
bar windows, and renders a stacked bar chart for each run.

Each figure uses:

- x-axis: time from run start (minutes)
- y-axis: average context usage in each bar window by default
- stacked layers: one stable layer per agent, ordered by first active time by default

## Script

- `post-process/visualization/stacked-per-agent/generate_all_figures.py`

## Single Run

Input file (default):

- `<run-dir>/post-processed/gateway/stack-context/context-usage-ranges.json`

Command:

```bash
python post-process/visualization/stacked-per-agent/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/stacked-per-agent/
```

It writes:

- `stacked-per-agent.window-120s.start-0.end-full.json`
- `stacked-per-agent.window-120s.start-0.end-full.png`
- `figures-manifest.json`

Optional parameters (single-run mode):

- `--ranges-input <path>`
- `--output-dir <path>`
- `--window-size-s <seconds>`
- `--start-s <seconds>`
- `--end-s <seconds>`
- `--agent-order {first-active,agent-key}`
- `--value-mode {average,integral}`
- `--legend {auto,show,hide}`
- `--legend-max-agents <positive-int>`
- `--title <text>`
- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Batch Processing

When you have many runs under one root directory:

```bash
python post-process/visualization/stacked-per-agent/generate_all_figures.py \
  --root-dir results/replay
```

Discovery rule:

- any subdirectory that has:
  - `post-processed/gateway/stack-context/context-usage-ranges.json`

Optional worker count:

```bash
python post-process/visualization/stacked-per-agent/generate_all_figures.py \
  --root-dir results/replay \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/visualization/stacked-per-agent/generate_all_figures.py \
  --root-dir results/replay \
  --dry-run
```

Batch-wide rendering options:

- `--window-size-s <seconds>`
- `--start-s <seconds>`
- `--end-s <seconds>`
- `--agent-order {first-active,agent-key}`
- `--value-mode {average,integral}`
- `--legend {auto,show,hide}`
- `--legend-max-agents <positive-int>`
- `--title <text>`
- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Note

This wrapper delegates the actual materialization and plotting work to:

- `figures/stacked-per-agent/materialize_stacked_per_agent.py`
- `figures/stacked-per-agent/plot_stacked_per_agent.py`
