This directory contains figure-generation scripts for post-processed
agent-output-throughput summaries.

It renders two figures for each run:

- a histogram of `output_throughput_tokens_per_s` using the extracted `1.0`
  token/s bins
- a scatter plot with:
  - x-axis: `output_tokens`
  - y-axis: `output_throughput_tokens_per_s`

The plotting style matches the other post-process visualizations:

- serif font
- readable grid lines
- summary stats box
- peak annotation on the scatter plot

## Script

- `post-process/visualization/agent-output-throughput/generate_all_figures.py`

## Single Run

Input file (default):

- `<run-dir>/post-processed/agent-output-throughput/agent-output-throughput.json`

Command:

```bash
python post-process/visualization/agent-output-throughput/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/agent-output-throughput/
```

It writes:

- `agent-output-throughput-histogram.<format>` (default `.png`)
- `agent-output-throughput-vs-output-tokens.<format>` (default `.png`)
- `figures-manifest.json`

Optional parameters (single-run mode):

- `--agent-output-input <path>`
- `--output-dir <path>`
- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Batch Processing

When you have many runs under one root directory:

```bash
python post-process/visualization/agent-output-throughput/generate_all_figures.py \
  --root-dir results/replay
```

Discovery rule:

- any subdirectory that has:
  - `post-processed/agent-output-throughput/agent-output-throughput.json`

Optional worker count:

```bash
python post-process/visualization/agent-output-throughput/generate_all_figures.py \
  --root-dir results/replay \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/visualization/agent-output-throughput/generate_all_figures.py \
  --root-dir results/replay \
  --dry-run
```

Batch-wide rendering options:

- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Note

`matplotlib` is required to render figures. If it is missing, install it in your
environment, for example:

```bash
pip install matplotlib
```
