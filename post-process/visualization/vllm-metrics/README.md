This directory contains figure-generation scripts for post-processed vLLM metrics.

It visualizes every extracted metric series as a line chart:

- x-axis: time (`time_from_start_s`)
- y-axis: metric value (`value`)
- annotation: `n`, `avg`, `min`, `max` from stats JSON

The plotting style is designed to be highly readable and academic:

- serif font
- clear grid lines
- high-contrast line colors
- stats box in each figure

## Script

- `post-process/visualization/vllm-metrics/generate_all_figures.py`

## Single Run

Input files (default):

- `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.json`
- `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.stats.json`

Command:

```bash
python post-process/visualization/vllm-metrics/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/vllm-metrics/
```

It writes:

- one figure per metric series (default `.png`)
- `figures-manifest.json` with generated file metadata

Optional parameters (single-run mode):

- `--timeseries-input <path>`
- `--stats-input <path>`
- `--output-dir <path>`
- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Batch Processing

When you have many runs under one root directory:

```bash
python post-process/visualization/vllm-metrics/generate_all_figures.py \
  --root-dir tests/output/con-driver
```

Discovery rule:

- any subdirectory that has:
  - `post-processed/vllm-log/gauge-counter-timeseries.json`
  - `post-processed/vllm-log/gauge-counter-timeseries.stats.json`

Optional worker count:

```bash
python post-process/visualization/vllm-metrics/generate_all_figures.py \
  --root-dir tests/output/con-driver \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/visualization/vllm-metrics/generate_all_figures.py \
  --root-dir tests/output/con-driver \
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
