This directory contains figure-generation scripts for post-processed prefill
concurrency timeseries.

It visualizes each run's prefill activity concurrency over time with one line
chart:

- x-axis: time (`time_offset_s`)
- y-axis: active prefill requests (`concurrency`)
- annotation: request/activity counts, sample count, avg/min/max concurrency,
  peak time, run duration, and tick size

The plotting style matches the existing post-process figures:

- serif font
- readable grid lines
- filled curve
- peak marker and stats box

## Script

- `post-process/visualization/prefill-concurrency/generate_all_figures.py`

## Single Run

Input file (default):

- `<run-dir>/post-processed/prefill-concurrency/prefill-concurrency-timeseries.json`

Command:

```bash
python post-process/visualization/prefill-concurrency/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/prefill-concurrency/
```

It writes:

- `prefill-concurrency.<format>` (default `.png`)
- `figures-manifest.json`

Optional parameters (single-run mode):

- `--timeseries-input <path>`
- `--output-dir <path>`
- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Batch Processing

When you have many runs under one root directory:

```bash
python post-process/visualization/prefill-concurrency/generate_all_figures.py \
  --root-dir results/replay
```

Discovery rule:

- any subdirectory that has:
  - `post-processed/prefill-concurrency/prefill-concurrency-timeseries.json`

Optional worker count:

```bash
python post-process/visualization/prefill-concurrency/generate_all_figures.py \
  --root-dir results/replay \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/visualization/prefill-concurrency/generate_all_figures.py \
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
